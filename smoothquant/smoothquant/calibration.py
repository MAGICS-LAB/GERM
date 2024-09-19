import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, train_file, validation_file, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    data_files = {"train": train_file, "validation": validation_file}
    if train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
        )
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    dataset = raw_datasets["train"]
    
    dataset = dataset.shuffle(seed=42)
    print(dataset)
    print(dataset[0])
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["sequence"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_encoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
    hidden_size = 768,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    data_files = {"train": train_file, "validation": validation_file}
    if train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
        )
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    dataset = raw_datasets["train"]
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["sequence"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.attention.self.Wqkv"]["input"] / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.attention.self.Wqkv"]["output"][:hidden_size] / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.attention.self.Wqkv"]["output"][hidden_size:2 * hidden_size]  / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.attention.self.Wqkv"]["output"][2 * hidden_size:]  / 127
        )
        scale_dict["gate_input_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.mlp.gated_layers"]["input"] / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[f"model.bert.encoder.layer.{idx}.mlp.wo"]["input"] / 127
        )
        # scale_dict["fc1_input_scale"] = (
        #     act_dict[f"model.bert.encoder.{idx}.fc1"]["input"] / 127
        # )
        # scale_dict["fc2_input_scale"] = (
        #     act_dict[f"model.bert.encoder.{idx}.fc2"]["input"] / 127
        # )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
