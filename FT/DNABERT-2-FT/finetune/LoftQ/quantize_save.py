# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BertConfig,
)


from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
    Q8bitBertUnpadSelfAttentionWithExtras,
    Q4bitBertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.bert_attention2 import (
    AttentionGateType,
    Q8bitBertSelfAttentionWithExtras,
    Q4bitBertSelfAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)

from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from safetensors import safe_open

class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        default="vanilla",
       
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
       
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=512
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True
    )

    parser.add_argument(
        '--num_labels',
        type=int,
        default=2
    )
    
    
    args = parser.parse_args()
    return args


def quantize_and_save():
    args = arg_parse()
    print(f"Trust remote code: {args.trust_remote_code}")

    # Download weights and configure LoRA
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        token=args.token
    )
    print(args.model_name_or_path.lower())
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["zhihan1996"]):
            config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, trust_remote_code=args.trust_remote_code,)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
                # _fast_init=False,
            )
            for layer_idx in range(len(model.bert.encoder.layer)):
                        old_self = model.bert.encoder.layer[layer_idx].attention.self
                        print("----------------------------------------------------------")
                        print("Inside BERT custom attention")
                        print("----------------------------------------------------------")
                        new_self = BertUnpadSelfAttentionWithExtras(
                            config,
                            position_embedding_type=None,
                            softmax_fn=SOFTMAX_MAPPING[args.attn_type],
                            ssm_eps=None,
                            tau=None,
                            max_seq_length=tokenizer.model_max_length,
                            skip_attn=False,
                            fine_tuning=True,
                        ).cuda()

                        # copy loaded weights
                        if args.model_name_or_path is not None:
                            new_self.load_state_dict(old_self.state_dict(), strict=False)
                        model.bert.encoder.layer[layer_idx].attention.self = new_self
            task_type = TaskType.SEQ_CLS
            target_modules = ["Wqkv"]  # embeddings not supported by peft

    elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert", "dnabert", "dna"]):
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, trust_remote_code=args.trust_remote_code,)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            # _fast_init=False,
        )
        for layer_idx in range(len(model.bert.encoder.layer)):
                    old_self = model.bert.encoder.layer[layer_idx].attention.self
                    print("----------------------------------------------------------")
                    print("Inside BERT custom attention")
                    print("----------------------------------------------------------")
                    new_self = BertUnpadSelfAttentionWithExtras(
                        config,
                        position_embedding_type=None,
                        softmax_fn=SOFTMAX_MAPPING[args.attn_type],
                        ssm_eps=None,
                        tau=None,
                        max_seq_length=tokenizer.model_max_length,
                        skip_attn=False,
                        fine_tuning=True,
                    ).cuda()

                    # copy loaded weights
                    if args.model_name_or_path is not None:
                        new_self.load_state_dict(old_self.state_dict(), strict=False)
                    model.bert.encoder.layer[layer_idx].attention.self = new_self
        task_type = TaskType.SEQ_CLS
        target_modules = ["Wqkv"]  # embeddings not supported by peft
    else:
        raise NotImplementedError("Other models not supported yet.")

    # Config of LoftQ
    loftq_config = LoftQConfig(loftq_bits=args.bits, loftq_iter=args.iter)

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=args.rank,
        lora_alpha=16 if task_type is TaskType.CAUSAL_LM and args.bits == 4 else args.rank,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )

    # Obtain LoftQ model
    lora_model = get_peft_model(model, lora_config)
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit" + f"-{args.rank}rank"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "loftq_init")

    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    print_model(base_model, "base_model")

    # convert safetensor to bin
    tensors = {}
    with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  # Don't apply LoftQ when loading again
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_dir, lora_dir = quantize_and_save()

# example command:
# python quantize_save_load.py \
# --model_name_or_path meta-llama/Llama-2-7b-hf \
# --token XXX \
# --bits 4 --iter 5 --rank 16 \
# --save_dir ./model_zoo/loftq/
