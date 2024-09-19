#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import json
import logging
import math
import os
import random
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from transformers.models.bert.configuration_bert import BertConfig
import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset, load_from_disk
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.quantized_dnabert import QuantizedBertForSequenceClassification
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)

from typing import Optional, Dict, Sequence, Tuple, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import csv
import sklearn

from peft import PeftModel, PeftConfig

lora=os.environ.get("lora")
print(lora)

logger = logging.getLogger("validate_sc")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

EXTRA_METRICS = True


def attach_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def main():
    args = parse_args()
    logger.info(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in
    # this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up
    # all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
        
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )
    accelerator.init_trackers("tb_logs_validation", init_kwargs={"wandb":{"name":args.run_name}})

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare HuggingFace config
    # In distributed training, the .from_pretrained methods guarantee that only one local process
    # can concurrently download model & vocab.
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Display config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(config))

    # Load tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this "
            "script. You can do it from another script, save it, and load it from here, "
            "using --tokenizer_name."
        )

    # Load and prepare model
    if args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.model_cache_dir,
            trust_remote_code=args.trust_remote_code
        )

    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)


    for layer_idx in range(len(model.bert.encoder.layer)):
        old_self = model.bert.encoder.layer[layer_idx].attention.self
        print("----------------------------------------------------------")
        print("Inside BERT custom attention")
        print("----------------------------------------------------------")
        new_self = BertUnpadSelfAttentionWithExtras(
            config,
            position_embedding_type=None,
            softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
            ssm_eps=None,
            tau=None,
            max_seq_length=args.max_seq_length,
            skip_attn=False,
            fine_tuning=False,
        )

        # copy loaded weights
        if args.model_name_or_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.bert.encoder.layer[layer_idx].attention.self = new_self
    

    # Gating -> load the model again to load missing alpha
    if args.model_name_or_path is not None and AttentionGateType.none.name != "none":
        state_dict = torch.load(str(Path(args.model_name_or_path) / "pytorch_model.bin"))
        new_state_dict = {}
        for name, val in state_dict.items():
            if "alpha" in name:
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict, strict=False)
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing token embeddings to fit tokenizer vocab size")
        model.resize_token_embeddings(len(tokenizer))

    

    # Display num params
    n_embeddings = count_params(model.bert.embeddings)
    n_encoder = count_params(model.bert.encoder)
    n_head = count_params(model.classifier)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )
    
    # Check sequence length
    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` "
                f"({tokenizer.model_max_length}). Picking 1024 instead. You can change that "
                f"default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum "
                f"length for the model ({tokenizer.model_max_length}). Using "
                f"max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=args.train_file, 
                                      kmer=-1)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=args.validation_file, 
                                     kmer=-1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    model = PeftModel.from_pretrained(model, lora)


    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.preprocessing_num_workers,
        shuffle=True,
    )    

    # Prepare everything with our `accelerator`.
    model,train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    logger.info("FP model:")
    logger.info(model)

    # Quantize:
    if args.quantize:
        click_config = get_quant_config()

        # override number of batches
        click_config.act_quant.num_batches = args.est_num_batches
        click_config.quant.n_bits = args.n_bits
        click_config.quant.n_bits_act = args.n_bits_act
        if args.no_weight_quant:
            click_config.quant.weight_quant = False
        if args.no_act_quant:
            click_config.quant.act_quant = False

        # Weight Ranges
        if args.ranges_weights == "minmax":
            pass
        elif args.ranges_weights in ("mse", "MSE"):
            click_config.quant.weight_quant_method = RangeEstimators.MSE
            click_config.quant.weight_opt_method = OptMethod.grid
        else:
            raise ValueError(f"Unknown weight range estimation: {args.ranges_weights}")

        # Acts ranges
        if args.percentile is not None:
            click_config.act_quant.options["percentile"] = args.percentile

        if args.ranges_acts == "running_minmax":
            click_config.act_quant.quant_method = RangeEstimators.running_minmax

        elif args.ranges_acts == "MSE":
            click_config.act_quant.quant_method = RangeEstimators.MSE
            if args.qmethod_acts == "symmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.grid)
            elif args.qmethod_acts == "asymmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.golden_section)

        elif args.ranges_acts.startswith("L"):
            click_config.act_quant.quant_method = RangeEstimators.Lp
            p_norm = float(args.ranges_acts.replace("L", ""))
            options = dict(p_norm=p_norm)
            if args.qmethod_acts == "symmetric_uniform":
                options["opt_method"] = OptMethod.grid
            elif args.qmethod_acts == "asymmetric_uniform":
                options["opt_method"] = OptMethod.golden_section
            click_config.act_quant.options = options

        else:
            raise NotImplementedError(f"Unknown act range estimation setting, '{args.ranges_acts}'")

        qparams = val_qparams(click_config)
        qparams["quant_dict"] = {}

        model = QuantizedBertForSequenceClassification(model, **qparams)
        model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

        logger.info("Quantized model:")
        logger.info(model)

        # Range estimation
        logger.info("** Estimate quantization ranges on training data **")
        pass_data_for_range_estimation(
            loader=train_dataloader,
            model=model,
            act_quant=click_config.quant.act_quant,
            max_num_batches=click_config.act_quant.num_batches,
        )
        model.fix_ranges()
        model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

    # attach hooks for activation stats (if needed)
    act_dict = {}
    if EXTRA_METRICS:
        act_dict = attach_act_hooks(model)

    num_layers = len(model.bert.encoder.layer)
    act_inf_norms = OrderedDict()
    act_kurtoses = OrderedDict()

    # *** Evaluation ***
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits  # Extract logits
        labels = batch["labels"]  # Assuming labels are in the batch

        # Gather loss for metrics
        loss_ = accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
        losses.append(loss_.detach().cpu())  # Detach and move to CPU to free GPU memory

        # Gather logits and labels for custom metric computation
        gathered_logits = accelerator.gather_for_metrics(logits).detach().cpu()
        gathered_labels = accelerator.gather_for_metrics(labels).detach().cpu()

        all_logits.append(gathered_logits)
        all_labels.append(gathered_labels)

        #print(model)
        # compute inf norms
        if EXTRA_METRICS:
            for j in range(num_layers):
                for name in (
                    f"bert.encoder.layer.{j}.mlp.wo",  # FFN output
                    f"bert.encoder.layer.{j}.mlp.layernorm",  # LN(FFN output + input)
                ):
                    x_inp, x_out = act_dict[name]

                    x = x_out

                    # inf-norm
                    x = x.view(x.size(0), -1)
                    inf_norms = x.norm(dim=1, p=np.inf)
                    if not name in act_inf_norms:
                        act_inf_norms[name] = AverageMeter()
                    for v in inf_norms:
                        act_inf_norms[name].update(v.item())

                    # kurtosis
                    if batch_idx <= 256:
                        kurt = kurtosis(x)
                        if not name in act_kurtoses:
                            act_kurtoses[name] = AverageMeter()
                        for v in kurt:
                            act_kurtoses[name].update(v.item())

                    # compute inf norm also for input
                    if "layerNorm" in name:
                        x = x_inp
                        x = x.view(x.size(0), -1)
                        inf_norms = x.norm(dim=1, p=np.inf)
                        name += ".input"
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())
        # break

    # Concatenate all gathered tensors
    losses = torch.cat(losses)
    logits = torch.cat(all_logits).numpy()  # Convert to numpy
    labels = torch.cat(all_labels).numpy()  # Convert to numpy
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    
    # Calculate custom metrics
    metrics = compute_metrics((logits, labels))

    # Logging
    logger.info(f"perplexity: {perplexity:.4f}")
    logger.info(f"loss: {eval_loss:.4f}")
    logger.info(f"accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"f1: {metrics['f1']:.4f}")
    logger.info(f"matthews_correlation: {metrics['matthews_correlation']:.4f}")
    logger.info(f"precision: {metrics['precision']:.4f}")
    logger.info(f"recall: {metrics['recall']:.4f}")


    if EXTRA_METRICS:
        for name, v in act_inf_norms.items():
            metrics[name] = v.avg

        max_ffn_out_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "wo" in k)
        max_LN_out_inf_norm = max(
            v.avg for k, v in act_inf_norms.items() if k.endswith("layernorm")
        )
        # max_LN_inp_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "input" in k)
        avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
        max_kurtosis = max(v.avg for v in act_kurtoses.values())

        metrics["max_ffn_out_inf_norm"] = max_ffn_out_inf_norm
        metrics["max_LN_out_inf_norm"] = max_LN_out_inf_norm
        #metrics["max_LN_inp_inf_norm"] = max_LN_inp_inf_norm
        metrics["avg_kurtosis"] = avg_kurtosis
        metrics["max_kurtosis"] = max_kurtosis

        logger.info(f"max FFN output inf norm: {max_ffn_out_inf_norm:.2f}")
        #logger.info(f"max FFN input + output inf norm: {max_LN_inp_inf_norm:.1f}")
        logger.info(f"max LN(FFN i + o) inf norm: {max_LN_out_inf_norm:.1f}")
        logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
        logger.info(f"Max Kurtosis: {max_kurtosis:.2f}")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
