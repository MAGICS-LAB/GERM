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
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.quantized_dnabert import QuantizedBertForMaskedLM
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)

logger = logging.getLogger("validate_mlm")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

EXTRA_METRICS = False


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
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.model_cache_dir,
            trust_remote_code=args.trust_remote_code
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)


    for layer_idx in range(len(model.bert.encoder.layer)):
        old_self = model.bert.encoder.layer[layer_idx].attention.self
        print("----------------------------------------------------------")
        print("Inside BERT custom attention")
        print("----------------------------------------------------------")
        new_self = BertUnpadSelfAttentionWithExtras(
            config,
            position_embedding_type=None,
            softmax_fn=SOFTMAX_MAPPING["vanilla"],
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
    n_head = count_params(model.cls)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )

    # Get the datasets.
    # In distributed training, the load_dataset function guarantee that only one local process can
    # concurrently download the dataset.


    if args.dataset_name is not None: # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
                streaming=args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
                streaming=args.streaming,
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=args.data_cache_dir
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )


    # Preprocessing the datasets.
    column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

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

    # Tokenize all the texts.
    # YB: removed line-by-line option as we'll likely never use it
    column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
    # Data collator
    # This one will take care of randomly masking the tokens.
    
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = tokenized_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    pad_to_multiple_of_8 = args.line_by_line and args.fp16 and not args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # DataLoaders creation:
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

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

        model = QuantizedBertForMaskedLM(model, **qparams)
        model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

        logger.info("Quantized model:")
        logger.info(model)

        # Range estimation
        logger.info("** Estimate quantization ranges on training data **")
        pass_data_for_range_estimation(
            loader=eval_dataloader,
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
    for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        loss_ = accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
        losses.append(loss_)

        #print(model)
        # compute inf norms
        if EXTRA_METRICS:
            for j in range(num_layers):
                for name in (
                    f"bert.encoder.layer.{j}.mlp.gated_layers",  # FFN output
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
                    if "LayerNorm" in name:
                        x = x_inp
                        x = x.view(x.size(0), -1)
                        inf_norms = x.norm(dim=1, p=np.inf)
                        name += ".input"
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())
        # break

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    logger.info(f"perplexity: {perplexity:.4f}")
    logger.info(f"loss: {eval_loss:.4f}")    
    # metrics
    metrics = OrderedDict([
        ("perplexity", perplexity),
        ("loss", eval_loss.item())
    ])

    if EXTRA_METRICS:
        for name, v in act_inf_norms.items():
            metrics[name] = v.avg

        max_ffn_out_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "gated_layers" in k)
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

        logger.info(f"max FFN output inf norm: {max_ffn_out_inf_norm:.1f}")
        #logger.info(f"max FFN input + output inf norm: {max_LN_inp_inf_norm:.1f}")
        logger.info(f"max LN(FFN i + o) inf norm: {max_LN_out_inf_norm:.1f}")
        logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
        logger.info(f"Max Kurtosis: {max_kurtosis:.1f}")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
