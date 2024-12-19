import os
import json
import logging
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from typing import Optional, Dict, Sequence, Tuple, List

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_dnabert_layer import QuantBertLayer
from quantize.int_linear import QuantLinear

import pdb
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BitsAndBytesConfig,
)
from typing import Optional, Dict, Sequence, Tuple, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import csv
import sklearn
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b",
    "zhihan1996/DNABERT-2-117M",
    "google-bert/bert-base-uncased",
    "magicslabnu/OutEffHop_bert_base",
]

# Add function to save results
def save_results(output_dir, ppl, metrics):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results = {
            "perplexity": ppl.item(),  # Convert to a regular number if it's a tensor
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "matthews_correlation": metrics["matthews_correlation"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(results, f)
        print(f"Results saved to {os.path.join(output_dir, 'all_results.json')}")

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

# Modify the Evaluator class to calculate custom metrics
class CustomEvaluator:
    def __init__(self, dataset, tokenizer, device, batch_size, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = n_samples

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer),
        )

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        all_logits = []
        all_labels = []

        for batch in tqdm(self.dataloader, desc="Evaluating..."):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids)
                lm_logits = outputs.logits

            if lm_logits.dtype == torch.bfloat16:
                lm_logits = lm_logits.to(torch.float32)

            if lm_logits.dim() == 3:
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:]
            elif lm_logits.dim() == 2:
                shift_logits = lm_logits.unsqueeze(1)  # Add a sequence dimension
                shift_labels = labels.unsqueeze(1)  # Add a sequence dimension
            else:
                raise ValueError(f"Unexpected number of dimensions: {lm_logits.dim()}")

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * shift_logits.size(1)
            nlls.append(neg_log_likelihood)

            all_logits.append(lm_logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / len(self.dataloader))
        
        # Calculate additional metrics using sklearn
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        metrics = calculate_metric_with_sklearn(logits, labels)

        return ppl, metrics

@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "bert" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.bert.encoder.layer)
            input_device = lm.model.bert.encoder.layer[0].device
            output_device = lm.model.bert.encoder.layer[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.bert.embeddings.to(input_device)
            lm.model.bert.pooler.to(output_device)
            lm.model.classifier.to(output_device)
        elif "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "bert" in args.net.lower():
            lm.model.bert.encoder = lm.model.bert.encoder.to(lm.device)
        elif "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args.eval:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        if "InstaDeepAI" in args.model:
            tokenizer.eos_token = tokenizer.pad_token

        # define datasets and data collator
        train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "train.csv"), 
                                        kmer=args.kmer)
        val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "dev.csv"), 
                                        kmer=args.kmer)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "test.csv"), 
                                        kmer=args.kmer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


        trainer = transformers.Trainer(model=lm,
                                    tokenizer=tokenizer,
                                    args=args,
                                    compute_metrics=compute_metrics,
                                    train_dataset=train_dataset,
                                    eval_dataset=val_dataset,
        )

        if args.do_eval:
            logger.info("*** Evaluate ***")
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [args.task_name]
            eval_datasets = [eval_dataset]
            if args.task_name == "mnli":
                tasks.append("mnli-mm")
                valid_mm_dataset = raw_datasets["validation_mismatched"]
                if args.max_eval_samples is not None:
                    max_eval_samples = min(len(valid_mm_dataset), args.max_eval_samples)
                    valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
                eval_datasets.append(valid_mm_dataset)
                combined = {}

            for eval_dataset, task in zip(eval_datasets, tasks):
                trainer.compute_metrics = compute_metrics
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                # Add additional metrics
                max_eval_samples = (
                    args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                if task == "mnli-mm":
                    metrics = {k + "_mm": v for k, v in metrics.items()}
                if task is not None and "mnli" in task:
                    combined.update(metrics)

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)



    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--data_path",type=str,default=None)
    parser.add_argument("--preprocessing_num_workers",type=int,default=8)

    parser.add_argument("--max_eval_samples",type=int,default=None)
    parser.add_argument("--kmer",type=int,default=-1)
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="eval batch size.")    
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--do_eval", type=str, default=None)
    parser.add_argument("--attn_softmax", type=str, default="vanilla")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = lm.model.config.max_position_embeddings
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        if "InstaDeepAI" in args.model:
            tokenizer.eos_token = tokenizer.pad_token

        # define datasets and data collator
        train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "train.csv"), 
                                        kmer=args.kmer)
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "dev.csv"), 
                                        kmer=args.kmer)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                        data_path=os.path.join(args.data_path, "test.csv"), 
                                        kmer=args.kmer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            num_workers=args.preprocessing_num_workers,
            shuffle=True,
        )
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir)
    # evaluate(lm, args,logger)
    lm.model.to(lm.device)
    print(lm.device)
    evaluator = CustomEvaluator(eval_dataset, tokenizer, "cuda", args.per_device_eval_batch_size, n_samples=args.max_eval_samples)
    ppl, metrics = evaluator.evaluate(lm.model)

    # Print perplexity and additional metrics
    print(f"Perplexity: {ppl}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"Matthews Correlation: {metrics['matthews_correlation']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")

    # Save results to file
    save_results(args.output_dir, ppl, metrics)


if __name__ == "__main__":
    print(sys.argv)
    main()
