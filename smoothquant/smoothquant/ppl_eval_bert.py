import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from smooth import smooth_lm
from transformers.models.bert.configuration_bert import BertConfig

from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING

from fake_quant import quantize_model, quantize_dnabert
import tqdm

from datasets import load_dataset
import argparse

from typing import Optional, Dict, Sequence, Tuple, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import csv
import sklearn
from sklearn import metrics
import logging



parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument(
    "--act_scales_path",
    type=str,
    default=None,
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--validation_file", type=str, default=None)
parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
parser.add_argument("--output_dir", type=str, default=None)


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples
seed=args.seed

torch.manual_seed(seed)

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



class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

# Modify the Evaluator class to calculate custom metrics
class CustomEvaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = n_samples

        self.dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer),
        )

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        all_logits = []
        all_labels = []

        for batch in tqdm.tqdm(self.dataloader, desc="Evaluating..."):
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

tokenizer = AutoTokenizer.from_pretrained(model_path)

# define datasets and data collator
eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.validation_file, kmer=-1)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

evaluator = CustomEvaluator(eval_dataset, tokenizer, "cuda", n_samples=n_samples)
config = BertConfig.from_pretrained(model_path, num_labels=eval_dataset.num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    config=config,
    cache_dir=args.cache_dir,
    trust_remote_code=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for layer_idx in range(len(model.bert.encoder.layer)):
    old_self = model.bert.encoder.layer[layer_idx].attention.self
    print("----------------------------------------------------------")
    print("Inside BERT custom attention")
    print("----------------------------------------------------------")
    new_self = BertUnpadSelfAttentionWithExtras(
        config,
        position_embedding_type=None,
        softmax_fn=SOFTMAX_MAPPING["softmax1"],
        ssm_eps=None,
        tau=None,
        max_seq_length=tokenizer.model_max_length,
        skip_attn=False,
        fine_tuning=False,
    ).cuda()

    # copy loaded weights
    if model_path is not None:
        new_self.load_state_dict(old_self.state_dict(), strict=False)
    model.bert.encoder.layer[layer_idx].attention.self = new_self


if args.smooth:
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)
if args.quantize:
    model = quantize_dnabert(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )
    print(model)

# Update the evaluation process
ppl, metrics = evaluator.evaluate(model)

# Print perplexity and additional metrics
print(f"Perplexity: {ppl}")
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1 Score: {metrics['f1']}")
print(f"Matthews Correlation: {metrics['matthews_correlation']}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")

# Save results to file
save_results(args.output_dir, ppl, metrics)