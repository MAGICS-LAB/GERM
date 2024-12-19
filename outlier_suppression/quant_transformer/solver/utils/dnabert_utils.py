import yaml
import os
from easydict import EasyDict
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
)
import transformers
import logging
from datasets import load_dataset
from hopfield_code.softmax import SOFTMAX_MAPPING
from torch import nn
import torch
import warnings
try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None

from typing import Optional, Dict, Sequence, Tuple, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import csv
import sklearn
import random

logger = logging.getLogger(__name__)

class BertUnpadSelfAttentionWithExtras(nn.Module):
  
    def __init__(self, config,
        position_embedding_type=None,
        softmax_fn=torch.nn.functional.softmax,
        ssm_eps=None,
        tau=None,
        max_seq_length=None,
        skip_attn=False,
        fine_tuning=False,):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.max_seq_length = max_seq_length
        self.softmax_fn = softmax_fn
        
        self.skip_attn = skip_attn
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)

        # Warn if defaulting to pytorch because of import issues
        if flash_attn_qkvpacked_func is None:
            warnings.warn(
                'Unable to import Triton; defaulting MosaicBERT attention implementation to pytorch (this will reduce throughput when using this model).'
            )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_in_batch: int, indices: torch.Tensor,
                attn_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Perform self-attention.
        If dropout is zero, then we can use the Triton kernel, so we do that. However, if not, we send through a standard PyTorch
        implementation of self-attention.
        The arguments are unpadded, and our implementations of attention require padded arguments,
        so we first call `pad_input`. Once we compute attention, we re-unpad our outputs for the other layers.
        The pad/unpad operations add overhead, but not sending pad tokens through ffs saves compute.
        It is possible to write an unpadded implementation of attention (in Triton and PyTorch), which we will eventually do.
        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen_in_batch: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen_in_batch)
            bias: (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        Returns:
            attention: (total_nnz, dim)
        """
        self.Wqkv = self.Wqkv.to(hidden_states.device).to(hidden_states.dtype)
        qkv = self.Wqkv(hidden_states)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1,
                        max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)
        if self.p_dropout or flash_attn_qkvpacked_func is None:
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = self.softmax_fn(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs).to(v.dtype)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1,
                                                                 3)  # b s h d
        else:
            # Triton implementation only supports 0 attention dropout
            convert_dtype = qkv.dtype not in [torch.float16, torch.bfloat16]
            if convert_dtype:
                # Triton implementation only supports fp16 and bf16
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.float16)
                bias_dtype = bias.dtype
                bias = bias.to(torch.float16)
                attention = flash_attn_qkvpacked_func(qkv, bias)
                attention = attention.to(orig_dtype)
                bias = bias.to(bias_dtype)
            else:
                attention = flash_attn_qkvpacked_func(qkv, bias)

        # attn_mask is 1 for attend and 0 for don't
        attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
        return rearrange(attention, 'nnz h d -> nnz (h d)')

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

    def shuffle(self):
        """Shuffle the dataset and return a new dataset."""
        indices = list(range(len(self.input_ids)))
        random.shuffle(indices)

        shuffled_input_ids = self.input_ids[indices]
        shuffled_attention_mask = self.attention_mask[indices]
        shuffled_labels = [self.labels[i] for i in indices]

        # Create a new dataset instance with shuffled data
        shuffled_dataset = SupervisedDataset.__new__(SupervisedDataset)
        shuffled_dataset.input_ids = shuffled_input_ids
        shuffled_dataset.attention_mask = shuffled_attention_mask
        shuffled_dataset.labels = shuffled_labels
        shuffled_dataset.num_labels = self.num_labels

        return shuffled_dataset

    def select(self, indices: List[int]):
        """Select a subset of the dataset based on indices and return a new dataset."""
        selected_input_ids = self.input_ids[indices]
        selected_attention_mask = self.attention_mask[indices]
        selected_labels = [self.labels[i] for i in indices]

        # Create a new dataset instance with selected data
        selected_dataset = SupervisedDataset.__new__(SupervisedDataset)
        selected_dataset.input_ids = selected_input_ids
        selected_dataset.attention_mask = selected_attention_mask
        selected_dataset.labels = selected_labels
        selected_dataset.num_labels = self.num_labels

        return selected_dataset


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


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def load_dataset_labels(data_args, model_args):
    # datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    return raw_datasets
    


def load_model(config_model, config_data):
    # num_labels first to indentity the classification heads
    config_kwargs = {
        "cache_dir": config_model.cache_dir,
        "trust_remote_code": config_model.trust_remote_code,
    }
    if config_model.config_name:
        config = AutoConfig.from_pretrained(config_model.config_name, **config_kwargs)
    elif config_model.model_name_or_path:
        config = AutoConfig.from_pretrained(config_model.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[config_model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Display config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(config))

    # Load tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": config_model.trust_remote_code,
    }
    if config_model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config_model.model_name_or_path, use_fast=not config_model.use_slow_tokenizer, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this "
            "script. You can do it from another script, save it, and load it from here, "
            "using --tokenizer_name."
        )

    # Load and prepare model
    if config_model.model_name_or_path:
        config = BertConfig.from_pretrained(config_model.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            config_model.model_name_or_path,
            from_tf=bool(".ckpt" in config_model.model_name_or_path),
            config=config,
            cache_dir=config_model.cache_dir,
            trust_remote_code=config_model.trust_remote_code
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
            softmax_fn=SOFTMAX_MAPPING["softmax1"],
            ssm_eps=None,
            tau=None,
            max_seq_length=config_model.max_seq_length,
            skip_attn=False,
            fine_tuning=False,
        )

        # copy loaded weights
        if config_model.model_name_or_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.bert.encoder.layer[layer_idx].attention.self = new_self
    

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

    return tokenizer, model


def preprocess_dataset(config_data, training_args, raw_datasets, tokenizer):
    # Preprocessing the datasets.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if config_data.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if config_data.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(config_data.max_seq_length, tokenizer.model_max_length)

    if config_data.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if config_data.pad_to_max_length else False

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

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not config_data.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=config_data.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not config_data.overwrite_cache,
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

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not config_data.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=config_data.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not config_data.overwrite_cache,
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

        with training_args.main_process_first(desc="grouping texts together"):
            if not config_data.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=config_data.preprocessing_num_workers,
                    load_from_cache_file=not config_data.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
    return tokenized_datasets

# 感觉没什么用，先留着
def check_return_data(raw_datasets, data_type, do_type, max_samples):
    if not do_type:
        return None
    if do_type and data_type not in raw_datasets:
        raise ValueError(f'do- {data_type} requires a {data_type} dataset')
    type_dataset = raw_datasets[data_type]
    if max_samples is not None:
        type_dataset = type_dataset.shuffle().select(range(max_samples))
    return type_dataset
