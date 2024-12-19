import torch
import os
from einops import rearrange
import math
from functools import partial
from typing import Optional, Tuple, Union
import warnings
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BertConfig,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
import argparse
from transformers import CONFIG_MAPPING
from smoothquant.calibration import get_act_scales
from hopfield_code.softmax import SOFTMAX_MAPPING
try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None
from torch import nn
import logging

from smoothquant.bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)



import warnings 
 
def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))

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



def build_model_and_tokenizer(args):
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()

    # Display config after changes
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

    if args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code
        )
    else:
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
            max_seq_length=args.seq_len,
            skip_attn=False,
            fine_tuning=False,
        )

        # copy loaded weights
        if args.model_name_or_path is not None:
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
    print(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )

    return  model,tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
            )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='/projects/p32013/DNA/smoothquant/cache'
    )
    parser.add_argument(
        '--model_type',
        default=None
    )
    parser.add_argument(
        "--trust_remote_code",
        action='store_true',
        default=True,
    )
    parser.add_argument(
        '--config_name',
        default=None
    )
    parser.add_argument(
        '--use_slow_tokenizer',
        action='store_true',
        default=False,
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args)

    if not os.path.exists(args.dataset_path) and not os.path.exists(args.train_file):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
        raise FileNotFoundError

    act_scales = get_act_scales(
        model, tokenizer, args.train_file, args.validation_file, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
