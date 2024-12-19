# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math
from functools import partial
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from einops import rearrange

from quantization.utils import BaseEnumOptions
# from enum import Enum
from softmax import clipped_softmax
from bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)
try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None


def logit(p, eps=1e-16):
    p = np.clip(p, eps, 1 - eps)
    return -np.log(1 / p - 1)


class AttentionGateType(BaseEnumOptions):
    none = 0
    unconditional_per_head = 1
    conditional_per_head = 2
    conditional_per_token = 3


class BertSelfAttentionWithExtras(nn.Module):
    def __init__(
        self,
        config,
        position_embedding_type=None,
        softmax_fn=torch.nn.functional.softmax,
        alpha=None,
        ssm_eps=None,
        tau=None,
        max_seq_length=None,
        skip_attn=False,
        attn_gate_type=AttentionGateType.none,
        attn_gate_init=None,
        attn_gate_mlp=False,
        attn_gate_mlp2=False,
        attn_gate_linear_all_features=False,
        fine_tuning=False,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

        # YB: capture the input and output of the softmax
        self.attn_scores = nn.Identity()  # before attention mask
        self.attn_probs_before_dropout = nn.Identity()
        self.attn_probs_after_dropout = nn.Identity()

        self.alpha = alpha
        self.ssm_eps = ssm_eps
        self.tau = tau
        self.max_seq_length = max_seq_length

        # define softmax function
        if self.alpha is not None:
            assert self.max_seq_length is not None
            gamma = -self.alpha / self.max_seq_length
            self.softmax_fn = partial(clipped_softmax, gamma=gamma, eta=1.0)
        else:
            self.softmax_fn = softmax_fn

        self.skip_attn = skip_attn

        # attention gating
        self.last_gate_avg_prob = None
        self.last_gate_all_probs = None

        self.attn_gate_type = attn_gate_type
        self.attn_gate_init = attn_gate_init
        self.attn_gate_mlp = attn_gate_mlp
        self.attn_gate_mlp2 = attn_gate_mlp2
        self.attn_gate_linear_all_features = attn_gate_linear_all_features

        self.alpha = None
        self.gate_fn = torch.sigmoid
        self.pooling_fn = partial(torch.mean, dim=1, keepdims=True)

        self.fine_tuning = fine_tuning

        # gate scaling factor
        self.gate_scaling_factor = 1.0
        if self.fine_tuning and self.attn_gate_init is not None:
            self.gate_scaling_factor = 1.0 / self.attn_gate_init

        # define gate
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            init_alpha = torch.zeros(size=(self.num_attention_heads,))
            self.alpha = nn.Parameter(init_alpha, requires_grad=True)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            if self.attn_gate_linear_all_features:
                self.alpha = nn.Linear(self.all_head_size, self.num_attention_heads, bias=True)

            else:  # separate predictors for each head
                module_list = []
                for _ in range(self.num_attention_heads):
                    if self.attn_gate_mlp:
                        fc = nn.Sequential(
                            nn.Linear(
                                self.attention_head_size, self.attention_head_size // 4, bias=True
                            ),
                            nn.ReLU(),
                            nn.Linear(self.attention_head_size // 4, 1, bias=True),
                        )
                    elif self.attn_gate_mlp2:
                        fc = nn.Sequential(
                            nn.Linear(
                                self.attention_head_size, self.attention_head_size, bias=True
                            ),
                            nn.ReLU(),
                            nn.Linear(self.attention_head_size, 1, bias=True),
                        )
                    else:
                        fc = nn.Linear(self.attention_head_size, 1, bias=True)

                        if self.attn_gate_init is not None:
                            init_bias = logit(self.attn_gate_init)
                            torch.nn.init.constant_(fc.bias, init_bias)

                        if self.fine_tuning:
                            # init to a very small values
                            torch.nn.init.normal_(fc.weight, mean=0.0, std=0.01)

                    module_list.append(fc)
                self.alpha = nn.ModuleList(module_list)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) ->  Tuple[torch.Tensor, torch.Tensor]:
        if self.skip_attn:
            out = torch.zeros_like(hidden_states)
            return (out,)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores + relative_position_scores_query + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # YB: for logging softmax input
        attention_scores = self.attn_scores(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # MN: uses our own SM function as specified in the config
        attention_probs = self.softmax_fn(attention_scores, dim=-1)

        # YB: for logging softmax output
        attention_probs = self.attn_probs_before_dropout(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # YB: for logging softmax output
        attention_probs = self.attn_probs_after_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # *** Gating ***
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            gate = self.gate_fn(self.alpha)  # (H,)
            context_layer *= gate.view(-1, 1, 1)  # (B, H, T, d_head)

            self.last_gate_avg_prob = gate.view(-1)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            x = hidden_states  # (B, T, d_model)

            if self.attn_gate_linear_all_features:  # assume per_token
                alpha = self.alpha(x)  # (B, T, H)
                gate = self.gate_fn(alpha)
                gate = gate.permute(0, 2, 1).contiguous()  # (B, H, T)
                gate = gate.unsqueeze(3)  # (B, H, T, 1)

            else:
                x = self.transpose_for_scores(x)  # (B, H, T, d_head)

                alpha = []
                for head_idx in range(self.num_attention_heads):
                    x_head = x[:, head_idx, ...]  # (B, T, d_head)
                    fc_head = self.alpha[head_idx]
                    alpha_head = fc_head(x_head)  # (B, T, 1)
                    if self.attn_gate_type == AttentionGateType.conditional_per_head:
                        alpha_head = self.pooling_fn(alpha_head)  # (B, 1, 1)
                    alpha.append(alpha_head)
                alpha = torch.stack(alpha, dim=1)  # (B, H, *, 1)
                gate = self.gate_fn(alpha)

            context_layer *= gate * self.gate_scaling_factor

            self.last_gate_all_probs = gate  # all gates to see the distributions
            avg_gate = gate.mean(dim=0)
            self.last_gate_avg_prob = avg_gate.view(self.num_attention_heads, -1).mean(dim=1)

        # <end elif>

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)


        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

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
