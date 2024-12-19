# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Tri Dao.

import copy
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import ModuleUtilsMixin, apply_chunking_to_forward
from DNABERT2.bert_layers import (
    BertLayer,
    BertUnpadSelfAttention,
    BertSelfOutput,
)
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
    QuantizedModule,
)
from quantization.base_quantized_model import QuantizedModel
from quantization.quantizers import QMethods
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from quantization.range_estimators import CurrentMinMaxEstimator
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
)
from transformers_language.utils import DotDict
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel

from .bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)

try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None

logger = logging.getLogger(__name__)
HAS_PAST_KEY_ATTR = tuple(map(int, transformers.__version__.split("."))) >= (4, 2, 0)


DEFAULT_QUANT_DICT = {
    # Attention
    "attn_mask_type": "add",
    # Clip `h` tensor
    "k_std": None,
    # LayerNorm
    "layer_norm_ver": "v1",
    "layer_norm_embd": False,
    "layer_norm_res_self_output": False,
    "layer_norm_res_output": False,
    "layer_norm_n_bits_unary": 8,
    "layer_norm_n_bits_binary": 8,
    "layer_norm_n_bits_params": 8,
}

def _make_quant_dict(partial_dict):
    quant_dict = DEFAULT_QUANT_DICT.copy()
    quant_dict.update(partial_dict)
    return DotDict(quant_dict)

class QuantLayerNorm(QuantizedModule):
    def __init__(self, org_module, input_quantizer, **quant_params):
        super().__init__(**quant_params)

        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        self.org_module = org_module
        self.input_quantizer = input_quantizer

        quant_params_ = quant_params.copy()
        quant_params_.update(dict(n_bits_act=self.quant_dict.layer_norm_n_bits_unary))
        self.ln_aq_mu2 = QuantizedActivation(**quant_params_)
        self.ln_aq_S = QuantizedActivation(**quant_params_)
        self.ln_aq_Sigma = QuantizedActivation(**quant_params_)
        self.ln_aq_v = QuantizedActivation(**quant_params_)

        quant_params_ = quant_params.copy()
        quant_params_.update(dict(n_bits_act=self.quant_dict.layer_norm_n_bits_binary))
        self.ln_aq_u = QuantizedActivation(**quant_params_)
        self.ln_aq_w = QuantizedActivation(**quant_params_)
        self.ln_aq_y = QuantizedActivation(**quant_params_)

        self.eps = 1e-12

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)  # mean across last dim
        mu = self.input_quantizer(mu)
        u_q = self.ln_aq_u(x - mu)

        approach = self.quant_dict.layer_norm_ver
        if approach == "v1":
            S = torch.mean(x**2.0, dim=-1, keepdim=True)
            S_q = self.ln_aq_S(S)
            mu2_q = self.ln_aq_mu2(mu * mu)
            Sigma_q = self.ln_aq_Sigma(F.relu(S_q - mu2_q, inplace=True))
        elif approach == "v2":
            Sigma = torch.mean(u_q**2.0, dim=-1, keepdim=True)
            Sigma_q = self.ln_aq_Sigma(Sigma)
        else:
            raise NotImplementedError(f"approach {approach} is not supported")

        v_q = self.ln_aq_v(torch.rsqrt(Sigma_q + self.eps))
        w_q = self.ln_aq_w(u_q * v_q)

        ## quantize gamma, beta
        gamma, beta = self.org_module.weight, self.org_module.bias

        q_gamma = SymmetricUniformQuantizer(
            n_bits=self.quant_dict.layer_norm_n_bits_params, per_channel=False
        )
        r_gamma = CurrentMinMaxEstimator()
        q_gamma.set_quant_range(*r_gamma(gamma))
        gamma_q = q_gamma(gamma)

        q_beta = SymmetricUniformQuantizer(
            n_bits=self.quant_dict.layer_norm_n_bits_params, per_channel=False
        )
        r_beta = (
            CurrentMinMaxEstimator()
        )  # MSE_Estimator(q_beta, opt_method=OptMethod.golden_section)
        q_beta.set_quant_range(*r_beta(beta))
        beta_q = q_beta(beta)

        y_q = self.ln_aq_y(w_q * gamma_q + beta_q)

        return y_q



class QuantBertEmbeddings(QuantizedModel):

    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
        super().__init__()
        quant_params_ = quant_params.copy()
        if "Et" in self.quant_dict:
          from quantization import OptMethod, RangeEstimators

          quant_params_["weight_range_method"] = RangeEstimators.MSE
          quant_params_["weight_range_options"] = dict(opt_method=OptMethod.golden_section)
        self.word_embeddings = quantize_model(org_model.word_embeddings, **quant_params_)
        # ALiBi doesn't use position embeddings
        self.token_type_embeddings = quantize_model(org_model.token_type_embeddings, **quant_params)

        self.dropout = org_model.dropout
        self.sum_input_token_type_embd_act_quantizer = QuantizedActivation(**quant_params)
        if self.quant_dict.layer_norm_embd:
            self.LayerNorm = QuantLayerNorm(
                org_module=org_model.LayerNorm,
                input_quantizer=self.sum_input_token_type_embd_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)
        
        # self.register_buffer('token_type_ids',
        #                      torch.zeros(config.max_position_embeddings,
        #                                  dtype=torch.long),
        #                      persistent=False)
        self.token_type_ids = org_model.token_type_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # great! ALiBi
            pass

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(input_shape,  # type: ignore
                                             dtype=torch.long,
                                             device=self.word_embeddings.device) # type: ignore  # yapf: disable

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.sum_input_token_type_embd_act_quantizer(embeddings)
        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedBertUnpadSelfAttentionWithExtras(QuantizedModel):
  
    def __init__(self,
        org_model, 
        position_embedding_type=None,
        softmax_fn=torch.nn.functional.softmax,
        ssm_eps=None,
        tau=None,
        max_seq_length=None,
        skip_attn=False,
        fine_tuning=False,**quant_params):
            
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
        super().__init__()

        self.num_attention_heads = org_model.num_attention_heads
        self.max_seq_length = org_model.max_seq_length
        self.softmax_fn = org_model.softmax_fn
        
        self.skip_attn = org_model.skip_attn
        self.attention_head_size = org_model.attention_head_size
        self.all_head_size = org_model.all_head_size
        self.dropout = org_model.dropout
        self.p_dropout = org_model.p_dropout
        self.Wqkv = quantize_model(org_model.Wqkv, **quant_params)
        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

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
            attention_scores = self.attn_scores_act_quantizer(attention_scores)

            attention_scores = attention_scores + bias
            attention_probs = self.softmax_fn(attention_scores, dim=-1)
            attention_probs = self.attn_probs_act_quantizer(attention_probs)
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

        context_layer = rearrange(attention, 'nnz h d -> nnz (h d)')
        context_layer = self.context_act_quantizer(context_layer)
        return context_layer


# Copy of transformer's library BertSelfOutput that will not be caught by surgery methods looking for HF BERT modules.
class QuantizedBertSelfOutput(QuantizedModel):

    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
        super().__init__()
        self.dense = quantize_model(org_model.dense, **quant_params)
        self.res_act_quantizer = QuantizedActivation(**quant_params)

        if self.quant_dict.layer_norm_res_self_output:
            self.LayerNorm = QuantLayerNorm(
                org_module=org_model.LayerNorm,
                input_quantizer=self.res_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

        self.dropout = org_model.dropout

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.res_act_quantizer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class QuantizedBertUnpadAttention(QuantizedModel):
    """Chains attention, Dropout, and LayerNorm for Mosaic BERT."""

    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()
        attention_specials = {
            BertSelfOutput: QuantizedBertSelfOutput,
            BertUnpadSelfAttentionWithExtras: QuantizedBertUnpadSelfAttentionWithExtras,
        }
        self.self =  quantize_model(
            org_model.self, specials=attention_specials, **quant_params
        )
        # self.self = QuantizedBertUnpadSelfAttentionWithExtras(org_model.self, **quant_params)
        self.output = QuantizedBertSelfOutput(org_model.output, **quant_params)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.

        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices,
                                attn_mask, bias)
        if subset_idx is not None:
            return self.output(index_first_axis(self_output, subset_idx),
                               index_first_axis(input_tensor, subset_idx))
        else:
            return self.output(self_output, input_tensor)


class QuantizedBertGatedLinearUnitMLP(QuantizedModel):
    """Applies the FFN at the end of each Mosaic BERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
        super().__init__()

        self.config = org_model.config
        self.gated_layers = quantize_model(org_model.gated_layers, **quant_params)
        self.act = quantize_model(org_model.act, **quant_params)
        self.wo = quantize_model(org_model.wo, **quant_params)
        self.dropout = org_model.dropout
        self.res_act_quantizer = QuantizedActivation(**quant_params)
        if self.quant_dict.layer_norm_res_output:
            self.layernorm = QuantLayerNorm(
                org_module=org_model.layernorm,
                input_quantizer=self.res_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.layernorm = quantize_model(org_model.layernorm, **quant_params)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, :self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class QuantizedBertLayer(QuantizedModel):
    """Composes the Mosaic BERT attention and FFN blocks into a single layer."""
    
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
        super(QuantizedBertLayer, self).__init__()
        self.attention = QuantizedBertUnpadAttention(org_model.attention, **quant_params)
        self.mlp = QuantizedBertGatedLinearUnitMLP(org_model.mlp, **quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        attention_output = self.attention(hidden_states, cu_seqlens, seqlen,
                                          subset_idx, indices, attn_mask, bias)
        layer_output = self.mlp(attention_output)
        return layer_output


# class QuantizedBertEncoder(QuantizedModel):
#     """A stack of BERT layers providing the backbone of Mosaic BERT.

#     This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertEncoder`,
#     but with substantial modifications to implement unpadding and ALiBi.

#     Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
#     at padded tokens, and pre-computes attention biases to implement ALiBi.
#     """

#     def __init__(self, org_model, **quant_params):
#         self.quant_dict = _make_quant_dict(quant_params["quant_dict"])
#         super().__init__()
        
#         self.layer = nn.ModuleList(
#             [QuantizedBertLayer(layer, **quant_params) for layer in org_model.layer])

#         self.num_attention_heads = org_model.num_attention_heads

#         # The alibi mask will be dynamically expanded if it is too small for
#         # the input the model receives. But it generally helps to initialize it
#         # to a reasonably large size to help pre-allocate CUDA memory.
#         # The default `alibi_starting_size` is 512.
#         self._current_alibi_size = int(org_model._current_alibi_size)
#         self.alibi = torch.zeros(
#             (1, self.num_attention_heads, self._current_alibi_size,
#              self._current_alibi_size))
#         self.rebuild_alibi_tensor(size=org_model._current_alibi_size)

#     def rebuild_alibi_tensor(self,
#                              size: int,
#                              device: Optional[Union[torch.device, str]] = None):
#         # Alibi
#         # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
#         # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
#         # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
#         # will be applied, it is necessary to construct the diagonal mask.
#         n_heads = self.num_attention_heads

#         def _get_alibi_head_slopes(n_heads: int) -> List[float]:

#             def get_slopes_power_of_2(n_heads: int) -> List[float]:
#                 start = (2**(-2**-(math.log2(n_heads) - 3)))
#                 ratio = start
#                 return [start * ratio**i for i in range(n_heads)]

#             # In the paper, they only train models that have 2^a heads for some a. This function
#             # has some good properties that only occur when the input is a power of 2. To
#             # maintain that even when the number of heads is not a power of 2, we use a
#             # workaround.
#             if math.log2(n_heads).is_integer():
#                 return get_slopes_power_of_2(n_heads)

#             closest_power_of_2 = 2**math.floor(math.log2(n_heads))
#             slopes_a = get_slopes_power_of_2(closest_power_of_2)
#             slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
#             slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
#             return slopes_a + slopes_b

#         context_position = torch.arange(size, device=device)[:, None]
#         memory_position = torch.arange(size, device=device)[None, :]
#         relative_position = torch.abs(memory_position - context_position)
#         # [n_heads, max_token_length, max_token_length]
#         relative_position = relative_position.unsqueeze(0).expand(
#             n_heads, -1, -1)
#         slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
#         alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
#         # [1, n_heads, max_token_length, max_token_length]
#         alibi = alibi.unsqueeze(0)
#         assert alibi.shape == torch.Size([1, n_heads, size, size])

#         self._current_alibi_size = size
#         self.alibi = alibi

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         output_all_encoded_layers: Optional[bool] = True,
#         subset_mask: Optional[torch.Tensor] = None,
#     ) -> List[torch.Tensor]:

#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(
#             dtype=torch.float32)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         attention_mask_bool = attention_mask.bool()
#         batch, seqlen = hidden_states.shape[:2]
#         # Unpad inputs and mask. It will remove tokens that are padded.
#         # Assume ntokens is total number of tokens (padded and non-padded)
#         # and ntokens_unpad is total number of non-padded tokens.
#         # Then unpadding performs the following compression of the inputs:
#         # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
#         hidden_states, indices, cu_seqlens, _ = unpad_input(
#             hidden_states, attention_mask_bool)

#         # Add alibi matrix to extended_attention_mask
#         if self._current_alibi_size < seqlen:
#             # Rebuild the alibi tensor when needed
#             warnings.warn(
#                 f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
#             )
#             self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
#         elif self.alibi.device != hidden_states.device:
#             # Device catch-up
#             self.alibi = self.alibi.to(hidden_states.device)
#         alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
#         attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
#         alibi_attn_mask = attn_bias + alibi_bias

#         all_encoder_layers = []
#         if subset_mask is None:
#             for layer_module in self.layer:
#                 hidden_states = layer_module(hidden_states,
#                                              cu_seqlens,
#                                              seqlen,
#                                              None,
#                                              indices,
#                                              attn_mask=attention_mask,
#                                              bias=alibi_attn_mask)
#                 if output_all_encoded_layers:
#                     all_encoder_layers.append(hidden_states)
#             # Pad inputs and mask. It will insert back zero-padded tokens.
#             # Assume ntokens is total number of tokens (padded and non-padded)
#             # and ntokens_unpad is total number of non-padded tokens.
#             # Then padding performs the following de-compression:
#             #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
#             hidden_states = pad_input(hidden_states, indices, batch, seqlen)
#         else:
#             for i in range(len(self.layer) - 1):
#                 layer_module = self.layer[i]
#                 hidden_states = layer_module(hidden_states,
#                                              cu_seqlens,
#                                              seqlen,
#                                              None,
#                                              indices,
#                                              attn_mask=attention_mask,
#                                              bias=alibi_attn_mask)
#                 if output_all_encoded_layers:
#                     all_encoder_layers.append(hidden_states)
#             subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
#                                        as_tuple=False).flatten()
#             hidden_states = self.layer[-1](hidden_states,
#                                            cu_seqlens,
#                                            seqlen,
#                                            subset_idx=subset_idx,
#                                            indices=indices,
#                                            attn_mask=attention_mask,
#                                            bias=alibi_attn_mask)

#         if not output_all_encoded_layers:
#             all_encoder_layers.append(hidden_states)
#         return all_encoder_layers


class QuantizedBertPooler(QuantizedModel):

    def __init__(self, org_model, **quant_params):
        super(QuantizedBertPooler, self).__init__()
        self.dense_act = quantize_model(
            nn.Sequential(org_model.dense, org_model.activation), **quant_params
        )

    def forward(self,
                hidden_states: torch.Tensor,
                pool: Optional[bool] = True) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output


class QuantizedBertPredictionHeadTransform(QuantizedModel):

    def __init__(self, org_model, **quant_params):
        super().__init__()
        if org_model.transform_act_fn == F.gelu:
            transform_act_fn = nn.GELU()
        else:
            raise ValueError(
                f'transform activation fn "{org_model.transform_act_fn}" ' f"is not supported"
            )
        self.dense_act = quantize_model(
            nn.Sequential(org_model.dense, org_model.transform_act_fn), **quant_params
        )
        self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class QuantizedBertModel(QuantizedModel, ModuleUtilsMixin):
    """Overall BERT model.

    Args:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, org_model, **quant_params):
        super().__init__()
        self.embeddings = QuantBertEmbeddings(org_model.embeddings, **quant_params)
        self.encoder = quantize_model(
            org_model.encoder, specials={BertLayer: QuantizedBertLayer}, **quant_params
        )
        self.pooler = (
            QuantizedBertPooler(org_model.pooler, **quant_params)
            if org_model.pooler is not None
            else None
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_all_encoded_layers: Optional[bool] = False,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids,
                                           position_ids)

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask)

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(
                sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][
                masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


###################
# Bert Heads
###################
class QuantizedBertLMPredictionHead(QuantizedModel):

    def __init__(self, org_model, **quant_params):
        super().__init__()
        self.transform = QuantizedBertPredictionHeadTransform(org_model.transform, **quant_params)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = quantize_model(org_model.decoder, **quant_params)
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states






class QuantizedBertForMaskedLM(QuantizedModel):

    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__()

        self.bert = QuantizedBertModel(org_model=org_model.bert, **quant_params)
        self.cls = org_model.cls
        self.config = org_model.config

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # labels should be a `torch.LongTensor` of shape
        # `(batch_size, sequence_length)`. These are used for computing the
        #  masked language modeling loss.
        #
        # Indices should be in `[-100, 0, ..., config.vocab_size]` (see
        # `input_ids` docstring) Tokens with indices set to `-100` are ignored
        # (masked), the loss is only computed for the tokens with labels in `[0,
        # ..., config.vocab_size]`
        #
        # Prediction scores are only computed for masked tokens and the (bs,
        # seqlen) dimensions are flattened
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')

        if labels is None:
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            masked_tokens_mask=masked_tokens_mask,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            masked_token_idx = torch.nonzero(labels.flatten() > 0,
                                             as_tuple=False).flatten()
            loss = loss_fct(prediction_scores,
                            labels.flatten()[masked_token_idx])

            assert input_ids is not None, 'Coding error; please open an issue'
            batch, seqlen = input_ids.shape[:2]
            prediction_scores = rearrange(index_put_first_axis(
                prediction_scores, masked_token_idx, batch * seqlen),
                                          '(b s) d -> b s d',
                                          b=batch)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs[0],
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')

        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_zeros((attention_mask.shape[0], 1))
        ],
                                   dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}



class QuantizedBertForSequenceClassification(QuantizedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__()

        self.num_labels = org_model.num_labels
        self.config = org_model.config

        self.bert = QuantizedBertModel(org_model=org_model.bert, **quant_params)
        classifier_dropout = (self.config.classifier_dropout
                              if self.config.classifier_dropout is not None else
                              self.config.hidden_dropout_prob)
        self.dropout = org_model.dropout
        self.classifier = org_model.classifier
        # Initialize weights and apply final processing
        # self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or
                                              labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=None,
        )
