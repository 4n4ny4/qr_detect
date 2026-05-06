import os
from typing import Optional, Union

import torch
from torch.nn import CrossEntropyLoss

from .transformers_compat import disable_optional_torchvision

disable_optional_torchvision()

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

# Mistral-specific imports. Tries the modern modeling_mistral re-exports first
# (transformers >= 4.46 after the AttentionInterface migration), falls back to
# shared utilities on older transformers.
try:
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS,
        MistralForCausalLM as TransformersMistralForCausalLM,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
except ImportError:
    from transformers.models.mistral.modeling_mistral import (
        MistralForCausalLM as TransformersMistralForCausalLM,
        apply_rotary_pos_emb,
    )
    try:
        from transformers.modeling_utils import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )
    except ImportError:
        from transformers.models.llama.modeling_llama import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )

from .custom_cache import DynamicCacheWithQuery


def _mistral_attention_forward_with_query_cache(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Mistral attention forward with QRRetriever's query-state cache support.

    Mirrors the OLMo monkey-patch in `custom_modeling_olmo.py`, adapted for
    Mistral. The single functional change vs upstream Mistral attention is the
    query-state slice into `DynamicCacheWithQuery`. Differences from OLMo:
      * No `clip_qkv` (Mistral has no `clip_qkv` config).
      * No sliding-window handling (Mistral-7B-Instruct-v0.3 ships with
        `sliding_window=null`; the chosen attention_interface handles it).
      * GQA (32 attention heads, 8 KV heads, head_dim=128) is handled inside
        the attention_interface via `repeat_kv`, so no extra logic is needed.
    """
    if past_key_values is None and "past_key_value" in kwargs:
        past_key_values = kwargs.pop("past_key_value")

    kwargs.pop("output_attentions", None)
    kwargs.pop("use_cache", None)

    if position_embeddings is None:
        raise ValueError(
            "Mistral QRRetriever support requires position_embeddings from MistralModel."
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if isinstance(past_key_values, DynamicCacheWithQuery):
            query_states_to_cache = query_states[:, :, past_key_values._query_indices, :]
            key_states, value_states = past_key_values.update(
                query_states_to_cache,
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        else:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

    attention_interface = eager_attention_forward
    attn_impl = getattr(self.config, "_attn_implementation", "eager")
    if attn_impl != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _mistral_mlp_forward_chunked(self, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Chunked MistralMLP forward. Caps peak FFN activation memory on long
    detection prefills (LME ~18-25K tokens after truncation, NQ similar).
    Mirrors the post-`bfdc025` `Qwen2MLP.forward` chunk loop. Chunk size set
    via `QRRETRIEVER_MLP_CHUNK_SIZE` (default 2048). Disabled when training
    or when chunk_size <= 0.
    """
    chunk_size = int(os.environ.get("QRRETRIEVER_MLP_CHUNK_SIZE", "2048"))
    if not torch.is_grad_enabled() and chunk_size > 0 and hidden_state.shape[-2] > chunk_size:
        return torch.cat(
            [
                self.down_proj(self.act_fn(self.gate_proj(chunk)) * self.up_proj(chunk))
                for chunk in hidden_state.split(chunk_size, dim=-2)
            ],
            dim=-2,
        )
    return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def _patch_mistral_query_cache_and_mlp(model) -> None:
    for layer in model.layers:
        layer.self_attn.forward = _mistral_attention_forward_with_query_cache.__get__(
            layer.self_attn,
            layer.self_attn.__class__,
        )
        layer.mlp.forward = _mistral_mlp_forward_chunked.__get__(
            layer.mlp,
            layer.mlp.__class__,
        )


class MistralForCausalLM(TransformersMistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _patch_mistral_query_cache_and_mlp(self.model)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        compute_logits: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }
        if output_attentions is not None:
            model_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            model_kwargs["output_hidden_states"] = output_hidden_states
        if return_dict is not None:
            model_kwargs["return_dict"] = return_dict

        outputs = self.model(**model_kwargs, **kwargs)
        hidden_states = outputs[0]
        past_key_values_out = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
        if past_key_values_out is None and len(outputs) > 1:
            past_key_values_out = outputs[1]
        hidden_states_out = getattr(outputs, "hidden_states", None)
        attentions_out = getattr(outputs, "attentions", None)

        if compute_logits:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = None

        loss = None
        if labels is not None:
            if logits is None:
                raise ValueError("Cannot compute loss when compute_logits=False.")
            if hasattr(self, "loss_function"):
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if return_dict is False:
            output = (logits, past_key_values_out, hidden_states_out, attentions_out)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values_out,
            hidden_states=hidden_states_out,
            attentions=attentions_out,
        )


__all__ = ["MistralForCausalLM"]
