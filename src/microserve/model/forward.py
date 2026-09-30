"""Thin wrappers around a Hugging Face model's forward pass for prefill and decode.

The KV cache here is Hugging Face's own `past_key_values`, passed through unchanged.

torch and transformers types are imported for type checking only, so importing this module
stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class PrefillOutput(NamedTuple):
    """Result of running the whole prompt through the model once.

    logits_last: `[B, V]` logits at the final prompt position only.
    past_kv:     the model's KV cache covering every prompt position.
    """

    logits_last: torch.Tensor
    past_kv: Any


class DecodeOutput(NamedTuple):
    """Result of feeding one new token per sequence through the model.

    logits:  `[B, V]` logits for the next token.
    past_kv: the KV cache, now one position longer.
    """

    logits: torch.Tensor
    past_kv: Any


def prefill(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> PrefillOutput:
    """One forward pass over the whole prompt, `input_ids` of shape `[B, T]`.

    Invariants:
      - `logits_last` equals the full forward's logits at position `T - 1`, and nothing
        else is kept (the `[B, T, V]` tensor is not returned or retained).
      - Passing an all-ones `attention_mask` gives the same result as passing none.
      - No autograd state is recorded.
    """
    raise NotImplementedError


def decode_step(
    model: PreTrainedModel,
    next_ids: torch.Tensor,
    past_kv: Any,
    attention_mask: torch.Tensor | None = None,
) -> DecodeOutput:
    """One forward pass for a single new token per sequence, `next_ids` of shape `[B, 1]`.

    `attention_mask`, when given, covers every position so far, including the new token.

    Invariants:
      - The logits equal what a full forward over `prompt + all tokens so far` would give at
        its last position, so incremental decoding matches recomputing from scratch.
      - No autograd state is recorded.
    """
    raise NotImplementedError
