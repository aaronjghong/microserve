"""Text generation on top of the model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def generate_greedy(
    model: PreTrainedModel,
    prompt_ids: list[int],
    max_tokens: int,
    eos_ids: list[int],
) -> list[int]:
    """Generate a continuation of one prompt by always taking the most likely next token.

    Prefill the prompt once, then decode one token at a time until an EOS token is produced
    or `max_tokens` tokens have been generated.

    Returns only the generated tokens, not the prompt.

    Invariants:
      - Token for token identical to Hugging Face `generate(do_sample=False)` on the same
        model, dtype, and device.
      - `len(result) <= max_tokens`.
      - If an EOS token is generated, it is the last element of the result: it is included,
        and generation stops right after it.
    """
    raise NotImplementedError
