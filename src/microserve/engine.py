"""Text generation on top of the model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from microserve.model.forward import decode_step, prefill
from collections.abc import Callable

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def generate_greedy(
    model: PreTrainedModel,
    prompt_ids: list[int],
    max_tokens: int,
    eos_ids: list[int],
    timing: tuple[Callable[[], float], list[float]] | None = None,
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
    import torch
    def logit_to_idx(logit: torch.Tensor) -> int:
        item = torch.argmax(logit, dim=1).item()

        if timing is not None:
            timing[1].append(timing[0]())
        return item
    
    device = next(model.parameters()).device

    if timing is not None:
        timing[1].append(timing[0]())
    out = prefill(model, torch.tensor(prompt_ids, device=device).view(1, len(prompt_ids)))

    logit_id = logit_to_idx(out.logits_last)
    past_kv = out.past_kv
    output = [logit_id]
    tokens = 1
    eos = output[-1] in eos_ids
    while (tokens < max_tokens and not eos):
        out = decode_step(model, torch.tensor([logit_id], device=device).view(1,1), past_kv)
        logit_id = logit_to_idx(out.logits)
        past_kv = out.past_kv
        output.append(logit_id)
        eos = output[-1] in eos_ids
        tokens += 1

    return output
