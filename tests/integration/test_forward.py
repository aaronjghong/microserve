"""Prefill and decode wrappers, checked against the model's plain forward pass."""

from __future__ import annotations

from typing import Any

import pytest

from microserve.model.forward import decode_step, prefill
from tests.golden_utils import TEST_DEVICE, load_prompts, tokenize_prompt

pytestmark = [pytest.mark.gpu, pytest.mark.model]

BF16_TOLERANCE = {"atol": 2e-2, "rtol": 2e-2}
DECODE_STEPS = 8
PROMPTS = load_prompts()
PROMPT_PARAMS = [pytest.param(p, id=p.id) for p in PROMPTS]


def _input_ids(tokenizer: Any, prompt: Any) -> Any:
    import torch

    return torch.tensor([tokenize_prompt(tokenizer, prompt)], device=TEST_DEVICE)


def _full_forward_last_logits(model: Any, input_ids: Any) -> Any:
    import torch

    with torch.inference_mode():
        return model(input_ids).logits[:, -1, :]


@pytest.mark.parametrize("prompt", PROMPT_PARAMS)
def test_prefill_matches_full_forward_last_logits(
    prompt: Any, reference_model: Any, reference_tokenizer: Any
) -> None:
    """Prefill returns exactly the last-position logits of an ordinary forward pass."""
    import torch

    input_ids = _input_ids(reference_tokenizer, prompt)

    out = prefill(reference_model, input_ids, torch.ones_like(input_ids))
    expected = _full_forward_last_logits(reference_model, input_ids)

    assert out.logits_last.shape == expected.shape
    torch.testing.assert_close(out.logits_last, expected, **BF16_TOLERANCE)


@pytest.mark.parametrize("prompt", PROMPT_PARAMS)
def test_decode_steps_match_full_forward(
    prompt: Any, reference_model: Any, reference_tokenizer: Any
) -> None:
    """Each incremental decode step matches recomputing the whole sequence from scratch.

    This proves the KV cache carries exactly the right state from step to step. Both
    paths are fed the same token at each step.
    """
    import torch

    sequence = _input_ids(reference_tokenizer, prompt)
    out = prefill(reference_model, sequence, torch.ones_like(sequence))
    logits, past_kv = out.logits_last, out.past_kv

    for step in range(DECODE_STEPS):
        next_ids = logits.argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, next_ids], dim=1)

        step_out = decode_step(reference_model, next_ids, past_kv, torch.ones_like(sequence))
        expected = _full_forward_last_logits(reference_model, sequence)

        torch.testing.assert_close(
            step_out.logits, expected, **BF16_TOLERANCE,
            msg=lambda m, step=step: f"decode step {step}: {m}",
        )
        logits, past_kv = step_out.logits, step_out.past_kv


@pytest.mark.parametrize("prompt", PROMPT_PARAMS)
def test_prefill_all_ones_mask_equals_no_mask(
    prompt: Any, reference_model: Any, reference_tokenizer: Any
) -> None:
    """For a single unpadded sequence, an all-ones attention mask changes nothing."""
    import torch

    input_ids = _input_ids(reference_tokenizer, prompt)

    with_mask = prefill(reference_model, input_ids, torch.ones_like(input_ids)).logits_last
    without_mask = prefill(reference_model, input_ids, None).logits_last

    assert torch.equal(with_mask, without_mask)
