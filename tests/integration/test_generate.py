"""Greedy generation, compared token for token with Hugging Face `generate`."""

from __future__ import annotations

from typing import Any

import pytest

from microserve.engine import generate_greedy
from tests.golden_utils import GREEDY_32, TEST_MODEL, load_golden, load_prompts, tokenize_prompt

pytestmark = [pytest.mark.gpu, pytest.mark.model]

MAX_TOKENS = 32
SHORT_MAX_TOKENS = 4
PROMPTS = load_prompts()


def _prompt_ids_matching_golden(tokenizer: Any, prompt: Any, golden: dict[str, Any]) -> list[int]:
    prompt_ids = tokenize_prompt(tokenizer, prompt)
    assert prompt_ids == golden["prompt_token_ids"], (
        f"prompt {prompt.id!r} tokenizes differently than when the goldens were made; "
        "the tokenizer or the prompt file changed"
    )
    return prompt_ids


@pytest.mark.parametrize("prompt", [pytest.param(p, id=p.id) for p in PROMPTS])
def test_greedy_matches_hf_golden(
    prompt: Any, reference_model: Any, reference_tokenizer: Any, golden_manifest: dict[str, Any]
) -> None:
    """Greedy generation produces exactly the tokens Hugging Face `generate` produced."""
    golden = load_golden(TEST_MODEL, prompt.id, GREEDY_32)
    prompt_ids = _prompt_ids_matching_golden(reference_tokenizer, prompt, golden)

    tokens = generate_greedy(
        reference_model, prompt_ids, MAX_TOKENS, golden_manifest["eos_token_ids"]
    )

    assert tokens == golden["token_ids"]


def test_greedy_stops_at_eos(
    reference_model: Any, reference_tokenizer: Any, golden_manifest: dict[str, Any]
) -> None:
    """Generation ends right after EOS instead of running on to the token limit."""
    eos_ids = golden_manifest["eos_token_ids"]
    early = []
    for prompt in PROMPTS:
        golden = load_golden(TEST_MODEL, prompt.id, GREEDY_32)
        tokens = golden["token_ids"]
        if len(tokens) < MAX_TOKENS and tokens and tokens[-1] in eos_ids:
            early.append((prompt, golden))
    assert early, "no fixture prompt's golden output ends with EOS early; add one to the fixtures"

    for prompt, golden in early:
        prompt_ids = _prompt_ids_matching_golden(reference_tokenizer, prompt, golden)
        tokens = generate_greedy(reference_model, prompt_ids, MAX_TOKENS, eos_ids)

        assert len(tokens) == len(golden["token_ids"]), f"prompt {prompt.id!r}"
        assert tokens == golden["token_ids"], f"prompt {prompt.id!r}"


def test_greedy_respects_max_tokens(
    reference_model: Any, reference_tokenizer: Any, golden_manifest: dict[str, Any]
) -> None:
    """With a small token limit, generation returns exactly that many tokens."""
    eos_ids = golden_manifest["eos_token_ids"]
    checked = 0
    for prompt in PROMPTS:
        golden = load_golden(TEST_MODEL, prompt.id, GREEDY_32)
        if len(golden["token_ids"]) <= SHORT_MAX_TOKENS:
            continue
        prompt_ids = _prompt_ids_matching_golden(reference_tokenizer, prompt, golden)

        tokens = generate_greedy(reference_model, prompt_ids, SHORT_MAX_TOKENS, eos_ids)

        assert tokens == golden["token_ids"][:SHORT_MAX_TOKENS], f"prompt {prompt.id!r}"
        checked += 1
    assert checked, f"no fixture prompt's golden output is longer than {SHORT_MAX_TOKENS} tokens"
