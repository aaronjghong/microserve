"""The benchmark tokenizes fixture prompts exactly as the correctness tests do."""

from __future__ import annotations

import pytest

from bench.naive_baseline import load_fixture_prompts
from tests.golden_utils import load_prompts, tokenize_prompt

pytestmark = [pytest.mark.model]


def test_bench_prompt_tokens_match_fixture_tokenization(reference_tokenizer: object) -> None:
    """Every fixture prompt becomes the same token IDs in the benchmark as in the tests.

    The benchmark's numbers are only comparable with the correctness results if both run
    on identical inputs, including chat-template rendering and BOS handling.
    """
    expected = [(p.id, tokenize_prompt(reference_tokenizer, p)) for p in load_prompts()]

    actual = load_fixture_prompts(reference_tokenizer)

    assert [pid for pid, _ in actual] == [pid for pid, _ in expected], "prompt ids or order differ"
    for (pid, got), (_, want) in zip(actual, expected):
        assert got == want, f"prompt {pid!r} tokenizes differently"
