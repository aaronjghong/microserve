"""Golden reference outputs exist, are complete, and belong to the current prompts."""

from __future__ import annotations

import re

from tests.golden_utils import (
    GREEDY_32,
    TEST_MODEL,
    golden_json_path,
    load_manifest,
    load_prompts,
    prompts_sha256,
)


def test_every_prompt_has_greedy_golden() -> None:
    """Every fixture prompt has a greedy golden, pinned to an exact model revision.

    The prompts file must also be unchanged since the goldens were made; otherwise every
    comparison against them is meaningless.
    """
    manifest = load_manifest(TEST_MODEL)
    missing = [
        p.id for p in load_prompts() if not golden_json_path(TEST_MODEL, p.id, GREEDY_32).exists()
    ]

    assert re.fullmatch(r"[0-9a-f]{40}", manifest["model_revision"])
    assert not missing, f"prompts without a {GREEDY_32} golden: {missing}"
    assert manifest["prompts_sha256"] == prompts_sha256(), (
        "prompts.json changed after the goldens were generated; add a new prompts file instead"
    )
