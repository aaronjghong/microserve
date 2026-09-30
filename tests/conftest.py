"""Shared fixtures and constants for the microserve test suite.

Heavy imports (torch, transformers) happen inside fixtures, so unit tests that don't use
them stay fast.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from tests.golden_utils import (
    GOLDEN_DEVICE,
    GOLDEN_DTYPE,
    TEST_MODEL,
    Prompt,
    load_manifest,
    load_prompts,
)

__all__ = ["TEST_MODEL"]

logger = logging.getLogger("microserve.tests")


@pytest.fixture(scope="session")
def gpu_env() -> dict[str, str | None]:
    """Hardware and software versions for this run, logged once per session.

    GPU tests take this fixture so every result can be traced back to the GPU, driver,
    and library versions that produced it. View with `--log-cli-level=INFO`.
    """
    from bench.env_report import env_report

    env = env_report()
    logger.info("gpu_env %s", json.dumps(env, sort_keys=True))
    return env


@pytest.fixture(scope="session")
def prompts() -> list[Prompt]:
    return load_prompts()


@pytest.fixture(scope="session")
def reference_tokenizer() -> Any:
    """The test model's tokenizer, loaded directly through Hugging Face."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture(scope="session")
def reference_model() -> Any:
    """The test model in bf16 on CUDA, loaded directly through Hugging Face.

    Forward and generation tests use this rather than microserve's own loader, so a
    loader bug shows up only in the loader tests.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL, dtype=torch.bfloat16)
    return model.to("cuda").eval()


@pytest.fixture(scope="session")
def golden_manifest() -> dict[str, Any]:
    """The golden manifest for the test model, checked against how the tests run.

    Greedy outputs are only comparable token for token when made in the same dtype on the
    same kind of device, so a mismatch fails here instead of as a confusing diff later.
    """
    manifest = load_manifest(TEST_MODEL)
    made_with = (manifest["dtype"], manifest["device"])
    assert made_with == (GOLDEN_DTYPE, GOLDEN_DEVICE), (
        f"goldens were generated with {made_with}, but tests run with "
        f"{(GOLDEN_DTYPE, GOLDEN_DEVICE)}; regenerate them with `uv run python -m tests.gen_golden`"
    )
    return manifest
