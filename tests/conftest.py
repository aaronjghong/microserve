"""Shared fixtures and constants for the microserve test suite."""

from __future__ import annotations

import json
import logging
import os

import pytest

TEST_MODEL: str = os.environ.get("TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

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
