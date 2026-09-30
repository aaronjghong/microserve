"""Fixtures applied to every integration test."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _record_gpu_env(request: pytest.FixtureRequest) -> None:
    """Log the hardware and library versions once per session for any GPU test."""
    if request.node.get_closest_marker("gpu"):
        request.getfixturevalue("gpu_env")
