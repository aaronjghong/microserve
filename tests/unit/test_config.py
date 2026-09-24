"""Validation and serialization of EngineConfig and SamplingParams."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from microserve.config import EngineConfig, SamplingParams

VALID_ENGINE: dict[str, Any] = {
    "model": "test-model",
    "dtype": "bf16",
    "max_seq_len": 64,
    "max_num_seqs": 1,
}
VALID_SAMPLING: dict[str, Any] = {"max_tokens": 1}


@pytest.mark.parametrize(
    ("cls", "base", "override"),
    [
        pytest.param(EngineConfig, VALID_ENGINE, {"max_seq_len": 100, "block_size": 16},
                     id="max_seq_len-not-multiple-of-block_size"),
        pytest.param(EngineConfig, VALID_ENGINE, {"gpu_memory_utilization": 0.0},
                     id="gpu_memory_utilization-zero"),
        pytest.param(EngineConfig, VALID_ENGINE, {"max_num_seqs": 0},
                     id="max_num_seqs-zero"),
        pytest.param(SamplingParams, VALID_SAMPLING, {"temperature": -0.1},
                     id="temperature-negative"),
        pytest.param(SamplingParams, VALID_SAMPLING, {"top_p": 0.0},
                     id="top_p-zero"),
        pytest.param(SamplingParams, VALID_SAMPLING, {"top_k": -1},
                     id="top_k-negative"),
        pytest.param(SamplingParams, VALID_SAMPLING, {"max_tokens": 0},
                     id="max_tokens-zero"),
    ],
)
def test_invalid_field_raises_validation_error(
    cls: type[BaseModel], base: dict[str, Any], override: dict[str, Any]
) -> None:
    """Each out-of-range value is rejected at construction with a ValidationError."""
    with pytest.raises(ValidationError):
        cls(**{**base, **override})


def test_is_greedy_iff_temperature_zero() -> None:
    """Temperature 0 means greedy decoding; any positive temperature does not."""
    assert SamplingParams(max_tokens=1, temperature=0).is_greedy is True
    assert SamplingParams(max_tokens=1, temperature=0.7).is_greedy is False


@pytest.mark.parametrize(
    "instance_factory",
    [
        pytest.param(
            lambda: EngineConfig(
                model="test-model",
                dtype="fp16",
                max_seq_len=64,
                max_num_seqs=4,
                block_size=32,
                gpu_memory_utilization=0.5,
                seed=7,
            ),
            id="EngineConfig",
        ),
        pytest.param(
            lambda: SamplingParams(
                max_tokens=5,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=3,
                stop=["\n", "END"],
                ignore_eos=True,
            ),
            id="SamplingParams",
        ),
    ],
)
def test_json_round_trip(instance_factory: Any) -> None:
    """Serializing to JSON and back yields an identical object (every field non-default)."""
    original = instance_factory()
    restored = type(original).model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.model_dump() == original.model_dump()
