"""Engine and sampling configuration.

Constructing either model runs its validators; an invalid value raises
`pydantic.ValidationError`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class EngineConfig(BaseModel):
    """Static configuration for one engine instance.

    Round-trips unchanged through `model_dump_json()` / `model_validate_json()`.
    Invariant: a constructed EngineConfig is always internally consistent — no later code
    re-checks these constraints.
    """

    model: str
    dtype: Literal["bf16", "fp16", "fp32"]
    max_seq_len: int
    max_num_seqs: int
    block_size: int = 16
    gpu_memory_utilization: float = 0.85
    seed: int = 0

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _check_gpu_memory_utilization(cls, v: float) -> float:
        """Enforce `0 < gpu_memory_utilization <= 1`.

        Must raise ValueError (pydantic wraps it as ValidationError), not any other
        exception type.
        """
        raise NotImplementedError

    @field_validator("max_num_seqs")
    @classmethod
    def _check_max_num_seqs(cls, v: int) -> int:
        """Enforce `max_num_seqs >= 1`."""
        raise NotImplementedError

    @model_validator(mode="after")
    def _check_block_alignment(self) -> Self:
        """Enforce `max_seq_len % block_size == 0` (cross-field, so it runs after field parsing).

        Invariant: a max-length sequence fills an integer number of KV-cache blocks.
        """
        raise NotImplementedError


class SamplingParams(BaseModel):
    """Per-request sampling configuration.

    Round-trips unchanged through `model_dump_json()` / `model_validate_json()`.
    Invariant: `stop` is never shared between instances (hence `default_factory`).
    """

    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    ignore_eos: bool = False

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        """Enforce `temperature >= 0`."""
        raise NotImplementedError

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float) -> float:
        """Enforce `0 < top_p <= 1`."""
        raise NotImplementedError

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v: int) -> int:
        """Enforce `top_k >= 0` (0 means disabled)."""
        raise NotImplementedError

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int) -> int:
        """Enforce `max_tokens >= 1`."""
        raise NotImplementedError

    @property
    def is_greedy(self) -> bool:
        """True iff `temperature == 0`.

        Invariant: the sampler dispatches greedy vs. stochastic on this and nothing else.
        Must be a plain property, not a field, so it is excluded from `model_dump`.
        """
        raise NotImplementedError
