"""Baseline latency of the naive single-request greedy loop.

Measures time to first token (TTFT) and time per output token (TPOT) for each fixture
prompt and writes one CSV row per prompt. Later optimizations are compared against these
numbers, so measure exactly the loop being served, with no batching and no cache tricks.

Run: `uv run python -m bench.naive_baseline`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel

CSV_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "prompt_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
)


def measure_prompt(
    model: PreTrainedModel,
    prompt_ids: list[int],
    max_tokens: int,
    eos_ids: list[int],
) -> dict[str, float | int]:
    """Time one greedy generation; returns every CSV column except `prompt_id`.

    ttft_ms: from the start of the request until the first generated token is available.
    tpot_ms: mean time per generated token after the first. Undefined for a single-token
             output; record it as NaN rather than 0.

    Invariant: GPU work is complete before each timestamp is taken, so asynchronous kernel
    launches are not mistaken for finished work.
    """
    raise NotImplementedError


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Write `rows` to `path` with exactly `CSV_COLUMNS` as the header, in that order."""
    raise NotImplementedError


def main() -> None:
    """Load the model, apply the warmup policy, measure every fixture prompt, write the CSV."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
