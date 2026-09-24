"""Environment report: the hardware and software versions a measurement depends on.

Emits the same JSON the `gpu_env` test fixture logs.
Run: `uv run python -m bench.env_report`
"""

from __future__ import annotations

ENV_REPORT_KEYS: frozenset[str] = frozenset(
    {"gpu_name", "driver", "cuda", "torch", "transformers", "python", "platform"}
)


def env_report() -> dict[str, str | None]:
    """Collect hardware and software versions relevant to reproducing a measurement.

    Returns a dict whose keys are exactly ENV_REPORT_KEYS.

    Invariants: never raises on a CPU-only box — `gpu_name` (and any other CUDA-only field)
    is None there; every value is a str or None, so the dict is JSON-serializable as-is.
    """
    raise NotImplementedError


def main() -> None:
    """Print `env_report()` as indented JSON to stdout."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
