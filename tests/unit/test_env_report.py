"""Environment report used to tag every measurement with its hardware and software."""

from __future__ import annotations

from bench.env_report import env_report

EXPECTED_KEYS = {"gpu_name", "driver", "cuda", "torch", "transformers", "python", "platform"}


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def test_env_report_keys_and_cpu_fallback() -> None:
    """The report has exactly the expected keys and succeeds without a GPU."""
    report = env_report()

    assert set(report) == EXPECTED_KEYS
    if not _cuda_available():
        assert report["gpu_name"] is None
