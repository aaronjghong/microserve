"""Environment report used to tag every measurement with its hardware and software."""

from __future__ import annotations

import shutil

from bench.env_report import env_report

EXPECTED_KEYS = {"gpu_name", "driver", "cuda", "torch", "transformers", "python", "platform"}


def _has_nvidia_driver() -> bool:
    # Checked via PATH rather than `torch.cuda.is_available()`: importing torch costs
    # several seconds, and a machine without the driver utility cannot have a CUDA GPU.
    return shutil.which("nvidia-smi") is not None


def test_env_report_keys_and_cpu_fallback() -> None:
    """The report has exactly the expected keys and succeeds without a GPU."""
    report = env_report()

    assert set(report) == EXPECTED_KEYS
    if not _has_nvidia_driver():
        assert report["gpu_name"] is None
