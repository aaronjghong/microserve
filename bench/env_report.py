"""Environment report: the hardware and software versions a measurement depends on.

Emits the same JSON the `gpu_env` test fixture logs.
Run: `uv run python -m bench.env_report`
"""

from __future__ import annotations
import platform
import subprocess
import json
from importlib.util import find_spec
from importlib.metadata import version

ENV_REPORT_KEYS: frozenset[str] = frozenset(
    {"gpu_name", "driver", "cuda", "torch", "transformers", "python", "platform"}
)


def env_report() -> dict[str, str | None]:
    """Collect hardware and software versions relevant to reproducing a measurement.

    Returns a dict whose keys are exactly ENV_REPORT_KEYS.

    Invariants: never raises on a CPU-only box — `gpu_name` (and any other CUDA-only field)
    is None there; every value is a str or None, so the dict is JSON-serializable as-is.
    """
    packages = ["torch", "transformers"]
    env_info = {}

    # Will always exist
    env_info["python"] = platform.python_version()
    env_info["platform"] = f"{platform.system()} ({platform.machine()})"

    # Check package versions
    for package in packages:
        if find_spec(package):
            env_info[package] = version(package)
        else:
            env_info[package] = None

    # Check for cuda and gpu versions through nvidia smi
    try:
        s = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], timeout=10).decode('utf-8')
        info = s.split('\n')[0].split(',') # Greedily take first GPU
        env_info["gpu_name"] = info[0].strip()  
        env_info["driver"] = info[1].strip()
    except Exception:
        env_info["driver"] = None
        env_info["gpu_name"] = None

    # Torch wheels on PyPI on Linux carry no "+" label so this might always return None unless we install from the cuda index
    env_info["cuda"] = None
    if (env_info["torch"] is not None):
        info = env_info["torch"].split("+") # Will be something like torch+cuXXX if cuda
        if (len(info) == 1 and info[1][0:2].lower() == "cu"):
            env_info["cuda"] = f"{info[1][2:4]}.{info[1][4:]}" # For now, always assume that first 2 digits are Major, rest are minor

    return env_info
    




def main() -> None:
    """Print `env_report()` as indented JSON to stdout."""
    print(json.dumps(env_report(), indent=4))


if __name__ == "__main__":
    main()
