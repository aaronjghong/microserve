"""Package metadata and test-suite health."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_matches_pyproject() -> None:
    """`microserve.__version__` must equal the version declared in pyproject.toml."""
    import microserve

    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]

    assert microserve.__version__ == declared


def test_every_unit_file_collects_tests() -> None:
    """Every unit test file contributes at least one test.

    Guards against an import error in one file silently dropping all of its tests from
    the run.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not gpu", "--collect-only", "-q", "tests/unit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    per_file = Counter(
        line.split("::", 1)[0] for line in result.stdout.splitlines() if "::" in line
    )

    unit_files = sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests" / "unit").glob("test_*.py")
    )
    empty = [f for f in unit_files if per_file[f] == 0]

    assert result.returncode == 0, f"collection failed:\n{result.stdout}\n{result.stderr}"
    assert not empty, f"files that collected no tests: {empty}"
