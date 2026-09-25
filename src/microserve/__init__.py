"""microserve — a measurable LLM serving engine.

Public attribute:
    __version__: str — must equal `[project].version` in pyproject.toml.
        Invariant: single source of truth; the version is never hard-coded in two places
        that can drift.
"""

from importlib.metadata import version
__version__ = version('microserve')