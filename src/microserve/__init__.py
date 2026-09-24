"""microserve — a measurable LLM serving engine.

Public attribute:
    __version__: str — must equal `[project].version` in pyproject.toml.
        Invariant: single source of truth; the version is never hard-coded in two places
        that can drift.
"""


def __getattr__(name: str) -> object:
    """PEP 562 module __getattr__ — placeholder until `__version__` is defined.

    Delete this function once you define `__version__` at module level.
    """
    if name == "__version__":
        raise NotImplementedError("microserve.__version__")
    raise AttributeError(f"module 'microserve' has no attribute {name!r}")
