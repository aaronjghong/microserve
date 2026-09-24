"""In-process metrics primitives and timing helpers.

Everything here is thread-safe: the engine loop, API handlers, and the /metrics scrape run
on different threads.

Keep this module importable without torch so CPU-only unit tests stay fast. Import torch
lazily inside `cuda_timer`.
"""

from __future__ import annotations

import random
import threading
from contextlib import AbstractContextManager
from types import TracebackType

# Prometheus client library defaults; a sensible starting point for latency in seconds.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
)
DEFAULT_MAX_SAMPLES: int = 100_000


class Counter:
    """Monotonically increasing value.

    `labels` is a fixed set of (key, value) pairs attached to this counter instance.

    Invariant: value never decreases; no increment is ever lost under concurrent inc().
    """

    def __init__(self, name: str, labels: tuple[tuple[str, str], ...] = ()) -> None:
        self.name: str = name
        self.labels: tuple[tuple[str, str], ...] = labels
        self._value: float = 0.0
        self._lock: threading.Lock = threading.Lock()
        raise NotImplementedError

    def inc(self, amount: float = 1.0) -> None:
        """Add `amount` (must be >= 0)."""
        raise NotImplementedError

    @property
    def value(self) -> float:
        """Current total."""
        raise NotImplementedError


class Gauge:
    """Value that can go up or down (e.g. running requests, KV blocks free).

    Invariant: `value` reflects the last set() plus all later inc()/dec(), with none lost.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._value: float = 0.0
        self._lock: threading.Lock = threading.Lock()
        raise NotImplementedError

    def set(self, v: float) -> None:
        """Replace the value."""
        raise NotImplementedError

    def inc(self, amount: float = 1.0) -> None:
        """Add `amount`."""
        raise NotImplementedError

    def dec(self, amount: float = 1.0) -> None:
        """Subtract `amount`."""
        raise NotImplementedError

    @property
    def value(self) -> float:
        """Current value."""
        raise NotImplementedError


class Histogram:
    """Distribution of observations with exact-sample percentiles.

    Keeps two views: cumulative bucket counts (for Prometheus export) and raw samples
    (for `percentile`). Raw samples are capped at `max_samples`; beyond that, reservoir
    sampling keeps a uniform random subset of everything observed.

    Invariants:
      - `count` and `sum` cover *every* observation, not just the retained samples.
      - `len(samples) == min(count, max_samples)`.
      - Retained samples are a uniform random subset of all observations.
    """

    def __init__(
        self,
        name: str,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        seed: int = 0,
    ) -> None:
        self.name: str = name
        self.buckets: tuple[float, ...] = buckets
        self.max_samples: int = max_samples
        self._bucket_counts: list[int] = []
        self._samples: list[float] = []
        self._count: int = 0
        self._sum: float = 0.0
        self._rng: random.Random = random.Random(seed)
        self._lock: threading.Lock = threading.Lock()
        raise NotImplementedError

    def observe(self, v: float) -> None:
        """Record one observation."""
        raise NotImplementedError

    def percentile(self, p: float) -> float:
        """The p-th percentile (0 <= p <= 100) of retained samples, by linear interpolation
        between order statistics — not bucket midpoints.

        Pinned convention: for samples 1..100, p50 == 50.5, p95 == 95.05, p99 ~= 99.01.
        """
        raise NotImplementedError

    @property
    def samples(self) -> list[float]:
        """A copy of the retained raw samples."""
        raise NotImplementedError

    @property
    def count(self) -> int:
        """Total observations ever made."""
        raise NotImplementedError

    @property
    def sum(self) -> float:
        """Sum of all observations ever made."""
        raise NotImplementedError


class Timer:
    """Context manager that observes elapsed wall-clock seconds into a Histogram on exit.

    Invariant: exactly one observation per `with` block, including when the body raises.
    """

    def __init__(self, histogram: Histogram) -> None:
        self.histogram: Histogram = histogram
        self._start: float | None = None
        raise NotImplementedError

    def __enter__(self) -> Timer:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError


class MetricsRegistry:
    """Owns every metric by name and renders them for /metrics.

    Invariants: one metric object per name (get-or-create, never duplicate); asking for an
    existing name as a different metric type is an error.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock: threading.Lock = threading.Lock()
        raise NotImplementedError

    def counter(self, name: str, labels: tuple[tuple[str, str], ...] = ()) -> Counter:
        """Get or create a Counter."""
        raise NotImplementedError

    def gauge(self, name: str) -> Gauge:
        """Get or create a Gauge."""
        raise NotImplementedError

    def histogram(self, name: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> Histogram:
        """Get or create a Histogram."""
        raise NotImplementedError

    def timer(self, name: str) -> Timer:
        """`with registry.timer("ttft_seconds"): ...` — times into histogram `name`."""
        raise NotImplementedError

    def render_prometheus(self) -> str:
        """All metrics in the Prometheus text exposition format (HELP/TYPE lines, histogram
        `_bucket{le=...}` / `_sum` / `_count` series, trailing newline).

        Output must parse with `prometheus_client.parser.text_string_to_metric_families`.
        """
        raise NotImplementedError


class TimingResult:
    """Holder populated by `cuda_timer` when its block exits.

    Invariant: `elapsed_ms` is None inside the block and set exactly once on exit.
    """

    def __init__(self) -> None:
        self.elapsed_ms: float | None = None
        raise NotImplementedError


def cuda_timer() -> AbstractContextManager[TimingResult]:
    """Time a block of GPU work in milliseconds.

        with cuda_timer() as t:
            ...
        t.elapsed_ms

    Uses `torch.cuda.Event(enable_timing=True)` when CUDA is available; otherwise falls
    back to `time.perf_counter` and logs a warning.

    Invariant: when the block exits, the recorded work has *finished*, not just been queued —
    an async kernel launched inside the block without a synchronize is still fully counted.
    """
    raise NotImplementedError


__all__: list[str] = [
    "DEFAULT_BUCKETS",
    "DEFAULT_MAX_SAMPLES",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "Timer",
    "TimingResult",
    "cuda_timer",
]
