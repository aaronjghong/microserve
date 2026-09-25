"""In-process metrics primitives and timing helpers.

Everything here is thread-safe: the engine loop, API handlers, and the /metrics scrape run
on different threads.

Keep this module importable without torch so CPU-only unit tests stay fast. Import torch
lazily inside `cuda_timer`.
"""

from __future__ import annotations

import random
import threading
import time
import warnings
from contextlib import AbstractContextManager, contextmanager
from typing import ContextManager
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

    def __init__(self, name: str, help: str, labels: tuple[tuple[str, str], ...] = ()) -> None:
        self.name: str = name
        self.help: str = help
        self.labels: tuple[tuple[str, str], ...] = labels
        self._value: float = 0.0
        self._lock: threading.Lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        """Add `amount` (must be >= 0)."""
        if not amount >= 0:
            raise ValueError("Amount must be >= 0")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Current total."""
        return self._value


class Gauge:
    """Value that can go up or down (e.g. running requests, KV blocks free).

    Invariant: `value` reflects the last set() plus all later inc()/dec(), with none lost.
    """

    def __init__(self, name: str, help: str) -> None:
        self.name: str = name
        self.help: str = help
        self._value: float = 0.0
        self._lock: threading.Lock = threading.Lock()

    def set(self, v: float) -> None:
        """Replace the value."""
        with self._lock:
            self._value = v

    def inc(self, amount: float = 1.0) -> None:
        """Add `amount`."""
        if not amount >= 0:
            raise ValueError("Amount must be >= 0")
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Subtract `amount`."""
        if not amount >= 0:
            raise ValueError("Amount must be >= 0")
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


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
        help: str,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        seed: int = 0,
    ) -> None:
        self.name: str = name
        self.help: str = help
        self.buckets: tuple[float, ...] = tuple(sorted(buckets))
        self.max_samples: int = max_samples
        self._bucket_counts: list[int] = [0 for bucket in buckets]
        self._samples: list[float] = []
        self._count: int = 0
        self._sum: float = 0.0
        self._rng: random.Random = random.Random(seed)
        self._lock: threading.Lock = threading.Lock()

    def observe(self, v: float) -> None:
        """Record one observation."""
        # Vitter's Algorithm
        with self._lock:
            if (self._count < self.max_samples):
                self._samples.append(v)
            else:
                # Replace random element in _samples with a probability of max_samples/count
                probability = float(self.max_samples / (self._count + 1)) # Count current count too
                r = self._rng.random()
                if (r < probability):
                    i = self._rng.randint(0, self.max_samples - 1)
                    self._samples[i] = v
            self._sum += v
            self._count += 1
            for i, bucket in enumerate(self.buckets):
                if v <= bucket:
                    self._bucket_counts[i] += 1 
                    break # We can sum it up later, no need to check all buckets


    def percentile(self, p: float) -> float:
        """The p-th percentile (0 <= p <= 100) of retained samples, by linear interpolation
        between order statistics — not bucket midpoints.

        Pinned convention: for samples 1..100, p50 == 50.5, p95 == 95.05, p99 ~= 99.01.
        """
        with self._lock:
            samples_sorted = sorted(self._samples)
            num = min(self._count, self.max_samples)
            i = p/100 * (num - 1)
            if (i < (num - 1)):
                f = i % 1
                k = int(i)
                return samples_sorted[k] + f * (samples_sorted[k+1] - samples_sorted[k])
            else:
                return samples_sorted[int(i)]

    @property
    def samples(self) -> list[float]:
        """A copy of the retained raw samples."""
        with self._lock:
            return self._samples.copy()

    @property
    def count(self) -> int:
        """Total observations ever made."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """Sum of all observations ever made."""
        with self._lock:
            return self._sum

    @property
    def bucket_counts(self) -> list[int]:
        with self._lock:
            return self._bucket_counts.copy()

class Timer:
    """Context manager that observes elapsed wall-clock seconds into a Histogram on exit.

    Invariant: exactly one observation per `with` block, including when the body raises.
    """

    def __init__(self, histogram: Histogram) -> None:
        self.histogram: Histogram = histogram
        self._start: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.histogram.observe(time.perf_counter() - self._start)


class MetricsRegistry:
    """Owns every metric by name and renders them for /metrics.

    A series is identified by its name plus its label set (label order doesn't matter), as in
    the Prometheus data model. Series sharing a name form one metric family.

    Invariants:
      - One metric object per series (get-or-create, never duplicate).
      - Every series under a name has the same metric type; asking for an existing name as a
        different type is an error, whatever the labels.
      - Each metric object's `name` is the plain metric name, never a composite key.
      - Every metric has a non-optional `help` description, rendered as the family's
        `# HELP` line.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, dict[tuple, Counter | Gauge | Histogram]] = {}
        self._lock: threading.Lock = threading.Lock()

    def counter(self, name: str, help: str, labels: tuple[tuple[str, str], ...] = ()) -> Counter:
        """Get or create a Counter."""
        # Sort all tuples in labels
        labels = tuple(sorted(labels))
        with self._lock:
            if (name not in self._metrics):
                self._metrics[name] = {labels: Counter(name, help, labels)}
            elif (() in self._metrics[name] and not isinstance(self._metrics[name][()], Counter)): # Assuming Gauge and Histograms have no labels
                raise ValueError(f"Cannot create/get Counter {name}, with labels {labels} as {name} is created as another type (Gauge/Histogram)")
            elif (labels not in self._metrics[name]):
                self._metrics[name][labels] = Counter(name, help, labels)
            return self._metrics[name][labels]
                

    def gauge(self, name: str, help: str) -> Gauge:
        """Get or create a Gauge."""
        with self._lock:
            if (name not in self._metrics):
                self._metrics[name] = {(): Gauge(name, help)}
            elif (not isinstance(self._metrics[name].get(()), Gauge)):
                raise ValueError(f"Cannot create/get Gauge {name} as it is created under another type (Counter/Histogram)")
            return self._metrics[name][()]

    def histogram(
        self, name: str, help: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS
    ) -> Histogram:
        """Get or create a Histogram."""
        with self._lock:
            if (name not in self._metrics):
                self._metrics[name] = {(): Histogram(name, help, buckets)}
            elif (not isinstance(self._metrics[name].get(()), Histogram)):
                raise ValueError(f"Cannot create/get Histogram {name} as it is created under another type (Counter/Gauge)")
            elif (self._metrics[name][()].buckets != tuple(sorted(buckets))):
                raise ValueError(f"Histogram {name} already exists with buckets {self._metrics[name][()].buckets}. Buckets {buckets} is invalid")
            return self._metrics[name][()]


    def timer(self, name: str, help: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> Timer:
        """`with registry.timer("ttft_seconds", "Time to first token."): ...` — times into
        histogram `name`, creating it with `help` if it doesn't exist yet."""
        hist = self.histogram(name, help, buckets)
        return Timer(hist)

    def render_prometheus(self) -> str:
        """All metrics in the Prometheus text exposition format: one `# HELP` and one `# TYPE`
        line per metric family, then every series in it (histograms as `_bucket{le=...}` /
        `_sum` / `_count`), ending with a trailing newline.

        Output must parse with `prometheus_client.parser.text_string_to_metric_families`.
        """
        strs = []
        with self._lock:
            for name, metric in self._metrics.items():
                # Greedily take first help message, assuming that counters with the same name share the same help message
                first_item = metric[next(iter(metric))]
                strs.append(fr"# HELP {name} {first_item.help.replace("\\", "\\\\").replace("\n", "\\n")}") 
                def type_to_str(t):
                    if (isinstance(t, Counter)):
                        return "counter"
                    elif (isinstance(t, Gauge)):
                        return "gauge"
                    elif (isinstance(t, Histogram)):
                        return "histogram"
                    else:
                        raise ValueError(f"Unsupported type in metrics {t}")
                strs.append(fr"# TYPE {name} {type_to_str(first_item)}")
                if (type(first_item) == Counter):
                    for labels in metric:
                        strs2 = []
                        for label in labels:
                            strs2.append(f"{label[0]}=\"{label[1].replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")}\"")
                        strs.append(fr"{name}{{{','.join(strs2)}}} {metric[labels].value}")
                elif (type(first_item) == Gauge):
                    strs.append(f"{name} {first_item.value}")
                elif (type(first_item) == Histogram):
                    count = 0
                    for i, b in enumerate(first_item.buckets): # Should already be sorted
                        count += first_item.bucket_counts[i]
                        strs.append(f"""{name}_bucket{{le="{b}"}} {count}""")
                    strs.append(f"""{name}_bucket{{le="+Inf"}} {first_item.count}""")
                    strs.append(f"{name}_sum {first_item.sum}")
                    strs.append(f"{name}_count {first_item.count}")
                strs.append('\n')
            return '\n'.join(strs)


class TimingResult:
    """Holder populated by `cuda_timer` when its block exits.

    Invariant: `elapsed_ms` is None inside the block and set exactly once on exit.
    """

    def __init__(self) -> None:
        self.elapsed_ms: float | None = None


@contextmanager
def cuda_timer() -> ContextManager[TimingResult]:
    """Time a block of GPU work in milliseconds.

        with cuda_timer() as t:
            ...
        t.elapsed_ms

    Uses `torch.cuda.Event(enable_timing=True)` when CUDA is available; otherwise falls
    back to `time.perf_counter` and logs a warning.

    Invariant: when the block exits, the recorded work has *finished*, not just been queued —
    an async kernel launched inside the block without a synchronize is still fully counted.
    """
    import torch.cuda
    t = TimingResult()
    if (torch.cuda.is_available()):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield t
        finally:
            end.record()
            torch.cuda.synchronize()
            t.elapsed_ms = start.elapsed_time(end)
    else:
        warnings.warn("Using cuda_timer but cuda is not available")
        start = time.perf_counter_ns()
        try:
            yield t
        finally:
            end = time.perf_counter_ns()
            t.elapsed_ms = (end - start) / 1000000



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
