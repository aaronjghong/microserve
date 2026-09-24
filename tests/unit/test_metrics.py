"""In-process metrics: percentile math, thread safety, Prometheus export, reservoir sampling."""

from __future__ import annotations

import random
import statistics
import threading

import pytest
from prometheus_client.parser import text_string_to_metric_families

from microserve.metrics import Counter, Histogram, MetricsRegistry


def test_percentile_linear_interpolation() -> None:
    """Percentiles interpolate linearly between order statistics.

    Pins the convention so p50/p95/p99 mean the same thing in every report.
    """
    h = Histogram("latency", buckets=(1.0, 10.0, 100.0))
    for v in range(1, 101):
        h.observe(float(v))

    assert h.percentile(50) == pytest.approx(50.5, abs=1e-9)
    assert h.percentile(95) == pytest.approx(95.05, abs=1e-9)
    assert h.percentile(99) == pytest.approx(99.01, abs=1e-9)


def test_counter_thread_safe_increments() -> None:
    """Concurrent inc() calls from many threads never lose an increment."""
    counter = Counter("hits")
    n_threads, per_thread = 8, 10_000
    start = threading.Barrier(n_threads)

    def work() -> None:
        start.wait()
        for _ in range(per_thread):
            counter.inc()

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.value == n_threads * per_thread


def test_render_prometheus_round_trip() -> None:
    """render_prometheus() output parses as Prometheus text and preserves every name and value."""
    registry = MetricsRegistry()
    registry.counter("requests", labels=(("route", "/v1"),)).inc(3)
    registry.gauge("running").set(2.5)
    hist = registry.histogram("latency_seconds", buckets=(0.1, 1.0))
    for v in (0.05, 0.5, 2.0):
        hist.observe(v)

    families = {f.name: f for f in text_string_to_metric_families(registry.render_prometheus())}

    assert {"requests", "running", "latency_seconds"} <= families.keys()

    (counter_sample,) = families["requests"].samples
    assert counter_sample.name in {"requests", "requests_total"}
    assert counter_sample.labels == {"route": "/v1"}
    assert counter_sample.value == 3

    (gauge_sample,) = families["running"].samples
    assert gauge_sample.value == 2.5

    hist_samples = families["latency_seconds"].samples
    buckets = {
        float(s.labels["le"]): s.value for s in hist_samples if s.name == "latency_seconds_bucket"
    }
    assert buckets == {0.1: 1, 1.0: 2, float("inf"): 3}
    (count,) = [s.value for s in hist_samples if s.name == "latency_seconds_count"]
    (total,) = [s.value for s in hist_samples if s.name == "latency_seconds_sum"]
    assert count == 3
    assert total == pytest.approx(2.55)


def test_reservoir_caps_samples_and_preserves_median() -> None:
    """Past the sample cap, retained samples stay bounded and remain representative."""
    rng = random.Random(1234)
    stream = [rng.random() for _ in range(250_000)]
    h = Histogram("uniform", buckets=(0.5, 1.0))
    for v in stream:
        h.observe(v)

    true_median = statistics.median(stream)

    assert len(h.samples) == 100_000
    assert abs(h.percentile(50) - true_median) <= 0.02 * true_median
