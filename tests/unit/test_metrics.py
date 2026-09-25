"""In-process metrics: percentile math, thread safety, Prometheus export, reservoir sampling."""

from __future__ import annotations

import random
import statistics
import threading
import time

import pytest
from prometheus_client.parser import text_string_to_metric_families

from microserve.metrics import Counter, Histogram, MetricsRegistry, Timer


def test_percentile_linear_interpolation() -> None:
    """Percentiles interpolate linearly between order statistics.

    Pins the convention so p50/p95/p99 mean the same thing in every report.
    """
    h = Histogram("latency", "Test latency.", buckets=(1.0, 10.0, 100.0))
    for v in range(1, 101):
        h.observe(float(v))

    assert h.percentile(50) == pytest.approx(50.5, abs=1e-9)
    assert h.percentile(95) == pytest.approx(95.05, abs=1e-9)
    assert h.percentile(99) == pytest.approx(99.01, abs=1e-9)


def test_counter_thread_safe_increments() -> None:
    """Concurrent inc() calls from many threads never lose an increment."""
    counter = Counter("hits", "Test hits.")
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
    """render_prometheus() output parses as Prometheus text and preserves every name, help
    text, and value."""
    registry = MetricsRegistry()
    registry.counter("requests", "Requests served.", labels=(("route", "/v1"),)).inc(3)
    registry.gauge("running", "Requests currently running.").set(2.5)
    hist = registry.histogram("latency_seconds", "Request latency in seconds.", buckets=(0.1, 1.0))
    for v in (0.05, 0.5, 2.0):
        hist.observe(v)

    families = {f.name: f for f in text_string_to_metric_families(registry.render_prometheus())}

    assert {"requests", "running", "latency_seconds"} <= families.keys()
    assert families["requests"].documentation == "Requests served."
    assert families["running"].documentation == "Requests currently running."
    assert families["latency_seconds"].documentation == "Request latency in seconds."

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


def test_render_prometheus_multiple_labels() -> None:
    """A series with several labels renders them all, correctly separated."""
    registry = MetricsRegistry()
    registry.counter(
        "requests", "Requests served.", labels=(("route", "/v1"), ("method", "GET"))
    ).inc(2)

    families = {f.name: f for f in text_string_to_metric_families(registry.render_prometheus())}

    (sample,) = families["requests"].samples
    assert sample.labels == {"route": "/v1", "method": "GET"}
    assert sample.value == 2


def test_render_prometheus_escapes_label_values_and_help() -> None:
    """Quotes, backslashes, and newlines in label values and help text survive a round trip."""
    tricky_label = 'say "hi" \\ C:\\path\nsecond line'
    tricky_help = "First line\nsecond line with a \\ backslash."
    registry = MetricsRegistry()
    registry.counter("requests", tricky_help, labels=(("route", tricky_label),)).inc()

    families = {f.name: f for f in text_string_to_metric_families(registry.render_prometheus())}

    assert families["requests"].documentation == tricky_help
    (sample,) = families["requests"].samples
    assert sample.labels == {"route": tricky_label}


def test_render_prometheus_concurrent_with_registration() -> None:
    """Rendering while another thread registers new metrics never raises or emits bad text.

    /metrics is scraped while the engine is running, so a scrape must tolerate the
    registry changing underneath it.
    """
    registry = MetricsRegistry()
    n_metrics = 3_000
    errors: list[BaseException] = []
    done = threading.Event()

    def register() -> None:
        try:
            for i in range(n_metrics):
                registry.counter(f"m_{i}", "Concurrently registered.").inc()
        except BaseException as e:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)
        finally:
            done.set()

    writer = threading.Thread(target=register)
    writer.start()
    renders = 0
    try:
        while not done.is_set():
            list(text_string_to_metric_families(registry.render_prometheus()))
            renders += 1
    except BaseException as e:  # noqa: BLE001
        errors.append(e)
    finally:
        writer.join()

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"
    families = {f.name for f in text_string_to_metric_families(registry.render_prometheus())}
    assert len(families) == n_metrics


def test_timer_observes_once_per_block() -> None:
    """A timed block records exactly one observation, in seconds, and `as` yields the timer."""
    registry = MetricsRegistry()

    with registry.timer("work_seconds", "Time spent working.") as t:
        time.sleep(0.02)

    hist = registry.histogram("work_seconds", "Time spent working.")
    assert isinstance(t, Timer)
    assert hist.count == 1
    assert 0.9 * 0.02 <= hist.sum < 1.0


def test_timer_observes_when_body_raises() -> None:
    """A block that raises still records its observation, and the exception propagates."""
    registry = MetricsRegistry()

    with pytest.raises(KeyError):
        with registry.timer("work_seconds", "Time spent working."):
            raise KeyError("body failed")

    assert registry.histogram("work_seconds", "Time spent working.").count == 1


def test_timer_nested_blocks_time_independently() -> None:
    """Overlapping blocks on the same histogram each measure their own duration.

    The outer block encloses the inner one, so it must record the longer time. A timer
    shared between the two blocks would restart the clock and record both as short.
    """
    registry = MetricsRegistry()
    step = 0.03

    with registry.timer("work_seconds", "Time spent working."):
        time.sleep(step)
        with registry.timer("work_seconds", "Time spent working."):
            time.sleep(step)

    hist = registry.histogram("work_seconds", "Time spent working.")
    assert hist.count == 2
    assert max(hist.samples) >= 0.9 * 2 * step
    assert min(hist.samples) >= 0.9 * step


def test_reservoir_caps_samples_and_preserves_median() -> None:
    """Past the sample cap, retained samples stay bounded and remain representative."""
    rng = random.Random(1234)
    stream = [rng.random() for _ in range(250_000)]
    h = Histogram("uniform", "Uniform test stream.", buckets=(0.5, 1.0))
    for v in stream:
        h.observe(v)

    true_median = statistics.median(stream)

    assert len(h.samples) == 100_000
    assert abs(h.percentile(50) - true_median) <= 0.02 * true_median


def test_reservoir_samples_uniformly_across_whole_stream() -> None:
    """Observations after the cap is reached are represented in proportion to their share.

    The stream shifts halfway through (100k zeros, then 150k ones), so a histogram that
    keeps only the first `max_samples` observations retains no ones at all, while a
    uniform reservoir retains about 60%. An i.i.d. stream can't tell the two apart.
    """
    early, late = 100_000, 150_000
    h = Histogram("shifting", "Shifting test stream.", buckets=(0.5, 1.0))
    for _ in range(early):
        h.observe(0.0)
    for _ in range(late):
        h.observe(1.0)

    late_fraction = sum(h.samples) / len(h.samples)

    assert late_fraction == pytest.approx(late / (early + late), abs=0.01)
    assert h.percentile(50) == 1.0
