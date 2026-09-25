"""cuda_timer behavior, checked against a fake `torch` so it runs without a GPU.

The fake mimics the CUDA semantics that matter for timing: events are recorded
asynchronously, and `elapsed_time` fails unless both events have completed, which only
happens after a synchronize. Absolute accuracy on real hardware is covered by the GPU
integration tests.
"""

from __future__ import annotations

import logging
import sys
import time
import types
import warnings

import pytest

from microserve.metrics import cuda_timer


class FakeEvent:
    def __init__(self, cuda: FakeCuda, enable_timing: bool = False, **_: object) -> None:
        self._cuda = cuda
        self.enable_timing = enable_timing
        self.time_ms: float | None = None
        self.completed = False

    def record(self, stream: object = None) -> None:
        self.time_ms = self._cuda.gpu_clock_ms
        self.completed = False
        self._cuda.pending.append(self)

    def synchronize(self) -> None:
        if self.time_ms is not None:
            self.completed = True

    def query(self) -> bool:
        return self.completed

    def elapsed_time(self, end: FakeEvent) -> float:
        if not (self.enable_timing and end.enable_timing):
            raise RuntimeError("Both events must be created with enable_timing=True.")
        if self.time_ms is None or end.time_ms is None:
            raise RuntimeError("Both events must be recorded before calculating elapsed time.")
        if not (self.completed and end.completed):
            raise RuntimeError("Both events must be completed before calculating elapsed time.")
        return end.time_ms - self.time_ms


class FakeStream:
    def __init__(self, cuda: FakeCuda) -> None:
        self._cuda = cuda

    def synchronize(self) -> None:
        self._cuda.synchronize()


class FakeCuda(types.ModuleType):
    def __init__(self, available: bool) -> None:
        super().__init__("torch.cuda")
        self._available = available
        self.gpu_clock_ms = 0.0
        self.pending: list[FakeEvent] = []

    def is_available(self) -> bool:
        return self._available

    def Event(self, enable_timing: bool = False, **kwargs: object) -> FakeEvent:  # noqa: N802
        return FakeEvent(self, enable_timing=enable_timing, **kwargs)

    def synchronize(self, device: object = None) -> None:
        for event in self.pending:
            event.completed = True
        self.pending.clear()

    def current_stream(self, device: object = None) -> FakeStream:
        return FakeStream(self)

    def run_kernel(self, ms: float) -> None:
        """Queue simulated GPU work: advances the GPU clock without completing anything."""
        self.gpu_clock_ms += ms


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool) -> FakeCuda:
    fake_cuda = FakeCuda(available=cuda_available)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    return fake_cuda


def test_cuda_timer_synchronizes_before_reading_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CUDA, the timer measures with events and waits for them to complete.

    Kernel work is queued without any synchronize inside the block; a timer that reads
    the events without synchronizing fails, and a timer that uses the host clock
    reports ~0 instead of the simulated 25 ms.
    """
    fake_cuda = _install_fake_torch(monkeypatch, cuda_available=True)

    with cuda_timer() as t:
        fake_cuda.run_kernel(25.0)

    assert t.elapsed_ms == pytest.approx(25.0)


def test_cuda_timer_falls_back_to_perf_counter_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without CUDA, the timer measures host wall-clock time and warns about it."""
    _install_fake_torch(monkeypatch, cuda_available=False)

    with warnings.catch_warnings(record=True) as warned, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        outer_start = time.perf_counter()
        with cuda_timer() as t:
            time.sleep(0.05)
        outer_ms = (time.perf_counter() - outer_start) * 1e3

    assert t.elapsed_ms is not None
    assert 0.9 * 50 <= t.elapsed_ms <= outer_ms
    logged = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert logged or warned, "falling back to the host clock must emit a warning"


@pytest.mark.parametrize("cuda_available", [True, False], ids=["cuda", "fallback"])
def test_cuda_timer_result_set_only_on_exit(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool
) -> None:
    """`elapsed_ms` is None inside the block and set on exit, even when the block raises."""
    _install_fake_torch(monkeypatch, cuda_available=cuda_available)

    with cuda_timer() as t:
        assert t.elapsed_ms is None
    assert isinstance(t.elapsed_ms, float)

    with pytest.raises(KeyError):
        with cuda_timer() as failing:
            raise KeyError("body failed")
    assert isinstance(failing.elapsed_ms, float)
