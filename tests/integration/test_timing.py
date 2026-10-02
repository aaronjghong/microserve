"""GPU timing helper accuracy.

torch is imported inside each test so this module still collects on machines without it.
"""

from __future__ import annotations

import statistics
import time

import pytest

from microserve.metrics import cuda_timer

# _sleep spins for a cycle count, so its duration follows the GPU clock: ~70-100 ms on an
# RTX Spark N1X. Aim for 50-100 ms.
SLEEP_CYCLES = 150_000_000
REPEATS = 5


def _wall_clock_ms(torch, cycles: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def _warm_up(torch) -> None:
    torch.cuda._sleep(1_000)
    torch.cuda.synchronize()


def _paired_medians(torch, timed_once) -> tuple[float, float]:
    """Median wall-clock and timer readings, sampled alternately.

    Interleaving means a GPU clock change shifts both series alike, instead of landing
    between a block of wall-clock samples and a block of timer samples.
    """
    wall, timed = [], []
    for _ in range(REPEATS):
        wall.append(_wall_clock_ms(torch, SLEEP_CYCLES))
        timed.append(timed_once())
    return statistics.median(wall), statistics.median(timed)


@pytest.mark.gpu
def test_cuda_timer_matches_synchronized_wall_clock(gpu_env) -> None:
    """The timer agrees with a synchronized wall-clock measurement of the same kernel to within 10%."""
    import torch

    def timed_once() -> float:
        torch.cuda.synchronize()
        with cuda_timer() as t:
            torch.cuda._sleep(SLEEP_CYCLES)
            torch.cuda.synchronize()
        return t.elapsed_ms

    _warm_up(torch)
    wall, measured = _paired_medians(torch, timed_once)

    assert measured == pytest.approx(wall, rel=0.10)


@pytest.mark.gpu
def test_cuda_timer_includes_unsynchronized_kernel(gpu_env) -> None:
    """A kernel launched asynchronously inside the block, with no synchronize, is still fully timed.

    Catches a timer that reads the clock when the launch returns instead of when the
    kernel finishes.
    """
    import torch

    def timed_once() -> float:
        torch.cuda.synchronize()
        with cuda_timer() as t:
            torch.cuda._sleep(SLEEP_CYCLES)
        return t.elapsed_ms

    _warm_up(torch)
    wall, measured = _paired_medians(torch, timed_once)

    assert measured >= 0.9 * wall
