"""GPU timing helper accuracy.

torch is imported inside each test so this module still collects on machines without it.
"""

from __future__ import annotations

import statistics
import time

import pytest

from microserve.metrics import cuda_timer

SLEEP_CYCLES = 100_000_000  # roughly 50-100 ms on current GPUs
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


@pytest.mark.gpu
def test_cuda_timer_matches_synchronized_wall_clock(gpu_env) -> None:
    """The timer agrees with a synchronized wall-clock measurement of the same kernel to within 10%."""
    import torch

    _warm_up(torch)
    wall = statistics.median(_wall_clock_ms(torch, SLEEP_CYCLES) for _ in range(REPEATS))

    timed = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        with cuda_timer() as t:
            torch.cuda._sleep(SLEEP_CYCLES)
            torch.cuda.synchronize()
        timed.append(t.elapsed_ms)
    measured = statistics.median(timed)

    assert measured == pytest.approx(wall, rel=0.10)


@pytest.mark.gpu
def test_cuda_timer_includes_unsynchronized_kernel(gpu_env) -> None:
    """A kernel launched asynchronously inside the block, with no synchronize, is still fully timed.

    Catches a timer that reads the clock when the launch returns instead of when the
    kernel finishes.
    """
    import torch

    _warm_up(torch)
    wall = statistics.median(_wall_clock_ms(torch, SLEEP_CYCLES) for _ in range(REPEATS))

    timed = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        with cuda_timer() as t:
            torch.cuda._sleep(SLEEP_CYCLES)
        timed.append(t.elapsed_ms)
    measured = statistics.median(timed)

    assert measured >= 0.9 * wall
