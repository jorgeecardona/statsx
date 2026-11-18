"""The same call gives the same numbers after a fork, in a spawned process, and in threads.

The Rust core keeps no global state and no random generator between calls, so a caller may warm
statsx up in the parent, then fork, or call it from many threads.
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from statsx.bootstrap import IID, bootstrap
from statsx.diagnostics import check_convergence, mcse_quantile
from statsx.evals import Responses, fit_rasch


def work(seed: int) -> tuple[float, ...]:
    """Run one call of several Rust-backed functions and return numbers that identify the result."""
    x = np.random.default_rng(seed).normal(size=(4, 500))
    c = check_convergence({"mu": x}).parameters[0]
    b = bootstrap(x[0], np.mean, IID(), n_resamples=200, seed=seed).estimate
    y = np.random.default_rng(seed).integers(0, 2, 120).astype(float)
    data = Responses.from_long(
        [f"m{i % 4}" for i in range(120)], [f"t{i % 15}" for i in range(120)], y
    )
    return (
        c.rhat,
        c.ess_bulk,
        b.estimate,
        float(fit_rasch(data).ability[0]),
        mcse_quantile(x, 0.9),
    )


SEEDS = list(range(6))


@pytest.fixture(scope="module")
def serial() -> list[tuple[float, ...]]:
    return [work(s) for s in SEEDS]  # also warms the extension up in this process


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_a_forked_or_spawned_process_gives_the_serial_numbers(
    method: str, serial: list[tuple[float, ...]]
) -> None:
    with mp.get_context(method).Pool(2) as pool:
        assert pool.map(work, SEEDS) == serial


def test_many_threads_calling_at_once_give_the_serial_numbers(
    serial: list[tuple[float, ...]],
) -> None:
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(work, SEEDS * 6)) == serial * 6
