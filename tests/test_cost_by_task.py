"""GitLab issue #12: ``cost_per_pass(..., tasks=...)`` resamples whole tasks."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx.evals import cost_per_pass

ATTEMPTS = 5


def _tasks(
    n_tasks: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[int]]:
    """Hard tasks fail more and cost more, so a task's cost and its results move together."""
    difficulty = rng.normal(0.0, 1.5, n_tasks)
    p = 1.0 / (1.0 + np.exp(-(0.5 - difficulty)))
    passed = (rng.random((n_tasks, ATTEMPTS)) < p[:, None]).astype(np.float64)
    cost = np.exp(1.0 + 0.8 * difficulty[:, None] + rng.normal(0.0, 0.3, (n_tasks, ATTEMPTS)))
    tasks = np.repeat(np.arange(n_tasks), ATTEMPTS).tolist()
    return passed.ravel(), cost.ravel(), tasks


def _truth() -> float:
    passed, cost, _ = _tasks(400_000, np.random.default_rng(99))
    return float(cost.sum() / passed.sum())


def test_resampling_tasks_covers_where_exchangeable_attempts_do_not() -> None:
    truth = _truth()
    rng = np.random.default_rng(0)
    by_task = flat = 0
    runs = 200
    for i in range(runs):
        passed, cost, tasks = _tasks(40, rng)
        by_task += (
            cost_per_pass(passed, cost, tasks=tasks)
            .interval(0.9, n_draws=2000, seed=i)
            .contains(truth)
        )
        flat += cost_per_pass(passed, cost).interval(0.9, n_draws=2000, seed=i).contains(truth)
    assert by_task / runs >= 0.84
    assert flat / runs < 0.7


def test_the_draws_are_total_cost_over_total_passes_of_a_task_resample() -> None:
    result = cost_per_pass([1, 0, 1, 1], [2.0, 4.0, 1.0, 3.0], tasks=["a", "a", "b", "b"])
    assert result.n_tasks == 2
    assert result.task_cost is not None and result.task_passes is not None
    assert result.task_cost.tolist() == [6.0, 4.0]
    assert result.task_passes.tolist() == [1.0, 2.0]
    draws = result.sample(5000, seed=1)
    possible = {6.0 / 1.0, 10.0 / 3.0, 4.0 / 2.0}
    assert set(np.round(draws, 9).tolist()) <= {round(v, 9) for v in possible}


def test_a_resample_with_no_pass_has_no_upper_bound() -> None:
    result = cost_per_pass([0, 0, 1, 0], [2.0, 4.0, 1.0, 3.0], tasks=["a", "a", "b", "b"])
    draws = result.sample(2000, seed=0)
    assert np.isinf(draws).any()
    assert math.isinf(result.interval(0.95, n_draws=2000).upper)


def test_without_tasks_the_result_is_unchanged() -> None:
    passed, cost = [1, 0, 1, 1, 0, 1], [2.0, 4.0, 1.0, 3.0, 5.0, 2.0]
    result = cost_per_pass(passed, cost)
    assert result.n_tasks == 0
    assert result.task_cost is None


def test_tasks_are_checked() -> None:
    with pytest.raises(ValueError, match="same length"):
        cost_per_pass([1, 0, 1], [1.0, 2.0, 3.0], tasks=["a", "b"])
    with pytest.raises(ValueError, match="at least 2"):
        cost_per_pass([1, 0, 1], [1.0, 2.0, 3.0], tasks=["a", "a", "a"])


def test_the_same_seed_gives_the_same_draws() -> None:
    passed, cost, tasks = _tasks(30, np.random.default_rng(5))
    result = cost_per_pass(passed, cost, tasks=tasks)
    assert np.array_equal(result.sample(100, seed=3), result.sample(100, seed=3))
