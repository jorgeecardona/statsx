from collections.abc import Callable

import numpy as np
import pytest

from statsx.identify import best_arm


def _bernoulli(p: list[float], seed: int) -> Callable[[int, int], int]:
    rng = np.random.default_rng(seed)
    return lambda arm, n: int(rng.binomial(n, p[arm]))


def test_the_arm_returned_is_the_best_arm_with_probability_at_least_one_minus_delta() -> None:
    p = [0.5, 0.6, 0.45, 0.4]
    runs, wrong, unsure = 300, 0, 0
    for seed in range(runs):
        r = best_arm(_bernoulli(p, seed), 4, delta=0.2, batch=25, max_pulls=400_000)
        wrong += r.confident and r.best != 1
        unsure += not r.confident
    assert unsure == 0
    assert wrong / runs <= 0.2
    # The bound is conservative, so in practice no run picks the wrong arm.
    assert wrong == 0


def test_an_epsilon_guarantee_returns_an_arm_within_epsilon_of_the_best() -> None:
    p = [0.50, 0.52, 0.51, 0.30]
    runs, bad = 100, 0
    pulls: list[int] = []
    for seed in range(runs):
        r = best_arm(_bernoulli(p, seed), 4, delta=0.1, epsilon=0.1, batch=20)
        assert r.confident
        bad += max(p) - p[r.best] > 0.1
        pulls.append(r.total_pulls)
    assert bad == 0
    exact = best_arm(_bernoulli(p, 0), 4, delta=0.1, batch=20, max_pulls=60_000)
    assert np.mean(pulls) < exact.total_pulls / 2


def test_arms_that_are_further_from_the_best_are_removed_earlier_and_pulled_less() -> None:
    r = best_arm(_bernoulli([0.2, 0.4, 0.6, 0.9], 1), 4, delta=0.05, batch=5)
    assert r.confident and r.best == 3 and r.active == (3,)
    assert r.removed_at[3] is None
    gone = [r.removed_at[a] for a in (0, 1, 2)]
    assert all(g is not None for g in gone)
    assert gone[0] <= gone[1] <= gone[2]  # type: ignore[operator]
    assert r.pulls[0] <= r.pulls[1] <= r.pulls[2] <= r.pulls[3]


def test_two_equal_arms_run_out_of_budget_and_the_result_says_no_guarantee() -> None:
    r = best_arm(_bernoulli([0.5, 0.5], 2), 2, max_pulls=2000, batch=10)
    assert not r.confident
    assert r.active == (0, 1)
    assert r.total_pulls <= 2000


def test_the_counts_add_up_and_a_larger_batch_checks_less_often() -> None:
    p = [0.3, 0.7, 0.5]
    small = best_arm(_bernoulli(p, 3), 3, batch=1)
    large = best_arm(_bernoulli(p, 3), 3, batch=50)
    assert small.best == large.best == 1
    assert small.rounds > large.rounds
    assert sum(small.successes) <= small.total_pulls
    assert small.total_pulls == sum(small.pulls)
    assert small.delta == 0.05 and small.epsilon == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_arms": 1},
        {"n_arms": 2, "delta": 0.0},
        {"n_arms": 2, "delta": 1.0},
        {"n_arms": 2, "epsilon": -0.1},
        {"n_arms": 2, "batch": 0},
        {"n_arms": 5, "batch": 10, "max_pulls": 40},
    ],
)
def test_bad_input_is_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"n_arms|delta|epsilon|batch|max_pulls"):
        best_arm(lambda a, n: 0, **kwargs)  # type: ignore[arg-type]


def test_a_pull_function_that_returns_too_many_successes_is_refused() -> None:
    with pytest.raises(ValueError, match="successes"):
        best_arm(lambda a, n: n + 1, 2)
