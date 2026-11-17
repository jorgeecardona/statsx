import math

import numpy as np
import pytest

from statsx.bandit import (
    bayes_ucb_arm,
    bayes_ucb_indices,
    successive_halving,
    thompson_allocation,
)
from statsx.conjugate import Beta

UNIFORM = Beta(1.0, 1.0)


def test_thompson_gives_each_arm_pulls_in_proportion_to_its_chance_of_being_best() -> None:
    # Uniform against Beta(2, 1): P(second is larger) = E[second] = 2/3 exactly.
    alloc = thompson_allocation([UNIFORM, Beta(2.0, 1.0)], 200_000, seed=1)
    assert sum(alloc.counts) == 200_000
    assert alloc.shares[1] == pytest.approx(2.0 / 3.0, abs=0.005)
    assert alloc.shares[0] + alloc.shares[1] == pytest.approx(1.0)


def test_thompson_is_repeatable_and_one_arm_gets_everything() -> None:
    a = thompson_allocation([UNIFORM, Beta(5.0, 5.0)], 100, seed=3)
    assert a == thompson_allocation([UNIFORM, Beta(5.0, 5.0)], 100, seed=3)
    assert thompson_allocation([Beta(3.0, 2.0)], 10).counts == (10,)


def test_bayes_ucb_index_is_the_posterior_quantile_at_the_stated_level() -> None:
    # Beta(1, 1) has quantile q; Beta(2, 1) has cdf x^2, so quantile sqrt(q).
    idx = bayes_ucb_indices([UNIFORM, Beta(2.0, 1.0)], 4)
    assert idx[0] == pytest.approx(0.75, abs=1e-9)
    assert idx[1] == pytest.approx(math.sqrt(0.75), abs=1e-9)
    # With c > 0 the level is 1 - 1 / (t (log n)^c).
    level = 1.0 - 1.0 / (10 * math.log(1000) ** 5)
    idx5 = bayes_ucb_indices([UNIFORM], 10, horizon=1000, c=5.0)
    assert idx5[0] == pytest.approx(level, abs=1e-9)
    assert bayes_ucb_arm([UNIFORM, Beta(2.0, 1.0)], 4) == 1


def _regret(policy: str, seed: int, horizon: int = 1000) -> float:
    p = np.array([0.3, 0.5, 0.6])
    rng = np.random.default_rng(seed)
    posts = [UNIFORM] * 3
    wins = np.zeros(3)
    pulls = np.zeros(3)
    for t in range(1, horizon + 1):
        if policy == "thompson":
            arm = int(np.argmax(thompson_allocation(posts, 1, seed=seed * horizon + t).counts))
        else:
            arm = bayes_ucb_arm(posts, t)
        reward = float(rng.random() < p[arm])
        wins[arm] += reward
        pulls[arm] += 1
        posts = [Beta(1.0 + wins[i], 1.0 + pulls[i] - wins[i]) for i in range(3)]
    return float((pulls * (p.max() - p)).sum())


@pytest.mark.parametrize("policy", ["thompson", "bayes_ucb"])
def test_the_regret_is_far_below_that_of_spending_the_budget_evenly(policy: str) -> None:
    even = 1000 * (0.6 - (0.3 + 0.5 + 0.6) / 3)
    regrets = [_regret(policy, s) for s in range(20)]
    assert np.mean(regrets) < 0.35 * even


def test_the_halving_schedule_follows_the_paper() -> None:
    log: list[tuple[int, int]] = []

    def pull(arm: int, n: int) -> int:
        log.append((arm, n))
        return n // 2 + (arm % 3)  # distinct, valid counts

    result = successive_halving(pull, 8, 240)
    assert [len(r.arms) for r in result.rounds] == [8, 4, 2]
    assert [r.pulls_per_arm for r in result.rounds] == [10, 20, 40]
    assert result.pulls_used == 8 * 10 + 4 * 20 + 2 * 40 <= 240
    five = successive_halving(lambda arm, n: n * (arm + 1) // 6, 5, 300)
    assert [len(r.arms) for r in five.rounds] == [5, 3, 2]
    assert five.best == 4


def test_the_halving_posteriors_use_all_pulls_of_each_arm() -> None:
    result = successive_halving(lambda arm, n: (n * (arm + 1)) // 5, 4, 80, prior=Beta(2.0, 3.0))
    for arm, post in enumerate(result.posteriors):
        assert post.alpha == 2.0 + result.successes[arm]
        assert post.beta == 3.0 + result.trials[arm] - result.successes[arm]
    assert sum(result.trials) == result.pulls_used


def _find_rate(budget: int, runs: int = 300) -> float:
    p = np.array([0.4] * 7 + [0.8])
    rng = np.random.default_rng(budget)
    hits = 0
    for _ in range(runs):
        result = successive_halving(lambda a, n: int(rng.binomial(n, p[a])), 8, budget)
        hits += result.best == 7
    return hits / runs


def test_more_budget_finds_the_best_arm_more_often() -> None:
    assert _find_rate(48) < _find_rate(2000)
    assert _find_rate(2000) > 0.95


@pytest.mark.parametrize(
    "call",
    [
        lambda: thompson_allocation([], 5),
        lambda: thompson_allocation([UNIFORM], 0),
        lambda: bayes_ucb_indices([UNIFORM], 0),
        lambda: bayes_ucb_indices([UNIFORM], 3, c=1.0),
        lambda: successive_halving(lambda a, n: 0, 1, 10),
        lambda: successive_halving(lambda a, n: 0, 8, 10),
        lambda: successive_halving(lambda a, n: n + 1, 4, 100),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"arm|n_pulls|\bt\b|horizon|budget|successes"):
        call()  # type: ignore[operator]
