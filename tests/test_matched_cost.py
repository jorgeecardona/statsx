"""GitLab issue #14: the cost per solved task of each model at the quality of a reference."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx.evals import (
    MAX_ATTEMPTS,
    _attempts_to_reach,  # pyright: ignore[reportPrivateUsage]
    cost_at_ability,
)

ABILITY = {"weak": -0.5, "strong": 1.0}
COST = {"weak": 1.0, "strong": 6.0}  # cost of one attempt, before the task effect
TASKS = 60


def _rows(
    seed: int, attempts: int = 3
) -> tuple[
    list[str], list[str], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0.0, 1.0, TASKS)
    models: list[str] = []
    tasks: list[str] = []
    passed: list[float] = []
    costs: list[float] = []
    for name, theta in ABILITY.items():
        for t in range(TASKS):
            p = 1.0 / (1.0 + math.exp(-(theta - difficulty[t])))
            for _ in range(attempts):
                models.append(name)
                tasks.append(f"t{t}")
                passed.append(float(rng.random() < p))
                costs.append(COST[name] * math.exp(0.4 * difficulty[t] + rng.normal(0.0, 0.2)))
    return models, tasks, np.asarray(passed), np.asarray(costs), difficulty


def _true_cost(name: str, difficulty: npt.NDArray[np.float64]) -> float:
    target = float((1.0 / (1.0 + np.exp(-(ABILITY["strong"] - difficulty)))).mean())
    p = 1.0 / (1.0 + np.exp(-(ABILITY[name] - difficulty)))
    cost = COST[name] * np.exp(0.4 * difficulty + 0.5 * 0.2**2)
    return _attempts_to_reach(p, cost, target)[1]


def test_the_reach_matches_the_target_exactly() -> None:
    p = np.array([0.2, 0.5, 0.9])
    cost = np.ones(3)
    attempts, per_solved = _attempts_to_reach(p, cost, 0.8)
    assert 2.0 < attempts < 4.0
    k = math.floor(attempts)
    w = attempts - k
    solved = (1 - w) * (1 - (1 - p) ** k).mean() + w * (1 - (1 - p) ** (k + 1)).mean()
    assert solved == pytest.approx(0.8)
    assert per_solved > 1.0


def test_a_model_at_or_above_the_reference_uses_one_attempt() -> None:
    attempts, per_solved = _attempts_to_reach(np.array([0.5, 0.9]), np.array([2.0, 4.0]), 0.6)
    assert attempts == 1.0
    assert per_solved == pytest.approx(6.0 / 1.4)


def test_an_unreachable_reference_gives_inf() -> None:
    attempts, per_solved = _attempts_to_reach(np.array([0.0, 0.5]), np.ones(2), 0.9)
    assert math.isinf(attempts) and math.isinf(per_solved)


def test_the_weak_model_needs_retries_and_the_interval_covers_the_truth() -> None:
    covered = {"weak": 0, "strong": 0}
    runs = 30
    for seed in range(runs):
        models, tasks, passed, costs, difficulty = _rows(seed)
        result = cost_at_ability(
            models, tasks, passed, costs, reference="strong", n_draws=400, seed=seed
        )
        for name in covered:
            covered[name] += result.interval(name, 0.9).contains(_true_cost(name, difficulty))
    assert covered["weak"] / runs >= 0.8
    assert covered["strong"] / runs >= 0.8


def test_the_attempts_and_the_ratio_read_as_expected() -> None:
    models, tasks, passed, costs, _ = _rows(1)
    result = cost_at_ability(models, tasks, passed, costs, reference="strong", n_draws=400)
    weak, strong = result.models.index("weak"), result.models.index("strong")
    assert result.attempts[strong] == 1.0
    assert 1.5 < result.attempts[weak] < MAX_ATTEMPTS
    assert 0.5 < result.reference_quality < 0.8
    ratio = result.ratio("weak", "strong")
    assert ratio.upper < 1.0  # six times cheaper per attempt beats the retries here
    assert result.p_cheaper("weak", "strong") > 0.95


def test_an_infra_offset_does_not_leak_into_the_cost() -> None:
    models, tasks, passed, costs, _ = _rows(2)
    infra = ["fast" if (m == "weak" and i % 2) else "base" for i, m in enumerate(models)]
    shifted = np.where(np.asarray(infra) == "fast", costs * 3.0, costs)
    plain = cost_at_ability(models, tasks, passed, costs, reference="strong", n_draws=400)
    with_infra = cost_at_ability(
        models, tasks, passed, shifted, reference="strong", infra=infra, n_draws=400
    )
    a, b = plain.interval("weak"), with_infra.interval("weak")
    assert abs(math.log((a.lower + a.upper) / (b.lower + b.upper))) < 0.15


def test_a_numeric_reference_and_bad_input() -> None:
    models, tasks, passed, costs, _ = _rows(3, attempts=2)
    low = cost_at_ability(models, tasks, passed, costs, reference=-3.0, n_draws=200)
    assert np.all(low.attempts == 1.0)
    with pytest.raises(ValueError, match="unknown model"):
        cost_at_ability(models, tasks, passed, costs, reference="nobody")
    with pytest.raises(ValueError, match="n_draws"):
        cost_at_ability(models, tasks, passed, costs, reference="weak", n_draws=10)
    result = cost_at_ability(models, tasks, passed, costs, reference="weak", n_draws=200)
    with pytest.raises(ValueError, match="unknown model"):
        result.interval("nobody")
