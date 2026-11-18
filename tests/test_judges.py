import math
from collections.abc import Callable

import numpy as np
import pytest

from statsx.pairwise import BradleyTerryFit, Comparisons, fit_bradley_terry

THETA = {"s0": 0.8, "s1": 0.0, "s2": -0.8}
BIAS = {"j1": 0.8, "j2": 0.0}  # judge j1 favours the answer shown first


def _rows(
    seed: int, n: int = 300, *, first_share: float = 0.5
) -> tuple[list[str], list[str], list[float], list[str], list[str]]:
    """Return first, second, outcome, judge and item.

    ``first_share`` is how often a pair is shown in the order that puts the stronger system first.
    """
    rng = np.random.default_rng(seed)
    names = list(THETA)
    first: list[str] = []
    second: list[str] = []
    outcome: list[float] = []
    judge: list[str] = []
    item: list[str] = []
    for i in range(n):
        a, b = rng.choice(names, size=2, replace=False)
        if THETA[str(a)] < THETA[str(b)] and rng.random() < 2.0 * first_share - 1.0:
            a, b = b, a
        j = str(rng.choice(list(BIAS)))
        eta = THETA[str(a)] - THETA[str(b)] + BIAS[j]
        first.append(str(a))
        second.append(str(b))
        outcome.append(float(rng.random() < 1.0 / (1.0 + math.exp(-eta))))
        judge.append(j)
        item.append(f"i{i}")
    return first, second, outcome, judge, item


def _data(seed: int, n: int = 300, *, first_share: float = 0.5) -> Comparisons:
    f, s, o, j, i = _rows(seed, n, first_share=first_share)
    return Comparisons.from_long(f, s, o, item=i, judge=j)


def _gap(fit: BradleyTerryFit) -> float:
    return float(fit.strength[fit.systems.index("s0")] - fit.strength[fit.systems.index("s2")])


def test_the_position_bias_of_each_judge_is_recovered() -> None:
    fit = fit_bradley_terry(_data(0, 1500), ties="half", n_bootstrap=200)
    assert fit.judges == ("j1", "j2")
    assert fit.position_bias[0] == pytest.approx(BIAS["j1"], abs=0.25)
    assert fit.position_bias[1] == pytest.approx(BIAS["j2"], abs=0.25)
    assert fit.position_bias_interval("j1").lower > 0.0
    assert fit.position_bias_interval("j2").lower < 0.0 < fit.position_bias_interval("j2").upper
    assert fit.p_first_wins("j1") == pytest.approx(1.0 / (1.0 + math.exp(-fit.position_bias[0])))


def test_the_90_percent_interval_of_the_bias_covers_the_truth_at_close_to_the_stated_rate() -> None:
    covered = 0
    for seed in range(60):
        fit = fit_bradley_terry(_data(seed), ties="half", n_bootstrap=100, seed=seed)
        interval = fit.position_bias_interval("j1", level=0.9)
        covered += interval.lower <= BIAS["j1"] <= interval.upper
    assert covered >= 50  # 83% of 60: the percentile bootstrap is a little narrow


def test_the_strengths_stay_right_when_the_stronger_system_is_shown_first_more_often() -> None:
    # Strong systems are mostly shown first and the judges favour the first answer: without the
    # judge term, the strong systems look stronger than they are.
    errors_with: list[float] = []
    errors_without: list[float] = []
    for seed in range(6):
        f, s, o, j, i = _rows(seed, 1500, first_share=0.9)
        with_judge = fit_bradley_terry(
            Comparisons.from_long(f, s, o, item=i, judge=j), ties="half", n_bootstrap=2
        )
        without = fit_bradley_terry(
            Comparisons.from_long(f, s, o, item=i), ties="half", n_bootstrap=2
        )
        truth = THETA["s0"] - THETA["s2"]
        errors_with.append(abs(_gap(with_judge) - truth))
        errors_without.append(abs(_gap(without) - truth))
    assert np.mean(errors_with) < 0.2
    assert np.mean(errors_without) > np.mean(errors_with) + 0.05


def test_a_judge_that_always_sees_the_same_system_first_is_refused_and_named() -> None:
    first = ["a", "a", "a", "a"]
    second = ["b", "b", "c", "c"]
    judge = ["strict", "strict", "strict", "strict"]
    with pytest.raises(ValueError, match="'strict'"):
        fit_bradley_terry(Comparisons.from_long(first, second, [1.0, 0.0, 1.0, 0.0], judge=judge))


def test_one_judge_with_both_orders_is_fitted() -> None:
    first = ["a", "b", "a", "b"] * 5
    second = ["b", "a", "b", "a"] * 5
    outcome = [1.0, 0.0, 1.0, 1.0] * 5
    fit = fit_bradley_terry(
        Comparisons.from_long(first, second, outcome, judge=["j"] * 20), n_bootstrap=10
    )
    assert fit.judges == ("j",)


def test_without_a_judge_column_the_fit_has_no_bias_terms() -> None:
    f, s, o, _, i = _rows(0, 100)
    fit = fit_bradley_terry(Comparisons.from_long(f, s, o, item=i), n_bootstrap=5)
    assert fit.judges == () and fit.position_bias.size == 0 and fit.bias_draws.shape == (5, 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: Comparisons.from_long(["a"], ["b"], [1.0], judge=["x", "y"]),
        lambda: fit_bradley_terry(_data(0, 50), bias_sd=0.0),
        lambda: fit_bradley_terry(_data(0, 50), n_bootstrap=3).position_bias_interval("zzz"),
        lambda: fit_bradley_terry(_data(0, 50), n_bootstrap=3).p_first_wins("zzz"),
    ],
)
def test_bad_input_is_refused(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match=r"."):
        call()
