import math

import numpy as np
import pytest

from statsx.pairwise import Comparisons, fit_bradley_terry, fit_by_judge

NAMES = ("s0", "s1", "s2")
BIAS = (0.3, -0.2, 0.0)
AGREE = (np.array([0.8, 0.0, -0.8]), np.array([0.8, 0.0, -0.8]))
REVERSED = (np.array([0.8, 0.0, -0.8]), np.array([-0.8, 0.0, 0.8]))


def _data(
    seed: int, strengths: tuple[np.ndarray, ...], n_items: int, *, item_sd: float = 0.5
) -> Comparisons:
    """Every judge judges every pair on every item, in a random order; an item shifts strengths."""
    rng = np.random.default_rng(seed)
    first: list[str] = []
    second: list[str] = []
    outcome: list[float] = []
    item: list[int] = []
    judge: list[str] = []
    for it in range(n_items):
        shift = rng.normal(0.0, item_sd, 3)
        for j, theta in enumerate(strengths):
            for a in range(3):
                for b in range(a + 1, 3):
                    x, y = (a, b) if rng.random() < 0.5 else (b, a)
                    d = theta[x] + shift[x] - theta[y] - shift[y] + BIAS[j]
                    first.append(NAMES[x])
                    second.append(NAMES[y])
                    outcome.append(float(rng.random() < 1.0 / (1.0 + math.exp(-d))))
                    item.append(it)
                    judge.append(f"j{j}")
    return Comparisons.from_long(first, second, outcome, item=item, judge=judge)


def test_judges_that_agree_are_rarely_called_different() -> None:
    rejected = [
        fit_by_judge(_data(500 + s, AGREE, 30), ties="half", n_bootstrap=100, seed=s).p_value < 0.05
        for s in range(40)
    ]
    # The nominal rate is 5%. With 40 runs, more than 8 rejections is very unlikely (p < 0.01).
    assert sum(rejected) <= 8


def test_judges_that_rank_the_systems_in_opposite_orders_are_found() -> None:
    result = fit_by_judge(_data(1, REVERSED, 40), ties="half", n_bootstrap=200)
    assert result.p_value < 0.001
    assert result.df == 2 and result.statistic > 10.0
    assert result.ranking("j0") == ("s0", "s1", "s2")
    assert result.ranking("j1") == ("s2", "s1", "s0")


def test_the_strengths_of_each_judge_are_recovered_and_the_difference_interval_holds_the_truth() -> (
    None
):
    result = fit_by_judge(_data(2, REVERSED, 150), ties="half", n_bootstrap=200)
    order = [NAMES.index(name) for name in result.systems]
    truth = np.vstack(REVERSED)[:, order]
    truth = truth - truth.mean(axis=1, keepdims=True)
    assert np.max(np.abs(result.strength - truth)) < 0.3
    box = result.difference_interval("j0", "j1", "s0", 0.95)
    assert box.contains(1.6) and box.lower > 0.0
    same = result.difference_interval("j0", "j1", "s1", 0.95)
    assert same.contains(0.0)


def test_the_pooled_fit_is_the_fit_with_one_set_of_strengths() -> None:
    data = _data(3, AGREE, 30)
    result = fit_by_judge(data, ties="half", n_bootstrap=60)
    pooled = fit_bradley_terry(data, ties="half", n_bootstrap=2)
    assert np.allclose(result.pooled.strength, pooled.strength)
    assert result.pooled.judges == ("j0", "j1")


def test_the_test_is_the_same_for_the_order_of_the_judges() -> None:
    data = _data(4, REVERSED, 30)
    swapped = Comparisons(
        data.systems,
        data.first,
        data.second,
        data.outcome,
        data.item,
        ("j1", "j0"),
        1 - data.judge if data.judge is not None else None,
    )
    a = fit_by_judge(data, ties="half", n_bootstrap=100, seed=7)
    b = fit_by_judge(swapped, ties="half", n_bootstrap=100, seed=7)
    assert a.p_value < 0.001 and b.p_value < 0.001


def test_bad_input_is_refused() -> None:
    data = _data(5, AGREE, 30)
    no_judge = Comparisons(data.systems, data.first, data.second, data.outcome, data.item)
    with pytest.raises(ValueError, match="at least two judges"):
        fit_by_judge(no_judge)
    with pytest.raises(ValueError, match="degrees of freedom"):
        fit_by_judge(data, ties="half", n_bootstrap=2)
    with pytest.raises(ValueError, match="ties must be"):
        fit_by_judge(data, ties="x")  # type: ignore[arg-type]
    result = fit_by_judge(data, ties="half", n_bootstrap=60)
    with pytest.raises(ValueError, match="unknown judge"):
        result.ranking("zzz")
    with pytest.raises(ValueError, match="unknown system"):
        result.difference_interval("j0", "j1", "zzz")


def test_a_judge_that_does_not_link_all_systems_is_refused_and_named() -> None:
    first = ["a", "b", "a", "b", "c", "d", "c", "d"]
    second = ["b", "a", "b", "a", "d", "c", "d", "c"]
    judge = ["x"] * 4 + ["y"] * 4
    data = Comparisons.from_long(
        first + first, second + second, [1.0, 0.0] * 8, judge=judge + judge
    )
    with pytest.raises(ValueError, match="judge 'x' does not compare every system"):
        fit_by_judge(data, ties="half")
