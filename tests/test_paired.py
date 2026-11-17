import math
from itertools import product

import numpy as np
import pytest

from statsx.paired import PairedTest, bootstrap_difference, mcnemar, sign_test, wilcoxon
from statsx.result import Estimate


def _binomial_two_sided(k: int, n: int) -> float:
    pmf = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, 2.0 * min(sum(pmf[: k + 1]), sum(pmf[k:])))


def test_the_sign_test_matches_the_binomial_sum_and_drops_ties() -> None:
    a = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.0])
    b = np.array([4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 15.0])  # 8 wins, 2 losses, 1 tie
    result = sign_test(a, b)
    assert isinstance(result, PairedTest)
    assert (result.statistic, result.n) == (8.0, 10)
    assert result.p_value == pytest.approx(_binomial_two_sided(8, 10), rel=1e-12)
    assert result.p_value == pytest.approx(0.109375, rel=1e-12)


def test_mcnemar_exact_is_a_binomial_test_on_the_disagreements() -> None:
    a = np.array([1] * 40 + [1] * 12 + [0] * 3 + [0] * 45)
    b = np.array([1] * 40 + [0] * 12 + [1] * 3 + [0] * 45)
    result = mcnemar(a, b)
    assert (result.statistic, result.n) == (9.0, 15)
    assert result.p_value == pytest.approx(_binomial_two_sided(12, 15), rel=1e-12)
    assert "exact" in result.test


def test_mcnemar_chi_square_matches_the_textbook_value() -> None:
    # Agresti's presidential approval example: 150 and 86 disagreements, chi-square 17.36 without
    # the continuity correction and (63)^2 / 236 = 16.82 with it.
    a = np.array([1] * 150 + [0] * 86)
    b = np.array([0] * 150 + [1] * 86)
    result = mcnemar(a, b, exact=False)
    assert result.statistic == pytest.approx(63.0**2 / 236.0, rel=1e-12)
    assert result.p_value == pytest.approx(math.erfc(math.sqrt(result.statistic / 2)), rel=1e-12)
    assert result.p_value < 1e-4
    assert mcnemar([1, 0], [1, 0], exact=False).p_value == 1.0
    assert mcnemar([1, 1], [1, 1]).p_value == 1.0


def _brute_force_wilcoxon(d: np.ndarray) -> tuple[float, float]:
    d = d[d != 0]
    order = np.argsort(np.abs(d), kind="stable")
    absd = np.abs(d)[order]
    ranks = np.empty(d.size)
    i = 0
    while i < d.size:
        j = i
        while j + 1 < d.size and absd[j + 1] == absd[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    observed = ranks[d[order] > 0].sum()
    sums = np.array([np.dot(ranks, s) for s in product((0, 1), repeat=d.size)])
    lower, upper = np.mean(sums <= observed + 1e-9), np.mean(sums >= observed - 1e-9)
    return float(observed), min(1.0, 2.0 * min(lower, upper))


def test_wilcoxon_matches_the_enumeration_of_all_sign_assignments_with_and_without_ties() -> None:
    rng = np.random.default_rng(0)
    for d in (
        rng.normal(0.4, 1.0, 11),
        np.round(rng.normal(0.5, 1.5, 12), 0),
        np.array([1, 1, -1, 2, 2, 2, -3, 4.0, 0]),
    ):
        statistic, p = _brute_force_wilcoxon(d)
        result = wilcoxon(d, np.zeros_like(d))
        assert result.statistic == pytest.approx(statistic)
        assert result.p_value == pytest.approx(p, rel=1e-9)


def test_wilcoxon_matches_the_hollander_wolfe_depression_example() -> None:
    x = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    result = wilcoxon(x, y)
    assert result.statistic == 40.0
    assert result.p_value == pytest.approx(0.03906, abs=1e-5)
    assert "exact" in result.assumption


def test_the_normal_approximation_agrees_with_the_exact_test_near_the_limit() -> None:
    rng = np.random.default_rng(3)
    d = rng.normal(0.3, 1.0, 50)
    exact = wilcoxon(d, np.zeros(50)).p_value
    # 51 non-zero differences switch to the approximation.
    approx = wilcoxon(np.append(d, 0.31), np.zeros(51)).p_value
    assert "normal" in wilcoxon(np.append(d, 0.31), np.zeros(51)).assumption
    assert approx == pytest.approx(exact, abs=0.01)


def test_the_tests_keep_their_size_and_have_power() -> None:
    rng = np.random.default_rng(1)
    size = {"sign": 0, "wilcoxon": 0}
    power = {"sign": 0, "wilcoxon": 0}
    for _ in range(400):
        null = rng.normal(0.0, 1.0, 30)
        size["sign"] += sign_test(null, np.zeros(30)).p_value < 0.05
        size["wilcoxon"] += wilcoxon(null, np.zeros(30)).p_value < 0.05
        shifted = rng.normal(0.6, 1.0, 30)
        power["sign"] += sign_test(shifted, np.zeros(30)).p_value < 0.05
        power["wilcoxon"] += wilcoxon(shifted, np.zeros(30)).p_value < 0.05
    assert size["sign"] / 400 <= 0.06 and size["wilcoxon"] / 400 <= 0.08
    # With symmetric differences Wilcoxon uses their size, so it is the stronger test.
    assert power["wilcoxon"] > power["sign"]


def test_the_bootstrap_difference_is_an_estimate_with_an_interval() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 60)
    b = a - rng.normal(0.4, 0.5, 60)
    result = bootstrap_difference(a, b, n_resamples=1000, seed=1)
    assert isinstance(result, Estimate)
    assert result.estimate == pytest.approx(float(np.mean(a - b)))
    assert result.interval.contains(0.4)
    assert result.n == 60
    assert "percentile" in result.assumption
    assert bootstrap_difference(a, b, method="bca", n_resamples=500, seed=1).interval.contains(0.4)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sign_test([1.0, 2.0], [1.0]),
        lambda: wilcoxon([], []),
        lambda: wilcoxon([1.0, float("nan")], [0.0, 0.0]),
        lambda: mcnemar([0, 2], [1, 1]),
        lambda: mcnemar([0, 1], [1]),
        lambda: bootstrap_difference([1.0], [1.0, 2.0]),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"paired|same length|finite|0/1"):
        call()  # type: ignore[operator]
