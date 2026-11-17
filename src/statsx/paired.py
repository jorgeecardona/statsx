"""Tests for paired results: McNemar, sign, Wilcoxon signed-rank, and a bootstrap difference (CO-1).

Paired means each item was measured twice (two systems on the same items, before and after). The
tests look at the differences, so they ignore the spread between items.

- ``mcnemar``: two systems that are right or wrong on the same items. Only the items where they
  disagree count. The exact test is a binomial test on those items; it is the default.
- ``sign_test``: the share of items where A beats B, against one half. It uses the direction of
  each difference only, so it is robust and weak.
- ``wilcoxon``: the signed-rank test. It uses direction and size of the differences, and is
  stronger than the sign test when the differences are roughly symmetric. Zero differences are
  dropped. For up to 50 non-zero differences the p-value is exact (it handles ties); above that,
  a normal approximation with a tie correction.
- ``bootstrap_difference``: the mean difference with a bootstrap interval, no distribution
  assumed.

References:
    McNemar, Q. (1947). Note on the sampling error of the difference between correlated
    proportions or percentages. *Psychometrika* 12(2):153-157.
    https://doi.org/10.1007/BF02295996.

    Edwards, A. L. (1948). Note on the "correction for continuity" in testing the significance
    of the difference between correlated proportions. *Psychometrika* 13(3):185-187.
    https://doi.org/10.1007/BF02289261.

    Dixon, W. J., & Mood, A. M. (1946). The statistical sign test. *Journal of the American
    Statistical Association* 41(236):557-566. https://doi.org/10.1080/01621459.1946.10501898.

    Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*
    1(6):80-83. https://doi.org/10.2307/3001968.

    Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods*,
    3rd ed., chapters 3 and 5. Wiley. https://doi.org/10.1002/9781119196037.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.bootstrap import IID, IntervalMethod, bootstrap
from statsx.result import Estimate

EXACT_LIMIT = 50
"""Largest number of non-zero differences for which Wilcoxon's p-value is exact."""


@dataclass(frozen=True, slots=True)
class PairedTest:
    """The result of a test on paired results.

    Args:
        test: The test that was run.
        statistic: The test statistic (see each function).
        p_value: The two-sided p-value.
        n: The number of items that entered the test.
        assumption: The assumption behind the p-value, in words.
    """

    test: str
    statistic: float
    p_value: float
    n: int
    assumption: str


def _binomial_cdf_half(k: int, n: int) -> float:
    """Return P(X <= k) for X ~ Binomial(n, 1/2), from the incomplete beta function."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return _accel.beta_cdf(np.array([0.5]), float(n - k), float(k + 1))[0]


def _two_sided_binomial(k: int, n: int) -> float:
    """Return the exact two-sided p-value of ``k`` successes in ``n`` fair trials."""
    if n == 0:
        return 1.0
    lower = _binomial_cdf_half(k, n)
    upper = 1.0 - _binomial_cdf_half(k - 1, n)
    return min(1.0, 2.0 * min(lower, upper))


def _pair(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        raise ValueError("paired results need two non-empty arrays of the same length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("paired results must be finite")
    return x - y


def mcnemar(
    a_correct: npt.ArrayLike, b_correct: npt.ArrayLike, *, exact: bool = True
) -> PairedTest:
    """Test whether two systems differ, from whether each is right on each item.

    Let ``b`` be the items only A gets right and ``c`` the items only B gets right. The exact
    test is a two-sided binomial test of ``b`` out of ``b + c`` against one half. With
    ``exact=False`` the statistic is ``(|b - c| - 1)^2 / (b + c)`` (with the continuity
    correction of Edwards 1948) against a chi-square with 1 degree of freedom. ``statistic`` is
    ``b - c`` for the exact test and the chi-square value otherwise.

    Raises:
        ValueError: If the arrays differ in length, are empty, or hold values other than 0 and 1.
    """
    xf = np.asarray(a_correct, dtype=np.float64).ravel()
    yf = np.asarray(b_correct, dtype=np.float64).ravel()
    binary = np.isin(xf, (0.0, 1.0)).all() and np.isin(yf, (0.0, 1.0)).all()
    if xf.size != yf.size or xf.size == 0 or not binary:
        raise ValueError("a_correct and b_correct must be 0/1 arrays of the same length")
    x, y = xf.astype(np.int64), yf.astype(np.int64)
    b = int(np.count_nonzero((x == 1) & (y == 0)))
    c = int(np.count_nonzero((x == 0) & (y == 1)))
    n = b + c
    if exact:
        return PairedTest(
            "mcnemar (exact)",
            float(b - c),
            _two_sided_binomial(b, n),
            n,
            "items are independent; under no difference each disagreement favours A or B with "
            "probability one half",
        )
    if n == 0:
        return PairedTest("mcnemar (chi-square)", 0.0, 1.0, 0, "no disagreements")
    chi2 = (abs(b - c) - 1.0) ** 2 / n if b != c else 0.0
    return PairedTest(
        "mcnemar (chi-square)",
        chi2,
        math.erfc(math.sqrt(chi2 / 2.0)),
        n,
        "items are independent; chi-square with 1 degree of freedom, continuity correction "
        "(needs about 25 or more disagreements)",
    )


def sign_test(a: npt.ArrayLike, b: npt.ArrayLike) -> PairedTest:
    """Test whether A beats B more often than B beats A; ties are dropped.

    ``statistic`` is the number of items where A is above B.

    Raises:
        ValueError: If the arrays differ in length, are empty, or are not finite.
    """
    d = _pair(a, b)
    wins, losses = int(np.sum(d > 0)), int(np.sum(d < 0))
    return PairedTest(
        "sign test",
        float(wins),
        _two_sided_binomial(wins, wins + losses),
        wins + losses,
        "items are independent; ties are dropped; under no difference each item favours A or B "
        "with probability one half",
    )


def _midranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def _exact_signed_rank_p(doubled_ranks: npt.NDArray[np.int64], observed: int) -> float:
    """Two-sided exact p-value: all 2^n sign assignments counted by dynamic programming.

    ``doubled_ranks`` are twice the mid-ranks, so they are integers even with ties. ``observed``
    is twice the positive rank sum.
    """
    total = int(doubled_ranks.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in doubled_ranks:
        counts[r:] = counts[r:] + counts[: total + 1 - r].copy()
    probs = counts / counts.sum()
    lower = float(probs[: observed + 1].sum())
    upper = float(probs[observed:].sum())
    return min(1.0, 2.0 * min(lower, upper))


def wilcoxon(a: npt.ArrayLike, b: npt.ArrayLike) -> PairedTest:
    """Wilcoxon signed-rank test that the paired differences are centred on zero.

    ``statistic`` is the sum of the ranks of the positive differences (the ranks are of the
    absolute differences, ties get their mean rank, zero differences are dropped).

    Raises:
        ValueError: If the arrays differ in length, are empty, or are not finite.
    """
    d = _pair(a, b)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return PairedTest("wilcoxon signed-rank", 0.0, 1.0, 0, "all differences are zero")
    ranks = _midranks(np.abs(d))
    positive = float(ranks[d > 0].sum())
    if n <= EXACT_LIMIT:
        doubled = np.rint(2.0 * ranks).astype(np.int64)
        p = _exact_signed_rank_p(doubled, round(2.0 * positive))
        how = "exact distribution of the signed-rank sum, ties included"
    else:
        _, counts = np.unique(np.abs(d), return_counts=True)
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(counts**3 - counts)) / 48.0
        z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
        p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
        how = "normal approximation with continuity and tie corrections"
    return PairedTest(
        "wilcoxon signed-rank",
        positive,
        p,
        n,
        f"items are independent and the differences are symmetric about their centre; {how}",
    )


def bootstrap_difference(  # noqa: PLR0913 (settings are keyword-only)
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    level: float = 0.95,
    method: Literal["percentile", "basic", "bca"] = "percentile",
    n_resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Return the mean paired difference A - B with a bootstrap interval.

    The items are resampled as pairs, so no distribution is assumed.

    Raises:
        ValueError: If the arrays differ in length, are empty, or are not finite.
    """
    d = _pair(a, b)
    interval_method: IntervalMethod = method
    result = bootstrap(
        d, np.mean, IID(), n_resamples=n_resamples, level=level, method=interval_method, seed=seed
    )
    return result.estimate
