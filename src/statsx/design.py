"""Power, sample size and minimum detectable effect for planning a run (DE-1, DE-2).

Every test has three functions: ``power_*`` (the chance the test finds the effect, for a given
sample size), ``sample_size_*`` (the smallest sample size that reaches a power) and
``minimum_detectable_effect_*`` (the smallest effect a fixed sample size finds with that power,
DE-2). Each returns a ``Design``.

Tests covered:

- Paired differences, or one sample against a value: the t test, with the exact power from the
  noncentral t distribution. ``n`` is the number of pairs; ``effect`` is the mean difference and
  ``sd`` the standard deviation of the differences.
- Two proportions: the pooled z test. The power is the normal approximation of Fleiss, Levin &
  Paik; ``exact=True`` on ``power_two_proportions`` counts every outcome of the two binomials
  instead. ``n`` is the number in each group. Exact power is not a smooth function of ``n``
  (it moves up and down), so ``sample_size_two_proportions`` gives the n of the approximation.
- Equivalence (TOST) of a paired difference, and of a difference of two proportions, against an
  ``EquivalenceMargin`` (the one definition of ``statsx.equivalence``). The paired power is exact
  (a numerical integral over the sample standard deviation); the proportions use the normal
  approximation of Blackwelder (1982).

Planning inputs (the effect, the spread, the baseline rate) are guesses. The power holds only if
they are right: use a cautious spread.

References:
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.,
    chapter 2. Lawrence Erlbaum. Power of the t test.

    Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and
    Proportions*, 3rd ed., chapter 4. Wiley. https://doi.org/10.1002/0471445428.

    Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power
    approach for assessing the equivalence of average bioavailability. *Journal of
    Pharmacokinetics and Biopharmaceutics* 15(6):657-680. https://doi.org/10.1007/BF01068419.

    Blackwelder, W. C. (1982). "Proving the null hypothesis" in clinical trials. *Controlled
    Clinical Trials* 3(4):345-353. https://doi.org/10.1016/0197-2456(82)90024-1.

    Owen, D. B. (1965). A special case of a bivariate non-central t-distribution. *Biometrika*
    52(3-4):437-446. https://doi.org/10.1093/biomet/52.3-4.437. The power of TOST.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.equivalence import EquivalenceMargin, as_margin

Alternative = Literal["two-sided", "greater"]
_NORMAL = NormalDist()
_MAX_N = 10_000_000
_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Design:
    """One planning answer: a sample size, an effect and the power they give.

    Args:
        test: The test the answer is for.
        n: Pairs (paired tests) or items per group (two proportions).
        effect: The true effect assumed: a difference in the unit of the data, or in proportion.
        power: The chance that the test finds the effect.
        alpha: The test's level.
        assumption: The planning assumption, in words.
    """

    test: str
    n: int
    effect: float
    power: float
    alpha: float
    assumption: str


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:  # noqa: PLR2004
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")


def _check_power(power: float) -> None:
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")


def _check_n(n: int, minimum: int) -> None:
    if int(n) != n or n < minimum:
        raise ValueError(f"n must be a whole number of at least {minimum}, got {n}")


def _smallest_n(power_of: Callable[[int], float], target: float, start: int) -> int:
    """Return the smallest n with ``power_of(n) >= target``, if the power rises with n."""
    low, high = start, start
    while power_of(high) < target:
        low, high = high, high * 2
        if high > _MAX_N:
            raise ValueError(
                "the sample size needed is more than 10 million: the effect is too small"
            )
    while low < high:
        mid = (low + high) // 2
        if power_of(mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def _solve_effect(
    power_of: Callable[[float], float], target: float, low: float, high: float
) -> float:
    """Return the effect in [low, high] where a rising ``power_of`` reaches ``target``."""
    if power_of(high) < target:
        raise ValueError("this sample size cannot reach that power for any effect in range")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if power_of(mid) >= target:
            high = mid
        else:
            low = mid
        if high - low < _TOLERANCE:
            break
    return high


def _t_power(effect: float, sd: float, n: int, alpha: float, alternative: Alternative) -> float:
    df = n - 1.0
    delta = effect * math.sqrt(n) / sd
    if alternative == "greater":
        critical = _accel.t_ppf(np.array([1.0 - alpha]), df)[0]
        return 1.0 - _accel.nct_cdf(np.array([critical]), df, delta)[0]
    critical = _accel.t_ppf(np.array([1.0 - alpha / 2.0]), df)[0]
    upper = 1.0 - _accel.nct_cdf(np.array([critical]), df, delta)[0]
    lower = _accel.nct_cdf(np.array([-critical]), df, delta)[0]
    return float(upper + lower)


def power_paired(
    effect: float,
    sd: float,
    n: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> Design:
    """Return the power of the paired t test to find a mean difference ``effect``.

    Args:
        effect: The true mean difference (positive for ``"greater"``).
        sd: The standard deviation of the paired differences. Must be > 0.
        n: Number of pairs, at least 2.
        alpha: The test's level, in (0, 0.5).
        alternative: ``"two-sided"`` or ``"greater"``.

    Raises:
        ValueError: If an input is out of range.
    """
    _check_alpha(alpha)
    _check_n(n, 2)
    if not sd > 0.0 or not math.isfinite(effect):
        raise ValueError("sd must be > 0 and effect finite")
    return Design(
        f"paired t test ({alternative})",
        int(n),
        effect,
        _t_power(effect, sd, int(n), alpha, alternative),
        alpha,
        "differences independent and about normal with the given sd",
    )


def sample_size_paired(
    effect: float,
    sd: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> Design:
    """Return the smallest number of pairs whose power for ``effect`` is at least ``power``."""
    _check_alpha(alpha)
    _check_power(power)
    if not sd > 0.0 or not effect != 0.0 or not math.isfinite(effect):
        raise ValueError("sd must be > 0 and effect finite and not 0")
    n = _smallest_n(lambda k: _t_power(effect, sd, k, alpha, alternative), power, 2)
    return power_paired(effect, sd, n, alpha=alpha, alternative=alternative)


def minimum_detectable_effect_paired(
    sd: float,
    n: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> Design:
    """Return the smallest mean difference that ``n`` pairs find with the given power (DE-2)."""
    _check_alpha(alpha)
    _check_power(power)
    _check_n(n, 2)
    if not sd > 0.0:
        raise ValueError("sd must be > 0")
    effect = _solve_effect(
        lambda e: _t_power(e, sd, int(n), alpha, alternative), power, 0.0, 20.0 * sd
    )
    return power_paired(effect, sd, int(n), alpha=alpha, alternative=alternative)


def _check_p(*values: float) -> None:
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("proportions must be in [0, 1]")


def _approx_power_proportions(
    p1: float, p2: float, n: int, alpha: float, alternative: Alternative
) -> float:
    z = _NORMAL.inv_cdf(1.0 - (alpha / 2.0 if alternative == "two-sided" else alpha))
    pbar = 0.5 * (p1 + p2)
    null_sd = math.sqrt(2.0 * pbar * (1.0 - pbar))
    alt_sd = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    if alt_sd == 0.0:
        return 1.0 if p1 != p2 else 0.0
    shift = abs(p1 - p2) * math.sqrt(n)
    upper = _NORMAL.cdf((shift - z * null_sd) / alt_sd)
    if alternative == "greater":
        return _NORMAL.cdf((((p2 - p1) * math.sqrt(n)) - z * null_sd) / alt_sd)
    return upper + _NORMAL.cdf((-shift - z * null_sd) / alt_sd)


def _log_binomial_pmf(n: int, p: float) -> npt.NDArray[np.float64]:
    k = np.arange(n + 1, dtype=np.float64)
    log_choose = np.array(
        [
            math.lgamma(n + 1.0) - math.lgamma(j + 1.0) - math.lgamma(n - j + 1.0)
            for j in range(n + 1)
        ]
    )
    if p <= 0.0:
        out = np.full(n + 1, -np.inf)
        out[0] = 0.0
        return out
    if p >= 1.0:
        out = np.full(n + 1, -np.inf)
        out[n] = 0.0
        return out
    return log_choose + k * math.log(p) + (n - k) * math.log1p(-p)


def _exact_power_proportions(
    p1: float, p2: float, n: int, alpha: float, alternative: Alternative
) -> float:
    z = _NORMAL.inv_cdf(1.0 - (alpha / 2.0 if alternative == "two-sided" else alpha))
    x1 = np.arange(n + 1, dtype=np.float64)[:, None]
    x2 = np.arange(n + 1, dtype=np.float64)[None, :]
    pooled = (x1 + x2) / (2.0 * n)
    spread = np.sqrt(np.maximum(pooled * (1.0 - pooled) * 2.0 / n, 1e-300))
    statistic = (x2 - x1) / n / spread  # group 2 minus group 1
    both_zero = (pooled == 0.0) | (pooled == 1.0)
    reject = np.abs(statistic) > z if alternative == "two-sided" else statistic > z
    reject = reject & ~both_zero
    mass: npt.NDArray[np.float64] = np.exp(
        _log_binomial_pmf(n, p1)[:, None] + _log_binomial_pmf(n, p2)[None, :]
    )
    return float(mass[reject].sum())


def power_two_proportions(  # noqa: PLR0913 (settings are keyword-only)
    p1: float,
    p2: float,
    n: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    exact: bool = False,
) -> Design:
    """Return the power of the pooled z test to tell two groups with rates ``p1`` and ``p2`` apart.

    Args:
        p1: The true rate in group 1 (the control).
        p2: The true rate in group 2. ``"greater"`` tests ``p2 > p1``.
        n: Items in each group, at least 2.
        alpha: The test's level, in (0, 0.5).
        alternative: ``"two-sided"`` or ``"greater"``.
        exact: Count every outcome of the two binomials instead of the normal approximation
            (n up to 2000).

    Raises:
        ValueError: If an input is out of range, or ``exact`` is asked for n above 2000.
    """
    _check_alpha(alpha)
    _check_p(p1, p2)
    _check_n(n, 2)
    if exact and n > 2000:  # noqa: PLR2004
        raise ValueError("exact power is for n up to 2000")
    compute = _exact_power_proportions if exact else _approx_power_proportions
    return Design(
        f"two-proportion z test ({alternative})",
        int(n),
        p2 - p1,
        float(compute(p1, p2, int(n), alpha, alternative)),
        alpha,
        "independent items; pooled z test; "
        + ("exact binomial enumeration" if exact else "normal approximation"),
    )


def sample_size_two_proportions(
    p1: float,
    p2: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> Design:
    """Return the smallest n per group whose approximate power is at least ``power``."""
    _check_alpha(alpha)
    _check_power(power)
    _check_p(p1, p2)
    if p1 == p2 or (alternative == "greater" and p2 < p1):
        raise ValueError("the effect must be non-zero (and positive for 'greater')")
    n = _smallest_n(lambda k: _approx_power_proportions(p1, p2, k, alpha, alternative), power, 2)
    return power_two_proportions(p1, p2, n, alpha=alpha, alternative=alternative)


def minimum_detectable_effect_two_proportions(  # noqa: PLR0913 (settings are keyword-only)
    p1: float,
    n: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    direction: Literal["increase", "decrease"] = "increase",
) -> Design:
    """Return the smallest change from rate ``p1`` that ``n`` per group finds with ``power`` (DE-2).

    ``direction`` says whether the second group's rate is above (``"increase"``) or below
    (``"decrease"``) ``p1``.
    """
    _check_alpha(alpha)
    _check_power(power)
    _check_p(p1)
    _check_n(n, 2)
    sign = 1.0 if direction == "increase" else -1.0
    room = (1.0 - p1) if direction == "increase" else p1
    if room <= 0.0:
        raise ValueError("there is no room to move in that direction")
    gap = _solve_effect(
        lambda g: _approx_power_proportions(p1, p1 + sign * g, int(n), alpha, alternative),
        power,
        0.0,
        room,
    )
    return power_two_proportions(p1, p1 + sign * gap, int(n), alpha=alpha, alternative=alternative)


@lru_cache(maxsize=1)
def _quadrature() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the nodes and weights of a 200-point Gauss-Legendre rule on [-1, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(200)
    return nodes, weights


def _tost_paired_power(  # noqa: PLR0913, PLR0917 (all inputs of one power)
    effect: float, sd: float, n: int, alpha: float, low: float, high: float
) -> float:
    """Exact TOST power: an integral over the sample sd of the chance the mean is inside."""
    df = n - 1.0
    tc = _accel.t_ppf(np.array([1.0 - alpha]), df)[0]
    root_n = math.sqrt(n)
    spread = 12.0 / math.sqrt(df)
    w_low, w_high = max(1e-6, 1.0 - min(spread, 0.999)), 1.0 + spread * 2.0
    nodes, weights = _quadrature()
    w = 0.5 * (w_high - w_low) * nodes + 0.5 * (w_high + w_low)
    scale = 0.5 * (w_high - w_low) * weights
    log_density = (
        math.log(2.0)
        + 0.5 * df * math.log(df / 2.0)
        - math.lgamma(df / 2.0)
        + (df - 1.0) * np.log(w)
        - 0.5 * df * w * w
    )
    upper_edge = (high - tc * sd * w / root_n - effect) * root_n / sd
    lower_edge = (low + tc * sd * w / root_n - effect) * root_n / sd
    cdf = np.vectorize(_NORMAL.cdf, otypes=[np.float64])
    inside: npt.NDArray[np.float64] = np.maximum(cdf(upper_edge) - cdf(lower_edge), 0.0)
    integrand: npt.NDArray[np.float64] = scale * np.exp(log_density) * inside
    return float(integrand.sum())


def power_equivalence_paired(
    effect: float,
    sd: float,
    margin: EquivalenceMargin | float,
    n: int,
    *,
    alpha: float = 0.05,
) -> Design:
    """Return the power of the paired TOST to show the difference is inside ``margin``.

    ``effect`` is the true mean difference (0 for no difference). The power is 0 when ``effect``
    is outside the margin and below ``alpha`` at its edge.

    Raises:
        ValueError: If an input is out of range.
    """
    _check_alpha(alpha)
    _check_n(n, 2)
    m = as_margin(margin)
    if not sd > 0.0 or not math.isfinite(effect):
        raise ValueError("sd must be > 0 and effect finite")
    return Design(
        "paired TOST (equivalence)",
        int(n),
        effect,
        _tost_paired_power(effect, sd, int(n), alpha, m.low, m.high),
        alpha,
        f"differences independent and about normal with the given sd; margin ({m.low}, {m.high})",
    )


def sample_size_equivalence_paired(
    sd: float,
    margin: EquivalenceMargin | float,
    *,
    effect: float = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
) -> Design:
    """Return the smallest number of pairs whose TOST power is at least ``power``.

    Raises:
        ValueError: If ``effect`` is not inside the margin (equivalence can never be shown).
    """
    _check_alpha(alpha)
    _check_power(power)
    m = as_margin(margin)
    if not sd > 0.0 or not m.low < effect < m.high:
        raise ValueError("sd must be > 0 and the true effect inside the margin")
    n = _smallest_n(lambda k: _tost_paired_power(effect, sd, k, alpha, m.low, m.high), power, 2)
    return power_equivalence_paired(effect, sd, m, n, alpha=alpha)


def _tost_proportions_power(  # noqa: PLR0913, PLR0917 (all inputs of one power)
    p1: float, p2: float, n: int, alpha: float, low: float, high: float
) -> float:
    z = _NORMAL.inv_cdf(1.0 - alpha)
    se = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n)
    if se == 0.0:
        return 1.0 if low < p2 - p1 < high else 0.0
    gap = p2 - p1
    return max(0.0, _NORMAL.cdf((high - gap) / se - z) + _NORMAL.cdf((gap - low) / se - z) - 1.0)


def power_equivalence_two_proportions(
    p1: float,
    p2: float,
    margin: EquivalenceMargin | float,
    n: int,
    *,
    alpha: float = 0.05,
) -> Design:
    """Return the power of the TOST on a difference of two rates (p2 minus p1) inside ``margin``.

    Normal approximation with the standard error at the true rates (Blackwelder 1982).

    Raises:
        ValueError: If an input is out of range.
    """
    _check_alpha(alpha)
    _check_p(p1, p2)
    _check_n(n, 2)
    m = as_margin(margin)
    return Design(
        "two-proportion TOST (equivalence)",
        int(n),
        p2 - p1,
        _tost_proportions_power(p1, p2, int(n), alpha, m.low, m.high),
        alpha,
        f"independent items; normal approximation; margin ({m.low}, {m.high})",
    )


def sample_size_equivalence_two_proportions(
    p1: float,
    p2: float,
    margin: EquivalenceMargin | float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> Design:
    """Return the smallest n per group whose approximate TOST power is at least ``power``."""
    _check_alpha(alpha)
    _check_power(power)
    _check_p(p1, p2)
    m = as_margin(margin)
    if not m.low < p2 - p1 < m.high:
        raise ValueError("the true difference must be inside the margin")
    n = _smallest_n(lambda k: _tost_proportions_power(p1, p2, k, alpha, m.low, m.high), power, 2)
    return power_equivalence_two_proportions(p1, p2, m, n, alpha=alpha)
