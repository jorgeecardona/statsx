"""One definition of practical equivalence, for the TOST test and the ROPE probability.

An ``EquivalenceMargin`` is the interval of differences that count as practically zero. Every
function that asks "are these the same in practice?" takes this one object, so a TOST test on
data and a ROPE probability on posterior draws (``statsx.posterior.compare``) mean the same thing:

- ``tost`` is the frequentist test. Two one-sided t tests reject "the difference is at or below
  the lower margin" and "the difference is at or above the upper margin". Both must reject, so
  the p-value is the larger of the two. It is equivalent to the ``1 - 2 alpha`` confidence
  interval of the difference lying inside the margin.
- ``rope_probability`` is the Bayesian counterpart: the share of posterior draws of the
  difference inside the margin (the region of practical equivalence).

Failing to show equivalence does not show a difference: a small sample often gives a wide
interval that is inside neither the margin nor outside it.

References:
    Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power
    approach for assessing the equivalence of average bioavailability. *Journal of
    Pharmacokinetics and Biopharmaceutics* 15(6):657-680. https://doi.org/10.1007/BF01068419.

    Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and
    meta-analyses. *Social Psychological and Personality Science* 8(4):355-362.
    https://doi.org/10.1177/1948550617697177.

    Welch, B. L. (1947). The generalization of "Student's" problem when several different
    population variances are involved. *Biometrika* 34(1-2):28-35.
    https://doi.org/10.1093/biomet/34.1-2.28. Standard error and degrees of freedom for two
    independent samples.

    Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation.
    *Advances in Methods and Practices in Psychological Science* 1(2):270-280.
    https://doi.org/10.1177/2515245918771304. The region of practical equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.result import Estimate, Interval


@dataclass(frozen=True, slots=True)
class EquivalenceMargin:
    """The differences that count as practically zero: ``low <= difference <= high``.

    Raises:
        ValueError: If ``low`` is not below ``high``, or the margin does not contain 0.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.low) and math.isfinite(self.high) and self.low < 0.0 < self.high
        ):
            raise ValueError(f"the margin must have low < 0 < high, got ({self.low}, {self.high})")

    @classmethod
    def symmetric(cls, delta: float) -> EquivalenceMargin:
        """Return the margin ``(-delta, delta)``; ``delta`` must be > 0."""
        if not delta > 0.0:
            raise ValueError(f"the margin delta must be > 0, got {delta}")
        return cls(-delta, delta)

    def contains(self, value: float) -> bool:
        """Return True when ``value`` is inside the margin, ends included."""
        return self.low <= value <= self.high


def as_margin(margin: EquivalenceMargin | float) -> EquivalenceMargin:
    """Return ``margin`` as an ``EquivalenceMargin``; a number ``d`` means ``(-d, d)``."""
    return margin if isinstance(margin, EquivalenceMargin) else EquivalenceMargin.symmetric(margin)


def rope_probability(difference_draws: npt.ArrayLike, margin: EquivalenceMargin | float) -> float:
    """Return the share of posterior draws of a difference that lie inside the margin.

    Raises:
        ValueError: If the draws are not a 1-D array of at least 2 finite values.
    """
    x = np.asarray(difference_draws, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):  # noqa: PLR2004
        raise ValueError("draws must be a 1-D array of at least 2 finite values")
    m = as_margin(margin)
    return float(np.mean((x >= m.low) & (x <= m.high)))


@dataclass(frozen=True, slots=True)
class EquivalenceTest:
    """The result of a TOST equivalence test.

    Args:
        equivalent: True when both one-sided tests reject at ``alpha``.
        p_value: The larger of the two one-sided p-values.
        difference: The estimated difference with its ``1 - 2 alpha`` confidence interval.
        margin: The margin that was tested.
        alpha: The level of each one-sided test.
        df: Degrees of freedom of the t distribution.
    """

    equivalent: bool
    p_value: float
    difference: Estimate
    margin: EquivalenceMargin
    alpha: float
    df: float


def _t_cdf(x: float, df: float) -> float:
    return _accel.t_cdf(np.array([x]), df)[0]


def _t_ppf(p: float, df: float) -> float:
    return _accel.t_ppf(np.array([p]), df)[0]


def tost(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    margin: EquivalenceMargin | float,
    *,
    alpha: float = 0.05,
    paired: bool = False,
) -> EquivalenceTest:
    """Test whether the mean of ``a`` minus the mean of ``b`` is practically zero (TOST).

    Args:
        a: Observations of the first group.
        b: Observations of the second group. With ``paired=True``, ``b[i]`` is paired with
            ``a[i]`` and the test is on the differences.
        margin: An ``EquivalenceMargin``, or a number ``d`` for ``(-d, d)``, in the unit of the
            data.
        alpha: The level of each one-sided test, in (0, 0.5).
        paired: Treat the groups as pairs.

    Raises:
        ValueError: If a group has fewer than 2 finite observations, the groups are not the
            same length when paired, the data have no spread, or ``alpha`` is out of range.
    """
    if not 0.0 < alpha < 0.5:  # noqa: PLR2004
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    m = as_margin(margin)
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if min(x.size, y.size) < 2 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):  # noqa: PLR2004
        raise ValueError("each group needs at least 2 finite observations")
    if paired:
        if x.size != y.size:
            raise ValueError("paired groups must have the same length")
        d = x - y
        estimate, se, df = float(d.mean()), float(d.std(ddof=1)) / math.sqrt(d.size), d.size - 1.0
        assumption = "paired differences are independent and about normal; two one-sided t tests"
    else:
        v1, v2 = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
        estimate, se = float(x.mean() - y.mean()), math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1**2 / (x.size - 1.0) + v2**2 / (y.size - 1.0)) if se > 0 else 1.0
        assumption = (
            "two independent groups, about normal, unequal variances allowed (Welch); "
            "two one-sided t tests"
        )
    if not se > 0.0:
        raise ValueError("the data have no spread: the test is undefined")
    p_lower = 1.0 - _t_cdf((estimate - m.low) / se, df)
    p_upper = _t_cdf((estimate - m.high) / se, df)
    half = _t_ppf(1.0 - alpha, df) * se
    interval = Interval(estimate - half, estimate + half, 1.0 - 2.0 * alpha)
    p_value = max(p_lower, p_upper)
    return EquivalenceTest(
        equivalent=p_value < alpha,
        p_value=p_value,
        difference=Estimate(
            estimate, interval, int(x.size if paired else x.size + y.size), assumption
        ),
        margin=m,
        alpha=alpha,
        df=float(df),
    )
