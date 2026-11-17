"""Intervals for a proportion: Wilson, Jeffreys, Clopper-Pearson, Agresti-Coull (ES-1).

``proportion_interval(successes, trials, method=...)`` returns an ``Estimate``: the sample
proportion, the interval, ``trials`` as the sample size, and the method as the assumption.

Small-N behaviour. "Coverage" is the share of samples whose interval holds the true proportion,
computed exactly from the binomial distribution over true proportions from 0.0005 to 0.9995. At
95% and 10 trials (the numbers are in decision D-23, and a test checks each claim below):

- ``"wilson"``: coverage is 95.4% on average, but it falls to 83.6% for a true proportion near 0
  (0.018) and stays above 91% between 0.1 and 0.9. Its interval is never empty and stays inside
  [0, 1]. A good default.
- ``"jeffreys"``: coverage is 95.3% on average and falls to 86.8% for some proportions (the
  lowest is at 0.78); it rises and falls with the proportion. The ends are set to 0 for 0
  successes and to 1 for all successes.
- ``"clopper_pearson"``: coverage never falls below 96.1% (it is exact, so at least 95%), and
  98.4% on average. It has the widest interval (mean width 0.51 against 0.44 for Wilson). Use
  it when under-coverage is not acceptable.
- ``"agresti_coull"``: coverage is 96.5% on average and never below 92.4%. Its interval is
  wider than Wilson's (0.46 against 0.44); its lower end can be below 0 and is then cut to 0.

With 30 trials the lowest coverage is 84.1% (Wilson, near 0), 88.8% (Jeffreys), 95.1%
(Clopper-Pearson) and 93.4% (Agresti-Coull).

References:
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
    *Journal of the American Statistical Association* 22(158):209-212.
    https://doi.org/10.1080/01621459.1927.10502953.

    Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits
    illustrated in the case of the binomial. *Biometrika* 26(4):404-413.
    https://doi.org/10.1093/biomet/26.4.404.

    Agresti, A., & Coull, B. A. (1998). Approximate is better than "exact" for interval
    estimation of binomial proportions. *The American Statistician* 52(2):119-126.
    https://doi.org/10.1080/00031305.1998.10480550.

    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial
    proportion. *Statistical Science* 16(2):101-133. https://doi.org/10.1214/ss/1009213286.
    Jeffreys interval and its ends at 0 and n; the comparison of all four.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal

import numpy as np

from statsx import _accel
from statsx.result import Estimate, Interval

Method = Literal["wilson", "jeffreys", "clopper_pearson", "agresti_coull"]

_ASSUMPTION = {
    "wilson": "independent trials with one success probability; Wilson score interval",
    "jeffreys": "independent trials with one success probability; Jeffreys interval "
    "(Beta(1/2, 1/2) prior)",
    "clopper_pearson": "independent trials with one success probability; Clopper-Pearson "
    "exact interval",
    "agresti_coull": "independent trials with one success probability; Agresti-Coull interval",
}


def _beta_ppf(p: float, a: float, b: float) -> float:
    return _accel.beta_ppf(np.array([p]), a, b)[0]


def interval_ends(successes: int, trials: int, level: float, method: Method) -> tuple[float, float]:
    """Return the lower and upper end of the interval; ``proportion_interval`` checks the input."""
    alpha = 1.0 - level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    n, x = float(trials), float(successes)
    match method:
        case "wilson":
            p = x / n
            denominator = 1.0 + z * z / n
            center = (p + z * z / (2.0 * n)) / denominator
            half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
            return max(0.0, center - half), min(1.0, center + half)
        case "agresti_coull":
            n_tilde = n + z * z
            p_tilde = (x + z * z / 2.0) / n_tilde
            half = z * math.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
            return max(0.0, p_tilde - half), min(1.0, p_tilde + half)
        case "clopper_pearson":
            lower = 0.0 if successes == 0 else _beta_ppf(alpha / 2.0, x, n - x + 1.0)
            upper = 1.0 if successes == trials else _beta_ppf(1.0 - alpha / 2.0, x + 1.0, n - x)
            return lower, upper
        case "jeffreys":
            lower = 0.0 if successes == 0 else _beta_ppf(alpha / 2.0, x + 0.5, n - x + 0.5)
            upper = (
                1.0 if successes == trials else _beta_ppf(1.0 - alpha / 2.0, x + 0.5, n - x + 0.5)
            )
            return lower, upper


def proportion_interval(
    successes: int, trials: int, *, method: Method = "wilson", level: float = 0.95
) -> Estimate:
    """Return the sample proportion with an interval for the true proportion.

    Args:
        successes: Number of successes, between 0 and ``trials``.
        trials: Number of trials, at least 1.
        method: ``"wilson"`` (default), ``"jeffreys"``, ``"clopper_pearson"`` or
            ``"agresti_coull"``. See the module notes for each one's small-N behaviour.
        level: Interval coverage in (0, 1).

    Raises:
        ValueError: If the counts are not integers with 0 <= successes <= trials and trials >= 1,
            ``level`` is not in (0, 1), or ``method`` is unknown.
    """
    if method not in _ASSUMPTION:
        raise ValueError(f"method must be one of {sorted(_ASSUMPTION)}, got {method!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if (
        int(successes) != successes
        or int(trials) != trials
        or not 0 <= successes <= trials
        or trials < 1
    ):
        raise ValueError(
            f"need integer counts with 0 <= successes <= trials, got {successes}/{trials}"
        )
    lower, upper = interval_ends(int(successes), int(trials), level, method)
    return Estimate(
        estimate=successes / trials,
        interval=Interval(lower, upper, level),
        n=int(trials),
        assumption=_ASSUMPTION[method],
    )
