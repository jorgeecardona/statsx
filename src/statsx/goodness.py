"""Goodness of fit at a stated effective sample size, with distances that do not grow with N (DI-2).

With a very large N, a chi-square test rejects almost any fit, because even a tiny misfit becomes
detectable. The p-value then says "the fit is not exact", which you knew. This module reports
three separate things:

- ``p_value``: the Pearson chi-square test, run at ``n_effective`` rows. The statistic is
  ``n_effective * sum((p_hat - pi)^2 / pi)``, where ``p_hat`` are the observed shares and ``pi``
  the expected shares. ``n_effective`` defaults to the number of rows ``n``. Set it lower when the
  rows are not independent (``n / design effect``), or to ask whether the misfit would show at the
  sample size you care about. The result always states both ``n`` and ``n_effective``.
- ``tvd``: the total variation distance, ``1/2 * sum(|p_hat - pi|)``, between observed and
  expected shares. It is the largest difference in the probability of any event, and does not
  depend on N. For continuous data it is measured on the bins, so it is a lower bound of the
  distance between the data and the fit. It includes sampling noise of about
  ``0.4 * sqrt(n_bins / n)``.
- ``wasserstein``: the Wasserstein-1 distance between the data and the fitted distribution, in the
  units of the data. It is the area between the two distribution functions. It is computed
  exactly, with no bins.

``n_to_reject`` is the effective sample size at which the same misfit would be rejected at level
``alpha``. A small value means the misfit is large. A value much larger than your real sample size
means the misfit is too small to matter at that size. It uses the observed shares, which include
sampling noise of about ``n_bins / n`` in the sum of squared differences, so for data that come
exactly from the fit it is about ``n``.

For continuous data, the bins hold equal probability under the fit, and their number follows
Mann & Wald: ``ceil(4 (2 (n - 1)^2 / z^2)^(1/5))`` with ``z`` the ``1 - alpha`` normal quantile,
reduced so that every bin expects at least 5 rows.

References:
    Pearson, K. (1900). On the criterion that a given system of deviations from the probable in
    the case of a correlated system of variables is such that it can be reasonably supposed to
    have arisen from random sampling. *Philosophical Magazine* 50(302):157-175.
    https://doi.org/10.1080/14786440009463897.

    Mann, H. B., & Wald, A. (1942). On the choice of the number of class intervals in the
    application of the chi square test. *Annals of Mathematical Statistics* 13(3):306-317.
    https://doi.org/10.1214/aoms/1177731569.

    Vallender, S. S. (1973). Calculation of the Wasserstein distance between probability
    distributions on the line. *Theory of Probability and its Applications* 18(4):784-786.
    https://doi.org/10.1137/1118101.

    Levin, D. A., Peres, Y., & Wilmer, E. L. *Markov Chains and Mixing Times*, proposition 4.2
    (total variation distance as half the sum of absolute differences). American Mathematical
    Society.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import numpy as np
import numpy.typing as npt

from statsx import _accel

_MIN_EXPECTED = 5
_BISECTIONS = 200


class ContinuousFit(Protocol):
    """A fitted continuous distribution, such as the fits in ``statsx.distributions``."""

    def cdf(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return P(X <= x)."""
        ...

    def ppf(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the quantile at each probability."""
        ...

    def quantile_integral(self, u: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the integral of the quantile function from 0 to ``u``."""
        ...


@dataclass(frozen=True, slots=True)
class GoodnessOfFit:
    """The result of a goodness-of-fit check.

    Args:
        statistic: The chi-square statistic at ``n_effective`` rows.
        df: Its degrees of freedom, ``n_bins - 1 - n_params``.
        p_value: The chance of a statistic at least this large if the fit is exact.
        n: The number of rows.
        n_effective: The number of rows the statistic stands for.
        n_bins: The number of categories or bins.
        tvd: Total variation distance between observed and expected shares.
        wasserstein: Wasserstein-1 distance to the fit, in data units; None for categories.
        n_to_reject: The effective sample size at which this misfit is rejected at ``alpha``;
            infinity when the shares match exactly.
        alpha: The level behind ``n_to_reject`` (and behind the default number of bins).
        assumption: The assumption behind the p-value, in words.
    """

    statistic: float
    df: int
    p_value: float
    n: int
    n_effective: float
    n_bins: int
    tvd: float
    wasserstein: float | None
    n_to_reject: float
    alpha: float
    assumption: str


def _shares(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError(f"{name} must be a non-empty array of finite, non-negative numbers")
    return x


def total_variation_distance(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    """Return ``1/2 * sum(|p - q|)`` for two probability vectors.

    Raises:
        ValueError: If the vectors differ in length, hold a negative or non-finite number, or do
            not add up to 1.
    """
    a, b = _shares(p, "p"), _shares(q, "q")
    if a.size != b.size:
        raise ValueError("p and q must have the same length")
    if abs(a.sum() - 1.0) > 1e-9 or abs(b.sum() - 1.0) > 1e-9:  # noqa: PLR2004
        raise ValueError("p and q must each add up to 1")
    return float(0.5 * np.abs(a - b).sum())


def _chi2_quantile(p: float, df: int) -> float:
    """Return the chi-square quantile at ``p`` by bisection on the cdf."""
    hi = float(df)
    while _accel.chi2_cdf(np.array([hi]), float(df))[0] < p:
        hi *= 2.0
    lo = 0.0
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if _accel.chi2_cdf(np.array([mid]), float(df))[0] < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _check_level(alpha: float, n_params: int) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_params < 0:
        raise ValueError(f"n_params must not be negative, got {n_params}")


def _effective(n: int, n_effective: float | None) -> float:
    value = float(n) if n_effective is None else float(n_effective)
    if not 0.0 < value <= n + 1e-9:
        raise ValueError(f"n_effective must be in (0, n], got {value} with n = {n}")
    return value


def _assemble(  # noqa: PLR0913 (settings are keyword-only)
    counts: npt.NDArray[np.float64],
    probs: npt.NDArray[np.float64],
    *,
    n_effective: float | None,
    n_params: int,
    alpha: float,
    wasserstein: float | None,
    how: str,
) -> GoodnessOfFit:
    n = round(counts.sum())
    eff = _effective(n, n_effective)
    df = counts.size - 1 - n_params
    if df < 1:
        raise ValueError(f"degrees of freedom are {df}: use more bins or fewer fitted parameters")
    shares = counts / n
    spread = float(np.sum((shares - probs) ** 2 / probs))
    statistic = eff * spread
    p_value = float(_accel.chi2_sf(np.array([statistic]), float(df))[0])
    to_reject = _chi2_quantile(1.0 - alpha, df) / spread if spread > 0.0 else math.inf
    return GoodnessOfFit(
        statistic=statistic,
        df=df,
        p_value=p_value,
        n=n,
        n_effective=eff,
        n_bins=counts.size,
        tvd=float(0.5 * np.abs(shares - probs).sum()),
        wasserstein=wasserstein,
        n_to_reject=to_reject,
        alpha=alpha,
        assumption=(
            f"rows are independent, or counted as {eff:g} independent rows; {how}; "
            "the p-value is exact for large expected counts and a fit that does not come from "
            "the same rows"
        ),
    )


def chi_square_test(
    observed: npt.ArrayLike,
    expected_probs: npt.ArrayLike,
    *,
    n_effective: float | None = None,
    n_params: int = 0,
    alpha: float = 0.05,
) -> GoodnessOfFit:
    """Test counts per category against expected shares, at a stated effective sample size.

    Args:
        observed: The count of rows in each category.
        expected_probs: The share expected in each category; positive, adding up to 1.
        n_effective: The number of independent rows the statistic stands for, at most the
            number of rows. Defaults to the number of rows.
        n_params: The number of parameters estimated from these rows to get ``expected_probs``;
            it lowers the degrees of freedom.
        alpha: The level for ``n_to_reject``.

    Raises:
        ValueError: If the inputs are malformed, ``n_effective`` is outside (0, n], or the
            degrees of freedom are below 1.
    """
    counts = _shares(observed, "observed")
    probs = _shares(expected_probs, "expected_probs")
    if counts.size != probs.size:
        raise ValueError("observed and expected_probs must have the same length")
    if np.any(probs <= 0.0) or abs(probs.sum() - 1.0) > 1e-9:  # noqa: PLR2004
        raise ValueError("expected_probs must be positive and add up to 1")
    _check_level(alpha, n_params)
    return _assemble(
        counts,
        probs,
        n_effective=n_effective,
        n_params=n_params,
        alpha=alpha,
        wasserstein=None,
        how="Pearson chi-square test on categories",
    )


def wasserstein_distance(values: npt.ArrayLike, fit: ContinuousFit) -> float:
    """Return the Wasserstein-1 distance between the rows and a fitted distribution.

    It is the integral over ``u`` in (0, 1) of ``|F_n^-1(u) - F^-1(u)|``, with ``F_n`` the
    empirical distribution of the rows. On each step of ``F_n`` the integral is exact, using
    ``fit.quantile_integral``.

    Raises:
        ValueError: If there are no values or one is not finite.
    """
    x = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a non-empty array of finite numbers")
    n = x.size
    edges = np.linspace(0.0, 1.0, n + 1)
    lower, upper = edges[:-1], edges[1:]
    star = np.clip(np.asarray(fit.cdf(x), dtype=np.float64), lower, upper)
    m_lower = np.asarray(fit.quantile_integral(lower), dtype=np.float64)
    m_upper = np.asarray(fit.quantile_integral(upper), dtype=np.float64)
    m_star = np.asarray(fit.quantile_integral(star), dtype=np.float64)
    below = x * (star - lower) - (m_star - m_lower)
    above = (m_upper - m_star) - x * (upper - star)
    return float(np.sum(below + above))


def _default_bins(n_effective: float, alpha: float) -> int:
    z = NormalDist().inv_cdf(1.0 - alpha)
    return math.ceil(4.0 * (2.0 * (n_effective - 1.0) ** 2 / z**2) ** 0.2)


def goodness_of_fit(  # noqa: PLR0913 (settings are keyword-only)
    values: npt.ArrayLike,
    fit: ContinuousFit,
    *,
    n_effective: float | None = None,
    n_params: int = 0,
    n_bins: int | None = None,
    alpha: float = 0.05,
) -> GoodnessOfFit:
    """Check how well a fitted continuous distribution describes the rows.

    The rows are put in bins of equal probability under the fit, and ``chi_square_test`` runs on
    the counts. ``wasserstein`` is computed on the rows directly.

    Args:
        values: The rows.
        fit: The fitted distribution, for example from ``statsx.distributions``.
        n_effective: As in ``chi_square_test``. Also sets the default number of bins.
        n_params: The number of parameters of ``fit`` that were estimated from these rows (2 for
            the fits in ``statsx.distributions``), or 0 when the fit comes from other data.
        n_bins: The number of bins. Defaults to the Mann-Wald rule, at most ``n / 5`` so that
            every bin expects at least 5 rows.
        alpha: The level for ``n_to_reject`` and the default number of bins.

    Raises:
        ValueError: If there are fewer than 25 rows, a value is not finite, ``n_effective`` is
            outside (0, n], or the degrees of freedom are below 1.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < _MIN_EXPECTED**2 or not np.all(np.isfinite(x)):
        raise ValueError("at least 25 finite values are needed")
    _check_level(alpha, n_params)
    eff = _effective(x.size, n_effective)
    most = x.size // _MIN_EXPECTED
    bins = min(_default_bins(eff, alpha), most) if n_bins is None else n_bins
    if not 2 <= bins <= most:  # noqa: PLR2004
        raise ValueError(f"n_bins must be from 2 to {most} (n / 5), got {bins}")
    edges = np.asarray(fit.ppf(np.arange(1, bins) / bins), dtype=np.float64)
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=bins)
    return _assemble(
        counts.astype(np.float64),
        np.full(bins, 1.0 / bins),
        n_effective=n_effective,
        n_params=n_params,
        alpha=alpha,
        wasserstein=wasserstein_distance(x, fit),
        how=f"Pearson chi-square test on {bins} bins of equal probability under the fit",
    )
