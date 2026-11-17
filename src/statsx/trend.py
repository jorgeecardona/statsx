"""Trend and forecast of a count series (TS-1, TS-2, TS-3, TS-5).

``fit_trend`` fits a negative binomial structural time series: a level, a slope, a random-walk
drift, and a Fourier season. The posterior comes from the nested Laplace approximation, corrected
by importance sampling. The fit is checked before it is returned: ``FitCheck`` lists the reasons
a fit fails, and ``fit_trend`` raises ``FitCheckError`` instead of returning a degraded fit.

References:
    Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent
    Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal
    Statistical Society B* 71(2):319-392. https://doi.org/10.1111/j.1467-9868.2008.00700.x.

    Tierney, L., & Kadane, J. B. (1986). Accurate approximations for posterior moments and
    marginal densities. *Journal of the American Statistical Association* 81(393):82-86.
    https://doi.org/10.1080/01621459.1986.10478240.

    Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed
    importance sampling. *Journal of Machine Learning Research* 25(72):1-58.

    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed.,
    chapter 4 (filtering, smoothing and forecasting). Oxford University Press.
    https://doi.org/10.1093/acprof:oso/9780199641178.001.0001.

    Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data*, 2nd ed.,
    chapter 3. Cambridge University Press. https://doi.org/10.1017/CBO9781139013567.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel

KHAT_CAP = 0.7
"""Largest k-hat accepted; with few draws the limit is lower (see ``khat_limit``)."""

ESS_MIN = 400.0
"""Smallest importance sampling effective sample size accepted."""

EDGE_MASS_MAX = 0.02
"""Largest posterior mass on an outer row or column of the hyperparameter grid."""

MIN_DRAWS = 400
"""Fewer draws cannot reach ``ESS_MIN``."""


def khat_limit(n_draws: int) -> float:
    """Return the largest k-hat accepted for ``n_draws`` draws, min(1 - 1/log10(S), 0.7)."""
    return min(1.0 - 1.0 / math.log10(n_draws), KHAT_CAP)


@dataclass(frozen=True, slots=True)
class FitCheck:
    """The numbers behind one fit check, and the reasons it fails.

    ``passed`` is True when no reason is listed.
    """

    khat: float
    khat_limit: float
    ess: float
    ess_min: float
    edge_mass: float
    edge_mass_max: float
    unconverged: int
    not_positive_definite: int
    draw_failures: int
    n_draws: int
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return True when no threshold failed."""
        return not self.reasons

    def require(self) -> None:
        """Raise ``FitCheckError`` when the check failed; return None otherwise."""
        if self.reasons:
            raise FitCheckError(self)


class FitCheckError(RuntimeError):
    """Raised when a fit fails its check; ``check`` holds the numbers and the reasons."""

    def __init__(self, check: FitCheck) -> None:
        super().__init__("fit check failed: " + "; ".join(check.reasons))
        self.check = check


@dataclass(frozen=True, slots=True)
class Band:
    """A credible band: ``lower`` and ``upper`` hold one value per month."""

    level: float
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


_ETA_LIMIT = 30.0
"""Same clamp on the log mean as the Rust fit."""

_DISPERSION_FLOOR = 1e-3

BandOf = Literal["counts", "mean", "trend"]


@dataclass(frozen=True, slots=True)
class Forecast:
    """Posterior draws for every fitted month and every forecast month (TS-2).

    Each array has shape (draws, fitted months + horizon).

    Args:
        trend: Draws of the log trend (level + slope + drift).
        mean: Draws of the expected count, the exponential of trend plus season.
        counts: Draws of a new count, from the negative binomial given ``mean``.
        n_fitted: Number of fitted months; later columns are forecast months.
    """

    trend: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    n_fitted: int

    @property
    def horizon(self) -> int:
        """Return the number of forecast months."""
        return self.counts.shape[1] - self.n_fitted

    def bands(
        self, levels: tuple[float, ...] = (0.5, 0.8, 0.95), *, of: BandOf = "counts"
    ) -> tuple[Band, ...]:
        """Return nested central credible bands, one per level, for every month.

        Args:
            levels: Increasing levels in (0, 1). Each band contains the one before it.
            of: ``"counts"`` for a new count (the default), ``"mean"`` for the expected count,
                ``"trend"`` for the log trend.

        Raises:
            ValueError: If ``levels`` is empty, not increasing, or outside (0, 1).
        """
        if (
            not levels
            or any(not 0.0 < v < 1.0 for v in levels)
            or list(levels) != sorted(set(levels))
        ):
            raise ValueError(f"levels must increase within (0, 1), got {levels}")
        draws = {"counts": self.counts, "mean": self.mean, "trend": self.trend}[of]
        out: list[Band] = []
        for level in levels:
            tail = (1.0 - level) / 2.0
            lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=0)
            out.append(Band(level, lower, upper))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class TrendFit:
    """A checked fit. Only ``fit_trend`` builds it, and only after the check passed.

    Args:
        latent: Draws of the latent field, shape (draws, columns + months): the fixed effects
            (intercept, slope per ``period`` months, then cosine and sine per harmonic), then the
            random-walk drift for each month.
        hyper: Draws of (log drift sd, log dispersion), shape (draws, 2).
        check: The fit check, which passed.
        period: Season length in months.
        harmonics: Number of Fourier harmonics.
    """

    latent: npt.NDArray[np.float64]
    hyper: npt.NDArray[np.float64]
    check: FitCheck
    period: int
    harmonics: int

    @property
    def n_months(self) -> int:
        """Return the number of months fitted."""
        return self.latent.shape[1] - 2 - 2 * self.harmonics

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """Return the draws of the slope of the log trend, per ``period`` months."""
        return self.latent[:, 1]

    @property
    def trend(self) -> npt.NDArray[np.float64]:
        """Return draws of the log trend, shape (draws, months): level + slope + drift."""
        months = self.n_months
        centred = (np.arange(months) - (months - 1) / 2) / self.period
        drift = self.latent[:, -months:]
        return self.latent[:, [0]] + self.latent[:, [1]] * centred + drift

    def band(self, level: float = 0.9) -> Band:
        """Return the central credible band of the log trend at ``level``.

        Raises:
            ValueError: If ``level`` is not in (0, 1).
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lower, upper = np.quantile(self.trend, [tail, 1.0 - tail], axis=0)
        return Band(level, lower, upper)

    def forecast(self, horizon: int, *, seed: int = 0) -> Forecast:
        """Return draws for the fitted months and the next ``horizon`` months.

        Each posterior draw continues its own random walk: the drift gains ``sigma`` times a
        standard normal step per month, so the trend band widens with the horizon. The season
        repeats. New counts are drawn from the negative binomial with that draw's dispersion.

        Args:
            horizon: Number of months ahead, at least 1.
            seed: Seed of the NumPy generator; the same seed gives the same draws.

        Raises:
            ValueError: If ``horizon`` is below 1.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        months, draws = self.n_months, self.latent.shape[0]
        columns = 2 + 2 * self.harmonics
        rng = np.random.default_rng(seed)
        sigma = np.exp(self.hyper[:, [0]])
        alpha = 1.0 / (np.exp(2.0 * self.hyper[:, [1]]) + _DISPERSION_FLOOR)
        fixed, drift = self.latent[:, :columns], self.latent[:, columns:]
        future = np.arange(months, months + horizon, dtype=np.float64)
        rows = _design_rows(future, months, self.period, self.harmonics)
        steps = sigma * rng.standard_normal((draws, horizon))
        drift_future = drift[:, [-1]] + np.cumsum(steps, axis=1)
        all_rows = _design(months, self.period, self.harmonics)
        centred = np.vstack([all_rows, rows])
        eta = fixed @ centred.T + np.hstack([drift, drift_future])
        eta = np.clip(eta, -_ETA_LIMIT, _ETA_LIMIT)
        trend = fixed[:, [0]] + fixed[:, [1]] * centred[:, 1] + np.hstack([drift, drift_future])
        mean = np.exp(eta)
        counts = rng.negative_binomial(alpha, alpha / (alpha + mean)).astype(np.float64)
        return Forecast(trend, mean, counts, months)

    def p_increasing(self) -> float:
        """Return the posterior probability that the slope is above 0 (TS-3)."""
        return float(np.mean(self.slope > 0.0))


def _design_rows(
    t: npt.NDArray[np.float64], months: int, period: int, harmonics: int
) -> npt.NDArray[np.float64]:
    """Return the fixed-effects design at month indices ``t``; the slope is centred on the fit."""
    columns = [np.ones(t.size), (t - (months - 1) / 2) / period]
    for k in range(1, harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        columns += [np.cos(angle), np.sin(angle)]
    return np.column_stack(columns)


def _design(months: int, period: int, harmonics: int) -> npt.NDArray[np.float64]:
    return _design_rows(np.arange(months, dtype=np.float64), months, period, harmonics)


def _check(
    diagnostics: tuple[float, float, float, float, float, float, float], n_draws: int
) -> FitCheck:
    khat, ess, edge, unconverged, not_pd, draw_failures, _grid = diagnostics
    limit = khat_limit(n_draws)
    reasons: list[str] = []
    if unconverged > 0:
        reasons.append(f"Newton did not converge at {int(unconverged)} grid points")
    if not_pd > 0:
        reasons.append(f"Hessian not positive definite at {int(not_pd)} grid points")
    if draw_failures > 0:
        reasons.append(f"{int(draw_failures)} draws had no converged Gaussian approximation")
    if edge > EDGE_MASS_MAX:
        reasons.append(f"grid-edge mass {edge:.3f} above {EDGE_MASS_MAX}")
    if not khat <= limit:  # also true for NaN
        reasons.append(f"k-hat {khat:.2f} above {limit:.2f}")
    if not ess >= ESS_MIN:
        reasons.append(f"importance sampling ESS {ess:.0f} below {ESS_MIN:.0f}")
    return FitCheck(
        khat, limit, ess, ESS_MIN, edge, EDGE_MASS_MAX,
        int(unconverged), int(not_pd), int(draw_failures), n_draws, tuple(reasons),
    )  # fmt: skip


def fit_trend(  # noqa: PLR0913 (model choices, all keyword-only)
    counts: npt.ArrayLike,
    *,
    period: int = 12,
    harmonics: int = 2,
    n_draws: int = 2000,
    seed: int = 0,
    drift_scale: float = 0.2,
    dispersion_scale: float = 1.0,
) -> TrendFit:
    """Fit a negative binomial structural series and return the checked posterior.

    The log mean is a level, a slope, a Fourier season and a random-walk drift. The intercept
    prior is centred on the log of the mean count; its sd is 2, and the sd of the slope and of
    each Fourier coefficient is 1. The drift sd and the dispersion ``phi`` (size = 1 / (phi^2 +
    0.001)) have half-normal priors.

    Args:
        counts: Non-negative counts, one per month, at least two seasons long.
        period: Season length in months.
        harmonics: Number of Fourier harmonics, at most ``period // 2``.
        n_draws: Number of posterior draws, at least 400.
        seed: Seed of the random number generator; the same seed gives the same draws.
        drift_scale: Scale of the half-normal prior on the drift sd.
        dispersion_scale: Scale of the half-normal prior on ``phi``.

    Returns:
        The fit, whose ``check`` passed.

    Raises:
        FitCheckError: If the fit check fails. It carries the numbers and the reasons.
        ValueError: If an argument is out of range.
    """
    y = np.ascontiguousarray(counts, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("counts must be a 1-D array of finite, non-negative numbers")
    if period < 2 or not 1 <= harmonics <= period // 2 or y.size < 2 * period:  # noqa: PLR2004
        raise ValueError("need period >= 2, 1 <= harmonics <= period // 2, and 2 seasons of data")
    if n_draws < MIN_DRAWS:
        raise ValueError(f"n_draws must be at least {MIN_DRAWS}, got {n_draws}")
    design = np.ascontiguousarray(_design(y.size, period, harmonics))
    columns = design.shape[1]
    prior_mean = np.zeros(columns)
    prior_mean[0] = math.log(float(y.mean()) + 0.1)
    prior_prec = np.ones(columns)
    prior_prec[0] = 0.25
    latent, hyper, diagnostics = _accel.fit_structural(
        y, design, prior_mean, prior_prec, drift_scale, dispersion_scale, n_draws, seed
    )
    check = _check(diagnostics, n_draws)
    check.require()
    return TrendFit(latent, hyper, check, period, harmonics)
