"""Rolling-origin backtest of a count forecast: CRPS and interval coverage.

``backtest`` refits ``fit_trend`` at each origin, forecasts the next months, and scores the
forecast draws against the counts that followed. An origin whose fit fails its check is
counted as refused and reported, not dropped. ``backtest_all`` runs every series it is given
and reports all of them: a forecast claim that rests on a chosen subset of series is not a
claim about the method.

References:
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
    estimation. *Journal of the American Statistical Association* 102(477):359-378.
    https://doi.org/10.1198/016214506000001437.

    Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review.
    *International Journal of Forecasting* 16(4):437-450.
    https://doi.org/10.1016/S0169-2070(00)00065-0.

    Roque, L., Cerqueira, V., Soares, C., & Torgo, L. (2025). Cherry-picking in time series
    forecasting: how to select datasets to make your model shine. *Proceedings of the AAAI
    Conference on Artificial Intelligence* 39(19):20192-20199.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx.trend import FitCheckError, Likelihood, fit_trend

DEFAULT_LEVELS = (0.5, 0.8, 0.95)


def crps_from_draws(draws: npt.ArrayLike, observed: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the CRPS of a forecast given as draws, for each observed value.

    CRPS = E|X - y| - E|X - X'| / 2, estimated from the draws; smaller is better, and it is in
    the units of the counts. The second term uses the sorted draws, so the cost is m log m.

    Args:
        draws: Forecast draws, shape (draws, steps).
        observed: The observed values, shape (steps,).

    Raises:
        ValueError: If the shapes do not match or there are fewer than 2 draws.
    """
    x = np.asarray(draws, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[1],) or x.shape[0] < 2:  # noqa: PLR2004
        raise ValueError("draws must be (draws >= 2, steps) and observed must be (steps,)")
    m = x.shape[0]
    first = np.mean(np.abs(x - y), axis=0)
    ranked = np.sort(x, axis=0)
    ranks: npt.NDArray[np.float64] = np.arange(1, m + 1, dtype=np.float64)
    weights: npt.NDArray[np.float64] = (2.0 * ranks - m - 1.0)[:, None]
    second: npt.NDArray[np.float64] = np.sum(weights * ranked, axis=0) / m**2
    return first - second


@dataclass(frozen=True, slots=True)
class OriginScore:
    """The score of the forecast made at one origin.

    ``crps`` and ``covered`` are empty when the fit was refused; ``reasons`` then says why.

    Args:
        origin: Index of the first forecast month; the fit used the months before it.
        reasons: The reasons the fit check failed, empty when it passed.
        crps: CRPS of each forecast month, shape (horizon,).
        covered: Whether the observed count was inside each band, shape (levels, horizon).
    """

    origin: int
    reasons: tuple[str, ...]
    crps: npt.NDArray[np.float64]
    covered: npt.NDArray[np.bool_]

    @property
    def refused(self) -> bool:
        """Return True when the fit failed its check and nothing was scored."""
        return bool(self.reasons)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """All origins of one series.

    Args:
        horizon: Number of months forecast at each origin.
        levels: Levels of the bands whose coverage is counted.
        scores: One score per origin, refused ones included.
    """

    horizon: int
    levels: tuple[float, ...]
    scores: tuple[OriginScore, ...]

    @property
    def n_origins(self) -> int:
        """Return the number of origins, refused ones included."""
        return len(self.scores)

    @property
    def n_refused(self) -> int:
        """Return the number of origins whose fit failed its check."""
        return sum(score.refused for score in self.scores)

    def _scored(self) -> list[OriginScore]:
        return [s for s in self.scores if not s.refused]

    @property
    def crps_by_step(self) -> npt.NDArray[np.float64]:
        """Return the mean CRPS at each step ahead, shape (horizon,); NaN if nothing scored."""
        scored = self._scored()
        if not scored:
            return np.full(self.horizon, np.nan)
        return np.mean([s.crps for s in scored], axis=0)

    @property
    def mean_crps(self) -> float:
        """Return the mean CRPS over all scored origins and steps; NaN if nothing scored."""
        return float(np.mean(self.crps_by_step))

    @property
    def coverage(self) -> npt.NDArray[np.float64]:
        """Return the share of observed counts inside each band, shape (levels,)."""
        scored = self._scored()
        if not scored:
            return np.full(len(self.levels), np.nan)
        return np.mean([s.covered.mean(axis=1) for s in scored], axis=0)


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """The backtest of every series given, by name. No series is left out."""

    results: Mapping[str, BacktestResult]

    @property
    def n_refused(self) -> int:
        """Return the number of refused origins over all series."""
        return sum(r.n_refused for r in self.results.values())

    @property
    def n_origins(self) -> int:
        """Return the number of origins over all series."""
        return sum(r.n_origins for r in self.results.values())

    @property
    def mean_crps(self) -> float:
        """Return the mean of the per-series mean CRPS, each series counting once."""
        return float(np.nanmean([r.mean_crps for r in self.results.values()]))


def backtest(  # noqa: PLR0913 (model choices, all keyword-only)
    counts: npt.ArrayLike,
    *,
    horizon: int,
    initial: int,
    step: int = 1,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    regressors: npt.ArrayLike | None = None,
    period: int = 12,
    harmonics: int = 2,
    likelihood: Likelihood = "negative_binomial",
    n_draws: int = 1000,
    seed: int = 0,
) -> BacktestResult:
    """Backtest ``fit_trend`` on one series with rolling origins.

    The origins are ``initial``, ``initial + step``, and so on while ``horizon`` months remain.
    At each one the model is fitted on the months before it and forecasts the next ``horizon``.

    Args:
        counts: Non-negative counts, one per month.
        horizon: Months forecast at each origin, at least 1.
        initial: First origin: the months fitted at the first origin. At least ``2 * period``.
        step: Months between origins, at least 1.
        levels: Increasing levels in (0, 1) whose coverage is counted.
        regressors: Optional regressors for every month, shape (months, regressors); the
            forecast months use their own rows.
        period: Season length in months.
        harmonics: Number of Fourier harmonics.
        likelihood: ``"negative_binomial"`` or ``"poisson"``.
        n_draws: Draws per fit, at least 400.
        seed: Seed; origin ``k`` uses ``seed + k``.

    Raises:
        ValueError: If no origin fits in the series, or an argument is out of range.
    """
    y = np.ascontiguousarray(counts, dtype=np.float64)
    if y.ndim != 1 or horizon < 1 or step < 1 or initial < 2 * period:
        raise ValueError("need 1-D counts, horizon >= 1, step >= 1 and initial >= 2 * period")
    if np.any(np.isnan(y)):
        raise ValueError(
            "backtest needs every month observed: a NaN would make the scores NaN. "
            "Use fit_trend for a series with months that were not observed"
        )
    origins = list(range(initial, y.size - horizon + 1, step))
    if not origins:
        raise ValueError(f"no origin fits: {y.size} months, initial {initial}, horizon {horizon}")
    x = None if regressors is None else np.asarray(regressors, dtype=np.float64)
    if x is not None and x.shape[0] != y.size:
        raise ValueError(f"regressors have {x.shape[0]} rows for {y.size} months")
    scores: list[OriginScore] = []
    for k, origin in enumerate(origins):
        future = y[origin : origin + horizon]
        try:
            fit = fit_trend(
                y[:origin],
                regressors=None if x is None else x[:origin],
                period=period,
                harmonics=harmonics,
                likelihood=likelihood,
                n_draws=n_draws,
                seed=seed + k,
            )
        except FitCheckError as error:
            empty = np.empty((0,), dtype=np.float64)
            scores.append(
                OriginScore(origin, error.check.reasons, empty, np.empty((0, 0), np.bool_))
            )
            continue
        forecast = fit.forecast(
            horizon,
            regressors=None if x is None else x[origin : origin + horizon],
            seed=seed + k,
        )
        draws = forecast.counts[:, -horizon:]
        covered = np.vstack(
            [
                (future >= band.lower[-horizon:]) & (future <= band.upper[-horizon:])
                for band in forecast.bands(levels)
            ]
        )
        scores.append(OriginScore(origin, (), crps_from_draws(draws, future), covered))
    return BacktestResult(horizon, tuple(levels), tuple(scores))


def backtest_all(
    series: Mapping[str, npt.ArrayLike],
    *,
    horizon: int,
    initial: int,
    **options: object,
) -> BacktestReport:
    """Backtest every series in ``series`` and return all of them, in the order given.

    Series that cannot be backtested raise ``ValueError``: none is skipped without a message.
    ``options`` are the keyword arguments of ``backtest``.
    """
    results = {
        name: backtest(counts, horizon=horizon, initial=initial, **options)  # type: ignore[arg-type]
        for name, counts in series.items()
    }
    return BacktestReport(results)
