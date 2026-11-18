import math

import numpy as np
import pytest

from statsx.backtest import backtest
from statsx.impact import causal_impact
from statsx.trend import fit_trend, supported_harmonics


def _counts(months: int, seed: int = 7, season: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(months, dtype=float)
    log_mean = 4.0 + 0.012 * t + season * np.sin(2 * np.pi * t / 12.0)
    return rng.poisson(np.exp(log_mean)).astype(float), log_mean


def _drifting(months: int, seed: int, season: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Counts whose log mean has a seasonal term and a small random-walk drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(months, dtype=float)
    log_mean = (
        4.0 + np.cumsum(rng.normal(0.0, 0.04, months)) + season * np.sin(2 * np.pi * t / 12.0)
    )
    return rng.poisson(np.exp(log_mean)).astype(float), log_mean


@pytest.mark.parametrize(
    ("n", "period", "expected"),
    [
        (11, 12, 0),
        (12, 12, 2),
        (16, 12, 3),
        (18, 12, 3),
        (23, 12, 4),
        (24, 12, 6),
        (60, 12, 6),
        (8, 4, 2),
        (7, 4, 0),
    ],
)
def test_the_supported_harmonics_follow_one_parameter_per_two_observations(
    n: int, period: int, expected: int
) -> None:
    assert supported_harmonics(n, period) == expected


@pytest.mark.parametrize(("months", "harmonics"), [(12, 1), (12, 2), (18, 2), (18, 3), (23, 3)])
def test_a_series_of_one_to_two_seasons_is_fitted_with_the_harmonics_it_supports(
    months: int, harmonics: int
) -> None:
    counts, log_mean = _counts(months)
    fit = fit_trend(counts, period=12, harmonics=harmonics, seed=42)
    median = np.median(fit.forecast(1, seed=0).mean[:, :months], axis=0)
    assert fit.n_months == fit.n_observed == months
    # The fitted mean follows the true mean on the log scale.
    assert float(np.sqrt(np.mean((np.log(median) - log_mean) ** 2))) < 0.25


def test_the_error_names_the_condition_that_failed_and_what_would_work() -> None:
    counts, _ = _counts(18)
    with pytest.raises(ValueError, match=r"18 observed months support at most 3 harmonics.*not 5"):
        fit_trend(counts, period=12, harmonics=5)
    with pytest.raises(ValueError, match=r"harmonics must be from 0 to period // 2 = 6, got 7"):
        fit_trend(_counts(60)[0], period=12, harmonics=7)
    with pytest.raises(ValueError, match=r"period must be at least 2"):
        fit_trend(_counts(60)[0], period=1, harmonics=0)
    with pytest.raises(ValueError, match=r"10 observed months support at most 0.*harmonics=0"):
        fit_trend(_counts(10)[0], period=12, harmonics=2)
    with pytest.raises(ValueError, match=r"need at least 4 observed months, got 3"):
        fit_trend([4.0, 5.0, 6.0], harmonics=0)


def test_a_fit_without_a_season_works_on_a_short_series_and_forecasts() -> None:
    counts, log_mean = _counts(10, season=0.0)
    fit = fit_trend(counts, harmonics=0, seed=1)
    assert fit.latent.shape[1] == 2 + 10
    median = np.median(fit.forecast(1, seed=0).mean[:, :10], axis=0)
    assert float(np.sqrt(np.mean((np.log(median) - log_mean) ** 2))) < 0.25
    assert fit.forecast(3).counts.shape[1] == 13


def test_a_nan_month_is_not_observed_and_infinities_and_negatives_are_refused_separately() -> None:
    counts, _ = _drifting(36, seed=3)
    counts[10] = float("nan")
    fit = fit_trend(counts, period=12, harmonics=2, seed=3)
    assert (fit.n_months, fit.n_observed) == (36, 35)
    assert np.isnan(fit.counts[10])
    assert fit.anomalies().direction[10] == 0
    bad_inf = counts.copy()
    bad_inf[3] = float("inf")
    with pytest.raises(ValueError, match="finite, or NaN"):
        fit_trend(bad_inf)
    bad_neg = counts.copy()
    bad_neg[3] = -1.0
    with pytest.raises(ValueError, match="must not be negative"):
        fit_trend(bad_neg)


def test_an_index_is_the_same_as_nan_in_the_dense_series() -> None:
    counts, _ = _drifting(36, seed=4)
    dense = counts.copy()
    dense[[10, 11, 20]] = np.nan
    keep = np.flatnonzero(~np.isnan(dense))
    from_index = fit_trend(dense[keep], index=keep + 100, seed=5)  # the offset does not matter
    from_nan = fit_trend(dense, seed=5)
    np.testing.assert_array_equal(from_index.latent, from_nan.latent)
    assert from_index.n_months == 36


def test_a_missing_month_keeps_the_season_in_phase_where_dropping_it_does_not() -> None:
    months = 48
    rng = np.random.default_rng(11)
    t = np.arange(months, dtype=float)
    truth = 4.5 + 0.8 * np.sin(2 * np.pi * t / 12.0)  # a strong season, no drift
    counts = rng.poisson(np.exp(truth)).astype(float)
    hole = 8
    with_nan = counts.copy()
    with_nan[hole] = np.nan
    good = fit_trend(with_nan, period=12, harmonics=1, seed=0)
    dropped = fit_trend(np.delete(counts, hole), period=12, harmonics=1, seed=0)
    keep = np.arange(months) != hole
    good_error = (
        np.log(np.median(good.forecast(1, seed=0).mean[:, :months], axis=0))[keep] - truth[keep]
    )
    # The dropped series puts every later month one place early: compare at the true positions.
    shifted = (
        np.insert(
            np.log(np.median(dropped.forecast(1, seed=0).mean[:, : months - 1], axis=0)),
            hole,
            np.nan,
        )[keep]
        - truth[keep]
    )
    assert float(np.sqrt(np.nanmean(good_error**2))) < 0.2
    assert float(np.sqrt(np.nanmean(shifted**2))) > 2.0 * float(np.sqrt(np.mean(good_error**2)))


@pytest.mark.parametrize("seed", [5, 6, 8])
def test_imputing_a_missing_month_narrows_the_band_by_pretending_to_know_it(seed: int) -> None:
    counts, _ = _drifting(36, seed=seed)
    counts[15] = float("nan")
    honest = fit_trend(counts, period=12, harmonics=2, seed=2)
    imputed_value = np.round(np.median(honest.forecast(1, seed=0).mean[:, 15]))
    filled = counts.copy()
    filled[15] = imputed_value
    pretend = fit_trend(filled, period=12, harmonics=2, seed=2)
    honest_band, pretend_band = honest.band(0.9), pretend.band(0.9)
    honest_width = float(honest_band.upper[15] - honest_band.lower[15])
    pretend_width = float(pretend_band.upper[15] - pretend_band.lower[15])
    assert honest_width > 1.05 * pretend_width
    assert math.isfinite(honest_width)


def test_regressors_follow_the_dense_axis_when_an_index_is_given() -> None:
    counts, _ = _drifting(24, seed=6)
    keep = np.array([i for i in range(24) if i != 5])
    rng = np.random.default_rng(1)
    regressor = rng.normal(size=24)
    fit = fit_trend(counts[keep], index=keep, regressors=regressor, seed=0)
    assert fit.n_months == 24 and fit.regressors.shape == (24, 1)
    with pytest.raises(ValueError, match="rows"):
        fit_trend(counts[keep], index=keep, regressors=regressor[keep])


@pytest.mark.parametrize(
    "index",
    [[0, 1, 2], [3, 2, 1, 0], [0, 1, 1, 3], [0.5, 1.0, 2.0, 3.0]],
)
def test_a_bad_index_is_refused(index: list[float]) -> None:
    with pytest.raises(ValueError, match="index"):
        fit_trend([4.0, 5.0, 6.0, 7.0], index=index, harmonics=0)


def test_impact_accepts_a_gap_before_the_intervention_and_backtest_refuses_gaps() -> None:
    counts, _ = _drifting(48, seed=4)
    controls = np.exp(_drifting(48, seed=9)[1])
    gap = counts.copy()
    gap[10] = float("nan")
    result = causal_impact(gap, controls, 36, harmonics=2, seed=1)
    assert result.effect.shape[1] == 12
    late = counts.copy()
    late[40] = float("nan")
    with pytest.raises(ValueError, match="after the intervention must all be observed"):
        causal_impact(late, controls, 36)
    with pytest.raises(ValueError, match="backtest needs every month observed"):
        backtest(gap, horizon=3, initial=30, n_draws=400)
