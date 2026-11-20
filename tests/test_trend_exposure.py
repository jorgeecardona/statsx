"""GitLab issue #15: ``exposure=`` enters the log mean as an offset with its coefficient fixed at 1."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx.trend import fit_trend

MONTHS = 72
T = np.arange(MONTHS)
LOG_RATE = -3.0 + 0.01 * T + 0.3 * np.sin(2 * np.pi * T / 12)


def _exposure() -> npt.NDArray[np.float64]:
    """Exposure that grows tenfold over the series, faster than the rate changes."""
    return 2000.0 * np.exp(np.linspace(0.0, math.log(10.0), MONTHS))


def _counts(exposure: npt.NDArray[np.float64], seed: int = 0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    mean = exposure * np.exp(LOG_RATE)
    return rng.negative_binomial(50, 50 / (50 + mean)).astype(np.float64)


def test_the_trend_is_the_log_rate_when_exposure_changes() -> None:
    exposure = _exposure()
    counts = _counts(exposure)
    fit = fit_trend(counts, exposure=exposure, seed=0)
    truth = -3.0 + 0.01 * T
    error = fit.trend.mean(axis=0) - truth
    assert np.sqrt(np.mean(error**2)) < 0.1
    slope_per_year = np.quantile(fit.slope, [0.05, 0.95])
    assert slope_per_year[0] < 0.12 < slope_per_year[1]


def test_without_exposure_the_slope_is_the_growth_of_the_counts() -> None:
    exposure = _exposure()
    fit = fit_trend(_counts(exposure), seed=0)
    assert np.quantile(fit.slope, 0.05) > 0.3  # log(10) / 6 years + 0.12 is about 0.5


def test_a_constant_exposure_only_moves_the_level() -> None:
    counts = _counts(np.full(MONTHS, 5000.0))
    plain = fit_trend(counts, seed=0)
    scaled = fit_trend(counts, exposure=np.full(MONTHS, 5000.0), seed=0)
    shift = plain.trend.mean(axis=0) - scaled.trend.mean(axis=0)
    assert np.allclose(shift, np.log(5000.0), atol=0.05)
    assert np.allclose(plain.slope.mean(), scaled.slope.mean(), atol=0.02)


def test_the_forecast_needs_the_future_exposure_and_scales_with_it() -> None:
    exposure = _exposure()
    fit = fit_trend(_counts(exposure), exposure=exposure, seed=0)
    with pytest.raises(ValueError, match="exposure"):
        fit.forecast(6)
    one = fit.forecast(6, exposure=np.full(6, 1000.0), seed=3)
    two = fit.forecast(6, exposure=np.full(6, 2000.0), seed=3)
    assert np.allclose(two.mean[:, MONTHS:], 2.0 * one.mean[:, MONTHS:])
    assert np.allclose(two.mean[:, :MONTHS], one.mean[:, :MONTHS])
    assert np.array_equal(two.trend, one.trend)


def test_a_rate_forecast_is_the_mean_at_exposure_one() -> None:
    exposure = _exposure()
    fit = fit_trend(_counts(exposure), exposure=exposure, seed=0)
    rate = fit.forecast(1, exposure=np.ones(1), seed=0).mean[:, MONTHS]
    assert 0.5 < np.median(rate) / np.exp(LOG_RATE[-1] + 0.01) < 2.0


def test_a_model_without_exposure_refuses_one_in_the_forecast() -> None:
    fit = fit_trend(_counts(np.full(MONTHS, 3000.0)), seed=0)
    with pytest.raises(ValueError, match="no exposure"):
        fit.forecast(3, exposure=np.ones(3))


@pytest.mark.parametrize(
    "bad",
    [np.zeros(MONTHS), -np.ones(MONTHS), np.full(MONTHS, np.nan), np.ones(MONTHS - 1)],
)
def test_exposure_must_be_positive_finite_and_one_per_month(bad: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="exposure"):
        fit_trend(_counts(np.full(MONTHS, 3000.0)), exposure=bad)


def test_exposure_works_with_regressors_and_the_free_coefficient_checks_proportionality() -> None:
    exposure = _exposure()
    counts = _counts(exposure, seed=1)
    fit = fit_trend(counts, regressors=np.log(exposure), harmonics=1, seed=0, require=False)
    low, high = np.quantile(fit.regressor_coefficients[:, 0], [0.025, 0.975])
    assert low < 1.0 < high
    both = fit_trend(
        counts, exposure=exposure, regressors=np.cos(2 * np.pi * T / 6), seed=0, require=False
    )
    assert both.forecast(2, exposure=exposure[-2:], regressors=np.zeros(2)).mean.shape[1] == 74


def test_anomalies_and_the_model_check_use_the_fitted_exposure() -> None:
    exposure = _exposure()
    counts = _counts(exposure, seed=2)
    fit = fit_trend(counts, exposure=exposure, seed=0)
    assert len(fit.anomalies().months) <= 8
    assert not fit.predictive_check().contradicted
    flat = fit_trend(counts, seed=0)
    assert np.all(np.isfinite(flat.anomalies().lower))


def test_a_poisson_fit_takes_exposure_too() -> None:
    exposure = _exposure()
    rng = np.random.default_rng(4)
    counts = rng.poisson(exposure * np.exp(LOG_RATE)).astype(np.float64)
    fit = fit_trend(counts, exposure=exposure, likelihood="poisson", seed=0)
    assert abs(float(np.median(fit.slope)) - 0.12) < 0.1
