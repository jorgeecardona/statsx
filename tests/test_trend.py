import math

import numpy as np
import pytest

from statsx import _accel
from statsx.trend import ESS_MIN, FitCheckError, fit_trend, khat_limit

MONTHS = 60


def _series(
    seed: int, level: float, slope: float, alpha: float, season: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return counts, the true log trend, and the true seasonal term."""
    rng = np.random.default_rng(seed)
    t = np.arange(MONTHS)
    centred = (t - (MONTHS - 1) / 2) / 12
    trend = math.log(level) + slope * centred + np.cumsum(rng.normal(0.0, 0.04, MONTHS))
    seasonal = season * np.cos(2 * np.pi * t / 12) + 0.4 * season * np.sin(4 * np.pi * t / 12)
    mu = np.exp(trend + seasonal)
    counts = rng.negative_binomial(alpha, alpha / (alpha + mu))
    return counts, trend, seasonal


def test_trend_follows_the_true_trend_without_the_season() -> None:
    counts, trend, seasonal = _series(seed=1, level=30.0, slope=0.3, alpha=20.0, season=0.8)
    fit = fit_trend(counts, seed=0)
    median = np.median(fit.trend, axis=0)
    # The error is small, and the 90% band covers most of the true trend.
    assert np.sqrt(np.mean((median - trend) ** 2)) < 0.2
    band = fit.band(0.9)
    assert np.mean((trend >= band.lower) & (trend <= band.upper)) > 0.8
    # The raw log counts follow the season; the error of the fitted trend does not.
    raw = np.log(counts + 0.5) - trend
    assert np.corrcoef(raw, seasonal)[0, 1] > 0.7
    assert abs(np.corrcoef(median - trend, seasonal)[0, 1]) < 0.3


def test_trend_works_on_low_counts() -> None:
    counts, trend, _ = _series(seed=4, level=1.0, slope=0.2, alpha=5.0)
    assert np.mean(counts == 0) > 0.2
    fit = fit_trend(counts, seed=0)
    assert np.sqrt(np.mean((np.median(fit.trend, axis=0) - trend) ** 2)) < 0.5


def test_probability_of_increase_follows_the_slope() -> None:
    up, _, _ = _series(seed=2, level=30.0, slope=0.8, alpha=20.0)
    down, _, _ = _series(seed=3, level=30.0, slope=-0.8, alpha=20.0)
    assert fit_trend(up, seed=0).p_increasing() > 0.95
    assert fit_trend(down, seed=0).p_increasing() < 0.05


def test_a_failed_fit_check_gives_an_error_with_reasons_and_no_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    real = _accel.fit_structural

    def bad_khat(*args: object) -> object:
        latent, hyper, diag = real(*args)  # type: ignore[arg-type]
        return latent, hyper, (0.95, *diag[1:])

    monkeypatch.setattr(_accel, "fit_structural", bad_khat)
    with pytest.raises(FitCheckError) as caught:
        fit_trend(counts, seed=0)
    check = caught.value.check
    assert not check.passed
    assert any("k-hat 0.95" in reason for reason in check.reasons)
    assert "k-hat" in str(caught.value)


@pytest.mark.parametrize(
    ("diagnostics", "reason"),
    [
        ((0.3, 900.0, 0.004, 2.0, 0.0, 0.0, 441.0), "Newton did not converge at 2"),
        ((0.3, 900.0, 0.004, 0.0, 1.0, 0.0, 441.0), "not positive definite at 1"),
        ((0.3, 900.0, 0.004, 0.0, 0.0, 3.0, 441.0), "3 draws had no converged"),
        ((0.3, 900.0, 0.2, 0.0, 0.0, 0.0, 441.0), "grid-edge mass 0.200"),
        ((float("nan"), 900.0, 0.004, 0.0, 0.0, 0.0, 441.0), "k-hat nan"),
        ((0.3, 120.0, 0.004, 0.0, 0.0, 0.0, 441.0), "ESS 120 below 400"),
    ],
)
def test_each_check_names_its_reason(
    monkeypatch: pytest.MonkeyPatch, diagnostics: tuple[float, ...], reason: str
) -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    real = _accel.fit_structural

    def patched(*args: object) -> object:
        latent, hyper, _ = real(*args)  # type: ignore[arg-type]
        return latent, hyper, diagnostics

    monkeypatch.setattr(_accel, "fit_structural", patched)
    with pytest.raises(FitCheckError) as caught:
        fit_trend(counts, seed=0)
    assert any(reason in r for r in caught.value.check.reasons), caught.value.check.reasons


def test_a_passing_fit_records_its_check() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    fit = fit_trend(counts, n_draws=1000, seed=0)
    assert fit.check.passed
    assert fit.check.n_draws == 1000
    assert fit.check.ess >= ESS_MIN
    assert fit.check.khat <= fit.check.khat_limit
    assert fit.latent.shape == (1000, 2 + 4 + MONTHS)
    assert fit.trend.shape == (1000, MONTHS)


def test_same_seed_gives_the_same_draws() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    a = fit_trend(counts, seed=5)
    b = fit_trend(counts, seed=5)
    c = fit_trend(counts, seed=6)
    assert np.array_equal(a.latent, b.latent)
    assert not np.array_equal(a.latent, c.latent)


def test_khat_limit_follows_the_draw_count() -> None:
    assert khat_limit(800) == pytest.approx(1 - 1 / math.log10(800))
    assert khat_limit(100_000) == 0.7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"counts": [1.0, -1.0] * 30},
        {"counts": [1.0, float("inf")] * 30},
        {"counts": [1.0] * 10},
        {"counts": [1.0] * 60, "n_draws": 100},
        {"counts": [1.0] * 60, "harmonics": 7},
        {"counts": [[1.0] * 60] * 2},
    ],
)
def test_bad_input_is_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"counts|n_draws|period"):
        fit_trend(**kwargs)  # type: ignore[arg-type]


def test_band_level_is_validated() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    with pytest.raises(ValueError, match="level"):
        fit_trend(counts, seed=0).band(1.5)


def test_forecast_bands_are_nested_and_widen_with_the_horizon() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    forecast = fit_trend(counts, seed=0).forecast(24, seed=0)
    assert forecast.horizon == 24
    assert forecast.counts.shape[1] == MONTHS + 24
    inner, middle, outer = forecast.bands((0.5, 0.8, 0.95))
    assert (inner.level, middle.level, outer.level) == (0.5, 0.8, 0.95)
    # Nested at every past and future month.
    assert np.all(outer.lower <= middle.lower) and np.all(middle.lower <= inner.lower)
    assert np.all(inner.upper <= middle.upper) and np.all(middle.upper <= outer.upper)
    # The 80% band of the trend is wider the further ahead it is.
    trend_band = forecast.bands((0.8,), of="trend")[0]
    width = trend_band.upper - trend_band.lower
    ahead = [MONTHS, MONTHS + 5, MONTHS + 11, MONTHS + 23]
    assert width[ahead[0]] < width[ahead[1]] < width[ahead[2]] < width[ahead[3]]
    # So does the 80% band of a new count.
    count_band = forecast.bands((0.8,))[0]
    count_width = count_band.upper - count_band.lower
    assert count_width[MONTHS + 23] > count_width[MONTHS]


def test_forecast_bands_cover_held_out_months() -> None:
    held = 12
    covered: dict[float, list[float]] = {0.5: [], 0.8: [], 0.95: []}
    for seed in range(15):
        full, _, _ = _series(seed=100 + seed, level=30.0, slope=0.2, alpha=5.0)
        # The generator has no months beyond MONTHS, so hold out the last ones.
        fit = fit_trend(full[:-held], seed=seed)
        bands = fit.forecast(held, seed=seed).bands((0.5, 0.8, 0.95))
        for band in bands:
            inside = (full[-held:] >= band.lower[-held:]) & (full[-held:] <= band.upper[-held:])
            covered[band.level].append(float(inside.mean()))
    rate = {level: float(np.mean(v)) for level, v in covered.items()}
    assert 0.35 < rate[0.5] < 0.7, rate
    assert 0.7 < rate[0.8] < 0.93, rate
    assert rate[0.95] > 0.88, rate


def test_forecast_repeats_the_season_and_draws_counts() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.0, alpha=20.0, season=0.8)
    forecast = fit_trend(counts, seed=0).forecast(12, seed=0)
    median = np.median(forecast.mean, axis=0)
    assert np.corrcoef(median[-12:], median[-24:-12])[0, 1] > 0.8
    assert np.all(forecast.counts >= 0)
    assert np.all(forecast.counts == np.round(forecast.counts))


def test_forecast_is_reproducible_and_validated() -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    fit = fit_trend(counts, seed=0)
    a, b, c = fit.forecast(6, seed=3), fit.forecast(6, seed=3), fit.forecast(6, seed=4)
    assert np.array_equal(a.counts, b.counts)
    assert not np.array_equal(a.counts, c.counts)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    with pytest.raises(ValueError, match="levels"):
        a.bands((0.8, 0.5))
    with pytest.raises(ValueError, match="levels"):
        a.bands(())


def test_months_outside_the_expected_band_are_flagged_as_anomalies() -> None:
    counts, _, _ = _series(seed=6, level=30.0, slope=0.2, alpha=20.0, season=0.6)
    spiked = counts.copy()
    spiked[30] = counts[30] * 6 + 40
    spiked[45] = 0
    report = fit_trend(spiked, seed=0).anomalies(0.95)
    assert report.level == 0.95
    assert report.direction[30] == 1
    assert report.direction[45] == -1
    assert {30, 45} <= set(report.months)
    assert np.all(report.lower <= report.upper)


def test_clean_series_are_rarely_flagged() -> None:
    flagged: list[float] = []
    for seed in range(8):
        counts, _, _ = _series(seed=200 + seed, level=30.0, slope=0.2, alpha=20.0)
        report = fit_trend(counts, seed=seed).anomalies(0.95)
        flagged.append(len(report.months) / MONTHS)
    # The nominal rate is 5%; the draws also include the posterior uncertainty, so it is lower.
    assert np.mean(flagged) < 0.1


def _with_regressor(seed: int, effect: float) -> tuple[np.ndarray, np.ndarray]:
    """Counts whose log mean has ``effect`` per unit of a regressor with sd 1.5."""
    rng = np.random.default_rng(seed)
    t = np.arange(MONTHS + 12)
    x = rng.normal(0.0, 1.5, MONTHS + 12)
    mu = np.exp(math.log(30.0) + effect * x + 0.4 * np.cos(2 * np.pi * t / 12))
    return rng.negative_binomial(20.0, 20.0 / (20.0 + mu)), x


def test_regressor_terms_are_learned_and_needed_for_the_forecast() -> None:
    counts, x = _with_regressor(seed=1, effect=0.4)
    fit = fit_trend(counts[:MONTHS], regressors=x[:MONTHS], seed=0)
    coefficient = fit.regressor_coefficients[:, 0]
    assert fit.n_regressors == 1
    low, high = np.quantile(coefficient, [0.05, 0.95])
    assert low < 0.4 < high
    assert low > 0.0
    forecast = fit.forecast(12, regressors=x[MONTHS:], seed=0)
    assert forecast.counts.shape == (fit.latent.shape[0], MONTHS + 12)
    with pytest.raises(ValueError, match="pass them"):
        fit.forecast(12)
    with pytest.raises(ValueError, match="shape"):
        fit.forecast(12, regressors=x[MONTHS:-1])


def test_forecast_follows_the_future_regressor() -> None:
    counts, x = _with_regressor(seed=2, effect=0.5)
    fit = fit_trend(counts[:MONTHS], regressors=x[:MONTHS], seed=0)
    high = fit.forecast(6, regressors=np.full(6, 3.0), seed=0)
    low = fit.forecast(6, regressors=np.full(6, -3.0), seed=0)
    assert np.median(high.mean[:, -6:]) > 3 * np.median(low.mean[:, -6:])


def test_poisson_on_overdispersed_counts_puts_the_noise_in_the_trend() -> None:
    counts, trend, _ = _series(seed=5, level=30.0, slope=0.2, alpha=3.0)
    poisson = fit_trend(counts, likelihood="poisson", seed=0)
    negbin = fit_trend(counts, likelihood="negative_binomial", seed=0)
    assert poisson.likelihood == "poisson"
    assert np.all(np.isnan(poisson.hyper[:, 1]))
    forecast = poisson.forecast(6, seed=0)
    assert np.all(forecast.counts == np.round(forecast.counts))
    error = {
        name: float(np.sqrt(np.mean((np.median(fit.trend, axis=0) - trend) ** 2)))
        for name, fit in (("poisson", poisson), ("negbin", negbin))
    }
    # Poisson has no dispersion, so the random-walk drift absorbs the extra variance.
    assert error["poisson"] > 2 * error["negbin"], error
    assert np.median(np.exp(poisson.hyper[:, 0])) > 5 * np.median(np.exp(negbin.hyper[:, 0]))


def test_poisson_recovers_the_trend_of_poisson_counts() -> None:
    rng = np.random.default_rng(9)
    t = np.arange(MONTHS)
    trend = math.log(20.0) + 0.3 * (t - 29.5) / 12
    counts = rng.poisson(np.exp(trend + 0.5 * np.cos(2 * np.pi * t / 12)))
    fit = fit_trend(counts, likelihood="poisson", seed=0)
    assert np.sqrt(np.mean((np.median(fit.trend, axis=0) - trend) ** 2)) < 0.15


def test_the_season_period_and_harmonics_are_chosen_by_the_caller() -> None:
    rng = np.random.default_rng(3)
    months = 48
    t = np.arange(months)
    trend = math.log(25.0) + 0.1 * (t - 23.5) / 4
    counts = rng.poisson(np.exp(trend + 0.6 * np.cos(2 * np.pi * t / 4)))
    fit = fit_trend(counts, period=4, harmonics=1, likelihood="poisson", seed=0)
    assert (fit.period, fit.harmonics) == (4, 1)
    assert fit.latent.shape[1] == 2 + 2 * 1 + months
    assert np.sqrt(np.mean((np.median(fit.trend, axis=0) - trend) ** 2)) < 0.2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"likelihood": "gamma"}, "likelihood"),
        ({"regressors": np.ones(MONTHS)}, "constant"),
        ({"regressors": np.arange(MONTHS - 1.0)}, "rows"),
        ({"regressors": np.full((MONTHS, 1), np.nan)}, "finite"),
    ],
)
def test_bad_model_choices_are_refused(kwargs: dict[str, object], message: str) -> None:
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    with pytest.raises(ValueError, match=message):
        fit_trend(counts, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", range(8))
def test_exactly_poisson_counts_are_fitted_not_refused(seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(36, dtype=float)
    counts = rng.poisson(np.exp(4.0 + 0.012 * t + 0.25 * np.sin(2 * np.pi * t / 12.0)))
    fit = fit_trend(counts, period=12, harmonics=3, n_draws=2000, seed=42)
    assert fit.check.passed


def test_fit_check_reason_names_a_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    real = _accel.fit_structural

    def patched(*args: object) -> tuple[object, object, tuple[float, ...]]:
        latent, hyper, diag = real(*args)  # type: ignore[arg-type]
        return latent, hyper, (diag[0], diag[1], 0.5, *diag[3:])

    monkeypatch.setattr(_accel, "fit_structural", patched)
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    with pytest.raises(FitCheckError, match=r"dispersion_scale"):
        fit_trend(counts, seed=0)


def test_one_except_catches_both_kinds_of_cannot_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from statsx.errors import StatsxError, UnfittableInput

    short = np.arange(12) + 5  # 12 months support 2 harmonics, not 3
    with pytest.raises(StatsxError) as shape:
        fit_trend(short, period=12, harmonics=3)
    assert isinstance(shape.value, UnfittableInput)
    assert isinstance(shape.value, ValueError)  # old handlers keep working
    assert not isinstance(shape.value, FitCheckError)

    real = _accel.fit_structural

    def failing(*args: object) -> tuple[object, object, tuple[float, ...]]:
        latent, hyper, diag = real(*args)  # type: ignore[arg-type]
        return latent, hyper, (np.inf, *diag[1:])

    monkeypatch.setattr(_accel, "fit_structural", failing)
    counts, _, _ = _series(seed=1, level=30.0, slope=0.3, alpha=20.0)
    with pytest.raises(StatsxError) as check:
        fit_trend(counts, seed=0)
    assert isinstance(check.value, FitCheckError)
    assert isinstance(check.value, RuntimeError)
    assert not isinstance(check.value, UnfittableInput)
