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
        {"counts": [1.0, float("nan")] * 30},
        {"counts": [1.0] * 10},
        {"counts": [1.0] * 60, "n_draws": 100},
        {"counts": [1.0] * 60, "harmonics": 0},
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
