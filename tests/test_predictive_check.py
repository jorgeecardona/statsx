"""AS-1: the posterior predictive check finds what the model cannot hold, and says what it cannot see."""

import numpy as np
import numpy.typing as npt
import pytest

from statsx.trend import ModelCheck, TrendFit, fit_trend

MONTHS = 96
T = np.arange(MONTHS)
SEASON = 0.5 * np.cos(2 * np.pi * T / 12) + 0.3 * np.sin(2 * np.pi * T / 12)


def _negative_binomial(log_mean: npt.NDArray[np.float64], seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    mean = np.exp(log_mean)
    return rng.negative_binomial(40, 40 / (40 + mean)).astype(np.float64)


def _well_specified(seed: int) -> npt.NDArray[np.float64]:
    return _negative_binomial(5.0 + 0.004 * T + SEASON, 1000 + seed)


def _fit(counts: npt.NDArray[np.float64], seed: int = 0) -> TrendFit:
    return fit_trend(counts, require=False, seed=seed)


def _p(check: ModelCheck, name: str) -> float:
    return next(d.p_value for d in check.discrepancies if d.name == name)


def test_well_specified_series_are_rarely_flagged() -> None:
    flagged = [
        _fit(_well_specified(seed), seed).predictive_check().contradicted for seed in range(12)
    ]
    assert sum(flagged) <= 2


def test_every_check_runs_on_a_plain_series() -> None:
    check = _fit(_well_specified(0)).predictive_check()
    assert [d.name for d in check.discrepancies] == ["tails", "zeros", "phase", "autocorrelation"]
    assert all(0.0 <= d.p_value <= 1.0 and d.what for d in check.discrepancies)


def test_an_outlier_is_flagged_as_tails() -> None:
    counts = _well_specified(1).copy()
    counts[40] = counts[40] * 3
    check = _fit(counts).predictive_check()
    assert "tails" in check.flagged
    assert check.contradicted


def test_a_season_the_harmonics_cannot_hold_is_flagged_as_phase() -> None:
    spike = np.where(T % 12 == 11, 1.0, 0.0) + np.where(T % 12 == 0, 0.5, 0.0)
    counts = _negative_binomial(5.0 + 0.004 * T + 0.2 * SEASON + spike, 2)
    check = _fit(counts).predictive_check()
    assert "phase" in check.flagged


def test_more_harmonics_hold_the_shape() -> None:
    spike = np.where(T % 12 == 11, 1.0, 0.0) + np.where(T % 12 == 0, 0.5, 0.0)
    counts = _negative_binomial(5.0 + 0.004 * T + 0.2 * SEASON + spike, 2)
    few = fit_trend(counts, harmonics=2, require=False, seed=0).predictive_check()
    many = fit_trend(counts, harmonics=6, require=False, seed=0).predictive_check()
    assert _p(many, "phase") > _p(few, "phase")


def test_a_surplus_of_zeros_is_flagged() -> None:
    rng = np.random.default_rng(3)
    counts = _negative_binomial(3.0 + 0.5 * SEASON, 3)
    counts = np.where(rng.random(MONTHS) < 0.2, 0.0, counts)
    check = _fit(counts).predictive_check()
    assert "zeros" in check.flagged


def test_counts_that_alternate_are_flagged_as_autocorrelation() -> None:
    alternate = np.where(T % 2 == 0, 0.25, -0.25)
    check = _fit(_negative_binomial(5.0 + 0.004 * T + SEASON + alternate, 4)).predictive_check()
    assert "autocorrelation" in check.flagged


def test_a_pulse_in_the_level_is_not_seen() -> None:
    """The drift follows a pulse, so the check passes. The docs name this limit."""
    pulse = np.where((T >= 40) & (T < 48), 0.7, 0.0)
    check = _fit(_negative_binomial(5.0 + 0.004 * T + SEASON + pulse, 5)).predictive_check()
    assert not check.contradicted


def test_the_check_is_reproducible_and_depends_on_the_seed() -> None:
    fit = _fit(_well_specified(2))
    assert fit.predictive_check(seed=1) == fit.predictive_check(seed=1)
    assert fit.predictive_check(seed=1) != fit.predictive_check(seed=2)


def test_months_that_were_not_observed_are_skipped() -> None:
    counts = _well_specified(3).copy()
    counts[[10, 11, 50]] = np.nan
    check = _fit(counts).predictive_check()
    assert [d.name for d in check.discrepancies] == ["tails", "zeros", "phase", "autocorrelation"]
    assert not np.isnan([d.observed for d in check.discrepancies]).any()


def test_a_poisson_fit_is_checked_too() -> None:
    rng = np.random.default_rng(6)
    counts = rng.poisson(np.exp(5.0 + SEASON)).astype(np.float64)
    fit = fit_trend(counts, likelihood="poisson", require=False, seed=0)
    assert len(fit.predictive_check().discrepancies) == 4


def test_a_series_under_two_cycles_drops_the_phase_check() -> None:
    counts = _well_specified(4)
    fit = fit_trend(counts[:30], harmonics=0, period=12, require=False, seed=0)
    assert "phase" in [d.name for d in fit.predictive_check().discrepancies]
    fit = fit_trend(counts[:20], harmonics=0, period=12, require=False, seed=0)
    assert "phase" not in [d.name for d in fit.predictive_check().discrepancies]


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_the_level_must_be_inside_zero_and_one(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        _fit(_well_specified(0)).predictive_check(level)


def test_a_looser_level_flags_at_least_as_much() -> None:
    fit = _fit(_well_specified(5))
    assert len(fit.predictive_check(0.5).flagged) >= len(fit.predictive_check(0.01).flagged)
