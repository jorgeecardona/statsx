import math

import numpy as np
import pytest

from statsx import backtest as backtest_module
from statsx.backtest import backtest, backtest_all, crps_from_draws
from statsx.trend import FitCheck, FitCheckError

MONTHS = 96


def _counts(seed: int, level: float = 30.0, season: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(MONTHS)
    trend = math.log(level) + 0.2 * (t - 47.5) / 12 + np.cumsum(rng.normal(0.0, 0.03, MONTHS))
    mu = np.exp(trend + season * np.cos(2 * np.pi * t / 12))
    return rng.negative_binomial(15.0, 15.0 / (15.0 + mu)).astype(np.float64)


def test_crps_matches_the_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    draws = rng.normal(5.0, 2.0, size=(60, 3))
    observed = np.array([4.0, 9.0, 5.5])
    brute: list[float] = [
        np.mean(np.abs(draws[:, j] - observed[j]))
        - 0.5 * np.mean(np.abs(draws[:, j][:, None] - draws[:, j][None, :]))
        for j in range(3)
    ]
    assert crps_from_draws(draws, observed) == pytest.approx(brute, rel=1e-12)


def test_crps_matches_the_closed_form_for_a_normal_forecast() -> None:
    # CRPS of N(0, 1) at y = 0 is (sqrt(2) - 1) / sqrt(pi) (Gneiting & Raftery 2007).
    draws = np.random.default_rng(1).normal(size=(20000, 1))
    got = crps_from_draws(draws, np.array([0.0]))[0]
    assert got == pytest.approx((math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), abs=0.01)


def test_backtest_scores_every_origin_with_crps_and_coverage() -> None:
    counts = _counts(seed=1)
    result = backtest(counts, horizon=6, initial=48, step=6, seed=0)
    origins = [s.origin for s in result.scores]
    assert origins == list(range(48, MONTHS - 6 + 1, 6))
    assert result.n_origins == len(origins)
    assert result.n_refused == 0
    assert result.crps_by_step.shape == (6,)
    assert result.coverage.shape == (3,)
    # The score gets worse the further ahead the forecast, and the bands are about nominal.
    assert result.crps_by_step[-1] > result.crps_by_step[0]
    assert 0.35 < result.coverage[0] < 0.8
    assert result.coverage[2] > 0.85


def test_backtest_shows_a_better_forecast_has_a_lower_crps() -> None:
    counts = _counts(seed=2, season=0.8)
    result = backtest(counts, horizon=6, initial=48, step=6, seed=0)
    # Baseline: the empirical distribution of all earlier counts, ignoring trend and season.
    baseline: list[float] = []
    for score in result.scores:
        past, future = counts[: score.origin], counts[score.origin : score.origin + 6]
        draws = np.repeat(past[:, None], 6, axis=1)
        baseline.append(float(crps_from_draws(draws, future).mean()))
    assert result.mean_crps < 0.8 * float(np.mean(baseline))


def test_every_series_is_reported_and_refused_origins_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = backtest_module.fit_trend
    calls = {"n": 0}

    def sometimes_refuse(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] > 4:  # every origin of the second series
            check = FitCheck(
                0.9, 0.7, 900.0, 400.0, 0.0, 0.02, 0, 0, 0, 1000, ("k-hat 0.90 above 0.70",)
            )
            raise FitCheckError(check)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backtest_module, "fit_trend", sometimes_refuse)
    series = {"a": _counts(seed=3), "b": _counts(seed=4)}
    report = backtest_all(series, horizon=12, initial=48, step=12)
    assert list(report.results) == ["a", "b"]
    assert report.results["a"].n_refused == 0
    assert report.results["b"].n_refused == report.results["b"].n_origins == 4
    assert report.results["b"].scores[0].reasons == ("k-hat 0.90 above 0.70",)
    assert math.isnan(report.results["b"].mean_crps)
    assert report.n_refused == 4
    assert report.n_origins == 8
    assert report.mean_crps == pytest.approx(report.results["a"].mean_crps)


def test_regressors_are_sliced_for_each_origin() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.5, MONTHS)
    t = np.arange(MONTHS)
    mu = np.exp(math.log(30.0) + 0.4 * x + 0.4 * np.cos(2 * np.pi * t / 12))
    counts = rng.negative_binomial(20.0, 20.0 / (20.0 + mu)).astype(np.float64)
    with_x = backtest(counts, horizon=6, initial=60, step=12, regressors=x, seed=0)
    without = backtest(counts, horizon=6, initial=60, step=12, seed=0)
    assert with_x.mean_crps < without.mean_crps


@pytest.mark.parametrize(
    ("horizon", "initial", "step"), [(0, 48, 1), (6, 10, 1), (6, 48, 0), (60, 48, 1)]
)
def test_bad_arguments_are_refused(horizon: int, initial: int, step: int) -> None:
    with pytest.raises(ValueError, match=r"origin|horizon"):
        backtest(_counts(seed=1), horizon=horizon, initial=initial, step=step)


def test_crps_needs_matching_shapes() -> None:
    with pytest.raises(ValueError, match="draws"):
        crps_from_draws(np.ones((10, 3)), np.ones(2))
