from itertools import pairwise

import numpy as np
import pytest

from statsx.calibration import Reliability, fit_isotonic, fit_platt, reliability


def _forecasts(seed: int, n: int = 5000, tilt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """True probabilities spread over (0, 1); the forecast is the logit scaled by ``tilt``."""
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2.0, 2.0, n)
    y = (rng.random(n) < true_p).astype(float)
    logit = np.log(true_p / (1.0 - true_p)) * tilt
    return y, 1.0 / (1.0 + np.exp(-logit))


def test_the_reliability_numbers_are_the_bin_means_and_the_ece_is_their_weighted_gap() -> None:
    y = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 1.0])
    p = np.array([0.05, 0.1, 0.15, 0.3, 0.35, 0.55, 0.65, 0.85, 0.9, 0.95])
    result = reliability(y, p, n_bins=5)
    assert isinstance(result, Reliability)
    # Bins [0, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1]: counts 3, 2, 1, 1, 3.
    assert list(result.counts) == [3, 2, 1, 1, 3]
    np.testing.assert_allclose(result.mean_predicted, [0.1, 0.325, 0.55, 0.65, 0.9])
    np.testing.assert_allclose(result.observed_rate, [1 / 3, 0.5, 1.0, 1.0, 2 / 3])
    gaps = np.abs(result.mean_predicted - result.observed_rate)
    assert result.ece == pytest.approx(float(np.sum(result.counts * gaps) / 10))
    assert result.mce == pytest.approx(float(gaps.max()))
    assert result.n == 10


def test_the_brier_score_equals_its_decomposition_exactly() -> None:
    for tilt in (1.0, 0.5, 2.0):
        y, p = _forecasts(1, tilt=tilt)
        for strategy in ("uniform", "quantile"):
            b = reliability(y, p, n_bins=10, strategy=strategy).brier  # type: ignore[arg-type]
            total = b.reliability - b.resolution + b.uncertainty + b.within_bin
            assert b.brier == pytest.approx(total, abs=1e-12)
            assert b.brier == pytest.approx(float(np.mean((p - y) ** 2)), abs=1e-15)


def test_with_one_value_per_bin_the_within_bin_terms_vanish() -> None:
    rng = np.random.default_rng(2)
    p = rng.choice([0.05, 0.25, 0.45, 0.65, 0.85], 2000)
    y = (rng.random(2000) < p).astype(float)
    b = reliability(y, p, n_bins=5).brier
    assert b.within_bin == pytest.approx(0.0, abs=1e-12)
    assert b.brier == pytest.approx(b.reliability - b.resolution + b.uncertainty, abs=1e-12)


def test_a_calibrated_classifier_has_a_small_ece_and_an_overconfident_one_a_large_one() -> None:
    y, p = _forecasts(3, tilt=1.0)
    y_over, p_over = _forecasts(3, tilt=3.0)
    good, over = reliability(y, p), reliability(y_over, p_over)
    assert good.ece < 0.03
    assert over.ece > 0.1
    assert over.brier.reliability > 5 * good.brier.reliability
    # Overconfident: the high bins are predicted above the rate, the low bins below it.
    assert over.mean_predicted[-1] > over.observed_rate[-1]
    assert over.mean_predicted[0] < over.observed_rate[0]


def test_quantile_bins_hold_about_the_same_number_of_items() -> None:
    y, p = _forecasts(4)
    result = reliability(y, p, n_bins=10, strategy="quantile")
    assert result.counts.sum() == 5000
    assert result.counts.max() - result.counts.min() <= 5
    uniform = reliability(y, p, n_bins=10, strategy="uniform")
    assert uniform.counts.max() > 2 * uniform.counts.min()


def test_a_perfect_and_a_constant_forecast() -> None:
    y = np.array([0, 1, 1, 0, 1, 0.0])
    perfect = reliability(y, y)
    assert perfect.ece == 0.0 and perfect.brier.brier == 0.0
    constant = reliability(y, np.full(6, 0.5), n_bins=4)
    assert list(constant.counts) == [6]
    assert constant.ece == pytest.approx(0.0, abs=1e-12)
    assert constant.brier.resolution == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("y", "p", "kwargs", "message"),
    [
        ([0, 1], [0.5], {}, "same length"),
        ([], [], {}, "same length"),
        ([0, 2], [0.5, 0.5], {}, "0 and 1"),
        ([0, 1], [0.5, 1.5], {}, "probabilities"),
        ([0, 1], [0.5, 0.5], {"n_bins": 0}, "n_bins"),
        ([0, 1], [0.5, 0.5], {"strategy": "random"}, "strategy"),
    ],
)
def test_bad_input_is_refused(
    y: list[float], p: list[float], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reliability(y, p, **kwargs)  # type: ignore[arg-type]


def _scores(rng: np.random.Generator, n: int, tilt: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Logit scores that are ``tilt`` times too confident, and their outcomes."""
    true_p = rng.beta(2.0, 2.0, n)
    y = (rng.random(n) < true_p).astype(float)
    return np.log(true_p / (1.0 - true_p)) * tilt, y


def test_platt_scaling_and_isotonic_regression_repair_an_overconfident_classifier() -> None:
    rng = np.random.default_rng(0)
    train_f, train_y = _scores(rng, 5000)
    test_f, test_y = _scores(rng, 20000)
    raw = reliability(test_y, 1.0 / (1.0 + np.exp(-test_f)))
    platt = fit_platt(train_f, train_y)
    isotonic = fit_isotonic(train_f, train_y)
    for fixed in (platt.predict(test_f), isotonic.predict(test_f)):
        after = reliability(test_y, fixed)
        assert after.ece < 0.03
        assert raw.ece > 0.1
        assert after.brier.brier < raw.brier.brier - 0.02
    # The logit was 3 times too large, so the fitted slope is about 1 / 3 and the intercept 0.
    assert platt.a == pytest.approx(1.0 / 3.0, abs=0.02)
    assert platt.b == pytest.approx(0.0, abs=0.05)


def test_the_isotonic_map_never_decreases_and_stays_a_probability() -> None:
    rng = np.random.default_rng(1)
    f, y = _scores(rng, 400)
    f = np.round(f, 1)  # repeated scores are pooled
    calibrator = fit_isotonic(f, y)
    assert np.all(np.diff(calibrator.thresholds) > 0)
    assert np.all(np.diff(calibrator.values) >= -1e-15)
    assert calibrator.values.min() >= 0.0 and calibrator.values.max() <= 1.0
    grid: np.ndarray = np.linspace(float(f.min()) - 5.0, float(f.max()) + 5.0, 500)
    assert np.all(np.diff(calibrator.predict(grid)) >= -1e-15)
    # Outside the training scores the map stays at its end values.
    assert calibrator.predict([f.min() - 100.0])[0] == calibrator.values[0]
    assert calibrator.predict([f.max() + 100.0])[0] == calibrator.values[-1]


def test_isotonic_regression_is_the_best_non_decreasing_fit() -> None:
    # The fit has the smallest squared error among all non-decreasing step fits (every way to cut
    # the sorted items into blocks with non-decreasing means).
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = 8
        f = np.sort(rng.normal(size=n))
        y = (rng.random(n) < 0.5).astype(float)
        if y.min() == y.max():
            continue
        fitted = fit_isotonic(f, y).values
        best = np.inf
        for cuts in range(2 ** (n - 1)):
            blocks: list[tuple[int, int]] = []
            start = 0
            for i in range(n - 1):
                if cuts >> i & 1:
                    blocks.append((start, i + 1))
                    start = i + 1
            blocks.append((start, n))
            means = [float(y[a:b].mean()) for a, b in blocks]
            if all(m1 <= m2 for m1, m2 in pairwise(means)):
                best = min(
                    best,
                    sum(
                        float(np.sum((y[a:b] - m) ** 2))
                        for (a, b), m in zip(blocks, means, strict=True)
                    ),
                )
        assert float(np.sum((y - fitted) ** 2)) == pytest.approx(best, abs=1e-9)


def test_platt_scaling_is_better_than_isotonic_regression_with_few_items() -> None:
    rng = np.random.default_rng(3)
    test_f, test_y = _scores(rng, 20000)
    platt_brier: list[float] = []
    isotonic_brier: list[float] = []
    for _ in range(40):
        f, y = _scores(rng, 40)
        if y.min() == y.max():
            continue
        platt_brier.append(reliability(test_y, fit_platt(f, y).predict(test_f)).brier.brier)
        isotonic_brier.append(reliability(test_y, fit_isotonic(f, y).predict(test_f)).brier.brier)
    assert np.mean(platt_brier) < np.mean(isotonic_brier)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit_platt([1.0, 2.0], [0.0]),
        lambda: fit_platt([1.0, float("inf")], [0.0, 1.0]),
        lambda: fit_platt([1.0, 2.0], [0.0, 0.0]),
        lambda: fit_isotonic([], []),
        lambda: fit_isotonic([1.0, 2.0], [0.0, 2.0]),
        lambda: fit_isotonic([1.0, 2.0], [1.0, 1.0]),
    ],
)
def test_calibrators_refuse_bad_input(call: object) -> None:
    with pytest.raises(ValueError, match=r"same length|finite|one class"):
        call()  # type: ignore[operator]
