import numpy as np
import pytest

from statsx.calibration import Reliability, reliability


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
