import math

import numpy as np
import pytest

from statsx.alwaysvalid import ConfidenceSequence, EProcess, confidence_sequence, e_process

RUNS, STEPS = 4000, 400


def _paths(seed: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, (RUNS, STEPS))


def test_the_confidence_sequence_holds_the_true_mean_at_every_step_with_probability_one_minus_alpha() -> (
    None
):
    x = _paths(0, mean=0.3)
    ever_wrong = 0
    for path in x:
        cs = confidence_sequence(path, sigma=1.0, alpha=0.05)
        ever_wrong += bool(np.any(cs.excludes(0.3)))
    assert ever_wrong / RUNS <= 0.05
    # It is conservative, since the bound is not tight for a normal mean.
    assert ever_wrong / RUNS < 0.03


def test_looking_after_every_observation_with_a_usual_interval_is_far_above_five_percent() -> None:
    x = _paths(1)
    n = np.arange(1, STEPS + 1)
    z = np.cumsum(x, axis=1) / np.sqrt(n)
    naive_wrong = float(np.mean(np.any(np.abs(z) > 1.96, axis=1)))
    cs_wrong = float(np.mean([np.any(confidence_sequence(p, sigma=1.0).excludes(0.0)) for p in x]))
    assert naive_wrong > 0.4
    assert cs_wrong <= 0.05


def test_the_e_value_rejects_a_true_claim_at_most_alpha_of_the_time_whenever_you_stop() -> None:
    x = _paths(2)
    rejected = sum(
        e_process(p, 0.0, sigma=1.0, alpha=0.05).first_rejection() is not None for p in x
    )
    assert rejected / RUNS <= 0.05


def test_the_expected_e_value_under_the_claim_is_one_by_integration() -> None:
    # The e-value is heavy-tailed, so simulation underestimates its mean; integrate over the sum.
    n = 100
    z = np.linspace(-40.0, 40.0, 8001)
    e = np.array(
        [e_process(np.r_[v * math.sqrt(n), np.zeros(n - 1)], 0.0, sigma=1.0).e_value[-1] for v in z]
    )
    density = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    assert np.trapezoid(e * density, z) == pytest.approx(1.0, abs=1e-6)


def test_a_false_claim_is_rejected_early_and_the_interval_excludes_it_at_the_same_step() -> None:
    path = np.random.default_rng(3).normal(0.5, 1.0, 600)
    e = e_process(path, 0.0, sigma=1.0, alpha=0.05)
    cs = confidence_sequence(path, sigma=1.0, alpha=0.05)
    assert isinstance(e, EProcess) and isinstance(cs, ConfidenceSequence)
    first = e.first_rejection()
    assert first is not None and first < 150
    assert cs.first_exclusion(0.0) == first
    # The two always agree: mu0 is outside the interval exactly when its e-value reaches 1 / alpha.
    for mu0 in (-0.2, 0.0, 0.3, 0.5, 0.9):
        np.testing.assert_array_equal(
            e_process(path, mu0, sigma=1.0).rejected(),
            confidence_sequence(path, sigma=1.0).excludes(mu0),
        )


def test_the_interval_matches_the_closed_form_and_shrinks_with_more_data() -> None:
    path = np.random.default_rng(4).normal(0.0, 1.0, 1000)
    cs = confidence_sequence(path, sigma=1.0, alpha=0.05, n_target=100.0)
    rho = 100.0 / (2 * math.log(20.0) + math.log(1 + 2 * math.log(20.0)))
    n = 100
    radius = math.sqrt((rho + n) * math.log((rho + n) / (rho * 0.05**2))) / n
    assert cs.upper[n - 1] - cs.mean[n - 1] == pytest.approx(radius, rel=1e-12)
    assert cs.mean[n - 1] - cs.lower[n - 1] == pytest.approx(radius, rel=1e-12)
    widths = cs.upper - cs.lower
    assert widths[999] < widths[99] < widths[9]
    # Wider than the fixed-sample interval 2 * 1.96 / sqrt(100) at n = 100, the price of validity.
    assert widths[99] > 2 * 1.96 / 10


def test_bounded_outcomes_work_with_sigma_one_half() -> None:
    rng = np.random.default_rng(5)
    wrong = 0
    for _ in range(2000):
        outcomes = (rng.random(300) < 0.3).astype(float)
        wrong += bool(np.any(confidence_sequence(outcomes, sigma=0.5).excludes(0.3)))
    assert wrong / 2000 <= 0.05


@pytest.mark.parametrize(
    "call",
    [
        lambda: confidence_sequence([], sigma=1.0),
        lambda: confidence_sequence([1.0, float("nan")], sigma=1.0),
        lambda: confidence_sequence([1.0], sigma=0.0),
        lambda: confidence_sequence([1.0], sigma=1.0, alpha=1.0),
        lambda: confidence_sequence([1.0], sigma=1.0, n_target=0.0),
        lambda: e_process([1.0], float("inf"), sigma=1.0),
        lambda: e_process([1.0], 0.0, sigma=-1.0),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"observations|sigma|alpha|n_target|mu0"):
        call()  # type: ignore[operator]
