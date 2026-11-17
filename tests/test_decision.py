import math
from statistics import NormalDist

import numpy as np
import pytest

from statsx.conjugate import Beta, Gamma, Normal
from statsx.decision import (
    break_even,
    break_even_probability,
    expected_value_of_information,
    simulate_cost,
)
from statsx.prior import lognormal_from_interval


def _model(fixed: np.ndarray, hours: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return fixed + hours * rate


def _identity(rate: np.ndarray) -> np.ndarray:
    return rate


def _per_hour(hours: np.ndarray) -> np.ndarray:
    return 60.0 * hours


def _scalar(rate: np.ndarray) -> float:
    return float(rate.mean())


def _log(rate: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.log(rate)


def _nothing() -> np.ndarray:
    return np.zeros(10)


def test_the_cost_distribution_has_the_exact_mean_and_a_fixed_seed() -> None:
    inputs = {"fixed": 100.0, "hours": Gamma(20.0, 2.0), "rate": Normal(50.0, 5.0)}
    result = simulate_cost(_model, inputs, n_draws=200_000, seed=7)
    # E[cost] = 100 + E[hours] * E[rate] for independent inputs.
    exact = 100.0 + (20.0 / 2.0) * 50.0
    mean = result.mean()
    assert mean.estimate == pytest.approx(
        exact, abs=4 * (mean.interval.upper - mean.interval.lower) / 4
    )
    assert mean.interval.contains(exact)
    assert result.inputs == ("hours", "rate")
    assert result.n_draws == 200_000
    again = simulate_cost(_model, inputs, n_draws=200_000, seed=7)
    assert np.array_equal(result.draws, again.draws)
    assert not np.array_equal(
        result.draws, simulate_cost(_model, inputs, n_draws=200_000, seed=8).draws
    )


def test_the_distribution_gives_quantiles_an_interval_and_a_tail_probability() -> None:
    # Cost = rate alone, a Normal(10, 2): the quantiles are the Normal quantiles.
    result = simulate_cost(_identity, {"rate": Normal(10.0, 2.0)}, n_draws=400_000, seed=1)
    quantiles = result.quantiles((0.05, 0.5, 0.95))
    assert quantiles[0.5] == pytest.approx(10.0, abs=0.02)
    assert quantiles[0.95] == pytest.approx(10.0 + 1.6449 * 2.0, abs=0.03)
    interval = result.interval(0.9)
    assert interval.lower == pytest.approx(quantiles[0.05])
    assert result.p_exceeds(12.0) == pytest.approx(1.0 - NormalDist(10.0, 2.0).cdf(12.0), abs=0.003)


def test_a_lognormal_input_from_an_expert_interval_can_drive_the_cost() -> None:
    hours = lognormal_from_interval(10.0, 100.0)
    result = simulate_cost(_per_hour, {"hours": hours}, n_draws=100_000, seed=3)
    interval = result.interval(0.9)
    assert interval.lower == pytest.approx(600.0, rel=0.03)
    assert interval.upper == pytest.approx(6000.0, rel=0.03)


def test_the_model_must_return_one_finite_cost_per_draw() -> None:
    with pytest.raises(ValueError, match="finite costs"):
        simulate_cost(_scalar, {"rate": Normal(1.0, 1.0)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite costs"):
        simulate_cost(_log, {"rate": Normal(0.0, 1.0)}, n_draws=1000)
    with pytest.raises(ValueError, match="at least one input"):
        simulate_cost(_nothing, {})
    with pytest.raises(ValueError, match="n_draws"):
        simulate_cost(_identity, {"rate": Normal(1.0, 1.0)}, n_draws=1)


def test_the_value_of_perfect_information_matches_the_closed_form() -> None:
    # Action 0 pays theta ~ Normal(mu, s), action 1 pays 0: EVPI = E[max(theta, 0)] - max(mu, 0).
    mu, s = 0.3, 1.0
    theta = np.random.default_rng(0).normal(mu, s, 1_000_000)
    result = expected_value_of_information(np.column_stack([theta, np.zeros_like(theta)]))
    z = mu / s
    exact = s * NormalDist().pdf(z) + mu * NormalDist().cdf(z) - max(mu, 0.0)
    assert result.evpi == pytest.approx(exact, abs=0.003)
    assert result.best_action == 0
    assert result.expected_utility[0] == pytest.approx(mu, abs=0.003)


def test_information_is_worth_nothing_when_one_action_always_wins() -> None:
    theta = np.abs(np.random.default_rng(1).normal(2.0, 0.1, 1000))
    result = expected_value_of_information(np.column_stack([theta, np.zeros_like(theta)]))
    assert result.evpi == pytest.approx(0.0, abs=1e-12)
    assert result.n_draws == 1000


def test_the_break_even_probability_is_the_loss_ratio() -> None:
    assert break_even_probability(gain_if_right=3.0, loss_if_wrong=1.0) == pytest.approx(0.25)
    assert break_even_probability(1.0, 1.0) == 0.5
    # At the break-even probability the expected utility of acting is zero.
    gain, loss = 7.0, 2.0
    p = break_even_probability(gain, loss)
    assert p * gain - (1.0 - p) * loss == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="> 0"):
        break_even_probability(0.0, 1.0)


def test_break_even_finds_the_threshold_where_the_net_benefit_changes_sign() -> None:
    assert break_even(lambda x: x * x - 2.0, 0.0, 2.0) == pytest.approx(math.sqrt(2.0), abs=1e-9)
    # Net benefit of shipping when the true rate has a Beta posterior and the bar is the threshold.
    posterior = Beta(30.0, 70.0).sample(200_000, seed=2)
    threshold = break_even(
        lambda bar: float(np.mean(posterior > bar)) - 0.5, 0.0, 1.0, tolerance=1e-6
    )
    assert threshold == pytest.approx(float(np.median(posterior)), abs=1e-4)
    assert break_even(lambda x: x, 0.0, 1.0) == 0.0
    with pytest.raises(ValueError, match="no sign change"):
        break_even(lambda x: x + 5.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="low < high"):
        break_even(lambda x: x, 1.0, 1.0)


def test_value_of_information_input_is_checked() -> None:
    with pytest.raises(ValueError, match="shape"):
        expected_value_of_information(np.ones(10))
    with pytest.raises(ValueError, match="shape"):
        expected_value_of_information(np.ones((10, 1)))
