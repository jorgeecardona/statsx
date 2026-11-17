import math
from statistics import NormalDist

import numpy as np
import pytest

from statsx import _accel
from statsx.conjugate import Beta, Gamma, Normal
from statsx.posterior import Comparison, compare
from statsx.prior import LogNormal, lognormal_from_interval


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _p_beta_greater(a1: int, b1: float, a2: float, b2: float) -> float:
    """Exact P(X > Y) for X ~ Beta(a1, b1), Y ~ Beta(a2, b2), integer a1 (Cook 2005)."""
    total = 0.0
    for i in range(a1):
        log_term = (
            _log_beta(a2 + i, b1 + b2) - math.log(b1 + i) - _log_beta(1 + i, b1) - _log_beta(a2, b2)
        )
        total += math.exp(log_term)
    return total


def test_p_a_greater_matches_the_exact_value_for_two_betas() -> None:
    a, b = Beta(81.0, 21.0), Beta(60.0, 42.0)
    result = compare(a, b, margin=0.02, n_draws=400_000, seed=1)
    exact = _p_beta_greater(81, 21.0, 60.0, 42.0)
    assert isinstance(result, Comparison)
    assert result.p_a_greater == pytest.approx(exact, abs=4 * result.mc_error + 1e-4)
    assert result.p_a_greater + result.p_b_greater == pytest.approx(1.0, abs=1e-9)
    assert result.n_draws == 400_000


def test_two_normals_match_the_exact_probabilities() -> None:
    a, b = Normal(1.0, 0.5), Normal(0.6, 0.3)
    result = compare(a, b, margin=0.2, n_draws=400_000, seed=2)
    mean, sd = 0.4, math.hypot(0.5, 0.3)
    exact_greater = 1.0 - NormalDist(mean, sd).cdf(0.0)
    exact_equivalent = NormalDist(mean, sd).cdf(0.2) - NormalDist(mean, sd).cdf(-0.2)
    assert result.p_a_greater == pytest.approx(exact_greater, abs=4 * result.mc_error)
    assert result.p_equivalent == pytest.approx(exact_equivalent, abs=4 * result.mc_error)
    assert result.difference.contains(mean)
    assert result.difference.lower == pytest.approx(mean - 1.96 * sd, abs=0.01)


def test_two_gammas_with_one_rate_follow_the_beta_relation() -> None:
    # A / (A + B) ~ Beta(a, b) when the rates are equal, so P(A > B) = 1 - I_0.5(a, b).
    result = compare(Gamma(12.0, 2.0), Gamma(9.0, 2.0), margin=0.5, n_draws=400_000, seed=3)
    exact = 1.0 - _accel.beta_cdf(np.array([0.5]), 12.0, 9.0)[0]
    assert result.p_a_greater == pytest.approx(exact, abs=4 * result.mc_error)


def test_draws_can_be_passed_instead_of_distributions() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(1.0, 0.5, 50_000)
    b = rng.normal(0.6, 0.3, 80_000)
    result = compare(a, b, margin=0.2)
    assert result.n_draws == 50_000
    exact = 1.0 - NormalDist(0.4, math.hypot(0.5, 0.3)).cdf(0.0)
    assert result.p_a_greater == pytest.approx(exact, abs=0.01)


def test_a_wider_margin_gives_a_higher_probability_of_equivalence() -> None:
    a, b = Beta(30.0, 70.0), Beta(33.0, 67.0)
    narrow = compare(a, b, margin=0.01, seed=5).p_equivalent
    middle = compare(a, b, margin=0.05, seed=5).p_equivalent
    wide = compare(a, b, margin=0.5, seed=5).p_equivalent
    assert narrow < middle < wide
    assert wide == 1.0


def test_identical_posteriors_are_a_coin_flip() -> None:
    result = compare(Beta(20.0, 30.0), Beta(20.0, 30.0), margin=0.05, n_draws=200_000, seed=6)
    assert result.p_a_greater == pytest.approx(0.5, abs=4 * result.mc_error)


def test_the_same_seed_gives_the_same_comparison() -> None:
    a, b = Beta(5.0, 7.0), Beta(6.0, 6.0)
    assert compare(a, b, margin=0.1, seed=3) == compare(a, b, margin=0.1, seed=3)
    assert compare(a, b, margin=0.1, seed=3) != compare(a, b, margin=0.1, seed=4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"margin": 0.0}, "margin"),
        ({"margin": 0.1, "level": 1.0}, "level"),
        ({"margin": 0.1, "n_draws": 1}, "n_draws"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare(Beta(2.0, 2.0), Beta(3.0, 3.0), **kwargs)  # type: ignore[arg-type]


def test_a_90_percent_interval_gives_the_matching_lognormal_prior() -> None:
    prior = lognormal_from_interval(2.0, 50.0)
    assert isinstance(prior, LogNormal)
    interval = prior.interval(0.9)
    assert interval.lower == pytest.approx(2.0, rel=1e-12)
    assert interval.upper == pytest.approx(50.0, rel=1e-12)
    assert prior.mu == pytest.approx((math.log(2.0) + math.log(50.0)) / 2.0)
    assert prior.sigma == pytest.approx((math.log(50.0) - math.log(2.0)) / (2 * 1.6448536269514722))
    assert prior.median == pytest.approx(10.0)
    draws = prior.sample(200_000, seed=1)
    assert np.mean((draws >= 2.0) & (draws <= 50.0)) == pytest.approx(0.9, abs=0.005)
    assert np.mean(draws < 2.0) == pytest.approx(0.05, abs=0.004)


def test_the_level_can_be_changed_and_the_prior_gets_wider_with_a_lower_level() -> None:
    strict = lognormal_from_interval(5.0, 20.0, level=0.99)
    loose = lognormal_from_interval(5.0, 20.0, level=0.5)
    assert strict.sigma < loose.sigma
    assert strict.interval(0.99).lower == pytest.approx(5.0, rel=1e-12)
    assert loose.interval(0.5).upper == pytest.approx(20.0, rel=1e-12)
    assert loose.mean > loose.median


@pytest.mark.parametrize(
    ("low", "high", "level"),
    [(0.0, 10.0, 0.9), (-1.0, 10.0, 0.9), (10.0, 10.0, 0.9), (10.0, 5.0, 0.9), (1.0, 10.0, 1.0)],
)
def test_bad_expert_intervals_are_refused(low: float, high: float, level: float) -> None:
    with pytest.raises(ValueError, match=r"bounds|level"):
        lognormal_from_interval(low, high, level)


def test_distributions_sample_with_a_seed() -> None:
    for dist in (Beta(2.0, 3.0), Gamma(2.0, 3.0), Normal(1.0, 2.0)):
        a, b = dist.sample(1000, seed=1), dist.sample(1000, seed=1)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, dist.sample(1000, seed=2))
    assert np.all(Beta(0.02, 0.03).sample(1000, seed=1) >= 0.0)
