import math

import numpy as np
import pytest

from statsx.conjugate import NormalInverseGamma
from statsx.distributions import Moments
from statsx.equivalence import tost
from statsx.eta import DurationPrior, fit_duration
from statsx.posterior import compare

A = [13.71, 13.58, 14.39, 13.80, 13.66]
B = [13.21, 13.30, 13.12, 13.45, 13.26]
WEAK = NormalInverseGamma(0.0, 1e-6, 1.0, 0.02**2)


def test_the_posterior_equals_the_intercept_only_duration_fit() -> None:
    prior = DurationPrior(coef_sd=1e3, noise_shape=1.0, noise_scale=0.02**2)
    model = fit_duration(A, {}, prior=prior)
    post = NormalInverseGamma(0.0, 1.0 / prior.coef_sd**2, prior.noise_shape, prior.noise_scale)
    post = post.update(np.log(A))
    assert post.mean == pytest.approx(float(model.coef_mean[0]), rel=1e-12)
    assert 1.0 / post.kappa == pytest.approx(float(model.coef_cov[0, 0]), rel=1e-12)
    assert post.shape == pytest.approx(model.noise_shape, rel=1e-12)
    assert post.scale == pytest.approx(model.noise_rate, rel=1e-10)


def test_updating_in_pieces_or_from_moments_gives_the_same_posterior() -> None:
    y = np.random.default_rng(0).normal(5.0, 2.0, 60)
    prior = NormalInverseGamma(1.0, 0.5, 2.0, 3.0)
    whole = prior.update(y)
    pieces = prior.update(y[:25]).update(y[25:])
    moments = prior.update_from_moments(Moments.from_values(y))
    for other in (pieces, moments):
        assert other.mean == pytest.approx(whole.mean, rel=1e-12)
        assert other.kappa == pytest.approx(whole.kappa, rel=1e-12)
        assert other.shape == pytest.approx(whole.shape, rel=1e-12)
        assert other.scale == pytest.approx(whole.scale, rel=1e-10)


def test_the_draws_follow_the_student_t_marginal_of_the_mean() -> None:
    post = NormalInverseGamma(0.0, 1e-6, 1.0, 0.02**2).update(np.log(A))
    exact = post.mean_interval(0.95)
    draws = post.sample(400_000, seed=3)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    assert lo == pytest.approx(exact.lower, abs=0.0015)
    assert hi == pytest.approx(exact.upper, abs=0.0015)
    assert post.mean_df == 7.0
    mu, sigma2 = post.sample_joint(10, seed=1)
    assert mu.shape == sigma2.shape == (10,) and np.all(sigma2 > 0.0)
    np.testing.assert_array_equal(post.sample(10, seed=1), mu)


def test_under_the_reference_prior_the_difference_matches_the_welch_interval() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.1, 20), rng.normal(0.9, 0.1, 20)
    c = compare(
        NormalInverseGamma.reference(a),
        NormalInverseGamma.reference(b),
        margin=0.05,
        level=0.9,
        n_draws=400_000,
        seed=2,
    )
    t = tost(a, b, 0.05)
    assert t.difference.interval.level == 0.9  # the TOST interval is 1 - 2 alpha
    middle = 0.5 * (c.difference.lower + c.difference.upper)
    assert middle == pytest.approx(t.difference.estimate, abs=2e-3)
    assert c.difference.lower == pytest.approx(t.difference.interval.lower, rel=0.03)
    assert c.difference.upper == pytest.approx(t.difference.interval.upper, rel=0.03)


@pytest.mark.parametrize("noise_sd", [0.005, 0.02, 0.04, 0.1])
def test_the_reference_posterior_covers_at_the_stated_rate_at_n_5_for_any_noise(
    noise_sd: float,
) -> None:
    rng = np.random.default_rng(int(noise_sd * 1e4))
    runs, hits = 6000, 0
    for _ in range(runs):
        y = rng.normal(0.0, noise_sd, 5)
        hits += NormalInverseGamma.reference(y).mean_interval(0.95).contains(0.0)
    assert hits / runs == pytest.approx(0.95, abs=0.012)


def test_the_reference_interval_is_the_classical_t_interval() -> None:
    y = np.array([13.71, 13.58, 14.39, 13.80, 13.66])
    post = NormalInverseGamma.reference(y)
    half = 2.7764451051977987 * y.std(ddof=1) / math.sqrt(5)  # t(0.975, 4 df)
    assert post.mean == pytest.approx(y.mean())
    assert post.mean_df == 4.0
    assert post.mean_interval().upper - post.mean == pytest.approx(half, rel=1e-9)
    np.testing.assert_allclose(
        NormalInverseGamma.reference(Moments.from_values(y)).scale, post.scale, rtol=1e-12
    )


def test_an_informative_noise_prior_covers_well_only_near_its_own_noise_level() -> None:
    # The prior noise sd is about 0.02. It covers 95% when the true noise is near that and
    # over-covers below it, but it falls under 93% once the true noise is 1.5 times larger.
    def coverage(noise_sd: float) -> float:
        rng = np.random.default_rng(int(noise_sd * 1e4) + 1)
        hits = sum(
            WEAK.update(rng.normal(0.0, noise_sd, 5)).mean_interval(0.95).contains(0.0)
            for _ in range(4000)
        )
        return hits / 4000

    assert coverage(0.01) >= 0.93
    assert coverage(0.02) >= 0.93
    assert coverage(0.04) < 0.93


def test_compare_takes_the_posteriors_directly_and_finds_version_b_faster() -> None:
    pa, pb = (WEAK.update(np.log(t)) for t in (A, B))
    c = compare(pa, pb, margin=math.log(1.02))
    assert c.p_a_greater > 0.99  # A - B > 0 on the log scale: B is faster
    assert 1.0 < math.exp(c.difference.lower) < math.exp(c.difference.upper) < 1.1


@pytest.mark.parametrize(
    "call",
    [
        lambda: NormalInverseGamma(0.0, 0.0, 1.0, 1.0),
        lambda: NormalInverseGamma(0.0, 1.0, -1.0, 1.0),
        lambda: NormalInverseGamma(float("nan"), 1.0, 1.0, 1.0),
        lambda: WEAK.update([]),
        lambda: NormalInverseGamma.reference([1.0]),
        lambda: NormalInverseGamma.reference([2.0, 2.0]),
        lambda: WEAK.update([1.0, float("inf")]),
        lambda: WEAK.mean_interval(1.0),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"kappa|values|level|reference"):
        call()  # type: ignore[operator]
