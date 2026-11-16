import pytest

from statsx.conjugate import (
    Beta,
    Dirichlet,
    Gamma,
    Normal,
    beta_binomial,
    dirichlet_multinomial,
    gamma_poisson,
    normal_normal,
)


def test_each_conjugate_update_returns_the_closed_form_posterior_as_a_named_object() -> None:
    # Closed forms from Gelman et al. (2013), BDA3, chapters 2 and 3.
    assert beta_binomial(Beta(1.0, 1.0), successes=7, trials=10) == Beta(8.0, 4.0)
    assert gamma_poisson(Gamma(2.0, 1.0), count=30, exposure=10) == Gamma(32.0, 11.0)
    post = normal_normal(Normal(0.0, 1.0), sigma=2.0, n=4, sample_mean=3.0)
    # prior precision 1, data precision 4/4 = 1: mean (0 + 3) / 2, sd sqrt(1/2)
    assert post.mean == pytest.approx(1.5)
    assert post.sd == pytest.approx(0.5**0.5)
    assert dirichlet_multinomial(Dirichlet((1.0, 1.0, 1.0)), [5, 0, 2]) == Dirichlet(
        (6.0, 1.0, 3.0)
    )


def test_posterior_moments() -> None:
    assert Beta(8.0, 4.0).mean == pytest.approx(2 / 3)
    assert Beta(2.0, 2.0).variance == pytest.approx(4 / (16 * 5))
    assert Gamma(32.0, 11.0).mean == pytest.approx(32 / 11)
    assert Dirichlet((6.0, 1.0, 3.0)).mean == pytest.approx((0.6, 0.1, 0.3))


def test_beta_binomial_refuses_more_successes_than_trials() -> None:
    with pytest.raises(ValueError, match="exceed trials"):
        beta_binomial(Beta(1.0, 1.0), successes=11, trials=10)


def test_priors_must_be_positive() -> None:
    with pytest.raises(ValueError, match="alpha"):
        beta_binomial(Beta(0.0, 1.0), successes=0, trials=0)
    with pytest.raises(ValueError, match="sigma"):
        normal_normal(Normal(0.0, 1.0), sigma=0.0, n=1, sample_mean=0.0)


def test_dirichlet_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="categories"):
        dirichlet_multinomial(Dirichlet((1.0, 1.0)), [1, 2, 3])
