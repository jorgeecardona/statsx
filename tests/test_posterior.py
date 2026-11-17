import numpy as np
import pytest

from statsx.posterior import credible_interval, hdi, quantiles


def test_quantiles_are_one_call_and_match_linear_interpolation() -> None:
    draws = np.random.default_rng(0).normal(10.0, 2.0, 5000)
    result = quantiles(draws, (0.025, 0.5, 0.975))
    assert result.probs == (0.025, 0.5, 0.975)
    assert result[0.5] == pytest.approx(np.quantile(draws, 0.5), rel=1e-12)
    assert result[0.5] == pytest.approx(10.0, abs=0.1)
    assert result[0.975] == pytest.approx(10.0 + 1.96 * 2.0, abs=0.2)
    assert quantiles([1.0, 2.0, 3.0, 4.0], (0.5,))[0.5] == 2.5


def test_credible_interval_has_equal_tails() -> None:
    draws = np.random.default_rng(1).gamma(2.0, 1.0, 20000)
    interval = credible_interval(draws, 0.9)
    assert interval.level == 0.9
    assert np.mean(draws < interval.lower) == pytest.approx(0.05, abs=0.005)
    assert np.mean(draws > interval.upper) == pytest.approx(0.05, abs=0.005)


def test_hdi_is_the_shortest_interval_with_the_stated_mass() -> None:
    draws = np.random.default_rng(2).gamma(2.0, 1.0, 20000)
    interval = hdi(draws, 0.9)
    inside = np.mean((draws >= interval.lower) & (draws <= interval.upper))
    assert inside == pytest.approx(0.9, abs=0.001)
    equal_tailed = credible_interval(draws, 0.9)
    # A right-skewed density: the HDI is shorter and starts lower than the equal-tailed interval.
    assert interval.upper - interval.lower < equal_tailed.upper - equal_tailed.lower
    assert interval.lower < equal_tailed.lower
    # Brute force over every window of the sorted draws.
    x = np.sort(draws)
    k = int(np.ceil(0.9 * x.size))
    assert interval.upper - interval.lower == pytest.approx(np.min(x[k:] - x[:-k]))


def test_hdi_of_a_symmetric_sample_matches_the_equal_tailed_interval() -> None:
    draws = np.random.default_rng(3).normal(0.0, 1.0, 100000)
    a, b = hdi(draws, 0.95), credible_interval(draws, 0.95)
    assert a.lower == pytest.approx(b.lower, abs=0.05)
    assert a.upper == pytest.approx(b.upper, abs=0.05)
    assert a.upper == pytest.approx(1.96, abs=0.05)


@pytest.mark.parametrize("bad", [[], [1.0], [1.0, float("nan")], [[1.0, 2.0], [3.0, 4.0]]])
def test_unusable_draws_are_refused(bad: list[object]) -> None:
    for call in (quantiles, credible_interval, hdi):
        with pytest.raises(ValueError, match="draws"):
            call(bad)  # type: ignore[arg-type]


def test_levels_and_probabilities_are_checked() -> None:
    with pytest.raises(ValueError, match="level"):
        hdi([1.0, 2.0, 3.0], 1.0)
    with pytest.raises(ValueError, match="level"):
        credible_interval([1.0, 2.0, 3.0], 0.0)
    with pytest.raises(ValueError, match="probs"):
        quantiles([1.0, 2.0, 3.0], (1.5,))
