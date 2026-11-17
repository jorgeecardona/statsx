import numpy as np
import pytest

from statsx import _accel
from statsx.conjugate import Normal
from statsx.equivalence import EquivalenceMargin, rope_probability, tost
from statsx.posterior import compare
from statsx.result import Estimate


def _groups(seed: int, shift: float, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, n), rng.normal(0.0, 1.0, n)


def test_student_t_functions_agree_with_the_normal_limit_and_each_other() -> None:
    # mpmath values are checked in the Rust tests; here the two functions invert each other.
    for df in (1.0, 4.5, 30.0, 1000.0):
        p = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
        t = np.array(_accel.t_ppf(p, df))
        np.testing.assert_allclose(_accel.t_cdf(t, df), p, rtol=1e-9)
    assert _accel.t_ppf(np.array([0.975]), 1e6)[0] == pytest.approx(1.959964, abs=1e-4)


def test_two_one_sided_tests_find_equivalence_when_the_groups_are_the_same() -> None:
    a, b = _groups(1, shift=0.0, n=200)
    result = tost(a, b, 0.5)
    assert result.equivalent
    assert result.p_value < 0.05
    assert isinstance(result.difference, Estimate)
    assert result.difference.interval.level == pytest.approx(0.90)
    assert -0.5 < result.difference.interval.lower < result.difference.interval.upper < 0.5
    assert "one-sided" in result.difference.assumption


def test_a_real_difference_is_not_equivalent_and_a_small_sample_is_inconclusive() -> None:
    a, b = _groups(2, shift=0.8, n=200)
    assert not tost(a, b, 0.5).equivalent
    a, b = _groups(3, shift=0.0, n=6)
    inconclusive = tost(a, b, 0.5)
    assert not inconclusive.equivalent
    # The interval is wide: it reaches outside the margin, so neither claim is supported.
    assert inconclusive.difference.interval.upper - inconclusive.difference.interval.lower > 0.5


def test_tost_is_the_same_as_the_interval_lying_inside_the_margin() -> None:
    for seed in range(60):
        a, b = _groups(seed, shift=0.1 * (seed % 5), n=30 + seed)
        result = tost(a, b, EquivalenceMargin(-0.6, 0.4), alpha=0.05)
        inside = result.difference.interval.lower > -0.6 and result.difference.interval.upper < 0.4
        assert result.equivalent == inside


def test_the_error_rate_at_the_edge_of_the_margin_is_at_most_alpha() -> None:
    wrongly = [tost(*_groups(seed, shift=0.5, n=60), 0.5).equivalent for seed in range(1000)]
    # The true difference sits at the margin, where equivalence is false: the rate is about alpha.
    assert float(np.mean(wrongly)) <= 0.07


def test_paired_data_use_the_differences() -> None:
    rng = np.random.default_rng(4)
    before = rng.normal(10.0, 3.0, 40)
    after = before + rng.normal(0.05, 0.3, 40)
    paired = tost(after, before, 0.3, paired=True)
    unpaired = tost(after, before, 0.3)
    assert paired.equivalent and not unpaired.equivalent
    assert paired.df == 39.0
    assert paired.difference.estimate == pytest.approx(float(np.mean(after - before)))
    with pytest.raises(ValueError, match="same length"):
        tost(after, before[:-1], 0.3, paired=True)


def test_the_same_margin_defines_the_rope_probability_and_the_comparison() -> None:
    margin = EquivalenceMargin(-0.1, 0.3)
    a, b = Normal(0.5, 0.2), Normal(0.4, 0.2)
    result = compare(a, b, margin=margin, n_draws=400_000, seed=1)
    assert result.margin == margin
    draws = a.sample(400_000, seed=1) - b.sample(400_000, seed=2)
    assert result.p_equivalent == rope_probability(draws, margin)
    # A number is the symmetric margin: one definition for every caller.
    assert compare(a, b, margin=0.2, seed=1).margin == EquivalenceMargin.symmetric(0.2)
    assert rope_probability(draws, 0.2) == rope_probability(draws, EquivalenceMargin(-0.2, 0.2))
    assert margin.contains(0.3) and not margin.contains(0.31)


@pytest.mark.parametrize(
    "build",
    [
        lambda: EquivalenceMargin(0.1, 0.2),
        lambda: EquivalenceMargin(-0.2, -0.1),
        lambda: EquivalenceMargin(0.2, -0.2),
        lambda: EquivalenceMargin.symmetric(0.0),
        lambda: tost([1.0, 2.0], [1.0, 2.0], 0.5, alpha=0.5),
        lambda: tost([1.0], [1.0, 2.0], 0.5),
        lambda: tost([1.0, 1.0], [2.0, 2.0], 0.5),
        lambda: rope_probability([1.0], 0.1),
    ],
)
def test_bad_input_is_refused(build: object) -> None:
    with pytest.raises(ValueError, match=r"margin|alpha|observations|spread|draws"):
        build()  # type: ignore[operator]
