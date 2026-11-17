import math

import numpy as np
import pytest

from statsx import _accel
from statsx.distributions import (
    LognormalFit,
    Moments,
    NormalFit,
    ShiftedExponentialFit,
    fit_normal,
)
from statsx.goodness import (
    chi_square_test,
    goodness_of_fit,
    total_variation_distance,
    wasserstein_distance,
)


def test_the_chi_square_test_matches_the_worked_die_example() -> None:
    # 60 rolls of a die, expected 10 each: statistic 13.4 on 5 df, p = 0.0199052203347744 (mpmath).
    r = chi_square_test([5, 8, 9, 8, 10, 20], [1 / 6] * 6)
    assert r.statistic == pytest.approx(13.4, abs=1e-12)
    assert r.df == 5
    assert r.p_value == pytest.approx(0.0199052203347744, rel=1e-10)
    assert (r.n, r.n_effective, r.n_bins) == (60, 60.0, 6)
    assert r.tvd == pytest.approx(0.5 * (5 + 2 + 1 + 2 + 0 + 10) / 60)
    assert r.wasserstein is None


def test_the_statistic_scales_with_the_effective_sample_size() -> None:
    full = chi_square_test([500, 300, 200], [0.4, 0.4, 0.2])
    tenth = chi_square_test([500, 300, 200], [0.4, 0.4, 0.2], n_effective=100)
    assert tenth.statistic == pytest.approx(full.statistic / 10.0, rel=1e-12)
    assert tenth.p_value > full.p_value
    assert tenth.n == 1000 and tenth.n_effective == 100.0
    assert tenth.tvd == full.tvd  # the distance does not depend on the sample size


def test_n_to_reject_is_where_the_p_value_crosses_alpha() -> None:
    r = chi_square_test([500, 300, 200], [0.4, 0.4, 0.2], alpha=0.05)
    phi = r.statistic / r.n_effective
    sf = _accel.chi2_sf(np.array([r.n_to_reject * phi]), float(r.df))[0]
    assert sf == pytest.approx(0.05, abs=1e-9)
    assert chi_square_test([40, 40, 20], [0.4, 0.4, 0.2]).n_to_reject == math.inf


def test_a_tiny_misfit_is_rejected_at_a_million_rows_but_not_at_the_effective_size() -> None:
    n = 1_000_000
    x = np.random.default_rng(0).normal(0.03, 1.0, n)  # shifted by 3% of a standard deviation
    fit = NormalFit(0.0, 1.0, n)
    big = goodness_of_fit(x, fit)
    assert big.p_value < 1e-3
    assert big.tvd < 0.02 and big.wasserstein == pytest.approx(0.03, abs=0.002)
    small = goodness_of_fit(x, fit, n_effective=1000)
    assert small.p_value > 0.05
    assert small.n_to_reject > 1000
    assert (small.n, small.n_effective) == (n, 1000.0)
    assert small.wasserstein == big.wasserstein


def test_the_p_value_is_calibrated_when_the_data_come_from_the_fit() -> None:
    fit = LognormalFit(0.3, 0.6, 1)
    rejected = 0
    runs = 300
    for seed in range(runs):
        x = fit.sample(1000, seed=seed)
        rejected += goodness_of_fit(x, fit).p_value < 0.05
    assert 0.02 <= rejected / runs <= 0.09


def test_the_wasserstein_distance_equals_the_area_between_the_distribution_functions() -> None:
    x = np.random.default_rng(1).normal(0.3, 1.2, 40)
    fit = NormalFit(0.0, 1.0, 1)
    grid = np.linspace(-12.0, 12.0, 800_001)
    ecdf = np.searchsorted(np.sort(x), grid, side="right") / x.size
    area = float(np.trapezoid(np.abs(ecdf - fit.cdf(grid)), grid))
    assert wasserstein_distance(x, fit) == pytest.approx(area, abs=2e-5)
    # Two normals with equal sd: the distance is the difference of the means.
    big = np.random.default_rng(2).normal(0.5, 1.0, 100_000)
    assert wasserstein_distance(big, fit) == pytest.approx(0.5, abs=0.01)


def test_the_distance_works_for_each_fit_and_is_small_for_data_from_the_fit() -> None:
    for fit in (
        NormalFit(2.0, 3.0, 1),
        LognormalFit(0.5, 0.4, 1),
        ShiftedExponentialFit(1.0, 2.0, 1),
    ):
        x = fit.sample(20_000, seed=5)
        assert wasserstein_distance(x, fit) < 0.05
        assert wasserstein_distance(x + 1.0, fit) > 0.9


def test_a_fit_from_the_same_rows_lowers_the_degrees_of_freedom() -> None:
    x = np.random.default_rng(3).normal(4.0, 2.0, 1000)
    fit = fit_normal(Moments.from_values(x))
    free = goodness_of_fit(x, fit)
    used = goodness_of_fit(x, fit, n_params=2)
    assert used.df == free.df - 2 and used.statistic == free.statistic
    assert (
        used.p_value < free.p_value
    )  # fewer degrees of freedom, same statistic: a smaller p-value


def test_total_variation_distance_is_half_the_sum_of_differences() -> None:
    assert total_variation_distance([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.25)
    assert total_variation_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: total_variation_distance([0.5, 0.6], [0.5, 0.5]),
        lambda: total_variation_distance([1.0], [0.5, 0.5]),
        lambda: chi_square_test([5, 5], [0.5, 0.4]),
        lambda: chi_square_test([5, 5, 5], [0.5, 0.5]),
        lambda: chi_square_test([5, 5], [0.5, 0.5], n_params=1),
        lambda: chi_square_test([5, 5, 5], [1 / 3] * 3, n_effective=100),
        lambda: chi_square_test([5, 5, 5], [1 / 3] * 3, n_effective=0),
        lambda: goodness_of_fit(np.arange(10.0), NormalFit(0.0, 1.0, 1)),
        lambda: goodness_of_fit(np.arange(100.0), NormalFit(0.0, 1.0, 1), n_bins=50),
        lambda: goodness_of_fit(np.arange(100.0), NormalFit(0.0, 1.0, 1), alpha=1.0),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"add up|same length|positive|n_effective|at least 25|n_bins|alpha|degrees",
    ):
        call()  # type: ignore[operator]
