from collections.abc import Callable

import numpy as np
import pytest

from statsx.sbc import check_calibration, sbc_ranks

N_DRAWS = 99
N_DATA = 5


def _replications(seed: int, *, scale: float = 1.0, shift: float = 0.0, n: int = 400) -> np.ndarray:
    """Return ranks for a normal mean with a normal prior.

    The posterior is exact at ``scale=1`` and ``shift=0``, and is made wrong by changing them.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n)
    y = rng.normal(mu[:, None], 1.0, size=(n, N_DATA))
    centre = y.sum(axis=1) / (N_DATA + 1)
    sd = scale / np.sqrt(N_DATA + 1)
    draws = rng.normal(centre[:, None] + shift, sd, size=(n, N_DRAWS))
    return sbc_ranks(mu, draws, seed=seed)


def test_an_exact_posterior_passes_at_about_the_stated_error_rate() -> None:
    rejected = sum(
        not check_calibration({"mu": _replications(seed)}, N_DRAWS).passed for seed in range(200)
    )
    assert rejected / 200 <= 0.08  # family-wise alpha is 0.05, so about 5% or fewer


def test_a_posterior_that_is_too_narrow_is_named_too_narrow() -> None:
    check = check_calibration({"mu": _replications(0, scale=0.6)}, N_DRAWS)
    assert not check.passed
    (mu,) = check.parameters
    assert mu.spread_z > 0
    assert any("too narrow" in r for r in check.reasons)


def test_a_posterior_that_is_too_wide_is_named_too_wide() -> None:
    check = check_calibration({"mu": _replications(1, scale=1.7)}, N_DRAWS)
    assert not check.passed
    assert check.parameters[0].spread_z < 0
    assert any("too wide" in r for r in check.reasons)


def test_a_shifted_posterior_is_named_biased_in_the_right_direction() -> None:
    below = check_calibration({"mu": _replications(2, shift=-0.5)}, N_DRAWS)
    assert any("sits below the truth" in r for r in below.reasons)
    above = check_calibration({"mu": _replications(3, shift=0.5)}, N_DRAWS)
    assert any("sits above the truth" in r for r in above.reasons)


def test_ranks_that_hit_every_value_equally_often_give_zero_statistics() -> None:
    ranks = np.tile(np.arange(N_DRAWS + 1), 5)
    (c,) = check_calibration({"x": ranks}, N_DRAWS).parameters
    assert c.mean_z == pytest.approx(0.0, abs=1e-12)
    assert c.spread_z == pytest.approx(0.0, abs=1e-12)
    assert c.chi_square_p == pytest.approx(1.0)
    assert sum(c.counts) == 500


def test_the_rank_is_the_number_of_draws_below_the_truth() -> None:
    posterior = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert sbc_ranks([2.5, 0.0], posterior).tolist() == [2, 0]
    assert sbc_ranks([9.0, 5.0], posterior).tolist() == [4, 4]


def test_ties_are_broken_at_random_and_stay_uniform() -> None:
    # A discrete parameter: the truth equals every draw, so the rank is uniform on 0..n_draws.
    ranks = sbc_ranks(np.zeros(4000), np.zeros((4000, 3)), seed=0)
    assert set(np.unique(ranks)) == {0, 1, 2, 3}
    assert check_calibration({"k": ranks}, 3).passed


def test_each_parameter_is_named_in_the_reasons() -> None:
    check = check_calibration(
        {"good": _replications(4), "bad": _replications(5, scale=0.5)}, N_DRAWS
    )
    assert all(r.startswith("bad:") for r in check.reasons)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sbc_ranks([1.0, 2.0], [[1.0, 2.0]]),
        lambda: sbc_ranks([1.0], [[1.0]]),
        lambda: sbc_ranks([np.nan], [[1.0, 2.0]]),
        lambda: check_calibration({"x": np.arange(10)}, 10),
        lambda: check_calibration({"x": np.full(30, 50)}, 10),
        lambda: check_calibration({"x": np.full(30, 0.5)}, 10),
        lambda: check_calibration({"x": np.zeros(30, dtype=int)}, 10, alpha=0.0),
    ],
)
def test_bad_input_is_refused(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match=r"."):
        call()
