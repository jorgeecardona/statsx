import itertools
import math

import numpy as np
import pytest

from statsx.drift import (
    classifier_test,
    kolmogorov_sf,
    ks_test,
    mmd_test,
    psi,
    psi_from_shares,
)


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    grid = np.unique(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), grid, side="right") / x.size
    fy = np.searchsorted(np.sort(y), grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def test_the_exact_ks_p_value_equals_the_count_over_every_possible_labelling() -> None:
    rng = np.random.default_rng(0)
    for n, m, shift in ((5, 4, 1.0), (6, 6, 0.3), (3, 7, 2.0)):
        x, y = rng.normal(shift, 1.0, n), rng.normal(0.0, 1.0, m)
        pooled = np.concatenate([x, y])
        observed = _ks_statistic(x, y)
        hits = total = 0
        for chosen in itertools.combinations(range(n + m), n):
            mask = np.zeros(n + m, dtype=bool)
            mask[list(chosen)] = True
            total += 1
            hits += _ks_statistic(pooled[mask], pooled[~mask]) >= observed - 1e-12
        result = ks_test(x, y)
        assert result.statistic == pytest.approx(observed, abs=1e-12)
        assert result.p_value == pytest.approx(hits / total, abs=1e-10)


def test_the_kolmogorov_limit_matches_the_published_critical_values() -> None:
    assert kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=2e-4)
    assert kolmogorov_sf(1.6276) == pytest.approx(0.01, abs=1e-4)
    assert kolmogorov_sf(0.0) == 1.0


def test_ks_rejects_at_the_stated_rate_with_and_without_the_exact_branch() -> None:
    rng = np.random.default_rng(1)
    for n, m in ((40, 30), (150, 120)):  # n * m = 1200 (exact) and 18000 (limit)
        rejected = sum(
            ks_test(rng.normal(size=n), rng.normal(size=m)).p_value < 0.05 for _ in range(1500)
        )
        assert 0.035 <= rejected / 1500 <= 0.065
    # A shift is found, and ties use the limit.
    assert ks_test(rng.normal(0.5, 1.0, 200), rng.normal(0.0, 1.0, 200)).p_value < 0.01
    tied = ks_test(np.repeat([1.0, 2.0, 3.0], 5), np.repeat([1.0, 2.0, 3.0], 4))
    assert "limit" in tied.assumption


def test_the_unbiased_mmd_has_the_closed_form_mean_for_two_normals() -> None:
    # E k(X, Y) = h / sqrt(h^2 + sx^2 + sy^2) * exp(-(mx - my)^2 / (2 (h^2 + sx^2 + sy^2)))
    h, mx, sx, my, sy = 1.0, 0.0, 1.0, 0.6, 1.5

    def expected(m1: float, s1: float, m2: float, s2: float) -> float:
        v = h * h + s1 * s1 + s2 * s2
        return h / math.sqrt(v) * math.exp(-((m1 - m2) ** 2) / (2.0 * v))

    truth = expected(mx, sx, mx, sx) + expected(my, sy, my, sy) - 2.0 * expected(mx, sx, my, sy)
    rng = np.random.default_rng(2)
    stats = np.array(
        [
            mmd_test(
                rng.normal(mx, sx, 80), rng.normal(my, sy, 70), bandwidth=h, n_permutations=1
            ).statistic
            for _ in range(600)
        ]
    )
    assert stats.mean() == pytest.approx(truth, abs=4.0 * stats.std() / math.sqrt(600))


def test_the_mmd_permutation_test_has_the_stated_level_and_finds_a_shift() -> None:
    rng = np.random.default_rng(3)
    rejected = sum(
        mmd_test(
            rng.normal(size=(40, 3)), rng.normal(size=(40, 3)), n_permutations=199, seed=i
        ).p_value
        < 0.05
        for i in range(300)
    )
    assert 0.02 <= rejected / 300 <= 0.085
    shifted = mmd_test(
        rng.normal(0.5, 1.0, (100, 3)), rng.normal(size=(100, 3)), n_permutations=499
    )
    assert shifted.p_value < 0.01 and shifted.statistic > 0.0


def test_the_classifier_auc_matches_the_normal_theory_value_for_a_shift() -> None:
    # One variable, shift d = 1, equal sd: the best AUC is Phi(d / sqrt(2)) = 0.760.
    rng = np.random.default_rng(4)
    result = classifier_test(rng.normal(1.0, 1.0, 4000), rng.normal(0.0, 1.0, 4000))
    assert result.statistic == pytest.approx(0.760, abs=0.02)
    assert result.p_value < 1e-10


def test_the_classifier_test_has_the_stated_level_and_needs_quadratic_terms_for_spread() -> None:
    rng = np.random.default_rng(5)
    rejected = sum(
        classifier_test(rng.normal(size=(120, 4)), rng.normal(size=(100, 4)), seed=i).p_value < 0.05
        for i in range(400)
    )
    assert 0.03 <= rejected / 400 <= 0.07
    wide, narrow = rng.normal(0.0, 2.0, (1500, 2)), rng.normal(0.0, 1.0, (1500, 2))
    assert classifier_test(wide, narrow).p_value > 0.01  # a linear model cannot see the spread
    assert classifier_test(wide, narrow, quadratic=True).p_value < 1e-6


def test_psi_equals_the_definition_and_its_p_value_has_the_stated_level() -> None:
    expected = (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.6 - 0.5) * math.log(0.6 / 0.5)
    assert psi_from_shares([0.5, 0.5], [0.4, 0.6]) == pytest.approx(expected, rel=1e-12)
    assert psi_from_shares([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert (
        4.0 < psi_from_shares([0.5, 0.5], [0.0, 1.0]) < 5.0
    )  # an empty bin is floored, not infinite
    rng = np.random.default_rng(6)
    rejected = sum(
        psi(rng.normal(size=1000), rng.normal(size=1000)).p_value < 0.05 for _ in range(1500)
    )
    assert 0.035 <= rejected / 1500 <= 0.065
    assert psi(rng.normal(size=2000), rng.normal(0.5, 1.0, 2000)).statistic > 0.25


@pytest.mark.parametrize(
    "call",
    [
        lambda: ks_test([1.0], [1.0, 2.0]),
        lambda: ks_test([1.0, 2.0], [1.0, float("nan")]),
        lambda: mmd_test(np.zeros((5, 2)) + np.arange(5)[:, None], np.zeros((5, 3))),
        lambda: mmd_test(np.ones((5, 2)), np.ones((5, 2))),
        lambda: mmd_test([1.0, 2.0], [3.0, 4.0], n_permutations=0),
        lambda: classifier_test(np.zeros((12, 2)), np.zeros((12, 2)), test_fraction=1.0),
        lambda: classifier_test(np.arange(12.0), np.arange(12.0)),
        lambda: psi_from_shares([0.5, 0.6], [0.5, 0.5]),
        lambda: psi_from_shares([1.0], [1.0]),
        lambda: psi([1.0, 1.0, 1.0, 1.0], [1.0, 2.0]),
        lambda: psi([1.0, 2.0, 3.0], [1.0, 2.0], n_bins=1),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"at least|columns|finite|bandwidth|n_permutations|test_fraction|rows|add up|shares|distinct",
    ):
        call()  # type: ignore[operator]
