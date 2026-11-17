import math

import numpy as np
import pytest

from statsx.effect import effect_size, hedges_j
from statsx.result import Estimate


def _groups(seed: int, delta: float, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(delta, 1.0, n), rng.normal(0.0, 1.0, n)


def test_the_effect_size_comes_with_an_interval_and_its_assumption() -> None:
    a, b = _groups(1, 0.5)
    result = effect_size(a, b, kind="cohen_d")
    assert isinstance(result, Estimate)
    assert result.n == 80
    assert result.interval.lower < result.estimate < result.interval.upper
    assert result.interval.level == 0.95
    assert "independent groups" in result.assumption


def test_cohens_d_matches_the_definition() -> None:
    a = np.array([5.0, 7.0, 9.0, 11.0])
    b = np.array([2.0, 4.0, 6.0])
    pooled_var = (3 * a.var(ddof=1) + 2 * b.var(ddof=1)) / 5
    want = (a.mean() - b.mean()) / math.sqrt(pooled_var)
    assert effect_size(a, b, kind="cohen_d").estimate == pytest.approx(want, rel=1e-12)
    assert effect_size(a, b, kind="mean_difference", method="bootstrap").estimate == pytest.approx(
        a.mean() - b.mean()
    )


def test_hedges_g_is_d_times_the_small_sample_correction() -> None:
    a, b = _groups(2, 0.8, n=12)
    d = effect_size(a, b, kind="cohen_d").estimate
    g = effect_size(a, b, kind="hedges_g").estimate
    assert g == pytest.approx(d * hedges_j(22.0), rel=1e-12)
    assert abs(g) < abs(d)
    # The correction matches the usual approximation 1 - 3 / (4 df - 1), and it tends to 1.
    assert hedges_j(10.0) == pytest.approx(1 - 3 / (4 * 10 - 1), abs=1e-3)
    assert hedges_j(2000.0) == pytest.approx(1.0, abs=1e-3)


def test_the_analytic_interval_covers_the_true_effect_at_about_the_nominal_rate() -> None:
    covered: list[bool] = []
    for seed in range(400):
        a, b = _groups(seed, 0.5, n=40)
        covered.append(effect_size(a, b, kind="hedges_g", level=0.9).interval.contains(0.5))
    assert 0.85 <= float(np.mean(covered)) <= 0.95


def test_the_bootstrap_interval_agrees_with_the_analytic_one() -> None:
    a, b = _groups(3, 0.6, n=60)
    analytic = effect_size(a, b, kind="hedges_g")
    boot = effect_size(a, b, kind="hedges_g", method="bootstrap", n_resamples=1000, seed=1)
    assert boot.estimate == analytic.estimate
    assert boot.interval.lower == pytest.approx(analytic.interval.lower, abs=0.12)
    assert boot.interval.upper == pytest.approx(analytic.interval.upper, abs=0.12)
    assert "bootstrap" in boot.assumption


def test_a_larger_sample_gives_a_narrower_interval() -> None:
    small = effect_size(*_groups(4, 0.5, n=15))
    large = effect_size(*_groups(4, 0.5, n=150))
    assert large.interval.upper - large.interval.lower < small.interval.upper - small.interval.lower


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "message"),
    [
        ([1.0], [1.0, 2.0], {}, "at least 2"),
        ([1.0, float("nan")], [1.0, 2.0], {}, "finite"),
        ([3.0, 3.0], [3.0, 3.0], {}, "constant"),
        ([1.0, 2.0], [3.0, 4.0], {"level": 1.0}, "level"),
        ([1.0, 2.0], [3.0, 4.0], {"kind": "mean_difference"}, "bootstrap"),
        ([1.0, 2.0], [3.0, 4.0], {"method": "exact"}, "method"),
    ],
)
def test_bad_input_is_refused(
    a: list[float], b: list[float], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        effect_size(a, b, **kwargs)  # type: ignore[arg-type]
