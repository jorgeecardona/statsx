import math
from collections.abc import Callable

import numpy as np
import pytest

from statsx import _accel
from statsx.design import (
    Design,
    minimum_detectable_effect_paired,
    minimum_detectable_effect_two_proportions,
    power_equivalence_paired,
    power_equivalence_two_proportions,
    power_paired,
    power_two_proportions,
    sample_size_equivalence_paired,
    sample_size_equivalence_two_proportions,
    sample_size_paired,
    sample_size_two_proportions,
)
from statsx.equivalence import EquivalenceMargin, tost


def test_paired_t_power_matches_the_published_value_and_a_simulation() -> None:
    # G*Power: effect 0.5 sd, alpha 0.05 two-sided, 34 pairs: power 0.8077; 34 pairs for 80%.
    result = power_paired(0.5, 1.0, 34)
    assert isinstance(result, Design)
    assert result.power == pytest.approx(0.8077, abs=2e-4)
    assert sample_size_paired(0.5, 1.0, power=0.8).n == 34
    # A small-sample case against a simulation of the t test itself.
    rng = np.random.default_rng(0)
    n, runs = 10, 20000
    x = rng.normal(0.9, 1.0, (runs, n))
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / math.sqrt(n))
    critical = _accel.t_ppf(np.array([0.975]), n - 1.0)[0]
    assert power_paired(0.9, 1.0, n).power == pytest.approx(
        float(np.mean(np.abs(t) > critical)), abs=0.01
    )


def test_the_one_sided_power_is_higher_and_no_effect_gives_alpha() -> None:
    two_sided = power_paired(0.4, 1.0, 40).power
    one_sided = power_paired(0.4, 1.0, 40, alternative="greater").power
    assert one_sided > two_sided
    assert power_paired(0.0, 1.0, 40).power == pytest.approx(0.05, abs=1e-6)


def test_two_proportions_power_and_sample_size_match_the_fleiss_formula_and_an_exact_count() -> (
    None
):
    # Fleiss, Levin & Paik: 0.5 against 0.6 at alpha 0.05 and power 0.8 needs 388 per group.
    assert sample_size_two_proportions(0.5, 0.6).n == 388
    approx = power_two_proportions(0.5, 0.6, 388)
    exact = power_two_proportions(0.5, 0.6, 388, exact=True)
    assert approx.power == pytest.approx(0.8007, abs=5e-4)
    assert exact.power == pytest.approx(approx.power, abs=0.02)
    assert "exact" in exact.assumption
    # The exact count equals a simulation of the test.
    rng = np.random.default_rng(1)
    n, runs = 60, 40000
    a, b = rng.binomial(n, 0.4, runs), rng.binomial(n, 0.6, runs)
    pooled = (a + b) / (2.0 * n)
    spread = np.sqrt(np.maximum(pooled * (1 - pooled) * 2.0 / n, 1e-300))
    z = (b - a) / n / spread
    simulated = float(np.mean((np.abs(z) > 1.959963984540054) & (pooled > 0) & (pooled < 1)))
    assert power_two_proportions(0.4, 0.6, n, exact=True).power == pytest.approx(
        simulated, abs=0.01
    )


def test_paired_equivalence_power_is_exact_and_matches_the_tost_test() -> None:
    rng = np.random.default_rng(2)
    for theta, n in ((0.0, 30), (0.2, 40)):
        planned = power_equivalence_paired(theta, 1.0, 0.5, n).power
        hits = sum(
            tost(rng.normal(theta, 1.0, n), np.zeros(n), 0.5, paired=True).equivalent
            for _ in range(4000)
        )
        assert planned == pytest.approx(hits / 4000, abs=0.02)
    # No power when the true difference is outside the margin; at most alpha at its edge.
    assert power_equivalence_paired(0.8, 1.0, 0.5, 60).power < 0.001
    assert power_equivalence_paired(0.5, 1.0, 0.5, 60).power <= 0.05
    assert power_equivalence_paired(0.0, 1.0, EquivalenceMargin(-0.3, 0.7), 50).power > 0.2


def test_proportion_equivalence_power_follows_the_normal_approximation() -> None:
    rng = np.random.default_rng(3)
    n, runs, z = 500, 6000, 1.6448536269514722
    a, b = rng.binomial(n, 0.5, runs) / n, rng.binomial(n, 0.55, runs) / n
    se = np.sqrt(a * (1 - a) / n + b * (1 - b) / n)
    simulated = float(np.mean(((b - a + 0.1) / se > z) & ((0.1 - b + a) / se > z)))
    planned = power_equivalence_two_proportions(0.5, 0.55, 0.1, n).power
    assert planned == pytest.approx(simulated, abs=0.04)
    assert power_equivalence_two_proportions(0.5, 0.7, 0.1, n).power < 1e-5


def _paired(n: int) -> float:
    return power_paired(0.3, 1.0, n).power


def _paired_tost(n: int) -> float:
    return power_equivalence_paired(0.0, 1.0, 0.4, n).power


def _proportions(n: int) -> float:
    return power_two_proportions(0.3, 0.4, n).power


def _proportions_tost(n: int) -> float:
    return power_equivalence_two_proportions(0.5, 0.5, 0.08, n).power


def test_a_sample_size_is_the_smallest_that_reaches_the_power() -> None:
    cases: list[tuple[Design, float, Callable[[int], float]]] = [
        (sample_size_paired(0.3, 1.0, power=0.9), 0.9, _paired),
        (sample_size_equivalence_paired(1.0, 0.4, power=0.8), 0.8, _paired_tost),
        (sample_size_two_proportions(0.3, 0.4, power=0.85), 0.85, _proportions),
        (
            sample_size_equivalence_two_proportions(0.5, 0.5, 0.08, power=0.8),
            0.8,
            _proportions_tost,
        ),
    ]
    for result, target, power_of in cases:
        assert power_of(result.n) >= target
        assert power_of(result.n - 1) < target
        assert result.power == pytest.approx(power_of(result.n))


def test_the_minimum_detectable_effect_for_a_fixed_n_reaches_the_power_and_shrinks_with_n() -> None:
    mde = minimum_detectable_effect_paired(1.0, 50, power=0.8)
    assert mde.power == pytest.approx(0.8, abs=1e-6)
    assert mde.effect == pytest.approx(0.4, abs=0.02)  # about 0.40 sd for 50 pairs
    assert minimum_detectable_effect_paired(1.0, 200).effect < mde.effect
    prop = minimum_detectable_effect_two_proportions(0.3, 400, power=0.8)
    assert prop.power == pytest.approx(0.8, abs=1e-6)
    assert prop.effect > 0.0
    down = minimum_detectable_effect_two_proportions(0.3, 400, power=0.8, direction="decrease")
    assert down.effect < 0.0
    assert minimum_detectable_effect_two_proportions(0.3, 1600).effect < prop.effect


@pytest.mark.parametrize(
    "call",
    [
        lambda: power_paired(0.5, 0.0, 10),
        lambda: power_paired(0.5, 1.0, 1),
        lambda: power_paired(0.5, 1.0, 10, alpha=0.5),
        lambda: sample_size_paired(0.0, 1.0),
        lambda: sample_size_paired(0.5, 1.0, power=1.0),
        lambda: power_two_proportions(0.5, 1.5, 10),
        lambda: power_two_proportions(0.5, 0.6, 3000, exact=True),
        lambda: sample_size_two_proportions(0.5, 0.5),
        lambda: sample_size_equivalence_paired(1.0, 0.5, effect=0.6),
        lambda: sample_size_equivalence_two_proportions(0.5, 0.7, 0.1),
        lambda: minimum_detectable_effect_two_proportions(1.0, 100),
        lambda: sample_size_two_proportions(0.5, 0.5001),
    ],
)
def test_bad_planning_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError, match=r"alpha|power|sd|whole|proportions|exact|effect|margin|room|million"
    ):
        call()  # type: ignore[operator]
