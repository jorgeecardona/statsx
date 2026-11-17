import math

import numpy as np
import pytest

from statsx import _accel
from statsx.design import (
    minimum_detectable_effect_two_sample,
    power_equivalence_two_sample,
    power_two_sample,
    sample_size_equivalence_two_sample,
    sample_size_two_sample,
)


def test_the_power_matches_cohens_classic_sample_sizes() -> None:
    # Cohen (1988): d = 0.5 needs 64 per group; G*Power gives power 0.8014596 at n = 64.
    assert power_two_sample(0.5, 1.0, 64).power == pytest.approx(0.8014596, abs=1e-6)
    assert sample_size_two_sample(0.5, 1.0).n == 64
    assert sample_size_two_sample(0.8, 1.0).n == 26
    assert sample_size_two_sample(0.2, 1.0).n == 394
    # The effect is in the unit of the data: the sd scales it away.
    assert sample_size_two_sample(0.05, 0.1).n == 64
    assert power_two_sample(0.5, 1.0, 64, alternative="greater").power > 0.87


def test_a_simulation_of_the_two_sample_test_has_the_analytic_power() -> None:
    n, effect, sd, runs = 30, 0.6, 1.0, 40_000
    rng = np.random.default_rng(0)
    a = rng.normal(effect, sd, (runs, n))
    b = rng.normal(0.0, sd, (runs, n))
    pooled = np.sqrt((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2.0)
    t = (a.mean(axis=1) - b.mean(axis=1)) / (pooled * math.sqrt(2.0 / n))
    critical = _accel.t_ppf(np.array([0.975]), 2.0 * (n - 1.0))[0]
    simulated = float(np.mean(np.abs(t) > critical))
    assert simulated == pytest.approx(power_two_sample(effect, sd, n).power, abs=0.01)
    # Under no effect the same test rejects at alpha.
    t0 = (b.mean(axis=1) - a.mean(axis=1) + effect) / (pooled * math.sqrt(2.0 / n))
    assert float(np.mean(np.abs(t0) > critical)) == pytest.approx(0.05, abs=0.01)


def test_the_minimum_detectable_effect_reaches_the_power_at_that_effect() -> None:
    five = minimum_detectable_effect_two_sample(1.0, 5)
    ten = minimum_detectable_effect_two_sample(1.0, 10)
    assert five.power == pytest.approx(0.8, abs=1e-6)
    assert ten.power == pytest.approx(0.8, abs=1e-6)
    assert ten.effect < five.effect
    # The effect scales with the sd.
    scaled = minimum_detectable_effect_two_sample(0.02, 5)
    assert scaled.effect == pytest.approx(0.02 * five.effect, rel=1e-9)


def test_the_equivalence_power_matches_a_simulation_and_the_sample_size_reaches_it() -> None:
    n, sd, margin, runs = 40, 1.0, 0.5, 40_000
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, sd, (runs, n))
    b = rng.normal(0.0, sd, (runs, n))
    pooled = np.sqrt((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2.0)
    se = pooled * math.sqrt(2.0 / n)
    diff = a.mean(axis=1) - b.mean(axis=1)
    tc = _accel.t_ppf(np.array([0.95]), 2.0 * (n - 1.0))[0]
    simulated = float(np.mean((diff - tc * se > -margin) & (diff + tc * se < margin)))
    analytic = power_equivalence_two_sample(0.0, sd, margin, n).power
    assert simulated == pytest.approx(analytic, abs=0.01)
    needed = sample_size_equivalence_two_sample(sd, margin)
    assert needed.power >= 0.8
    assert power_equivalence_two_sample(0.0, sd, margin, needed.n - 1).power < 0.8


@pytest.mark.parametrize(
    "call",
    [
        lambda: power_two_sample(0.5, 0.0, 10),
        lambda: power_two_sample(0.5, 1.0, 1),
        lambda: sample_size_two_sample(0.0, 1.0),
        lambda: minimum_detectable_effect_two_sample(1.0, 10, power=1.0),
        lambda: sample_size_equivalence_two_sample(1.0, 0.5, effect=0.6),
        lambda: power_equivalence_two_sample(0.0, -1.0, 0.5, 10),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"sd|n must|power|inside|effect"):
        call()  # type: ignore[operator]
