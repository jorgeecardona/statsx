import numpy as np
import pytest

from statsx.impact import ASSUMPTION, causal_impact
from statsx.result import Estimate

MONTHS = 60
START = 48


def _series(seed: int, effect: float, controls: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """Counts with a multiplicative effect from month 48, three controls sharing a factor.

    Returns the counts, the controls, and the true expected cumulative effect.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(MONTHS)
    factor = np.cumsum(rng.normal(0.0, 0.08, MONTHS))
    season = 0.4 * np.cos(2 * np.pi * t / 12)
    control = np.column_stack(
        [
            rng.negative_binomial(20.0, 20.0 / (20.0 + np.exp(np.log(40.0) + factor + season)))
            for _ in range(controls)
        ]
    )
    without = np.exp(np.log(30.0) + 0.8 * factor + season)
    mu = without * np.exp(effect * (t >= START))
    counts = rng.negative_binomial(20.0, 20.0 / (20.0 + mu))
    return counts, control, float((mu - without)[START:].sum())


def test_a_large_effect_is_found_with_its_size() -> None:
    counts, controls, truth = _series(seed=1, effect=0.8)
    impact = causal_impact(counts, controls, START, drift_scale=0.05, seed=0)
    cumulative = impact.cumulative(0.95)
    assert impact.p_positive() > 0.95
    assert cumulative.interval.lower > 0.0
    assert abs(cumulative.estimate - truth) < 0.35 * truth
    band = impact.band(0.95)
    assert band.lower.shape == (MONTHS - START,)
    assert np.all(band.lower <= band.upper)
    assert impact.average(0.95).estimate == pytest.approx(cumulative.estimate / (MONTHS - START))


def test_no_effect_is_rarely_called_an_effect() -> None:
    excluded: list[bool] = []
    for seed in range(15):
        counts, controls, _ = _series(seed=50 + seed, effect=0.0)
        impact = causal_impact(counts, controls, START, drift_scale=0.05, seed=seed)
        excluded.append(not impact.cumulative(0.95).interval.contains(0.0))
    # The nominal rate is 5%; allow for 15 series and for the extra risk of a wrong prior.
    assert np.mean(excluded) <= 0.2


def test_the_result_states_its_assumption_and_sample_size() -> None:
    counts, controls, _ = _series(seed=2, effect=0.3)
    impact = causal_impact(counts, controls, START, seed=0)
    result = impact.cumulative()
    assert isinstance(result, Estimate)
    assert result.n == MONTHS - START
    assert result.assumption == ASSUMPTION
    assert "controls" in result.assumption
    assert impact.effect.shape == (2000, MONTHS - START)
    assert impact.fit.n_regressors == 3
    assert impact.fit.counts.size == START


def test_one_control_can_be_passed_as_a_vector() -> None:
    counts, controls, _ = _series(seed=3, effect=0.0, controls=1)
    impact = causal_impact(counts, controls[:, 0], START, seed=0)
    assert impact.fit.n_regressors == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"intervention": 10}, "intervention"),
        ({"intervention": MONTHS}, "intervention"),
        ({"intervention": START, "level": 1.5}, "level"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], message: str) -> None:
    counts, controls, _ = _series(seed=4, effect=0.0)
    level = kwargs.pop("level", None)
    if level is not None:
        impact = causal_impact(counts, controls, START, seed=0)
        with pytest.raises(ValueError, match=message):
            impact.band(float(level))  # type: ignore[arg-type]
        return
    with pytest.raises(ValueError, match=message):
        causal_impact(counts, controls, **kwargs)  # type: ignore[arg-type]


def test_controls_must_match_the_months_and_be_non_negative() -> None:
    counts, controls, _ = _series(seed=5, effect=0.0)
    with pytest.raises(ValueError, match="one row per month"):
        causal_impact(counts, controls[:-1], START)
    with pytest.raises(ValueError, match="non-negative"):
        causal_impact(counts, controls - 100.0, START)
