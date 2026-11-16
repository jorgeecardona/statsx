import dataclasses

import pytest

from statsx import Estimate, Interval


def test_estimate_exposes_estimate_interval_n_and_assumption_as_frozen_fields() -> None:
    result = Estimate(
        estimate=0.4,
        interval=Interval(lower=0.3, upper=0.5, level=0.95),
        n=100,
        assumption="i.i.d. observations",
    )
    assert dataclasses.is_dataclass(result)
    assert {f.name for f in dataclasses.fields(result)} == {
        "estimate",
        "interval",
        "n",
        "assumption",
    }
    assert result.interval.contains(result.estimate)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n = 3  # type: ignore[misc]


def test_interval_refuses_lower_above_upper() -> None:
    with pytest.raises(ValueError, match="above upper"):
        Interval(lower=2.0, upper=1.0, level=0.9)


def test_interval_refuses_level_outside_unit() -> None:
    with pytest.raises(ValueError, match="level"):
        Interval(lower=0.0, upper=1.0, level=1.0)


def test_estimate_refuses_empty_assumption() -> None:
    with pytest.raises(ValueError, match="assumption"):
        Estimate(estimate=1.0, interval=Interval(0.0, 2.0, 0.5), n=1, assumption=" ")
