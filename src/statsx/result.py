"""Result types shared by every statsx estimator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    """A two-sided interval at a stated level.

    Args:
        lower: Lower end. Must not exceed ``upper``.
        upper: Upper end.
        level: Coverage in (0, 1), for example ``0.95``.

    Raises:
        ValueError: If ``lower > upper`` or ``level`` is not in (0, 1).
    """

    lower: float
    upper: float
    level: float

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if self.lower > self.upper:
            raise ValueError(f"lower {self.lower} is above upper {self.upper}")

    def contains(self, value: float) -> bool:
        """Return True when ``value`` lies in the closed interval."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class Estimate:
    """An estimate with its interval, its sample size, and the assumption it rests on.

    Every statsx function that returns a number with uncertainty returns this type (ES-6).

    Args:
        estimate: The point estimate.
        interval: The interval around it.
        n: The number of observations used. Must be at least 0.
        assumption: The model assumption in words, for example
            ``"i.i.d. resampling of observations"``. Must not be empty.

    Raises:
        ValueError: If ``n`` is negative or ``assumption`` is empty.
    """

    estimate: float
    interval: Interval
    n: int
    assumption: str

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"n must be at least 0, got {self.n}")
        if not self.assumption.strip():
            raise ValueError("assumption must not be empty")
