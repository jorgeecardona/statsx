"""Summaries of posterior draws: quantiles, equal-tailed credible interval, HDI (ES-4).

Each summary is one function. They work on any sample of draws of one quantity.

References:
    Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. *The American
    Statistician* 50(4):361-365. https://doi.org/10.1080/00031305.1996.10473566. Quantiles are
    their type 7: linear interpolation between order statistics.

    Chen, M.-H., & Shao, Q.-M. (1999). Monte Carlo estimation of Bayesian credible and HPD
    intervals. *Journal of Computational and Graphical Statistics* 8(1):69-92.
    https://doi.org/10.1080/10618600.1999.10474802. The HDI is the shortest interval holding
    the stated share of the sorted draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx.result import Interval


@dataclass(frozen=True, slots=True)
class Quantiles:
    """Quantiles of a sample: ``values[i]`` is the quantile at probability ``probs[i]``."""

    probs: tuple[float, ...]
    values: tuple[float, ...]

    def __getitem__(self, prob: float) -> float:
        """Return the quantile at ``prob``, which must be one of ``probs``."""
        return self.values[self.probs.index(prob)]


def _draws(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):  # noqa: PLR2004
        raise ValueError("draws must be a 1-D array of at least 2 finite values")
    return x


def _level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def quantiles(
    draws: npt.ArrayLike, probs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> Quantiles:
    """Return the quantiles of ``draws`` at ``probs``.

    Raises:
        ValueError: If ``draws`` is not a 1-D array of at least 2 finite values, or a
            probability is outside [0, 1].
    """
    x = _draws(draws)
    if not probs or any(not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError(f"probs must be in [0, 1], got {probs}")
    return Quantiles(tuple(probs), tuple(float(v) for v in np.quantile(x, probs)))


def credible_interval(draws: npt.ArrayLike, level: float = 0.95) -> Interval:
    """Return the equal-tailed credible interval: the same share of draws below and above it.

    Raises:
        ValueError: If ``level`` is not in (0, 1) or ``draws`` is not usable.
    """
    _level(level)
    x = _draws(draws)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(x, [tail, 1.0 - tail])
    return Interval(float(lower), float(upper), level)


def hdi(draws: npt.ArrayLike, level: float = 0.95) -> Interval:
    """Return the highest density interval: the shortest interval holding ``level`` of the draws.

    The interval is a single interval. For a sample with several modes it can contain a gap of
    low density; the sample then needs a set of intervals, which this function does not give.

    Raises:
        ValueError: If ``level`` is not in (0, 1) or ``draws`` is not usable.
    """
    _level(level)
    x = np.sort(_draws(draws))
    inside = max(1, min(x.size - 1, math.ceil(level * x.size)))
    widths = x[inside:] - x[: x.size - inside]
    start = int(np.argmin(widths))
    return Interval(float(x[start]), float(x[start + inside]), level)
