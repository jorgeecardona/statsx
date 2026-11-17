"""Summaries of posterior draws (ES-4), and the comparison of two posteriors (BA-2).

Each summary is one function. They work on any sample of draws of one quantity. ``compare``
gives P(A > B) and the probability of practical equivalence for two independent posteriors.

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
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from statsx.equivalence import EquivalenceMargin, as_margin
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


@runtime_checkable
class Sampler(Protocol):
    """A posterior that can be sampled: ``Beta``, ``Gamma``, ``Normal`` and ``LogNormal``."""

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws."""
        ...


@dataclass(frozen=True, slots=True)
class Comparison:
    """P(A > B), P(B > A), the probability of practical equivalence, and the difference.

    Args:
        p_a_greater: Probability that A is above B.
        p_b_greater: Probability that B is above A.
        p_equivalent: Probability that the difference A - B lies within ``margin`` of 0.
        margin: The practical equivalence margin that was used (see ``statsx.equivalence``).
        difference: Equal-tailed credible interval of A - B.
        n_draws: Number of paired draws behind the probabilities.
        mc_error: The largest Monte Carlo standard error of the three probabilities.
    """

    p_a_greater: float
    p_b_greater: float
    p_equivalent: float
    margin: EquivalenceMargin
    difference: Interval
    n_draws: int
    mc_error: float


def _posterior_draws(
    value: Sampler | npt.ArrayLike, n_draws: int, seed: int
) -> npt.NDArray[np.float64]:
    if isinstance(value, Sampler):
        return value.sample(n_draws, seed=seed)
    return _draws(value)


def compare(  # noqa: PLR0913 (two posteriors and the settings of the comparison)
    a: Sampler | npt.ArrayLike,
    b: Sampler | npt.ArrayLike,
    *,
    margin: EquivalenceMargin | float,
    level: float = 0.95,
    n_draws: int = 100_000,
    seed: int = 0,
) -> Comparison:
    """Compare two independent posteriors: P(A > B) and the probability of practical equivalence.

    Each posterior is either a distribution with a ``sample`` method (``Beta``, ``Gamma``,
    ``Normal``, ``LogNormal``) or an array of its draws. The two are treated as independent:
    draw ``i`` of A is paired with draw ``i`` of B, so the draws of one posterior must not be
    sorted or ordered together with the other. The probabilities are Monte Carlo estimates.

    Args:
        a: The first posterior.
        b: The second posterior.
        margin: Practical equivalence: A and B are equivalent when ``A - B`` is inside the
            margin. An ``EquivalenceMargin`` (the same object ``statsx.equivalence.tost`` takes),
            or a number ``d`` for ``(-d, d)``, in the unit of the quantity.
        level: Coverage of the credible interval of A - B.
        n_draws: Draws taken from a distribution; ignored for an array, which gives its length.
        seed: Seed of the draws (B uses ``seed + 1``).

    Raises:
        ValueError: If the margin is not valid, ``level`` is not in (0, 1), an array has fewer
            than 2 draws, or ``n_draws`` is below 2.
    """
    region = as_margin(margin)
    _level(level)
    if n_draws < 2:  # noqa: PLR2004
        raise ValueError("n_draws must be at least 2")
    x = _posterior_draws(a, n_draws, seed)
    y = _posterior_draws(b, n_draws, seed + 1)
    n = min(x.size, y.size)
    diff = x[:n] - y[:n]
    p_a = float(np.mean(diff > 0.0))
    p_b = float(np.mean(diff < 0.0))
    p_eq = float(np.mean((diff >= region.low) & (diff <= region.high)))
    error = max(math.sqrt(p * (1.0 - p) / n) for p in (p_a, p_b, p_eq))
    return Comparison(p_a, p_b, p_eq, region, credible_interval(diff, level), n, error)
