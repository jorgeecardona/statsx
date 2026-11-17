"""Sample allocation for a stratified evaluation set, for a target precision (DE-3).

A stratified sample draws a separate sample in each stratum (for example each class, language
or difficulty level) and combines them with the population weights ``W_h = N_h / N``. The
variance of the combined mean is ``sum_h W_h^2 S_h^2 (1 / n_h - 1 / N_h)``, so how the sample
is split between strata changes the precision. ``stratified_allocation`` finds the smallest
total sample size whose split reaches the target margin of error, with:

- ``"neyman"``: ``n_h`` proportional to ``N_h S_h`` (strata that are larger or more variable get
  more items). It gives the smallest variance for a fixed total. With a cost per item in each
  stratum, ``n_h`` is proportional to ``N_h S_h / sqrt(c_h)``.
- ``"proportional"``: ``n_h`` proportional to ``N_h``.

For a proportion, give the stratum proportions ``p_h`` instead of standard deviations (``S_h`` is
then ``sqrt(p_h (1 - p_h))``). Planning values are guesses: if they are off, so is the margin.

References:
    Neyman, J. (1934). On the two different aspects of the representative method: the method of
    stratified sampling and the method of purposive selection. *Journal of the Royal Statistical
    Society* 97(4):558-625. https://doi.org/10.2307/2342192.

    Cochran, W. G. (1977). *Sampling Techniques*, 3rd ed., section 5.5. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

Method = Literal["neyman", "proportional"]


@dataclass(frozen=True, slots=True)
class Allocation:
    """The sample size of each stratum and the margin of error it gives.

    Args:
        sizes: Items to draw in each stratum, in the order given.
        total: Total number of items.
        margin: The margin of error (half-width of the interval) the allocation reaches, in the
            unit of the quantity, at the stated level.
        level: Confidence level of the margin.
        method: How the sample was split.
    """

    sizes: tuple[int, ...]
    total: int
    margin: float
    level: float
    method: Method


def _variance(
    population: npt.NDArray[np.float64], sd: npt.NDArray[np.float64], n: npt.NDArray[np.float64]
) -> float:
    weights = population / population.sum()
    return float(np.sum(weights**2 * sd**2 * (1.0 / n - 1.0 / population)))


def _split(
    total: int, shares: npt.NDArray[np.float64], population: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Split ``total`` items by ``shares``, at least 1 and at most N_h in each stratum."""
    sizes = np.ones_like(shares)
    free = np.ones(shares.size, dtype=bool)
    remaining = float(total)
    for _ in range(shares.size):
        weight = shares[free] / shares[free].sum()
        trial = np.zeros_like(shares)
        trial[free] = remaining * weight
        over = free & (trial > population)
        if not over.any():
            sizes[free] = trial[free]
            break
        sizes[over] = population[over]
        remaining -= float(population[over].sum())
        free &= ~over
        if not free.any():
            break
    return np.clip(np.ceil(sizes - 1e-9), 1.0, population)


def _validated(
    sizes: npt.ArrayLike,
    sds: npt.ArrayLike | None,
    proportions: npt.ArrayLike | None,
    costs: npt.ArrayLike | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return population sizes, planning standard deviations and costs, after checking them."""
    population = np.asarray(sizes, dtype=np.float64).ravel()
    if population.size < 1 or np.any(population < 2) or not np.all(np.isfinite(population)):  # noqa: PLR2004
        raise ValueError("sizes must hold at least one stratum, each of at least 2 items")
    if (sds is None) == (proportions is None):
        raise ValueError("give either sds or proportions")
    if sds is not None:
        sd = np.asarray(sds, dtype=np.float64).ravel()
        if np.any(sd < 0.0) or not np.all(np.isfinite(sd)):
            raise ValueError("sds must be finite and >= 0")
    else:
        p = np.asarray(proportions, dtype=np.float64).ravel()
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("proportions must be in [0, 1]")
        sd = np.sqrt(p * (1.0 - p))
    if sd.size != population.size:
        raise ValueError("sizes and sds (or proportions) must have the same length")
    if not np.any(sd > 0.0):
        raise ValueError("every stratum has no variation: any sample size gives margin 0")
    cost = (
        np.ones_like(population) if costs is None else np.asarray(costs, dtype=np.float64).ravel()
    )
    if cost.size != population.size or np.any(cost <= 0.0):
        raise ValueError("costs must be > 0, one per stratum")
    return population, sd, cost


def stratified_allocation(  # noqa: PLR0913 (planning inputs, all keyword-only after the sizes)
    sizes: npt.ArrayLike,
    *,
    margin: float,
    sds: npt.ArrayLike | None = None,
    proportions: npt.ArrayLike | None = None,
    level: float = 0.95,
    method: Method = "neyman",
    costs: npt.ArrayLike | None = None,
) -> Allocation:
    """Return the smallest allocation whose margin of error is at most ``margin``.

    Args:
        sizes: Population size ``N_h`` of each stratum, at least 2 each.
        margin: Target half-width of the confidence interval of the population mean (or
            proportion), in its unit. Must be > 0.
        sds: Planning standard deviation of each stratum (give this or ``proportions``).
        proportions: Planning proportion of each stratum, in [0, 1].
        level: Confidence level, in (0, 1).
        method: ``"neyman"`` (default) or ``"proportional"``.
        costs: Cost per item in each stratum; used by ``"neyman"`` only.

    Raises:
        ValueError: If an input is out of range or inconsistent, every stratum has no variation,
            or the margin cannot be met even by sampling every item.
    """
    if not margin > 0.0 or not 0.0 < level < 1.0:
        raise ValueError("margin must be > 0 and level in (0, 1)")
    if method not in ("neyman", "proportional"):
        raise ValueError(f"method must be 'neyman' or 'proportional', got {method!r}")
    population, sd, cost = _validated(sizes, sds, proportions, costs)
    if method == "neyman":
        shares = population * np.maximum(sd, 1e-12) / np.sqrt(cost)
    else:
        shares = population.copy()
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    target = (margin / z) ** 2
    low, high = population.size, int(population.sum())
    while low < high:  # the smallest total whose allocation meets the target
        mid = (low + high) // 2
        if _variance(population, sd, _split(mid, shares, population)) <= target:
            high = mid
        else:
            low = mid + 1
    allocated = _split(low, shares, population)
    achieved = z * math.sqrt(_variance(population, sd, allocated))
    return Allocation(
        tuple(int(v) for v in allocated), int(allocated.sum()), achieved, level, method
    )
