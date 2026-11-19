"""Bootstrap with a named resample unit and a percentile, basic or BCa interval.

The resample unit is part of the result, so a reader can see what was resampled.
Resample indices come from the Rust core; the statistic runs in Python on NumPy arrays.

References:
    Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of
    Statistics* 7(1):1-26. https://doi.org/10.1214/aos/1176344552.

    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
    https://doi.org/10.1201/9780429246593. Percentile interval, and the jackknife acceleration of
    the BCa interval (chapter 14).

    Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American
    Statistical Association* 82(397):171-185. https://doi.org/10.1080/01621459.1987.10478410.
    BCa interval.

    Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*,
    chapters 3 and 5. Cambridge University Press. https://doi.org/10.1017/CBO9780511802843.
    Stratified and clustered resampling (chapter 3); the basic interval (chapter 5).

    Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations.
    *Annals of Statistics* 17(3):1217-1241. https://doi.org/10.1214/aos/1176347265.
    Moving block resampling.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.result import Estimate, Interval


@dataclass(frozen=True, slots=True)
class IID:
    """Resample single observations with replacement."""

    def describe(self) -> str:
        """Return the assumption in words."""
        return "i.i.d. resampling of observations"


@dataclass(frozen=True, slots=True)
class Stratified:
    """Resample observations with replacement within each stratum, keeping stratum sizes."""

    labels: Sequence[int] = field(repr=False)

    def describe(self) -> str:
        """Return the assumption in words."""
        return f"stratified resampling within {len(set(self.labels))} strata"


@dataclass(frozen=True, slots=True)
class Clustered:
    """Resample whole clusters with replacement; observations in a cluster stay together."""

    labels: Sequence[int] = field(repr=False)

    def describe(self) -> str:
        """Return the assumption in words."""
        return f"resampling of {len(set(self.labels))} clusters, observations kept together"


@dataclass(frozen=True, slots=True)
class Block:
    """Moving block resampling of a series: blocks of ``length`` consecutive observations."""

    length: int

    def describe(self) -> str:
        """Return the assumption in words."""
        return f"moving block resampling, block length {self.length}"


ResampleUnit = IID | Stratified | Clustered | Block


IntervalMethod = Literal["percentile", "basic", "bca"]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A bootstrap estimate, the resample unit and interval method it used, and the replicates."""

    estimate: Estimate
    unit: ResampleUnit
    replicates: npt.NDArray[np.float64] = field(repr=False)
    method: IntervalMethod = "percentile"


def _kind(unit: ResampleUnit) -> tuple[str, npt.NDArray[np.int64] | None, int]:
    match unit:
        case IID():
            return "iid", None, 0
        case Stratified(labels=labels):
            return "stratified", np.asarray(labels, dtype=np.int64), 0
        case Clustered(labels=labels):
            return "clustered", np.asarray(labels, dtype=np.int64), 0
        case Block(length=length):
            return "block", None, length


def bootstrap(  # noqa: PLR0913 (the last three are keyword-only settings)
    data: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64]], float],
    unit: ResampleUnit,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    method: IntervalMethod = "percentile",
    seed: int = 0,
) -> BootstrapResult:
    """Return the statistic on ``data`` with a bootstrap interval from the replicates.

    Args:
        data: Observations along the first axis.
        statistic: Maps an array of observations (same layout as ``data``) to a number.
        unit: What is resampled: ``IID()``, ``Stratified(labels)``, ``Clustered(labels)``, or
            ``Block(length)``. It is recorded in the result.
        n_resamples: Number of bootstrap replicates.
        level: Interval coverage in (0, 1).
        method: ``"percentile"``: the quantiles of the replicates. ``"basic"``: the quantiles
            reflected around the estimate, 2 t - q(1 - a/2) to 2 t - q(a/2). ``"bca"``:
            percentiles corrected for bias and skew (acceleration by the jackknife); it needs
            ``IID`` or ``Clustered`` resampling, since the jackknife drops one observation or
            one cluster.
        seed: Seed of the Rust generator; the same seed gives the same result.

    Raises:
        ValueError: If ``data`` is empty, labels do not match the data length, the block length
            is outside [1, n], there are fewer than 2 clusters, ``level`` is not in (0, 1),
            ``method`` is unknown, or ``"bca"`` is asked with ``Stratified`` or ``Block``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if method not in ("percentile", "basic", "bca"):
        raise ValueError(f"method must be 'percentile', 'basic' or 'bca', got {method!r}")
    if method == "bca" and isinstance(unit, Stratified | Block):
        raise ValueError(
            "bca needs a jackknife over observations or clusters: use IID or Clustered, "
            f"not {type(unit).__name__}"
        )
    values = np.asarray(data, dtype=np.float64)
    n = values.shape[0] if values.ndim > 0 else 0
    kind, labels, block_length = _kind(unit)
    indices, offsets = _accel.bootstrap_indices(
        n, kind, n_resamples, seed, labels=labels, block_length=block_length
    )
    replicates = np.array(
        [statistic(values[indices[offsets[r] : offsets[r + 1]]]) for r in range(n_resamples)],
        dtype=np.float64,
    )
    estimate = float(statistic(values))
    lower, upper = _interval(
        replicates,
        estimate,
        level=level,
        method=method,
        values=values,
        statistic=statistic,
        unit=unit,
    )
    return BootstrapResult(
        estimate=Estimate(
            estimate=estimate,
            interval=Interval(lower, upper, level),
            n=n,
            assumption=f"{unit.describe()}; {method} interval",
        ),
        unit=unit,
        replicates=replicates,
        method=method,
    )


def _interval(  # noqa: PLR0913 (all the inputs of one interval, keyword-only)
    replicates: npt.NDArray[np.float64],
    estimate: float,
    *,
    level: float,
    method: IntervalMethod,
    values: npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], float],
    unit: ResampleUnit,
) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    match method:
        case "percentile":
            lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
        case "basic":
            low, high = np.quantile(replicates, [tail, 1.0 - tail])
            lower, upper = 2.0 * estimate - high, 2.0 * estimate - low
        case "bca":
            normal = NormalDist()
            below = (
                np.count_nonzero(replicates < estimate)
                + 0.5 * np.count_nonzero(replicates == estimate)
            ) / replicates.size
            # Keep the proportion inside (0, 1): z0 is infinite when no replicate is on one side.
            half = 0.5 / float(replicates.size)
            z0: float = normal.inv_cdf(min(max(below, half), 1.0 - half))
            a = _acceleration(values, statistic, unit)
            probs = [
                normal.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
                for z in (normal.inv_cdf(tail), normal.inv_cdf(1.0 - tail))
            ]
            lower, upper = np.quantile(replicates, probs)
    return float(lower), float(upper)


def _acceleration(
    values: npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], float],
    unit: ResampleUnit,
) -> float:
    """Return the BCa acceleration from the jackknife, dropping one observation or cluster."""
    n = values.shape[0]
    drops: list[npt.NDArray[np.int64]]
    if isinstance(unit, Clustered):
        labels = np.asarray(unit.labels)
        drops = [np.flatnonzero(labels != c).astype(np.int64) for c in np.unique(labels)]
    else:
        everyone = np.arange(n, dtype=np.int64)
        drops = [np.delete(everyone, i) for i in range(n)]
    jack = np.array([statistic(values[keep]) for keep in drops], dtype=np.float64)
    centred = jack.mean() - jack
    denominator = 6.0 * math.pow(float(np.sum(centred**2)), 1.5)
    return 0.0 if denominator == 0.0 else float(np.sum(centred**3) / denominator)
