"""Bootstrap with a named resample unit (ES-2), and a percentile interval.

The resample unit is part of the result, so a reader can see what was resampled.
Resample indices come from the Rust core; the statistic runs in Python on NumPy arrays.

References:
    Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of
    Statistics* 7(1):1-26. https://doi.org/10.1214/aos/1176344552.

    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
    https://doi.org/10.1201/9780429246593. Percentile interval.

    Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*,
    chapter 3. Cambridge University Press. https://doi.org/10.1017/CBO9780511802843.
    Stratified and clustered resampling.

    Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations.
    *Annals of Statistics* 17(3):1217-1241. https://doi.org/10.1214/aos/1176347265.
    Moving block resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A bootstrap estimate, the resample unit it used, and the replicate statistics."""

    estimate: Estimate
    unit: ResampleUnit
    replicates: npt.NDArray[np.float64] = field(repr=False)


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
    seed: int = 0,
) -> BootstrapResult:
    """Return the statistic on ``data`` with a percentile interval from bootstrap replicates.

    Args:
        data: Observations along the first axis.
        statistic: Maps an array of observations (same layout as ``data``) to a number.
        unit: What is resampled: ``IID()``, ``Stratified(labels)``, ``Clustered(labels)``, or
            ``Block(length)``. It is recorded in the result.
        n_resamples: Number of bootstrap replicates.
        level: Interval coverage in (0, 1).
        seed: Seed of the Rust generator; the same seed gives the same result.

    Raises:
        ValueError: If ``data`` is empty, labels do not match the data length, the block length
            is outside [1, n], or there are fewer than 2 clusters.
    """
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
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
    return BootstrapResult(
        estimate=Estimate(
            estimate=float(statistic(values)),
            interval=Interval(float(lower), float(upper), level),
            n=n,
            assumption=unit.describe(),
        ),
        unit=unit,
        replicates=replicates,
    )
