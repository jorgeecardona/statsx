"""Updated ETA of a running task from its elapsed time and its progress (ET-3).

A task reports progress at some times: for example files done out of the files to do, so the
fraction ``f`` done at elapsed time ``e``. If the task ran at a constant rate, the time left
would be ``e * (1 - f) / f``. Real tasks start slowly, speed up, or have a slow last part, so
the model learns from past runs how the true time left relates to that extrapolation:

    log( left * f / (e * (1 - f)) ) = b0 + b1 * log(f) + noise

The fit uses the Normal-Inverse-Gamma regression of ``statsx.eta`` (decision D-7). The time left
is the constant-rate extrapolation times the exponential of the predictive draw, so it is
always positive and its interval comes from a Student-t on the log scale.

References:
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
    *Bayesian Data Analysis*, 3rd ed., chapter 14. CRC Press. https://doi.org/10.1201/b16018.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.eta import DurationPrior
from statsx.result import Estimate, Interval


@dataclass(frozen=True, slots=True)
class ProgressModel:
    """What past runs say about the time left, given the progress made.

    Args:
        n_runs: Number of past runs used.
        n_checkpoints: Number of progress reports used, over all runs.
    """

    coef_mean: npt.NDArray[np.float64] = field(repr=False)
    coef_cov: npt.NDArray[np.float64] = field(repr=False)
    noise_shape: float
    noise_rate: float
    n_runs: int
    n_checkpoints: int

    def remaining(
        self, elapsed: float, fraction: float, *, level: float = 0.9, seed: int = 0
    ) -> Estimate:
        """Return the time left (posterior predictive median) and its interval.

        Args:
            elapsed: Time since the task started, in the unit of the past runs. Must be > 0.
            fraction: Progress made, strictly between 0 and 1 (for example files done / files).
            level: Credible level of the interval.
            seed: Seed of the predictive draws; the same seed gives the same interval.

        Raises:
            ValueError: If ``elapsed`` is not positive or ``fraction`` is not in (0, 1).
        """
        if not (math.isfinite(elapsed) and elapsed > 0.0):
            raise ValueError("elapsed must be finite and > 0")
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
        row = np.array([[1.0, math.log(fraction)]])
        draws = _accel.nig_predict(
            self.coef_mean, self.coef_cov, self.noise_shape, self.noise_rate, row, 20_000, seed
        )[0]
        constant_rate = elapsed * (1.0 - fraction) / fraction
        tail = (1.0 - level) / 2.0
        lower, median, upper = np.quantile(draws, [tail, 0.5, 1.0 - tail])
        return Estimate(
            estimate=constant_rate * math.exp(median),
            interval=Interval(
                constant_rate * math.exp(lower), constant_rate * math.exp(upper), level
            ),
            n=self.n_checkpoints,
            assumption=(
                "the time left relates to the constant-rate extrapolation as in the past runs: "
                "log ratio linear in log(progress), normal noise; reports of one run are "
                "treated as independent"
            ),
        )


def fit_progress(
    durations: npt.ArrayLike,
    checkpoints: Sequence[Sequence[tuple[float, float]]],
    *,
    prior: DurationPrior | None = None,
) -> ProgressModel:
    """Return a progress model fitted to past runs and the progress they reported.

    Args:
        durations: Total duration of each past run.
        checkpoints: For each run, its progress reports as (elapsed time, fraction done). A
            report is used only if it is before the end of the run and its fraction is in
            (0, 1).
        prior: The prior of the regression; the default is weak.

    Raises:
        ValueError: If the lengths differ, a duration is not positive, or no report is usable.
    """
    prior = prior or DurationPrior()
    total = np.asarray(durations, dtype=np.float64)
    if total.ndim != 1 or total.size != len(checkpoints):
        raise ValueError(f"{total.size} durations for {len(checkpoints)} runs of checkpoints")
    if np.any(total <= 0.0) or not np.all(np.isfinite(total)):
        raise ValueError("durations must be finite and > 0")
    rows: list[tuple[float, float]] = []
    runs_used = 0
    for duration, reports in zip(total, checkpoints, strict=True):
        usable = [
            (e, f) for e, f in reports if 0.0 < e < duration and 0.0 < f < 1.0 and math.isfinite(e)
        ]
        runs_used += bool(usable)
        for e, f in usable:
            ratio = (duration - e) * f / (e * (1.0 - f))
            rows.append((math.log(f), math.log(ratio)))
    if not rows:
        raise ValueError(
            "no usable progress report: need 0 < elapsed < duration and 0 < fraction < 1"
        )
    data = np.array(rows, dtype=np.float64)
    design = np.ascontiguousarray(np.column_stack([np.ones(len(rows)), data[:, 0]]))
    mean, cov, shape, rate = _accel.nig_fit(
        design,
        np.ascontiguousarray(data[:, 1]),
        prior.coef_sd**2,
        prior.noise_shape,
        prior.noise_scale,
    )
    return ProgressModel(mean, cov, shape, rate, runs_used, len(rows))
