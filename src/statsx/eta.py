"""Task duration from input parameters, with a credible interval (ET-1, ET-2).

The model is ``log(duration) = b0 + sum_k b_k * log(1 + x_k) + e``, with normal noise ``e``.
A coefficient of 1 on ``log(1 + files)`` means duration grows in proportion to the file count.
The prior is the conjugate Normal-Inverse-Gamma, so the fit is closed form; the predictive
distribution is Student-t on the log scale, drawn exactly by the Rust core.

References:
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
    *Bayesian Data Analysis*, 3rd ed., chapter 14. CRC Press. https://doi.org/10.1201/b16018.

    O'Hagan, A., & Forster, J. (2004). *Kendall's Advanced Theory of Statistics, Vol. 2B:
    Bayesian Inference*, 2nd ed., chapter 11. Arnold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.result import Estimate, Interval


@dataclass(frozen=True, slots=True)
class DurationPrior:
    """Normal-Inverse-Gamma prior on the log-scale regression.

    Args:
        coef_sd: Prior sd of each coefficient, in units of the noise sd. 10 is weak.
        noise_shape: Inverse-Gamma shape of the noise variance.
        noise_scale: Inverse-Gamma scale of the noise variance. With shape 2 and scale 0.5 the
            prior mean noise variance is 0.5 (a typical run within a factor of about 2).
    """

    coef_sd: float = 10.0
    noise_shape: float = 2.0
    noise_scale: float = 0.5


@dataclass(frozen=True, slots=True)
class DurationModel:
    """A fitted duration model; ``predict`` gives the duration of a new run."""

    parameters: tuple[str, ...]
    coef_mean: npt.NDArray[np.float64] = field(repr=False)
    coef_cov: npt.NDArray[np.float64] = field(repr=False)
    noise_shape: float
    noise_rate: float
    n: int

    @property
    def coefficients(self) -> dict[str, float]:
        """Return the posterior mean of each coefficient, ``"intercept"`` first."""
        names = ("intercept", *self.parameters)
        return {k: float(v) for k, v in zip(names, self.coef_mean, strict=True)}

    def predict(
        self, params: Mapping[str, float], *, level: float = 0.9, seed: int = 0
    ) -> Estimate:
        """Return the predicted duration (posterior predictive median) and its interval.

        Args:
            params: A value for every parameter the model was fitted with.
            level: Credible level of the interval.
            seed: Seed of the predictive draws; the same seed gives the same interval.

        Raises:
            ValueError: If a parameter is missing or negative.
        """
        missing = set(self.parameters) - set(params)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        row = (
            _design([[params[k]] for k in self.parameters])
            if self.parameters
            else np.ones((1, 1), dtype=np.float64)
        )
        draws = _accel.nig_predict(
            self.coef_mean,
            self.coef_cov,
            self.noise_shape,
            self.noise_rate,
            row,
            20_000,
            seed,
        )[0]
        tail = (1.0 - level) / 2.0
        lower, median, upper = np.quantile(draws, [tail, 0.5, 1.0 - tail])
        return Estimate(
            estimate=math.exp(median),
            interval=Interval(math.exp(lower), math.exp(upper), level),
            n=self.n,
            assumption=(
                "log duration linear in log(1 + x) of "
                + ", ".join(self.parameters)
                + "; normal noise; Normal-Inverse-Gamma prior"
            ),
        )


def _design(columns: Sequence[npt.ArrayLike]) -> npt.NDArray[np.float64]:
    values = np.asarray(columns, dtype=np.float64).reshape(len(columns), -1)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite and >= 0")
    return np.column_stack([np.ones(values.shape[1]), np.log1p(values.T)])


def fit_duration(
    durations: npt.ArrayLike,
    params: Mapping[str, npt.ArrayLike],
    *,
    prior: DurationPrior | None = None,
) -> DurationModel:
    """Return a duration model fitted to past runs.

    Args:
        durations: Duration of each past run, in any unit; the predictions use the same unit.
        params: Parameter name to its value in each past run, same length as ``durations``.
            Values must be >= 0 (counts, sizes). An empty mapping fits the duration alone.
        prior: The prior; the default is weak.

    Raises:
        ValueError: If a duration is not positive, the lengths differ, or there are no runs.
    """
    prior = prior or DurationPrior()
    y = np.asarray(durations, dtype=np.float64)
    if y.size == 0:
        raise ValueError("need at least one past run")
    if np.any(y <= 0) or not np.all(np.isfinite(y)):
        raise ValueError("durations must be finite and > 0")
    names = tuple(params)
    columns = [np.asarray(params[k], dtype=np.float64).ravel() for k in names]
    for name, column in zip(names, columns, strict=True):
        if column.size != y.size:
            raise ValueError(f"{name} has {column.size} values for {y.size} runs")
    x = _design(columns) if names else np.ones((y.size, 1), dtype=np.float64)
    mean, cov, shape, rate = _accel.nig_fit(
        np.ascontiguousarray(x),
        np.log(y),
        prior.coef_sd**2,
        prior.noise_shape,
        prior.noise_scale,
    )
    return DurationModel(names, mean, cov, shape, rate, int(y.size))
