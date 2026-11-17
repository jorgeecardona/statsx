"""Causal effect of an intervention on a count series, from control series (TS-7).

``causal_impact`` fits ``fit_trend`` on the months before the intervention, with the control
series as regressors. It forecasts the months after it, using the controls' values for those
months, as the counterfactual: what the series would have been without the intervention. The
effect is the observed count minus that counterfactual, as posterior draws.

The effect is a causal effect only if the controls are not changed by the intervention and
their relation to the series is the same before and after it (the assumptions of Brodersen et
al. 2015). The data cannot check either assumption. The controls get a N(0, 1) prior on their
standardized coefficients, not the spike-and-slab prior of the paper (decision D-1 limits the
latent prior to a Gaussian).

References:
    Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring
    causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*
    9(1):247-274. https://doi.org/10.1214/14-AOAS788.

    Scott, S. L., & Varian, H. R. (2014). Predicting the present with Bayesian structural time
    series. *International Journal of Mathematical Modelling and Numerical Optimisation*
    5(1-2):4-23. https://doi.org/10.1504/IJMMNO.2014.059942.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx.result import Estimate, Interval
from statsx.trend import Band, Forecast, TrendFit, fit_trend

ASSUMPTION = (
    "the controls are not changed by the intervention, and their relation to the series "
    "is the same after it as before it"
)


@dataclass(frozen=True, slots=True)
class CausalImpact:
    """The posterior of the effect of an intervention, for each month after it.

    Args:
        intervention: Index of the first month the intervention could affect.
        observed: The observed counts from the intervention on.
        counterfactual: Draws of the counts without the intervention, shape (draws, months).
        fit: The fit on the months before the intervention.
        forecast: The forecast draws of the post-intervention months.
    """

    intervention: int
    observed: npt.NDArray[np.float64]
    counterfactual: npt.NDArray[np.float64]
    fit: TrendFit
    forecast: Forecast

    @property
    def effect(self) -> npt.NDArray[np.float64]:
        """Return draws of observed minus counterfactual, shape (draws, months)."""
        return self.observed - self.counterfactual

    def band(self, level: float = 0.95) -> Band:
        """Return the central credible band of the effect in each month after the intervention.

        Raises:
            ValueError: If ``level`` is not in (0, 1).
        """
        _check_level(level)
        tail = (1.0 - level) / 2.0
        lower, upper = np.quantile(self.effect, [tail, 1.0 - tail], axis=0)
        return Band(level, lower, upper)

    def cumulative(self, level: float = 0.95) -> Estimate:
        """Return the effect summed over all months after the intervention, with its interval."""
        return self._estimate(self.effect.sum(axis=1), level)

    def average(self, level: float = 0.95) -> Estimate:
        """Return the effect per month after the intervention, with its interval."""
        return self._estimate(self.effect.mean(axis=1), level)

    def p_positive(self) -> float:
        """Return the posterior probability that the cumulative effect is above 0."""
        return float(np.mean(self.effect.sum(axis=1) > 0.0))

    def _estimate(self, draws: npt.NDArray[np.float64], level: float) -> Estimate:
        _check_level(level)
        tail = (1.0 - level) / 2.0
        lower, upper = (float(v) for v in np.quantile(draws, [tail, 1.0 - tail]))
        return Estimate(
            float(np.mean(draws)), Interval(lower, upper, level), self.observed.size, ASSUMPTION
        )


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def causal_impact(  # noqa: PLR0913 (model choices, all keyword-only)
    counts: npt.ArrayLike,
    controls: npt.ArrayLike,
    intervention: int,
    *,
    log_controls: bool = True,
    period: int = 12,
    harmonics: int = 2,
    drift_scale: float = 0.2,
    n_draws: int = 2000,
    seed: int = 0,
) -> CausalImpact:
    """Return the posterior of the effect of an intervention on ``counts``.

    Args:
        counts: Non-negative counts of the treated series, one per month.
        controls: Control series, shape (months,) or (months, controls), the same length as
            ``counts``. They must not be affected by the intervention.
        intervention: Index of the first affected month. At least ``2 * period`` months must
            come before it and at least 1 after it.
        log_controls: Use ``log(1 + control)`` as the regressor, so a control acts on the log
            mean. Set False for controls that are already on that scale.
        period: Season length in months.
        harmonics: Number of Fourier harmonics.
        drift_scale: Scale of the half-normal prior on the drift sd. A smaller value gives a
            narrower counterfactual, and more power, when the series is known to drift little.
        n_draws: Number of posterior draws, at least 400.
        seed: Seed of the random number generator.

    Returns:
        The effect for every month from ``intervention`` on.

    Raises:
        FitCheckError: If the fit on the months before the intervention fails its check.
        ValueError: If the shapes or the intervention index are out of range.
    """
    y = np.ascontiguousarray(counts, dtype=np.float64)
    x = np.asarray(controls, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size:  # noqa: PLR2004 (months x controls)
        raise ValueError("counts must be 1-D and controls must have one row per month")
    if not 2 * period <= intervention < y.size:
        raise ValueError(
            f"intervention must leave {2 * period} months before it and 1 after, got "
            f"{intervention} for {y.size} months"
        )
    if log_controls:
        if np.any(x < 0.0):
            raise ValueError("log_controls needs non-negative controls")
        x = np.log1p(x)
    fit = fit_trend(
        y[:intervention],
        regressors=x[:intervention],
        period=period,
        harmonics=harmonics,
        drift_scale=drift_scale,
        n_draws=n_draws,
        seed=seed,
    )
    horizon = y.size - intervention
    forecast = fit.forecast(horizon, regressors=x[intervention:], seed=seed + 1)
    return CausalImpact(
        intervention, y[intervention:], forecast.counts[:, -horizon:], fit, forecast
    )
