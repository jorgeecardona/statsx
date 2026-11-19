"""Convergence checks for MCMC output and the Monte Carlo error of its summaries.

``check_convergence`` gives R-hat, ESS and divergences in one check.

``check_convergence`` measures; ``ConvergenceCheck.require`` refuses. Every sampler in statsx
calls ``require`` before it returns a result, so a failed fit gives an error with the reasons,
never a result.

The Monte Carlo standard error (MCSE) is the error in a posterior summary that comes from using a
finite, autocorrelated sample of draws. ``mcse_mean``, ``mcse_sd`` and ``mcse_quantile`` give it
for the posterior mean, standard deviation and a quantile, each from the effective sample size of
the quantity it summarizes. Report a number to the digit the MCSE supports.

References:
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C. (2021).
    Rank-normalization, folding, and localization: An improved R-hat for assessing convergence
    of MCMC. *Bayesian Analysis* 16(2):667-718. https://doi.org/10.1214/20-BA1221.

    Kenney, J. F., & Keeping, E. S. (1951). *Mathematics of Statistics, Part Two*, 2nd ed., p. 141.
    Van Nostrand. The delta-method variance of a sample standard deviation.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science*
    7(4):473-483. https://doi.org/10.1214/ss/1177011137.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from statsx import _accel

RHAT_MAX = 1.01
"""Vehtari et al. (2021) recommend R-hat below 1.01."""

ESS_MIN = 400.0
"""Vehtari et al. (2021) recommend bulk and tail ESS above 100 per chain for 4 chains."""


@dataclass(frozen=True, slots=True)
class ParameterDiagnostics:
    """R-hat, bulk ESS and tail ESS of one parameter."""

    name: str
    rhat: float
    ess_bulk: float
    ess_tail: float


@dataclass(frozen=True, slots=True)
class ConvergenceCheck:
    """The result of one convergence check; ``passed`` is False when any reason is listed."""

    parameters: tuple[ParameterDiagnostics, ...]
    divergences: int
    rhat_max: float
    ess_min: float
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return True when no threshold failed."""
        return not self.reasons

    def require(self) -> None:
        """Raise ``ConvergenceError`` when the check failed; return None otherwise."""
        if self.reasons:
            raise ConvergenceError(self)


class ConvergenceError(RuntimeError):
    """Raised when MCMC output fails the convergence check; ``check`` holds the numbers."""

    def __init__(self, check: ConvergenceCheck) -> None:
        super().__init__("MCMC did not converge: " + "; ".join(check.reasons))
        self.check = check


def check_convergence(
    draws: Mapping[str, npt.ArrayLike],
    *,
    divergences: int = 0,
    rhat_max: float = RHAT_MAX,
    ess_min: float = ESS_MIN,
) -> ConvergenceCheck:
    """Return R-hat, bulk ESS and tail ESS for each parameter, and the reasons it fails.

    Args:
        draws: Parameter name to an array of shape (chains, draws). Use at least 2 chains:
            with 1 chain, split R-hat compares only its two halves.
        divergences: Number of divergent transitions the sampler reported. Any is a failure.
        rhat_max: The check fails when R-hat is above this, or is NaN.
        ess_min: The check fails when bulk or tail ESS is below this, or is NaN.

    Raises:
        ValueError: If an array is not 2-D or has fewer than 8 draws per chain.
    """
    params: list[ParameterDiagnostics] = []
    reasons: list[str] = []
    for name, value in draws.items():
        array: npt.NDArray[np.float64] = np.ascontiguousarray(value, dtype=np.float64)
        if array.ndim != 2:  # noqa: PLR2004 (chains x draws)
            raise ValueError(f"{name}: draws must have shape (chains, draws), got {array.shape}")
        rhat, bulk, tail = _accel.convergence(array)
        params.append(ParameterDiagnostics(name, rhat, bulk, tail))
        if math.isnan(rhat) or rhat > rhat_max:
            reasons.append(f"{name}: R-hat {rhat:.3f} above {rhat_max}")
        if math.isnan(bulk) or bulk < ess_min:
            reasons.append(f"{name}: bulk ESS {bulk:.0f} below {ess_min:.0f}")
        if math.isnan(tail) or tail < ess_min:
            reasons.append(f"{name}: tail ESS {tail:.0f} below {ess_min:.0f}")
    if divergences > 0:
        reasons.append(f"{divergences} divergent transitions")
    return ConvergenceCheck(tuple(params), divergences, rhat_max, ess_min, tuple(reasons))


def _chains(draws: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array: npt.NDArray[np.float64] = np.ascontiguousarray(draws, dtype=np.float64)
    if array.ndim != 2:  # noqa: PLR2004 (chains x draws)
        raise ValueError(f"draws must have shape (chains, draws), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("draws must be finite")
    return array


def ess_mean(draws: npt.ArrayLike) -> float:
    """Return the effective sample size of the mean of the draws, as they are (not rank-normalized).

    Args:
        draws: Shape (chains, draws), at least 8 draws per chain. Chains are split in half.

    Returns:
        The ESS, or NaN when every draw is equal.

    Raises:
        ValueError: If ``draws`` is not 2-D, has fewer than 8 draws per chain, or is not finite.
    """
    return float(_accel.effective_sample_size(_chains(draws)))


def mcse_mean(draws: npt.ArrayLike) -> float:
    """Return the Monte Carlo standard error of the posterior mean: ``sd / sqrt(ESS)``.

    Raises:
        ValueError: If ``draws`` is not 2-D, has fewer than 8 draws per chain, or is not finite.
    """
    x = _chains(draws)
    return float(np.std(x, ddof=1) / math.sqrt(ess_mean(x)))


def mcse_sd(draws: npt.ArrayLike) -> float:
    """Return the Monte Carlo standard error of the posterior standard deviation.

    The variance of the sample variance comes from the fourth central moment and the ESS of the
    squared deviations, and the delta method turns it into a standard error of the standard
    deviation: ``Var(s) = Var(s^2) / (4 s^2)`` (Kenney & Keeping 1951).

    Raises:
        ValueError: If ``draws`` is not 2-D, has fewer than 8 draws per chain, or is not finite.
    """
    x = _chains(draws)
    squared = (x - x.mean()) ** 2
    variance = float(squared.mean())
    ess = ess_mean(squared)
    var_of_variance = (float(np.mean(squared**2)) - variance**2) / ess
    return math.sqrt(var_of_variance / variance / 4.0)


def mcse_quantile(draws: npt.ArrayLike, prob: float) -> float:
    """Return the Monte Carlo standard error of the posterior quantile at ``prob``.

    The ESS is that of the indicator ``draw <= quantile``. With it, the quantile level has a
    Beta(``ESS * prob + 1``, ``ESS * (1 - prob) + 1``) distribution, and the standard error is half
    the distance between the sorted draws at its 16th and 84th percentiles (the one-standard-
    deviation levels of a normal).

    Raises:
        ValueError: If ``prob`` is not in (0, 1), or ``draws`` is not 2-D, has fewer than 8 draws
            per chain, or is not finite.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    x = _chains(draws)
    flat = np.sort(x.ravel())
    cut = float(np.quantile(flat, prob))
    ess = ess_mean((x <= cut).astype(np.float64))
    if math.isnan(ess):
        return math.nan
    levels = np.array([NormalDist().cdf(-1.0), NormalDist().cdf(1.0)])
    a, b = _accel.beta_ppf(levels, ess * prob + 1.0, ess * (1.0 - prob) + 1.0)
    low = max(math.floor(a * flat.size) - 1, 0)
    high = min(math.ceil(b * flat.size) - 1, flat.size - 1)
    return float((flat[high] - flat[low]) / 2.0)
