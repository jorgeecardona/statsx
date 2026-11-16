"""One convergence check for MCMC output: R-hat, ESS, and divergences (BA-5).

``check_convergence`` measures; ``ConvergenceCheck.require`` refuses. Every sampler in statsx
calls ``require`` before it returns a result, so a failed fit gives an error with the reasons,
never a result.

References:
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C. (2021).
    Rank-normalization, folding, and localization: An improved R-hat for assessing convergence
    of MCMC. *Bayesian Analysis* 16(2):667-718. https://doi.org/10.1214/20-BA1221.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science*
    7(4):473-483. https://doi.org/10.1214/ss/1177011137.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

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
