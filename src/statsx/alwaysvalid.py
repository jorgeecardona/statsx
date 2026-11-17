"""Always-valid inference: confidence sequences and e-values, so you can stop whenever (SQ-1).

A usual 95% interval is valid at one planned sample size. If you look at it after every
observation and stop when it excludes a value, the chance of a false stop is far above 5%. The
tools here stay valid at every sample size at once, so you may watch continuously and stop early
without inflating the error rate:

- ``confidence_sequence`` gives, after each observation, an interval for the mean. With
  probability at least ``1 - alpha`` the true mean is inside the interval at every observation
  at once.
- ``e_process`` gives an e-value for the claim "the mean is ``mu0``": the evidence against it,
  as a product of likelihood ratios that has expected value at most 1 when the claim is true.
  The claim can be rejected at level ``alpha`` as soon as the e-value reaches ``1 / alpha``
  (Ville's inequality). The two agree: ``mu0`` is outside the confidence sequence exactly when
  its e-value is at least ``1 / alpha``.

The method is the normal mixture of Robbins (1970) for observations whose deviations from the
mean are sub-Gaussian with variance proxy ``sigma^2``: for normal data, ``sigma`` is the standard
deviation; for data in [0, 1] (such as 0/1 outcomes), ``sigma = 0.5`` is valid and conservative.
``sigma`` must be given: it is an assumption, not estimated. The price of always being valid is a
wider interval than a fixed-sample one at any one size.

References:
    Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm.
    *Annals of Mathematical Statistics* 41(5):1397-1409. https://doi.org/10.1214/aoms/1177696786.

    Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform, nonparametric,
    nonasymptotic confidence sequences. *Annals of Statistics* 49(2):1055-1080.
    https://doi.org/10.1214/20-AOS1991.

    Ville, J. (1939). *Etude critique de la notion de collectif*. Gauthier-Villars. The
    inequality P(sup of a non-negative supermartingale >= 1 / alpha) <= alpha.

    Ramdas, A., Grunwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe
    anytime-valid inference. *Statistical Science* 38(4):576-601.
    https://doi.org/10.1214/23-STS894.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    """A confidence sequence for a mean: one interval after each observation.

    Args:
        n: The number of observations at each step, 1 to the sample size.
        mean: The running mean.
        lower: Lower end of the interval at each step.
        upper: Upper end of the interval at each step.
        alpha: The error rate: all intervals hold the true mean with probability ``1 - alpha``.
        sigma: The sub-Gaussian scale that was assumed.
    """

    n: npt.NDArray[np.int64]
    mean: npt.NDArray[np.float64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    alpha: float
    sigma: float

    def excludes(self, value: float) -> npt.NDArray[np.bool_]:
        """Return, at each step, whether ``value`` is outside the interval."""
        return (value < self.lower) | (value > self.upper)

    def first_exclusion(self, value: float) -> int | None:
        """Return the first sample size at which ``value`` is outside the interval, or None."""
        hit = np.flatnonzero(self.excludes(value))
        return int(self.n[hit[0]]) if hit.size else None


@dataclass(frozen=True, slots=True)
class EProcess:
    """The e-value of the claim "the mean is ``mu0``" after each observation.

    Args:
        n: The number of observations at each step.
        e_value: The e-value at each step; large values are evidence against the claim.
        mu0: The claimed mean.
        alpha: The level at which the claim is rejected once ``e_value >= 1 / alpha``.
    """

    n: npt.NDArray[np.int64]
    e_value: npt.NDArray[np.float64]
    mu0: float
    alpha: float

    def rejected(self) -> npt.NDArray[np.bool_]:
        """Return, at each step, whether the claim can be rejected at level ``alpha``."""
        return self.e_value >= 1.0 / self.alpha

    def first_rejection(self) -> int | None:
        """Return the first sample size at which the claim is rejected, or None."""
        hit = np.flatnonzero(self.rejected())
        return int(self.n[hit[0]]) if hit.size else None


def _tuned_rho(sigma: float, alpha: float, n_target: float) -> float:
    """Return the mixture precision that gives the tightest interval near ``n_target``."""
    log_term = 2.0 * math.log(1.0 / alpha)
    return n_target * sigma**2 / (log_term + math.log(1.0 + log_term))


def _check(observations: npt.ArrayLike, sigma: float, alpha: float) -> npt.NDArray[np.float64]:
    x = np.asarray(observations, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a non-empty array of finite numbers")
    if not sigma > 0.0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and > 0, got {sigma}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return x


def confidence_radius(
    n: npt.ArrayLike, *, sigma: float, alpha: float = 0.05, n_target: float = 100.0
) -> npt.NDArray[np.float64]:
    """Return the half-width of the confidence sequence after ``n`` observations.

    The interval for the mean is the running mean plus or minus this radius. It depends only on
    ``n``, so a caller that keeps a running total can build the interval without the data.

    Raises:
        ValueError: If ``n`` is below 1, ``sigma`` is not positive, ``alpha`` is not in (0, 1),
            or ``n_target`` is not positive.
    """
    count = np.asarray(n, dtype=np.float64)
    if np.any(count < 1.0) or not np.all(np.isfinite(count)):
        raise ValueError("n must be at least 1")
    if not sigma > 0.0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and > 0, got {sigma}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not n_target > 0.0:
        raise ValueError(f"n_target must be > 0, got {n_target}")
    rho = _tuned_rho(sigma, alpha, n_target)
    variance = rho + count * sigma**2
    return np.sqrt(variance * np.log(variance / (rho * alpha**2))) / count


def confidence_sequence(
    observations: npt.ArrayLike,
    *,
    sigma: float,
    alpha: float = 0.05,
    n_target: float = 100.0,
) -> ConfidenceSequence:
    """Return the confidence sequence for the mean after each observation.

    The interval after ``n`` observations with sum ``S`` is ``S / n`` plus or minus
    ``sqrt((rho + n sigma^2) log((rho + n sigma^2) / (rho alpha^2))) / n``.

    Args:
        observations: The observations, in the order they arrived.
        sigma: Sub-Gaussian scale (the standard deviation for normal data; 0.5 for data in
            [0, 1]).
        alpha: The error rate over all sample sizes at once, in (0, 1).
        n_target: The sample size at which the interval is made tightest (it stays valid at
            every size). Must be > 0.

    Raises:
        ValueError: If the observations are empty or not finite, ``sigma`` is not positive,
            ``alpha`` is not in (0, 1), or ``n_target`` is not positive.
    """
    x = _check(observations, sigma, alpha)
    if not n_target > 0.0:
        raise ValueError(f"n_target must be > 0, got {n_target}")
    n = np.arange(1, x.size + 1)
    radius = confidence_radius(n, sigma=sigma, alpha=alpha, n_target=n_target)
    mean = np.cumsum(x) / n
    return ConfidenceSequence(n.astype(np.int64), mean, mean - radius, mean + radius, alpha, sigma)


def e_process(
    observations: npt.ArrayLike,
    mu0: float,
    *,
    sigma: float,
    alpha: float = 0.05,
    n_target: float = 100.0,
) -> EProcess:
    """Return the e-value of "the mean is ``mu0``" after each observation.

    The e-value is ``sqrt(rho / (rho + n sigma^2)) exp((S - n mu0)^2 / (2 (rho + n sigma^2)))``
    for the sum ``S`` of ``n`` observations: a mixture of likelihood ratios over the size of
    the possible shift, so it grows with the evidence for any shift, in either direction.

    Raises:
        ValueError: As ``confidence_sequence``, and if ``mu0`` is not finite.
    """
    x = _check(observations, sigma, alpha)
    if not math.isfinite(mu0) or not n_target > 0.0:
        raise ValueError("mu0 must be finite and n_target > 0")
    rho = _tuned_rho(sigma, alpha, n_target)
    n = np.arange(1, x.size + 1)
    deviation = np.cumsum(x) - n * mu0
    variance = rho + n * sigma**2
    log_e = 0.5 * np.log(rho / variance) + deviation**2 / (2.0 * variance)
    return EProcess(n.astype(np.int64), np.exp(np.minimum(log_e, 700.0)), mu0, alpha)
