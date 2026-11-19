"""Conjugate priors and their closed-form posterior updates.

Each prior is a frozen dataclass, and each update returns a new object of the same family.
The arithmetic runs in the Rust core.

References:
    Murphy, K. P. (2007). *Conjugate Bayesian analysis of the Gaussian distribution*. Technical
    report, University of British Columbia (the Normal-Inverse-Gamma update).

    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
    *Bayesian Data Analysis*, 3rd ed. CRC Press. https://doi.org/10.1201/b16018.
    Chapter 2 (binomial, normal with known variance, Poisson) and chapter 3 (multinomial).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.distributions import Moments
from statsx.result import Interval


@dataclass(frozen=True, slots=True)
class Beta:
    """Beta(``alpha``, ``beta``) distribution for a probability."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Return alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Return the variance of the distribution."""
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s * s * (s + 1.0))

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return _accel.sample_beta(self.alpha, self.beta, n, seed)


@dataclass(frozen=True, slots=True)
class Gamma:
    """Gamma distribution with ``shape`` and ``rate`` (mean is shape / rate)."""

    shape: float
    rate: float

    @property
    def mean(self) -> float:
        """Return shape / rate."""
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Return shape / rate squared."""
        return self.shape / (self.rate * self.rate)

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return _accel.sample_gamma(self.shape, self.rate, n, seed)


@dataclass(frozen=True, slots=True)
class Normal:
    """Normal distribution with ``mean`` and standard deviation ``sd``."""

    mean: float
    sd: float

    @property
    def variance(self) -> float:
        """Return sd squared."""
        return self.sd * self.sd

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return _accel.sample_normal(self.mean, self.sd, n, seed)


@dataclass(frozen=True, slots=True)
class Dirichlet:
    """Dirichlet distribution over category probabilities, one ``alpha`` per category."""

    alpha: tuple[float, ...]

    @property
    def mean(self) -> tuple[float, ...]:
        """Return each alpha divided by the sum of alpha."""
        total = sum(self.alpha)
        return tuple(a / total for a in self.alpha)


@dataclass(frozen=True, slots=True)
class NormalInverseGamma:
    """Normal-Inverse-Gamma distribution for the mean and variance of a normal sample.

    The model is ``y_i ~ Normal(mu, sigma2)`` with ``mu | sigma2 ~ Normal(mean, sigma2 / kappa)``
    and ``sigma2 ~ InverseGamma(shape, scale)``. ``update`` returns the posterior after data;
    ``sample`` draws the mean ``mu``, so the object can go straight into
    ``statsx.posterior.compare``.

    Args:
        mean: Prior (or posterior) mean of ``mu``.
        kappa: Prior pseudo-count of the mean: the variance of ``mu`` is ``sigma2 / kappa``. A
            small value (``1e-6``) is weak.
        shape: Inverse-Gamma shape of ``sigma2``, above 0.
        scale: Inverse-Gamma scale of ``sigma2``, above 0.
    """

    mean: float
    kappa: float
    shape: float
    scale: float

    def __post_init__(self) -> None:
        positive = self.kappa > 0.0 and self.shape > 0.0 and self.scale > 0.0
        finite = all(np.isfinite(v) for v in (self.mean, self.kappa, self.shape, self.scale))
        if not (positive and finite):
            raise ValueError("mean must be finite, and kappa, shape and scale finite and > 0")

    @classmethod
    def reference(cls, observations: npt.ArrayLike | Moments) -> NormalInverseGamma:
        """Return the posterior under the reference prior ``p(mu, sigma2) ~ 1 / sigma2``.

        With ``n`` values, mean ``ybar`` and sum of squared deviations ``m2``, the posterior is
        ``(ybar, n, (n - 1) / 2, m2 / 2)``. The marginal of ``mu`` is the Student t with ``n - 1``
        degrees of freedom, centred at ``ybar`` with scale ``s / sqrt(n)``, so ``mean_interval``
        equals the classical t interval (Gelman et al. 2013, section 3.2). Use it when you have no
        trusted prior for the noise; it keeps its coverage at small ``n``.

        Raises:
            ValueError: If there are fewer than 2 values or they are all equal.
        """
        m = observations if isinstance(observations, Moments) else Moments.from_values(observations)
        if m.n < 2 or m.m2 <= 0.0:  # noqa: PLR2004
            raise ValueError("a reference posterior needs at least 2 values that are not all equal")
        return cls(m.mean, float(m.n), (m.n - 1) / 2.0, m.m2 / 2.0)

    def update(self, observations: npt.ArrayLike) -> NormalInverseGamma:
        """Return the posterior after observing the values ``observations``.

        Raises:
            ValueError: If there are no values or one is not finite.
        """
        return self.update_from_moments(Moments.from_values(observations))

    def update_from_moments(self, moments: Moments) -> NormalInverseGamma:
        """Return the posterior after data summarized as ``Moments`` (count, mean, ``m2``).

        With ``n``, sample mean ``ybar`` and ``m2`` the sum of squared deviations, the posterior is
        ``kappa' = kappa + n``, ``mean' = (kappa mean + n ybar) / kappa'``, ``shape' = shape +
        n / 2`` and ``scale' = scale + m2 / 2 + kappa n (ybar - mean)^2 / (2 kappa')``.
        """
        n = float(moments.n)
        kappa = self.kappa + n
        mean = (self.kappa * self.mean + n * moments.mean) / kappa
        shift = self.kappa * n * (moments.mean - self.mean) ** 2 / (2.0 * kappa)
        return NormalInverseGamma(
            mean, kappa, self.shape + n / 2.0, self.scale + moments.m2 / 2.0 + shift
        )

    @property
    def mean_df(self) -> float:
        """Return the degrees of freedom of the Student t marginal of ``mu``: ``2 shape``."""
        return 2.0 * self.shape

    @property
    def mean_scale(self) -> float:
        """Return the scale of the Student t marginal of ``mu``: ``sqrt(scale / (shape kappa))``."""
        return float(np.sqrt(self.scale / (self.shape * self.kappa)))

    def mean_interval(self, level: float = 0.95) -> Interval:
        """Return the equal-tailed credible interval for ``mu``, exact from the Student t marginal.

        Raises:
            ValueError: If ``level`` is not in (0, 1).
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        q = _accel.t_ppf(np.array([0.5 + level / 2.0]), self.mean_df)[0]
        half = q * self.mean_scale
        return Interval(self.mean - half, self.mean + half, level)

    def sample_joint(
        self, n: int, *, seed: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``n`` draws of ``(mu, sigma2)``; the same seed gives the same draws."""
        sigma2 = 1.0 / _accel.sample_gamma(self.shape, self.scale, n, seed)
        z = _accel.sample_normal(0.0, 1.0, n, seed + 1)
        return self.mean + np.sqrt(sigma2 / self.kappa) * z, sigma2

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws of the mean ``mu``; the same seed gives the same draws."""
        return self.sample_joint(n, seed=seed)[0]


def beta_binomial(prior: Beta, successes: float, trials: float) -> Beta:
    """Return the Beta posterior after ``successes`` out of ``trials`` binomial trials.

    Raises:
        ValueError: If a parameter is not positive, a count is negative, or
            ``successes > trials``.
    """
    a, b = _accel.beta_binomial(prior.alpha, prior.beta, successes, trials)
    return Beta(a, b)


def gamma_poisson(prior: Gamma, count: float, exposure: float) -> Gamma:
    """Return the Gamma posterior of a Poisson rate after ``count`` events in ``exposure``.

    ``exposure`` is the total observation time (or the number of unit-time observations).

    Raises:
        ValueError: If a parameter is not positive or a count is negative.
    """
    shape, rate = _accel.gamma_poisson(prior.shape, prior.rate, count, exposure)
    return Gamma(shape, rate)


def normal_normal(prior: Normal, sigma: float, n: int, sample_mean: float) -> Normal:
    """Return the Normal posterior of a mean, given ``n`` observations with known sd ``sigma``.

    Raises:
        ValueError: If ``sigma`` or the prior sd is not positive, or ``n`` is negative.
    """
    mean, sd = _accel.normal_normal(prior.mean, prior.sd, sigma, float(n), sample_mean)
    return Normal(mean, sd)


def dirichlet_multinomial(prior: Dirichlet, counts: Sequence[float]) -> Dirichlet:
    """Return the Dirichlet posterior after multinomial ``counts``, one per category.

    Raises:
        ValueError: If the lengths differ, there are fewer than 2 categories, an alpha is not
            positive, or a count is negative.
    """
    return Dirichlet(tuple(_accel.dirichlet_multinomial(list(prior.alpha), list(counts))))
