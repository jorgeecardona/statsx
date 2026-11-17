"""Conjugate priors and their closed-form posterior updates (BA-1).

Each prior is a frozen dataclass, and each update returns a new object of the same family.
The arithmetic runs in the Rust core.

References:
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
