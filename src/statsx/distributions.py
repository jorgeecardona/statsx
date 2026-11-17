"""Fit normal, lognormal and shifted-exponential distributions from sufficient statistics (DI-1).

Sufficient statistics are a few numbers that hold all the information a model needs from the
rows. You can fit from them when the rows are gone, too large, or spread over many machines. Two
shards of statistics merge into the statistics of all the rows together, so a fit over shards
equals a fit over the pooled rows.

- ``Moments``: the count, the mean and the sum of squared deviations ``m2``. Enough for a normal
  distribution. For a lognormal distribution, take the moments of the logarithms of the values
  (``log_moments``).
- ``ExponentialStats``: the count, the minimum and the mean. Enough for a shifted exponential
  distribution: ``shift + Exponential(rate)``.

Each ``fit_*`` function returns the maximum likelihood fit, or with ``bias_corrected=True`` the
fit that removes the known small-sample bias:

- normal and lognormal: the variance is ``m2 / (n - 1)`` instead of ``m2 / n``;
- shifted exponential: the uniformly minimum variance unbiased estimates, ``shift = min -
  T / (n (n - 1))`` and ``rate = (n - 2) / T`` with ``T = n (mean - min)``. The maximum
  likelihood values are ``shift = min`` and ``rate = n / T``.

References:
    Casella, G., & Berger, R. L. (2002). *Statistical Inference*, 2nd ed., chapter 7 (maximum
    likelihood). Duxbury.

    Lehmann, E. L., & Casella, G. (1998). *Theory of Point Estimation*, 2nd ed., chapter 2 (unbiased
    estimation; the bias-corrected shifted exponential is derived from it and checked by
    simulation in the tests). Springer.
    https://doi.org/10.1007/b98854.

    Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for computing the sample
    variance: analysis and recommendations. *The American Statistician* 37(3):242-247.
    https://doi.org/10.1080/00031305.1983.10483115. The rule for merging two shards.

    Aitchison, J., & Brown, J. A. C. (1957). *The Lognormal Distribution*. Cambridge University
    Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from statsx import _accel

_SQRT2 = math.sqrt(2.0)
_erfc = np.vectorize(math.erfc, otypes=[np.float64])


def _normal_cdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(0.5 * _erfc(-z / _SQRT2), dtype=np.float64)


def _normal_ppf(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    inv = NormalDist().inv_cdf
    return np.array([inv(float(v)) for v in p.ravel()], dtype=np.float64).reshape(p.shape)


def _normal_pdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _unit_interval(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(p, dtype=np.float64)
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("probabilities must be strictly between 0 and 1")
    return values


def _closed_unit_interval(u: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(u, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("u must be between 0 and 1")
    return values


@dataclass(frozen=True, slots=True)
class Moments:
    """Sufficient statistics for a normal distribution.

    Args:
        n: The number of values, at least 1.
        mean: Their mean.
        m2: The sum of squared deviations from the mean, not negative.
    """

    n: int
    mean: float
    m2: float

    def __post_init__(self) -> None:
        if self.n < 1 or not math.isfinite(self.mean) or not math.isfinite(self.m2):
            raise ValueError("n must be at least 1, and mean and m2 must be finite")
        if self.m2 < 0.0:
            raise ValueError(f"m2 must not be negative, got {self.m2}")

    @classmethod
    def from_values(cls, values: npt.ArrayLike) -> Moments:
        """Return the moments of the rows ``values``.

        Raises:
            ValueError: If there are no values or one is not finite.
        """
        x = np.asarray(values, dtype=np.float64).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("values must be a non-empty array of finite numbers")
        mean = float(x.mean())
        return cls(int(x.size), mean, float(np.sum((x - mean) ** 2)))

    @classmethod
    def from_summary(cls, n: int, mean: float, variance: float, *, ddof: int = 1) -> Moments:
        """Return the moments from a count, a mean and a variance divided by ``n - ddof``.

        Raises:
            ValueError: If ``ddof`` is not 0 or 1, or the values are out of range.
        """
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof}")
        return cls(n, mean, variance * (n - ddof))

    def merge(self, other: Moments) -> Moments:
        """Return the moments of the rows of both shards together (Chan et al. 1983)."""
        n = self.n + other.n
        delta = other.mean - self.mean
        mean = self.mean + delta * other.n / n
        m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n
        return Moments(n, mean, m2)


def log_moments(values: npt.ArrayLike) -> Moments:
    """Return the moments of the logarithms of ``values``, for ``fit_lognormal``.

    Raises:
        ValueError: If there are no values or one is not positive and finite.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("a lognormal needs a non-empty array of positive, finite values")
    return Moments.from_values(np.log(x))


@dataclass(frozen=True, slots=True)
class ExponentialStats:
    """Sufficient statistics for a shifted exponential distribution.

    Args:
        n: The number of values, at least 1.
        minimum: The smallest value.
        mean: The mean, not below ``minimum``.
    """

    n: int
    minimum: float
    mean: float

    def __post_init__(self) -> None:
        if self.n < 1 or not math.isfinite(self.minimum) or not math.isfinite(self.mean):
            raise ValueError("n must be at least 1, and minimum and mean must be finite")
        if self.mean < self.minimum:
            raise ValueError(f"mean {self.mean} is below the minimum {self.minimum}")

    @classmethod
    def from_values(cls, values: npt.ArrayLike) -> ExponentialStats:
        """Return the statistics of the rows ``values``.

        Raises:
            ValueError: If there are no values or one is not finite.
        """
        x = np.asarray(values, dtype=np.float64).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("values must be a non-empty array of finite numbers")
        return cls(int(x.size), float(x.min()), float(x.mean()))

    def merge(self, other: ExponentialStats) -> ExponentialStats:
        """Return the statistics of the rows of both shards together."""
        n = self.n + other.n
        mean = (self.n * self.mean + other.n * other.mean) / n
        return ExponentialStats(n, min(self.minimum, other.minimum), mean)


@dataclass(frozen=True, slots=True)
class NormalFit:
    """A fitted normal distribution.

    Args:
        mean: The mean.
        sd: The standard deviation, above 0.
        n: The number of values behind the fit.
    """

    mean: float
    sd: float
    n: int

    def cdf(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return P(X <= x)."""
        return _normal_cdf((np.asarray(x, dtype=np.float64) - self.mean) / self.sd)

    def ppf(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the quantile at each probability ``p`` in (0, 1)."""
        return self.mean + self.sd * _normal_ppf(_unit_interval(p))

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return _accel.sample_normal(self.mean, self.sd, n, seed)

    def quantile_integral(self, u: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the integral of the quantile function from 0 to ``u``, for ``u`` in [0, 1]."""
        v = _closed_unit_interval(u)
        inner = np.clip(v, 1e-300, 1.0 - 1e-16)
        pdf = np.where((v <= 0.0) | (v >= 1.0), 0.0, _normal_pdf(_normal_ppf(inner)))
        return self.mean * v - self.sd * pdf


@dataclass(frozen=True, slots=True)
class LognormalFit:
    """A fitted lognormal distribution: ``log X`` is normal with mean ``mu`` and sd ``sigma``.

    Args:
        mu: The mean of the logarithm.
        sigma: The standard deviation of the logarithm, above 0.
        n: The number of values behind the fit.
    """

    mu: float
    sigma: float
    n: int

    def cdf(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return P(X <= x); 0 for ``x <= 0``."""
        v = np.asarray(x, dtype=np.float64)
        safe = np.where(v > 0.0, v, 1.0)
        z = (np.log(safe) - self.mu) / self.sigma
        return np.where(v > 0.0, _normal_cdf(z), 0.0)

    def ppf(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the quantile at each probability ``p`` in (0, 1)."""
        return np.exp(self.mu + self.sigma * _normal_ppf(_unit_interval(p)))

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return np.exp(_accel.sample_normal(self.mu, self.sigma, n, seed))

    def quantile_integral(self, u: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the integral of the quantile function from 0 to ``u``, for ``u`` in [0, 1]."""
        v = _closed_unit_interval(u)
        inner = np.clip(v, 1e-300, 1.0 - 1e-16)
        shifted = _normal_cdf(_normal_ppf(inner) - self.sigma)
        top = np.where(v >= 1.0, 1.0, shifted)
        return np.where(v <= 0.0, 0.0, math.exp(self.mu + 0.5 * self.sigma**2) * top)


@dataclass(frozen=True, slots=True)
class ShiftedExponentialFit:
    """A fitted shifted exponential: ``X = shift + Exponential(rate)``.

    Args:
        shift: The lowest possible value.
        rate: The rate, above 0; the mean above the shift is ``1 / rate``.
        n: The number of values behind the fit.
    """

    shift: float
    rate: float
    n: int

    def cdf(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return P(X <= x); 0 below the shift."""
        d = np.maximum(np.asarray(x, dtype=np.float64) - self.shift, 0.0)
        return -np.expm1(-self.rate * d)

    def ppf(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the quantile at each probability ``p`` in (0, 1)."""
        return self.shift - np.log1p(-_unit_interval(p)) / self.rate

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return self.shift + np.random.default_rng(seed).exponential(1.0 / self.rate, n)

    def quantile_integral(self, u: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the integral of the quantile function from 0 to ``u``, for ``u`` in [0, 1]."""
        v = _closed_unit_interval(u)
        inside = np.minimum(v, 0.5)  # a safe argument for the branch that is not used
        rest = np.where(v >= 1.0, 0.0, (1.0 - v) * np.log1p(-np.where(v >= 1.0, inside, v)))
        return self.shift * v + (rest + v) / self.rate


def _sd(moments: Moments, *, bias_corrected: bool) -> float:
    if moments.n < 2 or moments.m2 <= 0.0:  # noqa: PLR2004
        raise ValueError("a fit needs at least 2 values that are not all equal")
    return math.sqrt(moments.m2 / (moments.n - 1 if bias_corrected else moments.n))


def fit_normal(moments: Moments, *, bias_corrected: bool = False) -> NormalFit:
    """Fit a normal distribution to the moments of the values.

    Raises:
        ValueError: If there are fewer than 2 values or they are all equal.
    """
    return NormalFit(moments.mean, _sd(moments, bias_corrected=bias_corrected), moments.n)


def fit_lognormal(log_stats: Moments, *, bias_corrected: bool = False) -> LognormalFit:
    """Fit a lognormal distribution to the moments of the logarithms (see ``log_moments``).

    Raises:
        ValueError: If there are fewer than 2 values or their logarithms are all equal.
    """
    return LognormalFit(log_stats.mean, _sd(log_stats, bias_corrected=bias_corrected), log_stats.n)


def fit_shifted_exponential(
    stats: ExponentialStats, *, bias_corrected: bool = False
) -> ShiftedExponentialFit:
    """Fit ``shift + Exponential(rate)`` to the statistics of the values.

    Raises:
        ValueError: If there are fewer than 2 values (3 when ``bias_corrected``) or they are all
            equal.
    """
    n = stats.n
    total = n * (stats.mean - stats.minimum)
    least = 3 if bias_corrected else 2
    if n < least or total <= 0.0:
        raise ValueError(f"a fit needs at least {least} values that are not all equal")
    if bias_corrected:
        return ShiftedExponentialFit(stats.minimum - total / (n * (n - 1)), (n - 2) / total, n)
    return ShiftedExponentialFit(stats.minimum, n / total, n)
