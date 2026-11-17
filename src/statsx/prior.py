"""Priors from expert judgement: a lognormal from a calibrated interval (BA-3).

An expert gives a lower and an upper bound and says how sure they are that the true value lies
between them (the default is 90%). ``lognormal_from_interval`` returns the lognormal distribution
whose central interval at that level is exactly those bounds: the log of the value is normal, the
median of the log is the middle of the two logs, and the sd follows from the normal quantile.

The expert must be calibrated: the interval holds the value as often as they claim. The function
cannot check that; it only converts the judgement into a distribution.

References:
    O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P. H., Jenkinson, D. J.,
    Oakley, J. E., & Rakow, T. (2006). *Uncertain Judgements: Eliciting Experts' Probabilities*.
    Wiley. https://doi.org/10.1002/0470033312.

    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
    *Bayesian Data Analysis*, 3rd ed., appendix A (lognormal distribution). CRC Press.
    https://doi.org/10.1201/b16018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.result import Interval


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Lognormal distribution: ``log(X)`` is normal with mean ``mu`` and sd ``sigma``."""

    mu: float
    sigma: float

    @property
    def median(self) -> float:
        """Return exp(mu)."""
        return math.exp(self.mu)

    @property
    def mean(self) -> float:
        """Return exp(mu + sigma^2 / 2)."""
        return math.exp(self.mu + self.sigma**2 / 2.0)

    def quantile(self, p: float) -> float:
        """Return the value below which a share ``p`` of the distribution lies."""
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {p}")
        return math.exp(self.mu + self.sigma * NormalDist().inv_cdf(p))

    def interval(self, level: float) -> Interval:
        """Return the central interval that holds ``level`` of the distribution."""
        tail = (1.0 - level) / 2.0
        return Interval(self.quantile(tail), self.quantile(1.0 - tail), level)

    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
        """Return ``n`` draws; the same seed gives the same draws."""
        return np.exp(_accel.sample_normal(self.mu, self.sigma, n, seed))


def lognormal_from_interval(low: float, high: float, level: float = 0.9) -> LogNormal:
    """Return the lognormal whose central ``level`` interval is ``(low, high)``.

    Args:
        low: The expert's lower bound. Must be > 0.
        high: The expert's upper bound. Must be above ``low``.
        level: The probability the expert gives that the value lies between the bounds.

    Raises:
        ValueError: If ``low`` is not positive, ``high`` is not above ``low``, or ``level`` is
            not in (0, 1).
    """
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"need finite bounds with 0 < low < high, got {low} and {high}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    return LogNormal(
        (math.log(low) + math.log(high)) / 2.0, (math.log(high) - math.log(low)) / (2.0 * z)
    )
