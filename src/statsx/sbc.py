"""Simulation-based calibration: check that a posterior method is correct (BA-8).

Draw a parameter from the prior, simulate data from it, run your inference on that data, and
record the rank of the true parameter among ``n_draws`` posterior draws. Repeat. If the
inference is correct, every rank is uniform on ``0 .. n_draws``, whatever the prior and the model.

``sbc_ranks`` computes the ranks from the arrays your loop produced. ``check_calibration`` tests
them: a chi-square test of the whole histogram, a test of the mean rank (the posterior is biased
when it fails) and a test of the spread of the ranks (the posterior is too narrow when ranks pile
at both ends, too wide when they pile in the middle).

The ranks are independent across replications, but the posterior draws inside one replication
must be independent too. Thin MCMC draws until their autocorrelation is small, or the test
rejects a correct sampler.

References:
    Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian
    inference algorithms with simulation-based calibration. arXiv:1804.06788.

    Cook, S. R., Gelman, A., & Rubin, D. B. (2006). Validation of software for Bayesian models
    using posterior quantiles. *Journal of Computational and Graphical Statistics* 15(3):675-692.
    https://doi.org/10.1198/106186006X136976.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel

MAX_BINS = 20
"""Largest number of histogram bins used by the chi-square test."""

_MIN_EXPECTED = 5.0
"""Smallest expected count per bin for the chi-square approximation."""


@dataclass(frozen=True, slots=True)
class ParameterCalibration:
    """The calibration of one parameter.

    Args:
        name: The parameter name.
        n: Number of replications.
        counts: Rank histogram, one count per bin.
        chi_square_p: P-value of the chi-square test of uniformity.
        bias_p: Two-sided P-value that the mean rank is the middle one.
        spread_p: Two-sided P-value that the spread of the ranks is the uniform one.
        spread_z: Positive when ranks pile at the ends (posterior too narrow), negative when they
            pile in the middle (posterior too wide).
        mean_z: Positive when the true value ranks high (posterior sits below the truth).
    """

    name: str
    n: int
    counts: tuple[int, ...]
    chi_square_p: float
    bias_p: float
    spread_p: float
    spread_z: float
    mean_z: float


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """The result of one calibration check; ``passed`` is True when no reason is listed."""

    n_draws: int
    parameters: tuple[ParameterCalibration, ...]
    alpha: float
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return True when no test rejected uniformity."""
        return not self.reasons


def sbc_ranks(
    truth: npt.ArrayLike, posterior: npt.ArrayLike, *, seed: int = 0
) -> npt.NDArray[np.int64]:
    """Return the rank of each true value among its posterior draws.

    Args:
        truth: The parameter drawn from the prior in each replication, shape (replications,).
        posterior: Posterior draws of that parameter, shape (replications, n_draws).
        seed: Breaks ties at random, which only matters for a discrete parameter.

    Returns:
        Integers from 0 to ``n_draws``: the number of draws below the true value, plus a random
        share of the draws equal to it.

    Raises:
        ValueError: If the shapes do not match, there are fewer than 2 draws, or a value is not
            finite.
    """
    t = np.asarray(truth, dtype=np.float64)
    p = np.asarray(posterior, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 2 or p.shape[0] != t.size or p.shape[1] < 2:  # noqa: PLR2004
        raise ValueError(
            "truth must have shape (replications,) and posterior (replications, n_draws >= 2)"
        )
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("truth and posterior must be finite")
    below: npt.NDArray[np.int64] = np.count_nonzero(p < t[:, None], axis=1).astype(np.int64)
    equal: npt.NDArray[np.int64] = np.count_nonzero(p == t[:, None], axis=1).astype(np.int64)
    ties: npt.NDArray[np.int64] = np.random.default_rng(seed).integers(0, equal + 1)
    return below + ties


def _n_bins(n_ranks: int, n: int) -> int:
    """Return the largest bin count up to ``MAX_BINS`` that divides ``n_ranks`` and keeps 5 each."""
    for bins in range(min(MAX_BINS, n_ranks), 1, -1):
        if n_ranks % bins == 0 and n / bins >= _MIN_EXPECTED:
            return bins
    return 1


def _normal_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _check_one(name: str, ranks: npt.NDArray[np.int64], n_draws: int) -> ParameterCalibration:
    n_ranks = n_draws + 1
    n = ranks.size
    bins = _n_bins(n_ranks, n)
    counts = np.bincount(ranks * bins // n_ranks, minlength=bins)
    if bins > 1:
        expected = n / bins
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        chi_p = float(_accel.chi2_sf(np.array([chi2]), float(bins - 1))[0])
    else:
        chi_p = math.nan
    # Midpoint scores of a uniform rank, with their exact mean, spread and variance of the spread.
    v = (np.arange(n_ranks) + 0.5) / n_ranks - 0.5
    spread_mean = float(np.mean(v**2))
    spread_var = float(np.mean(v**4) - spread_mean**2)
    u = (ranks + 0.5) / n_ranks - 0.5
    mean_z = float(np.mean(u)) / math.sqrt(float(np.mean(v**2)) / n)
    spread_z = (float(np.mean(u**2)) - spread_mean) / math.sqrt(spread_var / n)
    return ParameterCalibration(
        name,
        n,
        tuple(int(c) for c in counts),
        chi_p,
        _normal_two_sided(mean_z),
        _normal_two_sided(spread_z),
        spread_z,
        mean_z,
    )


def check_calibration(
    ranks: Mapping[str, npt.ArrayLike], n_draws: int, *, alpha: float = 0.05
) -> CalibrationCheck:
    """Test that the ranks of each parameter are uniform, and say how they are not.

    Args:
        ranks: Parameter name to its ranks from ``sbc_ranks``, one per replication.
        n_draws: Number of posterior draws behind each rank. Ranks run from 0 to ``n_draws``.
        alpha: Family-wise level. Each of the three tests of each parameter is compared with
            ``alpha`` divided by the number of tests (Bonferroni).

    Raises:
        ValueError: If ``alpha`` is not in (0, 1), a rank is outside 0 to ``n_draws``, or a
            parameter has fewer than 20 replications.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    parameters: list[ParameterCalibration] = []
    for name, value in ranks.items():
        r = np.asarray(value)
        if r.ndim != 1 or r.size < 20:  # noqa: PLR2004
            raise ValueError(f"{name}: need at least 20 replications, got shape {r.shape}")
        if not np.issubdtype(r.dtype, np.integer) or r.min() < 0 or r.max() > n_draws:
            raise ValueError(f"{name}: ranks must be integers from 0 to {n_draws}")
        parameters.append(_check_one(name, r.astype(np.int64), n_draws))
    cut = alpha / (3.0 * max(len(parameters), 1))
    reasons: list[str] = []
    for c in parameters:
        if c.chi_square_p < cut:
            reasons.append(f"{c.name}: rank histogram is not uniform (p = {c.chi_square_p:.2g})")
        if c.bias_p < cut:
            side = "below" if c.mean_z > 0 else "above"
            reasons.append(
                f"{c.name}: the posterior sits {side} the truth on average (p = {c.bias_p:.2g})"
            )
        if c.spread_p < cut:
            shape = "too narrow" if c.spread_z > 0 else "too wide"
            reasons.append(f"{c.name}: the posterior is {shape} (p = {c.spread_p:.2g})")
    return CalibrationCheck(n_draws, tuple(parameters), alpha, tuple(reasons))
