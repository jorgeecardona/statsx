"""Calibration of a classifier's probabilities: measure it (CA-1) and fix it (CA-2).

A classifier is calibrated when, among the items it gives probability ``p``, a share ``p`` are
positive. ``reliability`` bins the predicted probabilities and compares the mean prediction in
each bin with the share of positives, which are the numbers of a reliability diagram. From the
same bins it gives:

- ECE, the expected calibration error: the bin gaps, weighted by bin size. MCE is the largest gap
  among bins with items.
- The Brier score and its decomposition into reliability (calibration error), resolution (how
  much the bins separate positives from negatives) and uncertainty (the variance of the outcome).
  With bins, the decomposition has two more terms from the spread of the predictions inside a
  bin; they are zero when every bin holds one predicted value. The identity is exact:
  ``brier = reliability - resolution + uncertainty + within_bin``.

When a classifier is miscalibrated, ``fit_platt`` (a sigmoid on the score) and ``fit_isotonic`` (a
non-decreasing step-free curve) learn a map from its scores to probabilities on held-out data.
Platt scaling has 2 parameters, so it is stable with few items, but it can only stretch and shift
the score's logit. Isotonic regression fits any non-decreasing shape, so it needs more items and
can overfit with fewer: when the miscalibration was a sigmoid of the logit (an overconfident
classifier), isotonic had a worse test Brier score than Platt with 40 and 200 training items
and the same from 1000. Fit the map on data the classifier was not
trained on, and check the result on a third set.

ECE depends on the number of bins and on how the bins are cut (equal width, or equal count). It
is biased upward with few items per bin: for a perfectly calibrated classifier, 10 bins give a
mean ECE of about 0.10 at 100 items, 0.03 at 1000 and 0.01 at 10000. Read ECE with the bin
counts, and the Brier score as the overall check.

References:
    DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and evaluation of forecasters.
    *The Statistician* 32(1/2):12-22. https://doi.org/10.2307/2987588. Reliability diagram.

    Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied
    Meteorology* 12(4):595-600. https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2.

    Stephenson, D. B., Coelho, C. A. S., & Jolliffe, I. T. (2008). Two extra components in the
    Brier score decomposition. *Weather and Forecasting* 23(4):752-757.
    https://doi.org/10.1175/2007WAF2006116.1. The within-bin terms.

    Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to
    regularized likelihood methods. In *Advances in Large Margin Classifiers*, 61-74. MIT Press.

    Lin, H.-T., Lin, C.-J., & Weng, R. C. (2007). A note on Platt's probabilistic outputs for
    support vector machines. *Machine Learning* 68(3):267-276.
    https://doi.org/10.1007/s10994-007-5018-6. The Newton method with backtracking.

    Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass
    probability estimates. *Proceedings of KDD 2002*, 694-699.
    https://doi.org/10.1145/775047.775151. Isotonic regression for calibration.

    Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). *Statistical
    Inference under Order Restrictions*. Wiley. Pool-adjacent-violators algorithm.

    Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated
    probabilities using Bayesian binning into quantiles. *Proceedings of the AAAI Conference
    on Artificial Intelligence* 29(1). ECE and MCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel

Strategy = Literal["uniform", "quantile"]


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """The Brier score as ``reliability - resolution + uncertainty + within_bin``."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin: float


@dataclass(frozen=True, slots=True)
class Reliability:
    """The numbers of a reliability diagram, ECE, MCE and the Brier decomposition.

    Each array has one entry per bin that holds at least one item, in increasing order of the
    predicted probability.

    Args:
        counts: Number of items in the bin.
        mean_predicted: Mean predicted probability in the bin.
        observed_rate: Share of positives in the bin.
        ece: Expected calibration error: bin gaps weighted by bin size.
        mce: Maximum calibration error: the largest bin gap.
        brier: The Brier score and its decomposition.
        n: Number of items.
    """

    counts: npt.NDArray[np.int64]
    mean_predicted: npt.NDArray[np.float64]
    observed_rate: npt.NDArray[np.float64]
    ece: float
    mce: float
    brier: BrierDecomposition
    n: int


def check_labels(
    y_true: npt.ArrayLike, p: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return labels and probabilities as float arrays after checking them."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    q = np.asarray(p, dtype=np.float64).ravel()
    if y.size != q.size or y.size == 0:
        raise ValueError("y_true and p must be non-empty arrays of the same length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must hold only 0 and 1")
    if not np.all((q >= 0.0) & (q <= 1.0)):
        raise ValueError("p must hold probabilities in [0, 1]")
    return y, q


def _bin_index(
    q: npt.NDArray[np.float64], n_bins: int, strategy: Strategy
) -> npt.NDArray[np.int64]:
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(q, np.linspace(0.0, 1.0, n_bins + 1))
    index = np.searchsorted(edges[1:-1], q, side="right")
    return index.astype(np.int64)


def reliability(
    y_true: npt.ArrayLike,
    p: npt.ArrayLike,
    *,
    n_bins: int = 10,
    strategy: Strategy = "uniform",
) -> Reliability:
    """Return the reliability numbers, ECE, MCE and the Brier decomposition of probabilities.

    Args:
        y_true: The outcome of each item, 0 or 1.
        p: The predicted probability of 1 for each item.
        n_bins: Number of bins, at least 1.
        strategy: ``"uniform"`` for bins of equal width on [0, 1] (the usual reliability
            diagram), or ``"quantile"`` for bins with about the same number of items.

    Raises:
        ValueError: If the arrays differ in length or are empty, ``y_true`` is not 0 or 1, ``p``
            is outside [0, 1], ``n_bins`` is below 1, or ``strategy`` is unknown.
    """
    y, q = check_labels(y_true, p)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
    index = _bin_index(q, n_bins, strategy)
    n = y.size
    base_rate = float(y.mean())
    counts: list[int] = []
    mean_p: list[float] = []
    rate: list[float] = []
    reliability_term = resolution = within_var = within_cov = 0.0
    for k in range(n_bins):
        mask = index == k
        size = int(mask.sum())
        if size == 0:
            continue
        f, o = q[mask], y[mask]
        f_bar, o_bar = float(f.mean()), float(o.mean())
        counts.append(size)
        mean_p.append(f_bar)
        rate.append(o_bar)
        reliability_term += size * (f_bar - o_bar) ** 2
        resolution += size * (o_bar - base_rate) ** 2
        within_var += float(np.sum((f - f_bar) ** 2))
        within_cov += float(np.sum((f - f_bar) * (o - o_bar)))
    counts_a = np.array(counts, dtype=np.int64)
    mean_a = np.array(mean_p, dtype=np.float64)
    rate_a = np.array(rate, dtype=np.float64)
    gaps = np.abs(mean_a - rate_a)
    decomposition = BrierDecomposition(
        brier=float(np.mean((q - y) ** 2)),
        reliability=reliability_term / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin=(within_var - 2.0 * within_cov) / n,
    )
    return Reliability(
        counts=counts_a,
        mean_predicted=mean_a,
        observed_rate=rate_a,
        ece=float(np.sum(counts_a * gaps) / n),
        mce=float(gaps.max()),
        brier=decomposition,
        n=n,
    )


@dataclass(frozen=True, slots=True)
class PlattScaler:
    """The map ``p = 1 / (1 + exp(-(a * score + b)))`` learned by ``fit_platt``."""

    a: float
    b: float

    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return calibrated probabilities for ``scores``."""
        z = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """A non-decreasing map from scores to probabilities, learned by ``fit_isotonic``.

    ``thresholds`` are the distinct training scores in increasing order and ``values`` the fitted
    probability at each. Between two thresholds the map is linear, and outside them it is
    constant.
    """

    thresholds: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return calibrated probabilities for ``scores``."""
        return np.interp(np.asarray(scores, dtype=np.float64), self.thresholds, self.values)


def fit_platt(scores: npt.ArrayLike, y_true: npt.ArrayLike) -> PlattScaler:
    """Fit Platt scaling: a sigmoid of the score, by maximum likelihood on smoothed targets.

    Args:
        scores: The classifier's scores (any real numbers; logits work best).
        y_true: The outcome of each item, 0 or 1.

    Raises:
        ValueError: If the arrays differ in length or are empty, a score is not finite, ``y_true``
            is not 0 or 1, or only one class is present.
    """
    f = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if f.size != y.size or f.size == 0:
        raise ValueError("scores and y_true must be non-empty arrays of the same length")
    if not np.all(np.isfinite(f)) or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("scores must be finite and y_true must hold only 0 and 1")
    if y.min() == y.max():
        raise ValueError("y_true holds one class only: there is nothing to calibrate")
    a, b = _accel.platt_fit(f, y)
    return PlattScaler(a, b)


def fit_isotonic(scores: npt.ArrayLike, y_true: npt.ArrayLike) -> IsotonicCalibrator:
    """Fit isotonic regression of the outcome on the score (pool adjacent violators).

    Items with the same score are pooled first. The fitted curve never decreases, so a higher
    score never gives a lower probability.

    Args:
        scores: The classifier's scores (any finite numbers).
        y_true: The outcome of each item, 0 or 1.

    Raises:
        ValueError: If the arrays differ in length or are empty, a score is not finite, ``y_true``
            is not 0 or 1, or only one class is present.
    """
    f = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if f.size != y.size or f.size == 0:
        raise ValueError("scores and y_true must be non-empty arrays of the same length")
    if not np.all(np.isfinite(f)) or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("scores must be finite and y_true must hold only 0 and 1")
    if y.min() == y.max():
        raise ValueError("y_true holds one class only: there is nothing to calibrate")
    thresholds, inverse, counts = np.unique(f, return_inverse=True, return_counts=True)
    rate = np.bincount(inverse, weights=y) / counts
    fitted = _accel.isotonic_fit(rate, counts.astype(np.float64))
    return IsotonicCalibrator(thresholds, np.asarray(fitted, dtype=np.float64))
