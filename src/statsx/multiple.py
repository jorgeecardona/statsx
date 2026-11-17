"""Corrections for many tests: Holm, Benjamini-Hochberg, Storey q-values (CO-3).

``adjust`` takes the p-values of m tests and returns adjusted values in the input order and which
tests are rejected at level ``alpha``.

- ``"holm"`` controls the family-wise error rate (the chance of any false rejection) under any
  dependence.
- ``"bh"`` (Benjamini-Hochberg) controls the false discovery rate (the expected share of false
  rejections among the rejections) when the tests are independent or positively dependent.
- ``"storey"`` gives q-values: Benjamini-Hochberg scaled by an estimate of the share of true
  nulls, pi0. It rejects more than ``"bh"`` when many tests have a real effect. pi0 is estimated
  at one threshold ``lambda_`` (default 0.5), as p-values above it are mostly nulls; it is capped
  at 1.

References:
    Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
    Journal of Statistics* 6(2):65-70.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
    powerful approach to multiple testing. *Journal of the Royal Statistical Society B*
    57(1):289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x.

    Storey, J. D. (2002). A direct approach to false discovery rates. *Journal of the Royal
    Statistical Society B* 64(3):479-498. https://doi.org/10.1111/1467-9868.00346.

    Storey, J. D., & Tibshirani, R. (2003). Statistical significance for genomewide studies.
    *Proceedings of the National Academy of Sciences* 100(16):9440-9445.
    https://doi.org/10.1073/pnas.1530509100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Method = Literal["holm", "bh", "storey"]


@dataclass(frozen=True, slots=True)
class Adjustment:
    """The result of a multiple-testing correction, in the order of the input p-values.

    Args:
        method: The correction that was used.
        alpha: The level: the error rate that the method controls.
        adjusted: Adjusted p-values (Holm, Benjamini-Hochberg) or q-values (Storey).
        rejected: True where the adjusted value is at most ``alpha``.
        pi0: The estimated share of true nulls (Storey only; None for the other methods).
    """

    method: Method
    alpha: float
    adjusted: npt.NDArray[np.float64]
    rejected: npt.NDArray[np.bool_]
    pi0: float | None

    @property
    def n_rejected(self) -> int:
        """Return the number of rejected tests."""
        return int(self.rejected.sum())


def adjust(
    pvalues: npt.ArrayLike,
    *,
    method: Method = "bh",
    alpha: float = 0.05,
    lambda_: float = 0.5,
) -> Adjustment:
    """Return adjusted p-values and the rejections of ``len(pvalues)`` tests.

    Args:
        pvalues: One p-value per test, each in [0, 1].
        method: ``"holm"``, ``"bh"`` (default) or ``"storey"``.
        alpha: The error rate to control, in (0, 1).
        lambda_: The threshold of the pi0 estimate, in [0, 1); used by ``"storey"`` only.

    Raises:
        ValueError: If a p-value is outside [0, 1] or not a number, there are none, ``alpha`` is
            not in (0, 1), the method is unknown, or ``lambda_`` is outside [0, 1).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("pvalues must be a non-empty 1-D array of numbers in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("holm", "bh", "storey"):
        raise ValueError(f"method must be 'holm', 'bh' or 'storey', got {method!r}")
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    pi0: float | None = None
    if method == "holm":
        # p_(i) * (m - i + 1), made non-decreasing from the smallest p-value up.
        step = np.maximum.accumulate(sorted_p * (m - ranks + 1.0))
    else:
        scale = 1.0
        if method == "storey":
            pi0 = min(1.0, float(np.mean(p > lambda_)) / (1.0 - lambda_))
            scale = pi0
        # scale * p_(i) * m / i, made non-decreasing from the largest p-value down.
        step = np.minimum.accumulate((scale * sorted_p * m / ranks)[::-1])[::-1]
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(step, 1.0)
    return Adjustment(method, alpha, adjusted, adjusted <= alpha, pi0)
