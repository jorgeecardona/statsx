"""Stopping boundaries for a planned number of looks: O'Brien-Fleming and Pocock (SQ-2).

When you plan K looks at the data (for example after every quarter of the sample) and stop if
the effect is clear, testing at 5% at each look gives an overall error rate far above 5%. A
group sequential design uses a different critical value at each look so that the overall
chance of a false stop is ``alpha``.

- ``"obrien_fleming"``: very strict at the first looks and close to the fixed-sample value at
  the end. Stopping early is rare, and the last look keeps most of the power.
- ``"pocock"``: the same critical value at every look. Stopping early is easier, and the final
  look is stricter than a fixed-sample test.

The test statistic at look ``k`` is the standardized difference ``Z_k`` computed from all the
data so far. Stop at the first look where ``|Z_k|`` (or ``Z_k`` for a one-sided design) is at
least ``z_bounds[k]``. The bounds are found by the recursive numerical integration of Armitage,
McPherson & Rowe (1969), on the Brownian motion of the partial sums, for the stated
``information times`` (the share of the final information at each look; equal spacing by
default).

References:
    Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical
    trials. *Biometrika* 64(2):191-199. https://doi.org/10.1093/biomet/64.2.191.

    O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical trials.
    *Biometrics* 35(3):549-556. https://doi.org/10.2307/2530245.

    Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests on
    accumulating data. *Journal of the Royal Statistical Society A* 132(2):235-244.
    https://doi.org/10.2307/2343787.

    Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
    Clinical Trials*, chapter 2. Chapman & Hall/CRC. https://doi.org/10.1201/9781584888581.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

Kind = Literal["obrien_fleming", "pocock"]
_NORMAL = NormalDist()
_GRID = 801
_TOLERANCE = 1e-11
_TIME_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class GroupSequentialDesign:
    """Critical values for a planned number of looks.

    Args:
        kind: ``"obrien_fleming"`` or ``"pocock"``.
        alpha: The overall error rate.
        two_sided: True when the test stops for ``|Z| >= bound``, False for ``Z >= bound``.
        times: Information time of each look, increasing, the last one 1.
        z_bounds: The critical value of the standardized statistic at each look.
        nominal_p: The p-value level that matches each bound (two-sided or one-sided as set).
        alpha_spent: The chance of stopping for the first time at each look when there is no
            effect; it adds up to ``alpha``.
    """

    kind: Kind
    alpha: float
    two_sided: bool
    times: tuple[float, ...]
    z_bounds: tuple[float, ...]
    nominal_p: tuple[float, ...]
    alpha_spent: tuple[float, ...]


def _crossing_probabilities(
    bounds: npt.NDArray[np.float64], times: npt.NDArray[np.float64], *, two_sided: bool
) -> npt.NDArray[np.float64]:
    """Return P(first crossing at look k) for a Brownian motion, by recursive integration.

    ``bounds[k]`` is the boundary on the scale of B(t_k) = sqrt(t_k) * Z_k. The density of the
    paths that have not stopped is carried on a grid, starting from a point mass at 0.
    """
    spent = np.zeros(bounds.size)
    cdf: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = np.vectorize(
        _NORMAL.cdf, otypes=[np.float64]
    )
    grid: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64)
    density: npt.NDArray[np.float64] = np.ones(1, dtype=np.float64)
    weights: npt.NDArray[np.float64] = np.ones(1, dtype=np.float64)
    previous_time = 0.0
    for k, (b_raw, t_raw) in enumerate(zip(bounds.tolist(), times.tolist(), strict=True)):
        b, t = float(b_raw), float(t_raw)
        lo = -b if two_sided else -b - 9.0 * math.sqrt(t) - 1.0
        sd = math.sqrt(t - previous_time)
        before = float(np.sum(density * weights))
        inside = np.asarray(cdf((b - grid) / sd) - cdf((lo - grid) / sd), dtype=np.float64)
        stay = float(np.sum(density * weights * inside))
        spent[k] = before - stay
        new_grid: npt.NDArray[np.float64] = np.linspace(lo, b, _GRID)
        kernel = np.exp(-0.5 * ((new_grid[:, None] - grid[None, :]) / sd) ** 2) / (
            sd * math.sqrt(2.0 * math.pi)
        )
        density = kernel @ (density * weights)
        weights = np.full(_GRID, float(new_grid[1] - new_grid[0]), dtype=np.float64)
        weights[[0, -1]] *= 0.5
        grid, previous_time = new_grid, t
    return spent


def _alpha_spent_total(
    scale: float, kind: Kind, times: npt.NDArray[np.float64], *, two_sided: bool
) -> float:
    bounds = scale * np.sqrt(times) if kind == "pocock" else np.full(times.size, scale)
    return float(_crossing_probabilities(bounds, times, two_sided=two_sided).sum())


def _solve_scale(
    kind: Kind, times: npt.NDArray[np.float64], alpha: float, *, two_sided: bool
) -> float:
    """Return the boundary scale whose overall error rate is ``alpha``, by the secant method."""

    def gap(scale: float) -> float:
        return _alpha_spent_total(scale, kind, times, two_sided=two_sided) - alpha

    x0, x1 = 2.0, 2.6
    f0, f1 = gap(x0), gap(x1)
    for _ in range(40):
        if abs(f1) < _TOLERANCE or f1 == f0:
            break
        x2 = min(max(x1 - f1 * (x1 - x0) / (f1 - f0), 0.5), 10.0)
        x0, f0, x1, f1 = x1, f1, x2, gap(x2)
    return x1


def group_sequential_bounds(
    n_looks: int,
    *,
    kind: Kind = "obrien_fleming",
    alpha: float = 0.05,
    two_sided: bool = True,
    times: npt.ArrayLike | None = None,
) -> GroupSequentialDesign:
    """Return the critical value at each of ``n_looks`` planned looks.

    Args:
        n_looks: Number of looks, from 1 to 20.
        kind: ``"obrien_fleming"`` (default) or ``"pocock"``.
        alpha: The overall error rate, in (0, 0.5).
        two_sided: Stop for a large ``|Z|`` (default), or for a large ``Z``.
        times: Information time of each look (increasing, ending at 1); equally spaced when
            omitted.

    Raises:
        ValueError: If ``n_looks`` is outside 1 to 20, ``alpha`` is out of range, ``kind`` is
            unknown, or ``times`` is not increasing, positive and ending at 1.
    """
    if int(n_looks) != n_looks or not 1 <= n_looks <= 20:  # noqa: PLR2004
        raise ValueError(f"n_looks must be a whole number from 1 to 20, got {n_looks}")
    if not 0.0 < alpha < 0.5:  # noqa: PLR2004
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if kind not in ("obrien_fleming", "pocock"):
        raise ValueError(f"kind must be 'obrien_fleming' or 'pocock', got {kind!r}")
    t = (
        np.arange(1, n_looks + 1) / n_looks
        if times is None
        else np.asarray(times, dtype=np.float64).ravel()
    )
    if (
        t.size != n_looks
        or t[0] <= 0.0
        or np.any(np.diff(t) <= 0.0)
        or abs(t[-1] - 1.0) > _TIME_TOLERANCE
    ):
        raise ValueError("times must have n_looks values, increasing, positive, ending at 1")
    scale = _solve_scale(kind, t, alpha, two_sided=two_sided)
    bounds = scale * np.sqrt(t) if kind == "pocock" else np.full(n_looks, scale)
    z = bounds / np.sqrt(t)
    tail = 2.0 if two_sided else 1.0
    return GroupSequentialDesign(
        kind=kind,
        alpha=alpha,
        two_sided=two_sided,
        times=tuple(float(v) for v in t),
        z_bounds=tuple(float(v) for v in z),
        nominal_p=tuple(tail * (1.0 - _NORMAL.cdf(float(v))) for v in z),
        alpha_spent=tuple(
            float(v) for v in _crossing_probabilities(bounds, t, two_sided=two_sided)
        ),
    )
