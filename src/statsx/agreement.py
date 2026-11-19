"""Agreement between raters: Cohen's kappa, Fleiss' kappa, Krippendorff's alpha.

All three compare the agreement seen with the agreement expected by chance. 1 is perfect
agreement, 0 is the level of chance, and a negative value is worse than chance. Each function
returns an ``Estimate`` with a percentile bootstrap interval that resamples the items (units).

- ``cohen_kappa``: two raters, categories as labels. Optional linear or quadratic weights make
  a near miss count less than a far miss (for ordered categories).
- ``fleiss_kappa``: any fixed number of raters per item, nominal categories, input as counts.
- ``krippendorff_alpha``: any number of raters, missing ratings allowed (a unit needs at least
  two ratings), levels nominal, ordinal, interval or ratio.

Kappa is low when one category is very common, even with high raw agreement: read it with the
category shares.

References:
    Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
    Psychological Measurement* 20(1):37-46. https://doi.org/10.1177/001316446002000104.

    Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision for scaled
    disagreement or partial credit. *Psychological Bulletin* 70(4):213-220.
    https://doi.org/10.1037/h0026256.

    Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological
    Bulletin* 76(5):378-382. https://doi.org/10.1037/h0031619.

    Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. Departmental Papers
    (ASC), University of Pennsylvania. https://repository.upenn.edu/asc_papers/43.

    Krippendorff, K. (2016). Bootstrapping distributions for Krippendorff's alpha.
    https://www.asc.upenn.edu/sites/default/files/2021-03/Alphabootstrap.pdf. Resampling units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx.bootstrap import IID, bootstrap
from statsx.result import Estimate

Weights = Literal["linear", "quadratic"]
Level = Literal["nominal", "ordinal", "interval", "ratio"]


def _kappa_from_table(table: npt.NDArray[np.float64], weight: npt.NDArray[np.float64]) -> float:
    """Return weighted kappa of a k x k table of counts, 1 on the diagonal of ``weight``."""
    p = table / table.sum()
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    observed = float(np.sum(weight * p))
    expected = float(np.sum(weight * np.outer(rows, cols)))
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def cohen_kappa(  # noqa: PLR0913 (settings are keyword-only)
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    weights: Weights | None = None,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Return Cohen's kappa of two raters' labels with a bootstrap interval over the items.

    Args:
        a: The first rater's label for each item.
        b: The second rater's label for each item.
        weights: None for plain kappa; ``"linear"`` or ``"quadratic"`` to give partial credit to
            near misses. The labels must then be numbers, and they are ordered by value.
        level: Coverage of the interval, in (0, 1).
        n_resamples: Number of bootstrap resamples of the items.
        seed: Seed of the bootstrap.

    Raises:
        ValueError: If the arrays differ in length or hold fewer than 2 items, there is only one
            category (so chance agreement is perfect), or ``weights`` is used with labels that are
            not numbers.
    """
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    if x.size != y.size or x.size < 2:  # noqa: PLR2004
        raise ValueError("a and b must have the same length, at least 2 items")
    categories = np.unique(np.concatenate([x, y]))
    k = categories.size
    if k < 2:  # noqa: PLR2004
        raise ValueError("kappa is undefined with a single category")
    if weights is None:
        weight = np.eye(k)
    else:
        if not np.issubdtype(categories.dtype, np.number):
            raise ValueError("weights need numeric labels")
        gap = np.abs(categories[:, None] - categories[None, :]).astype(np.float64)
        gap = gap / gap.max()
        weight = 1.0 - gap if weights == "linear" else 1.0 - gap**2
    index = {c: i for i, c in enumerate(categories.tolist())}
    codes = np.column_stack(
        [[index[v] for v in x.tolist()], [index[v] for v in y.tolist()]]
    ).astype(np.int64)

    def statistic(pairs: npt.NDArray[np.float64]) -> float:
        table = np.zeros((k, k))
        np.add.at(table, (pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)), 1.0)
        return _kappa_from_table(table, weight)

    result = bootstrap(
        codes.astype(np.float64), statistic, IID(), n_resamples=n_resamples, level=level, seed=seed
    )
    name = "kappa" if weights is None else f"{weights}-weighted kappa"
    return Estimate(
        result.estimate.estimate,
        result.estimate.interval,
        int(x.size),
        f"two raters, items independent; Cohen's {name}; bootstrap over items",
    )


def _fleiss(counts: npt.NDArray[np.float64]) -> float:
    n_items = counts.shape[0]
    raters = float(counts[0].sum())
    p_category: npt.NDArray[np.float64] = counts.sum(axis=0) / (n_items * raters)
    p_item: npt.NDArray[np.float64] = (np.sum(counts**2, axis=1) - raters) / (
        raters * (raters - 1.0)
    )
    p_bar, p_expected = float(p_item.mean()), float(np.sum(p_category**2))
    if p_expected == 1.0:
        return float("nan")
    return (p_bar - p_expected) / (1.0 - p_expected)


def fleiss_kappa(
    counts: npt.ArrayLike,
    *,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Return Fleiss' kappa from a table of counts, with a bootstrap interval over the items.

    Args:
        counts: Shape (items, categories); ``counts[i, j]`` is how many raters put item ``i`` in
            category ``j``. Every item must have the same number of raters, at least 2.
        level: Coverage of the interval, in (0, 1).
        n_resamples: Number of bootstrap resamples of the items.
        seed: Seed of the bootstrap.

    Raises:
        ValueError: If the table is not 2-D with at least 2 items and 2 categories, a count is
            negative or not a whole number, the row sums differ or are below 2, or all raters
            always choose one category.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 2:  # noqa: PLR2004
        raise ValueError("counts must have shape (items >= 2, categories >= 2)")
    if np.any(c < 0.0) or np.any(c != np.round(c)):
        raise ValueError("counts must be non-negative whole numbers")
    sums = c.sum(axis=1)
    if np.any(sums != sums[0]) or sums[0] < 2:  # noqa: PLR2004
        raise ValueError("every item needs the same number of raters, at least 2")
    if np.isnan(_fleiss(c)):
        raise ValueError("kappa is undefined when every rating is in one category")
    result = bootstrap(c, _fleiss, IID(), n_resamples=n_resamples, level=level, seed=seed)
    return Estimate(
        result.estimate.estimate,
        result.estimate.interval,
        int(c.shape[0]),
        f"{int(sums[0])} raters per item, items independent; Fleiss' kappa; bootstrap over items",
    )


def _coincidences(
    ratings: list[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the sorted values and the coincidence matrix of the units (Krippendorff 2011)."""
    values = np.unique(np.concatenate(ratings))
    index = {v: i for i, v in enumerate(values.tolist())}
    coincidence = np.zeros((values.size, values.size))
    for r in ratings:
        codes = np.array([index[v] for v in r.tolist()])
        for i, ci in enumerate(codes):
            for j, cj in enumerate(codes):
                if i != j:
                    coincidence[ci, cj] += 1.0 / (r.size - 1.0)
    return values, coincidence


def _squared_difference(
    values: npt.NDArray[np.float64], n_c: npt.NDArray[np.float64], level: Level
) -> npt.NDArray[np.float64]:
    """Return the matrix of squared differences between values for a level of measurement."""
    k = values.size
    if level == "nominal":
        return 1.0 - np.eye(k)
    if level == "interval":
        return (values[:, None] - values[None, :]) ** 2
    if level == "ratio":
        return ((values[:, None] - values[None, :]) / (values[:, None] + values[None, :])) ** 2
    cum = np.cumsum(n_c)
    below = cum - n_c
    delta2 = np.zeros((k, k))
    for c in range(k):
        for d in range(k):
            lo, hi = min(c, d), max(c, d)
            if c != d:
                delta2[c, d] = (cum[hi] - below[lo] - (n_c[lo] + n_c[hi]) / 2.0) ** 2
    return delta2


def _alpha(units: npt.NDArray[np.float64], level: Level) -> float:
    """Krippendorff's alpha of a (units, raters) array with NaN for a missing rating."""
    ratings = [row[~np.isnan(row)] for row in units]
    ratings = [r for r in ratings if r.size >= 2]  # noqa: PLR2004
    if not ratings:
        return float("nan")
    values, coincidence = _coincidences(ratings)
    n_c = coincidence.sum(axis=1)
    n = float(n_c.sum())
    delta2 = _squared_difference(values, n_c, level)
    observed = float(np.sum(coincidence * delta2)) / n
    expected = float(np.sum(np.outer(n_c, n_c) * delta2)) / (n * (n - 1.0))
    if expected == 0.0:
        return float("nan")
    return 1.0 - observed / expected


def krippendorff_alpha(
    ratings: npt.ArrayLike,
    *,
    level_of_measurement: Level = "nominal",
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Return Krippendorff's alpha with a bootstrap interval over the units.

    Args:
        ratings: Shape (raters, units); use ``nan`` for a rating that is missing. A unit with
            fewer than 2 ratings is left out.
        level_of_measurement: ``"nominal"`` (any two different values disagree fully),
            ``"ordinal"``, ``"interval"`` (squared difference) or ``"ratio"`` (needs values
            that are not all zero and not of opposite sign).
        level: Coverage of the interval, in (0, 1).
        n_resamples: Number of bootstrap resamples of the units.
        seed: Seed of the bootstrap.

    Raises:
        ValueError: If the array is not 2-D, fewer than 2 units have 2 or more ratings, or all
            ratings are one value (alpha is then undefined).
    """
    data = np.asarray(ratings, dtype=np.float64)
    if data.ndim != 2:  # noqa: PLR2004
        raise ValueError("ratings must have shape (raters, units)")
    units = np.ascontiguousarray(data.T)
    usable = int(np.sum(np.sum(~np.isnan(units), axis=1) >= 2))  # noqa: PLR2004
    if usable < 2 or np.isnan(_alpha(units, level_of_measurement)):  # noqa: PLR2004
        raise ValueError("alpha is undefined: need 2 units with 2 ratings and more than one value")

    def statistic(sample: npt.NDArray[np.float64]) -> float:
        return _alpha(sample, level_of_measurement)

    result = bootstrap(units, statistic, IID(), n_resamples=n_resamples, level=level, seed=seed)
    return Estimate(
        result.estimate.estimate,
        result.estimate.interval,
        usable,
        f"{level_of_measurement} measurement, missing ratings allowed, units independent; "
        "Krippendorff's alpha; bootstrap over units",
    )
