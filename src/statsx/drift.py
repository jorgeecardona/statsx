"""Compare two samples for drift: KS, MMD, a classifier two-sample test, and PSI (DI-3).

The question is whether sample ``a`` (a reference) and sample ``b`` (new data) come from the same
distribution. Each test looks for a different kind of change, so the choice matters.

- ``ks_test``: one number per row. The statistic is the largest gap between the two empirical
  distribution functions. The p-value is exact when there are no ties and ``n * m <= 10000``, and
  the Smirnov limit otherwise. It sees any change in one variable, and is weak for a change in
  the tails only.
- ``mmd_test``: rows with several columns. The statistic is the unbiased squared maximum mean
  discrepancy with a Gaussian kernel, and the p-value comes from a permutation test. It needs the
  ``(n + m)^2`` kernel matrix in memory.
- ``classifier_test``: rows with several columns. A logistic regression learns to tell the two
  samples apart on half of the rows. The statistic is its AUC on the other half; 0.5 means no
  drift. The p-value is the exact Mann-Whitney test, valid because under no drift the held-out
  scores of the two samples are exchangeable. A linear classifier sees shifts of the mean; set
  ``quadratic=True`` to see changes of spread and correlation as well.
- ``psi``: one number per row, or counts you already binned. The population stability index is
  ``sum((b_i - a_i) * ln(b_i / a_i))`` over bins of the reference. It equals the symmetrized
  Kullback-Leibler divergence of the two binned distributions. The p-value is approximate.

References:
    Smirnov, N. (1948). Table for estimating the goodness of fit of empirical distributions.
    *Annals of Mathematical Statistics* 19(2):279-281. https://doi.org/10.1214/aoms/1177730256.
    The limit distribution.

    Hodges, J. L. (1958). The significance probability of the Smirnov two-sample test. *Arkiv for
    Matematik* 3(5):469-486. https://doi.org/10.1007/BF02589501. The exact distribution by a count
    of lattice paths.

    Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., & Smola, A. (2012). A kernel
    two-sample test. *Journal of Machine Learning Research* 13:723-773.
    https://jmlr.org/papers/v13/gretton12a.html. The unbiased statistic and the median bandwidth.

    Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical
    Applications in Genetics and Molecular Biology* 9(1):Article 39.
    https://doi.org/10.2202/1544-6115.1585.

    Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.
    https://openreview.net/forum?id=SJkXfE5xx.

    Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is
    stochastically larger than the other. *Annals of Mathematical Statistics* 18(1):50-60.
    https://doi.org/10.1214/aoms/1177730491.

    Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley. The population stability index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx._ranks import midranks

KS_EXACT_LIMIT = 10_000
"""Largest ``n * m`` for which the KS p-value is exact (without ties)."""

_PSI_FLOOR = 1e-4
_SERIES_TOLERANCE = 1e-16
_MIN_PER_CLASS = 10


@dataclass(frozen=True, slots=True)
class DriftTest:
    """The result of a two-sample drift test.

    Args:
        test: The test that was run.
        statistic: The test statistic (see each function).
        p_value: The p-value; small values mean the samples differ. Approximate for ``psi``.
        n_a: The number of rows of the reference sample.
        n_b: The number of rows of the new sample.
        assumption: The assumption behind the p-value, in words.
    """

    test: str
    statistic: float
    p_value: float
    n_a: int
    n_b: int
    assumption: str


def _column(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)):  # noqa: PLR2004
        raise ValueError(f"{name} must hold at least 2 finite numbers")
    return x


def _rows(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 2 or not np.all(np.isfinite(x)):  # noqa: PLR2004
        raise ValueError(f"{name} must be rows of finite numbers, at least 2 rows")
    return x


def kolmogorov_sf(lam: float) -> float:
    """Return the Kolmogorov limit ``P(K >= lam) = 2 sum (-1)^(k-1) exp(-2 k^2 lam^2)``."""
    if lam < 0.2:  # noqa: PLR2004 (the series needs many terms; the value is 1 to 1e-10)
        return 1.0
    total = 0.0
    for k in range(1, 200):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < _SERIES_TOLERANCE:
            break
    return min(max(total, 0.0), 1.0)


def _ks_exact_sf(n: int, m: int, gap: int) -> float:
    """Return P(D >= gap / (n m)) without ties, by counting lattice paths (Hodges 1958).

    ``f[i][j]`` is the share of the paths to ``(i, j)`` that stay where ``|i m - j n| < gap``.
    """
    previous = np.zeros(m + 1)
    for i in range(n + 1):
        current = np.zeros(m + 1)
        for j in range(m + 1):
            if abs(i * m - j * n) >= gap:
                continue
            if i == 0 and j == 0:
                current[0] = 1.0
                continue
            up = i * previous[j] if i > 0 else 0.0
            left = j * current[j - 1] if j > 0 else 0.0
            current[j] = (up + left) / (i + j)
        previous = current
    return float(min(max(1.0 - previous[m], 0.0), 1.0))


def ks_test(a: npt.ArrayLike, b: npt.ArrayLike) -> DriftTest:
    """Test whether two samples of one variable come from the same distribution.

    The statistic is ``D = max |F_a - F_b|``. With no ties and ``n * m <= 10000`` the p-value is
    exact. Otherwise it is the Smirnov limit at ``D * sqrt(n m / (n + m))``.

    Raises:
        ValueError: If a sample has fewer than 2 values or a value is not finite.
    """
    x, y = _column(a, "a"), _column(b, "b")
    n, m = x.size, y.size
    pooled = np.concatenate([x, y])
    grid = np.unique(pooled)
    fx = np.searchsorted(np.sort(x), grid, side="right") / n
    fy = np.searchsorted(np.sort(y), grid, side="right") / m
    d = float(np.max(np.abs(fx - fy)))
    has_ties = grid.size < pooled.size
    if not has_ties and n * m <= KS_EXACT_LIMIT:
        gap = round(d * n * m)
        p = _ks_exact_sf(n, m, gap)
        how = "exact distribution, no ties"
    else:
        p = kolmogorov_sf(d * math.sqrt(n * m / (n + m)))
        how = "Smirnov limit distribution"
    return DriftTest(
        "Kolmogorov-Smirnov",
        d,
        p,
        n,
        m,
        f"each sample is independent and identically distributed; {how}",
    )


def _pairwise_sq_dist(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sq = np.sum(z * z, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)


def mmd_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    bandwidth: float | None = None,
    n_permutations: int = 1000,
    seed: int = 0,
) -> DriftTest:
    """Test two samples of rows with several columns by the maximum mean discrepancy.

    The statistic is the unbiased estimate of ``MMD^2`` with the kernel
    ``exp(-|x - y|^2 / (2 h^2))``. ``h`` is the median of the pairwise distances in the pooled
    sample unless ``bandwidth`` is given. The p-value is ``(1 + #{permuted >= observed}) /
    (1 + n_permutations)`` over random re-labelling of the pooled rows (Phipson and Smyth 2010).
    The statistic is near 0 without drift and can be slightly negative.

    Raises:
        ValueError: If a sample has fewer than 2 rows, the columns differ, a value is not finite,
            ``bandwidth`` is not positive, or ``n_permutations`` is below 1.
    """
    x, y = _rows(a, "a"), _rows(b, "b")
    if x.shape[1] != y.shape[1]:
        raise ValueError("a and b must have the same number of columns")
    if n_permutations < 1 or (bandwidth is not None and not bandwidth > 0.0):
        raise ValueError("n_permutations must be at least 1 and bandwidth positive")
    n, m = x.shape[0], y.shape[0]
    z = np.vstack([x, y])
    sq = _pairwise_sq_dist(z)
    if bandwidth is None:
        upper = sq[np.triu_indices(n + m, k=1)]
        bandwidth = float(np.sqrt(np.median(upper)))
        if bandwidth <= 0.0:
            raise ValueError("all rows are equal: the bandwidth would be 0")
    kernel = np.exp(-sq / (2.0 * bandwidth**2))
    np.fill_diagonal(kernel, 0.0)

    def statistic(indicator: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Columns of ``indicator`` are 0/1 vectors that mark the rows labelled as sample a.
        other = 1.0 - indicator
        ku = kernel @ indicator
        kv = kernel @ other
        xx = np.sum(indicator * ku, axis=0)
        yy = np.sum(other * kv, axis=0)
        xy = np.sum(indicator * kv, axis=0)
        return xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2.0 * xy / (n * m)

    observed_indicator = np.zeros((n + m, 1))
    observed_indicator[:n, 0] = 1.0
    observed = float(statistic(observed_indicator)[0])
    rng = np.random.default_rng(seed)
    perms = np.zeros((n + m, n_permutations))
    for column in range(n_permutations):
        perms[rng.permutation(n + m)[:n], column] = 1.0
    permuted = statistic(perms)
    p = (1.0 + float(np.sum(permuted >= observed - 1e-15))) / (1.0 + n_permutations)
    return DriftTest(
        "MMD (Gaussian kernel, permutation)",
        observed,
        p,
        n,
        m,
        f"rows are independent and identically distributed in each sample; bandwidth {bandwidth:.4g}",
    )


def _features(z: npt.NDArray[np.float64], *, quadratic: bool) -> npt.NDArray[np.float64]:
    if not quadratic:
        return z
    d = z.shape[1]
    products = [z[:, i] * z[:, j] for i in range(d) for j in range(i, d)]
    return np.column_stack([z, *products])


def _logistic_fit(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], l2: float
) -> npt.NDArray[np.float64]:
    """Return the weights (the last is the intercept) of an L2-penalised logistic regression."""
    design = np.column_stack([x, np.ones(x.shape[0])])
    w = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -35.0, 35.0)))
        grad = design.T @ (p - y) + penalty * w
        hess = (design * (p * (1.0 - p))[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hess + 1e-10 * np.eye(w.size), grad)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-9:  # noqa: PLR2004
            break
    return w


def classifier_test(  # noqa: PLR0913 (settings are keyword-only)
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    quadratic: bool = False,
    l2: float = 1.0,
    test_fraction: float = 0.5,
    seed: int = 0,
) -> DriftTest:
    """Test two samples of rows by how well a classifier tells them apart.

    A logistic regression with an L2 penalty ``l2`` is trained on the rows outside a random held-out
    share, with the same share held out from each sample. The statistic is its AUC on the held-out
    rows. The p-value is the two-sided Mann-Whitney test of the held-out scores, with a continuity
    and tie correction; it is valid with no drift because the scores of the two samples are then
    exchangeable.

    Args:
        a: The reference rows.
        b: The new rows.
        quadratic: Add squares and products of the columns, so changes of spread and correlation
            can be found as well as shifts of the mean.
        l2: The penalty on the weights; the columns are standardized first.
        test_fraction: The share of each sample held out, in (0, 1).
        seed: The seed of the split.

    Raises:
        ValueError: If the samples are malformed, the columns differ, ``test_fraction`` is not in
            (0, 1), or fewer than 10 rows of a sample are left in the training or the test part.
    """
    x, y = _rows(a, "a"), _rows(b, "b")
    if x.shape[1] != y.shape[1]:
        raise ValueError("a and b must have the same number of columns")
    if not 0.0 < test_fraction < 1.0 or l2 < 0.0:
        raise ValueError("test_fraction must be in (0, 1) and l2 not negative")
    rng = np.random.default_rng(seed)

    def split(
        rows: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        order = rng.permutation(rows.shape[0])
        cut = round(rows.shape[0] * test_fraction)
        return rows[order[cut:]], rows[order[:cut]]

    train_a, test_a = split(x)
    train_b, test_b = split(y)
    if min(len(train_a), len(train_b), len(test_a), len(test_b)) < _MIN_PER_CLASS:
        raise ValueError("at least 10 rows of each sample must be left for training and testing")
    train = _features(np.vstack([train_a, train_b]), quadratic=quadratic)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    labels = np.concatenate([np.zeros(len(train_a)), np.ones(len(train_b))])
    w = _logistic_fit((train - mean) / scale, labels, l2)

    def score(rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        feats = (_features(rows, quadratic=quadratic) - mean) / scale
        return np.asarray(feats @ w[:-1] + w[-1], dtype=np.float64)

    s_a, s_b = score(test_a), score(test_b)
    n1, n2 = len(s_a), len(s_b)
    ranks = midranks(np.concatenate([s_a, s_b]))
    u = float(ranks[n1:].sum() - n2 * (n2 + 1) / 2.0)
    auc = u / (n1 * n2)
    _, counts = np.unique(np.concatenate([s_a, s_b]), return_counts=True)
    total = n1 + n2
    variance = (
        n1 * n2 / 12.0 * (total + 1.0 - float(np.sum(counts**3 - counts)) / (total * (total - 1.0)))
    )
    z = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0) / math.sqrt(variance) if variance > 0.0 else 0.0
    return DriftTest(
        "classifier two-sample test (AUC)",
        auc,
        math.erfc(z / math.sqrt(2.0)),
        x.shape[0],
        y.shape[0],
        "held-out scores of the two samples are exchangeable without drift; "
        f"{'quadratic' if quadratic else 'linear'} logistic regression, {n1 + n2} held-out rows",
    )


def psi_from_shares(
    reference: npt.ArrayLike, current: npt.ArrayLike, *, floor: float = _PSI_FLOOR
) -> float:
    """Return ``sum((b_i - a_i) * ln(b_i / a_i))`` for two sets of bin shares.

    Shares below ``floor`` are raised to ``floor`` first, because an empty bin has an infinite
    logarithm.

    Raises:
        ValueError: If the shares differ in length, are negative, or do not each add up to 1.
    """
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(current, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2 or np.any(a < 0.0) or np.any(b < 0.0):  # noqa: PLR2004
        raise ValueError("shares must be two non-negative vectors of the same length, at least 2")
    if abs(a.sum() - 1.0) > 1e-9 or abs(b.sum() - 1.0) > 1e-9:  # noqa: PLR2004
        raise ValueError("each set of shares must add up to 1")
    a, b = np.maximum(a, floor), np.maximum(b, floor)
    return float(np.sum((b - a) * np.log(b / a)))


def psi(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    n_bins: int = 10,
    floor: float = _PSI_FLOOR,
) -> DriftTest:
    """Return the population stability index of a variable, with an approximate p-value.

    The bins are the quantiles of the reference sample (equal shares in it), and the first and last
    bin are open. The p-value uses ``n_eff * PSI ~ chi-square(bins - 1)`` with
    ``n_eff = n m / (n + m)``, which follows from the second-order expansion
    ``(b - a) ln(b / a) ~ (b - a)^2 / a``. It is approximate: check it by simulation for small
    samples or many bins. Common rules of thumb (0.1 and 0.25) are conventions, not tests.

    Raises:
        ValueError: If a sample has fewer than 2 values, ``n_bins`` is below 2, or the reference
            has fewer than ``n_bins`` distinct values.
    """
    x, y = _column(reference, "reference"), _column(current, "current")
    if n_bins < 2:  # noqa: PLR2004
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    inner = np.unique(np.quantile(x, np.arange(1, n_bins) / n_bins))
    if inner.size < n_bins - 1:
        raise ValueError("the reference has too few distinct values for that many bins")
    a = np.bincount(np.searchsorted(inner, x, side="right"), minlength=n_bins) / x.size
    b = np.bincount(np.searchsorted(inner, y, side="right"), minlength=n_bins) / y.size
    value = psi_from_shares(a, b, floor=floor)
    n_eff = x.size * y.size / (x.size + y.size)
    p = float(_accel.chi2_sf(np.array([n_eff * value]), float(n_bins - 1))[0])
    return DriftTest(
        "population stability index",
        value,
        p,
        x.size,
        y.size,
        f"independent rows; {n_bins} bins of equal share in the reference; chi-square approximation",
    )
