"""Clustering: k-means, Gaussian mixtures, DBSCAN and HDBSCAN, and ways to choose the number.

``kmeans`` runs Lloyd's iteration from k-means++ seeds, several times, and keeps the fit with the
lowest within-cluster sum of squares. The kernel is Rust and uses no library besides NumPy at the
Python level. A fit does not depend on the number of threads.

k-means finds the partition that minimizes the within-cluster sum of squares. That is the maximum
likelihood partition when the clusters are round, of equal spread and of similar size. When they
are not, k-means still returns k clusters: that it returned a result is no evidence the clusters
exist. ``choose_k`` compares the data with a uniform reference, so a result of ``k = 1`` is the
library saying it sees no structure. ``silhouette`` shows which points sit between clusters.

``gaussian_mixture`` fits a mixture of Gaussians by expectation maximization. Each point gets a
probability for each component, and each component has its own mean and covariance. The model
assumes the components are Gaussian. ``choose_components`` picks the number of components by BIC.
When a cluster is skewed or has heavy tails, BIC adds components to describe it: a larger number
than the clusters you expect is a sign the Gaussian assumption fails, not a finding.

``dbscan`` and ``hdbscan`` find groups of any shape from the density of the points, label the
points that belong to no group as noise (-1), and need no number of clusters. ``dbscan`` cuts the
density at one distance ``eps``. ``hdbscan`` keeps the groups that persist over all distances, so
groups of different density are found together. Distances are exact, with no neighbour index; the
cost is O(n^2 d) time. Both assume that distance in your space measures similarity. A result of
all noise is the library saying it sees no group at those settings.

References:
    Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information
    Theory* 28(2):129-137. https://doi.org/10.1109/TIT.1982.1056489.

    Arthur, D., & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding.
    *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, pp. 1027-1035.

    Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of
    cluster analysis. *Journal of Computational and Applied Mathematics* 20:53-65.
    https://doi.org/10.1016/0377-0427(87)90125-7.

    Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for
    discovering clusters in large spatial databases with noise. *Proceedings of the 2nd
    International Conference on Knowledge Discovery and Data Mining (KDD-96)*, pp. 226-231.

    Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
    hierarchical density estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD
    2013)*, LNCS 7819, pp. 160-172. https://doi.org/10.1007/978-3-642-37456-2_14.

    Campello, R. J. G. B., Moulavi, D., Zimek, A., & Sander, J. (2015). Hierarchical density
    estimates for data clustering, visualization, and outlier detection. *ACM Transactions on
    Knowledge Discovery from Data* 10(1), article 5. https://doi.org/10.1145/2733381.

    McInnes, L., & Healy, J. (2017). Accelerated hierarchical density based clustering. *2017 IEEE
    International Conference on Data Mining Workshops (ICDMW)*, pp. 33-42.
    https://doi.org/10.1109/ICDMW.2017.12.

    Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data
    via the EM algorithm. *Journal of the Royal Statistical Society B* 39(1):1-38.
    https://doi.org/10.1111/j.2517-6161.1977.tb01600.x.

    Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*
    6(2):461-464. https://doi.org/10.1214/aos/1176344136.

    Fraley, C., & Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and
    density estimation. *Journal of the American Statistical Association* 97(458):611-631.
    https://doi.org/10.1198/016214502760047131.

    McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.

    Keribin, C. (2000). Consistent estimation of the order of mixture models. *Sankhya A*
    62(1):49-66.

    Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data
    set via the gap statistic. *Journal of the Royal Statistical Society B* 63(2):411-423.
    https://doi.org/10.1111/1467-9868.00293.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx._arrays import MATRIX, as_points, resolve_threads

_BLOCK = 512
"""Rows per block of the silhouette distance matrix."""


@dataclass(frozen=True, slots=True)
class KMeansFit:
    """A k-means fit.

    Args:
        centroids: The mean of each cluster, shape (k, d).
        labels: The cluster of each row, shape (n,).
        inertia: The within-cluster sum of squared distances to the centroid (the objective).
        n_iter: Lloyd iterations of the kept run.
        converged: True when the kept run stopped because labels or centroids stopped moving,
            False when it hit ``max_iter``.
        n_init: Runs from different seeds; the kept one has the lowest inertia.
    """

    centroids: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64]
    inertia: float
    n_iter: int
    converged: bool
    n_init: int

    @property
    def k(self) -> int:
        """Return the number of clusters."""
        return self.centroids.shape[0]

    @property
    def sizes(self) -> npt.NDArray[np.int64]:
        """Return the number of rows in each cluster."""
        return np.bincount(self.labels, minlength=self.k).astype(np.int64)

    def predict(self, x: npt.ArrayLike, *, threads: int | None = None) -> npt.NDArray[np.int64]:
        """Return the nearest centroid of each row of ``x``.

        Raises:
            ValueError: If ``x`` does not have the columns of the fit.
        """
        labels, _ = self.distances(x, threads=threads)
        return labels

    def distances(
        self, x: npt.ArrayLike, *, threads: int | None = None
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the nearest centroid of each row of ``x`` and the squared distance to it."""
        a = np.ascontiguousarray(x, dtype=np.float64)
        if a.ndim != MATRIX or a.shape[1] != self.centroids.shape[1]:
            raise ValueError(f"x must have {self.centroids.shape[1]} columns")
        labels, dist = _accel.kmeans_assign(a, self.centroids, resolve_threads(threads))
        return np.asarray(labels, dtype=np.int64), np.asarray(dist, dtype=np.float64)


def kmeans(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 0,
    threads: int | None = None,
) -> KMeansFit:
    """Return the k-means fit of the rows of ``x`` with ``k`` clusters.

    Each of ``n_init`` runs seeds ``k`` centres by k-means++ (Arthur & Vassilvitskii 2007) and
    iterates Lloyd's assign-and-average steps (Lloyd 1982) until no label changes, or the centres
    move less than ``tol`` times the mean variance of the columns, or ``max_iter`` is reached. An
    empty cluster moves onto the points worst served by their centroid. The run with the lowest
    inertia is kept. The same ``seed`` gives the same fit, for any ``threads``.

    Args:
        x: The points, shape (n, d), finite.
        k: Number of clusters, from 1 to n.
        n_init: Runs from different seeds. One run can end in a poor local minimum.
        max_iter: Most Lloyd iterations per run.
        tol: Stop when the sum of squared centre moves is below ``tol`` times the mean variance of
            the columns of ``x``.
        seed: Seed of the generator.
        threads: Worker threads. The default is the number of cores, at most 4. The result does
            not depend on it.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array with at least 2 rows, ``k`` is not in
            [1, n], or ``n_init``, ``max_iter`` or ``tol`` are out of range.
    """
    a = as_points(x)
    n = a.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and n = {n}, got {k}")
    if n_init < 1 or max_iter < 1 or tol < 0.0:
        raise ValueError("n_init and max_iter must be at least 1, and tol at least 0")
    scale = float(np.mean(np.var(a, axis=0)))
    labels, centroids, inertia, n_iter, converged = _accel.kmeans_fit(
        a, k, max_iter, tol * scale, seed, n_init, resolve_threads(threads)
    )
    return KMeansFit(
        np.asarray(centroids, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        float(inertia),
        int(n_iter),
        bool(converged),
        n_init,
    )


@dataclass(frozen=True, slots=True)
class Silhouette:
    """The silhouette of a labelling (Rousseeuw 1987).

    For a point ``i`` in cluster ``A``, ``a(i)`` is its mean distance to the other points of ``A``
    and ``b(i)`` the smallest mean distance to the points of another cluster. The silhouette is
    ``(b - a) / max(a, b)``, from -1 to 1. A point alone in its cluster has 0.

    Args:
        values: The silhouette of each point.
        labels: The labels it was computed for.
    """

    values: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64]

    @property
    def mean(self) -> float:
        """Return the mean silhouette over all points."""
        return float(self.values.mean())

    @property
    def by_cluster(self) -> npt.NDArray[np.float64]:
        """Return the mean silhouette of each cluster, indexed by label."""
        k = int(self.labels.max()) + 1
        sums = np.bincount(self.labels, weights=self.values, minlength=k)
        counts = np.bincount(self.labels, minlength=k)
        return np.divide(sums, counts, out=np.zeros(k), where=counts > 0)

    @property
    def n_negative(self) -> int:
        """Return how many points sit nearer another cluster than their own."""
        return int(np.count_nonzero(self.values < 0.0))


def silhouette(x: npt.ArrayLike, labels: npt.ArrayLike) -> Silhouette:
    """Return the silhouette of each point for a labelling of the rows of ``x``.

    The cost is O(n^2 d) in time and O(512 n) in memory (distances are taken 512 rows at a time).
    Use a sample for more than about 20,000 rows.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, the labels do not match its rows, or
            there are fewer than 2 clusters.
    """
    a = as_points(x)
    lab = np.asarray(labels).ravel()
    if lab.size != a.shape[0] or not np.issubdtype(lab.dtype, np.integer) or lab.min() < 0:
        raise ValueError("labels must be one non-negative integer per row of x")
    lab = lab.astype(np.int64)
    k = int(lab.max()) + 1
    counts = np.bincount(lab, minlength=k)
    if np.count_nonzero(counts) < 2:  # noqa: PLR2004
        raise ValueError("the silhouette needs at least 2 clusters")
    onehot = np.zeros((a.shape[0], k))
    onehot[np.arange(a.shape[0]), lab] = 1.0
    sq = np.einsum("ij,ij->i", a, a)
    values = np.empty(a.shape[0])
    for start in range(0, a.shape[0], _BLOCK):
        rows = slice(start, start + _BLOCK)
        d2 = sq[rows, None] + sq[None, :] - 2.0 * (a[rows] @ a.T)
        dist = np.sqrt(np.maximum(d2, 0.0))
        sums = dist @ onehot  # distance from each row to each cluster, summed
        own = lab[rows]
        idx = np.arange(own.size)
        size_own = counts[own]
        within = np.where(size_own > 1, sums[idx, own] / np.maximum(size_own - 1, 1), 0.0)
        other = sums / np.maximum(counts[None, :], 1)
        other[idx, own] = np.inf
        other[:, counts == 0] = np.inf
        nearest = other.min(axis=1)
        top = np.maximum(within, nearest)
        values[rows] = np.where((size_own > 1) & (top > 0.0), (nearest - within) / top, 0.0)
    return Silhouette(values, lab)


ReferenceKind = Literal["pca", "box"]


@dataclass(frozen=True, slots=True)
class GapResult:
    """The gap statistic over a range of cluster counts (Tibshirani, Walther & Hastie 2001).

    ``gap[k]`` is the mean over reference data sets of ``log W_k`` minus ``log W_k`` of the data,
    where ``W_k`` is the within-cluster sum of squares of the best k-means fit. A large gap means
    the data are more tightly clustered at ``k`` than uniform data are.

    Args:
        ks: The cluster counts tried, from 1.
        gap: The gap at each k.
        se: The standard error of the reference ``log W_k``, times ``sqrt(1 + 1/B)``.
        log_inertia: ``log W_k`` of the data at each k.
        k: The smallest k with ``gap[k] >= gap[k + 1] - se[k + 1]``. When no k meets it, the
            largest k tried, and ``found`` is False.
        found: True when a k met the rule.
        reference: ``"pca"`` or ``"box"``.
        n_reference: Number of reference data sets.
    """

    ks: tuple[int, ...]
    gap: npt.NDArray[np.float64]
    se: npt.NDArray[np.float64]
    log_inertia: npt.NDArray[np.float64]
    k: int
    found: bool
    reference: ReferenceKind
    n_reference: int


def _reference(
    a: npt.NDArray[np.float64], kind: ReferenceKind, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Return one uniform reference data set the size of ``a``."""
    n = a.shape[0]
    if kind == "box":
        lo, hi = a.min(axis=0), a.max(axis=0)
        return lo + rng.random(a.shape) * (hi - lo)
    centred = a - a.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ vt.T
    lo, hi = rotated.min(axis=0), rotated.max(axis=0)
    fresh = lo + rng.random((n, rotated.shape[1])) * (hi - lo)
    return fresh @ vt + a.mean(axis=0)


def choose_k(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    k_max: int = 10,
    *,
    n_reference: int = 20,
    n_init: int = 5,
    reference: ReferenceKind = "pca",
    seed: int = 0,
    threads: int | None = None,
) -> GapResult:
    """Return the gap statistic for ``k = 1 .. k_max`` and the k it picks.

    Each k-means fit uses ``n_init`` runs. The reference data sets are uniform over the range of
    the data: ``"pca"`` aligns the box with the principal axes (the version Tibshirani, Walther
    and Hastie recommend), ``"box"`` uses the axes of the columns. The pick is the smallest k with
    ``gap[k] >= gap[k + 1] - se[k + 1]``.

    A pick of ``k = 1`` says the data are not more clustered than uniform data in their range. It
    does not say the data are one round cluster. Clusters that touch, or that differ a lot in
    spread, may not be found. The gap compares with one null model; look at ``silhouette`` too.

    Args:
        x: The points, shape (n, d), finite.
        k_max: Largest number of clusters tried, at most n.
        n_reference: Reference data sets, at least 2.
        n_init: k-means runs per fit.
        reference: ``"pca"`` or ``"box"``.
        seed: Seed of the generator.
        threads: Worker threads, as in ``kmeans``.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, ``k_max`` is not in [1, n],
            ``n_reference`` is below 2, or ``reference`` is unknown.
    """
    a = as_points(x)
    if not 1 <= k_max <= a.shape[0]:
        raise ValueError(f"k_max must be between 1 and n = {a.shape[0]}, got {k_max}")
    if n_reference < 2:  # noqa: PLR2004
        raise ValueError(f"n_reference must be at least 2, got {n_reference}")
    if reference not in ("pca", "box"):
        raise ValueError(f"reference must be 'pca' or 'box', got {reference!r}")
    rng = np.random.default_rng(seed)
    ks = tuple(range(1, k_max + 1))

    def log_w(points: npt.NDArray[np.float64], salt: int) -> npt.NDArray[np.float64]:
        out = np.empty(len(ks))
        for i, k in enumerate(ks):
            fit = kmeans(points, k, n_init=n_init, seed=seed + salt, threads=threads)
            out[i] = math.log(max(fit.inertia, np.finfo(np.float64).tiny))
        return out

    observed = log_w(a, 0)
    refs = np.vstack([log_w(_reference(a, reference, rng), 1 + b) for b in range(n_reference)])
    gap = refs.mean(axis=0) - observed
    se = refs.std(axis=0, ddof=0) * math.sqrt(1.0 + 1.0 / n_reference)
    pick, found = ks[-1], False
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - se[i + 1]:
            pick, found = ks[i], True
            break
    return GapResult(ks, gap, se, observed, pick, found, reference, n_reference)


CovarianceKind = Literal["full", "diag", "spherical"]
Criterion = Literal["bic", "aic"]

_COVARIANCE_CODE: dict[str, int] = {"full": 0, "diag": 1, "spherical": 2}


def _check_covariance(kind: str) -> int:
    if kind not in _COVARIANCE_CODE:
        raise ValueError(f"covariance must be 'full', 'diag' or 'spherical', got {kind!r}")
    return _COVARIANCE_CODE[kind]


def _n_parameters(k: int, d: int, kind: str) -> int:
    """Free parameters of a mixture: weights, means and covariances."""
    per_component = {"full": d * (d + 1) // 2, "diag": d, "spherical": 1}[kind]
    return (k - 1) + k * d + k * per_component


@dataclass(frozen=True, slots=True)
class GaussianMixtureFit:
    """A Gaussian mixture fitted by EM.

    Args:
        weights: The share of each component, shape (k,), sums to 1.
        means: The mean of each component, shape (k, d).
        covariances: Shape (k, d, d) for ``"full"``, (k, d) for ``"diag"`` (the variances), and
            (k,) for ``"spherical"`` (one variance per component).
        covariance_type: ``"full"``, ``"diag"`` or ``"spherical"``.
        log_likelihood: The log-likelihood of the data at the fit, summed over rows.
        n_samples: Rows the model was fitted to.
        responsibilities: The probability of each component for each row, shape (n, k).
        n_iter: EM iterations of the kept run.
        converged: True when the log-likelihood changed by less than ``tol`` per row, False when
            the run stopped at ``max_iter``.
        trace: The log-likelihood at each iteration of the kept run. It does not decrease.
        n_init: Runs from different starts; the kept one has the highest log-likelihood.
        n_degenerate: Runs dropped because a component held too few points.
    """

    weights: npt.NDArray[np.float64]
    means: npt.NDArray[np.float64]
    covariances: npt.NDArray[np.float64]
    covariance_type: CovarianceKind
    log_likelihood: float
    n_samples: int
    responsibilities: npt.NDArray[np.float64]
    n_iter: int
    converged: bool
    trace: npt.NDArray[np.float64]
    n_init: int
    n_degenerate: int

    @property
    def k(self) -> int:
        """Return the number of components."""
        return self.weights.shape[0]

    @property
    def n_parameters(self) -> int:
        """Return the number of free parameters (weights, means and covariances)."""
        return _n_parameters(self.k, self.means.shape[1], self.covariance_type)

    @property
    def bic(self) -> float:
        """Return ``-2 log L + p log n`` (Schwarz 1978). Lower is better."""
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(self.n_samples)

    @property
    def aic(self) -> float:
        """Return ``-2 log L + 2 p``. Lower is better. It adds components more readily than BIC."""
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    @property
    def labels(self) -> npt.NDArray[np.int64]:
        """Return the most probable component of each fitted row."""
        return np.argmax(self.responsibilities, axis=1).astype(np.int64)

    @property
    def sizes(self) -> npt.NDArray[np.float64]:
        """Return the expected number of rows in each component (the sum of its responsibilities)."""
        return self.responsibilities.sum(axis=0)

    @property
    def uncertainty(self) -> npt.NDArray[np.float64]:
        """Return ``1 - max responsibility`` for each fitted row: 0 is sure, near 1 is a toss."""
        return 1.0 - self.responsibilities.max(axis=1)

    def _score(
        self, x: npt.ArrayLike, threads: int | None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = np.ascontiguousarray(x, dtype=np.float64)
        if a.ndim != MATRIX or a.shape[1] != self.means.shape[1]:
            raise ValueError(f"x must have {self.means.shape[1]} columns")
        resp, log_density = _accel.gmm_score(
            a,
            self.weights,
            self.means,
            np.ascontiguousarray(self.covariances.ravel()),
            _COVARIANCE_CODE[self.covariance_type],
            resolve_threads(threads),
        )
        return np.asarray(resp, dtype=np.float64), np.asarray(log_density, dtype=np.float64)

    def predict_proba(
        self, x: npt.ArrayLike, *, threads: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Return the probability of each component for each row of ``x``, shape (n, k)."""
        return self._score(x, threads)[0]

    def predict(self, x: npt.ArrayLike, *, threads: int | None = None) -> npt.NDArray[np.int64]:
        """Return the most probable component of each row of ``x``."""
        return np.argmax(self._score(x, threads)[0], axis=1).astype(np.int64)

    def score_samples(
        self, x: npt.ArrayLike, *, threads: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Return the log density of the mixture at each row of ``x``."""
        return self._score(x, threads)[1]

    def sample(
        self, n: int, *, seed: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Return ``n`` draws from the fitted mixture and the component each came from."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        rng = np.random.default_rng(seed)
        k, d = self.means.shape
        labels = rng.choice(k, size=n, p=self.weights).astype(np.int64)
        noise = rng.standard_normal((n, d))
        out = np.empty((n, d))
        for j in range(k):
            rows = labels == j
            if self.covariance_type == "full":
                factor = np.linalg.cholesky(self.covariances[j])
                out[rows] = self.means[j] + noise[rows] @ factor.T
            elif self.covariance_type == "diag":
                out[rows] = self.means[j] + noise[rows] * np.sqrt(self.covariances[j])
            else:
                out[rows] = self.means[j] + noise[rows] * math.sqrt(float(self.covariances[j]))
        return out, labels


def gaussian_mixture(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    k: int,
    *,
    covariance: CovarianceKind = "full",
    n_init: int = 10,
    max_iter: int = 500,
    tol: float = 1e-6,
    reg: float = 1e-6,
    seed: int = 0,
    threads: int | None = None,
) -> GaussianMixtureFit:
    """Return the Gaussian mixture with ``k`` components fitted to the rows of ``x`` by EM.

    Each run starts from a k-means partition (k-means++ seeds, as in ``kmeans``) and alternates
    the E-step, the probability of each component for each row, with the M-step, the weighted
    mean and covariance of each component (Dempster, Laird & Rubin 1977). It stops when the mean
    log-likelihood per row changes by less than ``tol``, or at ``max_iter``. The run with the
    highest log-likelihood is kept. The same ``seed`` gives the same fit, for any ``threads``.
    More runs make a poor local optimum less likely but cannot rule it out.

    A component can collapse onto a few points, where the likelihood has no upper bound (McLachlan
    and Peel 2000). A run is dropped when a component holds fewer than ``d + 1`` points in effect
    (``"full"``) or 2 (``"diag"``, ``"spherical"``). ``reg`` times the mean variance of the columns
    is added to every variance, so a covariance stays positive definite.

    The model assumes each cluster is Gaussian. A log-likelihood that rose and settled is no
    evidence of that. Skewed or heavy-tailed clusters are split into several components.

    Args:
        x: The points, shape (n, d), finite.
        k: Number of components, from 1 to n.
        covariance: ``"full"`` (any ellipse, d(d+1)/2 numbers per component), ``"diag"`` (axes
            aligned with the columns) or ``"spherical"`` (one variance per component).
        n_init: Runs from different starts. EM stops at the optimum nearest its start, so one run
            can be far from the best. The first run starts from a converged k-means partition and
            the others from k-means++ seeds after 0 or 3 Lloyd steps.
        max_iter: Most EM iterations per run.
        tol: Stop when the mean log-likelihood per row changes by less than this.
        reg: Ridge on the variances, as a fraction of the mean variance of the columns of ``x``.
        seed: Seed of the generator.
        threads: Worker threads. The default is the number of cores, at most 4. The result does
            not depend on it.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array with at least 2 rows, ``k`` is not in
            [1, n], an option is out of range, or every run was dropped as degenerate.
    """
    a = as_points(x)
    n, d = a.shape
    code = _check_covariance(covariance)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and n = {n}, got {k}")
    if n_init < 1 or max_iter < 1 or tol < 0.0 or reg < 0.0:
        raise ValueError("n_init and max_iter must be at least 1, and tol and reg at least 0")
    scale = float(np.mean(np.var(a, axis=0)))
    min_count = float(d + 1) if covariance == "full" else 2.0
    weights, means, covs, resp, log_lik, n_iter, converged, trace, dropped = _accel.gmm_fit(
        a, k, code, max_iter, tol, reg * scale, min_count, seed, n_init, resolve_threads(threads)
    )
    shape = {"full": (k, d, d), "diag": (k, d), "spherical": (k,)}[covariance]
    return GaussianMixtureFit(
        np.asarray(weights, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(covs, dtype=np.float64).reshape(shape),
        covariance,
        float(log_lik),
        n,
        np.asarray(resp, dtype=np.float64),
        int(n_iter),
        bool(converged),
        np.asarray(trace, dtype=np.float64),
        n_init,
        int(dropped),
    )


@dataclass(frozen=True, slots=True)
class ComponentSearch:
    """Mixtures with 1 to ``k_max`` components, compared by BIC or AIC.

    Args:
        ks: The component counts tried, from 1.
        bic: BIC at each k. NaN where every run was dropped as degenerate.
        aic: AIC at each k, NaN in the same places.
        log_likelihood: The log-likelihood at each k, NaN in the same places.
        k: The k with the lowest value of ``criterion``.
        criterion: ``"bic"`` or ``"aic"``.
        covariance_type: The covariance form used for every k.
        best: The fit at ``k``.
    """

    ks: tuple[int, ...]
    bic: npt.NDArray[np.float64]
    aic: npt.NDArray[np.float64]
    log_likelihood: npt.NDArray[np.float64]
    k: int
    criterion: Criterion
    covariance_type: CovarianceKind
    best: GaussianMixtureFit


def choose_components(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    k_max: int = 10,
    *,
    covariance: CovarianceKind = "full",
    criterion: Criterion = "bic",
    n_init: int = 10,
    max_iter: int = 500,
    tol: float = 1e-6,
    reg: float = 1e-6,
    seed: int = 0,
    threads: int | None = None,
) -> ComponentSearch:
    """Return the BIC (or AIC) of mixtures with ``k = 1 .. k_max`` components and the k it picks.

    BIC is Schwarz's criterion (1978): ``-2 log L + p log n``, where ``p`` counts weights, means
    and covariances. Its regularity conditions do not hold for mixtures, since a smaller model sits
    on the edge of a larger one. Keribin (2000) proves it still finds the true number when the
    components are Gaussian. When they are not, it adds components to describe the shape: a pick
    above the number of groups you expect points at skewed or heavy-tailed clusters.

    Args:
        x: The points, shape (n, d), finite.
        k_max: Largest number of components tried, at most n.
        covariance: ``"full"``, ``"diag"`` or ``"spherical"``, as in ``gaussian_mixture``.
        criterion: ``"bic"`` or ``"aic"``.
        n_init: EM runs per k.
        max_iter: Most EM iterations per run.
        tol: Convergence bound, as in ``gaussian_mixture``.
        reg: Ridge, as in ``gaussian_mixture``.
        seed: Seed of the generator.
        threads: Worker threads, as in ``kmeans``.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, ``k_max`` is not in [1, n], an option
            is out of range, or no k gave a fit.
    """
    a = as_points(x)
    _check_covariance(covariance)
    if not 1 <= k_max <= a.shape[0]:
        raise ValueError(f"k_max must be between 1 and n = {a.shape[0]}, got {k_max}")
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
    ks = tuple(range(1, k_max + 1))
    fits: list[GaussianMixtureFit | None] = []
    for k in ks:
        try:
            fits.append(
                gaussian_mixture(
                    a,
                    k,
                    covariance=covariance,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    reg=reg,
                    seed=seed,
                    threads=threads,
                )
            )
        except ValueError:
            fits.append(None)  # every run had a collapsed component: this k is not supported
    bic = np.array([f.bic if f else np.nan for f in fits])
    aic = np.array([f.aic if f else np.nan for f in fits])
    log_lik = np.array([f.log_likelihood if f else np.nan for f in fits])
    score = bic if criterion == "bic" else aic
    if np.all(np.isnan(score)):
        raise ValueError("no number of components gave a fit: the data have too few rows")
    pick = int(np.nanargmin(score))
    best = fits[pick]
    assert best is not None  # noqa: S101 (np.nanargmin never lands on NaN)
    return ComponentSearch(ks, bic, aic, log_lik, ks[pick], criterion, covariance, best)


@dataclass(frozen=True, slots=True)
class DBSCANFit:
    """A DBSCAN clustering (Ester, Kriegel, Sander & Xu 1996).

    A point is a core point when at least ``min_samples`` points, itself included, lie within
    ``eps`` of it. A cluster is a set of core points joined through steps of at most ``eps``,
    plus the border points within ``eps`` of one of its core points. A border point within
    ``eps`` of core points of several clusters joins the cluster discovered first, the one with
    the lowest-index core point. Every other point is noise.

    Args:
        labels: The cluster of each point, from 0, or -1 for noise.
        core_mask: True for the core points.
        eps: The radius used.
        min_samples: The neighbourhood size used.
    """

    labels: npt.NDArray[np.int64]
    core_mask: npt.NDArray[np.bool_]
    eps: float
    min_samples: int

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters, not counting noise."""
        return int(self.labels.max()) + 1 if self.labels.size else 0

    @property
    def n_noise(self) -> int:
        """Return the number of noise points."""
        return int(np.count_nonzero(self.labels < 0))

    @property
    def n_border(self) -> int:
        """Return the number of border points: in a cluster, but not core points."""
        return int(np.count_nonzero((self.labels >= 0) & ~self.core_mask))

    @property
    def sizes(self) -> npt.NDArray[np.int64]:
        """Return the number of points in each cluster, noise left out."""
        return np.bincount(self.labels[self.labels >= 0], minlength=self.n_clusters).astype(
            np.int64
        )


def dbscan(
    x: npt.ArrayLike, eps: float, *, min_samples: int = 5, threads: int | None = None
) -> DBSCANFit:
    """Return the DBSCAN clustering of the rows of ``x`` for radius ``eps``.

    Distances are Euclidean and exact. The cost is O(n^2 d) time and O(n d) memory. The result
    does not depend on the order of the rows (except that a border point shared by two clusters
    goes to the one with the lowest-index core point) or on ``threads``.

    ``eps`` has no default: it sets the scale of what counts as dense. ``k_distances`` shows
    the distance to the k-th neighbour of each point, from which Ester et al. read ``eps`` at the
    first valley of the sorted curve.

    Args:
        x: The points, shape (n, d), finite.
        eps: The radius, positive.
        min_samples: Points within ``eps``, the point itself included, that make a core point.
        threads: Worker threads, as in ``kmeans``.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, ``eps`` is not positive and finite, or
            ``min_samples`` is below 1.
    """
    a = as_points(x)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be positive and finite, got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}")
    labels, core = _accel.dbscan_fit(a, float(eps), min_samples, resolve_threads(threads))
    return DBSCANFit(
        np.asarray(labels, dtype=np.int64),
        np.asarray(core, dtype=np.bool_),
        float(eps),
        min_samples,
    )


def k_distances(x: npt.ArrayLike, k: int, *, threads: int | None = None) -> npt.NDArray[np.float64]:
    """Return the distance from each row of ``x`` to its ``k``-th nearest other row.

    The sorted curve of these values is the sorted k-dist graph of Ester et al. (1996, section
    4.2). With ``min_samples = k + 1`` in ``dbscan`` (its count includes the point itself), a
    point is a core point exactly when its k-distance is at most ``eps``.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, or ``k`` is not in [1, n - 1].
    """
    a = as_points(x)
    if not 1 <= k < a.shape[0]:
        raise ValueError(f"k must be between 1 and n - 1 = {a.shape[0] - 1}, got {k}")
    return np.asarray(_accel.kth_distances(a, k + 1, resolve_threads(threads)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CondensedTree:
    """The condensed cluster tree of HDBSCAN (Campello, Moulavi & Sander 2013).

    Clusters are numbered from 0, the root, and a parent has a lower number than its children.
    Each row says that ``child`` left cluster ``parent`` at density ``lambda_value`` (the inverse
    of the distance). A ``child`` below ``n_points`` is a point. A ``child`` of ``n_points + c``
    is the cluster ``c``, which split off from the parent there.

    Args:
        n_points: Number of points.
        parent: The cluster each row leaves.
        child: A point, or ``n_points`` plus a cluster number.
        lambda_value: The density at which the child leaves the parent.
        child_size: Points in the child (1 for a point).
        stability: The stability of each cluster: the sum over its points of ``lambda_max -
            lambda_min``, the density the point leaves at minus the density the cluster is born at.
        label: The label of each cluster if it was chosen, else -1.
    """

    n_points: int
    parent: npt.NDArray[np.int64]
    child: npt.NDArray[np.int64]
    lambda_value: npt.NDArray[np.float64]
    child_size: npt.NDArray[np.int64]
    stability: npt.NDArray[np.float64]
    label: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HDBSCANFit:
    """An HDBSCAN* clustering (Campello, Moulavi & Sander 2013; Campello et al. 2015).

    Args:
        labels: The cluster of each point, from 0, or -1 for noise.
        core_distances: The distance of each point to its ``min_samples``-th nearest point, the
            point itself being the first.
        mst: The minimum spanning tree of the mutual reachability distance, shape (n - 1, 3):
            the two points of each edge and its weight.
        tree: The condensed tree.
        min_cluster_size: The smallest group that counts as a cluster.
        min_samples: The neighbourhood size of the core distance.
        allow_single_cluster: Whether the whole data set could be returned as one cluster.
    """

    labels: npt.NDArray[np.int64]
    core_distances: npt.NDArray[np.float64]
    mst: npt.NDArray[np.float64]
    tree: CondensedTree
    min_cluster_size: int
    min_samples: int
    allow_single_cluster: bool

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters, not counting noise."""
        return int(self.labels.max()) + 1 if self.labels.size else 0

    @property
    def n_noise(self) -> int:
        """Return the number of noise points."""
        return int(np.count_nonzero(self.labels < 0))

    @property
    def sizes(self) -> npt.NDArray[np.int64]:
        """Return the number of points in each cluster, noise left out."""
        return np.bincount(self.labels[self.labels >= 0], minlength=self.n_clusters).astype(
            np.int64
        )

    @property
    def stabilities(self) -> npt.NDArray[np.float64]:
        """Return the stability of each chosen cluster, indexed by label."""
        chosen = self.tree.label >= 0
        out = np.zeros(self.n_clusters)
        out[self.tree.label[chosen]] = self.tree.stability[chosen]
        return out


def hdbscan(
    x: npt.ArrayLike,
    min_cluster_size: int = 5,
    *,
    min_samples: int | None = None,
    allow_single_cluster: bool = False,
    threads: int | None = None,
) -> HDBSCANFit:
    """Return the HDBSCAN* clustering of the rows of ``x``.

    The steps follow Campello, Moulavi and Sander (2013), with the notation of McInnes and Healy
    (2017, section 2.2). The core distance of a point is its distance to its ``min_samples``-th
    nearest point (the point itself is the first). The mutual reachability distance of two points
    is the largest of their core distances and their distance. The single linkage hierarchy of
    that distance is read from a minimum spanning tree, which is exact here. The hierarchy is
    condensed: a branch with fewer than ``min_cluster_size`` points is dropped, its points
    "falling out" of the parent at ``lambda = 1 / distance``. The stability of a cluster is the sum,
    over its points, of the density they leave at minus the density it was born at. The result is
    the set of non-overlapping clusters with the largest total stability. Points in no chosen
    cluster are noise (-1).

    The root of the tree is the whole data set. It is not a candidate unless
    ``allow_single_cluster`` is True. With the default, data with one dense group and nothing
    else comes back as noise: HDBSCAN then answers "which groups stand out from each other", not
    "is there a group here". This default is a choice of statsx (D-65), not a rule of the papers.

    The cost is O(n^2 d) time and O(n d) memory. The result does not depend on ``threads``.

    Args:
        x: The points, shape (n, d), finite.
        min_cluster_size: The smallest group that counts as a cluster, at least 2.
        min_samples: Neighbourhood size of the core distance. The default is
            ``min_cluster_size``. A larger value gives smoother density and more noise.
        allow_single_cluster: Let the root be chosen, so that the whole set can be one cluster.
        threads: Worker threads, as in ``kmeans``.

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, ``min_cluster_size`` is below 2, or
            ``min_samples`` is not in [1, n].
    """
    a = as_points(x)
    n = a.shape[0]
    if min_cluster_size < 2:  # noqa: PLR2004
        raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")
    samples = min_cluster_size if min_samples is None else min_samples
    if not 1 <= samples <= n:
        raise ValueError(f"min_samples must be between 1 and n = {n}, got {samples}")
    point, core, mst, parent, child, lam, size, stability, label = _accel.hdbscan_fit(
        a, samples, min_cluster_size, allow_single_cluster, resolve_threads(threads)
    )
    tree = CondensedTree(
        n,
        np.asarray(parent, dtype=np.int64),
        np.asarray(child, dtype=np.int64),
        np.asarray(lam, dtype=np.float64),
        np.asarray(size, dtype=np.int64),
        np.asarray(stability, dtype=np.float64),
        np.asarray(label, dtype=np.int64),
    )
    return HDBSCANFit(
        np.asarray(point, dtype=np.int64),
        np.asarray(core, dtype=np.float64),
        np.asarray(mst, dtype=np.float64),
        tree,
        min_cluster_size,
        samples,
        allow_single_cluster,
    )
