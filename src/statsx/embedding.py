"""UMAP: a low-dimensional map of points, built from a graph of nearest neighbours.

``umap`` follows McInnes, Healy & Melville (2018). It has two phases. First it builds a weighted
graph: each point is joined to its ``n_neighbors`` nearest points, the weight of an edge falls with
the distance beyond the nearest neighbour (section 3.1 of the paper), and the two directions of an
edge are merged by the probabilistic sum ``a + b - a b``. Then it places the points in
``dimensions`` dimensions so that points joined by heavy edges sit close and unjoined points sit
apart. It starts from the eigenvectors of the normalized graph Laplacian (Algorithm 4) and moves
the points by stochastic gradient descent on the fuzzy set cross entropy with negative sampling
(section 4.1, Algorithm 5).

Everything is written in Rust and NumPy: the neighbours are exact (no approximate index), the
eigenvectors come from LOBPCG, the curve ``a``, ``b`` from Levenberg-Marquardt, and the layout
from the algorithm of the paper. A result does not depend on the number of threads.

A picture of a map is not evidence of groups. UMAP keeps the neighbours of each point and gives up
distances and densities (section 6 of the paper): the size of a cluster on the map, the gap
between two clusters and the order of the clusters mean nothing. UMAP also "can tend to find
manifold structure within the noise of a dataset" (section 6), most of all with few points.
``neighbor_preservation`` measures how many of the nearest neighbours the map keeps, and
``UMAPFit.cross_entropy`` is the value of the objective that the layout lowers. Cluster the data,
not the map, when the groups matter.

Not in the paper, so not here: mapping new points into an existing map (the paper lists it as
future work, section 7), other distances than the Euclidean one, and supervised maps.

References:
    McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and
    Projection for Dimension Reduction. arXiv:1802.03426 (version 3, 2020).
    https://arxiv.org/abs/1802.03426.

    Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver: locally optimal block
    preconditioned conjugate gradient method. *SIAM Journal on Scientific Computing* 23(2):517-541.
    https://doi.org/10.1137/S1064827500366124.

    Duersch, J. A., Shao, M., Yang, C., & Gu, M. (2018). A robust and efficient implementation of
    LOBPCG. *SIAM Journal on Scientific Computing* 40(5):C655-C676.
    https://doi.org/10.1137/17M1129830.

    Levenberg, K. (1944). A method for the solution of certain non-linear problems in least
    squares. *Quarterly of Applied Mathematics* 2(2):164-168.

    Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters.
    *Journal of the Society for Industrial and Applied Mathematics* 11(2):431-441.

    Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). Visualizing large-scale and high-dimensional
    data. *Proceedings of the 25th International Conference on World Wide Web*, pp. 287-297.
    https://doi.org/10.1145/2872427.2883041.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx._arrays import as_points, resolve_threads

_CURVE_END = 3.0
"""The curve of the paper is fitted on distances in (0, 3]: ``Psi`` has fallen to ``exp(-3)``."""

_CURVE_POINTS = 300
"""Distances on that grid, spaced evenly."""

_SPECTRAL_ITERATIONS = 1000
"""Largest number of LOBPCG iterations for the start."""

_SPECTRAL_TOLERANCE = 1e-6
"""Residual norm below which an eigenvector counts as found."""

_BOX = 10.0
"""The start is scaled so that its largest coordinate is this far from the origin."""

_CELL = 25.0
"""Distance between the centres of two connected components in the start."""

_JITTER = 1e-4
"""Standard deviation of the noise added to the start, so that no two points coincide."""

_LARGE = 10_000
"""Points above which the default number of epochs drops from 500 to 200."""

_LM_ITERATIONS = 200

_MIN_NEIGHBORS = 2
"""Fewest neighbours: the weights of one neighbour always sum to 1, so sigma is undefined."""


@dataclass(frozen=True, slots=True)
class FuzzyGraph:
    """The weighted graph of the points: the 1-skeleton of the fuzzy simplicial set.

    The graph is symmetric and stored in compressed rows. The weights are in (0, 1], read as the
    probability that the two points are joined.

    Args:
        indptr: Row starts, shape (n + 1,). Row ``i`` holds ``indices[indptr[i]:indptr[i + 1]]``.
        indices: The column of each entry, ascending within a row.
        weights: The weight of each entry.
    """

    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]

    @property
    def n_vertices(self) -> int:
        """The number of points."""
        return int(self.indptr.shape[0]) - 1

    @property
    def n_edges(self) -> int:
        """The number of undirected edges."""
        return int(self.indices.shape[0]) // 2

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        """The row of each entry, shape (2 * n_edges,)."""
        return np.repeat(np.arange(self.n_vertices, dtype=np.int64), np.diff(self.indptr))

    def dense(self) -> npt.NDArray[np.float64]:
        """The (n, n) weight matrix. It holds n^2 values: use it for small n only."""
        n = self.n_vertices
        out = np.zeros((n, n))
        out[self.rows, self.indices] = self.weights
        return out


@dataclass(frozen=True, slots=True)
class UMAPFit:
    """A UMAP map and what it was built from.

    Args:
        embedding: The position of each point, shape (n, dimensions).
        graph: The weighted graph of the points.
        neighbors: The ``n_neighbors`` nearest other points of each point, shape (n, k), nearest
            first.
        distances: The distance to each of those, shape (n, k).
        rho: The distance from each point to its nearest neighbour with a distance above zero.
        sigma: The scale of each point, solved so that the weights of its neighbours sum to
            ``log2(n_neighbors)``.
        a: The ``a`` of ``1 / (1 + a r^(2 b))``, the membership of a pair at distance ``r`` in the map.
        b: The ``b`` of the same curve.
        min_dist: The distance below which the curve is fitted to 1.
        n_epochs: Passes of the layout.
        negative_sample_rate: Negative samples per sampled edge.
        seed: The seed of the start and the layout.
        n_graph_components: Connected components of the graph. Each one gets its own start.
        n_random_start: Points whose component was too small for a spectral start and got a
            random one.
        spectral_converged: True when every spectral start reached the tolerance. When False the
            start is the last iterate, which is still a smooth map of the graph but not the
            eigenvector.
    """

    embedding: npt.NDArray[np.float64]
    graph: FuzzyGraph
    neighbors: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    a: float
    b: float
    min_dist: float
    n_epochs: int
    negative_sample_rate: int
    seed: int
    n_graph_components: int
    n_random_start: int
    spectral_converged: bool

    @property
    def n_points(self) -> int:
        """The number of points."""
        return int(self.embedding.shape[0])

    @property
    def dimensions(self) -> int:
        """The dimension of the map."""
        return int(self.embedding.shape[1])

    @property
    def n_neighbors(self) -> int:
        """The number of neighbours of each point."""
        return int(self.neighbors.shape[1])

    def cross_entropy(self, *, threads: int | None = None) -> float:
        """Return the fuzzy set cross entropy of the map (Definition 10 of the paper).

        It is summed over all pairs of points, with weight 0 for a pair that is not an edge. It is
        0 when the map reproduces the graph exactly, and the layout lowers it. It costs
        O(n^2 d), so it is not stored.

        Args:
            threads: Worker threads. The value does not depend on it.
        """
        return float(
            _accel.umap_cross_entropy(
                self.embedding,
                self.graph.indptr,
                self.graph.indices,
                self.graph.weights,
                self.a,
                self.b,
                resolve_threads(threads),
            )
        )


def fit_curve(min_dist: float) -> tuple[float, float]:
    """Return ``(a, b)`` so that ``1 / (1 + a r^(2 b))`` follows the curve of the paper.

    The curve is 1 for ``r <= min_dist`` and ``exp(-(r - min_dist))`` beyond it (Definition 11).
    The fit is nonlinear least squares on 300 evenly spaced distances in (0, 3], by the
    Levenberg-Marquardt method with Marquardt's scaling, on ``log a`` and ``log b`` so that both
    stay positive. The paper does not say how the fit is done or on which distances.

    Args:
        min_dist: The distance in [0, 1] below which two points in the map count as joined.
    """
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError(f"min_dist must be in [0, 1], got {min_dist}")
    r = np.linspace(0.0, _CURVE_END, _CURVE_POINTS)[1:]
    target = np.where(r <= min_dist, 1.0, np.exp(-(r - min_dist)))
    log_r = np.log(r)

    def model(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, b = np.exp(p)
        return 1.0 / (1.0 + a * np.exp(2.0 * b * log_r))

    p = np.zeros(2)
    damping = 1e-3
    residual = model(p) - target
    sse = float(residual @ residual)
    gain = 0.0
    for _ in range(_LM_ITERATIONS):
        a, b = np.exp(p)
        t = a * np.exp(2.0 * b * log_r)
        slope = -1.0 / (1.0 + t) ** 2
        jac = np.column_stack([slope * t, slope * t * 2.0 * b * log_r])
        gradient = jac.T @ residual
        hessian = jac.T @ jac
        improved = False
        while damping < 1e12:  # noqa: PLR2004 (give up when the step is hopeless)
            step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)), -gradient)
            trial = model(p + step) - target
            trial_sse = float(trial @ trial)
            if trial_sse < sse:
                gain = sse - trial_sse
                p, residual, sse = p + step, trial, trial_sse
                damping = max(damping / 3.0, 1e-12)
                improved = True
                break
            damping *= 4.0
        if not improved or gain <= 1e-15 * sse:
            break
    a, b = np.exp(p)
    return float(a), float(b)


def _sub_graph(graph: FuzzyGraph, members: npt.NDArray[np.int64]) -> FuzzyGraph:
    """The graph on ``members``, a union of connected components, renumbered from 0."""
    position = np.full(graph.n_vertices, -1, dtype=np.int64)
    position[members] = np.arange(members.shape[0], dtype=np.int64)
    counts = graph.indptr[members + 1] - graph.indptr[members]
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    first = np.repeat(graph.indptr[members] - indptr[:-1], counts)
    entries = first + np.arange(int(indptr[-1]), dtype=np.int64)
    return FuzzyGraph(indptr, position[graph.indices[entries]], graph.weights[entries])


def _start(
    graph: FuzzyGraph, dimensions: int, seed: int, threads: int
) -> tuple[npt.NDArray[np.float64], int, int, bool]:
    """Return the start, the number of components, the points started at random and convergence.

    Each connected component gets the eigenvectors of its own normalized Laplacian after the
    trivial one (Algorithm 4), scaled to a box that holds a share of the volume equal to the
    share of the points. A component with fewer than ``dimensions + 2`` points has no such
    eigenvectors and starts at random. The components are placed on a grid. The paper only warns
    that disconnected graphs need care (section 4.3).
    """
    rng = np.random.default_rng(seed)
    n = graph.n_vertices
    labels = _accel.umap_components(graph.indptr, graph.indices)
    n_components = int(labels.max()) + 1
    members = [np.flatnonzero(labels == c) for c in range(n_components)]
    order = sorted(range(n_components), key=lambda c: (-members[c].shape[0], c))
    side = int(np.ceil(n_components ** (1.0 / dimensions)))
    largest = members[order[0]].shape[0]
    start = np.empty((n, dimensions))
    random_points = 0
    converged = True
    for rank, c in enumerate(order):
        idx = members[c]
        sub = graph if n_components == 1 else _sub_graph(graph, idx)
        found = None
        if idx.shape[0] >= dimensions + 2:
            found = _accel.umap_spectral(
                sub.indptr,
                sub.indices,
                sub.weights,
                dimensions,
                _SPECTRAL_ITERATIONS,
                _SPECTRAL_TOLERANCE,
                seed,
                threads,
            )
        if found is None:
            coords = rng.uniform(-1.0, 1.0, size=(idx.shape[0], dimensions))
            random_points += idx.shape[0]
        else:
            coords, _, ok, _ = found
            converged = converged and bool(ok)
        peak = float(np.max(np.abs(coords)))
        scale = _BOX * (idx.shape[0] / largest) ** (1.0 / dimensions)
        centre = np.array(np.unravel_index(rank, (side,) * dimensions), dtype=np.float64) * _CELL
        start[idx] = coords * (scale / peak if peak > 0.0 else 0.0) + centre
    start += rng.normal(0.0, _JITTER, size=start.shape)
    return start, n_components, random_points, converged


def umap(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    *,
    n_neighbors: int = 15,
    dimensions: int = 2,
    min_dist: float = 0.1,
    n_epochs: int | None = None,
    negative_sample_rate: int = 5,
    seed: int = 0,
    threads: int | None = None,
) -> UMAPFit:
    """Return a UMAP map of the rows of ``x``.

    The neighbours are exact, so the cost is O(n^2 d) for them and O(n_neighbors n n_epochs) for
    the layout. The result does not depend on ``threads``. A different ``seed`` gives a different
    map: the layout is a stochastic search for a minimum, not a unique answer.

    The default ``n_epochs`` is 500 up to 10,000 points and 200 above. The paper leaves the number
    to the caller. ``negative_sample_rate`` 5 is the value of LargeVis (Tang et al. 2016), whose
    sampling the paper follows.

    Args:
        x: The points, shape (n, d), finite.
        n_neighbors: Neighbours per point, from 2 to n - 1. Small values keep fine structure and
            can split the graph into pieces; large values keep the large-scale shape (section
            4.3 of the paper).
        dimensions: The dimension of the map.
        min_dist: How close two points may sit in the map, in [0, 1]. It changes the look of the
            map, not the graph (section 4.3 of the paper).
        n_epochs: Passes over the edges.
        negative_sample_rate: Points drawn at random per sampled edge to push away from.
        seed: Seed of the start and the layout.
        threads: Worker threads. Defaults to the CPU count, at most 4.

    Raises:
        ValueError: If ``n_neighbors`` is not in [2, n - 1], or a setting is out of range.
    """
    a = as_points(x)
    n = a.shape[0]
    if not _MIN_NEIGHBORS <= n_neighbors <= n - 1:
        raise ValueError(f"n_neighbors must be in [2, n - 1] = [2, {n - 1}], got {n_neighbors}")
    if dimensions < 1:
        raise ValueError(f"dimensions must be at least 1, got {dimensions}")
    if negative_sample_rate < 0:
        raise ValueError(f"negative_sample_rate must be at least 0, got {negative_sample_rate}")
    epochs = n_epochs if n_epochs is not None else (500 if n <= _LARGE else 200)
    if epochs < 1:
        raise ValueError(f"n_epochs must be at least 1, got {epochs}")
    curve_a, curve_b = fit_curve(min_dist)
    workers = resolve_threads(threads)

    idx, dist = _accel.umap_knn(a, n_neighbors, workers)
    rho, sigma, indptr, indices, weights = _accel.umap_fuzzy(idx, dist)
    graph = FuzzyGraph(indptr, indices, weights)
    start, n_components, n_random, converged = _start(graph, dimensions, seed, workers)
    embedding = _accel.umap_layout(
        start,
        graph.indptr,
        graph.indices,
        graph.weights,
        curve_a,
        curve_b,
        epochs,
        negative_sample_rate,
        seed,
        workers,
    )
    return UMAPFit(
        embedding=embedding,
        graph=graph,
        neighbors=idx,
        distances=dist,
        rho=rho,
        sigma=sigma,
        a=curve_a,
        b=curve_b,
        min_dist=min_dist,
        n_epochs=epochs,
        negative_sample_rate=negative_sample_rate,
        seed=seed,
        n_graph_components=n_components,
        n_random_start=n_random,
        spectral_converged=converged,
    )


def neighbor_preservation(
    x: npt.ArrayLike, embedding: npt.ArrayLike, k: int = 15, *, threads: int | None = None
) -> float:
    """Return the share of nearest neighbours that the map keeps.

    For each point it takes the ``k`` nearest points in ``x`` and the ``k`` nearest in
    ``embedding``, and the result is the mean share of the first set that is in the second.
    1 means the map keeps every neighbourhood, and a map of unrelated points scores about
    ``k / (n - 1)``. It checks the thing UMAP tries to keep. It says nothing about distances, the
    size of clusters or the gaps between them, which the map does not keep.

    Args:
        x: The original points, shape (n, d).
        embedding: Their positions in the map, shape (n, m).
        k: Neighbours per point, from 1 to n - 1.
        threads: Worker threads. The value does not depend on it.
    """
    original = as_points(x)
    mapped = as_points(embedding)
    n = original.shape[0]
    if mapped.shape[0] != n:
        raise ValueError("x and embedding must have the same number of rows")
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n - 1] = [1, {n - 1}], got {k}")
    workers = resolve_threads(threads)
    near_x, _ = _accel.umap_knn(original, k, workers)
    near_y, _ = _accel.umap_knn(mapped, k, workers)
    shared = (near_x[:, :, None] == near_y[:, None, :]).any(axis=2).sum(axis=1)
    return float(shared.mean()) / k
