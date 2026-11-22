"""UMAP: each step against the definition in McInnes, Healy & Melville (2018), written plainly."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx import _accel
from statsx.clustering import hdbscan
from statsx.embedding import FuzzyGraph, fit_curve, neighbor_preservation, umap

F64 = npt.NDArray[np.float64]


def _blobs(n_per: int = 60, dim: int = 6, seed: int = 0) -> F64:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, 0.5, size=(n_per, dim)) for c in (0.0, 2.0, 4.0)])


def _swiss_roll(n: int, seed: int) -> F64:
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
    h = 10 * rng.uniform(size=n)
    return np.column_stack([t * np.cos(t), h, t * np.sin(t)])


def _distance(a: F64, b: F64) -> float:
    return math.sqrt(float(np.sum((a - b) ** 2)))


def _pairwise(x: F64) -> F64:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def _entropy(y: F64, graph: FuzzyGraph, a: float, b: float) -> float:
    """Definition 10 over all pairs, with weight 0 for a pair that is not an edge."""
    rows, cols = np.triu_indices(y.shape[0], 1)
    mu = graph.dense()[rows, cols]
    nu = 1.0 / (1.0 + a * _pairwise(y)[rows, cols] ** (2.0 * b))
    edge = mu > 0.0
    partial = mu < 1.0
    total = float(np.sum(mu[edge] * np.log(mu[edge] / nu[edge])))
    total += float(np.sum((1.0 - mu[partial]) * np.log((1.0 - mu[partial]) / (1.0 - nu[partial]))))
    return total


def test_the_curve_fit_is_a_least_squares_minimum_and_matches_the_known_constants() -> None:
    a, b = fit_curve(0.1)
    # The constants that the reference implementation of the paper reports for min_dist 0.1:
    # about 1.58 and 0.90 (the uwot documentation lists them).
    assert a == pytest.approx(1.58, abs=0.01)
    assert b == pytest.approx(0.90, abs=0.01)
    r = np.linspace(0.0, 3.0, 300)[1:]
    target = np.where(r <= 0.1, 1.0, np.exp(-(r - 0.1)))

    def sse(a_: float, b_: float) -> float:
        return float(np.sum((1.0 / (1.0 + a_ * r ** (2.0 * b_)) - target) ** 2))

    best = sse(a, b)
    for da, db in [(1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3), (1e-3, 1e-3), (-1e-3, -1e-3)]:
        assert sse(a + da, b + db) >= best


def test_the_curve_is_one_for_zero_min_dist_only_at_the_origin_and_validates_its_range() -> None:
    a0 = fit_curve(0.0)[0]
    a1 = fit_curve(0.5)[0]
    assert a0 > a1  # a larger min_dist flattens the curve near 0
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="min_dist"):
            fit_curve(bad)


def test_the_neighbours_are_the_exact_nearest_points() -> None:
    x = _blobs(40, 5, seed=3)
    fit = umap(x, n_neighbors=10, n_epochs=2)
    d = _pairwise(x)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind="stable")[:, :10]
    assert np.array_equal(fit.neighbors, order)
    assert np.allclose(fit.distances, np.take_along_axis(d, order, axis=1), rtol=0, atol=1e-12)


def test_neighbour_distances_are_exact_when_many_are_tied() -> None:
    gx, gy = np.meshgrid(np.arange(8.0), np.arange(8.0))
    x = np.column_stack([gx.ravel(), gy.ravel()]) + 1e6
    fit = umap(x, n_neighbors=6, n_epochs=2)
    d = _pairwise(x)
    np.fill_diagonal(d, np.inf)
    assert np.array_equal(fit.distances, np.sort(d, axis=1)[:, :6])
    for i in range(x.shape[0]):
        assert np.array_equal(d[i, fit.neighbors[i]], fit.distances[i])


def test_rho_sigma_and_the_graph_follow_section_3_1() -> None:
    x = _blobs(50, 6, seed=1)
    k = 12
    fit = umap(x, n_neighbors=k, n_epochs=2)
    n = x.shape[0]
    a = np.zeros((n, n))
    for i in range(n):
        d = fit.distances[i]
        rho = d[d > 0].min()
        assert fit.rho[i] == rho
        # sigma solves sum_j exp(-max(0, d_ij - rho) / sigma) = log2(k).
        total = np.exp(-np.maximum(0.0, d - rho) / fit.sigma[i]).sum()
        assert total == pytest.approx(math.log2(k), abs=1e-9)
        a[i, fit.neighbors[i]] = np.exp(-np.maximum(0.0, d - rho) / fit.sigma[i])
        assert a[i].max() == pytest.approx(1.0)  # local connectivity: one edge of weight 1
    b = a + a.T - a * a.T
    assert np.allclose(fit.graph.dense(), b, rtol=0, atol=1e-14)
    assert fit.graph.n_edges == int(np.count_nonzero(np.triu(b, 1)))
    assert np.array_equal(fit.graph.dense(), fit.graph.dense().T)


def test_coinciding_points_do_not_break_the_graph() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 4))
    x = np.vstack([base, base[:10], base[:10]])  # ten points three times
    fit = umap(x, n_neighbors=8, n_epochs=20)
    assert np.all(np.isfinite(fit.embedding))
    assert np.all(fit.sigma > 0.0) and np.all(np.isfinite(fit.graph.weights))
    assert np.all((fit.graph.weights > 0.0) & (fit.graph.weights <= 1.0))


def test_the_spectral_vectors_equal_a_dense_eigendecomposition() -> None:
    x = np.random.default_rng(2).normal(size=(120, 5))
    fit = umap(x, n_neighbors=15, n_epochs=1)
    assert fit.n_graph_components == 1  # the trivial eigenvalue 1 is simple
    graph = fit.graph
    w = graph.dense()
    root = 1.0 / np.sqrt(w.sum(axis=1))
    values, vectors = np.linalg.eigh(root[:, None] * w * root[None, :])
    values, vectors = values[::-1], vectors[:, ::-1]
    dim = 2
    assert values[dim] - values[dim + 1] > 1e-3  # a gap, so the subspace is well defined
    found = _accel.umap_spectral(graph.indptr, graph.indices, graph.weights, dim, 2000, 1e-10, 0, 2)
    assert found is not None
    vecs, vals, converged, _ = found
    assert converged
    assert np.allclose(vals, values[1 : dim + 1], atol=1e-9)
    true = vectors[:, 1 : dim + 1]
    # The same plane: the projectors agree.
    assert np.allclose(vecs @ vecs.T, true @ true.T, atol=1e-6)


def test_the_cross_entropy_equals_definition_10() -> None:
    x = _blobs(25, 4, seed=4)
    fit = umap(x, n_neighbors=8, n_epochs=30)
    assert fit.cross_entropy() == pytest.approx(
        _entropy(fit.embedding, fit.graph, fit.a, fit.b), rel=1e-9
    )


def _sequential_layout(  # noqa: PLR0913, PLR0917 (the settings of the layout)
    y0: F64, graph: FuzzyGraph, a: float, b: float, epochs: int, n_neg: int, seed: int
) -> F64:
    """Algorithm 5 as written: one point at a time, every update seen by the next."""
    rng = np.random.default_rng(seed)
    y = y0.copy()
    n = y.shape[0]
    for e in range(epochs):
        alpha = 1.0 - e / epochs
        for i in range(n):
            for entry in range(graph.indptr[i], graph.indptr[i + 1]):
                if rng.uniform() >= graph.weights[entry]:
                    continue
                j = graph.indices[entry]
                diff = y[i] - y[j]
                r2 = float(diff @ diff)
                if r2 > 0:
                    coef = -2 * a * b * r2 ** (b - 1) / (1 + a * r2**b)
                    y[i] += alpha * np.clip(coef * diff, -4, 4)
                for _ in range(n_neg):
                    c = int(rng.integers(n))
                    if c == i:
                        continue
                    diff = y[i] - y[c]
                    r2 = float(diff @ diff)
                    coef = 2 * b / ((0.001 + r2) * (1 + a * r2**b))
                    y[i] += alpha * np.clip(coef * diff, -4, 4)
    return y


def test_the_layout_reaches_the_cross_entropy_of_the_sequential_algorithm() -> None:
    x = _blobs(25, 5, seed=5)
    fit = umap(x, n_neighbors=6, n_epochs=1)
    n = x.shape[0]
    rng = np.random.default_rng(9)
    y0 = rng.normal(0.0, 5.0, size=(n, 2))
    epochs = 100
    ours = _accel.umap_layout(
        y0, fit.graph.indptr, fit.graph.indices, fit.graph.weights, fit.a, fit.b, epochs, 5, 3, 2
    )
    theirs = _sequential_layout(y0, fit.graph, fit.a, fit.b, epochs, 5, 3)
    start = _entropy(y0, fit.graph, fit.a, fit.b)
    ce_ours = _entropy(ours, fit.graph, fit.a, fit.b)
    ce_theirs = _entropy(theirs, fit.graph, fit.a, fit.b)
    assert ce_ours < 0.5 * start and ce_theirs < 0.5 * start
    assert ce_ours == pytest.approx(ce_theirs, rel=0.15)


def test_the_layout_lowers_the_cross_entropy_and_keeps_the_neighbours_of_a_swiss_roll() -> None:
    x = _swiss_roll(600, seed=7)
    short = umap(x, n_epochs=1)
    full = umap(x)
    assert full.cross_entropy() < 0.8 * short.cross_entropy()
    centred = x - x.mean(axis=0)
    pca = centred @ np.linalg.svd(centred, full_matrices=False)[2][:2].T
    kept = neighbor_preservation(x, full.embedding, 15)
    assert kept > 0.6
    assert kept > neighbor_preservation(x, pca, 15) + 0.2


def test_three_separate_groups_stay_separate_in_the_map() -> None:
    rng = np.random.default_rng(6)
    x = np.vstack([rng.normal(10.0 * c, 0.5, size=(80, 8)) for c in range(3)])
    fit = umap(x)
    labels = np.repeat([0, 1, 2], 80)
    groups = [fit.embedding[labels == c] for c in range(3)]
    centres = [g.mean(axis=0) for g in groups]
    radii = [max(_distance(p, centres[c]) for p in groups[c]) for c in range(3)]
    gaps = [_distance(centres[i], centres[j]) for i in range(3) for j in range(i)]
    assert fit.n_graph_components == 3
    assert min(gaps) > 2 * max(radii)


def test_the_map_does_not_depend_on_the_number_of_threads() -> None:
    x = _blobs(50, 6, seed=8)
    one = umap(x, n_epochs=40, threads=1)
    many = umap(x, n_epochs=40, threads=3)
    assert np.array_equal(one.embedding, many.embedding)
    assert np.array_equal(one.graph.weights, many.graph.weights)
    other = umap(x, n_epochs=40, threads=3, seed=1)
    assert not np.array_equal(one.embedding, other.embedding)


def test_a_graph_in_pieces_gets_a_start_for_each_piece() -> None:
    rng = np.random.default_rng(0)
    big = rng.normal(0.0, 1.0, size=(60, 3))
    far = rng.normal(100.0, 0.01, size=(3, 3))  # three points: too few for two eigenvectors
    fit = umap(np.vstack([big, far]), n_neighbors=2, n_epochs=50)
    assert fit.n_graph_components >= 2
    assert fit.n_random_start >= 3
    assert np.all(np.isfinite(fit.embedding))


def test_pure_noise_gets_clusters_on_the_map_that_the_data_does_not_have() -> None:
    # Section 6 of the paper: UMAP finds structure in noise. HDBSCAN finds none in the data, and
    # finds groups in the map of the same data.
    for seed in range(4):
        x = np.random.default_rng(seed).uniform(size=(400, 10))
        assert hdbscan(x, min_cluster_size=15).n_clusters == 0
        assert hdbscan(umap(x).embedding, min_cluster_size=15).n_clusters >= 2


def test_neighbor_preservation_equals_its_definition() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 5))
    y = x[:, :2] + rng.normal(0.0, 0.3, size=(80, 2))
    k = 7
    dx, dy = _pairwise(x), _pairwise(y)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    shares = [len(set(np.argsort(dx[i])[:k]) & set(np.argsort(dy[i])[:k])) / k for i in range(80)]
    assert neighbor_preservation(x, y, k) == pytest.approx(float(np.mean(shares)))
    assert neighbor_preservation(x, x, k) == 1.0
    unrelated = neighbor_preservation(x, rng.normal(size=(80, 2)), k)
    assert unrelated < 0.2


def test_the_fit_exposes_its_inputs() -> None:
    x = _blobs(30, 4, seed=2)
    fit = umap(x, n_neighbors=9, dimensions=3, min_dist=0.3, n_epochs=10, seed=5)
    assert (fit.n_points, fit.dimensions, fit.n_neighbors) == (90, 3, 9)
    assert (fit.n_epochs, fit.min_dist, fit.seed, fit.negative_sample_rate) == (10, 0.3, 5, 5)
    assert fit.graph.n_vertices == 90
    assert fit.graph.rows.shape == fit.graph.indices.shape
    assert fit.embedding.shape == (90, 3)
    assert (fit.a, fit.b) == fit_curve(0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_neighbors": 1}, "n_neighbors"),
        ({"n_neighbors": 90}, "n_neighbors"),
        ({"dimensions": 0}, "dimensions"),
        ({"min_dist": 2.0}, "min_dist"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"negative_sample_rate": -1}, "negative_sample_rate"),
        ({"threads": 0}, "threads"),
    ],
)
def test_settings_out_of_range_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        umap(_blobs(30, 4), **kwargs)  # type: ignore[arg-type]


def test_bad_points_are_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        umap(np.array([[0.0, 1.0], [np.nan, 1.0], [1.0, 2.0], [3.0, 4.0]]), n_neighbors=2)
    with pytest.raises(ValueError, match="shape"):
        umap(np.arange(10.0))
    with pytest.raises(ValueError, match="same number"):
        neighbor_preservation(np.zeros((5, 2)), np.zeros((4, 2)), 2)
    with pytest.raises(ValueError, match="k must"):
        neighbor_preservation(np.zeros((5, 2)), np.zeros((5, 2)), 5)
