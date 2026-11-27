"""UMAP: each step against the definition in McInnes, Healy & Melville (2018), written plainly."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from statsx import _accel
from statsx.clustering import hdbscan
from statsx.embedding import FuzzyGraph, Metric, UMAPFit, fit_curve, neighbor_preservation, umap

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
        y0,
        fit.graph.indptr,
        fit.graph.indices,
        fit.graph.weights,
        fit.a,
        fit.b,
        epochs,
        5,
        1.0,
        3,
        2,
    )
    theirs = _sequential_layout(y0, fit.graph, fit.a, fit.b, epochs, 5, 3)
    start = _entropy(y0, fit.graph, fit.a, fit.b)
    ce_ours = _entropy(ours, fit.graph, fit.a, fit.b)
    ce_theirs = _entropy(theirs, fit.graph, fit.a, fit.b)
    assert ce_ours < 0.5 * start and ce_theirs < 0.5 * start
    # Both are stochastic, and the same code gave ratios of 0.84 to 1.10 on two machines.
    assert ce_ours == pytest.approx(ce_theirs, rel=0.3)


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


def _cosine_distances(x: F64) -> F64:
    """1 - cos for every pair, from the definition."""
    norms = np.sqrt((x * x).sum(axis=1))
    return 1.0 - (x @ x.T) / (norms[:, None] * norms[None, :])


def _scaled_points(n: int = 80, dim: int = 10, seed: int = 3) -> tuple[F64, F64]:
    """Random directions with lengths from 0.1 to 100, and the same points at length 1."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, dim))
    lengths = rng.uniform(0.1, 100.0, size=(n, 1))
    return direction * lengths, direction


def test_the_cosine_neighbours_and_distances_equal_the_exact_ones() -> None:
    x, _ = _scaled_points()
    fit = umap(x, n_neighbors=10, metric="cosine", n_epochs=1)
    full = _cosine_distances(x)
    np.fill_diagonal(full, np.inf)
    order = np.argsort(full, axis=1)[:, :10]
    assert fit.metric == "cosine"
    assert np.array_equal(np.sort(fit.neighbors, axis=1), np.sort(order, axis=1))
    expected = np.take_along_axis(full, order, axis=1)
    assert np.allclose(fit.distances, expected, rtol=0.0, atol=1e-12)


def test_the_cosine_map_ignores_the_length_of_a_row() -> None:
    x, direction = _scaled_points()
    long = umap(x, n_neighbors=10, metric="cosine", n_epochs=1)
    unit = umap(direction, n_neighbors=10, metric="cosine", n_epochs=1)
    assert np.array_equal(long.neighbors, unit.neighbors)
    assert np.allclose(long.distances, unit.distances, rtol=0.0, atol=1e-12)
    assert np.allclose(long.rho, unit.rho, rtol=0.0, atol=1e-12)


def test_a_scaled_copy_of_a_row_is_at_cosine_distance_zero() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 8))
    x = np.vstack([x, 3.0 * x[0]])
    fit = umap(x, n_neighbors=8, metric="cosine", n_epochs=1)
    assert fit.neighbors[0, 0] == 40
    assert fit.distances[0, 0] == 0.0
    full = _cosine_distances(x)
    others = np.sort(full[0, 1:40])
    # rho is the smallest distance above zero, not the zero of the copy (section 3.1)
    assert fit.rho[0] == pytest.approx(others[0], abs=1e-12)


def test_the_cosine_distance_keeps_groups_by_direction_when_lengths_differ_a_lot() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=10)
    a /= np.sqrt(a @ a)
    b = rng.normal(size=10)
    b -= (b @ a) * a
    b /= np.sqrt(b @ b)
    angle = 0.4  # radians between the two directions
    axes = np.stack([a, np.cos(angle) * a + np.sin(angle) * b])
    group = np.repeat([0, 1], 100)
    direction = axes[group] + rng.normal(0.0, 0.05, size=(200, 10))
    x = direction * np.exp(rng.uniform(0.0, np.log(100.0), size=(200, 1)))

    def purity(metric: Metric) -> float:
        fit = umap(x, n_neighbors=10, metric=metric, n_epochs=1)
        return float((group[fit.neighbors] == group[:, None]).mean())

    # a point close to the origin is near the other group in length-driven distance
    assert purity("cosine") > 0.995
    assert purity("euclidean") < 0.985


def test_neighbor_preservation_with_the_cosine_distance_equals_its_definition() -> None:
    x, _ = _scaled_points(n=60)
    rng = np.random.default_rng(1)
    y = rng.normal(size=(60, 2))
    full_x = _cosine_distances(x)
    full_y = _pairwise(y)
    np.fill_diagonal(full_x, np.inf)
    np.fill_diagonal(full_y, np.inf)
    k = 7
    shares = [
        len(set(np.argsort(full_x[i])[:k]) & set(np.argsort(full_y[i])[:k])) / k for i in range(60)
    ]
    assert neighbor_preservation(x, y, k, metric="cosine") == pytest.approx(float(np.mean(shares)))


def test_the_cosine_distance_refuses_a_row_of_zeros_and_an_unknown_metric_is_refused() -> None:
    x, _ = _scaled_points(n=20)
    x[4] = 0.0
    with pytest.raises(ValueError, match="row of zeros"):
        umap(x, n_neighbors=5, metric="cosine")
    # the Euclidean distance has no such problem
    assert umap(x, n_neighbors=5, n_epochs=1).n_points == 20
    with pytest.raises(ValueError, match="metric must be one of"):
        umap(x, n_neighbors=5, metric="manhattan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric must be one of"):
        neighbor_preservation(x, x[:, :2], 5, metric="manhattan")  # type: ignore[arg-type]


def _split_roll() -> tuple[F64, F64]:
    roll = _swiss_roll(1250, 0)
    return roll[:1000], roll[1000:]


def test_a_new_point_starts_at_the_weighted_mean_of_its_neighbours() -> None:
    train, test = _split_roll()
    fit = umap(train, n_neighbors=12, n_epochs=2)
    placed = fit.transform(test[:60], n_epochs=0)
    full = np.sqrt(((test[:60, None, :] - train[None, :, :]) ** 2).sum(axis=2))
    assert np.array_equal(placed.neighbors, np.argsort(full, axis=1, kind="stable")[:, :12])
    for i in range(60):
        d = placed.distances[i]
        # rho is 0 for a new point (local connectivity 0), so the weight is exp(-d / sigma)
        w = np.exp(-d / placed.sigma[i])
        assert w.sum() == pytest.approx(math.log2(12), abs=1e-9)  # the equation of section 3.1
        mean = (w[:, None] * fit.embedding[placed.neighbors[i]]).sum(axis=0) / w.sum()
        assert np.allclose(placed.embedding[i], mean, rtol=0.0, atol=1e-12)
    assert placed.n_points == 60
    assert placed.n_epochs == 0


def _shared_with_map(train_map: F64, placed_map: F64, neighbors: npt.NDArray[np.int64]) -> float:
    near, _ = _accel.umap_knn_between(train_map, placed_map, neighbors.shape[1], 2)
    return float((neighbors[:, :, None] == near[:, None, :]).any(axis=2).mean())


def test_new_points_keep_their_neighbours_about_as_well_as_the_training_points() -> None:
    train, test = _split_roll()
    fit = umap(train)
    placed = fit.transform(test)
    k = fit.n_neighbors
    # the share of each new point's k nearest training points that are among its k nearest in the map
    shared = _shared_with_map(fit.embedding, placed.embedding, placed.neighbors[:, :k])
    own = neighbor_preservation(train, fit.embedding)
    assert shared > 0.6
    assert shared > own - 0.15


def test_a_training_point_placed_again_lands_near_its_own_position() -> None:
    train, _ = _split_roll()
    fit = umap(train)
    again = fit.transform(train[:200]).embedding
    moved = np.sqrt(((again - fit.embedding[:200]) ** 2).sum(axis=1))
    between = _pairwise(fit.embedding[:200]) + np.diag(np.full(200, np.inf))
    assert np.median(moved) < 0.6 * np.median(between.min(axis=1))


def test_the_placement_does_not_depend_on_threads_and_follows_the_seed() -> None:
    train, test = _split_roll()
    fit = umap(train, n_epochs=50)
    one = fit.transform(test, threads=1)
    three = fit.transform(test, threads=3)
    assert np.array_equal(one.embedding, three.embedding)
    assert np.array_equal(one.embedding, fit.transform(test).embedding)
    assert not np.array_equal(one.embedding, fit.transform(test, seed=99).embedding)
    # the first row of a call has the same stream whatever follows it
    assert np.array_equal(fit.transform(test[:1]).embedding[0], one.embedding[0])
    assert one.seed == fit.seed


def test_the_cosine_placement_finds_the_exact_cosine_neighbours() -> None:
    x, _ = _scaled_points(n=100)
    new = _scaled_points(n=15, seed=8)[0]
    fit = umap(x, n_neighbors=8, metric="cosine", n_epochs=5)
    placed = fit.transform(new, n_epochs=3)
    cross = 1.0 - (new @ x.T) / (
        np.sqrt((new * new).sum(axis=1))[:, None] * np.sqrt((x * x).sum(axis=1))[None, :]
    )
    order = np.argsort(cross, axis=1)[:, :8]
    assert np.array_equal(np.sort(placed.neighbors, axis=1), np.sort(order, axis=1))
    assert np.allclose(placed.distances, np.take_along_axis(cross, order, axis=1), atol=1e-12)
    assert np.array_equal(placed.neighbors, fit.transform(2.5 * new, n_epochs=3).neighbors)


def test_transform_refuses_bad_input_and_the_fit_keeps_its_own_copy_of_the_data() -> None:
    x = _blobs(40, 5)
    fit = umap(x, n_neighbors=6, n_epochs=2)
    with pytest.raises(ValueError, match="columns"):
        fit.transform(x[:, :4])
    with pytest.raises(ValueError, match="n_epochs"):
        fit.transform(x[:3], n_epochs=-1)
    with pytest.raises(ValueError, match="finite"):
        fit.transform(np.full((2, 5), np.nan))
    cosine = umap(x + 5.0, n_neighbors=6, metric="cosine", n_epochs=2)
    with pytest.raises(ValueError, match="row of zeros"):
        cosine.transform(np.zeros((1, 5)))
    assert fit.transform(x[:1]).embedding.shape == (1, 2)
    # a change of the caller's array must not change the fit
    before = fit.transform(x[:5], n_epochs=0).embedding.copy()
    x[:] = 0.0
    assert np.array_equal(fit.transform(_blobs(40, 5)[:5], n_epochs=0).embedding, before)
    assert not fit.data.flags.writeable


def _same_fit(a: UMAPFit, b: UMAPFit) -> bool:
    arrays = ("embedding", "data", "neighbors", "distances", "rho", "sigma")
    graph = ("indptr", "indices", "weights")
    plain = (
        "metric",
        "a",
        "b",
        "min_dist",
        "n_epochs",
        "negative_sample_rate",
        "seed",
        "n_graph_components",
        "n_random_start",
        "spectral_converged",
        "learning_rate",
        "start",
        "neighbor_source",
        "supervision",
    )
    return (
        all(np.array_equal(getattr(a, f), getattr(b, f)) for f in arrays)
        and all(np.array_equal(getattr(a.graph, f), getattr(b.graph, f)) for f in graph)
        and all(getattr(a, f) == getattr(b, f) for f in plain)
    )


def test_a_saved_fit_loads_back_exactly_and_places_new_points_the_same_way(
    tmp_path: Path,
) -> None:
    train, test = _split_roll()
    fit = umap(train, n_epochs=40, seed=2**63 + 5, metric="cosine")
    path = tmp_path / "model.bin"  # the name is used as given: no .npz is added
    fit.to_npz(path)
    assert path.exists()
    loaded = UMAPFit.from_npz(path)
    assert _same_fit(fit, loaded)
    assert not loaded.data.flags.writeable
    assert np.array_equal(
        fit.transform(test[:30], n_epochs=5).embedding,
        loaded.transform(test[:30], n_epochs=5).embedding,
    )
    assert loaded.cross_entropy() == fit.cross_entropy()
    buffer = io.BytesIO()
    fit.to_npz(buffer)
    buffer.seek(0)
    assert _same_fit(fit, UMAPFit.from_npz(buffer))


def _saved(fit: UMAPFit, **changes: object) -> io.BytesIO:
    """A file of ``fit`` with some arrays replaced, as a damaged or edited file would be."""
    plain = io.BytesIO()
    fit.to_npz(plain)
    plain.seek(0)
    with np.load(plain) as store:
        arrays: dict[str, object] = {name: store[name] for name in store.files}
    arrays.update(changes)
    out = io.BytesIO()
    np.savez(out, **arrays)  # type: ignore[arg-type]
    out.seek(0)
    return out


def test_a_file_that_is_not_a_fit_of_this_version_is_refused() -> None:
    x = _blobs(30, 4)
    fit = umap(x, n_neighbors=5, n_epochs=2)
    with pytest.raises(ValueError, match="format version 99"):
        UMAPFit.from_npz(_saved(fit, format_version=np.int64(99)))
    with pytest.raises(ValueError, match="lacks"):
        UMAPFit.from_npz(_without(fit, "sigma"))
    with pytest.raises(ValueError, match="not a UMAPFit file"):
        UMAPFit.from_npz(io.BytesIO(b"not a file"))
    # a file with a pickled object array is refused, so loading never runs code
    pickled = io.BytesIO()
    np.savez(pickled, format_version=np.array([object()], dtype=object))
    pickled.seek(0)
    with pytest.raises(ValueError, match="not a UMAPFit file"):
        UMAPFit.from_npz(pickled)


def _without(fit: UMAPFit, name: str) -> io.BytesIO:
    plain = io.BytesIO()
    fit.to_npz(plain)
    plain.seek(0)
    with np.load(plain) as store:
        arrays = {k: store[k] for k in store.files if k != name}
    out = io.BytesIO()
    np.savez(out, **arrays)  # type: ignore[arg-type]
    out.seek(0)
    return out


def test_a_file_with_values_a_fit_cannot_have_is_refused() -> None:
    x = _blobs(30, 4)
    fit = umap(x, n_neighbors=5, n_epochs=2)
    n = fit.n_points
    bad_neighbors = fit.neighbors.copy()
    bad_neighbors[0, 0] = n  # one past the last row
    with pytest.raises(ValueError, match="cannot have"):
        UMAPFit.from_npz(_saved(fit, neighbors=bad_neighbors))
    bad_indices = fit.graph.indices.copy()
    bad_indices[0] = -1
    with pytest.raises(ValueError, match="cannot have"):
        UMAPFit.from_npz(_saved(fit, graph_indices=bad_indices))
    nan_points = fit.data.copy()
    nan_points[0, 0] = np.nan
    with pytest.raises(ValueError, match="cannot have"):
        UMAPFit.from_npz(_saved(fit, data=nan_points))
    with pytest.raises(ValueError, match="do not fit together"):
        UMAPFit.from_npz(_saved(fit, embedding=fit.embedding[:-1]))
    with pytest.raises(ValueError, match="metric must be one of"):
        UMAPFit.from_npz(_saved(fit, metric=np.array("manhattan")))


def _groups(n: int, seed: int) -> tuple[F64, F64, F64]:
    """Return ``n`` points in 8 Gaussian groups of 64 columns, ``n // 6`` more, and the centres."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 4.0, (8, 64))

    def draw(m: int) -> F64:
        return centres[rng.integers(0, 8, m)] + rng.normal(0.0, 1.0, (m, 64))

    return draw(n), draw(n // 6), centres


def _movement(a: F64, b: F64) -> float:
    return float(np.median(np.linalg.norm(a - b, axis=1)))


def test_zero_epochs_return_the_given_start_in_a_new_array() -> None:
    x = _blobs(30, 4)
    given = np.random.default_rng(3).normal(0.0, 5.0, (90, 2))
    fit = umap(x, init=given, n_epochs=0)
    assert np.array_equal(fit.embedding, given)
    assert not np.shares_memory(fit.embedding, given)
    assert (fit.start, fit.n_epochs) == ("init", 0)
    assert fit.cross_entropy() > umap(x, init=given).cross_entropy()


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_layout_never_writes_the_callers_start(dtype: type[np.floating]) -> None:
    x = _blobs(30, 4)
    given = np.ascontiguousarray(umap(x, n_epochs=50).embedding, dtype=dtype)
    before = given.copy()
    given.flags.writeable = False
    fit = umap(x, init=given, learning_rate=1.0)
    assert np.array_equal(given, before)
    assert not np.shares_memory(fit.embedding, given)
    assert _movement(fit.embedding, np.asarray(before, dtype=np.float64)) > 0.1


def test_a_turned_start_gives_a_turned_map() -> None:
    x = _groups(300, seed=1)[0]
    old = umap(x, seed=42).embedding
    turned = (old - old.mean(axis=0)) @ np.array([[0.0, 1.0], [-1.0, 0.0]]) + old.mean(axis=0)
    again = umap(x, init=turned, seed=5).embedding
    assert _movement(again, turned) < 0.1 * _movement(again, old)


def test_the_step_size_defaults_to_one_cold_and_a_tenth_with_a_start() -> None:
    x = _groups(300, seed=2)[0]
    cold = umap(x, seed=42)
    warm = umap(x, init=cold.embedding)
    assert (cold.learning_rate, cold.start) == (1.0, "spectral")
    assert (warm.learning_rate, warm.start) == (0.1, "init")
    assert np.array_equal(warm.embedding, umap(x, init=cold.embedding, learning_rate=0.1).embedding)
    small = umap(x, init=cold.embedding, learning_rate=0.01).embedding
    full = umap(x, init=cold.embedding, learning_rate=1.0).embedding
    assert _movement(small, cold.embedding) < 0.5 * _movement(full, cold.embedding)


def test_a_moving_window_placed_in_the_old_map_is_about_as_good_as_a_cold_map() -> None:
    x, fresh, _ = _groups(600, seed=0)
    old = umap(x, seed=42)
    window = np.vstack([x[100:], fresh])
    cold = umap(window, seed=42)
    warm = umap(window, init=old.transform(window).embedding, seed=42)
    kept_cold = neighbor_preservation(window, cold.embedding)
    assert neighbor_preservation(window, warm.embedding) >= kept_cold - 0.03
    assert warm.cross_entropy() <= 1.05 * cold.cross_entropy()  # the same graph for both
    same = old.embedding[100:]
    assert _movement(warm.embedding[:500], same) < 0.3 * _movement(cold.embedding[:500], same)


def test_a_warm_map_does_not_depend_on_threads_and_counts_the_pieces_of_its_graph() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 1.0, (60, 3)), rng.normal(100.0, 0.01, (3, 3))])
    cold = umap(x, n_neighbors=2, n_epochs=20)
    one = umap(x, n_neighbors=2, n_epochs=20, init=cold.embedding, threads=1)
    many = umap(x, n_neighbors=2, n_epochs=20, init=cold.embedding, threads=3)
    assert np.array_equal(one.embedding, many.embedding)
    assert one.n_graph_components == cold.n_graph_components >= 2
    assert (one.n_random_start, one.spectral_converged) == (0, True)


def test_rows_that_start_on_the_same_position_come_apart() -> None:
    x = _blobs(30, 4)
    given = np.random.default_rng(1).normal(0.0, 5.0, (90, 2))
    given[:10] = given[0]
    fit = umap(x, init=given, n_epochs=50)
    assert np.unique(fit.embedding[:10], axis=0).shape[0] == 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"init": np.zeros((89, 2))}, "init must have shape"),
        ({"init": np.eye(90, 3)}, "init must have shape"),
        ({"init": np.full((90, 2), np.nan)}, "init must be finite"),
        ({"init": np.eye(90, 2) * 1e200}, "init must be finite and at most"),
        ({"init": np.zeros((90, 2))}, "fewer than 2 dimensions"),
        ({"init": np.outer(np.arange(90.0), [1.0, 2.0])}, "fewer than 2 dimensions"),
        ({"init": np.eye(90, 2), "n_epochs": -1}, "n_epochs must be at least 0"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": 1.5}, "learning_rate"),
        ({"learning_rate": math.nan}, "learning_rate"),
    ],
)
def test_a_bad_start_or_step_size_is_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        umap(_blobs(30, 4), **kwargs)  # type: ignore[arg-type]


def test_a_warm_fit_survives_a_save_and_a_load() -> None:
    x = _blobs(30, 4)
    start = umap(x, n_epochs=20).embedding
    for epochs in (0, 10):
        fit = umap(x, init=start, n_epochs=epochs, learning_rate=0.3)
        buffer = io.BytesIO()
        fit.to_npz(buffer)
        buffer.seek(0)
        assert _same_fit(fit, UMAPFit.from_npz(buffer))


def test_a_file_of_format_version_1_loads_as_a_cold_fit() -> None:
    # written by statsx 0.1.79: two groups of 20 points, n_neighbors 5, 20 epochs, seed 7
    old = UMAPFit.from_npz(Path(__file__).parent / "data" / "umap_fit_v1.npz")
    assert (old.learning_rate, old.start) == (1.0, "spectral")
    assert (old.n_points, old.n_neighbors, old.n_epochs, old.seed) == (40, 5, 20, 7)
    # the layout of 0.1.79 and of this version are the same, up to the last digits of a processor
    now = umap(old.data, n_neighbors=5, n_epochs=20, seed=7)
    assert now.graph.weights == pytest.approx(old.graph.weights, rel=1e-9)


def test_a_warm_file_with_values_a_fit_cannot_have_is_refused() -> None:
    x = _blobs(30, 4)
    fit = umap(x, n_neighbors=5, n_epochs=2)
    warm = umap(x, n_neighbors=5, n_epochs=0, init=fit.embedding)
    for damaged in (
        _saved(warm, counts=np.array([0, 5, 1, 1, 1])),
        _saved(warm, start=np.array("random")),
        _saved(fit, counts=np.array([0, 5, 1, 0, 1])),
        _saved(fit, reals=np.array([fit.a, fit.b, fit.min_dist, 2.0])),
    ):
        with pytest.raises(ValueError, match="cannot have"):
            UMAPFit.from_npz(damaged)
    with pytest.raises(ValueError, match="do not fit together"):
        UMAPFit.from_npz(_saved(fit, reals=np.array([fit.a, fit.b, fit.min_dist])))
    with pytest.raises(ValueError, match="lacks"):
        UMAPFit.from_npz(_without(fit, "start"))
