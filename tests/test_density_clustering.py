"""CL-3: DBSCAN and HDBSCAN, checked against the definitions in the papers, written out naively."""

import itertools
from collections import deque
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import dbscan, hdbscan, k_distances, kmeans

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

LOWEST = 1e-100  # the floor on a distance in lambda = 1 / distance


def _distances(x: Floats) -> Floats:
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)


def _blobs(seed: int, per: int = 80) -> tuple[Floats, Ints]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]])
    x = np.vstack([c + 0.5 * rng.standard_normal((per, 2)) for c in centres])
    return x, np.repeat(np.arange(3), per).astype(np.int64)


def _partition(labels: Ints) -> frozenset[frozenset[int]]:
    """The groups of a labelling, noise left out."""
    groups: dict[int, set[int]] = {}
    for i, label in enumerate(labels.tolist()):
        if label >= 0:
            groups.setdefault(label, set()).add(i)
    return frozenset(frozenset(g) for g in groups.values())


# --- DBSCAN, as Ester et al. (1996) write the algorithm -------------------------------------


def naive_dbscan(  # noqa: C901 (the paper's loop)
    x: Floats, eps: float, min_pts: int, distances: Callable[[Floats], Floats] = _distances
) -> Ints:
    """The sequential algorithm of the paper: scan the points, expand a cluster from each core."""
    n = len(x)
    dist = distances(x)
    near = [np.flatnonzero(dist[i] <= eps).tolist() for i in range(n)]  # holds the point itself
    unclassified, noise = -2, -1
    label = [unclassified] * n
    cluster = 0
    for i in range(n):
        if label[i] != unclassified:
            continue
        if len(near[i]) < min_pts:
            label[i] = noise
            continue
        seeds: deque[int] = deque()
        for j in near[i]:
            if label[j] == unclassified:
                seeds.append(j)
            if label[j] in (unclassified, noise):
                label[j] = cluster
        while seeds:
            p = seeds.popleft()
            if len(near[p]) < min_pts:
                continue
            for q in near[p]:
                if label[q] == unclassified:
                    seeds.append(q)
                if label[q] in (unclassified, noise):
                    label[q] = cluster
        cluster += 1
    return np.array(label, dtype=np.int64)


@pytest.mark.parametrize("seed", range(4))
def test_dbscan_equals_the_algorithm_of_the_paper(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.vstack([_blobs(seed, 60)[0], rng.uniform(-3, 10, (40, 2))])
    for eps, min_pts in [(0.5, 4), (0.8, 5), (1.2, 8)]:
        fit = dbscan(x, eps, min_samples=min_pts)
        assert np.array_equal(fit.labels, naive_dbscan(x, eps, min_pts))


def test_dbscan_counts_a_neighbour_exactly_at_eps_on_a_grid() -> None:
    grid = np.array(list(itertools.product(range(12), range(12))), dtype=np.float64)
    grid = np.vstack([grid, [[30.0, 30.0]]])
    for eps, min_pts in [(1.0, 4), (1.0, 5), (np.sqrt(2.0), 9), (2.0, 13)]:
        fit = dbscan(grid, float(eps), min_samples=min_pts)
        assert np.array_equal(fit.labels, naive_dbscan(grid, float(eps), min_pts))


def test_dbscan_is_exact_for_points_far_from_the_origin() -> None:
    rng = np.random.default_rng(3)
    x = 1e6 + np.vstack([rng.standard_normal((60, 3)), rng.standard_normal((60, 3)) + 8.0])
    assert np.array_equal(dbscan(x, 1.5).labels, naive_dbscan(x, 1.5, 5))


def test_dbscan_in_many_dimensions_equals_the_naive_one() -> None:
    rng = np.random.default_rng(4)
    x = np.vstack([rng.standard_normal((70, 30)), rng.standard_normal((70, 30)) + 2.0])
    assert np.array_equal(dbscan(x, 8.5, min_samples=6).labels, naive_dbscan(x, 8.5, 6))


def test_dbscan_reports_core_border_and_noise() -> None:
    x, _ = _blobs(1)
    x = np.vstack([x, [[40.0, 40.0]]])
    fit = dbscan(x, 0.7)
    assert fit.n_clusters == 3
    assert fit.labels[-1] == -1
    assert not fit.core_mask[-1]
    assert fit.n_border == int(np.count_nonzero((fit.labels >= 0) & ~fit.core_mask))
    assert fit.sizes.sum() == len(x) - fit.n_noise
    counts = (_distances(x) <= 0.7).sum(axis=1)
    assert np.array_equal(fit.core_mask, counts >= 5)


def test_dbscan_follows_the_shape_of_two_moons_where_kmeans_cannot() -> None:
    rng = np.random.default_rng(5)
    t = rng.uniform(0.0, np.pi, 300)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1.0 - np.cos(t), 0.4 - np.sin(t)])
    x = np.vstack([upper, lower]) + 0.05 * rng.standard_normal((600, 2))
    truth = np.repeat([0, 1], 300)
    fit = dbscan(x, 0.2, min_samples=5)
    assert fit.n_clusters == 2
    hits = max(float((fit.labels == truth).mean()), float((fit.labels == 1 - truth).mean()))
    assert hits > 0.97
    km = kmeans(x, 2, seed=1).labels
    assert max(float((km == truth).mean()), float((km == 1 - truth).mean())) < 0.9


def test_k_distances_are_the_distance_to_the_kth_other_point() -> None:
    x, _ = _blobs(2)
    expected = np.sort(_distances(x), axis=1)[:, 4]
    assert np.allclose(k_distances(x, 4), expected, rtol=1e-12)


def test_a_point_is_core_exactly_when_its_k_distance_is_at_most_eps() -> None:
    x, _ = _blobs(3)
    eps = 0.6
    fit = dbscan(x, eps, min_samples=5)
    assert np.array_equal(fit.core_mask, k_distances(x, 4) <= eps)


@pytest.mark.parametrize("function", ["dbscan", "hdbscan", "k_distances"])
def test_density_results_do_not_depend_on_the_number_of_threads(function: str) -> None:
    rng = np.random.default_rng(6)
    x = np.vstack([rng.standard_normal((2500, 4)), rng.standard_normal((2500, 4)) + 5.0])
    if function == "dbscan":
        one, four = dbscan(x, 0.9, threads=1), dbscan(x, 0.9, threads=4)
        assert np.array_equal(one.labels, four.labels)
    elif function == "hdbscan":
        a, b = hdbscan(x, 20, threads=1), hdbscan(x, 20, threads=4)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.mst, b.mst)
        assert np.array_equal(a.tree.stability, b.tree.stability)
    else:
        assert np.array_equal(k_distances(x, 5, threads=1), k_distances(x, 5, threads=4))


def test_dbscan_checks_its_arguments() -> None:
    x, _ = _blobs(0)
    with pytest.raises(ValueError, match="eps"):
        dbscan(x, 0.0)
    with pytest.raises(ValueError, match="eps"):
        dbscan(x, float("inf"))
    with pytest.raises(ValueError, match="min_samples"):
        dbscan(x, 1.0, min_samples=0)
    with pytest.raises(ValueError, match="finite"):
        dbscan(np.array([[0.0, np.nan], [1.0, 1.0]]), 1.0)
    with pytest.raises(ValueError, match="k must"):
        k_distances(x, 0)
    with pytest.raises(ValueError, match="k must"):
        k_distances(x, len(x))


# --- HDBSCAN*, as Campello et al. (2013) and McInnes and Healy (2017) define it --------------


def _mutual_reachability(
    x: Floats, k: int, distances: Callable[[Floats], Floats] = _distances
) -> tuple[Floats, Floats]:
    dist = distances(x)
    core = np.sort(dist, axis=1)[:, k - 1]  # the point itself is the first
    reach = np.maximum(np.maximum(core[:, None], core[None, :]), dist)
    np.fill_diagonal(reach, 0.0)
    return core, reach


def _prim(weights: Floats) -> list[tuple[int, int, float]]:
    n = len(weights)
    best: Floats = weights[0].copy()
    source: Ints = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    done[0] = True
    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        masked: Floats = np.where(done, np.inf, best)
        v = int(np.argmin(masked))
        edges.append((int(source[v]), v, float(masked[v])))
        done[v] = True
        row: Floats = weights[v]
        closer: npt.NDArray[np.bool_] = row < best
        best = np.where(closer, row, best)
        source = np.where(closer, v, source)
    return edges


class _Cluster:
    def __init__(self, birth: float) -> None:
        self.birth = birth
        self.points: list[int] = []
        self.stability = 0.0
        self.children: list[_Cluster] = []


def _grow(
    members: list[int], edges: list[tuple[int, int, float]], birth: float, m: int
) -> _Cluster:
    """Take edges off the heaviest first, as the hierarchy is read from the top (section 2.2)."""
    cluster = _Cluster(birth)
    while True:
        heaviest = max(w for _, _, w in edges)
        lam = 1.0 / max(heaviest, LOWEST)
        kept = [e for e in edges if e[2] < heaviest]
        parent = {p: p for p in members}

        def root(p: int, parent: dict[int, int] = parent) -> int:
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        for a, b, _ in kept:
            parent[root(a)] = root(b)
        groups: dict[int, list[int]] = {}
        for p in members:
            groups.setdefault(root(p), []).append(p)
        big = [g for g in groups.values() if len(g) >= m]
        small = [p for g in groups.values() if len(g) < m for p in g]
        cluster.points += small
        cluster.stability += (lam - birth) * len(small)
        if len(big) == 0:
            return cluster
        if len(big) == 1:
            members = big[0]
            inside = set(members)
            edges = [e for e in kept if e[0] in inside and e[1] in inside]
            continue
        for g in big:
            inside = set(g)
            sub = [e for e in kept if e[0] in inside and e[1] in inside]
            cluster.children.append(_grow(g, sub, lam, m))
            cluster.stability += (lam - birth) * len(g)
        return cluster


def _everything(cluster: _Cluster) -> list[int]:
    out = list(cluster.points)
    for child in cluster.children:
        out += _everything(child)
    return out


def _select(cluster: _Cluster, *, root: bool, allow_single: bool) -> tuple[float, list[_Cluster]]:
    below = [_select(c, root=False, allow_single=allow_single) for c in cluster.children]
    total = sum(v for v, _ in below)
    chosen = [c for _, group in below for c in group]
    if (not root or allow_single) and cluster.stability >= total:
        return cluster.stability, [cluster]
    return total, chosen


def naive_hdbscan(
    x: Floats,
    m: int,
    k: int,
    *,
    allow_single: bool = False,
    distances: Callable[[Floats], Floats] = _distances,
) -> tuple[frozenset[frozenset[int]], list[float], float]:
    """Return the chosen groups, their stabilities, and the total weight of the tree."""
    _, reach = _mutual_reachability(x, k, distances)
    edges = _prim(reach)
    top = _grow(list(range(len(x))), edges, 0.0, m)
    _, chosen = _select(top, root=True, allow_single=allow_single)
    groups = frozenset(frozenset(_everything(c)) for c in chosen)
    return groups, sorted(c.stability for c in chosen), sum(w for _, _, w in edges)


def _three_densities(seed: int) -> Floats:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.standard_normal((90, 2)) * 0.25,
            rng.standard_normal((90, 2)) * 0.25 + [2.2, 0.0],
            rng.standard_normal((90, 2)) * 1.6 + [10.0, 8.0],
            rng.uniform(-4, 14, (25, 2)),
        ]
    )


@pytest.mark.parametrize("seed", range(4))
def test_core_distances_and_the_tree_weight_equal_the_definitions(seed: int) -> None:
    x = _three_densities(seed)
    fit = hdbscan(x, 10, min_samples=7)
    core, reach = _mutual_reachability(x, 7)
    assert np.allclose(fit.core_distances, core, rtol=1e-12)
    assert fit.mst.shape == (len(x) - 1, 3)
    assert fit.mst[:, 2].sum() == pytest.approx(sum(w for _, _, w in _prim(reach)), rel=1e-12)
    for a, b, w in fit.mst:
        assert w == pytest.approx(reach[int(a), int(b)], rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_hdbscan_equals_the_hierarchy_read_from_the_definitions(seed: int) -> None:
    x = _three_densities(seed)
    for m, k in [(10, 10), (15, 5), (8, 8)]:
        fit = hdbscan(x, m, min_samples=k)
        groups, stability, _ = naive_hdbscan(x, m, k)
        assert _partition(fit.labels) == groups
        assert np.allclose(np.sort(fit.stabilities), stability, rtol=1e-9)


def _merging_densities(seed: int) -> Floats:
    """Two tight groups 1.5 apart and one spread-out group, plus scattered points."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.standard_normal((90, 2)) * 0.2,
            rng.standard_normal((90, 2)) * 0.2 + [1.5, 0.0],
            rng.standard_normal((90, 2)) * 1.6 + [10.0, 8.0],
            rng.uniform(-4, 14, (25, 2)),
        ]
    )


def _groups_found(labels: Ints) -> int:
    """How many of the three planted groups sit in a cluster that is mostly that group."""
    truth = np.repeat([0, 1, 2, 3], [90, 90, 90, 25])
    found = 0
    for g in range(3):
        mine = labels[truth == g]
        mine = mine[mine >= 0]
        if mine.size == 0:
            continue
        top = int(np.bincount(mine).argmax())
        if np.count_nonzero(mine == top) > 80 and np.count_nonzero(labels == top) < 110:
            found += 1
    return found


@pytest.mark.parametrize("seed", range(3))
def test_hdbscan_finds_groups_of_different_density_that_no_single_eps_finds(seed: int) -> None:
    x = _merging_densities(seed)
    assert _groups_found(hdbscan(x, 15).labels) == 3
    best = max(
        _groups_found(dbscan(x, float(eps), min_samples=15).labels)
        for eps in np.linspace(0.1, 3.0, 59)
    )
    # A radius small enough to split the tight groups leaves the spread-out one as noise, and a
    # radius large enough to hold it joins the tight ones.
    assert best == 2


def test_one_dense_group_is_noise_unless_the_root_is_allowed() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 2))
    assert hdbscan(x, 10).n_clusters == 0
    assert hdbscan(x, 10).n_noise == 200
    one = hdbscan(x, 10, allow_single_cluster=True)
    assert one.n_clusters == 1
    groups, _, _ = naive_hdbscan(x, 10, 10, allow_single=True)
    assert _partition(one.labels) == groups


def test_the_condensed_tree_is_consistent() -> None:
    x = _three_densities(1)
    fit = hdbscan(x, 12)
    tree = fit.tree
    n = len(x)
    assert tree.n_points == n
    points = tree.child[tree.child < n]
    assert np.array_equal(np.sort(points), np.arange(n))  # each point leaves exactly once
    assert np.all(np.diff(tree.parent[tree.child >= n]) >= -tree.parent.max())
    split = tree.child >= n
    assert np.all(tree.parent[split] < tree.child[split] - n)
    assert np.all(tree.lambda_value > 0.0)
    assert np.all(tree.stability >= 0.0)
    assert np.array_equal(
        np.sort(fit.labels[fit.labels >= 0]), np.sort(fit.labels[fit.labels >= 0])
    )
    assert fit.sizes.sum() == n - fit.n_noise
    assert fit.stabilities.shape == (fit.n_clusters,)


def test_a_cluster_is_never_smaller_than_min_cluster_size() -> None:
    x = _three_densities(2)
    for m in (5, 20, 40):
        fit = hdbscan(x, m)
        assert fit.n_clusters == 0 or fit.sizes.min() >= m


def test_min_samples_one_makes_the_tree_the_single_linkage_tree() -> None:
    x = _three_densities(3)[:150]
    fit = hdbscan(x, 6, min_samples=1)
    assert np.all(fit.core_distances == 0.0)
    assert fit.mst[:, 2].sum() == pytest.approx(
        sum(w for _, _, w in _prim(_distances(x))), rel=1e-12
    )


def test_coinciding_points_give_finite_stabilities() -> None:
    rng = np.random.default_rng(7)
    x = np.vstack([np.zeros((30, 2)), rng.standard_normal((60, 2)) + np.array([6.0, 6.0])])
    fit = hdbscan(x, 5)
    assert np.all(np.isfinite(fit.tree.stability))
    assert np.all(np.isfinite(fit.tree.lambda_value))
    assert fit.n_clusters >= 1


def test_hdbscan_checks_its_arguments() -> None:
    x, _ = _blobs(0)
    with pytest.raises(ValueError, match="min_cluster_size"):
        hdbscan(x, 1)
    with pytest.raises(ValueError, match="min_samples"):
        hdbscan(x, 5, min_samples=0)
    with pytest.raises(ValueError, match="min_samples"):
        hdbscan(x, 5, min_samples=len(x) + 1)
    with pytest.raises(ValueError, match="finite"):
        hdbscan(np.array([[0.0, np.inf], [1.0, 1.0], [2.0, 2.0]]), 2)


def test_three_blobs_are_found_by_hdbscan() -> None:
    x, truth = _blobs(9)
    fit = hdbscan(x, 20)
    assert fit.n_clusters == 3
    assert _partition(fit.labels) >= {
        frozenset(np.flatnonzero(fit.labels == g).tolist()) for g in range(3)
    }
    mapping = {
        g: np.bincount(fit.labels[truth == g][fit.labels[truth == g] >= 0]).argmax()
        for g in range(3)
    }
    assert len(set(mapping.values())) == 3


def test_a_small_min_cluster_size_finds_clusters_in_one_gaussian_cloud() -> None:
    """The docs give these counts: a result is no evidence of structure."""

    def with_a_cluster(size: int) -> int:
        return sum(
            hdbscan(np.random.default_rng(seed).standard_normal((200, 2)), size).n_clusters >= 1
            for seed in range(300)
        )

    assert (with_a_cluster(10), with_a_cluster(20), with_a_cluster(30)) == (192, 2, 0)


# --- The cosine distance --------------------------------------------------------------------


def _cosine(x: Floats) -> Floats:
    """1 - cos for every pair, from the definition, with 0 on the diagonal."""
    norms = np.sqrt((x * x).sum(axis=1))
    out: Floats = np.maximum(0.0, 1.0 - (x @ x.T) / (norms[:, None] * norms[None, :]))
    np.fill_diagonal(out, 0.0)
    return out


def _directions(seed: int) -> Floats:
    """Rows around two close directions and one wide one, plus scattered ones, any length."""
    rng = np.random.default_rng(seed)
    axes = np.array([[1.0, 0, 0, 0], [1.0, 0.35, 0, 0], [0, 0, 1.0, 0]])
    spread = (0.04, 0.04, 0.2)
    groups = [a + s * rng.standard_normal((70, 4)) for a, s in zip(axes, spread, strict=True)]
    x = np.vstack([*groups, rng.standard_normal((25, 4))])
    return x * rng.uniform(0.1, 50.0, size=(len(x), 1))


@pytest.mark.parametrize("seed", range(4))
def test_cosine_dbscan_equals_the_algorithm_on_cosine_distances(seed: int) -> None:
    x = _directions(seed)
    for eps, min_pts in [(0.004, 5), (0.02, 8), (0.2, 4)]:
        fit = dbscan(x, eps, min_samples=min_pts, metric="cosine")
        assert np.array_equal(fit.labels, naive_dbscan(x, eps, min_pts, _cosine))
        assert fit.metric == "cosine"
        assert fit.eps == eps


def test_cosine_k_distances_equal_the_definition() -> None:
    x = _directions(0)
    expected = np.sort(_cosine(x), axis=1)[:, 6]  # the point itself is at position 0
    assert np.allclose(k_distances(x, 6, metric="cosine"), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_cosine_hdbscan_equals_the_hierarchy_of_the_cosine_distance(seed: int) -> None:
    x = _directions(seed)
    for m, k in [(10, 10), (15, 5), (8, 8)]:
        fit = hdbscan(x, m, min_samples=k, metric="cosine")
        core, _ = _mutual_reachability(x, k, _cosine)
        assert np.allclose(fit.core_distances, core, rtol=0.0, atol=1e-12)
        groups, stability, weight = naive_hdbscan(x, m, k, distances=_cosine)
        assert fit.mst[:, 2].sum() == pytest.approx(weight, abs=1e-10)
        assert _partition(fit.labels) == groups
        assert np.allclose(np.sort(fit.stabilities), stability, rtol=1e-6)


def test_cosine_densities_ignore_the_length_of_a_row() -> None:
    x = _directions(1)
    unit = x / np.sqrt((x * x).sum(axis=1))[:, None]
    long = hdbscan(x, 10, metric="cosine")
    short = hdbscan(unit, 10, metric="cosine")
    assert np.array_equal(long.labels, short.labels)
    assert np.array_equal(
        dbscan(x, 0.01, metric="cosine").labels, dbscan(unit, 0.01, metric="cosine").labels
    )


def test_cosine_scaled_copies_are_at_distance_zero_and_stabilities_stay_finite() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal((6, 5))
    x = np.vstack([base * s for s in (1.0, 2.0, 3.0, 7.0, 11.0, 13.0)])
    fit = hdbscan(x, 4, metric="cosine")
    assert np.array_equal(fit.core_distances, np.zeros(len(x)))
    assert np.all(np.isfinite(fit.stabilities))
    assert fit.n_clusters == 6


@pytest.mark.parametrize("function", ["dbscan", "hdbscan", "k_distances"])
def test_cosine_density_functions_refuse_zero_rows_and_unknown_metrics(function: str) -> None:
    x = _directions(0)
    calls: dict[str, Callable[[Floats, str], object]] = {
        "dbscan": lambda a, m: dbscan(a, 0.1, metric=m),  # type: ignore[arg-type]
        "hdbscan": lambda a, m: hdbscan(a, metric=m),  # type: ignore[arg-type]
        "k_distances": lambda a, m: k_distances(a, 3, metric=m),  # type: ignore[arg-type]
    }
    with pytest.raises(ValueError, match="metric must be one of"):
        calls[function](x, "manhattan")
    x[3] = 0.0
    with pytest.raises(ValueError, match="row of zeros"):
        calls[function](x, "cosine")
