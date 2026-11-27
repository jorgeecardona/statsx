"""CL-1: k-means, the silhouette and the gap statistic."""

import itertools
import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import (
    Metric,
    assign_clusters,
    choose_k,
    elbow_k,
    find_knees,
    kmeans,
    silhouette,
)

F64 = npt.NDArray[np.float64]

CENTRES = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])


def _blobs(
    per: int, seed: int, spread: float = 0.5
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    x = np.vstack([c + spread * rng.standard_normal((per, 2)) for c in CENTRES])
    return x, np.repeat(np.arange(3), per)


def _same_partition(a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> bool:
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_planted_clusters_are_found_and_the_fit_is_consistent() -> None:
    x, truth = _blobs(60, 0)
    fit = kmeans(x, 3, seed=1)
    assert _same_partition(fit.labels, truth)
    assert fit.converged
    for c in range(3):
        assert np.allclose(fit.centroids[c], x[fit.labels == c].mean(axis=0))
    direct = sum(float(((x[fit.labels == c] - fit.centroids[c]) ** 2).sum()) for c in range(3))
    assert fit.inertia == pytest.approx(direct, rel=1e-12)
    assert np.array_equal(fit.predict(x), fit.labels)
    assert fit.sizes.tolist() == [60, 60, 60]


def test_the_fit_reaches_the_best_partition_on_a_small_set() -> None:
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0.0, 0.3, 4), rng.normal(5.0, 0.3, 3), rng.normal(11.0, 0.3, 3)])
    x = x[:, None]
    best = np.inf
    for assignment in itertools.product(range(3), repeat=x.shape[0]):
        a = np.asarray(assignment)
        total = 0.0
        for c in range(3):
            rows = x[a == c]
            if rows.size:
                total += float(((rows - rows.mean(axis=0)) ** 2).sum())
        best = min(best, total)
    assert kmeans(x, 3, n_init=10, seed=0).inertia == pytest.approx(best, rel=1e-9)


def test_the_result_does_not_depend_on_the_number_of_threads() -> None:
    x, _ = _blobs(900, 1, spread=1.5)
    one = kmeans(x, 4, seed=2, threads=1)
    four = kmeans(x, 4, seed=2, threads=4)
    assert np.array_equal(one.labels, four.labels)
    assert np.array_equal(one.centroids, four.centroids)
    assert one.inertia == four.inertia


def test_more_runs_never_give_a_worse_fit() -> None:
    x, _ = _blobs(80, 2, spread=2.5)
    assert kmeans(x, 5, n_init=10, seed=4).inertia <= kmeans(x, 5, n_init=1, seed=4).inertia


def test_the_extremes_are_exact() -> None:
    x, _ = _blobs(10, 3)
    assert kmeans(x, 1).centroids[0] == pytest.approx(x.mean(axis=0))
    assert kmeans(x, x.shape[0], n_init=1).inertia == pytest.approx(0.0, abs=1e-9)


def test_duplicate_points_do_not_break_the_seeding() -> None:
    x = np.vstack([np.zeros((20, 2)), np.ones((20, 2))])
    fit = kmeans(x, 3, seed=0)
    assert fit.inertia == pytest.approx(0.0, abs=1e-12)
    assert fit.centroids.shape == (3, 2)


def test_predict_and_input_checks() -> None:
    x, _ = _blobs(20, 4)
    fit = kmeans(x, 3)
    labels, dist = fit.distances(CENTRES)
    assert sorted(labels.tolist()) == [0, 1, 2]
    assert np.all(dist < 1.0)
    with pytest.raises(ValueError, match="columns"):
        fit.predict(np.ones((3, 5)))
    for bad in (
        lambda: kmeans(x, 0),
        lambda: kmeans(x, x.shape[0] + 1),
        lambda: kmeans(np.array([[1.0, np.nan], [2.0, 3.0]]), 1),
        lambda: kmeans(np.ones(5), 1),
        lambda: kmeans(x, 2, n_init=0),
        lambda: kmeans(x, 2, threads=0),
    ):
        with pytest.raises(ValueError, match=r"k must|finite|shape|at least"):
            bad()


def _silhouette_by_hand(
    x: npt.NDArray[np.float64], lab: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    n = x.shape[0]
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    out = np.zeros(n)
    for i in range(n):
        own = lab == lab[i]
        if own.sum() == 1:
            continue
        a = dist[i, own].sum() / (own.sum() - 1)
        b = min(dist[i, lab == c].mean() for c in set(lab.tolist()) if c != lab[i])
        out[i] = (b - a) / max(a, b)
    return out


def test_the_silhouette_equals_the_definition() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((40, 3))
    lab = np.concatenate([np.zeros(15), np.ones(14), np.full(10, 2), [3]]).astype(np.int64)
    result = silhouette(x, lab)
    assert np.allclose(result.values, _silhouette_by_hand(x, lab))
    assert result.values[-1] == 0.0  # a point alone in its cluster


def test_the_silhouette_separates_real_clusters_from_random_labels() -> None:
    x, truth = _blobs(50, 6)
    good = silhouette(x, truth)
    assert good.mean > 0.8
    assert good.n_negative == 0
    shuffled = silhouette(x, np.random.default_rng(0).permutation(truth))
    assert shuffled.mean < 0.1
    assert good.by_cluster.shape == (3,)


def test_silhouette_checks_its_input() -> None:
    x, truth = _blobs(10, 7)
    with pytest.raises(ValueError, match="at least 2 clusters"):
        silhouette(x, np.zeros(30, dtype=np.int64))
    with pytest.raises(ValueError, match="one non-negative integer"):
        silhouette(x, truth[:5])


def test_the_gap_statistic_finds_three_clusters() -> None:
    x, _ = _blobs(50, 8)
    result = choose_k(x, 6, n_reference=20, seed=1)
    assert result.k == 3
    assert result.found
    assert result.ks == (1, 2, 3, 4, 5, 6)
    assert result.gap.shape == (6,)


def test_the_gap_statistic_says_one_for_uniform_data_most_of_the_time() -> None:
    picks: list[int] = []
    for seed in range(12):
        x = np.random.default_rng(100 + seed).random((120, 2))
        picks.append(choose_k(x, 5, n_reference=15, seed=seed).k)
    assert sum(k == 1 for k in picks) >= 9


def test_choose_k_checks_its_input() -> None:
    x, _ = _blobs(10, 9)
    with pytest.raises(ValueError, match="k_max"):
        choose_k(x, 0)
    with pytest.raises(ValueError, match="n_reference"):
        choose_k(x, 3, n_reference=1)
    with pytest.raises(ValueError, match="reference"):
        choose_k(x, 3, reference="sphere")  # type: ignore[arg-type]


def test_the_box_reference_also_finds_the_clusters() -> None:
    x, _ = _blobs(50, 10)
    assert choose_k(x, 5, reference="box", seed=2).k == 3


def test_find_knees_reproduces_figure_2_of_the_kneedle_paper() -> None:
    # y = -1/(x + 0.1) + 5 on ten points in [0, 1]: Figure 2 reports the knee at x = 0.22, with
    # S = 1, detected after the point x = 0.55.
    x = np.linspace(0.0, 1.0, 10)
    y = -1.0 / (x + 0.1) + 5.0
    found = find_knees(x, y)
    assert [round(float(x[i]), 2) for i in found.indices] == [0.22]
    threshold = found.difference[2] - 1.0 / 9.0
    first_below = next(j for j in range(3, 10) if found.difference[j] < threshold)
    assert round(float(x[first_below]), 2) == 0.56
    # the knee is only known once that point is seen
    assert find_knees(x[:first_below], y[:first_below]).difference.shape == (first_below,)


def _knees_by_definition(x: F64, y: F64, s: float) -> list[int]:
    """Steps 2 to 6 of the algorithm, written from the paper."""
    n = x.shape[0]
    px: list[float] = [float(v) for v in x]
    py: list[float] = [float(v) for v in y]
    xs = [(v - min(px)) / (max(px) - min(px)) for v in px]
    ys = [(v - min(py)) / (max(py) - min(py)) for v in py]
    d = [b - a for a, b in zip(xs, ys, strict=True)]
    step = sum(xs[i + 1] - xs[i] for i in range(n - 1)) / (n - 1)
    maxima = [i for i in range(1, n - 1) if d[i - 1] < d[i] > d[i + 1]]
    out: list[int] = []
    for i in maxima:
        threshold = d[i] - s * step
        for j in range(i + 1, n):
            if d[j] < threshold:
                out.append(i)
                break
            if d[j] > d[j - 1]:  # the difference turned up before it fell far enough
                break
    return out


def test_find_knees_equals_the_algorithm_on_random_curves() -> None:
    rng = np.random.default_rng(0)
    declared = 0
    for _ in range(300):
        n = int(rng.integers(3, 25))
        x = np.cumsum(rng.uniform(0.1, 1.0, n))
        y = np.cumsum(rng.uniform(-0.3, 1.0, n))
        for s in (0.0, 1.0, 3.0):
            expected = _knees_by_definition(x, y, s)
            assert list(find_knees(x, y, sensitivity=s).indices) == expected
            declared += len(expected)
    assert declared > 100  # the comparison saw many knees


def test_find_knees_checks_its_input() -> None:
    x = np.arange(5.0)
    with pytest.raises(ValueError, match="at least 3"):
        find_knees(x[:2], x[:2])
    with pytest.raises(ValueError, match="increasing"):
        find_knees(x[::-1], x)
    with pytest.raises(ValueError, match="constant"):
        find_knees(x, np.ones(5))
    with pytest.raises(ValueError, match="finite"):
        find_knees(x, np.full(5, np.nan))
    with pytest.raises(ValueError, match="sensitivity"):
        find_knees(x, x**0.5, sensitivity=-1.0)
    # a straight line has no bend
    assert find_knees(x, x).indices == ()


def _planted(n_groups: int, n_per: int = 150, dim: int = 8, seed: int = 0) -> F64:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 8.0, size=(n_groups, dim))
    return np.vstack([c + rng.normal(0.0, 1.0, size=(n_per, dim)) for c in centres])


@pytest.mark.parametrize("n_groups", [3, 5, 7])
def test_the_elbow_finds_planted_clusters_for_any_seed(n_groups: int) -> None:
    x = _planted(n_groups)
    picks = [elbow_k(x, 14, seed=seed) for seed in (0, 7, 42)]
    assert [p.k for p in picks] == [n_groups] * 3
    assert all(p.is_knee and p.inertia_rises == 0 for p in picks)


def test_the_elbow_result_holds_the_curve_it_used() -> None:
    x = _planted(4)
    result = elbow_k(x, 9, k_min=2, seed=3)
    assert result.ks == tuple(range(2, 10))
    for k, value in zip(result.ks, result.inertia, strict=True):
        assert value == kmeans(x, k, n_init=5, seed=3).inertia
    logs = np.log(result.inertia)
    turned = (logs.max() - logs) / (logs.max() - logs.min())
    line = (np.array(result.ks) - 2) / 7
    assert np.allclose(result.difference, turned - line, rtol=0, atol=1e-12)
    assert result.k == result.ks[int(np.argmax(result.difference))]
    assert result.k in result.knees
    linear = elbow_k(x, 9, seed=3, scale="linear")
    assert linear.scale == "linear"
    assert np.array_equal(linear.inertia, result.inertia)
    assert not np.allclose(linear.difference, result.difference)


def test_the_elbow_picks_a_k_for_uniform_data_which_has_no_clusters() -> None:
    x = np.random.default_rng(0).uniform(size=(500, 5))
    result = elbow_k(x, 10)
    assert result.k > 1  # it cannot answer "no clusters"
    assert choose_k(x, 6, n_reference=15).k == 1


def test_the_elbow_checks_its_input() -> None:
    x = _planted(3, 20)
    with pytest.raises(ValueError, match="k_min"):
        elbow_k(x, 3, k_min=2)
    with pytest.raises(ValueError, match="k_max <= n"):
        elbow_k(x, 61)
    with pytest.raises(ValueError, match="scale"):
        elbow_k(x, 6, scale="sqrt")  # type: ignore[arg-type]
    repeated = np.repeat(np.eye(3), 5, axis=0)
    with pytest.raises(ValueError, match="inertia 0"):
        elbow_k(repeated, 6)


def _cosine_objective(x: F64, labels: npt.NDArray[np.int64], k: int) -> float:
    """Sum of 1 - cos to the unit mean of each cluster: the objective of spherical k-means."""
    unit = x / np.sqrt((x * x).sum(axis=1))[:, None]
    total = 0.0
    for c in range(k):
        members = unit[labels == c]
        if len(members) == 0:
            continue
        centre = members.sum(axis=0)
        total += float(len(members) - np.sqrt(centre @ centre))
    return total


def _rays(n_per: int, seed: int) -> tuple[F64, npt.NDArray[np.int64]]:
    """Rows around three directions, with lengths from 0.1 to 100."""
    rng = np.random.default_rng(seed)
    axes = np.eye(6)[:3]
    x = np.vstack([a + rng.normal(0.0, 0.12, size=(n_per, 6)) for a in axes])
    x = x * np.exp(rng.uniform(np.log(0.1), np.log(100.0), size=(len(x), 1)))
    return x, np.repeat(np.arange(3), n_per)


def test_cosine_kmeans_reaches_the_best_partition_on_a_small_set() -> None:
    x = np.random.default_rng(4).standard_normal((10, 3))
    best = min(
        _cosine_objective(x, np.array(labels), 3)
        for labels in itertools.product(range(3), repeat=10)
    )
    fit = kmeans(x, 3, metric="cosine", n_init=30)
    assert fit.inertia == pytest.approx(best, rel=1e-9)


def test_a_cosine_fit_has_unit_centres_and_labels_by_the_largest_cosine() -> None:
    x, truth = _rays(80, 0)
    fit = kmeans(x, 3, metric="cosine")
    unit = x / np.sqrt((x * x).sum(axis=1))[:, None]
    assert fit.metric == "cosine"
    assert np.allclose(np.sqrt((fit.centroids**2).sum(axis=1)), 1.0, rtol=0.0, atol=1e-12)
    cos = unit @ fit.centroids.T
    assert np.array_equal(fit.labels, np.argmax(cos, axis=1))
    for c in range(3):
        mean = unit[fit.labels == c].sum(axis=0)
        assert np.allclose(fit.centroids[c], mean / np.sqrt(mean @ mean), rtol=0.0, atol=1e-12)
    assert fit.inertia == pytest.approx(_cosine_objective(x, fit.labels, 3), rel=1e-9)
    # the planted directions, whatever the lengths
    assert all(len(set(fit.labels[truth == c].tolist())) == 1 for c in range(3))


def test_cosine_kmeans_ignores_the_length_of_a_row_and_predicts_by_cosine() -> None:
    x, _ = _rays(50, 1)
    unit = x / np.sqrt((x * x).sum(axis=1))[:, None]
    long = kmeans(x, 3, metric="cosine")
    short = kmeans(unit, 3, metric="cosine")
    assert np.array_equal(long.labels, short.labels)
    assert np.allclose(long.centroids, short.centroids, rtol=0.0, atol=1e-12)
    new, _ = _rays(20, 2)
    labels, distance = long.distances(new)
    unit_new = new / np.sqrt((new * new).sum(axis=1))[:, None]
    cos = unit_new @ long.centroids.T
    assert np.array_equal(labels, np.argmax(cos, axis=1))
    assert np.allclose(distance, 1.0 - cos.max(axis=1), rtol=0.0, atol=1e-12)
    assert np.array_equal(long.predict(new * 1000.0), labels)


def test_the_cosine_fit_does_not_depend_on_the_number_of_threads() -> None:
    x, _ = _rays(1500, 3)
    one = kmeans(x, 3, metric="cosine", threads=1)
    four = kmeans(x, 3, metric="cosine", threads=4)
    assert np.array_equal(one.labels, four.labels)
    assert np.array_equal(one.centroids, four.centroids)
    assert one.inertia == four.inertia


def test_the_cosine_elbow_finds_the_directions_whatever_the_lengths() -> None:
    x, _ = _rays(80, 4)
    result = elbow_k(x, 9, metric="cosine")
    assert result.k == 3
    assert result.inertia[1] == kmeans(x, 3, metric="cosine", n_init=5).inertia
    # by the Euclidean distance the lengths decide
    assert elbow_k(x, 9, scale="linear").inertia[1] > 100.0 * result.inertia[1]


def test_cosine_kmeans_refuses_zero_rows_and_unknown_metrics() -> None:
    x, _ = _rays(10, 0)
    with pytest.raises(ValueError, match="metric must be one of"):
        kmeans(x, 2, metric="manhattan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric must be one of"):
        elbow_k(x, 5, metric="manhattan")  # type: ignore[arg-type]
    fit = kmeans(x, 3, metric="cosine")
    x[0] = 0.0
    with pytest.raises(ValueError, match="row of zeros"):
        kmeans(x, 2, metric="cosine")
    with pytest.raises(ValueError, match="row of zeros"):
        fit.predict(x)
    # the Euclidean distance has no such problem
    assert kmeans(x, 2).k == 2


def test_the_gap_margins_are_the_rule_and_the_pick_is_the_first_that_meets_it() -> None:
    x = _planted(3, 80, 4)
    result = choose_k(x, 6, n_reference=10, seed=1)
    expected = [
        result.gap[i] - (result.gap[i + 1] - result.se[i + 1]) for i in range(len(result.ks) - 1)
    ]
    assert np.array_equal(result.margins, expected)
    first = next(i for i, m in enumerate(expected) if m >= 0)
    assert result.found
    assert result.k == result.ks[first] == 3
    closest = min(abs(expected[i]) / result.se[i + 1] for i in range(first + 1))
    assert result.decision_margin == closest
    assert result.decision_margin > 1.0  # well separated: no comparison was close


def test_a_flat_gap_curve_has_a_pick_that_moves_and_a_small_decision_margin() -> None:
    # fastumap and statsx issue 17: unit vectors in many dimensions, 8 planted centres
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(8, 128))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    x = centres[rng.integers(0, 8, size=600)] + 0.35 * rng.normal(size=(600, 128))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    results = [
        choose_k(x, 6, n_reference=b, n_init=1, reference="box", seed=0) for b in (2, 3, 5, 10)
    ]
    assert len({r.k for r in results}) > 1  # the pick depends on n_reference
    assert all(r.found for r in results)
    assert all(r.decision_margin < 1.0 for r in results)  # and each result says so


def test_k_min_starts_the_gap_rule_later_on_the_same_curve() -> None:
    x = _planted(3, 60, 4)
    full = choose_k(x, 7, n_reference=8, seed=2)
    later = choose_k(x, 7, k_min=4, n_reference=8, seed=2)
    assert later.ks == (4, 5, 6, 7)
    assert np.array_equal(later.gap, full.gap[3:])
    assert np.array_equal(later.se, full.se[3:])
    assert np.array_equal(later.margins, full.margins[3:])
    met = [k for k, m in zip(later.ks, later.margins, strict=False) if m >= 0]
    assert later.k == (met[0] if met else 7)
    single = choose_k(x, 3, k_min=3, n_reference=2)
    assert single.k == 3
    assert not single.found
    assert math.isnan(single.decision_margin)
    with pytest.raises(ValueError, match="k_min"):
        choose_k(x, 3, k_min=4)
    with pytest.raises(ValueError, match="k_min"):
        choose_k(x, 3, k_min=0)


@pytest.mark.parametrize("offset", [1e8, 1e11])
def test_kmeans_and_the_silhouette_are_unchanged_far_from_the_origin(offset: float) -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    x = np.repeat(centres, 200, axis=0) + 0.3 * rng.normal(size=(600, 3))
    near = kmeans(x, 3)
    far = kmeans(x + offset, 3)
    # the same partition, whatever the numbering
    assert len(set(zip(near.labels.tolist(), far.labels.tolist(), strict=True))) == 3
    assert far.inertia == pytest.approx(near.inertia, rel=1e-4)
    truth = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    found = assign_clusters(x + offset, centres + offset)
    assert np.array_equal(found.nearest, truth)
    assert silhouette(x + offset, near.labels).mean == pytest.approx(
        silhouette(x, near.labels).mean, rel=1e-4
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_assign_clusters_equals_the_nearest_centre_by_definition(metric: Metric) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6)) + 0.5
    centres = rng.normal(size=(7, 6)) * rng.uniform(0.1, 20.0, size=(7, 1))
    if metric == "euclidean":
        full = np.sqrt(((x[:, None, :] - centres[None]) ** 2).sum(axis=2))
    else:
        norms = np.sqrt((x * x).sum(axis=1))[:, None] * np.sqrt((centres**2).sum(axis=1))[None]
        full = 1.0 - (x @ centres.T) / norms
    limit = float(np.median(full.min(axis=1)))
    found = assign_clusters(x, centres, metric=metric, max_distance=limit)
    assert np.array_equal(found.nearest, np.argmin(full, axis=1))
    assert np.allclose(found.distances, full.min(axis=1), rtol=0.0, atol=1e-12)
    assert np.array_equal(found.labels, np.where(found.distances > limit, -1, found.nearest))
    assert found.n_noise == int((found.distances > limit).sum()) > 0
    assert found.metric == metric
    assert found.max_distance == limit
    without = assign_clusters(x, centres, metric=metric)
    assert np.array_equal(without.labels, without.nearest)
    assert without.n_noise == 0


def test_a_row_at_the_limit_keeps_its_label_and_a_tie_goes_to_the_first_centre() -> None:
    centres = np.array([[0.0, 0.0], [10.0, 0.0]])
    x = np.array([[3.0, 4.0], [5.0, 0.0], [3.0, 4.000001]])
    found = assign_clusters(x, centres, max_distance=5.0)
    assert found.distances[0] == 5.0
    assert found.labels.tolist() == [0, 0, -1]
    assert assign_clusters(x[:1], centres[:1], max_distance=0.0).labels.tolist() == [-1]


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_a_fit_assigns_in_its_own_distance(metric: Metric) -> None:
    x, _ = _rays(60, 5)
    fit = kmeans(x, 3, metric=metric)
    again = fit.assign(x)
    assert again.metric == metric
    assert np.array_equal(again.labels, fit.labels)
    assert np.array_equal(again.labels, fit.predict(x))
    squared = fit.distances(x)[1]
    expected = np.sqrt(squared) if metric == "euclidean" else squared
    assert np.allclose(again.distances, expected, rtol=1e-9, atol=1e-12)
    far = fit.assign(x, max_distance=float(np.quantile(again.distances, 0.9)))
    assert 0 < far.n_noise <= len(x) // 10 + 1


@pytest.mark.parametrize(
    ("scale", "offset", "columns"),
    [(1.0, 0.0, 3), (1.0, 1e9, 40), (2.0**-140, 0.0, 40), (2.0**133, 0.0, 40), (1.0, 0.0, 1100)],
    ids=["small", "far", "below float32", "above float32", "wide"],
)
def test_the_screen_of_the_assignment_never_changes_a_label(
    scale: float, offset: float, columns: int
) -> None:
    # whole numbers times a power of two, so every squared distance is exact and many tie
    rng = np.random.default_rng(columns)
    centres = rng.integers(-2, 3, size=(25, columns)).astype(np.float64)
    centres[10] = centres[3]
    centres[11] = centres[3]
    centres[11, 0] += 1.0
    x = rng.integers(-2, 3, size=(5000, columns)).astype(np.float64)
    x[:25] = centres
    x, centres = x * scale + offset, centres * scale + offset
    squared = np.stack([((x - c) ** 2).sum(axis=1) for c in centres], axis=1)
    found = assign_clusters(x, centres)
    assert np.array_equal(found.nearest, squared.argmin(axis=1))
    assert np.array_equal(found.distances, np.sqrt(squared.min(axis=1)))
    assert found.nearest[10] == 3  # a copy of a centre goes to the first of the two


def test_assign_clusters_checks_its_input() -> None:
    x = np.ones((3, 2))
    with pytest.raises(ValueError, match="columns"):
        assign_clusters(x, np.ones((2, 3)))
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="max_distance"):
            assign_clusters(x, x, max_distance=bad)
    with pytest.raises(ValueError, match="metric must be one of"):
        assign_clusters(x, x, metric="manhattan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="row of zeros"):
        assign_clusters(x, np.zeros((1, 2)), metric="cosine")
    with pytest.raises(ValueError, match="finite"):
        assign_clusters(x, np.full((1, 2), np.nan))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_the_dispersion_is_the_mean_distance_of_each_cluster_to_its_centre(metric: Metric) -> None:
    x, _ = _rays(700, 6)  # more rows than one block
    fit = kmeans(x, 3, metric=metric)
    for c in range(3):
        members = x[fit.labels == c]
        centre = fit.centroids[c]
        if metric == "euclidean":
            distance = np.sqrt(((members - centre) ** 2).sum(axis=1))
        else:
            distance = 1.0 - (members @ centre) / np.sqrt((members**2).sum(axis=1))
        assert fit.dispersion[c] == pytest.approx(float(distance.mean()), rel=1e-9)
        assert fit.sizes[c] == len(members)


def test_a_wide_cluster_has_a_larger_dispersion_than_a_tight_one_of_the_same_size() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 0.1, size=(200, 4)), rng.normal(20.0, 3.0, size=(200, 4))])
    fit = kmeans(x, 2)
    tight, wide = int(fit.labels[0]), int(fit.labels[-1])
    assert fit.sizes[tight] == fit.sizes[wide] == 200
    assert fit.dispersion[wide] > 10 * fit.dispersion[tight]


def test_kmeans_accepts_arrays_that_are_not_row_major() -> None:
    # the Rust kernel reads the caller's rows where they are, so other layouts are converted first
    x = _planted(3, 100, 12)
    columns_first = np.asfortranarray(x)
    every_other = np.repeat(x, 2, axis=0)[::2]
    expected = kmeans(x, 3, seed=1)
    for layout in (columns_first, every_other, x.astype(np.float32).astype(np.float64)):
        fit = kmeans(layout, 3, seed=1)
        assert np.array_equal(fit.labels, expected.labels)
        assert fit.inertia == pytest.approx(expected.inertia, rel=1e-6)
    assert np.array_equal(expected.predict(columns_first), expected.labels)
    turned = replace(expected, centroids=np.asfortranarray(expected.centroids))
    assert np.array_equal(turned.predict(x), expected.labels)
    assert np.array_equal(assign_clusters(every_other, turned.centroids).labels, expected.labels)


def test_the_distances_do_not_depend_on_how_the_columns_are_grouped() -> None:
    # the squared distance is summed in blocks of eight columns; check widths around the block
    rng = np.random.default_rng(2)
    for width in (1, 7, 8, 9, 31, 1024):
        x = rng.normal(size=(40, width))
        centres = rng.normal(size=(3, width))
        found = assign_clusters(x, centres)
        squared = found.distances**2
        full = ((x[:, None, :] - centres[None]) ** 2).sum(axis=2)
        assert np.allclose(squared, full.min(axis=1), rtol=1e-12)
