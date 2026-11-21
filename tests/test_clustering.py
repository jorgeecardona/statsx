"""CL-1: k-means, the silhouette and the gap statistic."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import choose_k, kmeans, silhouette

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
