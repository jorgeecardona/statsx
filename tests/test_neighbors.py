"""Neighbours that the caller gives to ``umap``, and the measured recall of a table (MG-9)."""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
import pytest

from statsx.embedding import (
    Metric,
    UMAPFit,
    nearest_neighbors,
    neighbor_preservation,
    neighbor_recall,
    umap,
)

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def _points(n: int = 200, dim: int = 8, seed: int = 0) -> F64:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n, dim)) + rng.integers(0, 4, (n, 1)) * 3.0


def _exact(x: F64, k: int, metric: Metric) -> tuple[I64, F64]:
    """The ``k`` nearest other rows of each row, from the full distance matrix."""
    if metric == "euclidean":
        full = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    else:
        unit = x / np.linalg.norm(x, axis=1)[:, None]
        full = 1.0 - unit @ unit.T
    np.fill_diagonal(full, np.inf)
    idx = np.argsort(full, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(full, idx, axis=1)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_the_nearest_neighbours_are_the_ones_of_the_full_distance_matrix(metric: Metric) -> None:
    x = _points()
    table = nearest_neighbors(x, 12, metric=metric)
    idx, dist = _exact(x, 12, metric)
    assert np.array_equal(table.indices, idx)
    assert table.distances == pytest.approx(dist, rel=1e-9, abs=1e-12)
    assert table.metric == metric


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_the_table_of_the_exact_search_gives_the_same_fit(metric: Metric, offset: float) -> None:
    x = _points() + offset
    found = umap(x, metric=metric, n_epochs=30)
    table = nearest_neighbors(x, 15, metric=metric)
    for given in (table.indices, (table.indices, table.distances)):
        fit = umap(x, metric=metric, n_epochs=30, neighbors=given)
        for name in ("embedding", "neighbors", "distances", "rho", "sigma"):
            assert np.array_equal(getattr(fit, name), getattr(found, name)), name
        assert (fit.neighbor_source, found.neighbor_source) == ("given", "exact")


def test_a_wider_table_in_any_order_with_the_row_itself_gives_the_same_fit() -> None:
    x = _points()
    x[7] = x[3]  # a copy of a row: another library can list it before the row itself
    found = umap(x, n_neighbors=10, n_epochs=20)
    wide = nearest_neighbors(x, 25).indices
    own = np.arange(200)[:, None]
    shuffled = np.random.default_rng(1).permuted(np.hstack([wide, own]), axis=1)
    fit = umap(x, n_neighbors=10, n_epochs=20, neighbors=shuffled.astype(np.uint32))
    assert np.array_equal(fit.neighbors, found.neighbors)
    assert np.array_equal(fit.embedding, found.embedding)


def test_distances_in_float32_are_accepted_and_the_fit_keeps_exact_ones() -> None:
    x = _points(dim=64)
    table = nearest_neighbors(x, 15)
    rough = table.distances.astype(np.float32)
    fit = umap(x, n_epochs=5, neighbors=(table.indices, rough))
    assert np.array_equal(fit.distances, table.distances)


@pytest.mark.parametrize(
    ("metric", "change", "message"),
    [
        ("euclidean", "squared", "squared distances"),
        ("cosine", "similarity", "cosine similarities"),
        ("euclidean", "doubled", "are not the euclidean distances"),
        ("euclidean", "short", "have shape"),
    ],
)
def test_distances_that_are_not_those_of_the_data_are_refused(
    metric: Metric, change: str, message: str
) -> None:
    x = _points()
    table = nearest_neighbors(x, 15, metric=metric)
    d = table.distances
    stated = {"squared": d**2, "similarity": 1.0 - d, "doubled": 2 * d, "short": d[:, :3]}[change]
    with pytest.raises(ValueError, match=message):
        umap(x, metric=metric, neighbors=(table.indices, stated))


def test_a_table_that_cannot_be_a_table_of_neighbours_is_refused() -> None:
    x = _points()
    good = nearest_neighbors(x, 15).indices

    def changed(value: int, column: int = 0) -> I64:
        bad = good.copy()
        bad[4, column] = value
        return bad

    cases: list[tuple[object, str]] = [
        (changed(-1), "holds -1"),
        (changed(200), "holds 200"),
        (changed(int(good[4, 1])), "row 4 of neighbors names a row twice"),
        (changed(4), "at least 15 other rows"),
        (good[:, :14], "at least 15 other rows"),
        (good[:150], "must have shape"),
        (good.astype(np.float64), "whole row numbers"),
        ((good, good, good), "must be \\(indices, distances\\)"),
    ]
    for table, message in cases:
        with pytest.raises(ValueError, match=message):
            umap(x, neighbors=table)  # type: ignore[arg-type]


def test_recall_is_one_for_the_exact_table_and_falls_by_what_is_missing() -> None:
    x = _points(300)
    table = nearest_neighbors(x, 15).indices
    whole = neighbor_recall(x, table)
    assert (whole.recall, whole.standard_error, whole.worst_decile) == (1.0, 0.0, 1.0)
    assert (whole.n_sample, whole.k) == (300, 15)
    # replace the nearest neighbour of every row by its farthest row
    worse = table.copy()
    full = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    worse[:, 0] = full.argmax(axis=1)
    part = neighbor_recall(x, worse, k=15)
    assert part.recall == pytest.approx(14 / 15)
    assert part.per_row == pytest.approx(np.full(300, 14 / 15))


def test_recall_counts_a_neighbour_at_the_same_distance_as_found() -> None:
    x = np.repeat(_points(40), 6, axis=0)  # every row has 5 copies
    table = nearest_neighbors(x, 5).indices
    # the other five copies of each row, listed from the last to the first
    copies = [[j for j in range(i // 6 * 6, i // 6 * 6 + 6) if j != i] for i in range(240)]
    swapped = np.array(copies)[:, ::-1]
    assert neighbor_recall(x, swapped).recall == 1.0
    assert neighbor_recall(x, table, k=3).recall == 1.0


def test_recall_on_a_sample_estimates_the_recall_of_all_rows() -> None:
    x = _points(600)
    table = nearest_neighbors(x, 10).indices
    rng = np.random.default_rng(2)
    full = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    bad = rng.choice(600, 150, replace=False)
    table[bad, 0] = full[bad].argmax(axis=1)
    truth = neighbor_recall(x, table, sample=600)
    assert truth.recall == pytest.approx(1 - 150 / 6000)
    assert truth.worst_decile == pytest.approx(0.9)
    estimates = [neighbor_recall(x, table, sample=100, seed=s) for s in range(40)]
    errors = np.array([e.recall - truth.recall for e in estimates])
    # about 95% of the estimates lie within two standard errors of the truth
    assert np.mean([abs(e.recall - truth.recall) <= 2 * e.standard_error for e in estimates]) > 0.85
    assert abs(errors.mean()) < estimates[0].standard_error
    one = neighbor_recall(x, table, sample=100, seed=3, threads=1)
    many = neighbor_recall(x, table, sample=100, seed=3, threads=3)
    assert np.array_equal(one.rows, many.rows) and np.array_equal(one.per_row, many.per_row)
    assert one.n_sample == 100


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_a_fit_measures_its_own_neighbours(metric: Metric) -> None:
    x = _points()
    assert umap(x, metric=metric, n_epochs=2).neighbor_recall(sample=50).recall == 1.0
    half = nearest_neighbors(x, 30, metric=metric).indices[:, ::2]
    fit = umap(x, metric=metric, n_epochs=2, neighbors=half)
    assert 0.4 < fit.neighbor_recall().recall < 0.7


def test_recall_refuses_settings_out_of_range() -> None:
    x = _points()
    table = nearest_neighbors(x, 5).indices
    with pytest.raises(ValueError, match="sample must be at least 2"):
        neighbor_recall(x, table, sample=1)
    with pytest.raises(ValueError, match="k must be in"):
        neighbor_recall(x, table, k=200)
    with pytest.raises(ValueError, match="at least 6 other rows"):
        neighbor_recall(x, table, k=6)
    with pytest.raises(ValueError, match="k must be in"):
        nearest_neighbors(x, 0)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_preservation_on_a_sample_is_the_mean_over_those_rows(metric: Metric) -> None:
    x = _points()
    y = umap(x, metric=metric, n_epochs=30).embedding
    k = 9
    near_x = _exact(x, k, metric)[0]
    near_y = _exact(y, k, "euclidean")[0]
    share = np.array([len(set(near_x[i]) & set(near_y[i])) / k for i in range(200)])
    rows = np.sort(np.random.default_rng(4).choice(200, 60, replace=False))
    got = neighbor_preservation(x, y, k, metric=metric, sample=60, seed=4)
    assert got == pytest.approx(float(share[rows].mean()))
    everyone = neighbor_preservation(x, y, k, metric=metric, sample=1000)
    assert everyone == neighbor_preservation(x, y, k, metric=metric)


def test_a_sample_leaves_out_the_row_itself_when_it_has_many_copies() -> None:
    x = np.repeat(_points(10), 20, axis=0)
    # every row has 19 copies at distance 0, so a search for 6 can miss the row itself
    assert neighbor_preservation(x, x, 5, sample=150) == pytest.approx(
        neighbor_preservation(x, x, 5), abs=0.05
    )
    assert neighbor_recall(x, nearest_neighbors(x, 5).indices, sample=150).recall == 1.0


def test_a_fit_from_a_given_table_survives_a_save_and_older_files_say_exact() -> None:
    x = _points()
    fit = umap(x, n_epochs=2, neighbors=nearest_neighbors(x, 15).indices)
    buffer = io.BytesIO()
    fit.to_npz(buffer)
    buffer.seek(0)
    assert UMAPFit.from_npz(buffer).neighbor_source == "given"
    buffer.seek(0)
    with np.load(buffer) as store:
        arrays = {name: store[name] for name in store.files}

    def saved(**changes: object) -> io.BytesIO:
        out = io.BytesIO()
        np.savez(out, **{**arrays, **changes})  # type: ignore[arg-type]
        out.seek(0)
        return out

    version_2 = {k: v for k, v in arrays.items() if k != "neighbor_source"}
    older = io.BytesIO()
    np.savez(older, **{**version_2, "format_version": np.int64(2)})  # type: ignore[arg-type]
    older.seek(0)
    assert UMAPFit.from_npz(older).neighbor_source == "exact"
    itself = fit.neighbors.copy()
    itself[0, 0] = 0
    twice = fit.neighbors.copy()
    twice[0, 0] = twice[0, 1]
    for damaged in (
        saved(neighbors=itself),
        saved(neighbors=twice),
        saved(neighbor_source=np.array("guessed")),
    ):
        with pytest.raises(ValueError, match="cannot have"):
            UMAPFit.from_npz(damaged)
