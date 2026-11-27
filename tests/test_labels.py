"""Labels that guide a UMAP map, and how much of its separation they explain."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from statsx.embedding import UMAPFit, label_agreement, umap

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def _classes(n: int, shift: float, seed: int) -> tuple[F64, I64]:
    """Three classes in 10 columns; ``shift`` moves each class along its own column."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    centres = np.zeros((3, 10))
    centres[[0, 1, 2], [0, 1, 2]] = shift
    return rng.standard_normal((n, 10)) + centres[y], y.astype(np.int64)


def _map(
    x: F64,
    labels: npt.ArrayLike | None = None,
    label_weight: float | None = None,
    threads: int | None = None,
) -> UMAPFit:
    return umap(
        x,
        n_neighbors=8,
        n_epochs=60,
        seed=3,
        labels=labels,
        label_weight=label_weight,
        threads=threads,
    )


def _same_map(a: UMAPFit, b: UMAPFit) -> bool:
    return (
        np.array_equal(a.graph.indptr, b.graph.indptr)
        and np.array_equal(a.graph.indices, b.graph.indices)
        and np.array_equal(a.graph.weights, b.graph.weights)
        and np.array_equal(a.embedding, b.embedding)
    )


def _formula(fit: UMAPFit) -> F64:
    """The graph of a fit with labels, from the formula of D-80 in NumPy."""
    assert fit.supervision is not None
    y, w = fit.supervision.labels, fit.supervision.label_weight
    rho, sigma = fit.rho[:, None], fit.sigma[:, None]
    directed = np.exp(-np.maximum(0.0, fit.distances - rho) / sigma)
    near = y[fit.neighbors]
    differs = (y[:, None] >= 0) & (near >= 0) & (near != y[:, None])
    directed = np.where(differs, directed * (1.0 - w), directed)
    directed /= directed.max(axis=1, keepdims=True)
    a = np.zeros((fit.n_points, fit.n_points))
    np.put_along_axis(a, fit.neighbors, directed, axis=1)
    return a + a.T - a * a.T


@pytest.mark.parametrize(
    ("labels", "label_weight"),
    [
        (np.zeros(120, dtype=np.int64), 0.9),
        (np.full(120, -1), 0.9),
        (np.arange(120) % 3, 0.0),
    ],
    ids=["one class", "no row labelled", "weight 0"],
)
def test_labels_that_say_nothing_give_the_bytes_of_the_map_without_them(
    labels: I64, label_weight: float
) -> None:
    x, _ = _classes(120, 1.0, 0)
    plain = _map(x)
    guided = _map(x, labels, label_weight)
    assert _same_map(plain, guided)
    assert plain.supervision is None
    assert guided.supervision is not None


@pytest.mark.parametrize("label_weight", [0.3, 0.9, 0.999])
def test_the_graph_follows_the_formula_and_every_row_keeps_a_weight_of_one(
    label_weight: float,
) -> None:
    x, y = _classes(150, 1.0, 1)
    y[::5] = -1
    fit = _map(x, y, label_weight)
    dense = fit.graph.dense()
    assert dense == pytest.approx(_formula(fit), abs=1e-12)
    assert dense.max(axis=1) == pytest.approx(1.0, abs=1e-12)


def test_the_neighbours_stay_those_of_the_data_and_unlabelled_pairs_keep_their_weight() -> None:
    x, y = _classes(150, 1.0, 1)
    y[:75] = -1
    plain = _map(x)
    fit = _map(x, y, 0.9)
    for name in ("neighbors", "distances", "rho", "sigma"):
        assert np.array_equal(getattr(fit, name), getattr(plain, name))
    assert np.array_equal(fit.graph.dense()[:75, :75], plain.graph.dense()[:75, :75])
    assert not np.array_equal(fit.graph.dense()[75:, 75:], plain.graph.dense()[75:, 75:])


def test_the_cross_label_share_is_the_weight_between_different_labels() -> None:
    x, y = _classes(150, 1.0, 2)
    y[::4] = -1
    fit = _map(x, y, 0.5)
    directed = np.exp(-np.maximum(0.0, fit.distances - fit.rho[:, None]) / fit.sigma[:, None])
    near = y[fit.neighbors]
    both = (y[:, None] >= 0) & (near >= 0)
    expected = directed[both & (near != y[:, None])].sum() / directed[both].sum()
    assert fit.supervision is not None
    assert fit.supervision.cross_label_share == pytest.approx(expected, rel=1e-12)


def test_the_share_is_none_when_no_two_labelled_rows_are_neighbours() -> None:
    x, _ = _classes(150, 1.0, 2)
    y = np.full(150, -1)
    y[0] = 1
    fit = _map(x, y, 0.5)
    assert fit.supervision is not None
    assert fit.supervision.cross_label_share is None
    file = io.BytesIO()
    fit.to_npz(file)
    file.seek(0)
    loaded = UMAPFit.from_npz(file)
    assert loaded.supervision is not None
    assert loaded.supervision.cross_label_share is None


def test_random_labels_separate_the_map_but_not_the_new_rows() -> None:
    # one Gaussian: the labels have nothing to do with the data
    x, y = _classes(600, 0.0, 4)
    new, new_y = _classes(200, 0.0, 5)
    fit = umap(x, labels=y, label_weight=0.99, seed=1)
    placed = fit.transform(new).embedding
    on_map = label_agreement(fit.embedding, y)
    held_out = label_agreement(fit.embedding, y, query=placed, query_labels=new_y)
    assert on_map > 0.9
    assert held_out < 0.45  # three classes: chance is 1/3


def test_on_separate_classes_new_rows_agree_about_as_often_as_in_the_data() -> None:
    x, y = _classes(600, 3.0, 6)
    new, new_y = _classes(200, 3.0, 7)
    fit = umap(x, labels=y, label_weight=0.9, seed=1)
    placed = fit.transform(new).embedding
    data = label_agreement(x, y, query=new, query_labels=new_y)
    held_out = label_agreement(fit.embedding, y, query=placed, query_labels=new_y)
    assert held_out > data - 0.03


def test_transform_does_not_use_the_labels() -> None:
    x, y = _classes(150, 1.0, 8)
    fit = _map(x, y, 0.9)
    bare = dataclasses.replace(fit, supervision=None)
    new, _ = _classes(20, 1.0, 9)
    assert np.array_equal(fit.transform(new).embedding, bare.transform(new).embedding)


def test_a_map_with_labels_does_not_depend_on_threads_and_keeps_its_own_labels() -> None:
    x, y = _classes(150, 1.0, 10)
    one = _map(x, y, 0.9, threads=1)
    three = _map(x, y, 0.9, threads=3)
    assert _same_map(one, three)
    assert one.supervision is not None
    assert not one.supervision.labels.flags.writeable
    before = one.supervision.labels.copy()
    y[:] = 0
    assert np.array_equal(one.supervision.labels, before)


def test_zero_epochs_with_a_start_and_labels_return_the_start_with_the_new_graph() -> None:
    x, y = _classes(150, 1.0, 11)
    plain = _map(x)
    fit = umap(x, labels=y, label_weight=0.9, init=plain.embedding, n_epochs=0, n_neighbors=8)
    assert np.array_equal(fit.embedding, plain.embedding)
    assert fit.graph.dense() == pytest.approx(_formula(fit), abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"labels": np.zeros(60, dtype=int)}, "needs label_weight"),
        ({"label_weight": 0.5}, "needs labels"),
        ({"labels": np.zeros(60, dtype=int), "label_weight": 1.0}, "below 1"),
        ({"labels": np.zeros(60, dtype=int), "label_weight": -0.1}, "below 1"),
        ({"labels": np.zeros(60, dtype=int), "label_weight": np.nan}, "below 1"),
        ({"labels": np.zeros(59, dtype=int), "label_weight": 0.5}, "shape"),
        ({"labels": np.zeros(60), "label_weight": 0.5}, "integers"),
        ({"labels": np.zeros(60, dtype=bool), "label_weight": 0.5}, "integers"),
        ({"labels": np.full(60, -2), "label_weight": 0.5}, "-1 or above"),
    ],
)
def test_bad_labels_and_weights_are_refused(kwargs: dict[str, object], message: str) -> None:
    x, _ = _classes(60, 1.0, 12)
    with pytest.raises(ValueError, match=message):
        umap(x, n_neighbors=8, n_epochs=2, **kwargs)  # type: ignore[arg-type]


def _saved(fit: UMAPFit, **changes: object) -> io.BytesIO:
    file = io.BytesIO()
    fit.to_npz(file)
    file.seek(0)
    with np.load(file) as store:
        arrays = {name: store[name] for name in store.files}
    arrays.update(changes)  # type: ignore[arg-type]
    out = io.BytesIO()
    np.savez(out, **arrays)  # type: ignore[arg-type]
    out.seek(0)
    return out


def test_a_map_with_labels_survives_a_save_and_a_load() -> None:
    x, y = _classes(150, 1.0, 13)
    fit = _map(x, y, 0.75)
    loaded = UMAPFit.from_npz(_saved(fit))
    assert _same_map(fit, loaded)
    assert fit.supervision is not None
    assert loaded.supervision is not None
    assert np.array_equal(loaded.supervision.labels, fit.supervision.labels)
    assert loaded.supervision.label_weight == 0.75
    assert loaded.supervision.cross_label_share == fit.supervision.cross_label_share


def test_a_saved_file_with_labels_a_fit_cannot_have_is_refused() -> None:
    x, y = _classes(90, 1.0, 14)
    fit = umap(x, labels=y, label_weight=0.5, n_neighbors=8, n_epochs=2)
    for damaged in (
        _saved(fit, labels=y[:-1]),
        _saved(fit, labels=y - 2),
        _saved(fit, label_weight=np.float64(1.0)),
        _saved(fit, cross_label_share=np.float64(1.5)),
    ):
        with pytest.raises(ValueError, match="cannot have"):
            UMAPFit.from_npz(damaged)
    with pytest.raises(ValueError, match="integers"):
        UMAPFit.from_npz(_saved(fit, labels=y.astype(np.float64)))
    with np.load(_saved(fit)) as store:
        rest = {name: store[name] for name in store.files if name != "labels"}
    damaged = io.BytesIO()
    np.savez(damaged, **rest)  # type: ignore[arg-type]
    damaged.seek(0)
    with pytest.raises(ValueError, match="lacks"):
        UMAPFit.from_npz(damaged)


def test_a_file_of_format_version_3_loads_as_a_fit_without_labels() -> None:
    # written by statsx 0.1.81: two groups of 20 points, a given table, 20 epochs, seed 7
    old = UMAPFit.from_npz(Path(__file__).parent / "data" / "umap_fit_v3.npz")
    assert (old.supervision, old.neighbor_source) == (None, "given")
    assert (old.n_points, old.n_neighbors, old.n_epochs, old.seed) == (40, 5, 20, 7)


def _squared(query: F64, x: F64) -> F64:
    return ((query[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)


def _agreement(d: F64, y: I64, own: I64, k: int) -> float:
    """The share of the ``k`` smallest entries of each row of ``d`` with the label of the row."""
    near = np.argsort(d, axis=1, kind="stable")[:, :k]
    return float((y[near] == own[:, None]).mean())


def test_label_agreement_equals_its_definition() -> None:
    x, y = _classes(80, 1.0, 15)
    new, new_y = _classes(30, 1.0, 16)
    others = _squared(x, x)
    np.fill_diagonal(others, np.inf)
    assert label_agreement(x, y, 7) == pytest.approx(_agreement(others, y, y, 7))
    assert label_agreement(x, y, 7, query=new, query_labels=new_y) == pytest.approx(
        _agreement(_squared(new, x), y, new_y, 7)
    )


def test_label_agreement_with_the_cosine_distance_ignores_the_length_of_a_row() -> None:
    x, y = _classes(80, 1.0, 17)
    scaled = x * np.random.default_rng(0).uniform(0.1, 10.0, (80, 1))
    assert label_agreement(scaled, y, 7, metric="cosine") == pytest.approx(
        label_agreement(x, y, 7, metric="cosine")
    )


def test_label_agreement_checks_its_input() -> None:
    x, y = _classes(30, 1.0, 18)
    with pytest.raises(ValueError, match="0 or above"):
        label_agreement(x, np.full(30, -1))
    with pytest.raises(ValueError, match="shape"):
        label_agreement(x, y[:-1])
    with pytest.raises(ValueError, match="go together"):
        label_agreement(x, y, query=x)
    with pytest.raises(ValueError, match="k must be"):
        label_agreement(x, y, 30)
    with pytest.raises(ValueError, match="columns"):
        label_agreement(x, y, query=x[:, :3], query_labels=y)
    assert label_agreement(x, y, 30, query=x, query_labels=y) > 0.0
