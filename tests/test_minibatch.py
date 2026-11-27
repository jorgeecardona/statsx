"""Mini-batch k-means (Sculley 2010), Algorithm 1."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import kmeans, minibatch_kmeans

F64 = npt.NDArray[np.float64]


def _groups(n: int, columns: int, seed: int) -> F64:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((8, columns)) * 3.0
    return centres[rng.integers(0, 8, n)] + rng.standard_normal((n, columns))


@pytest.mark.parametrize("columns", [3, 300])
def test_a_minibatch_fit_labels_each_row_by_its_nearest_centre(columns: int) -> None:
    x = _groups(4000, columns, 1)
    fit = minibatch_kmeans(x, 8, batch_size=500, n_batches=20)
    squared = np.stack([((x - c) ** 2).sum(axis=1) for c in fit.centroids], axis=1)
    assert np.array_equal(fit.labels, squared.argmin(axis=1))
    assert np.array_equal(fit.predict(x), fit.labels)
    assert fit.inertia == pytest.approx(squared.min(axis=1).sum(), rel=1e-12)
    distance = np.sqrt(squared.min(axis=1))
    expected = [distance[fit.labels == c].mean() for c in range(8)]
    assert fit.dispersion == pytest.approx(expected, rel=1e-9)
    assert (fit.method, fit.batch_size, fit.n_iter, fit.converged, fit.n_init) == (
        "minibatch",
        500,
        20,
        False,
        1,
    )
    assert fit.metric == "euclidean"


def test_a_lloyd_fit_says_so() -> None:
    fit = kmeans(_groups(200, 4, 2), 3)
    assert (fit.method, fit.batch_size) == ("lloyd", None)


def test_one_batch_of_one_centre_is_the_mean_of_the_rows_drawn() -> None:
    # with one centre the step 1 / count makes it the mean of every row it has had, the seed
    # row excluded: the paper's counts start at 0
    x = np.arange(10.0)[:, None] * np.ones((1, 2))
    fit = minibatch_kmeans(x, 1, batch_size=100_000, n_batches=1, seed=5)
    assert fit.centroids[0] == pytest.approx(x.mean(axis=0), abs=0.05)


def test_the_inertia_is_near_that_of_lloyd_on_separate_groups() -> None:
    x = _groups(20_000, 16, 3)
    full = kmeans(x, 8, n_init=3)
    part = minibatch_kmeans(x, 8, batch_size=1000, n_batches=100)
    assert part.inertia < 1.02 * full.inertia


def test_minibatch_gives_the_same_bytes_for_any_threads_and_for_float32() -> None:
    single = _groups(6000, 40, 4).astype(np.float32)
    runs = [
        minibatch_kmeans(data, 12, batch_size=3000, n_batches=10, seed=7, threads=threads)
        for data in (single, single.astype(np.float64))
        for threads in (1, 3)
    ]
    for other in runs[1:]:
        assert np.array_equal(other.centroids, runs[0].centroids)
        assert np.array_equal(other.labels, runs[0].labels)
        assert other.inertia == runs[0].inertia
    again = minibatch_kmeans(single, 12, batch_size=3000, n_batches=10, seed=8)
    assert not np.array_equal(again.centroids, runs[0].centroids)


def test_a_batch_may_be_larger_than_the_data() -> None:
    x = _groups(30, 3, 5)
    assert minibatch_kmeans(x, 30, batch_size=100, n_batches=2).k == 30


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be"),
        ({"k": 51}, "k must be"),
        ({"k": 2, "batch_size": 0}, "at least 1"),
        ({"k": 2, "n_batches": 0}, "at least 1"),
    ],
)
def test_minibatch_refusals(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        minibatch_kmeans(_groups(50, 3, 6), **kwargs)  # type: ignore[arg-type]
