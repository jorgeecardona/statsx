"""float32 points in k-means: read where they are, with the answer of their float64 copy."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import Metric, assign_clusters, kmeans

F32 = npt.NDArray[np.float32]


def _groups(n: int, columns: int, seed: int) -> F32:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((6, columns)) * 3.0
    x = centres[rng.integers(0, 6, n)] + rng.standard_normal((n, columns))
    return x.astype(np.float32)


@pytest.mark.parametrize(
    ("columns", "k"),
    [(5, 3), (20, 30), (300, 20)],
    ids=["every distance", "float64 product", "float32 product"],
)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_float32_input_gives_the_bytes_of_its_float64_copy(
    columns: int, k: int, metric: Metric
) -> None:
    single = _groups(3000, columns, columns)
    double = single.astype(np.float64)
    a = kmeans(single, k, metric=metric, n_init=2, seed=4)
    b = kmeans(double, k, metric=metric, n_init=2, seed=4)
    for name in ("centroids", "labels", "dispersion"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True)
    assert (a.inertia, a.n_iter, a.converged) == (b.inertia, b.n_iter, b.converged)
    assert a.centroids.dtype == np.float64
    new_single = _groups(500, columns, columns + 1)
    new_double = new_single.astype(np.float64)
    assert np.array_equal(a.predict(new_single), a.predict(new_double))
    assert np.array_equal(a.distances(new_single)[1], a.distances(new_double)[1])
    first = assign_clusters(new_single, a.centroids, metric=metric)
    second = assign_clusters(new_double, a.centroids, metric=metric)
    assert np.array_equal(first.nearest, second.nearest)
    assert np.array_equal(first.distances, second.distances)


def test_float32_input_in_any_layout_is_read() -> None:
    single = _groups(200, 6, 0)
    plain = kmeans(single, 3)
    assert np.array_equal(kmeans(np.asfortranarray(single), 3).labels, plain.labels)
    assert np.array_equal(kmeans(single.astype(">f4"), 3).labels, plain.labels)
    assert np.array_equal(plain.predict(single[::2]), plain.labels[::2])


def test_float32_input_that_is_not_finite_is_refused() -> None:
    single = _groups(10_000, 3, 0)
    single[9_000, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        kmeans(single, 3)


_PEAK = """
import numpy as np
from statsx.clustering import assign_clusters, kmeans

def peak() -> int:
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmHWM"))

rng = np.random.default_rng(0)
x = rng.standard_normal((200_000, 256), dtype=np.float32)
before = peak()
fit = kmeans(x, 10, n_init=1, max_iter=3, threads=2)
fit.predict(x, threads=2)
assign_clusters(x, fit.centroids, threads=2)
print((peak() - before) / x.nbytes)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="reads the peak memory from /proc")
def test_float32_input_is_not_copied() -> None:
    done = subprocess.run(  # noqa: S603 (this interpreter and a program written above)
        [sys.executable, "-c", _PEAK], capture_output=True, text=True, check=True
    )
    # a float64 copy of the data would add twice its size
    assert float(done.stdout) < 0.5
