"""PCA against its definitions, and what it does for clustering wide data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import elbow_k, kmeans
from statsx.embedding import pca

F64 = npt.NDArray[np.float64]


def _data(n: int, d: int, seed: int = 0) -> F64:
    """Correlated columns with a spread of variances, away from the origin."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(d, d)) * np.linspace(3.0, 0.1, d)
    out: F64 = rng.normal(size=(n, d)) @ mix + rng.uniform(-50.0, 50.0, size=d)
    return out


def _svd(x: F64) -> tuple[F64, F64]:
    """Singular values and right singular vectors of the centred data."""
    _, singular, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    return singular, vt


@pytest.mark.parametrize(("n", "d"), [(500, 12), (30, 80), (10_000, 7)])
def test_the_components_equal_the_singular_vectors_of_the_centred_data(n: int, d: int) -> None:
    # (10000, 7) spans several blocks of rows; (30, 80) has more columns than rows
    x = _data(n, d)
    k = 5
    fit = pca(x, k)
    singular, vt = _svd(x)
    assert fit.n_components == k
    for i in range(k):
        assert abs(float(fit.components[i] @ vt[i])) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(fit.explained_variance, singular[:k] ** 2 / (n - 1), rtol=1e-9)
    assert np.allclose(fit.components @ fit.components.T, np.eye(k), rtol=0.0, atol=1e-12)
    assert np.allclose(fit.mean, x.mean(axis=0), rtol=1e-12)
    assert np.allclose(fit.scores, (x - x.mean(axis=0)) @ fit.components.T, rtol=0.0, atol=1e-9)


def test_the_variance_and_the_reconstruction_error_follow_the_definitions() -> None:
    x = _data(400, 9, seed=1)
    k = 4
    fit = pca(x, k)
    assert fit.total_variance == pytest.approx(float(np.var(x, axis=0, ddof=1).sum()), rel=1e-12)
    assert np.allclose(fit.explained_variance, np.var(fit.scores, axis=0, ddof=1), rtol=1e-9)
    assert np.all(np.diff(fit.explained_variance) <= 0)
    assert fit.explained_variance_ratio.sum() < 1.0
    # Eckart-Young: what k components lose is the sum of the other squared singular values
    back = fit.scores @ fit.components + fit.mean
    singular, _ = _svd(x)
    assert float(((x - back) ** 2).sum()) == pytest.approx(
        float((singular[k:] ** 2).sum()), rel=1e-9
    )
    # all components keep everything
    full = pca(x, 9)
    assert np.allclose(full.scores @ full.components + full.mean, x, rtol=0.0, atol=1e-9)
    assert full.explained_variance_ratio.sum() == pytest.approx(1.0, rel=1e-12)


def test_the_sign_of_each_component_is_fixed_and_new_rows_project_the_same_way() -> None:
    x = _data(300, 10, seed=2)
    fit = pca(x, 6)
    for row in fit.components:
        assert row[np.argmax(np.abs(row))] > 0
    assert np.array_equal(fit.transform(x), fit.scores)
    assert fit.transform(x[:1]).shape == (1, 6)
    with pytest.raises(ValueError, match="columns"):
        fit.transform(x[:, :9])
    with pytest.raises(ValueError, match="finite"):
        fit.transform(np.full((2, 10), np.inf))


def _wide_groups(n: int = 2000, dims: int = 256) -> tuple[F64, npt.NDArray[np.int64]]:
    """The data of statsx issue 17, smaller: 8 directions and noise much longer than them."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(8, dims))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    members = rng.integers(0, 8, size=n)
    x = centres[members] + 0.2 * rng.normal(size=(n, dims))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, members.astype(np.int64)


def test_principal_components_first_let_the_elbow_find_the_groups_of_wide_data() -> None:
    x, members = _wide_groups()
    reduced = pca(x, 8).scores
    result = elbow_k(reduced, 14)
    assert result.k == 8
    assert result.is_knee
    fit = kmeans(reduced, 8)
    table = np.zeros((8, 8), dtype=np.int64)
    np.add.at(table, (fit.labels, members), 1)
    assert table.max(axis=1).sum() > 0.9 * len(x)


def test_principal_components_lose_groups_that_differ_in_a_small_direction() -> None:
    rng = np.random.default_rng(3)
    group = np.repeat([0, 1], 200)
    # the groups differ in the third column only, which has far less variance than the others
    x = rng.normal(size=(400, 3)) * [20.0, 10.0, 0.1]
    x[:, 2] += group * 1.0
    kept = kmeans(pca(x, 2).scores, 2).labels
    agree = max(float((kept == group).mean()), float((kept != group).mean()))
    assert agree < 0.65
    all_three = kmeans(x[:, 2:], 2).labels
    assert max(float((all_three == group).mean()), float((all_three != group).mean())) > 0.99


def test_pca_checks_its_input() -> None:
    x = _data(10, 20)
    for bad in (0, 10, 21):
        with pytest.raises(ValueError, match="n_components"):
            pca(x, bad)
    assert pca(x, 9).n_components == 9
    with pytest.raises(ValueError, match="all rows are equal"):
        pca(np.ones((5, 3)), 1)
    with pytest.raises(ValueError, match="finite"):
        pca(np.full((5, 3), np.nan), 1)
