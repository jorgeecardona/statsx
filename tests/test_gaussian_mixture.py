"""CL-2: Gaussian mixtures by EM, and the choice of the number of components by BIC."""

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx.clustering import (
    CovarianceKind,
    choose_components,
    gaussian_mixture,
    kmeans,
)

WEIGHTS = np.array([0.5, 0.3, 0.2])
MEANS = np.array([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]])
COVS = np.array(
    [
        [[1.0, 0.6], [0.6, 0.8]],
        [[0.5, 0.0], [0.0, 1.5]],
        [[1.2, -0.5], [-0.5, 0.7]],
    ]
)


def _mixture(n: int, seed: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    labels = rng.choice(3, size=n, p=WEIGHTS).astype(np.int64)
    x = np.empty((n, 2))
    for j in range(3):
        rows = labels == j
        x[rows] = (
            MEANS[j] + rng.standard_normal((int(rows.sum()), 2)) @ np.linalg.cholesky(COVS[j]).T
        )
    return x, labels


def _match(
    fit_means: npt.NDArray[np.float64], truth: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Return, for each true component, the fitted component with the nearest mean."""
    dist = np.linalg.norm(truth[:, None, :] - fit_means[None, :, :], axis=2)
    return np.argmin(dist, axis=1).astype(np.int64)


def test_a_planted_mixture_is_recovered() -> None:
    x, truth = _mixture(3000, 0)
    fit = gaussian_mixture(x, 3, seed=1)
    order = _match(fit.means, MEANS)
    assert sorted(order.tolist()) == [0, 1, 2]
    assert np.allclose(fit.means[order], MEANS, atol=0.2)
    assert np.allclose(fit.weights[order], WEIGHTS, atol=0.04)
    assert np.allclose(fit.covariances[order], COVS, atol=0.25)
    matches: npt.NDArray[np.bool_] = order[truth] == fit.labels
    agree = float(matches.mean())
    assert agree > 0.97
    assert fit.converged
    assert fit.n_degenerate == 0


@pytest.mark.parametrize("kind", ["full", "diag", "spherical"])
def test_the_log_likelihood_never_decreases(kind: CovarianceKind) -> None:
    x, _ = _mixture(800, 1)
    fit = gaussian_mixture(x, 3, covariance=kind, seed=2)
    assert np.all(np.diff(fit.trace) >= -1e-6 * np.abs(fit.trace[:-1]))
    assert fit.trace[-1] == fit.log_likelihood


def test_the_log_likelihood_is_the_density_at_the_data() -> None:
    x, _ = _mixture(500, 2)
    fit = gaussian_mixture(x, 3, seed=3)
    by_hand = np.zeros(len(x))
    for w, m, c in zip(fit.weights, fit.means, fit.covariances, strict=True):
        diff = x - m
        quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(c), diff)
        by_hand += w * np.exp(-0.5 * quad) / (2 * math.pi * math.sqrt(np.linalg.det(c)))
    assert np.allclose(fit.score_samples(x), np.log(by_hand))
    assert fit.log_likelihood == pytest.approx(float(np.log(by_hand).sum()), rel=1e-10)
    assert np.allclose(fit.predict_proba(x), fit.responsibilities, atol=1e-12)
    assert np.allclose(fit.responsibilities.sum(axis=1), 1.0)
    assert np.array_equal(fit.predict(x), fit.labels)


def test_one_component_is_the_sample_mean_and_covariance() -> None:
    x, _ = _mixture(400, 3)
    fit = gaussian_mixture(x, 1, reg=0.0)
    assert np.allclose(fit.means[0], x.mean(axis=0))
    assert np.allclose(fit.covariances[0], np.cov(x.T, bias=True))
    assert fit.weights[0] == 1.0


def test_diag_and_spherical_shapes_and_parameter_counts() -> None:
    x, _ = _mixture(600, 4)
    full = gaussian_mixture(x, 3, covariance="full")
    diag = gaussian_mixture(x, 3, covariance="diag")
    sph = gaussian_mixture(x, 3, covariance="spherical")
    assert full.covariances.shape == (3, 2, 2)
    assert diag.covariances.shape == (3, 2)
    assert sph.covariances.shape == (3,)
    assert (full.n_parameters, diag.n_parameters, sph.n_parameters) == (17, 14, 11)
    assert full.bic == pytest.approx(-2 * full.log_likelihood + 17 * math.log(600))
    assert full.aic == pytest.approx(-2 * full.log_likelihood + 34)
    assert full.log_likelihood > diag.log_likelihood > sph.log_likelihood


def test_round_equal_clusters_give_the_kmeans_partition() -> None:
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]])
    x = np.vstack([c + 0.4 * rng.standard_normal((100, 2)) for c in centres])
    mixture = gaussian_mixture(x, 3, covariance="spherical", seed=1)
    km = kmeans(x, 3, seed=1)
    pairs = set(zip(mixture.labels.tolist(), km.labels.tolist(), strict=True))
    assert len(pairs) == 3


def test_the_fit_does_not_depend_on_the_number_of_threads() -> None:
    x, _ = _mixture(5000, 6)
    one = gaussian_mixture(x, 3, seed=7, threads=1, n_init=2)
    four = gaussian_mixture(x, 3, seed=7, threads=4, n_init=2)
    assert one.log_likelihood == four.log_likelihood
    assert np.array_equal(one.means, four.means)
    assert np.array_equal(one.covariances, four.covariances)
    assert np.array_equal(one.responsibilities, four.responsibilities)


def test_the_same_seed_gives_the_same_fit() -> None:
    x, _ = _mixture(600, 7)
    a = gaussian_mixture(x, 3, seed=4)
    b = gaussian_mixture(x, 3, seed=4)
    assert np.array_equal(a.means, b.means)
    assert a.log_likelihood == b.log_likelihood


def test_the_fit_scores_at_least_the_true_parameters() -> None:
    x, _ = _mixture(1500, 8)
    fit = gaussian_mixture(x, 3, seed=1)
    truth = np.zeros(len(x))
    for w, m, c in zip(WEIGHTS, MEANS, COVS, strict=True):
        diff = x - m
        quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(c), diff)
        truth += w * np.exp(-0.5 * quad) / (2 * math.pi * math.sqrt(np.linalg.det(c)))
    assert fit.log_likelihood >= float(np.log(truth).sum()) - 1e-6


def test_sampling_from_a_fit_and_refitting_returns_the_same_mixture() -> None:
    x, _ = _mixture(2000, 9)
    fit = gaussian_mixture(x, 3, seed=1)
    draws, labels = fit.sample(6000, seed=2)
    assert draws.shape == (6000, 2)
    assert np.allclose(np.bincount(labels, minlength=3) / 6000, fit.weights, atol=0.03)
    again = gaussian_mixture(draws, 3, seed=1)
    order = _match(again.means, fit.means)
    assert np.allclose(again.means[order], fit.means, atol=0.2)
    for kind in ("diag", "spherical"):
        small = gaussian_mixture(x, 3, covariance=kind)  # type: ignore[arg-type]
        out, _ = small.sample(10, seed=1)
        assert out.shape == (10, 2)


def test_bic_finds_three_components_in_gaussian_data() -> None:
    picks = [choose_components(_mixture(900, s)[0], 6, seed=s).k for s in range(8)]
    assert picks.count(3) >= 7


def test_bic_adds_components_to_a_skewed_cluster() -> None:
    """One lognormal cluster is not Gaussian, so BIC says more than one. The docs name this."""
    rng = np.random.default_rng(1)
    x = np.exp(0.9 * rng.standard_normal((2000, 2)))
    assert choose_components(x, 5, seed=1).k > 1


def test_bic_says_one_component_for_one_gaussian_cluster() -> None:
    rng = np.random.default_rng(2)
    picks = [choose_components(rng.standard_normal((600, 3)), 4, seed=s).k for s in range(6)]
    assert picks.count(1) >= 5


def test_aic_asks_for_at_least_as_many_components_as_bic() -> None:
    x, _ = _mixture(700, 10)
    assert choose_components(x, 6, criterion="aic").k >= choose_components(x, 6).k


def test_the_search_reports_each_k_and_the_best_fit() -> None:
    x, _ = _mixture(500, 11)
    search = choose_components(x, 4)
    assert search.ks == (1, 2, 3, 4)
    assert search.bic.shape == (4,)
    assert search.best.k == search.k
    assert search.bic[search.k - 1] == pytest.approx(search.best.bic)
    assert search.covariance_type == "full"


def test_a_k_with_no_supported_fit_is_nan_in_the_search() -> None:
    x, _ = _mixture(12, 12)
    search = choose_components(x, 12, covariance="spherical")
    assert np.isnan(search.bic[-1])
    assert not np.isnan(search.bic[search.k - 1])


def test_a_collapsing_component_is_dropped_not_reported() -> None:
    x = np.vstack([np.zeros((10, 2)), [[5.0, 5.0]], [[9.0, 9.0]]])
    with pytest.raises(ValueError, match="too few points"):
        gaussian_mixture(x, 3)


def test_a_constant_column_does_not_break_the_fit() -> None:
    x, _ = _mixture(400, 13)
    x = np.column_stack([x, np.ones(400)])
    fit = gaussian_mixture(x, 3, seed=1)
    assert np.isfinite(fit.log_likelihood)


def test_uncertainty_is_high_between_overlapping_components() -> None:
    rng = np.random.default_rng(3)
    x = np.vstack(
        [rng.standard_normal((400, 2)), rng.standard_normal((400, 2)) + np.array([2.0, 0.0])]
    )
    fit = gaussian_mixture(x, 2, seed=1)
    assert fit.uncertainty.mean() > 0.1
    apart = np.vstack(
        [rng.standard_normal((400, 2)), rng.standard_normal((400, 2)) + np.array([20.0, 0.0])]
    )
    assert gaussian_mixture(apart, 2, seed=1).uncertainty.mean() < 1e-3
    assert np.isclose(fit.sizes.sum(), 800.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be"),
        ({"k": 1000}, "k must be"),
        ({"k": 2, "covariance": "tied"}, "covariance"),
        ({"k": 2, "n_init": 0}, "n_init"),
        ({"k": 2, "tol": -1.0}, "tol"),
        ({"k": 2, "threads": 0}, "threads"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], message: str) -> None:
    x, _ = _mixture(50, 14)
    with pytest.raises(ValueError, match=message):
        gaussian_mixture(x, **kwargs)  # type: ignore[arg-type]


def test_bad_data_and_bad_new_points_are_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        gaussian_mixture(np.array([[0.0, np.nan], [1.0, 2.0], [3.0, 4.0]]), 1)
    x, _ = _mixture(100, 15)
    fit = gaussian_mixture(x, 2)
    with pytest.raises(ValueError, match="columns"):
        fit.predict(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="n must"):
        fit.sample(0)
    with pytest.raises(ValueError, match="criterion"):
        choose_components(x, 3, criterion="hqic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_max"):
        choose_components(x, 0)


def _stretched() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Two long flat groups one above the other, and a small round one far away."""
    rng = np.random.default_rng(1)
    long_axis = np.array([[3.0, 0.0], [0.0, 0.3]])
    a = rng.standard_normal((400, 2)) @ long_axis
    b = rng.standard_normal((400, 2)) @ long_axis + np.array([0.0, 2.0])
    c = rng.standard_normal((200, 2)) * 0.5 + np.array([7.0, 7.0])
    return np.vstack([a, b, c]), np.repeat([0, 1, 2], [400, 400, 200]).astype(np.int64)


def _best_agreement(labels: npt.NDArray[np.int64], truth: npt.NDArray[np.int64]) -> float:
    return max(
        float((np.array(p)[labels] == truth).mean()) for p in itertools.permutations(range(3))
    )


def test_more_starts_reach_a_higher_likelihood_where_one_start_is_stuck() -> None:
    """EM keeps the optimum nearest its start. A k-means start cuts the long groups across."""
    x, truth = _stretched()
    one = gaussian_mixture(x, 3, seed=1, n_init=1)
    many = gaussian_mixture(x, 3, seed=1, n_init=5)
    assert many.log_likelihood > one.log_likelihood + 100.0
    assert _best_agreement(many.labels, truth) > 0.97
    assert _best_agreement(kmeans(x, 3, seed=1).labels, truth) < 0.8
    assert _best_agreement(one.labels, truth) < 0.8


def test_ten_starts_find_the_best_optimum_on_every_seed_and_one_start_does_not() -> None:
    x, _ = _stretched()
    best = gaussian_mixture(x, 3, seed=1, n_init=30).log_likelihood
    ten = [gaussian_mixture(x, 3, seed=s).log_likelihood for s in range(10)]
    assert all(v > best - 1.0 for v in ten)
    one = [gaussian_mixture(x, 3, seed=s, n_init=1).log_likelihood for s in range(5)]
    assert all(v < best - 400.0 for v in one)
