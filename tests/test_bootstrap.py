import numpy as np
import numpy.typing as npt
import pytest

from statsx.bootstrap import IID, Block, Clustered, Stratified, bootstrap


def test_the_result_records_the_resample_unit_for_each_of_the_four_units() -> None:
    data = np.arange(20, dtype=float)
    labels = [i // 5 for i in range(20)]
    for unit in (IID(), Stratified(labels), Clustered(labels), Block(4)):
        result = bootstrap(data, np.mean, unit, n_resamples=200, seed=1)
        assert result.unit is unit
        assert unit.describe() in result.estimate.assumption
        assert result.estimate.interval.contains(result.estimate.estimate)


def test_iid_percentile_interval_covers_the_mean_at_about_the_nominal_rate() -> None:
    rng = np.random.default_rng(0)
    covered = 0
    trials = 200
    for t in range(trials):
        sample = rng.normal(loc=3.0, scale=1.0, size=50)
        result = bootstrap(sample, np.mean, IID(), n_resamples=400, level=0.9, seed=t)
        covered += result.estimate.interval.contains(3.0)
    # nominal 0.90; the percentile interval undercovers a little at n = 50
    assert 0.82 <= covered / trials <= 0.96


def test_stratified_keeps_stratum_sizes() -> None:
    data = np.array([0.0] * 8 + [100.0] * 2)
    labels = [0] * 8 + [1] * 2
    result = bootstrap(data, np.mean, Stratified(labels), n_resamples=300, seed=2)
    # each replicate has exactly 8 zeros and 2 hundreds: the mean never moves
    assert np.all(result.replicates == 20.0)


def test_clustered_keeps_clusters_together() -> None:
    data = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    labels = [0, 0, 0, 1, 1, 1]
    result = bootstrap(data, len, Clustered(labels), n_resamples=100, seed=3)
    assert np.all(result.replicates == 6)
    means = bootstrap(data, np.mean, Clustered(labels), n_resamples=200, seed=3).replicates
    assert set(np.unique(means)) <= {1.0, 3.0, 5.0}


def test_block_replicates_are_made_of_consecutive_runs() -> None:
    data = np.arange(12, dtype=float)

    def max_run_break(x: npt.NDArray[np.float64]) -> float:
        # number of places where the next value is not previous + 1
        breaks: npt.NDArray[np.bool_] = np.diff(x) != 1.0
        return float(np.count_nonzero(breaks))

    result = bootstrap(data, max_run_break, Block(4), n_resamples=200, seed=4)
    assert np.all(result.replicates <= 2)  # 3 blocks of 4: at most 2 breaks


def test_same_seed_same_result() -> None:
    data = np.arange(30, dtype=float)
    a = bootstrap(data, np.median, IID(), seed=9)
    b = bootstrap(data, np.median, IID(), seed=9)
    assert a.estimate == b.estimate


def test_refuses_bad_block_length_and_label_mismatch() -> None:
    with pytest.raises(ValueError, match="block_length"):
        bootstrap(np.arange(5.0), np.mean, Block(6))
    with pytest.raises(ValueError, match="labels"):
        bootstrap(np.arange(5.0), np.mean, Stratified([0, 1]))


def _skewed(seed: int, n: int = 40) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).lognormal(0.0, 1.0, n)


def test_the_basic_interval_reflects_the_percentile_interval_around_the_estimate() -> None:
    data = _skewed(1)
    pct = bootstrap(data, np.mean, IID(), n_resamples=1000, level=0.9, seed=3)
    basic = bootstrap(data, np.mean, IID(), n_resamples=1000, level=0.9, method="basic", seed=3)
    theta = pct.estimate.estimate
    assert basic.method == "basic"
    assert basic.estimate.interval.lower == pytest.approx(2 * theta - pct.estimate.interval.upper)
    assert basic.estimate.interval.upper == pytest.approx(2 * theta - pct.estimate.interval.lower)
    assert "basic interval" in basic.estimate.assumption


def test_the_bca_interval_follows_its_definition() -> None:
    from statistics import NormalDist

    data = _skewed(2)
    result = bootstrap(data, np.mean, IID(), n_resamples=2000, level=0.9, method="bca", seed=4)
    theta, reps = result.estimate.estimate, result.replicates
    # Bias correction from the share of replicates below the estimate, acceleration from the
    # leave-one-out estimates (Efron 1987).
    z0 = NormalDist().inv_cdf(np.mean(reps < theta))
    jack = np.array([np.delete(data, i).mean() for i in range(data.size)])
    d = jack.mean() - jack
    a = np.sum(d**3) / (6.0 * np.sum(d**2) ** 1.5)
    levels: list[float] = [
        NormalDist().cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        for z in (NormalDist().inv_cdf(0.05), NormalDist().inv_cdf(0.95))
    ]
    want = np.quantile(reps, levels)
    assert result.method == "bca"
    assert result.estimate.interval.lower == pytest.approx(want[0], rel=1e-9)
    assert result.estimate.interval.upper == pytest.approx(want[1], rel=1e-9)
    # A right-skewed statistic has a positive acceleration, so the upper end moves out.
    pct = bootstrap(data, np.mean, IID(), n_resamples=2000, level=0.9, seed=4)
    assert result.estimate.interval.upper > pct.estimate.interval.upper


def test_bca_with_clusters_drops_one_cluster_at_a_time() -> None:
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(12), 5)
    data = rng.normal(size=12)[labels] + rng.normal(scale=0.3, size=60)
    unit = Clustered(labels.tolist())
    result = bootstrap(data, np.mean, unit, n_resamples=800, method="bca", seed=1)
    assert (
        result.estimate.interval.lower < result.estimate.estimate < result.estimate.interval.upper
    )
    assert "resampling of 12 clusters" in result.estimate.assumption


def test_bca_is_refused_for_units_the_jackknife_cannot_serve() -> None:
    data = np.arange(20, dtype=float)
    for unit in (Block(4), Stratified([0] * 10 + [1] * 10)):
        with pytest.raises(ValueError, match="bca"):
            bootstrap(data, np.mean, unit, method="bca")
    with pytest.raises(ValueError, match="method"):
        bootstrap(data, np.mean, IID(), method="studentized")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        bootstrap(data, np.mean, IID(), level=1.0)
