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
        assert result.estimate.assumption == unit.describe()
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
