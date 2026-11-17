import math

import numpy as np
import pytest

from statsx.distributions import (
    ExponentialStats,
    LognormalFit,
    Moments,
    NormalFit,
    ShiftedExponentialFit,
    fit_lognormal,
    fit_normal,
    fit_shifted_exponential,
    log_moments,
)


def test_a_normal_fit_from_statistics_equals_the_fit_from_the_rows() -> None:
    x = np.random.default_rng(0).normal(3.0, 2.0, 500)
    fit = fit_normal(Moments.from_values(x))
    assert fit.mean == pytest.approx(x.mean(), rel=1e-12)
    assert fit.sd == pytest.approx(x.std(ddof=0), rel=1e-12)
    corrected = fit_normal(Moments.from_values(x), bias_corrected=True)
    assert corrected.sd == pytest.approx(x.std(ddof=1), rel=1e-12)
    from_table = fit_normal(Moments.from_summary(500, float(x.mean()), float(x.var(ddof=1))))
    assert from_table.sd == pytest.approx(x.std(ddof=0), rel=1e-9)
    assert fit.n == 500


def test_merging_shards_gives_the_statistics_of_the_pooled_rows() -> None:
    rng = np.random.default_rng(1)
    shards = [rng.normal(5.0, 3.0, n) for n in (10, 400, 37)]
    merged = Moments.from_values(shards[0])
    for s in shards[1:]:
        merged = merged.merge(Moments.from_values(s))
    whole = Moments.from_values(np.concatenate(shards))
    assert merged.n == whole.n
    assert merged.mean == pytest.approx(whole.mean, rel=1e-12)
    assert merged.m2 == pytest.approx(whole.m2, rel=1e-12)
    e = ExponentialStats.from_values(shards[0] - 100).merge(
        ExponentialStats.from_values(shards[1] - 100)
    )
    w = ExponentialStats.from_values(np.concatenate(shards[:2]) - 100)
    assert (e.n, e.minimum) == (w.n, w.minimum)
    assert e.mean == pytest.approx(w.mean, rel=1e-12)


def test_the_lognormal_fit_is_the_normal_fit_of_the_logarithms() -> None:
    x = np.random.default_rng(2).lognormal(1.0, 0.5, 2000)
    fit = fit_lognormal(log_moments(x))
    assert fit.mu == pytest.approx(1.0, abs=0.03)
    assert fit.sigma == pytest.approx(0.5, abs=0.03)
    assert fit.mu == pytest.approx(float(np.log(x).mean()), rel=1e-12)
    assert float(fit.cdf(math.exp(fit.mu))) == pytest.approx(0.5, abs=1e-12)
    assert float(fit.cdf(-1.0)) == 0.0


def test_the_shifted_exponential_fit_has_the_stated_closed_form() -> None:
    x = np.array([2.0, 2.5, 3.0, 4.5, 8.0])
    stats = ExponentialStats.from_values(x)
    mle = fit_shifted_exponential(stats)
    total = float((x - x.min()).sum())
    assert mle.shift == 2.0
    assert mle.rate == pytest.approx(5.0 / total)
    ub = fit_shifted_exponential(stats, bias_corrected=True)
    assert ub.shift == pytest.approx(2.0 - total / 20.0)
    assert ub.rate == pytest.approx(3.0 / total)


def test_the_bias_corrected_shifted_exponential_is_unbiased_and_the_mle_is_not() -> None:
    rng = np.random.default_rng(3)
    true_shift, true_rate, n, runs = 10.0, 0.5, 8, 40_000
    x = true_shift + rng.exponential(1.0 / true_rate, (runs, n))
    mins, means = x.min(axis=1), x.mean(axis=1)
    mle = [
        fit_shifted_exponential(ExponentialStats(n, a, b))
        for a, b in zip(mins[:4000], means[:4000], strict=True)
    ]
    mle_shift = np.mean([f.shift for f in mle])
    # E[min] = shift + 1 / (n rate) = 10.25: the maximum likelihood shift is too high.
    assert mle_shift == pytest.approx(10.25, abs=0.02)
    fits = [
        fit_shifted_exponential(ExponentialStats(n, a, b), bias_corrected=True)
        for a, b in zip(mins, means, strict=True)
    ]
    assert np.mean([f.shift for f in fits]) == pytest.approx(true_shift, abs=0.02)
    assert np.mean([f.rate for f in fits]) == pytest.approx(true_rate, rel=0.02)


def test_cdf_and_ppf_are_inverses_and_samples_follow_the_fit() -> None:
    fits = [
        NormalFit(2.0, 3.0, 10),
        LognormalFit(0.5, 0.4, 10),
        ShiftedExponentialFit(1.0, 2.0, 10),
    ]
    p = np.array([0.01, 0.2, 0.5, 0.9, 0.999])
    for fit in fits:
        np.testing.assert_allclose(fit.cdf(fit.ppf(p)), p, atol=1e-10)
        draws = fit.sample(20000, seed=4)
        np.testing.assert_array_equal(draws, fit.sample(20000, seed=4))
        assert float(np.mean(draws <= fit.ppf(0.9))) == pytest.approx(0.9, abs=0.01)
    assert float(fits[0].cdf(2.0)) == pytest.approx(0.5)
    assert float(fits[2].cdf(0.5)) == 0.0
    assert float(fits[2].cdf(1.5)) == pytest.approx(1.0 - math.exp(-1.0))


@pytest.mark.parametrize(
    "call",
    [
        lambda: Moments(0, 0.0, 0.0),
        lambda: Moments(5, 0.0, -1.0),
        lambda: Moments.from_values([]),
        lambda: Moments.from_summary(5, 0.0, 1.0, ddof=2),
        lambda: ExponentialStats(5, 3.0, 2.0),
        lambda: log_moments([1.0, -2.0]),
        lambda: fit_normal(Moments(1, 0.0, 0.0)),
        lambda: fit_normal(Moments(5, 1.0, 0.0)),
        lambda: fit_shifted_exponential(ExponentialStats(2, 1.0, 2.0), bias_corrected=True),
        lambda: fit_shifted_exponential(ExponentialStats(5, 1.0, 1.0)),
        lambda: NormalFit(0.0, 1.0, 5).ppf(1.0),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"n must|m2|values|ddof|below the minimum|positive|fit needs|probabilities",
    ):
        call()  # type: ignore[operator]
