import numpy as np
import pytest

from statsx.agreement import Level, cohen_kappa, fleiss_kappa, krippendorff_alpha
from statsx.result import Estimate

NAN = float("nan")

# Ten subjects, 14 raters, 5 categories: the worked example of Fleiss (1971), kappa 0.210.
FLEISS = np.array(
    [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
)  # fmt: skip

# Four raters, 15 units, missing ratings as nan. The four values are from the `krippendorff`
# package (checked once, outside the test suite).
ALPHA_DATA = np.array(
    [
        [NAN, NAN, NAN, NAN, NAN, 3, 4, 1, 2, 1, 1, 3, 3, NAN, 3],
        [1, NAN, 2, 1, 3, 3, 4, 3, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
        [NAN, NAN, 2, 1, 3, 4, 4, NAN, 2, 1, 1, 3, 3, NAN, 4],
        [1, NAN, 2, 1, 3, 3, 4, 2, 2, 1, 1, 3, 3, NAN, NAN],
    ]
)  # fmt: skip
ALPHA_REFERENCE = {
    "nominal": 0.7572992700729927,
    "ordinal": 0.8922583826429981,
    "interval": 0.8916761687571265,
    "ratio": 0.8952406333459721,
}


def test_cohens_kappa_matches_the_textbook_example() -> None:
    # 50 items: both yes 20, A yes and B no 5, A no and B yes 10, both no 15. po = 0.7, pe = 0.5.
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    result = cohen_kappa(a, b, n_resamples=1000, seed=1)
    assert isinstance(result, Estimate)
    assert result.estimate == pytest.approx(0.4, abs=1e-12)
    assert result.n == 50
    assert result.interval.lower < 0.4 < result.interval.upper
    assert "Cohen" in result.assumption


def test_weighted_kappa_gives_partial_credit_to_near_misses() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 6, 200)
    near = np.clip(truth + rng.choice([-1, 0, 1], 200, p=[0.15, 0.7, 0.15]), 1, 5)
    plain = cohen_kappa(truth, near, n_resamples=200).estimate
    linear = cohen_kappa(truth, near, weights="linear", n_resamples=200).estimate
    quadratic = cohen_kappa(truth, near, weights="quadratic", n_resamples=200).estimate
    assert plain < linear < quadratic <= 1.0
    # Brute force from the definition of weighted kappa.
    k = 5
    table = np.zeros((k, k))
    for t, n in zip(truth, near, strict=True):
        table[t - 1, n - 1] += 1
    p = table / table.sum()
    w = 1.0 - ((np.arange(k)[:, None] - np.arange(k)[None, :]) / (k - 1)) ** 2
    expected = np.sum(w * np.outer(p.sum(1), p.sum(0)))
    assert quadratic == pytest.approx((np.sum(w * p) - expected) / (1 - expected), rel=1e-12)


def test_fleiss_kappa_matches_the_worked_example() -> None:
    result = fleiss_kappa(FLEISS, n_resamples=1000, seed=1)
    assert result.estimate == pytest.approx(0.2099, abs=5e-4)
    assert result.n == 10
    assert "14 raters" in result.assumption
    assert result.interval.lower < result.estimate < result.interval.upper


def test_fleiss_kappa_agrees_with_cohen_for_two_raters() -> None:
    # With 2 raters Fleiss' kappa is Cohen's kappa with pooled category shares (Scott's pi): equal
    # when both raters use the categories equally often.
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    b = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    counts = np.zeros((a.size, 2))
    counts[np.arange(a.size), a] += 1
    counts[np.arange(a.size), b] += 1
    assert fleiss_kappa(counts, n_resamples=100).estimate == pytest.approx(
        cohen_kappa(a, b, n_resamples=100).estimate, abs=1e-12
    )


@pytest.mark.parametrize("level_name", ["nominal", "ordinal", "interval", "ratio"])
def test_alpha_matches_the_reference_values_with_missing_ratings(level_name: Level) -> None:
    result = krippendorff_alpha(
        ALPHA_DATA, level_of_measurement=level_name, n_resamples=300, seed=1
    )
    assert result.estimate == pytest.approx(ALPHA_REFERENCE[level_name], abs=1e-12)
    assert result.n == 13  # two of the 15 units have no rating at all
    assert "missing ratings allowed" in result.assumption


def test_nominal_alpha_follows_its_definition_over_pairs_of_values() -> None:
    # alpha = 1 - D_o / D_e, with D_o from pairs inside units and D_e from all pairs of values.
    units = ALPHA_DATA.T
    inside, n = 0.0, 0.0
    pooled: list[float] = []
    for row in units:
        r = row[~np.isnan(row)]
        if r.size < 2:
            continue
        pooled.extend(r.tolist())
        n += r.size
        inside += sum(
            2.0 * (r[i] != r[j]) / (r.size - 1.0) for i in range(r.size) for j in range(i)
        )
    values = np.array(pooled)
    different = sum(2.0 * (values[i] != values[j]) for i in range(values.size) for j in range(i))
    alpha = 1.0 - (inside / n) / (different / (n * (n - 1.0)))
    assert krippendorff_alpha(ALPHA_DATA, n_resamples=100).estimate == pytest.approx(
        alpha, rel=1e-12
    )


def test_perfect_agreement_gives_one_and_random_ratings_give_about_zero() -> None:
    rng = np.random.default_rng(3)
    same = np.tile(rng.integers(1, 5, 40), (3, 1)).astype(float)
    same[0, 0] = 4.0
    same[1, 0] = 4.0
    same[2, 0] = 4.0
    assert krippendorff_alpha(same, n_resamples=100).estimate == pytest.approx(1.0)
    noise = rng.integers(1, 5, (3, 400)).astype(float)
    assert abs(krippendorff_alpha(noise, n_resamples=100).estimate) < 0.1
    assert (
        abs(
            fleiss_kappa(
                np.apply_along_axis(
                    lambda r: np.bincount(r, minlength=4), 0, rng.integers(0, 4, (3, 400))
                ).T,
                n_resamples=100,
            ).estimate
        )
        < 0.1
    )
    assert (
        abs(cohen_kappa(rng.integers(0, 3, 500), rng.integers(0, 3, 500), n_resamples=100).estimate)
        < 0.1
    )


def test_the_interval_narrows_with_more_items() -> None:
    rng = np.random.default_rng(4)
    widths: list[float] = []
    for n in (40, 400):
        truth = rng.integers(0, 3, n)
        other = np.where(rng.random(n) < 0.8, truth, rng.integers(0, 3, n))
        interval = cohen_kappa(truth, other, n_resamples=500, seed=2).interval
        widths.append(interval.upper - interval.lower)
    assert widths[1] < widths[0]


@pytest.mark.parametrize(
    "call",
    [
        lambda: cohen_kappa([1, 2], [1]),
        lambda: cohen_kappa([1, 1, 1], [1, 1, 1]),
        lambda: cohen_kappa(["a", "b", "a"], ["a", "b", "b"], weights="linear"),
        lambda: fleiss_kappa([[3, 0], [0, 3]][:1]),
        lambda: fleiss_kappa([[2, 1], [1, 1]]),
        lambda: fleiss_kappa([[3, 0], [3, 0]]),
        lambda: fleiss_kappa([[1.5, 1.5], [2, 1]]),
        lambda: krippendorff_alpha([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        lambda: krippendorff_alpha([[1.0, NAN], [NAN, 2.0]]),
        lambda: krippendorff_alpha([1.0, 2.0]),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError, match=r"undefined|same length|numeric|shape|raters|whole|categories"
    ):
        call()  # type: ignore[operator]
