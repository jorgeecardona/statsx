from statistics import NormalDist

import numpy as np
import pytest

from statsx.multiple import Method, adjust

# The 15 p-values of Benjamini and Hochberg (1995), section 4: BH at 0.05 rejects the first 4,
# Holm rejects the first 3.
BH_1995 = [
    0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
]  # fmt: skip


def test_holm_and_benjamini_hochberg_reject_what_the_1995_example_says() -> None:
    p = np.array(BH_1995)
    holm = adjust(p, method="holm", alpha=0.05)
    bh = adjust(p, method="bh", alpha=0.05)
    assert holm.n_rejected == 3
    assert bh.n_rejected == 4
    assert list(np.flatnonzero(bh.rejected)) == [0, 1, 2, 3]
    assert bh.method == "bh" and holm.pi0 is None


def _brute_holm(p: np.ndarray) -> np.ndarray:
    m = p.size
    # Holm (1979) step-down: the adjusted value of a test is the largest, over all tests at
    # least as small, of (m - number of smaller p-values) * p, capped at 1.
    terms = np.array([(m - np.sum(p < p[j])) * p[j] for j in range(m)], dtype=np.float64)
    return np.array([min(1.0, float(terms[p <= p[i]].max())) for i in range(m)], dtype=np.float64)


def _brute_bh(p: np.ndarray, scale: float = 1.0) -> np.ndarray:
    m = p.size
    # Step-up: the smallest, over all tests at least as large, of scale * m * p / rank.
    terms = np.array([scale * m * p[j] / np.sum(p <= p[j]) for j in range(m)], dtype=np.float64)
    return np.array([min(1.0, float(terms[p >= p[i]].min())) for i in range(m)], dtype=np.float64)


def test_adjusted_values_match_the_definitions_by_brute_force() -> None:
    p = np.random.default_rng(1).uniform(size=40) ** 2
    np.testing.assert_allclose(adjust(p, method="holm").adjusted, _brute_holm(p), rtol=1e-12)
    np.testing.assert_allclose(adjust(p, method="bh").adjusted, _brute_bh(p), rtol=1e-12)
    storey = adjust(p, method="storey", lambda_=0.5)
    assert storey.pi0 == pytest.approx(min(1.0, np.mean(p > 0.5) / 0.5))
    np.testing.assert_allclose(storey.adjusted, _brute_bh(p, storey.pi0 or 1.0), rtol=1e-12)


def test_results_come_back_in_the_input_order_and_adjusted_values_never_shrink() -> None:
    p = np.array([0.04, 0.001, 0.5, 0.03, 0.2])
    for method in ("holm", "bh", "storey"):
        result = adjust(p, method=method)  # type: ignore[arg-type]
        assert np.all(result.adjusted >= p - 1e-15) or method == "storey"
        assert np.all((result.adjusted >= 0.0) & (result.adjusted <= 1.0))
        # Listed in the order of the p-values, the adjusted values never decrease.
        assert np.all(np.diff(result.adjusted[np.argsort(p, kind="stable")]) >= -1e-15)
    assert adjust(p, method="holm").adjusted[1] == pytest.approx(0.005)


def _simulate(seed: int, m: int = 200, share_true: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """P-values of m z-tests; the first share_true of them have a real effect of 3.5 sd."""
    rng = np.random.default_rng(seed)
    n_true = int(m * share_true)
    z = rng.normal(size=m)
    z[:n_true] += 3.5
    p = np.array([1.0 - NormalDist().cdf(v) for v in z])
    return p, np.arange(m) < n_true


def test_benjamini_hochberg_keeps_the_false_discovery_rate_below_alpha() -> None:
    fdp: list[float] = []
    for seed in range(300):
        p, truth = _simulate(seed)
        rejected = adjust(p, method="bh", alpha=0.1).rejected
        fdp.append(0.0 if rejected.sum() == 0 else np.sum(rejected & ~truth) / rejected.sum())
    # BH controls FDR at alpha * pi0 = 0.1 * 0.8 = 0.08 for independent tests.
    assert float(np.mean(fdp)) <= 0.1


def test_holm_keeps_the_family_wise_error_rate_below_alpha() -> None:
    any_false: list[bool] = []
    for seed in range(300):
        p, truth = _simulate(seed)
        any_false.append(bool(np.any(adjust(p, method="holm", alpha=0.1).rejected & ~truth)))
    assert float(np.mean(any_false)) <= 0.1 + 0.04


def test_storey_rejects_more_than_benjamini_hochberg_when_many_effects_are_real() -> None:
    more = 0
    for seed in range(50):
        p, _ = _simulate(seed, share_true=0.5)
        bh = adjust(p, method="bh", alpha=0.05)
        storey = adjust(p, method="storey", alpha=0.05)
        more += storey.n_rejected >= bh.n_rejected
        assert storey.pi0 is not None and storey.pi0 < 0.75
    assert more == 50


@pytest.mark.parametrize(
    ("p", "kwargs", "message"),
    [
        ([], {}, "pvalues"),
        ([0.1, 1.2], {}, "pvalues"),
        ([0.1, float("nan")], {}, "pvalues"),
        ([0.1], {"alpha": 1.0}, "alpha"),
        ([0.1], {"method": "bonferroni"}, "method"),
        ([0.1], {"method": "storey", "lambda_": 1.0}, "lambda_"),
    ],
)
def test_bad_input_is_refused(p: list[float], kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        adjust(p, **kwargs)  # type: ignore[arg-type]


def test_methods_are_the_documented_ones() -> None:
    methods: tuple[Method, ...] = ("holm", "bh", "storey")
    assert all(adjust([0.01, 0.02], method=m).method == m for m in methods)
