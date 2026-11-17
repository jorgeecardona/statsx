import math
from statistics import NormalDist

import numpy as np
import pytest

from statsx.allocation import Allocation, stratified_allocation

# Cochran (1977), section 5.5 example shape: three strata of different size and spread.
SIZES = [4000, 2500, 1000]
SDS = [4.0, 10.0, 20.0]


def _variance(sizes: list[int], sds: list[float], n: list[int]) -> float:
    total = sum(sizes)
    return sum(
        (big / total) ** 2 * s**2 * (1 / k - 1 / big)
        for big, s, k in zip(sizes, sds, n, strict=True)
    )


def test_the_neyman_allocation_follows_n_h_proportional_to_size_times_sd() -> None:
    result = stratified_allocation(SIZES, sds=SDS, margin=0.5)
    assert isinstance(result, Allocation)
    products = np.array(SIZES) * np.array(SDS)
    expected_share = products / products.sum()
    share = np.array(result.sizes) / result.total
    np.testing.assert_allclose(share, expected_share, atol=0.01)
    assert result.method == "neyman"
    assert sum(result.sizes) == result.total


def test_the_allocation_is_the_smallest_total_that_meets_the_margin() -> None:
    z = NormalDist().inv_cdf(0.975)
    result = stratified_allocation(SIZES, sds=SDS, margin=0.5)
    assert result.margin <= 0.5 + 1e-12
    assert result.margin == pytest.approx(z * math.sqrt(_variance(SIZES, SDS, list(result.sizes))))
    smaller = stratified_allocation(SIZES, sds=SDS, margin=0.5)
    # One item fewer in the largest stratum misses the target.
    reduced = list(smaller.sizes)
    reduced[int(np.argmax(reduced))] -= 1
    assert z * math.sqrt(_variance(SIZES, SDS, reduced)) > 0.5 - 1e-3


def test_neyman_needs_fewer_items_than_proportional_allocation_for_the_same_margin() -> None:
    neyman = stratified_allocation(SIZES, sds=SDS, margin=0.4)
    proportional = stratified_allocation(SIZES, sds=SDS, margin=0.4, method="proportional")
    assert neyman.total < proportional.total
    share = np.array(proportional.sizes) / proportional.total
    np.testing.assert_allclose(share, np.array(SIZES) / sum(SIZES), atol=0.01)


def test_proportions_replace_standard_deviations_and_costs_shift_the_split() -> None:
    by_p = stratified_allocation([5000, 5000], proportions=[0.5, 0.05], margin=0.02)
    by_sd = stratified_allocation([5000, 5000], sds=[0.5, math.sqrt(0.05 * 0.95)], margin=0.02)
    assert by_p.sizes == by_sd.sizes
    assert by_p.sizes[0] > by_p.sizes[1]
    even = stratified_allocation([5000, 5000], sds=[1.0, 1.0], margin=0.05)
    costly = stratified_allocation([5000, 5000], sds=[1.0, 1.0], margin=0.05, costs=[1.0, 9.0])
    assert abs(even.sizes[0] - even.sizes[1]) <= 1
    assert costly.sizes[0] > 2.5 * costly.sizes[1]


def test_a_small_stratum_is_capped_at_its_size_and_the_rest_is_reallocated() -> None:
    result = stratified_allocation([30, 10000, 10000], sds=[50.0, 1.0, 1.0], margin=0.05)
    assert result.sizes[0] == 30
    assert result.sizes[1] >= 1 and result.sizes[2] >= 1
    assert result.margin <= 0.05 + 1e-12


def test_a_tighter_margin_needs_more_items() -> None:
    wide = stratified_allocation(SIZES, sds=SDS, margin=1.0)
    narrow = stratified_allocation(SIZES, sds=SDS, margin=0.25)
    assert narrow.total > wide.total


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sizes": [10, 1], "sds": [1.0, 1.0]}, "sizes"),
        ({"sizes": [10, 10]}, "either"),
        ({"sizes": [10, 10], "sds": [1.0], "margin": 0.1}, "same length"),
        (
            {"sizes": [10, 10], "sds": [1.0, 1.0], "proportions": [0.5, 0.5], "margin": 0.1},
            "either",
        ),
        ({"sizes": [10, 10], "proportions": [0.5, 1.5], "margin": 0.1}, "proportions"),
        ({"sizes": [10, 10], "sds": [0.0, 0.0], "margin": 0.1}, "no variation"),
        ({"sizes": [10, 10], "sds": [1.0, 1.0], "margin": 0.0}, "margin"),
        ({"sizes": [10, 10], "sds": [1.0, 1.0], "margin": 0.1, "costs": [1.0, 0.0]}, "costs"),
        ({"sizes": [10, 10], "sds": [1.0, 1.0], "margin": 0.1, "method": "equal"}, "method"),
    ],
)
def test_bad_input_is_refused(kwargs: dict[str, object], message: str) -> None:
    kwargs.setdefault("margin", 0.1)
    with pytest.raises(ValueError, match=message):
        stratified_allocation(**kwargs)  # type: ignore[arg-type]
