import math

import numpy as np
import pytest

from statsx.proportion import Method, interval_ends, proportion_interval
from statsx.result import Estimate

# mpmath at 40 digits: Wilson and Agresti-Coull closed forms; Clopper-Pearson and Jeffreys from
# the Beta quantiles (with the ends set to 0 and 1 for x = 0 and x = n).
REFERENCE: dict[tuple[int, int], dict[Method, tuple[float, float]]] = {
    (5, 10): {
        "wilson": (0.23659309051256397, 0.7634069094874361),
        "agresti_coull": (0.23659309051256397, 0.7634069094874361),
        "clopper_pearson": (0.18708602844739852, 0.8129139715526015),
        "jeffreys": (0.2235286702527052, 0.7764713297472948),
    },
    (81, 263): {
        "wilson": (0.2552885198782742, 0.36620957698280004),
        "agresti_coull": (0.25522066518999675, 0.3662774316710775),
        "clopper_pearson": (0.25273674558527115, 0.36762192260135135),
        "jeffreys": (0.2545219349910235, 0.3656474991740824),
    },
    (0, 20): {
        "wilson": (0.0, 0.16112515805281938),
        "agresti_coull": (0.0, 0.18980956054248885),
        "clopper_pearson": (0.0, 0.16843347098308534),
        "jeffreys": (0.0, 0.11663898290487543),
    },
    (1, 29): {
        "wilson": (0.00611321429276266, 0.17175521879320288),
        "agresti_coull": (0.0, 0.1862865085617061),
        "clopper_pearson": (0.0008726468835799219, 0.17764429548872293),
        "jeffreys": (0.0037461736490570735, 0.15007768600223176),
    },
    (20, 20): {
        "wilson": (0.8388748419471806, 1.0),
        "agresti_coull": (0.8101904394575111, 1.0),
        "clopper_pearson": (0.8315665290169146, 1.0),
        "jeffreys": (0.8833610170951246, 1.0),
    },
}
METHODS: tuple[Method, ...] = ("wilson", "jeffreys", "clopper_pearson", "agresti_coull")


@pytest.mark.parametrize(("counts", "methods"), list(REFERENCE.items()))
def test_each_interval_matches_the_reference_values(
    counts: tuple[int, int], methods: dict[Method, tuple[float, float]]
) -> None:
    x, n = counts
    for method, (lower, upper) in methods.items():
        result = proportion_interval(x, n, method=method)
        assert result.interval.lower == pytest.approx(lower, abs=1e-10), method
        assert result.interval.upper == pytest.approx(upper, abs=1e-10), method


def test_the_result_has_the_proportion_the_size_and_the_method() -> None:
    result = proportion_interval(7, 10, method="clopper_pearson", level=0.9)
    assert isinstance(result, Estimate)
    assert result.estimate == 0.7
    assert result.n == 10
    assert result.interval.level == 0.9
    assert "Clopper-Pearson" in result.assumption
    assert proportion_interval(7, 10).interval.contains(0.7)


def _coverage(n: int, method: Method) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact coverage of the 95% interval over a grid of true proportions."""
    grid = np.linspace(0.0005, 0.9995, 2000)
    x = np.arange(n + 1)
    log_choose = np.array(
        [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in x]
    )
    pmf = np.exp(
        log_choose[:, None]
        + x[:, None] * np.log(grid)[None, :]
        + (n - x)[:, None] * np.log1p(-grid)[None, :]
    )
    ends = np.array([interval_ends(int(k), n, 0.95, method) for k in x])
    inside = (ends[:, 0][:, None] <= grid[None, :]) & (grid[None, :] <= ends[:, 1][:, None])
    width = float(np.mean(ends[:, 1] - ends[:, 0]))
    return grid, (pmf * inside).sum(axis=0), width


def test_the_small_n_behaviour_stated_in_the_docs_is_true_at_ten_trials() -> None:
    grid, wilson, w_width = _coverage(10, "wilson")
    _, jeffreys, _ = _coverage(10, "jeffreys")
    _, exact, e_width = _coverage(10, "clopper_pearson")
    _, agresti, a_width = _coverage(10, "agresti_coull")
    middle = (grid > 0.1) & (grid < 0.9)
    # Wilson: near 95% on average, low only near 0 and 1.
    assert wilson.mean() == pytest.approx(0.954, abs=0.005)
    assert wilson.min() == pytest.approx(0.836, abs=0.005)
    assert grid[wilson.argmin()] < 0.05 or grid[wilson.argmin()] > 0.95
    assert wilson[middle].min() > 0.91
    # Jeffreys: near 95% on average, down to about 87%.
    assert jeffreys.mean() == pytest.approx(0.953, abs=0.005)
    assert jeffreys.min() == pytest.approx(0.868, abs=0.005)
    # Clopper-Pearson: never below 95%, the widest.
    assert exact.min() >= 0.95
    assert exact.mean() == pytest.approx(0.984, abs=0.005)
    assert e_width > a_width > w_width
    # Agresti-Coull: never far below nominal, above it on average.
    assert agresti.min() == pytest.approx(0.924, abs=0.005)
    assert agresti.mean() == pytest.approx(0.965, abs=0.005)


def test_the_ends_are_set_at_zero_and_all_successes() -> None:
    for method in METHODS:
        low = proportion_interval(0, 15, method=method).interval
        high = proportion_interval(15, 15, method=method).interval
        assert low.lower == pytest.approx(0.0, abs=1e-12)
        assert high.upper == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= low.upper < 0.3
        assert 0.7 < high.lower <= 1.0


def test_a_larger_sample_gives_a_narrower_interval_and_a_higher_level_a_wider_one() -> None:
    for method in METHODS:
        small = proportion_interval(30, 100, method=method).interval
        large = proportion_interval(300, 1000, method=method).interval
        assert large.upper - large.lower < small.upper - small.lower
        wide = proportion_interval(30, 100, method=method, level=0.99).interval
        assert wide.upper - wide.lower > small.upper - small.lower


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"successes": 11, "trials": 10}, "counts"),
        ({"successes": -1, "trials": 10}, "counts"),
        ({"successes": 1, "trials": 0}, "counts"),
        ({"successes": 1.5, "trials": 10}, "counts"),
        ({"successes": 1, "trials": 10, "level": 1.0}, "level"),
        ({"successes": 1, "trials": 10, "method": "normal"}, "method"),
    ],
)
def test_bad_input_is_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        proportion_interval(**kwargs)  # type: ignore[arg-type]
