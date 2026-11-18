import math

import numpy as np
import pytest

from statsx.pairwise import Comparisons, fit_bradley_terry


def _simulate(
    theta: np.ndarray,
    n_items: int,
    seed: int,
    *,
    nu: float = 0.0,
    item_noise: float = 0.0,
) -> Comparisons:
    """Every pair on every item; Davidson outcomes; an item shifts each system by its own noise."""
    rng = np.random.default_rng(seed)
    k = theta.size
    first: list[str] = []
    second: list[str] = []
    outcome: list[float] = []
    item: list[int] = []
    for it in range(n_items):
        shift = rng.normal(0.0, item_noise, k)
        for i in range(k):
            for j in range(i + 1, k):
                d = (theta[i] + shift[i]) - (theta[j] + shift[j])
                z = 2.0 * math.cosh(d / 2.0) + nu
                u = rng.random() * z
                win, loss = math.exp(d / 2.0), math.exp(-d / 2.0)
                result = 1.0 if u < win else 0.0 if u < win + loss else 0.5
                first.append(f"s{i}")
                second.append(f"s{j}")
                outcome.append(result)
                item.append(it)
    return Comparisons.from_long(first, second, outcome, item=item)


def _log_posterior(data: Comparisons, x: np.ndarray, ties: str, prior_sd: float) -> float:
    k = len(data.systems)
    d = x[data.first] - x[data.second]
    y = data.outcome
    if ties == "half":
        value = y * -np.logaddexp(0.0, -d) + (1.0 - y) * -np.logaddexp(0.0, d)
        return float(np.sum(value) - 0.5 * float(x[:k] @ x[:k]) / prior_sd**2)
    tau = x[k]
    nu = math.exp(tau)
    z = 2.0 * np.cosh(d / 2.0) + nu
    value = np.where(y == 1.0, d / 2.0, np.where(y == 0.0, -d / 2.0, tau)) - np.log(z)
    return float(np.sum(value) - 0.5 * float(x[:k] @ x[:k]) / prior_sd**2 - 0.5 * tau**2 / 9.0)


@pytest.mark.parametrize("ties", ["half", "davidson"])
def test_the_fit_is_the_maximum_of_the_penalized_log_likelihood(ties: str) -> None:
    data = _simulate(np.array([0.8, 0.0, -0.3, -0.5]), 60, 0, nu=0.7)
    fit = fit_bradley_terry(data, ties=ties, n_bootstrap=2)  # type: ignore[arg-type]
    k = len(data.systems)
    x = np.concatenate([fit.strength, [math.log(fit.tie_parameter)] if ties == "davidson" else []])
    h = 1e-5
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = h
        grad = (
            _log_posterior(data, x + e, ties, fit.prior_sd)
            - _log_posterior(data, x - e, ties, fit.prior_sd)
        ) / (2 * h)
        assert abs(grad) < 1e-4
    # Any move that keeps the mean at 0 lowers the objective.
    rng = np.random.default_rng(1)
    base = _log_posterior(data, x, ties, fit.prior_sd)
    for _ in range(20):
        v = x.copy()
        move = rng.normal(0.0, 0.05, k)
        v[:k] += move - move.mean()
        assert _log_posterior(data, v, ties, fit.prior_sd) <= base + 1e-9


def test_two_systems_have_the_closed_form_bradley_terry_and_davidson_fits() -> None:
    wins, losses, ties = 60, 30, 10
    outcome = [1.0] * wins + [0.0] * losses + [0.5] * ties
    data = Comparisons.from_long(["a"] * 100, ["b"] * 100, outcome)
    half = fit_bradley_terry(data, ties="half", prior_sd=1e4, n_bootstrap=2)
    # Ties count half: 65 wins out of 100.
    assert half.strength[0] - half.strength[1] == pytest.approx(math.log(65 / 35), rel=1e-4)
    dav = fit_bradley_terry(data, ties="davidson", prior_sd=1e4, n_bootstrap=2)
    d = math.log(wins / losses)
    f_t = ties / 100
    nu = f_t * (math.exp(d / 2) + math.exp(-d / 2)) / (1.0 - f_t)
    assert dav.strength[0] - dav.strength[1] == pytest.approx(d, abs=2e-3)
    assert dav.tie_parameter == pytest.approx(nu, rel=0.03)  # a Normal(0, 3^2) prior on log nu
    p_a, p_b, p_tie = dav.probabilities("a", "b")
    assert (p_a, p_b, p_tie) == pytest.approx((0.6, 0.3, 0.1), abs=0.01)


def test_strengths_and_the_tie_parameter_are_recovered_from_a_large_sample() -> None:
    theta = np.array([0.9, 0.2, -0.4, -0.7])
    data = _simulate(theta, 800, 2, nu=0.8)
    fit = fit_bradley_terry(data, n_bootstrap=2)
    assert np.max(np.abs(fit.strength - (theta - theta.mean()))) < 0.08
    assert fit.tie_parameter == pytest.approx(0.8, abs=0.1)
    assert fit.n_items == 800 and fit.n_comparisons == 800 * 6


def test_a_system_that_always_wins_gets_a_finite_strength() -> None:
    data = Comparisons.from_long(["a"] * 10 + ["b"] * 10, ["b"] * 10 + ["c"] * 10, [1.0] * 20)
    fit = fit_bradley_terry(data, ties="half", n_bootstrap=20)
    assert np.all(np.isfinite(fit.strength)) and fit.strength[0] > fit.strength[1] > fit.strength[2]
    assert fit.p_better("a", "c") > 0.99


def _row_level(data: Comparisons) -> Comparisons:
    return Comparisons(
        data.systems, data.first, data.second, data.outcome, np.arange(data.outcome.size)
    )


def test_item_level_intervals_are_wider_than_row_level_ones_when_rows_of_an_item_depend() -> None:
    theta = np.array([0.8, 0.2, -0.3, -0.7])
    ratios: list[float] = []
    for seed in range(20):
        data = _simulate(theta, 20, 900 + seed, nu=0.5, item_noise=1.5)
        item = fit_bradley_terry(data, n_bootstrap=100, seed=seed).difference_interval(
            "s0", "s3", 0.9
        )
        row = fit_bradley_terry(_row_level(data), n_bootstrap=100, seed=seed).difference_interval(
            "s0", "s3", 0.9
        )
        ratios.append((item.upper - item.lower) / (row.upper - row.lower))
    assert float(np.mean(ratios)) > 1.03
    # With one row per item the two resamplings are the same.
    single = _simulate(theta, 20, 1, nu=0.5)
    flat = Comparisons(
        single.systems, single.first, single.second, single.outcome, np.arange(single.outcome.size)
    )
    one = fit_bradley_terry(flat, n_bootstrap=100, seed=3).difference_interval("s0", "s3", 0.9)
    two = fit_bradley_terry(_row_level(single), n_bootstrap=100, seed=3).difference_interval(
        "s0", "s3", 0.9
    )
    assert (one.lower, one.upper) == (two.lower, two.upper)


def test_the_item_level_interval_covers_the_population_value_of_the_estimand() -> None:
    # With item noise the marginal win rates are flatter than the strengths that made them, so the
    # estimand is the fit on a very large sample of the same process, not the strengths put in.
    theta = np.array([0.8, 0.2, -0.3, -0.7])
    population = fit_bradley_terry(
        _simulate(theta, 3000, 12345, nu=0.5, item_noise=1.0), n_bootstrap=2
    )
    truth = float(population.strength[0] - population.strength[3])
    assert truth < theta[0] - theta[3]
    hits = 0
    runs = 40
    for seed in range(runs):
        data = _simulate(theta, 20, 900 + seed, nu=0.5, item_noise=1.0)
        interval = fit_bradley_terry(data, n_bootstrap=100, seed=seed).difference_interval(
            "s0", "s3", 0.9
        )
        hits += interval.contains(truth)
    # A percentile bootstrap over 20 items is a little narrow: 90% nominal, about 82% observed.
    assert 0.72 <= hits / runs <= 0.97


def test_probabilities_add_up_and_the_intervals_order_the_systems() -> None:
    data = _simulate(np.array([0.7, 0.0, -0.7]), 150, 3, nu=0.4)
    fit = fit_bradley_terry(data, n_bootstrap=300)
    for a, b in (("s0", "s1"), ("s1", "s2"), ("s0", "s2")):
        assert sum(fit.probabilities(a, b)) == pytest.approx(1.0)
    interval = fit.difference_interval("s0", "s2", 0.95)
    assert interval.lower > 0.0 and fit.p_better("s0", "s2") == 1.0
    assert fit.strength_interval("s0").contains(float(fit.strength[0]))
    assert abs(fit.strength.sum()) < 1e-9


@pytest.mark.parametrize(
    "call",
    [
        lambda: Comparisons.from_long([], [], []),
        lambda: Comparisons.from_long(["a"], ["a"], [1.0]),
        lambda: Comparisons.from_long(["a"], ["b"], ["maybe"]),
        lambda: Comparisons.from_long(["a"], ["b"], [0.3]),
        lambda: Comparisons.from_long(["a", "a"], ["b", "b"], [1.0, 0.0], item=[1]),
        lambda: fit_bradley_terry(Comparisons.from_long(["a"], ["b"], [1.0]), prior_sd=0.0),
        lambda: fit_bradley_terry(Comparisons.from_long(["a"], ["b"], [1.0]), n_bootstrap=1),
        lambda: fit_bradley_terry(Comparisons.from_long(["a"], ["b"], [1.0]), ties="x"),  # type: ignore[arg-type]
        lambda: fit_bradley_terry(
            Comparisons.from_long(["a"], ["b"], [1.0]), n_bootstrap=3
        ).p_better("a", "z"),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"same length|compared with itself|outcome must|prior_sd|ties must|unknown",
    ):
        call()  # type: ignore[operator]
