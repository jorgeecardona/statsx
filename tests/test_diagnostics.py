import math
from collections.abc import Callable

import numpy as np
import pytest

from statsx.diagnostics import (
    ConvergenceError,
    check_convergence,
    ess_mean,
    mcse_mean,
    mcse_quantile,
    mcse_sd,
)


def _iid(seed: int, chains: int = 4, draws: int = 1000) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(chains, draws))


def test_one_check_reports_rhat_ess_and_divergences_and_refuses_when_any_fails() -> None:
    good = check_convergence({"mu": _iid(0)})
    assert good.passed
    good.require()
    (mu,) = good.parameters
    assert mu.rhat < 1.01
    assert mu.ess_bulk > 3000
    assert mu.ess_tail > 2500

    stuck = _iid(1)
    stuck[3] += 3.0  # one chain sits elsewhere
    for check in (
        check_convergence({"mu": stuck}),
        check_convergence({"mu": _iid(2)}, divergences=2),
        check_convergence({"mu": _iid(3, draws=60)}),  # too few draws: ESS below 400
    ):
        assert not check.passed
        with pytest.raises(ConvergenceError) as info:
            check.require()
        assert info.value.check is check


def test_reasons_name_the_parameter_and_the_threshold() -> None:
    stuck = _iid(4)
    stuck[0] -= 5.0
    check = check_convergence({"mu": _iid(5), "sigma": stuck}, divergences=1)
    assert any(r.startswith("sigma: R-hat") for r in check.reasons)
    assert not any(r.startswith("mu:") for r in check.reasons)
    assert "1 divergent transitions" in check.reasons


def test_constant_draws_fail_with_nan_not_pass() -> None:
    check = check_convergence({"c": np.ones((4, 100))})
    assert not check.passed


def test_ar1_bulk_ess_is_a_third_of_draws_for_phi_one_half() -> None:
    rng = np.random.default_rng(6)
    phi, n = 0.5, 5000
    x = np.zeros((4, n))
    noise = rng.normal(size=(4, n)) * np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + noise[:, t]
    (p,) = check_convergence({"x": x}).parameters
    assert p.ess_bulk / (4 * n) == pytest.approx(1 / 3, abs=0.05)


def test_refuses_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        check_convergence({"mu": np.zeros(100)})


def test_matches_arviz_on_a_fixed_ar1_sample() -> None:
    # Expected numbers from ArviZ 0.22.0 (rhat method="rank", ess "bulk" and "tail") on
    # these exact draws; ArviZ is a reference only, not a dependency.
    rng = np.random.default_rng(11)
    x = np.zeros((4, 800))
    e = rng.normal(size=(4, 800))
    for t in range(1, 800):
        x[:, t] = 0.7 * x[:, t - 1] + e[:, t]
    x[2] += 0.4
    (p,) = check_convergence({"x": x}).parameters
    assert p.rhat == pytest.approx(1.0130120703226426, rel=1e-10)
    assert p.ess_bulk == pytest.approx(499.05054140767186, rel=1e-10)
    assert p.ess_tail == pytest.approx(1106.2186622564639, rel=1e-10)


def _ar1(seed: int, phi: float, chains: int = 4, draws: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, draws))
    state = rng.normal(size=chains)
    for t in range(draws):
        state = phi * state + np.sqrt(1.0 - phi**2) * rng.normal(size=chains)
        x[:, t] = state
    return x


def _spread_of_estimates(phi: float, summary: Callable[[np.ndarray], float]) -> float:
    return float(np.std([summary(_ar1(seed, phi)) for seed in range(300)], ddof=1))


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_mcse_mean_matches_the_spread_of_the_mean_over_repeated_runs(phi: float) -> None:
    reported = float(np.mean([mcse_mean(_ar1(seed, phi)) for seed in range(60)]))
    actual = _spread_of_estimates(phi, lambda x: float(x.mean()))
    assert reported == pytest.approx(actual, rel=0.15)
    # theory for an AR(1) with unit variance: sqrt((1 + phi) / (1 - phi) / N), N = 2000
    assert reported == pytest.approx(np.sqrt((1 + phi) / (1 - phi) / 2000), rel=0.15)


def test_mcse_sd_matches_the_spread_of_the_sd_over_repeated_runs() -> None:
    reported = float(np.mean([mcse_sd(_ar1(seed, 0.5)) for seed in range(60)]))
    actual = _spread_of_estimates(0.5, lambda x: float(x.std(ddof=1)))
    assert reported == pytest.approx(actual, rel=0.2)


@pytest.mark.parametrize("prob", [0.05, 0.5, 0.95])
def test_mcse_quantile_matches_the_spread_of_the_quantile_over_repeated_runs(prob: float) -> None:
    reported = float(np.mean([mcse_quantile(_ar1(seed, 0.5), prob) for seed in range(60)]))
    actual = _spread_of_estimates(0.5, lambda x: float(np.quantile(x, prob)))
    assert reported == pytest.approx(actual, rel=0.25)


def test_the_effective_sample_size_of_the_mean_is_a_third_of_the_draws_for_phi_one_half() -> None:
    assert ess_mean(_ar1(0, 0.5, draws=5000)) / 20000 == pytest.approx(1 / 3, abs=0.05)


def test_constant_draws_give_nan_and_bad_input_is_refused() -> None:
    flat = np.ones((4, 100))
    assert math.isnan(ess_mean(flat)) and math.isnan(mcse_mean(flat))
    for call in (
        lambda: mcse_mean(np.ones(100)),
        lambda: mcse_mean(np.ones((2, 4))),
        lambda: mcse_mean(np.array([[np.nan] * 20])),
        lambda: mcse_quantile(_iid(0), 1.0),
        lambda: mcse_sd(np.ones((2, 2, 2))),
    ):
        with pytest.raises(ValueError, match=r"."):
            call()
