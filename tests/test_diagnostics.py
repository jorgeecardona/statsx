import numpy as np
import pytest

from statsx.diagnostics import ConvergenceError, check_convergence


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
