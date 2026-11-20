"""GitLab issue #13: runs that hit the token budget are censored, not dropped and not taken as is."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from statsx.evals import (
    _censored_derivatives,  # pyright: ignore[reportPrivateUsage]
    _censored_loglik,  # pyright: ignore[reportPrivateUsage]
    fit_tokens,
)

TASKS = 30
RATIO = 1.5  # model b uses 1.5 times the tokens of a


def _runs(
    rng: np.random.Generator, budget: float
) -> tuple[list[str], list[str], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Two models, two attempts per task; long tasks hit the budget, more often for b."""
    task_effect = rng.normal(8.0, 0.8, TASKS)
    models: list[str] = []
    tasks: list[str] = []
    need: list[float] = []
    for m, shift in (("a", 0.0), ("b", math.log(RATIO))):
        for t in range(TASKS):
            for _ in range(2):
                models.append(m)
                tasks.append(f"t{t}")
                need.append(math.exp(task_effect[t] + shift + rng.normal(0.0, 0.3)))
    tokens = np.asarray(need)
    cut = tokens >= budget
    return models, tasks, np.minimum(tokens, budget), cut


def test_the_censored_fit_recovers_the_ratio_where_the_plain_fits_do_not() -> None:
    rng = np.random.default_rng(0)
    covered = 0
    plain: list[float] = []
    dropped: list[float] = []
    runs = 150
    for _ in range(runs):
        models, tasks, tokens, cut = _runs(rng, budget=math.exp(8.6))
        fit = fit_tokens(models, tasks, tokens, censored=cut)
        covered += fit.ratio("b", "a", 0.9).contains(RATIO)
        whole = fit_tokens(models, tasks, tokens).log_ratio("b", "a")
        plain.append(math.exp((whole.lower + whole.upper) / 2))
        keep = ~cut
        kept = fit_tokens(
            [m for m, k in zip(models, keep, strict=True) if k],
            [t for t, k in zip(tasks, keep, strict=True) if k],
            tokens[keep],
        )
        part = kept.log_ratio("b", "a")
        dropped.append(math.exp((part.lower + part.upper) / 2))
    assert 0.84 <= covered / runs <= 0.96
    assert np.median(plain) < 1.35
    assert np.median(dropped) < 1.45


def test_without_censored_rows_the_fit_is_the_exact_one() -> None:
    rng = np.random.default_rng(1)
    models, tasks, tokens, _ = _runs(rng, budget=1e12)
    exact = fit_tokens(models, tasks, tokens)
    flagged = fit_tokens(models, tasks, tokens, censored=np.zeros(tokens.size, dtype=bool))
    assert np.array_equal(flagged.effects, exact.effects)
    assert flagged.noise_sd == exact.noise_sd
    assert flagged.n_censored == 0


def test_the_gradient_and_hessian_match_finite_differences() -> None:
    rng = np.random.default_rng(2)
    design = np.column_stack([np.ones(40), rng.normal(size=40)])
    y = 1.0 + 0.5 * design[:, 1] + rng.normal(0.0, 0.4, 40)
    cut = y > 1.3
    y = np.minimum(y, 1.3)
    theta = np.array([0.9, 0.4, math.log(0.5)])
    gradient, hessian = _censored_derivatives(design, y, cut, theta)
    h = 1e-6
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        numeric = (
            _censored_loglik(design, y, cut, theta + step)
            - _censored_loglik(design, y, cut, theta - step)
        ) / (2 * h)
        assert numeric == pytest.approx(gradient[j], rel=1e-5, abs=1e-6)
        g_plus, _ = _censored_derivatives(design, y, cut, theta + step)
        g_minus, _ = _censored_derivatives(design, y, cut, theta - step)
        assert np.allclose((g_plus - g_minus) / (2 * h), hessian[:, j], rtol=1e-4, atol=1e-5)


def test_far_in_the_tail_the_likelihood_stays_finite() -> None:
    design = np.ones((3, 1))
    y = np.array([0.0, 0.0, 60.0])
    cut = np.array([False, False, True])
    value = _censored_loglik(design, y, cut, np.array([0.0, 0.0]))
    gradient, hessian = _censored_derivatives(design, y, cut, np.array([0.0, 0.0]))
    assert math.isfinite(value)
    assert np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))


def test_a_model_with_only_censored_rows_is_refused_and_named() -> None:
    models = ["a", "a", "b", "b", "a", "b"]
    tasks = ["t1", "t2", "t1", "t2", "t3", "t3"]
    tokens = [100.0, 200.0, 500.0, 500.0, 150.0, 500.0]
    with pytest.raises(ValueError, match=r"model \['b'\]"):
        fit_tokens(models, tasks, tokens, censored=[False, False, True, True, False, True])


def test_a_task_where_every_run_hit_the_budget_is_left_out_and_named() -> None:
    models = ["a", "b"] * 4
    tasks = ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"]
    tokens = [100.0, 160.0, 210.0, 300.0, 500.0, 500.0, 120.0, 190.0]
    cut = [False, False, False, False, True, True, False, False]
    fit = fit_tokens(models, tasks, tokens, censored=cut)
    assert fit.left_out == ("t3",)
    assert "t3" not in fit.tasks
    assert fit.n_censored == 0


def test_the_flags_must_match_the_rows() -> None:
    with pytest.raises(ValueError, match="one flag per row"):
        fit_tokens(
            ["a", "b", "a", "b"], ["t1", "t1", "t2", "t2"], [1.0, 2.0, 3.0, 4.0], censored=[True]
        )
