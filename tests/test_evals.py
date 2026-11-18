import math

import numpy as np
import pytest

from statsx import _accel
from statsx.evals import Responses, cost_per_pass, fit_rasch, fit_tokens
from statsx.posterior import compare


def _simulate(
    n_models: int,
    n_tasks: int,
    seed: int,
    *,
    missing: float = 0.0,
    sd: float = 1.0,
) -> tuple[Responses, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, sd, n_models)
    b = rng.normal(0.0, sd, n_tasks)
    models: list[str] = []
    tasks: list[str] = []
    passed: list[float] = []
    for i in range(n_models):
        for j in range(n_tasks):
            if rng.random() < missing:
                continue
            p = 1.0 / (1.0 + math.exp(-(theta[i] - b[j])))
            models.append(f"m{i}")
            tasks.append(f"t{j}")
            passed.append(float(rng.random() < p))
    return Responses.from_long(models, tasks, passed), theta, b


def _log_posterior(x: np.ndarray, data: Responses, sd: float) -> float:
    n_m = len(data.models)
    eta = x[data.model_index] - x[n_m + data.task_index]
    y = data.passed
    log_lik = np.sum(y * -np.logaddexp(0.0, -eta) + (1.0 - y) * -np.logaddexp(0.0, eta))
    return float(log_lik - 0.5 * np.sum(x**2) / sd**2)


def test_the_mode_has_a_zero_gradient_and_the_covariance_is_minus_the_inverse_hessian() -> None:
    data, _, _ = _simulate(4, 12, 0)
    fit = fit_rasch(data, ability_sd=1.5, difficulty_sd=1.5)
    x = np.concatenate([fit.ability, fit.difficulty])
    n = x.size
    h = 1e-4
    grad = np.zeros(n)
    hess = np.zeros((n, n))
    f0 = _log_posterior(x, data, 1.5)
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        grad[i] = (_log_posterior(x + e, data, 1.5) - _log_posterior(x - e, data, 1.5)) / (2 * h)
        for j in range(n):
            g = np.zeros(n)
            g[j] = h
            hess[i, j] = (
                _log_posterior(x + e + g, data, 1.5)
                - _log_posterior(x + e - g, data, 1.5)
                - _log_posterior(x - e + g, data, 1.5)
                + _log_posterior(x - e - g, data, 1.5)
            ) / (4 * h * h)
    assert f0 > _log_posterior(x + 0.05, data, 1.5)
    np.testing.assert_allclose(grad, 0.0, atol=1e-6)
    np.testing.assert_allclose(fit.covariance, np.linalg.inv(-hess), rtol=1e-4, atol=1e-6)


def test_abilities_and_difficulties_recover_the_truth_with_enough_data() -> None:
    data, theta, _ = _simulate(6, 300, 1)
    fit = fit_rasch(data)
    assert np.max(np.abs(fit.ability - fit.ability.mean() - (theta - theta.mean()))) < 0.25
    # Difficulties need many models: with 30 models each task has 30 results.
    wide, _, b = _simulate(30, 80, 2)
    assert np.corrcoef(fit_rasch(wide).difficulty, b)[0, 1] > 0.85
    # The fitted solve rate matches the observed solve rate.
    predicted = np.mean(
        [
            fit.p_solve(data.models[int(m)], data.tasks[int(t)])
            for m, t in zip(data.model_index.tolist(), data.task_index.tolist(), strict=True)
        ]
    )
    assert predicted == pytest.approx(float(data.passed.mean()), abs=0.01)


def test_the_ability_difference_interval_has_its_stated_coverage() -> None:
    hits = total = 0
    for seed in range(300):
        data, theta, _ = _simulate(6, 40, 1000 + seed)
        fit = fit_rasch(data, ability_sd=1.0, difficulty_sd=1.0)
        for a, b in ((0, 1), (2, 3), (4, 5)):
            interval = fit.ability_difference(data.models[a], data.models[b], level=0.9)
            hits += interval.contains(
                float(theta[int(data.models[a][1:])] - theta[int(data.models[b][1:])])
            )
            total += 1
    assert 0.86 <= hits / total <= 0.94


def test_p_better_matches_the_difference_and_draws_agree_with_the_gaussian() -> None:
    data, _, _ = _simulate(5, 60, 3)
    fit = fit_rasch(data)
    a, b = data.models[0], data.models[1]
    diff = fit.ability_difference(a, b, level=0.9)
    assert (fit.p_better(a, b) > 0.5) == ((diff.lower + diff.upper) > 0.0)
    assert fit.p_better(a, b) == pytest.approx(1.0 - fit.p_better(b, a), abs=1e-12)
    draws = fit.ability_draws(100_000, seed=1)
    assert float(np.mean(draws[:, 0] > draws[:, 1])) == pytest.approx(fit.p_better(a, b), abs=0.01)


def test_missing_cells_link_models_through_shared_tasks_and_unlinked_data_is_refused() -> None:
    data, _, _ = _simulate(6, 60, 4, missing=0.6)
    assert data.n_components == 1
    assert len(fit_rasch(data).ability) == 6
    # Two groups with no task in common are on different scales.
    split = Responses.from_long(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        ["t1", "t2", "t1", "t2", "t3", "t4", "t3", "t4"],
        [1, 0, 0, 0, 1, 1, 0, 1],
    )
    assert split.n_components == 2
    with pytest.raises(ValueError, match="not linked"):
        fit_rasch(split)
    assert len(fit_rasch(split, allow_disconnected=True).ability) == 4


def test_a_model_that_solves_everything_gets_a_finite_ability() -> None:
    data = Responses.from_long(
        ["strong"] * 10 + ["weak"] * 10,
        [f"t{i}" for i in range(10)] * 2,
        [1.0] * 10 + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    fit = fit_rasch(data)
    assert np.all(np.isfinite(fit.ability)) and fit.ability[0] > fit.ability[1]
    assert fit.p_better("strong", "weak") > 0.95


def test_repeated_attempts_and_scores_between_0_and_1_are_accepted() -> None:
    data = Responses.from_long(
        ["a", "a", "a", "b", "b", "b"], ["t", "t", "u", "t", "u", "u"], [1, 0.5, 1, 0, 0, 0.25]
    )
    fit = fit_rasch(data)
    assert fit.n_rows == 6 and fit.ability[0] > fit.ability[1]


@pytest.mark.parametrize(
    "call",
    [
        lambda: Responses.from_long(["a"], ["t", "u"], [1.0]),
        lambda: Responses.from_long([], [], []),
        lambda: Responses.from_long(["a"], ["t"], [1.5]),
        lambda: fit_rasch(Responses.from_long(["a"], ["t"], [1.0]), ability_sd=0.0),
        lambda: fit_rasch(Responses.from_long(["a"], ["t"], [1.0])).p_better("a", "zzz"),
        lambda: fit_rasch(Responses.from_long(["a"], ["t"], [1.0])).ability_interval("a", 1.0),
        lambda: fit_rasch(Responses.from_long(["a"], ["t"], [1.0])).p_solve("a", "zzz"),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match=r"same length|between 0 and 1|positive|unknown|level"):
        call()  # type: ignore[operator]


def test_cost_per_pass_interval_covers_the_true_cost_over_pass_rate() -> None:
    # Pass rate 0.4, lognormal cost with mu 0 and sigma 0.5: the cost per pass is exp(0.125) / 0.4.
    truth = math.exp(0.125) / 0.4
    rng = np.random.default_rng(0)
    hits = 0
    runs = 300
    for i in range(runs):
        passed = (rng.random(120) < 0.4).astype(float)
        costs = rng.lognormal(0.0, 0.5, 120)
        hits += cost_per_pass(passed, costs).interval(0.9, n_draws=4000, seed=i).contains(truth)
    assert 0.84 <= hits / runs <= 0.96


def test_cost_per_pass_draws_follow_the_ratio_and_work_in_compare() -> None:
    a = cost_per_pass([1, 1, 1, 0] * 25, np.random.default_rng(1).lognormal(1.0, 0.3, 100))
    b = cost_per_pass([1, 0, 0, 0] * 25, np.random.default_rng(2).lognormal(1.0, 0.3, 100))
    # Same mean cost, pass rate 0.75 against 0.25: a is about three times cheaper per pass.
    da, db = a.sample(50_000, seed=1), b.sample(50_000, seed=7)
    assert float(np.median(db / da)) == pytest.approx(3.0, rel=0.15)
    result = compare(np.log(da), np.log(db), margin=math.log(1.1))
    assert result.p_b_greater > 0.99


def test_the_token_ratio_equals_the_paired_t_interval_on_a_complete_design() -> None:
    rng = np.random.default_rng(3)
    task_effect = rng.normal(5.0, 1.0, 12)
    a = np.exp(task_effect + 0.0 + rng.normal(0.0, 0.2, 12))
    b = np.exp(task_effect + 0.5 + rng.normal(0.0, 0.2, 12))
    models = ["a"] * 12 + ["b"] * 12
    tasks = [f"t{i}" for i in range(12)] * 2
    fit = fit_tokens(models, tasks, np.concatenate([a, b]))
    d = np.log(b) - np.log(a)
    q = _accel.t_ppf(np.array([0.975]), 11.0)[0]
    half = q * d.std(ddof=1) / math.sqrt(12)
    interval = fit.log_ratio("b", "a")
    assert fit.df == 11
    assert 0.5 * (interval.lower + interval.upper) == pytest.approx(d.mean(), abs=1e-9)
    assert interval.upper - d.mean() == pytest.approx(half, rel=1e-9)
    assert fit.p_fewer("a", "b") > 0.99
    assert (
        fit.ratio("b", "a").contains(math.exp(0.5))
        or abs(math.log(fit.ratio("b", "a").lower) - 0.5) < 0.4
    )


def test_the_token_ratio_interval_covers_with_missing_cells_and_unequal_task_sets() -> None:
    rng = np.random.default_rng(4)
    hits = 0
    runs = 400
    for _ in range(runs):
        effect = rng.normal(5.0, 1.0, 30)
        rows: list[tuple[str, str, float]] = []
        for j in range(30):
            for model, shift in (("a", 0.0), ("b", 0.3), ("c", -0.2)):
                if rng.random() < 0.35:
                    continue
                rows.append(
                    (model, f"t{j}", float(np.exp(effect[j] + shift + rng.normal(0.0, 0.3))))
                )
        fit = fit_tokens([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
        hits += fit.log_ratio("b", "c", 0.9).contains(0.5)
    assert 0.85 <= hits / runs <= 0.95


def test_the_effects_are_not_penalized_for_the_tasks_a_model_was_given() -> None:
    # Model b is only given the long tasks, model a is given all of them. Same ratio per task.
    models: list[str] = []
    tasks: list[str] = []
    tokens: list[float] = []
    for j in range(20):
        base = 100.0 * (j + 1)
        models.append("a")
        tasks.append(f"t{j}")
        tokens.append(base)
        if j >= 10:
            models.append("b")
            tasks.append(f"t{j}")
            tokens.append(base * 2.0)
    fit = fit_tokens(
        models,
        tasks,
        np.array(tokens) * np.exp(np.random.default_rng(5).normal(0, 1e-3, len(tokens))),
    )
    assert fit.ratio("b", "a").lower == pytest.approx(2.0, rel=0.01)
    assert (
        float(np.mean([t for m, t in zip(models, tokens, strict=True) if m == "b"]))
        > 3.0 * float(np.mean([t for m, t in zip(models, tokens, strict=True) if m == "a"])) / 2.0
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: cost_per_pass([1, 0], [1.0, -1.0]),
        lambda: cost_per_pass([1], [1.0]),
        lambda: cost_per_pass([1, 2], [1.0, 2.0]),
        lambda: cost_per_pass([1, 0, 1], [1.0, 1.0, 1.0]),
        lambda: fit_tokens(["a", "b"], ["t", "u"], [10.0, 20.0]),
        lambda: fit_tokens(["a", "a"], ["t", "u"], [10.0, 0.0]),
        lambda: fit_tokens(["a", "b", "a"], ["t", "t", "u"], [1.0, 2.0, 3.0]),
        lambda: fit_tokens(["a", "b", "a", "b"], ["t", "t", "u", "u"], [1.0, 2.0, 3.0, 4.0]).ratio(
            "a", "zzz"
        ),
    ],
)
def test_bad_cost_and_token_input_is_refused(call: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"same length|between 0 and 1|above 0|linked|degrees|unknown|all equal|at least",
    ):
        call()  # type: ignore[operator]
