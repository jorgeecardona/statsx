import math
from collections.abc import Callable

import numpy as np
import pytest

from statsx.evals import Responses, calibrate_infra, fit_rasch, fit_tokens

N_TASKS = 30
SHIFT = math.log(1.2)  # the second infra uses 20% more tokens
MODEL_RATIO = 1.5  # model A uses 1.5 times the tokens of model B


def _token_rows(seed: int) -> tuple[list[str], list[str], list[str], np.ndarray]:
    """A runs on g1 only, B on g2 only, C on both: C is the bridge."""
    rng = np.random.default_rng(seed)
    task_effect = rng.normal(5.0, 1.0, N_TASKS)
    effect = {"A": math.log(MODEL_RATIO), "B": 0.0, "C": 0.3}
    cells = [("A", "g1"), ("B", "g2"), ("C", "g1"), ("C", "g2")]
    models: list[str] = []
    infras: list[str] = []
    tasks: list[str] = []
    tokens: list[float] = []
    for model, infra in cells:
        for t in range(N_TASKS):
            models.append(model)
            infras.append(infra)
            tasks.append(f"t{t}")
            mean = effect[model] + task_effect[t] + (SHIFT if infra == "g2" else 0.0)
            tokens.append(math.exp(rng.normal(mean, 0.25)))
    return models, tasks, infras, np.array(tokens)


def test_the_model_ratio_is_recovered_and_the_90_percent_interval_covers_it() -> None:
    covered = covered_infra = 0
    for seed in range(100):
        models, tasks, infras, tokens = _token_rows(seed)
        fit = fit_tokens(models, tasks, tokens, infra=infras)
        r = fit.ratio("A", "B", level=0.9)
        covered += r.lower <= MODEL_RATIO <= r.upper
        e = fit.infra_effect("g2", "g1", level=0.9)
        covered_infra += e.lower <= 1.2 <= e.upper
    assert covered >= 85
    assert covered_infra >= 85


def test_without_the_infra_effect_the_ratio_is_off_by_the_infra_shift() -> None:
    models, tasks, infras, tokens = _token_rows(0)
    naive = fit_tokens(models, tasks, tokens).ratio("A", "B")
    adjusted = fit_tokens(models, tasks, tokens, infra=infras).ratio("A", "B")
    # A ran on g1 and B on g2, so the naive ratio is 1.5 / 1.2
    assert math.sqrt(naive.lower * naive.upper) == pytest.approx(MODEL_RATIO / 1.2, abs=0.07)
    assert math.sqrt(adjusted.lower * adjusted.upper) == pytest.approx(MODEL_RATIO, abs=0.1)


def test_the_reference_infra_has_effect_one_against_itself_and_the_reverse_is_the_inverse() -> None:
    models, tasks, infras, tokens = _token_rows(1)
    fit = fit_tokens(models, tasks, tokens, infra=infras)
    one = fit.infra_effect("g1", "g1")
    assert one.lower == pytest.approx(1.0) and one.upper == pytest.approx(1.0)
    forward = fit.infra_log_effect("g2", "g1")
    backward = fit.infra_log_effect("g1", "g2")
    assert forward.lower == pytest.approx(-backward.upper)


def test_models_that_never_share_an_infra_are_refused_and_named() -> None:
    models, tasks, infras, tokens = _token_rows(2)
    keep = [i for i, m in enumerate(models) if m != "C"]
    args = ([models[i] for i in keep], [tasks[i] for i in keep], tokens[keep])
    with pytest.raises(ValueError, match=r"\[A\].*\[B\]"):
        fit_tokens(*args, infra=[infras[i] for i in keep])
    data = Responses.from_long(args[0], args[1], np.ones(len(keep)), [infras[i] for i in keep])
    with pytest.raises(ValueError, match=r"\[A\].*\[B\]"):
        fit_rasch(data)


def test_one_infra_for_every_row_changes_nothing() -> None:
    models, tasks, _, tokens = _token_rows(3)
    plain = fit_tokens(models, tasks, tokens)
    one = fit_tokens(models, tasks, tokens, infra=["g1"] * len(models))
    assert one.noise_sd == pytest.approx(plain.noise_sd)
    a, b = plain.ratio("A", "B"), one.ratio("A", "B")
    assert (a.lower, a.upper) == pytest.approx((b.lower, b.upper))


def _rasch_rows(seed: int) -> tuple[Responses, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    theta = {"A": rng.normal(0, 1.5), "B": rng.normal(0, 1.5), "C": rng.normal(0, 1.5)}
    difficulty = rng.normal(0, 1.5, 80)
    shift = float(rng.normal(0, 1.0))  # g2 against g1
    models: list[str] = []
    tasks: list[str] = []
    infras: list[str] = []
    passed: list[float] = []
    for model, infra in (("A", "g1"), ("B", "g2"), ("C", "g1"), ("C", "g2")):
        for t in range(80):
            eta = theta[model] - difficulty[t] + (shift if infra == "g2" else 0.0)
            models.append(model)
            tasks.append(f"t{t}")
            infras.append(infra)
            passed.append(float(rng.random() < 1.0 / (1.0 + math.exp(-eta))))
    return (
        Responses.from_long(models, tasks, passed, infras),
        np.array([theta["A"] - theta["B"]]),
        shift,
    )


def test_the_rasch_infra_shift_and_the_ability_gap_are_covered_at_close_to_the_stated_rate() -> (
    None
):
    shift_covered = gap_covered = 0
    for seed in range(100):
        data, gap, shift = _rasch_rows(seed)
        fit = fit_rasch(data)
        s = fit.infra_shift_difference("g2", "g1", level=0.9)
        shift_covered += s.lower <= shift <= s.upper
        g = fit.ability_difference("A", "B", level=0.9)
        gap_covered += g.lower <= gap[0] <= g.upper
    assert shift_covered >= 80
    assert gap_covered >= 80


def test_rasch_abilities_are_read_on_the_reference_infra() -> None:
    data, _, _ = _rasch_rows(5)
    fit = fit_rasch(data)
    assert fit.infras == ("g1", "g2")
    assert fit.infra_shift[0] == 0.0
    base = fit.p_solve("C", "t0")
    on_g2 = fit.p_solve("C", "t0", infra="g2")
    assert (on_g2 > base) == (fit.infra_shift[1] > 0.0)
    assert fit.infra_shift_difference("g1", "g1").lower == 0.0


def test_the_bridge_between_two_infras_is_the_model_that_ran_on_both() -> None:
    models, _, infras, _ = _token_rows(0)
    (bridge,) = calibrate_infra(models, infras)
    assert (bridge.infra_a, bridge.infra_b) == ("g1", "g2")
    assert bridge.models == ("C",)
    assert bridge.n_rows == 2 * N_TASKS
    (none,) = calibrate_infra(["A", "B"], ["g1", "g2"])
    assert none.models == () and none.n_rows == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: Responses.from_long(["a"], ["t"], [1.0], infra=["g", "h"]),
        lambda: calibrate_infra(["a"], []),
        lambda: fit_tokens(["a", "b"], ["t", "t"], [1.0, 2.0], infra=["g"]),
        lambda: fit_rasch(Responses.from_long(["a"], ["t"], [1.0], ["g"]), infra_sd=0.0),
    ],
)
def test_bad_input_is_refused(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match=r"."):
        call()


def test_an_unknown_infra_is_refused() -> None:
    models, tasks, infras, tokens = _token_rows(4)
    fit = fit_tokens(models, tasks, tokens, infra=infras)
    with pytest.raises(ValueError, match="unknown infra"):
        fit.infra_effect("g1", "zzz")
