import math

import numpy as np
import pytest

from statsx.progress import fit_progress


def _runs(seed: int, n: int) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    """Tasks whose progress is (elapsed / duration) ** gamma with gamma near 0.7: they start slow."""
    rng = np.random.default_rng(seed)
    durations = np.exp(rng.normal(math.log(600.0), 0.5, n))
    gammas = np.exp(rng.normal(math.log(0.7), 0.15, n))
    reports: list[list[tuple[float, float]]] = []
    for d, g in zip(durations, gammas, strict=True):
        run: list[tuple[float, float]] = []
        for u in np.linspace(0.1, 0.9, 9):
            f = min(0.999, (u**g) * math.exp(rng.normal(0.0, 0.03)))
            run.append((float(u * d), float(f)))
        reports.append(run)
    return durations, reports


def _new_task(rng: np.random.Generator) -> tuple[float, float, float]:
    """Return elapsed, fraction done and the true time left of a new task."""
    d = math.exp(rng.normal(math.log(600.0), 0.5))
    g = math.exp(rng.normal(math.log(0.7), 0.15))
    u = rng.uniform(0.1, 0.9)
    f = min(0.999, (u**g) * math.exp(rng.normal(0.0, 0.03)))
    return u * d, f, d - u * d


def test_eta_from_progress_beats_the_constant_rate_guess_and_covers_the_truth() -> None:
    durations, reports = _runs(seed=1, n=60)
    model = fit_progress(durations, reports)
    rng = np.random.default_rng(2)
    hits = 0
    learned: list[float] = []
    constant: list[float] = []
    for i in range(200):
        elapsed, fraction, left = _new_task(rng)
        estimate = model.remaining(elapsed, fraction, level=0.9, seed=i)
        hits += estimate.interval.contains(left)
        learned.append(float(abs(estimate.estimate - left) / left))
        constant.append(float(abs(elapsed * (1 - fraction) / fraction - left) / left))
    assert model.n_runs == 60
    assert model.n_checkpoints == 540
    assert 0.8 <= hits / 200 <= 0.97
    assert np.median(learned) < 0.7 * np.median(constant)


def test_the_interval_narrows_as_the_task_progresses() -> None:
    durations, reports = _runs(seed=3, n=60)
    model = fit_progress(durations, reports)
    early = model.remaining(60.0, 0.15)
    late = model.remaining(540.0, 0.9)
    assert late.estimate < early.estimate
    early_width = early.interval.upper / early.interval.lower
    late_width = late.interval.upper / late.interval.lower
    assert late_width <= early_width * 1.2
    assert late.interval.lower > 0.0
    assert "constant-rate" in late.assumption
    assert late.n == 540


def test_same_seed_gives_the_same_interval() -> None:
    durations, reports = _runs(seed=4, n=30)
    model = fit_progress(durations, reports)
    assert model.remaining(200.0, 0.4, seed=7) == model.remaining(200.0, 0.4, seed=7)


def test_unusable_reports_are_skipped_and_bad_input_is_refused() -> None:
    durations = [100.0, 200.0, 150.0]
    reports = [
        [(10.0, 0.1), (50.0, 0.5), (150.0, 0.6), (20.0, 1.0)],  # last two are unusable
        [(40.0, 0.2), (120.0, 0.7)],
        [],
    ]
    model = fit_progress(durations, reports)
    assert model.n_runs == 2
    assert model.n_checkpoints == 4
    with pytest.raises(ValueError, match="usable"):
        fit_progress([100.0], [[(200.0, 0.5)]])
    with pytest.raises(ValueError, match="runs of checkpoints"):
        fit_progress([100.0, 200.0], [[(10.0, 0.1)]])
    with pytest.raises(ValueError, match="durations"):
        fit_progress([0.0], [[(10.0, 0.1)]])
    with pytest.raises(ValueError, match="fraction"):
        model.remaining(10.0, 1.0)
    with pytest.raises(ValueError, match="elapsed"):
        model.remaining(0.0, 0.5)
