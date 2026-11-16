import math

import numpy as np
import pytest

from statsx.eta import fit_duration


def _runs(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # True model: 0.2 s per file plus 5 s per GB, times log-normal noise (sd 0.2 on log scale).
    rng = np.random.default_rng(seed)
    files = rng.integers(10, 5000, size=n).astype(float)
    gb = rng.uniform(0.1, 50.0, size=n)
    duration = (0.2 * files + 5.0 * gb) * np.exp(rng.normal(0.0, 0.2, size=n))
    return duration, files, gb


def test_fit_learns_duration_from_input_parameters() -> None:
    rng = np.random.default_rng(0)
    files = rng.integers(10, 5000, size=300).astype(float)
    duration = 0.2 * files * np.exp(rng.normal(0.0, 0.2, size=300))
    model = fit_duration(duration, {"files": files})
    assert model.n == 300
    assert model.parameters == ("files",)
    # duration is close to proportional to files: the log-log slope is near 1
    assert model.coefficients["files"] == pytest.approx(1.0, abs=0.05)


def test_predicted_duration_has_a_credible_interval_that_covers_new_runs() -> None:
    duration, files, gb = _runs(1, 400)
    model = fit_duration(duration, {"files": files, "gb": gb})
    new_duration, new_files, new_gb = _runs(2, 200)
    covered = 0
    for d, f, g in zip(new_duration, new_files, new_gb, strict=True):
        pred = model.predict({"files": f, "gb": g}, level=0.9)
        assert pred.interval.lower < pred.estimate < pred.interval.upper
        covered += pred.interval.contains(d)
    # nominal 0.90 for new runs from the same process
    assert 0.82 <= covered / 200 <= 0.97


def test_more_files_predicts_longer() -> None:
    duration, files, _ = _runs(3, 200)
    model = fit_duration(duration, {"files": files})
    assert model.predict({"files": 4000}).estimate > 5 * model.predict({"files": 200}).estimate


def test_fit_with_no_parameters_predicts_the_typical_duration() -> None:
    durations = np.exp(np.random.default_rng(4).normal(math.log(60.0), 0.1, size=100))
    pred = fit_duration(durations, {}).predict({})
    assert pred.estimate == pytest.approx(60.0, rel=0.05)


def test_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="durations"):
        fit_duration([1.0, 0.0], {"files": [1, 2]})
    with pytest.raises(ValueError, match="values for"):
        fit_duration([1.0, 2.0], {"files": [1]})
    model = fit_duration([1.0, 2.0, 3.0], {"files": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing"):
        model.predict({})
