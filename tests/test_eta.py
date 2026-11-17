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
    with pytest.raises(ValueError, match="unknown parameters"):
        model.predict({"fies": 3.0})


def test_updating_with_each_new_run_equals_refitting_from_all_runs() -> None:
    duration, files, gb = _runs(seed=11, n=60)
    first = 25
    model = fit_duration(duration[:first], {"files": files[:first], "gb": gb[:first]})
    for i in range(first, 60):
        model = model.update(duration[i], {"files": files[i], "gb": gb[i]})
    batch = fit_duration(duration, {"files": files, "gb": gb})
    assert model.n == batch.n == 60
    np.testing.assert_allclose(model.coef_mean, batch.coef_mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(model.coef_cov, batch.coef_cov, rtol=1e-9, atol=1e-12)
    assert model.noise_shape == batch.noise_shape
    assert model.noise_rate == pytest.approx(batch.noise_rate, rel=1e-9)
    new = {"files": 800.0, "gb": 12.0}
    a, b = model.predict(new, seed=1), batch.predict(new, seed=1)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9)
    assert a.interval.lower == pytest.approx(b.interval.lower, rel=1e-9)
    assert a.interval.upper == pytest.approx(b.interval.upper, rel=1e-9)


def test_update_returns_a_new_model_and_checks_its_input() -> None:
    duration, files, _ = _runs(seed=12, n=20)
    model = fit_duration(duration, {"files": files})
    newer = model.update(duration[0], {"files": files[0]})
    assert newer is not model
    assert model.n == 20
    assert newer.n == 21
    with pytest.raises(ValueError, match="duration"):
        model.update(0.0, {"files": 10.0})
    with pytest.raises(ValueError, match="missing"):
        model.update(5.0, {})
    with pytest.raises(ValueError, match="parameters"):
        model.update(5.0, {"files": -1.0})


def test_update_works_with_no_parameters() -> None:
    model = fit_duration([10.0, 12.0, 9.0], {})
    batch = fit_duration([10.0, 12.0, 9.0, 11.0], {})
    updated = model.update(11.0, {})
    np.testing.assert_allclose(updated.coef_mean, batch.coef_mean, rtol=1e-12)


def test_unknown_parameters_give_a_wider_interval_from_the_known_ones() -> None:
    duration, files, gb = _runs(seed=21, n=200)
    model = fit_duration(duration, {"files": files, "gb": gb})
    both = model.predict({"files": 2000.0, "gb": 25.0}, seed=1)
    files_only = model.predict({"files": 2000.0}, seed=1)
    nothing = model.predict({}, seed=1)
    width = {
        k: v.interval.upper / v.interval.lower
        for k, v in (("both", both), ("files", files_only), ("none", nothing))
    }
    # Each parameter left out widens the interval.
    assert width["both"] < width["files"] < width["none"]
    # The interval from the known parameter still covers the true mean duration here.
    truth = 0.2 * 2000.0 + 5.0 * 25.0
    assert files_only.interval.contains(truth)
    assert "unknown before the run: gb" in files_only.assumption
    assert "unknown" not in both.assumption


def test_interval_with_unknown_parameters_covers_new_runs() -> None:
    train, files, gb = _runs(seed=22, n=300)
    model = fit_duration(train, {"files": files, "gb": gb})
    new, new_files, _ = _runs(seed=23, n=300)
    hits = 0
    for i in range(100):
        pred = model.predict({"files": float(new_files[i])}, level=0.9, seed=i)
        hits += pred.interval.contains(float(new[i]))
    # The nominal rate is 90%.
    assert 0.8 <= hits / 100 <= 0.98


def test_update_keeps_what_unknown_parameter_predictions_need() -> None:
    duration, files, gb = _runs(seed=24, n=60)
    model = fit_duration(duration[:30], {"files": files[:30], "gb": gb[:30]})
    for i in range(30, 60):
        model = model.update(duration[i], {"files": files[i], "gb": gb[i]})
    batch = fit_duration(duration, {"files": files, "gb": gb})
    a, b = model.predict({"files": 900.0}, seed=2), batch.predict({"files": 900.0}, seed=2)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-8)
    assert a.interval.upper == pytest.approx(b.interval.upper, rel=1e-8)
