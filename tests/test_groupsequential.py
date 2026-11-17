import numpy as np
import pytest

from statsx.groupsequential import GroupSequentialDesign, group_sequential_bounds


def test_pocock_constants_match_the_published_table() -> None:
    # Pocock (1977), two-sided 5%: 2.178, 2.289, 2.361, 2.413 for 2, 3, 4, 5 equally spaced looks.
    for looks, constant in ((2, 2.178), (3, 2.289), (4, 2.361), (5, 2.413)):
        design = group_sequential_bounds(looks, kind="pocock")
        assert isinstance(design, GroupSequentialDesign)
        assert design.z_bounds == pytest.approx((constant,) * looks, abs=6e-4)


def test_obrien_fleming_bounds_match_the_published_values_for_five_looks() -> None:
    # Jennison and Turnbull (2000), two-sided 5%, 5 looks: C = 2.040 and c_k = C * sqrt(5 / k).
    design = group_sequential_bounds(5, kind="obrien_fleming")
    assert design.z_bounds == pytest.approx((4.562, 3.226, 2.634, 2.281, 2.040), abs=6e-4)
    assert design.z_bounds[0] > design.z_bounds[-1] > 1.96
    one_look = group_sequential_bounds(1, kind="obrien_fleming")
    assert one_look.z_bounds[0] == pytest.approx(1.95996, abs=1e-4)


def test_the_error_spent_adds_up_to_alpha_and_the_designs_differ_in_when_they_spend_it() -> None:
    pocock = group_sequential_bounds(5, kind="pocock")
    obf = group_sequential_bounds(5, kind="obrien_fleming")
    for design in (pocock, obf):
        assert sum(design.alpha_spent) == pytest.approx(design.alpha, abs=1e-7)
        assert all(p > 0 for p in design.alpha_spent)
    # Pocock spends most at the first look; O'Brien-Fleming spends almost nothing there.
    assert pocock.alpha_spent[0] > 0.01
    assert obf.alpha_spent[0] < 1e-4
    assert obf.alpha_spent[-1] > pocock.alpha_spent[-1]
    assert obf.nominal_p[0] < 1e-5
    assert pocock.nominal_p[-1] < obf.nominal_p[-1] < 0.05


def test_a_simulation_of_the_stopping_rule_has_the_stated_error_rate() -> None:
    rng = np.random.default_rng(0)
    looks, runs, block = 4, 60000, 25
    steps = rng.normal(size=(runs, looks, block)).sum(axis=2)
    partial = np.cumsum(steps, axis=1)
    z = partial / np.sqrt(block * np.arange(1, looks + 1))
    for kind in ("pocock", "obrien_fleming"):
        bounds = np.array(group_sequential_bounds(looks, kind=kind).z_bounds)  # type: ignore[arg-type]
        stopped = np.any(np.abs(z) >= bounds, axis=1)
        assert float(np.mean(stopped)) == pytest.approx(0.05, abs=0.005)
    # Testing at 1.96 at every look instead is far above 5%.
    assert float(np.mean(np.any(np.abs(z) >= 1.96, axis=1))) > 0.1


def test_one_sided_designs_and_unequal_looks_are_supported() -> None:
    one = group_sequential_bounds(3, kind="obrien_fleming", two_sided=False)
    assert sum(one.alpha_spent) == pytest.approx(0.05, abs=1e-7)
    assert one.z_bounds[-1] > 1.6449  # above the fixed-sample one-sided value
    assert one.nominal_p[-1] < 0.05
    unequal = group_sequential_bounds(3, kind="pocock", times=[0.2, 0.5, 1.0])
    assert unequal.times == (0.2, 0.5, 1.0)
    assert sum(unequal.alpha_spent) == pytest.approx(0.05, abs=1e-7)
    assert unequal.alpha_spent[0] > unequal.alpha_spent[2] * 1.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_looks": 0},
        {"n_looks": 21},
        {"n_looks": 3, "alpha": 0.6},
        {"n_looks": 3, "kind": "haybittle"},
        {"n_looks": 3, "times": [0.5, 0.4, 1.0]},
        {"n_looks": 3, "times": [0.2, 0.5, 0.9]},
        {"n_looks": 3, "times": [0.5, 1.0]},
    ],
)
def test_bad_designs_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"n_looks|alpha|kind|times"):
        group_sequential_bounds(**kwargs)  # type: ignore[arg-type]
