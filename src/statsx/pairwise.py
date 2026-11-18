"""Rank systems from pairwise results: Bradley-Terry and Davidson, with item-level resampling (CO-2).

Each row says that system ``a`` was compared with system ``b`` on one item (a prompt, a task, a
match) and won, lost or tied. The model gives each system a strength ``theta`` on the log scale.
With ``d = theta_a - theta_b``:

- ``ties="half"``: the Bradley-Terry model, ``P(a wins) = sigmoid(d)``. A tie counts as half a win
  and half a loss.
- ``ties="davidson"``: Davidson's model, with ``Z = e^{d/2} + e^{-d/2} + nu`` and
  ``P(a wins) = e^{d/2} / Z``, ``P(b wins) = e^{-d/2} / Z`` and ``P(tie) = nu / Z``. The tie
  parameter ``nu`` is estimated. With ``nu = 0`` it is the Bradley-Terry model.

The fit is the maximum of the log likelihood with a Gaussian prior ``theta ~ Normal(0, s^2)``
(``prior_sd``), found by Newton steps. The prior keeps the strengths finite when a system never
loses; the strengths are reported with mean 0.

Intervals come from resampling the items, not the rows. All rows of one item are drawn together,
because rows of one item are not independent (the same prompt, judged for several pairs). Treating
the rows as independent gives intervals that are too narrow. A bootstrap interval is not a
credible interval: ``p_better`` is the share of resamples in which ``a`` is above ``b``.

References:
    Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The
    method of paired comparisons. *Biometrika* 39(3/4):324-345. https://doi.org/10.2307/2334029.

    Davidson, R. R. (1970). On extending the Bradley-Terry model to accommodate ties in paired
    comparison experiments. *Journal of the American Statistical Association* 65(329):317-328.
    https://doi.org/10.1080/01621459.1970.10481082.

    Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*
    7(1):1-26. https://doi.org/10.1214/aos/1176344552.

    Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal
    Statistical Society B* 69(3):369-390. https://doi.org/10.1111/j.1467-9868.2007.00593.x.
    Resampling whole clusters.

    Chiang, W.-L., et al. (2024). Chatbot Arena: an open platform for evaluating LLMs by human
    preference. *ICML*, PMLR 235. https://proceedings.mlr.press/v235/chiang24b.html. Bradley-Terry
    with bootstrap intervals for language models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from statsx.result import Interval

Ties = Literal["davidson", "half"]
_NEWTON_STEPS = 60
_DECREMENT = 1e-10
_TIE = 0.5
_MIN_STEP = 1e-8
_TAU_SD = 3.0  # the prior sd of log(nu), the tie parameter
_OUTCOMES = {"a": 1.0, "win": 1.0, "b": 0.0, "loss": 0.0, "tie": 0.5}


@dataclass(frozen=True, slots=True)
class Comparisons:
    """Pairwise results, one row per comparison.

    Args:
        systems: The system names, in the order of their index.
        first: The index of the first system of each row.
        second: The index of the second system of each row.
        outcome: 1 if the first won, 0 if the second won, 0.5 for a tie.
        item: The item of each row, as a whole number; rows with the same item are resampled
            together.
    """

    systems: tuple[str, ...]
    first: npt.NDArray[np.int64]
    second: npt.NDArray[np.int64]
    outcome: npt.NDArray[np.float64]
    item: npt.NDArray[np.int64]

    @classmethod
    def from_long(
        cls,
        first: Sequence[str],
        second: Sequence[str],
        outcome: Sequence[str | float],
        *,
        item: Sequence[object] | None = None,
    ) -> Comparisons:
        """Build the table from columns of the same length.

        ``outcome`` is ``"a"`` or ``"win"`` (the first won), ``"b"`` or ``"loss"``, ``"tie"``, or
        the numbers 1, 0 and 0.5. ``item`` names the item of each row; by default every row is its
        own item.

        Raises:
            ValueError: If the columns differ in length or are empty, a system is compared with
                itself, or an outcome is not one of the values above.
        """
        n = len(first)
        if n == 0 or len(second) != n or len(outcome) != n or (item is not None and len(item) != n):
            raise ValueError(
                "first, second, outcome and item must be non-empty and of the same length"
            )
        names = tuple(dict.fromkeys([*first, *second]))
        index = {name: i for i, name in enumerate(names)}
        result = np.empty(n, dtype=np.float64)
        for r, value in enumerate(outcome):
            key = value.lower() if isinstance(value, str) else value
            if isinstance(key, str):
                if key not in _OUTCOMES:
                    raise ValueError(
                        f"outcome must be a, b, tie, win, loss, 1, 0 or 0.5, got {value!r}"
                    )
                result[r] = _OUTCOMES[key]
            elif key in (0, 0.5, 1):
                result[r] = float(key)
            else:
                raise ValueError(
                    f"outcome must be a, b, tie, win, loss, 1, 0 or 0.5, got {value!r}"
                )
        a = np.array([index[s] for s in first], dtype=np.int64)
        b = np.array([index[s] for s in second], dtype=np.int64)
        if np.any(a == b):
            raise ValueError("a system cannot be compared with itself")
        if item is None:
            items = np.arange(n, dtype=np.int64)
        else:
            labels = {name: i for i, name in enumerate(dict.fromkeys(item))}
            items = np.array([labels[v] for v in item], dtype=np.int64)
        return cls(names, a, b, result, items)


@dataclass(frozen=True, slots=True)
class _Terms:
    """Log likelihood of each row and its derivatives with respect to ``d`` and ``tau``."""

    value: npt.NDArray[np.float64]
    d1: npt.NDArray[np.float64]
    dd: npt.NDArray[np.float64]
    t1: npt.NDArray[np.float64]
    tt: npt.NDArray[np.float64]
    dt: npt.NDArray[np.float64]


def _terms(
    d: npt.NDArray[np.float64], y: npt.NDArray[np.float64], tau: float, ties: Ties
) -> _Terms:
    zero = np.zeros_like(d)
    if ties == "half":
        p = 1.0 / (1.0 + np.exp(-np.clip(d, -35.0, 35.0)))
        value = y * -np.logaddexp(0.0, -d) + (1.0 - y) * -np.logaddexp(0.0, d)
        return _Terms(value, y - p, -p * (1.0 - p), zero, zero, zero)
    nu = math.exp(tau)
    a = np.clip(d / 2.0, -30.0, 30.0)
    z = 2.0 * np.cosh(a) + nu
    sinh, cosh = np.sinh(a), np.cosh(a)
    win, loss, tie = (y == 1.0), (y == 0.0), (y == _TIE)
    numerator = np.where(win, a, np.where(loss, -a, tau))
    d1 = np.where(win, 0.5, np.where(loss, -0.5, 0.0)) - sinh / z
    dd = -(cosh / 2.0 / z - sinh**2 / z**2)
    t1 = np.where(tie, 1.0, 0.0) - nu / z
    tt = -(nu / z - nu**2 / z**2)
    dt = sinh * nu / z**2
    return _Terms(numerator - np.log(z), d1, dd, t1, tt, dt)


def _newton(
    data: Comparisons,
    weight: npt.NDArray[np.float64],
    ties: Ties,
    prior_sd: float,
    start: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the maximum of the weighted penalized log likelihood: strengths, then ``tau``."""
    k = len(data.systems)
    use_tau = ties == "davidson"
    size = k + int(use_tau)
    x = start.copy()
    precision = 1.0 / prior_sd**2

    def objective(v: npt.NDArray[np.float64]) -> float:
        d = v[data.first] - v[data.second]
        t = _terms(d, data.outcome, float(v[k]) if use_tau else 0.0, ties)
        penalty = 0.5 * precision * float(v[:k] @ v[:k])
        if use_tau:
            penalty += 0.5 * float(v[k]) ** 2 / _TAU_SD**2
        return float(np.sum(weight * t.value) - penalty)

    for _ in range(_NEWTON_STEPS):
        d = x[data.first] - x[data.second]
        t = _terms(d, data.outcome, float(x[k]) if use_tau else 0.0, ties)
        grad = np.zeros(size)
        grad[:k] = np.bincount(data.first, weights=weight * t.d1, minlength=k) - np.bincount(
            data.second, weights=weight * t.d1, minlength=k
        )
        grad[:k] -= precision * x[:k]
        neg_hess = np.zeros((size, size))
        wdd = weight * t.dd
        for rows, cols, sign in (
            (data.first, data.first, 1.0),
            (data.second, data.second, 1.0),
            (data.first, data.second, -1.0),
            (data.second, data.first, -1.0),
        ):
            flat = np.bincount(rows * k + cols, weights=sign * wdd, minlength=k * k)
            neg_hess[:k, :k] -= flat.reshape(k, k)
        neg_hess[:k, :k] += precision * np.eye(k)
        if use_tau:
            grad[k] = float(np.sum(weight * t.t1))
            wdt = weight * t.dt
            cross = np.bincount(data.first, weights=wdt, minlength=k) - np.bincount(
                data.second, weights=wdt, minlength=k
            )
            neg_hess[:k, k] = -cross
            neg_hess[k, :k] = -cross
            neg_hess[k, k] = (
                -float(np.sum(weight * t.tt)) + 1.0 / 9.0
            )  # a Normal(0, 3^2) prior on tau
            grad[k] -= x[k] / _TAU_SD**2
        step = np.linalg.solve(neg_hess, grad)
        scale, base = 1.0, objective(x)
        while scale > _MIN_STEP and objective(x + scale * step) < base - 1e-13 * (1.0 + abs(base)):
            scale /= 2.0
        x = x + scale * step
        if float(grad @ step) < _DECREMENT * (1.0 + float(weight.sum())):
            break
    else:
        raise ValueError("Newton's method did not converge")
    return x


def _centered(x: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.float64]:
    out = x.copy()
    out[:k] -= out[:k].mean()
    return out


@dataclass(frozen=True, slots=True)
class BradleyTerryFit:
    """A fitted Bradley-Terry or Davidson model with bootstrap draws over items.

    Args:
        systems: The system names.
        strength: The fitted strength of each system, with mean 0.
        tie_parameter: ``nu`` of Davidson's model (0 for ``ties="half"``).
        ties: ``"davidson"`` or ``"half"``.
        prior_sd: The prior sd of a strength.
        n_comparisons: The number of rows.
        n_items: The number of items resampled.
        draws: The strengths of each bootstrap resample, shape (resamples, systems), mean 0.
    """

    systems: tuple[str, ...]
    strength: npt.NDArray[np.float64]
    tie_parameter: float
    ties: Ties
    prior_sd: float
    n_comparisons: int
    n_items: int
    draws: npt.NDArray[np.float64]

    def _index(self, name: str) -> int:
        if name not in self.systems:
            raise ValueError(f"unknown system {name!r}")
        return self.systems.index(name)

    def strength_interval(self, system: str, level: float = 0.95) -> Interval:
        """Return the bootstrap percentile interval of the strength of ``system``."""
        return _percentile(self.draws[:, self._index(system)], level)

    def difference_interval(self, a: str, b: str, level: float = 0.95) -> Interval:
        """Return the bootstrap percentile interval of ``strength(a) - strength(b)``."""
        diff = self.draws[:, self._index(a)] - self.draws[:, self._index(b)]
        return _percentile(diff, level)

    def p_better(self, a: str, b: str) -> float:
        """Return the share of item resamples in which ``a`` is stronger than ``b``."""
        diff = self.draws[:, self._index(a)] - self.draws[:, self._index(b)]
        return float(np.mean(diff > 0.0))

    def probabilities(self, a: str, b: str) -> tuple[float, float, float]:
        """Return the fitted probabilities that ``a`` wins, ``b`` wins and they tie."""
        d = float(self.strength[self._index(a)] - self.strength[self._index(b)])
        if self.ties == "half":
            p = 1.0 / (1.0 + math.exp(-d))
            return p, 1.0 - p, 0.0
        z = 2.0 * math.cosh(d / 2.0) + self.tie_parameter
        return math.exp(d / 2.0) / z, math.exp(-d / 2.0) / z, self.tie_parameter / z


def _percentile(values: npt.NDArray[np.float64], level: float) -> Interval:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    lo, hi = np.quantile(values, [0.5 - level / 2.0, 0.5 + level / 2.0])
    return Interval(float(lo), float(hi), level)


def fit_bradley_terry(
    data: Comparisons,
    *,
    ties: Ties = "davidson",
    prior_sd: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> BradleyTerryFit:
    """Fit strengths to pairwise results and resample the items for intervals.

    Args:
        data: The comparisons.
        ties: ``"davidson"`` (the default, the tie probability is modelled) or ``"half"`` (a tie is
            half a win).
        prior_sd: The sd of the Gaussian prior on a strength, in log-odds units.
        n_bootstrap: The number of resamples of the items, at least 2.
        seed: The seed of the resampling.

    Raises:
        ValueError: If ``prior_sd`` is not positive, ``n_bootstrap`` is below 2, ``ties`` is
            unknown, or Newton's method does not converge.
    """
    if ties not in ("davidson", "half"):
        raise ValueError(f"ties must be 'davidson' or 'half', got {ties!r}")
    if not prior_sd > 0.0 or n_bootstrap < 2:  # noqa: PLR2004
        raise ValueError("prior_sd must be positive and n_bootstrap at least 2")
    k = len(data.systems)
    start = np.zeros(k + int(ties == "davidson"))
    full = _centered(_newton(data, np.ones(data.outcome.size), ties, prior_sd, start), k)
    n_items = int(data.item.max()) + 1
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n_items, n_items), minlength=n_items).astype(
            np.float64
        )
        sample = _newton(data, counts[data.item], ties, prior_sd, full)
        draws[b] = _centered(sample, k)[:k]
    nu = math.exp(float(full[k])) if ties == "davidson" else 0.0
    return BradleyTerryFit(
        data.systems, full[:k], nu, ties, prior_sd, int(data.outcome.size), n_items, draws
    )
