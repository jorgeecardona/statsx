"""Decisions under uncertainty: a cost distribution, break-even points, value of information.

``simulate_cost`` (DC-1) draws the uncertain inputs of a cost model with a fixed seed and returns
the distribution of the cost. ``expected_value_of_information`` and ``break_even`` (DC-2) say how
much more information is worth and where a decision changes.

References:
    Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. *Journal of the American
    Statistical Association* 44(247):335-341. https://doi.org/10.1080/01621459.1949.10483310.

    Raiffa, H., & Schlaifer, R. (1961). *Applied Statistical Decision Theory*. Harvard University
    Press. The expected value of perfect information and the loss ratio of a two-action decision.

    Claxton, K. (1999). The irrelevance of inference: a decision-making approach to the
    stochastic evaluation of health care technologies. *Journal of Health Economics*
    18(3):341-364. https://doi.org/10.1016/S0167-6296(98)00039-3. Value of information from
    posterior draws of a utility.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from statsx.posterior import Quantiles, Sampler, credible_interval, quantiles
from statsx.result import Estimate, Interval


@dataclass(frozen=True, slots=True)
class CostDistribution:
    """The distribution of a cost from Monte Carlo draws of its uncertain inputs.

    Args:
        draws: Cost draws, one per Monte Carlo draw of the inputs.
        inputs: Names of the inputs that were uncertain.
        seed: The seed that gave these draws; the same seed gives the same draws.
    """

    draws: npt.NDArray[np.float64] = field(repr=False)
    inputs: tuple[str, ...]
    seed: int

    @property
    def n_draws(self) -> int:
        """Return the number of Monte Carlo draws."""
        return int(self.draws.size)

    def mean(self, level: float = 0.95) -> Estimate:
        """Return the expected cost with the interval of its Monte Carlo error."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        mean = float(self.draws.mean())
        half = _z(level) * float(self.draws.std(ddof=1)) / math.sqrt(self.n_draws)
        return Estimate(
            estimate=mean,
            interval=Interval(mean - half, mean + half, level),
            n=self.n_draws,
            assumption="independent Monte Carlo draws of the inputs; interval is the Monte Carlo "
            "error of the mean",
        )

    def interval(self, level: float = 0.9) -> Interval:
        """Return the central interval that holds ``level`` of the cost distribution."""
        return credible_interval(self.draws, level)

    def quantiles(self, probs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> Quantiles:
        """Return the quantiles of the cost."""
        return quantiles(self.draws, probs)

    def p_exceeds(self, limit: float) -> float:
        """Return the probability that the cost is above ``limit``."""
        return float(np.mean(self.draws > limit))


def _z(level: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + level / 2.0)


def simulate_cost(
    model: Callable[..., npt.ArrayLike],
    inputs: Mapping[str, Sampler | float],
    *,
    n_draws: int = 10_000,
    seed: int = 0,
) -> CostDistribution:
    """Return the distribution of ``model`` when its inputs are uncertain.

    Each input that is a distribution with a ``sample`` method (``Beta``, ``Gamma``, ``Normal``,
    ``LogNormal``) is drawn ``n_draws`` times, independently of the others; a plain number is a
    constant. ``model`` is called once with all inputs as keyword arguments, each an array of
    ``n_draws`` values (a constant is repeated), and returns an array of ``n_draws`` costs, so
    write it with NumPy operations. The input ``k`` in the order given uses seed ``seed + k``.

    Args:
        model: The cost model, a function of the input names.
        inputs: Input name to a distribution or a number.
        n_draws: Number of Monte Carlo draws, at least 2.
        seed: Seed of the draws.

    Raises:
        ValueError: If ``n_draws`` is below 2, there is no input, or the model does not return
            one finite cost per draw.
    """
    if n_draws < 2:  # noqa: PLR2004
        raise ValueError("n_draws must be at least 2")
    if not inputs:
        raise ValueError("need at least one input")
    values: dict[str, npt.NDArray[np.float64]] = {}
    uncertain: list[str] = []
    for k, (name, spec) in enumerate(inputs.items()):
        if isinstance(spec, Sampler):
            values[name] = spec.sample(n_draws, seed=seed + k)
            uncertain.append(name)
        else:
            values[name] = np.full(n_draws, float(spec))
    cost = np.asarray(model(**values), dtype=np.float64)
    if cost.shape != (n_draws,) or not np.all(np.isfinite(cost)):
        raise ValueError(f"the model must return {n_draws} finite costs, got shape {cost.shape}")
    return CostDistribution(cost, tuple(uncertain), seed)


@dataclass(frozen=True, slots=True)
class ValueOfInformation:
    """Expected value of perfect information for a choice between actions.

    Args:
        evpi: Expected gain from learning the uncertain inputs before choosing. At least 0.
        best_action: Index of the action with the highest expected utility now.
        expected_utility: Expected utility of each action now.
        n_draws: Number of draws behind the expectations.
    """

    evpi: float
    best_action: int
    expected_utility: tuple[float, ...]
    n_draws: int


def expected_value_of_information(utilities: npt.ArrayLike) -> ValueOfInformation:
    """Return the expected value of perfect information from draws of each action's utility.

    ``utilities[i, a]`` is the utility of action ``a`` when the uncertain inputs take their
    ``i``-th joint draw. EVPI = E[max over actions] - max over actions of E[utility]: what you
    would pay, at most, to learn the inputs before you choose. It is 0 when one action is best
    in every draw.

    Raises:
        ValueError: If ``utilities`` is not a finite array of shape (draws >= 2, actions >= 2).
    """
    u = np.asarray(utilities, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] < 2 or u.shape[1] < 2 or not np.all(np.isfinite(u)):  # noqa: PLR2004
        raise ValueError("utilities must be finite with shape (draws >= 2, actions >= 2)")
    expected = u.mean(axis=0)
    evpi = float(np.mean(u.max(axis=1)) - expected.max())
    return ValueOfInformation(
        evpi, int(np.argmax(expected)), tuple(float(v) for v in expected), u.shape[0]
    )


def break_even_probability(gain_if_right: float, loss_if_wrong: float) -> float:
    """Return the probability above which acting pays: loss / (gain + loss).

    For a two-action decision where acting gains ``gain_if_right`` when the state is good and
    loses ``loss_if_wrong`` when it is bad, acting has the higher expected utility when
    P(good) is above this value.

    Raises:
        ValueError: If the gain or the loss is not positive.
    """
    if not (gain_if_right > 0.0 and loss_if_wrong > 0.0):
        raise ValueError("gain_if_right and loss_if_wrong must be > 0")
    return loss_if_wrong / (gain_if_right + loss_if_wrong)


def break_even(
    net_benefit: Callable[[float], float],
    low: float,
    high: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Return the threshold in ``[low, high]`` where ``net_benefit`` changes sign, by bisection.

    Use it when the net benefit of an action is a function of a threshold (a price, a score
    cut-off, a probability): the root is the break-even point.

    Raises:
        ValueError: If ``net_benefit`` has the same sign at both ends, or ``low >= high``.
    """
    if not low < high:
        raise ValueError(f"need low < high, got {low} and {high}")
    f_low, f_high = net_benefit(low), net_benefit(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError(
            f"no sign change: net_benefit({low}) = {f_low}, net_benefit({high}) = {f_high}"
        )
    a, b = low, high
    for _ in range(max_iterations):
        mid = 0.5 * (a + b)
        f_mid = net_benefit(mid)
        if f_mid == 0.0 or 0.5 * (b - a) < tolerance:
            return mid
        if f_low * f_mid < 0.0:
            b = mid
        else:
            a, f_low = mid, f_mid
    return 0.5 * (a + b)
