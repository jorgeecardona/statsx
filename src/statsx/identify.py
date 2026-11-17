"""Pick the best arm and stop when it is certain, with an error guarantee (SQ-4).

``best_arm`` pulls every arm still in the race, and after each round removes the arms that are
clearly worse. It stops when one arm is left, or when every other arm is within ``epsilon`` of
the leader. With probability at least ``1 - delta`` it never removes the best arm, so:

- with ``epsilon = 0`` the arm it returns is the best arm;
- with ``epsilon > 0`` the arm it returns has a success probability within ``epsilon`` of the
  best.

The method is successive elimination (Even-Dar, Mannor & Mansour 2006), which removes an arm when
its empirical mean is more than twice the confidence radius below the best. This module writes that
rule as "upper end below the best lower end" and uses a time-uniform radius in place of their sum
over time steps (Jamieson et al. 2014 also avoid a union bound over time). Each arm has an
always-valid confidence interval for its success probability at level ``delta / K`` (see
``statsx.alwaysvalid``). An arm is removed when its upper end is below the largest lower end of
the arms in the race. The guarantee holds when all ``K`` intervals hold at every round at once,
which has probability at least ``1 - delta`` by the union bound. Because the intervals are valid at
every sample size, you may stop at any round without losing the guarantee.

When ``max_pulls`` runs out before the rule stops, the result says ``confident=False`` and
returns the leader with the arms still in the race: the guarantee does not apply.

References:
    Even-Dar, E., Mannor, S., & Mansour, Y. (2006). Action elimination and stopping conditions
    for the multi-armed bandit and reinforcement learning problems. *Journal of Machine
    Learning Research* 7:1079-1105. https://jmlr.org/papers/v7/evendar06a.html.

    Jamieson, K., Malloy, M., Nowak, R., & Bubeck, S. (2014). lil' UCB: An optimal exploration
    algorithm for multi-armed bandits. *COLT*, PMLR 35:423-439.
    https://proceedings.mlr.press/v35/jamieson14.html.

    Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform, nonparametric,
    nonasymptotic confidence sequences. *Annals of Statistics* 49(2):1055-1080.
    https://doi.org/10.1214/20-AOS1991.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx.alwaysvalid import confidence_radius

_BERNOULLI_SIGMA = 0.5


@dataclass(frozen=True, slots=True)
class Identification:
    """The outcome of ``best_arm``.

    Args:
        best: The arm returned: the leader when the rule stopped.
        confident: True when the stopping rule fired, so the guarantee applies. False when
            ``max_pulls`` ran out first.
        active: The arms still in the race at the end.
        pulls: Pulls of each arm.
        successes: Successes of each arm.
        removed_at: For each arm, the round in which it was removed, or None if it stayed.
        rounds: The number of rounds run.
        delta: The error rate of the guarantee.
        epsilon: The tolerance of the guarantee.
    """

    best: int
    confident: bool
    active: tuple[int, ...]
    pulls: tuple[int, ...]
    successes: tuple[int, ...]
    removed_at: tuple[int | None, ...]
    rounds: int
    delta: float
    epsilon: float

    @property
    def total_pulls(self) -> int:
        """Return the pulls over all arms."""
        return sum(self.pulls)


def _pull_round(
    pull: Callable[[int, int], int],
    active: list[int],
    batch: int,
    pulls: list[int],
    wins: list[int],
) -> None:
    """Pull each arm in the race ``batch`` times and add the results to the totals."""
    for arm in active:
        got = int(pull(arm, batch))
        if not 0 <= got <= batch:
            raise ValueError(f"pull returned {got} successes for {batch} pulls")
        pulls[arm] += batch
        wins[arm] += got


def _prune(
    active: list[int], means: npt.NDArray[np.float64], radius: float, epsilon: float
) -> tuple[int, list[int], bool]:
    """Return the leader, the arms that stay, and whether every other arm is within epsilon."""
    lower, upper = means - radius, means + radius
    leader_pos = int(np.argmax(lower))
    # Close is judged before removal: every other arm is within epsilon of the leader.
    close = bool(np.all(np.delete(upper, leader_pos) <= lower[leader_pos] + epsilon))
    keep = [arm for pos, arm in enumerate(active) if upper[pos] >= lower[leader_pos]]
    return active[leader_pos], keep, close


def best_arm(  # noqa: PLR0913 (settings are keyword-only)
    pull: Callable[[int, int], int],
    n_arms: int,
    *,
    delta: float = 0.05,
    epsilon: float = 0.0,
    max_pulls: int = 100_000,
    batch: int = 1,
    n_target: float = 100.0,
) -> Identification:
    """Find the arm with the highest success probability, and stop when it is certain.

    Args:
        pull: ``pull(arm, n)`` pulls ``arm`` ``n`` times and returns the number of successes
            (each pull is a 0/1 outcome).
        n_arms: The number of arms, at least 2.
        delta: The chance that the guarantee fails, in (0, 1).
        epsilon: The tolerance: stop when every other arm is within ``epsilon`` of the leader.
            The default 0 asks for the exact best arm, which can take very many pulls when two
            arms are close.
        max_pulls: The most pulls over all arms. The rule is checked after each round.
        batch: Pulls of each arm in the race per round. A larger batch makes fewer checks, and
            the guarantee holds.
        n_target: The sample size at which the intervals are tightest (see
            ``statsx.alwaysvalid.confidence_sequence``).

    Raises:
        ValueError: If ``n_arms`` is below 2, ``delta`` is not in (0, 1), ``epsilon`` is
            negative, ``batch`` or ``max_pulls`` is below 1, ``max_pulls`` is below ``n_arms * batch``, or ``pull`` returns a count outside
            0 to ``n``.
    """
    if n_arms < 2:  # noqa: PLR2004
        raise ValueError(f"n_arms must be at least 2, got {n_arms}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if epsilon < 0.0 or batch < 1 or max_pulls < 1:
        raise ValueError("epsilon must be at least 0, batch and max_pulls at least 1")
    if max_pulls < n_arms * batch:
        raise ValueError(f"max_pulls must allow one round: at least {n_arms * batch}")
    alpha = delta / n_arms
    active: list[int] = list(range(n_arms))
    pulls = [0] * n_arms
    wins = [0] * n_arms
    removed: list[int | None] = [None] * n_arms
    rounds = 0
    confident = False
    leader = 0
    while True:
        if sum(pulls) + batch * len(active) > max_pulls:
            break
        rounds += 1
        _pull_round(pull, active, batch, pulls, wins)
        radius = float(
            confidence_radius(
                pulls[active[0]], sigma=_BERNOULLI_SIGMA, alpha=alpha, n_target=n_target
            )
        )
        means = np.array([wins[a] / pulls[a] for a in active])
        leader, keep, close = _prune(active, means, radius, epsilon)
        for arm in active:
            if arm not in keep:
                removed[arm] = rounds
        active = keep
        if close or len(active) == 1:
            confident = True
            break
    return Identification(
        best=leader,
        confident=confident,
        active=tuple(active),
        pulls=tuple(pulls),
        successes=tuple(wins),
        removed_at=tuple(removed),
        rounds=rounds,
        delta=delta,
        epsilon=epsilon,
    )
