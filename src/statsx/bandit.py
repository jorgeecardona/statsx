"""Spend a budget over arms: Thompson sampling, Bayes-UCB and successive halving (SQ-3).

Each arm has a Beta posterior for its success probability (see ``statsx.conjugate.Beta``). Three
policies decide where the next pulls go:

- ``thompson_allocation``: draw one value from each posterior per pull and give the pull to the
  arm with the largest draw. An arm gets pulls in proportion to its posterior probability of being
  the best. It works for one pull at a time or for a batch.
- ``bayes_ucb_indices`` and ``bayes_ucb_arm``: give the next pull to the arm whose posterior
  quantile at level ``1 - 1 / (t (log n)^c)`` is the largest.
- ``successive_halving``: split a fixed budget over ``ceil(log2 n)`` rounds, pull every surviving
  arm equally in each round, and keep the better half. It returns the one arm left.

References:
    Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in
    view of the evidence of two samples. *Biometrika* 25(3/4):285-294.
    https://doi.org/10.2307/2332286.

    Agrawal, S., & Goyal, N. (2012). Analysis of Thompson sampling for the multi-armed bandit
    problem. *COLT*, PMLR 23:39.1-39.26. https://proceedings.mlr.press/v23/agrawal12.html.

    Kaufmann, E., Cappe, O., & Garivier, A. (2012). On Bayesian upper confidence bounds for
    bandit problems. *AISTATS*, PMLR 22:592-600.
    https://proceedings.mlr.press/v22/kaufmann12.html.

    Karnin, Z., Koren, T., & Somekh, O. (2013). Almost optimal exploration in multi-armed
    bandits. *ICML*, PMLR 28(3):1238-1246. https://proceedings.mlr.press/v28/karnin13.html.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx.conjugate import Beta

_SEED_STRIDE = 1_000_003


@dataclass(frozen=True, slots=True)
class Allocation:
    """How a batch of pulls is split over the arms.

    Args:
        counts: Pulls given to each arm; they add up to ``n_pulls``.
        shares: ``counts`` divided by ``n_pulls``.
    """

    counts: tuple[int, ...]
    shares: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HalvingRound:
    """One round of successive halving.

    Args:
        arms: The arms that were pulled in this round.
        pulls_per_arm: Pulls each of those arms got.
        means: The success rate of each arm in this round only, in the order of ``arms``.
    """

    arms: tuple[int, ...]
    pulls_per_arm: int
    means: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HalvingResult:
    """The outcome of successive halving.

    Args:
        best: The arm that survived every round.
        rounds: What happened in each round.
        successes: Total successes of each arm over all rounds.
        trials: Total pulls of each arm over all rounds.
        posteriors: The Beta posterior of each arm from all its pulls.
        pulls_used: Total pulls; never more than the budget.
    """

    best: int
    rounds: tuple[HalvingRound, ...]
    successes: tuple[int, ...]
    trials: tuple[int, ...]
    posteriors: tuple[Beta, ...]
    pulls_used: int


def _check_arms(posteriors: Sequence[Beta]) -> None:
    if len(posteriors) < 1:
        raise ValueError("at least one arm is needed")


def thompson_allocation(posteriors: Sequence[Beta], n_pulls: int, *, seed: int = 0) -> Allocation:
    """Split ``n_pulls`` over the arms by Thompson sampling.

    For each pull, one value is drawn from every posterior and the pull goes to the arm with the
    largest value. Ties go to the lowest arm number.

    Raises:
        ValueError: If there are no arms or ``n_pulls`` is below 1.
    """
    _check_arms(posteriors)
    if n_pulls < 1:
        raise ValueError(f"n_pulls must be at least 1, got {n_pulls}")
    draws = np.stack(
        [p.sample(n_pulls, seed=seed + i * _SEED_STRIDE) for i, p in enumerate(posteriors)]
    )
    winners = np.argmax(draws, axis=0)
    counts = np.bincount(winners, minlength=len(posteriors))
    return Allocation(tuple(int(c) for c in counts), tuple(float(c) / n_pulls for c in counts))


def bayes_ucb_indices(
    posteriors: Sequence[Beta], t: int, *, horizon: int | None = None, c: float = 0.0
) -> npt.NDArray[np.float64]:
    """Return the Bayes-UCB index of each arm at round ``t``.

    The index is the posterior quantile at level ``1 - 1 / (t (log horizon)^c)``. With ``c = 0``
    (the default, the value Kaufmann et al. found best in experiments) the level is ``1 - 1 / t``
    and ``horizon`` is not used. Their regret theorem needs ``c >= 5``.

    Args:
        t: The round, counting from 1.
        horizon: The total number of rounds ``n``; required when ``c > 0``.
        c: The exponent of the level.

    Raises:
        ValueError: If there are no arms, ``t`` is below 1, ``c`` is negative, or ``c > 0`` and
            ``horizon`` is missing or not above 1.
    """
    _check_arms(posteriors)
    if t < 1 or c < 0.0:
        raise ValueError("t must be at least 1 and c at least 0")
    if c > 0.0:
        if horizon is None or horizon <= 1:
            raise ValueError("horizon must be above 1 when c > 0")
        level = 1.0 - 1.0 / (t * math.log(horizon) ** c)
    else:
        level = 1.0 - 1.0 / t
    p = np.array([min(max(level, 0.0), 1.0)])
    return np.array(
        [_accel.beta_ppf(p, post.alpha, post.beta)[0] for post in posteriors], dtype=np.float64
    )


def bayes_ucb_arm(
    posteriors: Sequence[Beta], t: int, *, horizon: int | None = None, c: float = 0.0
) -> int:
    """Return the arm with the largest Bayes-UCB index at round ``t``; ties go to the lowest.

    Raises:
        ValueError: As ``bayes_ucb_indices``.
    """
    return int(np.argmax(bayes_ucb_indices(posteriors, t, horizon=horizon, c=c)))


def successive_halving(
    pull: Callable[[int, int], int],
    n_arms: int,
    budget: int,
    *,
    prior: Beta = Beta(1.0, 1.0),  # noqa: B008 (a frozen dataclass)
) -> HalvingResult:
    """Find the best arm with a fixed budget of pulls.

    There are ``ceil(log2 n_arms)`` rounds. In round ``r`` each of the ``|S_r|`` surviving arms
    is pulled ``floor(budget / (|S_r| ceil(log2 n_arms)))`` times, and the ``ceil(|S_r| / 2)`` arms
    with the highest success rate in that round remain. The success rate of a round uses that
    round's pulls only, as in the paper. Ties go to the lowest arm number.

    Args:
        pull: ``pull(arm, n)`` pulls ``arm`` ``n`` times and returns the number of successes.
        n_arms: The number of arms, at least 2.
        budget: The most pulls allowed.
        prior: The Beta prior for the posteriors in the result.

    Raises:
        ValueError: If ``n_arms`` is below 2, the budget is too small to pull every arm once in
            every round, or ``pull`` returns a count outside 0 to ``n``.
    """
    if n_arms < 2:  # noqa: PLR2004
        raise ValueError(f"n_arms must be at least 2, got {n_arms}")
    n_rounds = math.ceil(math.log2(n_arms))
    if budget < n_arms * n_rounds:
        raise ValueError(
            f"budget {budget} is too small: at least n_arms * ceil(log2 n_arms) = "
            f"{n_arms * n_rounds} pulls are needed"
        )
    alive: list[int] = list(range(n_arms))
    successes = [0] * n_arms
    trials = [0] * n_arms
    rounds: list[HalvingRound] = []
    for _ in range(n_rounds):
        per_arm = budget // (len(alive) * n_rounds)
        means: list[float] = []
        for arm in alive:
            won = int(pull(arm, per_arm))
            if not 0 <= won <= per_arm:
                raise ValueError(f"pull returned {won} successes for {per_arm} pulls")
            successes[arm] += won
            trials[arm] += per_arm
            means.append(won / per_arm)
        rounds.append(HalvingRound(tuple(alive), per_arm, tuple(means)))
        order: list[int] = np.argsort(-np.asarray(means), kind="stable").tolist()
        alive = sorted(alive[i] for i in order[: math.ceil(len(alive) / 2)])
    return HalvingResult(
        best=alive[0],
        rounds=tuple(rounds),
        successes=tuple(successes),
        trials=tuple(trials),
        posteriors=tuple(
            Beta(prior.alpha + s, prior.beta + t - s)
            for s, t in zip(successes, trials, strict=True)
        ),
        pulls_used=sum(trials),
    )
