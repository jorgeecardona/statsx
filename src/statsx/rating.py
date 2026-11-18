"""Online ratings from a stream of match results: Elo and Glicko-2 (CO-9).

Use this when results arrive one match (Elo) or one rating period (Glicko-2) at a time and you want
the ratings updated in place. To rank systems from a fixed table of comparisons, with intervals,
use ``statsx.pairwise`` instead.

- Elo: ``elo_update`` moves two ratings by ``k * (score - expected)``. The expected score is
  ``1 / (1 + 10^(-(ra - rb) / 400))``, the same 400-points-per-factor-10 scale as
  ``BradleyTerryFit.elo``.
- Glicko-2: ``glicko2_update`` takes the games of one rating period and returns the new rating, the
  rating deviation (RD) and the volatility. The algorithm is the one in Glickman's example
  document, including the Illinois iteration for the volatility (revised 2012).

An update is a point estimate. A rating from ``elo_update`` has no interval. A Glicko-2 rating
carries RD, and ``rating -/+ 2 * RD`` is the 95% interval Glickman gives for it.

References:
    Elo, A. E. (1978). The Rating of Chessplayers, Past and Present. Arco, New York.

    Glickman, M. E. (2001). Dynamic paired comparison models with stochastic variances. Journal of
    Applied Statistics 28(6):673-689. https://doi.org/10.1080/02664760120059219

    Glickman, M. E. (2022). Example of the Glicko-2 system (Boston University, March 22, 2022).
    http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_SCALE = 173.7178  # Glicko points per unit of the Glicko-2 scale (Glickman 2022, step 2)
_EPSILON = 1e-6  # tolerance of the volatility iteration (Glickman 2022, step 5)
_MAX_ITERATIONS = 200
_SCORES = (0.0, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class Glicko2Rating:
    """A Glicko-2 rating on the original Glicko scale.

    Args:
        rating: The rating ``r``. An unrated player starts at 1500.
        rd: The rating deviation. An unrated player starts at 350.
        volatility: The expected fluctuation of the rating. An unrated player starts at 0.06
            (Glickman: this value depends on the application).
    """

    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rating) and math.isfinite(self.rd)):
            raise ValueError("rating and rd must be finite")
        if self.rd <= 0.0 or not (math.isfinite(self.volatility) and self.volatility > 0.0):
            raise ValueError("rd and volatility must be positive")

    def interval(self) -> tuple[float, float]:
        """Return ``rating -/+ 2 * rd``, the 95% interval of Glickman (2022)."""
        return self.rating - 2.0 * self.rd, self.rating + 2.0 * self.rd


@dataclass(frozen=True, slots=True)
class Game:
    """One game of a rating period, seen from the player being updated.

    Args:
        opponent: The rating of the opponent at the start of the period.
        score: 1 for a win, 0.5 for a draw, 0 for a loss.
    """

    opponent: Glicko2Rating
    score: float

    def __post_init__(self) -> None:
        if self.score not in _SCORES:
            raise ValueError(f"score must be 0, 0.5 or 1, got {self.score}")


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Return the expected score of ``a`` against ``b``: ``1 / (1 + 10^(-(ra - rb) / 400))``."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def elo_update(
    rating_a: float, rating_b: float, score: float, *, k: float = 32.0
) -> tuple[float, float]:
    """Return the new ratings of ``a`` and ``b`` after one game.

    Args:
        rating_a: The rating of the first player.
        rating_b: The rating of the second player.
        score: The result for ``a``: 1 a win, 0.5 a draw, 0 a loss.
        k: The step size, the most points one game can move a rating. It is a choice of the
            rating pool (32 is common), not a constant of the method.

    The update is zero-sum: ``a`` gains what ``b`` loses.
    """
    if score not in _SCORES:
        raise ValueError(f"score must be 0, 0.5 or 1, got {score}")
    if not (math.isfinite(k) and k > 0.0):
        raise ValueError(f"k must be positive, got {k}")
    step = k * (score - elo_expected(rating_a, rating_b))
    return rating_a + step, rating_b - step


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi**2)


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(sigma: float, phi: float, v: float, delta: float, tau: float) -> float:
    """Return the new volatility by the Illinois iteration of Glickman (2022), step 5."""
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi * phi + v + ex
        return ex * (delta * delta - phi * phi - v - ex) / (2.0 * denom * denom) - (x - a) / (
            tau * tau
        )

    lo = a
    if delta * delta > phi * phi + v:
        hi = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
            if k > _MAX_ITERATIONS:
                raise ValueError("the volatility bracket was not found; check tau and the inputs")
        hi = a - k * tau
    f_lo, f_hi = f(lo), f(hi)
    for _ in range(_MAX_ITERATIONS):
        if abs(hi - lo) <= _EPSILON:
            return math.exp(lo / 2.0)
        c = lo + (lo - hi) * f_lo / (f_hi - f_lo)
        f_c = f(c)
        if f_c * f_hi <= 0.0:
            lo, f_lo = hi, f_hi
        else:
            f_lo /= 2.0
        hi, f_hi = c, f_c
    raise ValueError("the volatility iteration did not converge")


def glicko2_update(
    player: Glicko2Rating, games: Sequence[Game], *, tau: float = 0.5
) -> Glicko2Rating:
    """Return the rating of ``player`` after one rating period of ``games``.

    The games of a period are treated as simultaneous: every opponent rating is the one at the start
    of the period. Glickman advises an average of at least 10 to 15 games per player per period.
    A player with no games keeps the rating and volatility, and the RD grows (step 6 only).

    Args:
        player: The rating at the start of the period.
        games: The games of the period.
        tau: The system constant that limits how fast the volatility changes. Glickman suggests
            0.3 to 1.2 and a value tested for predictive accuracy; smaller values suit data with
            very improbable results.
    """
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be positive, got {tau}")
    mu = (player.rating - 1500.0) / _SCALE
    phi = player.rd / _SCALE
    sigma = player.volatility
    if not games:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return Glicko2Rating(player.rating, phi_star * _SCALE, sigma)
    terms = [
        (_g(o.rd / _SCALE), _expected(mu, (o.rating - 1500.0) / _SCALE, o.rd / _SCALE), s)
        for o, s in ((game.opponent, game.score) for game in games)
    ]
    v = 1.0 / sum(g * g * e * (1.0 - e) for g, e, _ in terms)
    improvement = sum(g * (s - e) for g, e, s in terms)
    delta = v * improvement
    new_sigma = _new_volatility(sigma, phi, v, delta, tau)
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * improvement
    return Glicko2Rating(_SCALE * new_mu + 1500.0, _SCALE * new_phi, new_sigma)
