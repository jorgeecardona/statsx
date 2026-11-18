# Update ratings from a stream of results

Use this when results arrive one game at a time (Elo) or one rating period at a time (Glicko-2) and
you want the ratings updated as they come. The module is `statsx.rating`. To rank systems from a
fixed table, with intervals, use [pairwise results](pairwise.md) instead.

## Elo, one game

The expected score of `a` against `b` is `1 / (1 + 10^(-(ra - rb) / 400))`. The update moves both
ratings by `k * (score - expected)`, so it is zero-sum (Elo 1978).

```python
from statsx.rating import elo_expected, elo_update

print(round(elo_expected(1100.0, 1500.0), 4))
print(elo_update(1500.0, 1500.0, 1.0, k=20.0))
```

```text
0.0909
(1510.0, 1490.0)
```

`k` is a choice of the rating pool, not a constant of the method. The scale is the one used by
`BradleyTerryFit.elo`, so a rating from the fit and a rating from the stream are comparable.

## Glicko-2, one rating period

Each player has a rating, a rating deviation (RD) and a volatility. The games of one period count
as simultaneous. `glicko2_update` returns the new rating, RD and volatility for one player.

```python
from statsx.rating import Game, Glicko2Rating, glicko2_update

player = Glicko2Rating(1500.0, 200.0, 0.06)
games = [
    Game(Glicko2Rating(1400.0, 30.0), 1.0),
    Game(Glicko2Rating(1550.0, 100.0), 0.0),
    Game(Glicko2Rating(1700.0, 300.0), 0.0),
]
new = glicko2_update(player, games, tau=0.5)
print(round(new.rating, 1), round(new.rd, 1), round(new.volatility, 3))
```

```text
1464.1 151.5 0.06
```

Glickman's worked example prints 1464.06, 151.52 and 0.05999. The small difference in the last
digit is the rounding of intermediate values in the document; the test allows 0.01 points. A player with no games keeps the rating and
the volatility, and the RD grows.

`rating -/+ 2 * rd` is the 95% interval Glickman gives (`Glicko2Rating.interval()`).

## What it assumes

- The update is a point estimate. Elo has no interval.
- Glicko-2 works best with about 10 to 15 games per player per period (Glickman). With one game
  per period the volatility term is poorly determined.
- `tau` limits how fast the volatility changes. Glickman suggests 0.3 to 1.2 and a value tested
  for predictive accuracy.
- Both methods assume one strength per player that does not depend on the order of the games,
  apart from the drift Glicko-2 allows.
- The package checks the Glicko-2 update against the worked example of Glickman (2022). It does not
  check whether the model fits your data.

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco, New York.
- Glickman, M. E. (2001). Dynamic paired comparison models with stochastic variances. *Journal of
  Applied Statistics* 28(6):673-689. <https://doi.org/10.1080/02664760120059219>
- Glickman, M. E. (2022). Example of the Glicko-2 system. Boston University, March 22, 2022.
  <http://www.glicko.net/glicko/glicko2.pdf>
