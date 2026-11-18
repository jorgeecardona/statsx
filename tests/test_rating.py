import math

import pytest

from statsx.rating import Game, Glicko2Rating, elo_expected, elo_update, glicko2_update


def test_glicko2_reproduces_the_worked_example_of_glickman() -> None:
    player = Glicko2Rating(1500.0, 200.0, 0.06)
    games = [
        Game(Glicko2Rating(1400.0, 30.0), 1.0),
        Game(Glicko2Rating(1550.0, 100.0), 0.0),
        Game(Glicko2Rating(1700.0, 300.0), 0.0),
    ]
    new = glicko2_update(player, games, tau=0.5)
    # The numbers printed in the document: 1464.06, 151.52 and 0.05999.
    assert new.rating == pytest.approx(1464.06, abs=0.01)
    assert new.rd == pytest.approx(151.52, abs=0.01)
    assert new.volatility == pytest.approx(0.05999, abs=1e-5)


def test_a_player_without_games_keeps_the_rating_and_the_rd_grows() -> None:
    player = Glicko2Rating(1650.0, 80.0, 0.07)
    new = glicko2_update(player, [])
    assert new.rating == 1650.0 and new.volatility == 0.07
    phi = 80.0 / 173.7178
    assert new.rd == pytest.approx(173.7178 * math.sqrt(phi**2 + 0.07**2))
    assert new.rd > player.rd


def test_a_surprising_result_raises_the_volatility_and_an_expected_one_does_not() -> None:
    player = Glicko2Rating(1500.0, 60.0, 0.06)
    peers = [Glicko2Rating(1500.0, 60.0)] * 12
    shock = glicko2_update(player, [Game(o, 1.0) for o in peers])
    even = glicko2_update(player, [Game(o, 1.0 if i % 2 else 0.0) for i, o in enumerate(peers)])
    assert shock.volatility > player.volatility > even.volatility
    assert shock.rating > 1500.0 > glicko2_update(player, [Game(o, 0.0) for o in peers]).rating


def test_more_games_shrink_the_rd() -> None:
    player = Glicko2Rating()
    opp = Glicko2Rating(1500.0, 100.0)
    few = glicko2_update(player, [Game(opp, 1.0)])
    many = glicko2_update(player, [Game(opp, 1.0), Game(opp, 0.0)] * 8)
    assert many.rd < few.rd < player.rd


def test_elo_expected_score_is_one_in_eleven_at_400_points_below() -> None:
    assert elo_expected(1500.0, 1500.0) == 0.5
    assert elo_expected(1100.0, 1500.0) == pytest.approx(1.0 / 11.0)
    assert elo_expected(1900.0, 1500.0) + elo_expected(1500.0, 1900.0) == pytest.approx(1.0)


def test_elo_update_is_zero_sum_and_moves_by_k_times_the_surprise() -> None:
    a, b = elo_update(1500.0, 1500.0, 1.0, k=20.0)
    assert (a, b) == (1510.0, 1490.0)
    a, b = elo_update(1700.0, 1500.0, 0.5, k=32.0)
    assert a + b == pytest.approx(3200.0)
    assert a == pytest.approx(1700.0 + 32.0 * (0.5 - elo_expected(1700.0, 1500.0)))


def test_online_elo_recovers_the_odds_it_is_fed() -> None:
    # Two players, a wins 3 games in 4 forever: the rating gap settles near 400 log10(3).
    a, b = 1500.0, 1500.0
    for i in range(4000):
        a, b = elo_update(a, b, 0.0 if i % 4 == 3 else 1.0, k=4.0)
    assert a - b == pytest.approx(400.0 * math.log10(3.0), abs=15.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: Game(Glicko2Rating(), 0.7),
        lambda: Glicko2Rating(1500.0, 0.0),
        lambda: Glicko2Rating(1500.0, 50.0, -0.1),
        lambda: glicko2_update(Glicko2Rating(), [], tau=0.0),
        lambda: elo_update(1500.0, 1500.0, 2.0),
        lambda: elo_update(1500.0, 1500.0, 1.0, k=0.0),
    ],
)
def test_bad_input_is_refused(call: object) -> None:
    with pytest.raises(ValueError, match="must"):
        call()  # type: ignore[operator]
