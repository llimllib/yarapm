from datetime import date, timedelta

import duckdb
import pytest

from yarapm.harness import Game, History
from yarapm.minutes import CAP, ProjectedMinutes, allocate, rosters


def test_allocate_scales_to_240_and_caps():
    minutes = allocate({"star": 60} | {f"p{i}": 25 for i in range(7)})
    assert sum(minutes.values()) == pytest.approx(240)
    assert minutes["star"] == CAP
    # what's over the cap goes to the others in proportion
    assert minutes["p0"] == pytest.approx((240 - CAP) / 7)

    assert allocate({p: 1 for p in "abcdefgh"}) == {p: 30 for p in "abcdefgh"}
    # with too few players, everyone is capped
    assert allocate({"a": 1, "b": 2}) == {"a": CAP, "b": CAP}


def tables(con, games):
    """
    `results` and `player_games` from games: (team, opp, {player: (min,
    started)} for each team), a day apart
    """
    results, players = [], []
    start = date(2023, 11, 1)
    for i, (team, opp, lineups) in enumerate(games):
        game_id, day = f"g{i:03}", start + timedelta(days=i)
        results.append((2024, game_id, day, "regular_season", team, opp, True))
        for t, lineup in lineups.items():
            for p, (m, started) in lineup.items():
                players.append((2024, game_id, day, t, p, m, started))
    con.execute("""CREATE TEMP TABLE results (season INTEGER, game_id VARCHAR,
        game_date DATE, game_type VARCHAR, team_id VARCHAR, opp_team_id VARCHAR,
        home BOOLEAN)""")
    con.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", results)
    con.execute("""CREATE TEMP TABLE player_games (season INTEGER, game_id VARCHAR,
        game_date DATE, team_id VARCHAR, player_id VARCHAR, min DOUBLE,
        started BOOLEAN)""")
    con.executemany("INSERT INTO player_games VALUES (?, ?, ?, ?, ?, ?, ?)", players)


def team(starters, bench, start_min=36.0):
    """Five starters at `start_min` and the bench sharing the rest"""
    lineup = {p: (start_min, True) for p in starters}
    rest = (240 - 5 * start_min) / len(bench)
    return lineup | {p: (rest, False) for p in bench}


A_STARTERS = ["a1", "a2", "a3", "a4", "a5"]
B = team(["b1", "b2", "b3", "b4", "b5"], ["b6", "b7", "b8"])


def test_minutes_when_playing():
    con = duckdb.connect()
    games = [("A", "B", {"A": team(A_STARTERS, ["a6", "a7", "a8"]), "B": B})] * 9
    # a6 starts in place of a1 in the last game
    last = team(["a6", "a2", "a3", "a4", "a5"], ["a7", "a8"], start_min=40)
    games.append(("A", "B", {"A": last, "B": B}))
    tables(con, games)

    plain = ProjectedMinutes("known", half_life=1e9, start_bonus=0)
    plain.fit(History(con, date(2024, 1, 1), 2024))
    # starters' average over ten games: nine of 36 and one of 40
    assert plain.minutes("a2") == pytest.approx((9 * 36 + 40) / 10)
    assert plain.minutes("new") == 10

    bonus = ProjectedMinutes("known", half_life=1e9, start_bonus=5)
    bonus.fit(History(con, date(2024, 1, 1), 2024))
    # a6 started one of their ten games, and the last one
    assert bonus.minutes("a6") == pytest.approx(plain.minutes("a6") + 5 * 0.9)


def test_known_availability_uses_who_played():
    con = duckdb.connect()
    tables(con, [("A", "B", {"A": team(A_STARTERS, ["a6", "a7", "a8"]), "B": B})] * 5)
    model = ProjectedMinutes("known", half_life=5, start_bonus=0)
    model.fit(History(con, date(2024, 1, 1), 2024))
    # a1 is out of the next game: their minutes go to the rest
    con.execute("""INSERT INTO player_games VALUES
        (2024, 'x', '2024-01-01', 'A', 'a2', 30, true),
        (2024, 'x', '2024-01-01', 'A', 'a6', 30, false)""")
    game = Game("x", 2024, date(2024, 1, 1), "regular_season", "A", "B", True)
    minutes = model.predict([game])[("x", "A")]
    assert set(minutes) == {"a2", "a6"}
    assert sum(minutes.values()) == pytest.approx(84)  # each capped at 42


def test_rosters_follow_trades():
    con = duckdb.connect()
    a = team(A_STARTERS, ["a6", "a7", "a8"])
    c = team(["c1", "c2", "c3", "c4", "c5"], ["c6", "c7", "a8"])
    tables(
        con,
        [("A", "B", {"A": a, "B": B})] * 3
        # a8 is traded to C; a5 misses A's last two games
        + [("C", "B", {"C": c, "B": B})]
        + [
            (
                "A",
                "B",
                {"A": {k: v for k, v in a.items() if k not in ("a5", "a8")}, "B": B},
            )
        ]
        * 2,
    )
    teams = rosters(History(con, date(2024, 1, 1), 2024))
    assert "a8" not in teams["A"]
    assert teams["C"]["a8"] == (0, 1)
    assert teams["A"]["a5"] == (2, 3)
    assert teams["A"]["a1"] == (0, 5)
