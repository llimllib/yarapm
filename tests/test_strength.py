from datetime import date

import duckdb
import pytest

from tests.test_minutes import B, tables, team
from yarapm.harness import History
from yarapm.minutes import ProjectedMinutes
from yarapm.strength import Lineup, fill, lineups, played_for, team_ratings

A_STARTERS = ["a1", "a2", "a3", "a4", "a5"]


def fit(con, cutoff):
    projection = ProjectedMinutes("unknown", half_life=5, start_bonus=0)
    history = History(con, cutoff, 2024)
    projection.fit(history)
    return projection, history


def test_fill_gives_out_minutes_in_order():
    minutes = fill(
        {
            "star": 36,
            "a": 34,
            "b": 32,
            "c": 30,
            "d": 28,
            "e": 26,
            "f": 24,
            "g": 22,
            "h": 20,
        }
    )
    # the top eight get theirs, 232 minutes; h gets the last 8
    assert minutes == {
        "star": 36, "a": 34, "b": 32, "c": 30, "d": 28, "e": 26, "f": 24, "g": 22,
        "h": 8,
    }  # fmt: skip
    # capped at 42
    assert fill({"x": 50, **{p: 30 for p in "abcdefg"}})["x"] == 42
    # too few minutes: scaled up to 240
    assert fill({p: 20 for p in "abcdefgh"}) == {p: 30 for p in "abcdefgh"}


def test_full_strength_leaves_out_last_seasons_fill_ins():
    con = duckdb.connect()
    # last season ended with a fill-in, f, playing 40 minutes
    tank = team(["f", "a2", "a3", "a4", "a5"], ["a6", "a7", "a8"], start_min=40)
    tables(con, [("A", "B", {"A": tank, "B": B})] * 12, season=2023)
    # this season, two games without them
    a = team(A_STARTERS, ["a6", "a7", "a8"])
    tables(con, [("A", "B", {"A": a, "B": B})] * 2, season=2024)
    projection, history = fit(con, date(2024, 1, 1))

    lineup = lineups(projection, history)["A"]
    assert "f" not in lineup.full
    assert set(lineup.full) == set(a)
    assert sum(lineup.full.values()) == pytest.approx(240)
    # the model's roster still has them, so today's lineup does too (with
    # no chance of playing here: last season, no one missed a game)
    assert "f" in lineup.chances and "f" in lineup.today
    assert sum(lineup.today.values()) == pytest.approx(240)


def test_traded_players_count_for_their_new_team():
    con = duckdb.connect()
    a = team(A_STARTERS, ["a6", "a7", "a8"])
    c = team(["c1", "c2", "c3", "c4", "c5"], ["c6", "c7", "a8"])
    # a8 plays for A, then is traded to C
    tables(
        con,
        [("A", "B", {"A": a, "B": B})] * 3 + [("C", "B", {"C": c, "B": B})],
        season=2024,
    )
    teams = played_for(History(con, date(2024, 1, 1), 2024))
    assert "a8" not in teams["A"]
    assert "a8" in teams["C"]
    # and games after the cutoff don't count
    assert played_for(History(con, date(2023, 11, 2), 2024)) == {
        "A": set(a),
        "B": set(B),
    }


def test_before_a_teams_first_game_its_roster_is_full_strength():
    con = duckdb.connect()
    a = team(A_STARTERS, ["a6", "a7", "a8"])
    tables(con, [("A", "B", {"A": a, "B": B})] * 12, season=2023)
    # the season starts on November 1st
    projection, history = fit(con, date(2023, 11, 1))
    assert set(lineups(projection, history)["A"].full) == set(a)


def test_team_ratings_are_centered():
    # each player is worth their minutes / 48 on offense, and 0 on defense
    def strength(minutes):
        return sum(m for p, m in minutes.items() if p == "star") / 48, 0.0

    star = {"star": 48.0, "other": 192.0}
    rest = {"other": 240.0}
    rated, mean = team_ratings(
        {
            "A": Lineup(star, star, {}),
            "B": Lineup(star, rest, {}),
            "C": Lineup(rest, rest, {}),
        },
        strength,
    )
    assert (mean.full_off, mean.off) == pytest.approx((2 / 3, 1 / 3))
    assert rated["A"].full_off == pytest.approx(1 / 3)
    assert rated["C"].full_off == pytest.approx(-2 / 3)
    # B is as good as A at full strength, and is missing its star today
    assert rated["B"].full_off == rated["A"].full_off
    assert rated["B"].off == pytest.approx(-1 / 3)
    assert sum(r.off for r in rated.values()) == pytest.approx(0)
