from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from yarapm.harness import Game, History
from yarapm.minutes import CAP, ProjectedMinutes, allocate, availability, rosters


def utc(*args: int) -> datetime:
    """A time in UTC without a zone, as `History` has them"""
    return datetime(*args, tzinfo=UTC).replace(tzinfo=None)


def test_allocate_scales_to_240_and_caps():
    minutes = allocate({"star": 60} | {f"p{i}": 25 for i in range(7)})
    assert sum(minutes.values()) == pytest.approx(240)
    assert minutes["star"] == CAP
    # what's over the cap goes to the others in proportion
    assert minutes["p0"] == pytest.approx((240 - CAP) / 7)

    assert allocate({p: 1 for p in "abcdefgh"}) == {p: 30 for p in "abcdefgh"}
    # with too few players, everyone is capped
    assert allocate({"a": 1, "b": 2}) == {"a": CAP, "b": CAP}


def tables(con, games, season=2024):
    """
    `results` and `player_games` from games: (team, opp, {player: (min,
    started)} for each team), a day apart from November 1st. Adds to them
    if they exist
    """
    results, players = [], []
    start = date(season - 1, 11, 1)
    for i, (team, opp, lineups) in enumerate(games):
        game_id, day = f"g{season}{i:03}", start + timedelta(days=i)
        results.append((season, game_id, day, "regular_season", team, opp, True))
        for t, lineup in lineups.items():
            for p, (m, started) in lineup.items():
                players.append((season, game_id, day, t, p, m, started))
    con.execute("""CREATE TEMP TABLE IF NOT EXISTS results (season INTEGER,
        game_id VARCHAR, game_date DATE, game_type VARCHAR, team_id VARCHAR,
        opp_team_id VARCHAR, home BOOLEAN)""")
    con.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", results)
    con.execute("""CREATE TEMP TABLE IF NOT EXISTS player_games (season INTEGER,
        game_id VARCHAR,
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


def test_availability_ignores_this_seasons_games():
    con = duckdb.connect()
    a = team(A_STARTERS, ["a6", "a7", "a8"])
    tables(con, [("A", "B", {"A": a, "B": B})] * 12, season=2023)
    before = availability(History(con, date(2023, 11, 1), 2024))
    # a new roster this season: last season's players don't play
    c = team(["c1", "c2", "c3", "c4", "c5"], ["c6", "c7", "c8"])
    tables(con, [("A", "B", {"A": c, "B": B})] * 12, season=2024)
    assert availability(History(con, date(2024, 1, 1), 2024)) == before


def schedule_and_injuries(con, injuries):
    """
    A's next three games after five, from 2023-11-06, in nba.schedule
    (as `load` makes it), and ESPN's injury lists: (fetched_at, player,
    status, fantasy_status, return_date)
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE schedule AS
        SELECT game_id, 2024 AS season, game_date, 'regular_season' AS game_type,
            'A' AS team_id, 'B' AS opp_team_id, true AS home,
            game_date + INTERVAL 24 HOUR AS start_time
        FROM (VALUES ('x1', DATE '2023-11-06'), ('x2', DATE '2023-11-08'),
            ('x3', DATE '2023-11-10')) v(game_id, game_date)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE injuries (season INTEGER,
        fetched_at TIMESTAMP, team_id VARCHAR, player_id VARCHAR, status VARCHAR,
        fantasy_status VARCHAR, return_date DATE)""")
    con.executemany("INSERT INTO injuries VALUES (2024, ?, 'A', ?, ?, ?, ?)", injuries)


SCHEDULED = [
    Game(g, 2024, d, "regular_season", "A", "B", True)
    for g, d in [
        ("x1", date(2023, 11, 6)),
        ("x2", date(2023, 11, 8)),
        ("x3", date(2023, 11, 10)),
    ]
]


def unknown(con, as_of=None):
    """ProjectedMinutes with unknown availability, everyone at 0.9 to play"""
    model = ProjectedMinutes("unknown", half_life=5, start_bonus=0)
    model.chances, model.chances_season = {(0, 5): 0.9}, 2024
    model.fit(History(con, date(2023, 11, 6), 2024, as_of))
    return model


def test_injury_list_sets_the_chance_of_playing():
    con = duckdb.connect()
    tables(con, [("A", "B", {"A": team(A_STARTERS, ["a6", "a7", "a8"]), "B": B})] * 5)
    listed = utc(2023, 11, 6, 12)
    schedule_and_injuries(
        con,
        [
            (listed, "a1", "out", "OUT", date(2023, 11, 9)),
            (listed, "a2", "out", "GTD", date(2023, 11, 7)),
            (listed, "a3", "day_to_day", "GTD", None),
            (listed, "a4", "suspension", "OUT", date(2023, 11, 7)),
            # after the first tip-off on the cutoff day: not seen
            (utc(2023, 11, 7, 1), "a5", "out", "OUT", date(2023, 12, 1)),
        ],
    )
    model = unknown(con)
    x1, x2, x3 = SCHEDULED
    chances = {g.game_id: model.playing_chances("A", g) for g in SCHEDULED}
    # the team's next game
    assert {p: chances["x1"][p] for p in ["a1", "a2", "a3", "a4", "a5"]} == {
        "a1": 0.0,
        "a2": 0.5,
        "a3": 0.7,
        "a4": 0.0,
        "a5": 0.9,
    }
    # with no game, the next one: the lineups in yarapm.strength
    assert model.playing_chances("A") == chances["x1"]
    # later games before the return date
    assert chances["x2"]["a1"] == 0.0
    assert chances["x2"]["a2"] == chances["x2"]["a3"] == 0.9
    assert chances["x3"] == {p: 0.9 for p in chances["x3"]}

    # a player listed out gets no minutes, and the rest share 240
    minutes = model.predict(SCHEDULED)
    assert minutes["x1", "A"]["a1"] == 0
    assert minutes["x3", "A"]["a1"] > 0
    assert sum(minutes["x1", "A"].values()) == pytest.approx(240)
    assert minutes["x1", "B"] == minutes["x3", "B"]

    # as of a later time: the later list, whose next game for A is x2
    later = unknown(con, datetime(2023, 11, 7, 2, tzinfo=UTC))
    assert later.playing_chances("A", x1)["a1"] == 0.9
    assert later.playing_chances("A", x2)["a5"] == 0.0
    assert later.playing_chances("A", x3)["a5"] == 0.0


def test_no_injury_list_changes_nothing():
    con = duckdb.connect()
    tables(con, [("A", "B", {"A": team(A_STARTERS, ["a6", "a7", "a8"]), "B": B})] * 5)
    # no lists loaded, as in the tests and for a History without `load`
    before = unknown(con)
    assert before.injuries is None
    expected = before.predict(SCHEDULED)
    assert expected["x1", "A"]["a1"] > 0
    # loaded, but only fetched after the first tip-off
    schedule_and_injuries(
        con, [(utc(2023, 11, 7, 1), "a1", "out", "OUT", date(2023, 12, 1))]
    )
    after = unknown(con)
    assert after.injuries is None
    assert after.predict(SCHEDULED) == expected


def test_the_pattern_alone_ignores_the_list():
    con = duckdb.connect()
    tables(con, [("A", "B", {"A": team(A_STARTERS, ["a6", "a7", "a8"]), "B": B})] * 5)
    expected = unknown(con).predict(SCHEDULED)
    schedule_and_injuries(
        con, [(utc(2023, 11, 6, 12), "a1", "out", "OUT", date(2023, 12, 1))]
    )
    assert unknown(con).predict(SCHEDULED)["x1", "A"]["a1"] == 0
    pattern = ProjectedMinutes("unknown", half_life=5, start_bonus=0, injuries=False)
    pattern.chances, pattern.chances_season = {(0, 5): 0.9}, 2024
    pattern.fit(History(con, date(2023, 11, 6), 2024))
    assert pattern.injuries is None
    assert pattern.predict(SCHEDULED) == expected
