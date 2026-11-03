import duckdb
import pytest

from yarapm.segments import build

GAME = "0022500001"
HOME, AWAY = "H", "A"
HOME_PLAYERS = ["h1", "h2", "h3", "h4", "h5"]
AWAY_PLAYERS = ["a1", "a2", "a3", "a4", "a5"]


def made(team, t, value=2, period=1):
    return {
        "action_type": "Made Shot",
        "team_id": team,
        "time": t,
        "period": period,
        "shot_result": "Made",
        "shot_value": value,
        "pts": value,
    }


def missed(team, t, value=2):
    return {
        "action_type": "Missed Shot",
        "team_id": team,
        "time": t,
        "shot_result": "Missed",
        "shot_value": value,
    }


def free_throw(team, t, n, of, make=True):
    return {
        "action_type": "Free Throw",
        "sub_type": f"Free Throw {n} of {of}",
        "team_id": team,
        "time": t,
        "shot_result": "Made" if make else "Missed",
        "pts": 1 if make else 0,
    }


def rebound(team, t):
    return {"action_type": "Rebound", "sub_type": "Unknown", "team_id": team, "time": t}


def foul(team, t):
    return {"action_type": "Foul", "sub_type": "Shooting", "team_id": team, "time": t}


def substitution(team, t):
    return {"action_type": "Substitution", "team_id": team, "time": t}


def game(
    con: duckdb.DuckDBPyConnection,
    events: list[dict],
    stints: list[tuple[str, str, float, float, int]] | None = None,
    box: dict[str, tuple[int, int]] | None = None,
) -> None:
    """
    One 720-second period. Without `stints`, the starters play it all with
    plus-minus from the events. `box` is team -> (pts, poss), from the
    events and segments if not given
    """
    con.execute(
        "CREATE OR REPLACE TEMP TABLE games AS SELECT ? AS game_id, "
        "DATE '2025-11-01' AS game_date, 'regular_season' AS game_type, "
        "? AS home_team_id, false AS neutral",
        [GAME, HOME],
    )
    rows = []
    for i, e in enumerate(events, 1):
        rows.append(
            (
                GAME,
                i,
                e.get("period", 1),
                e["time"],
                e.get("team_id"),
                e.get("person_id"),
                e["action_type"],
                e.get("sub_type"),
                e.get("shot_result"),
                e.get("shot_value"),
                e.get("pts", 0),
            )
        )
    con.execute("""CREATE OR REPLACE TEMP TABLE events (game_id VARCHAR,
        action_id INTEGER, period INTEGER, time DOUBLE, team_id VARCHAR,
        person_id VARCHAR, action_type VARCHAR, sub_type VARCHAR,
        shot_result VARCHAR, shot_value INTEGER, pts INTEGER)""")
    con.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    if stints is None:
        margin = sum(
            e.get("pts", 0) * (1 if e["team_id"] == HOME else -1) for e in events
        )
        stints = [(HOME, p, 0, 720, margin) for p in HOME_PLAYERS] + [
            (AWAY, p, 0, 720, -margin) for p in AWAY_PLAYERS
        ]
    con.execute("""CREATE OR REPLACE TEMP TABLE rotations (game_id VARCHAR,
        team_id VARCHAR, player_id VARCHAR, in_time DOUBLE, out_time DOUBLE,
        plus_minus INTEGER)""")
    con.executemany(
        "INSERT INTO rotations VALUES (?, ?, ?, ?, ?, ?)", [(GAME, *s) for s in stints]
    )
    con.execute("""CREATE OR REPLACE TEMP TABLE box (game_id VARCHAR,
        team_id VARCHAR, pts INTEGER, poss INTEGER)""")
    if box is None:
        pts = {
            t: sum(e.get("pts", 0) for e in events if e.get("team_id") == t)
            for t in (HOME, AWAY)
        }
        box = {t: (pts[t], 0) for t in (HOME, AWAY)}
    con.executemany(
        "INSERT INTO box VALUES (?, ?, ?, ?)", [(GAME, t, *v) for t, v in box.items()]
    )


def rows(con, *columns):
    cols = ", ".join(columns)
    return con.sql(
        f"SELECT seg, off_team_id, {cols} FROM segments ORDER BY seg, off_team_id"
    ).fetchall()


@pytest.fixture
def con():
    return duckdb.connect()


def test_one_lineup_two_rows(con):
    game(con, [made(HOME, 10), made(AWAY, 20, 3)])
    build(con)
    assert con.sql(
        "SELECT seg, off_team_id, def_team_id, off_home, off_player_ids, pts FROM segments ORDER BY off_team_id"
    ).fetchall() == [
        (1, AWAY, HOME, False, AWAY_PLAYERS, 3),
        (1, HOME, AWAY, True, HOME_PLAYERS, 2),
    ]


def substitution_stints(home_margins: tuple[int, int]):
    """h5 is replaced by h6 at 100; home's margin before and after"""
    before, after = home_margins
    return (
        [(HOME, p, 0, 720, before + after) for p in HOME_PLAYERS[:4]]
        + [(HOME, "h5", 0, 100, before), (HOME, "h6", 100, 720, after)]
        + [(AWAY, p, 0, 720, -(before + after)) for p in AWAY_PLAYERS]
    )


def test_segments_split_at_substitutions(con):
    events = [made(HOME, 50), substitution(HOME, 100), made(AWAY, 150, 3)]
    game(con, events, substitution_stints((2, -3)))
    build(con)
    assert rows(con, "start_time", "end_time", "pts") == [
        (1, AWAY, 0, 100, 0),
        (1, HOME, 0, 100, 2),
        (2, AWAY, 100, 720, 3),
        (2, HOME, 100, 720, 0),
    ]
    assert con.sql(
        "SELECT off_player_ids FROM segments WHERE seg = 2 AND off_team_id = 'H'"
    ).fetchall() == [(["h1", "h2", "h3", "h4", "h6"],)]


def test_play_on_substitution_tick_follows_its_order(con):
    # a basket listed before the sub is the old lineup's, after it the new one's
    events = [made(HOME, 100), substitution(HOME, 100), made(AWAY, 100, 3)]
    game(con, events, substitution_stints((2, -3)))
    build(con)
    assert rows(con, "pts") == [(1, AWAY, 0), (1, HOME, 2), (2, AWAY, 3), (2, HOME, 0)]


def test_free_throws_go_to_the_lineup_on_for_the_foul(con):
    events = [
        foul(AWAY, 100),
        substitution(HOME, 100),
        free_throw(HOME, 100, 1, 2),
        free_throw(HOME, 100, 2, 2),
    ]
    game(con, events, substitution_stints((2, 0)))
    build(con)
    assert rows(con, "pts", "fta", "ftm") == [
        (1, AWAY, 0, 0, 0),
        (1, HOME, 2, 2, 2),
        (2, AWAY, 0, 0, 0),
        (2, HOME, 0, 0, 0),
    ]


def test_free_throws_after_a_new_foul_go_to_the_new_lineup(con):
    events = [substitution(HOME, 100), foul(AWAY, 100), free_throw(HOME, 100, 1, 1)]
    game(con, events, substitution_stints((0, 1)))
    build(con)
    assert rows(con, "pts") == [(1, AWAY, 0), (1, HOME, 0), (2, AWAY, 0), (2, HOME, 1)]


def test_possessions(con):
    events = [
        missed(HOME, 10),
        rebound(HOME, 11),  # offensive rebound: same possession
        made(HOME, 12),
        missed(AWAY, 30),  # away's first
        rebound(HOME, 31),  # home's second
        free_throw(HOME, 40, 1, 2, make=False),
        rebound(HOME, 40),  # dead ball between free throws
        free_throw(HOME, 40, 2, 2, make=False),
        rebound(AWAY, 41),  # away's second
        {
            "action_type": "Turnover",
            "sub_type": "Bad Pass",
            "team_id": AWAY,
            "time": 50,
        },
        made(HOME, 60, 3),  # home's third, and the period ends on a make: away's third
    ]
    game(con, events)
    build(con)
    assert rows(con, "poss", "oreb", "dreb", "tov") == [
        (1, AWAY, 3, 0, 1, 1),
        (1, HOME, 3, 1, 1, 0),
    ]


def test_possession_credited_to_the_lineup_that_plays_it(con):
    # a defensive rebound, then a substitution, then the shot: the new lineup's
    events = [
        missed(AWAY, 50),
        rebound(HOME, 60),
        substitution(HOME, 100),
        made(HOME, 110),
    ]
    game(con, events, substitution_stints((0, 2)))
    build(con)
    assert rows(con, "poss") == [(1, AWAY, 1), (1, HOME, 0), (2, AWAY, 1), (2, HOME, 1)]


def test_checks(con):
    events = [made(HOME, 50), substitution(HOME, 100), made(AWAY, 150, 3)]
    game(con, events, substitution_stints((2, -3)), box={HOME: (2, 1), AWAY: (3, 2)})
    build(con)
    assert con.sql(
        "SELECT pm_diff, pts_match, poss_diff, ok FROM game_checks"
    ).fetchall() == [(0, True, 1, True)]

    # h5's stint is off by more than a tick's free throws
    stints = substitution_stints((2, -3))
    stints[4] = (HOME, "h5", 0, 100, 5)
    game(con, events, stints, box={HOME: (2, 1), AWAY: (3, 2)})
    build(con)
    assert con.sql("SELECT pm_diff, ok FROM game_checks").fetchall() == [(3, False)]

    game(con, events, substitution_stints((2, -3)), box={HOME: (4, 1), AWAY: (3, 2)})
    build(con)
    assert con.sql("SELECT pts_match, ok FROM game_checks").fetchall() == [
        (False, False)
    ]
