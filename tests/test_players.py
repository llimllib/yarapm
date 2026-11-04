from datetime import date, timedelta
from itertools import permutations

import duckdb
import numpy as np
import pytest

from yarapm.harness import Game, History
from yarapm.players import PlayerRatings

TEAMS = {t: [f"{t}{i}" for i in range(6)] for t in "ABCD"}
LEAGUE = 110.0
HOME = 1.5  # per 100 possessions, for the home offense and against the away one


def ratings(seed=1):
    rng = np.random.default_rng(seed)
    players = [p for roster in TEAMS.values() for p in roster]
    return (
        dict(zip(players, rng.normal(0, 3, len(players)))),
        dict(zip(players, rng.normal(0, 2, len(players)))),
    )


def efficiency(offense, defense, off_players, def_players, home):
    return (
        LEAGUE
        + HOME * home
        + sum(offense[p] for p in off_players)
        - sum(defense[p] for p in def_players)
    )


def tables(con, offense, defense, seed=2):
    """
    `results`, `all_segments` and `player_games` for a season of games between every pair of teams, each in ten segments of random
    lineups scoring exactly their expected points
    """
    rng = np.random.default_rng(seed)
    results, segments, players = [], [], []
    start = date(2023, 11, 1)
    games = [pair for _ in range(3) for pair in permutations(TEAMS, 2)]
    for i, (team, opp) in enumerate(games):
        game_id, day = f"g{i:03}", start + timedelta(days=i)
        minutes: dict[str, float] = {}
        for seg in range(10):
            on = {
                t: sorted(rng.choice(TEAMS[t], 5, replace=False)) for t in (team, opp)
            }
            for p in on[team] + on[opp]:
                minutes[p] = minutes.get(p, 0) + 4.8
            for off, dfn, home in [(team, opp, 1), (opp, team, -1)]:
                eff = efficiency(offense, defense, on[off], on[dfn], home)
                segments.append(
                    (
                        2024,
                        game_id,
                        day,
                        "regular_season",
                        off,
                        dfn,
                        home == 1,
                        on[off],
                        on[dfn],
                        10,
                        10 * eff / 100,
                    )
                )
        results.append(
            (2024, game_id, day, "regular_season", team, opp, True, 100, 100)
        )
        for p, m in minutes.items():
            players.append((2024, game_id, day, p[0], p, m))

    con.execute("""CREATE TEMP TABLE results (season INTEGER, game_id VARCHAR,
        game_date DATE, game_type VARCHAR, team_id VARCHAR, opp_team_id VARCHAR,
        home BOOLEAN, poss DOUBLE, opp_poss DOUBLE)""")
    con.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", results)
    con.execute("""CREATE OR REPLACE TEMP TABLE results AS
        SELECT *, 0.0 AS pts, 0.0 AS opp_pts, 48.0 AS min FROM results""")
    con.execute("""CREATE TEMP TABLE all_segments (season INTEGER, game_id VARCHAR,
        game_date DATE, game_type VARCHAR, off_team_id VARCHAR, def_team_id VARCHAR,
        off_home BOOLEAN, off_player_ids VARCHAR[], def_player_ids VARCHAR[],
        poss INTEGER, pts DOUBLE)""")
    con.executemany(
        "INSERT INTO all_segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", segments
    )
    con.execute("""CREATE TEMP TABLE player_games (season INTEGER, game_id VARCHAR,
        game_date DATE, team_id VARCHAR, player_id VARCHAR, min DOUBLE)""")
    con.executemany("INSERT INTO player_games VALUES (?, ?, ?, ?, ?, ?)", players)


def test_predicts_from_ratings_and_minutes():
    offense, defense = ratings()
    con = duckdb.connect()
    tables(con, offense, defense)
    model = PlayerRatings(
        half_life=1e9,
        ridge=1e-6,
        season_ridge=1,
        level_half_life=1e9,
        prior_weight=0,
        tempo_half_life=1e9,
        tempo_ridge=1e-6,
        minutes="actual",
    )
    model.fit(History(con, date(2024, 6, 1), 2024))
    assert model.home == pytest.approx(HOME, abs=1e-4)

    # the game's minutes: A's starters play 36, its sixth man 60 (an
    # overtime); B's are even
    con.execute("""INSERT INTO player_games VALUES
        (2024, 'x', '2024-06-01', 'A', 'A0', 36), (2024, 'x', '2024-06-01', 'A', 'A1', 36),
        (2024, 'x', '2024-06-01', 'A', 'A2', 36), (2024, 'x', '2024-06-01', 'A', 'A3', 36),
        (2024, 'x', '2024-06-01', 'A', 'A4', 36), (2024, 'x', '2024-06-01', 'A', 'A5', 60),
        (2024, 'x', '2024-06-01', 'B', 'B0', 48), (2024, 'x', '2024-06-01', 'B', 'B1', 48),
        (2024, 'x', '2024-06-01', 'B', 'B2', 48), (2024, 'x', '2024-06-01', 'B', 'B3', 48),
        (2024, 'x', '2024-06-01', 'B', 'B4', 48)""")
    game = Game("x", 2024, date(2024, 6, 1), "regular_season", "A", "B", True)
    (prediction,) = model.predict([game])

    # weights add up to five a side
    share = {f"A{i}": 5 * 36 / 240 for i in range(5)} | {"A5": 5 * 60 / 240}
    a_off = sum(share[p] * offense[p] for p in share)
    a_def = sum(share[p] * defense[p] for p in share)
    b = [f"B{i}" for i in range(5)]
    # ratings are identified up to a shift the league average absorbs, so
    # compare margins, and each side's points against the average
    expected = (
        a_off - sum(defense[p] for p in b) - (sum(offense[p] for p in b) - a_def)
    ) + 2 * HOME
    assert prediction.poss == pytest.approx(100, rel=1e-6)
    assert prediction.pts - prediction.opp_pts == pytest.approx(expected, abs=1e-3)
