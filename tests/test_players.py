from datetime import date, timedelta
from itertools import permutations

import duckdb
import numpy as np
import pytest

from yarapm import rapm
from yarapm.harness import Game, History
from yarapm.players import PlayerRatings, age_basis
from yarapm.predict import live_models

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


def tables(con, offense, defense, seed=2, season=2024, create=True):
    """
    `results`, `all_segments` and `player_games` for a season of games between every pair of teams, each in ten segments of random
    lineups scoring exactly their expected points. With `create` false,
    the season is added to the tables
    """
    rng = np.random.default_rng(seed)
    results, segments, players = [], [], []
    start = date(season - 1, 11, 1)
    games = [pair for _ in range(3) for pair in permutations(TEAMS, 2)]
    for i, (team, opp) in enumerate(games):
        game_id, day = f"g{season}{i:03}", start + timedelta(days=i)
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
                        season,
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
            (season, game_id, day, "regular_season", team, opp, True, 100, 100)
        )
        for p, m in minutes.items():
            players.append((season, game_id, day, p[0], p, m))

    if not create:
        con.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 48)",
            results,
        )
        con.executemany(
            "INSERT INTO all_segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            segments,
        )
        con.executemany("INSERT INTO player_games VALUES (?, ?, ?, ?, ?, ?)", players)
        return
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
        aging=False,
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


def test_injuries_setting_keeps_run_ids():
    # `injuries` is in the settings only when false, so runs from before #46
    # keep their ids, and `report` only when true, for those before #52
    settings = PlayerRatings(minutes="unknown").settings
    assert "injuries" not in settings and settings["report"] is True
    assert "report" not in PlayerRatings(minutes="unknown", report=False).settings
    models = live_models()
    pattern = models["rapm_pattern"]
    assert pattern.name == "rapm_pattern" and models["rapm"].name == "rapm"
    assert pattern.settings == {
        k: v for k, v in models["rapm"].settings.items() if k != "report"
    } | {"injuries": False}


def test_aging_carries_ratings_to_this_season():
    """
    Players aged along a curve between 2023 and 2024: with `aging`, a fit
    on both seasons rates them as they are in 2024
    """
    now_off, now_def = ratings()
    rng = np.random.default_rng(3)
    born = {p: float(rng.integers(1985, 2004)) for p in now_off}
    curve = np.array([[0.3, -0.4, 0.2], [0.1, 0.2, 0.0]])

    def then(now, side):
        return {
            p: r
            - (
                age_basis(np.array(2024 - born[p]))
                - age_basis(np.array(2023 - born[p]))
            )
            @ curve[side]
            for p, r in now.items()
        }

    con = duckdb.connect()
    tables(con, now_off, now_def)
    tables(con, then(now_off, 0), then(now_def, 1), seed=4, season=2023, create=False)
    con.execute("CREATE TEMP TABLE bio (player_id VARCHAR, born DOUBLE)")
    con.executemany("INSERT INTO bio VALUES (?, ?)", list(born.items()))

    def fit(aging):
        model = PlayerRatings(
            half_life=1e9,
            ridge=1e-6,
            season_ridge=1,
            level_half_life=1e9,
            prior_weight=0,
            tempo_half_life=1e9,
            tempo_ridge=1e-6,
            minutes="actual",
            aging=aging,
            aging_ridge=1e-6,
        )
        model.fit(History(con, date(2024, 6, 1), 2024))
        assert model.ratings is not None
        # ratings are identified up to a shift: compare them centered
        got = np.array([model.ratings.rating(p) for p in now_off])
        want = np.array([(now_off[p], now_def[p]) for p in now_off])
        return np.abs((got - got.mean(0)) - (want - want.mean(0))).max()

    assert fit(aging=True) == pytest.approx(0, abs=1e-3)
    assert fit(aging=False) > 0.05


def test_rest_is_fit_and_predicted():
    """
    A team on the second night of a back-to-back scores 2 fewer points per
    100 possessions and allows 3 more: with `rest`, the fit learns both,
    and a game's prediction moves by them
    """
    offense, defense = ratings()
    con = duckdb.connect()
    tables(con, offense, defense)
    con.execute("""CREATE TEMP TABLE rest AS
        WITH t AS (
            SELECT season, game_id, game_date, team_id FROM results
            UNION ALL SELECT season, game_id, game_date, opp_team_id FROM results
        )
        SELECT game_id, team_id, date_diff('day', lag(game_date) OVER (
            PARTITION BY season, team_id ORDER BY game_date), game_date) - 1 AS rest
        FROM t""")
    assert con.execute("SELECT count(*) FROM rest WHERE rest = 0").fetchall()[0][0]
    con.execute("""UPDATE all_segments s SET pts = pts + poss / 100 * (
        - 2 * (SELECT coalesce(rest, 9) = 0 FROM rest r
            WHERE r.game_id = s.game_id AND r.team_id = s.off_team_id)::INT
        + 3 * (SELECT coalesce(rest, 9) = 0 FROM rest r
            WHERE r.game_id = s.game_id AND r.team_id = s.def_team_id)::INT)""")
    # the game to predict: A on a back-to-back, B rested
    con.execute("INSERT INTO rest VALUES ('x', 'A', 0), ('x', 'B', 1), ('y', 'A', 1)")
    model = PlayerRatings(
        half_life=1e9,
        ridge=1e-6,
        season_ridge=1,
        level_half_life=1e9,
        prior_weight=0,
        tempo_half_life=1e9,
        tempo_ridge=1e-6,
        minutes="actual",
        aging=False,
        rest=1,
    )
    assert model.settings["rest"] == 1
    model.fit(History(con, date(2024, 6, 1), 2024))
    assert model.rest_effects == pytest.approx(np.array([[-2], [3]]), abs=1e-3)

    for game_id in "xy":
        con.execute(
            "INSERT INTO player_games SELECT 2024, ?, '2024-06-01', p[1], p, 48 "
            "FROM unnest(['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3', "
            "'B4']) t(p)",
            [game_id],
        )
    tired, rested = model.predict(
        [
            Game(g, 2024, date(2024, 6, 1), "regular_season", "A", "B", True)
            for g in "xy"
        ]
    )
    assert tired.pts - rested.pts == pytest.approx(-2, abs=1e-3)
    assert tired.opp_pts - rested.opp_pts == pytest.approx(3, abs=1e-3)


def test_refit_day_by_day_matches_a_fresh_fit(monkeypatch):
    """
    A model refit every day keeps its rows and normal equations (`Window`)
    and updates them: the same ratings as a model fit fresh on each day,
    as rows are added and the last season's age out
    """
    # solved closely, so the two differ only by rounding
    monkeypatch.setattr(rapm, "TOLERANCE", 1e-13)
    now_off, now_def = ratings()
    rng = np.random.default_rng(5)
    con = duckdb.connect()
    tables(con, now_off, now_def)
    tables(con, *ratings(seed=6), seed=7, season=2023, create=False)
    con.execute("CREATE TEMP TABLE bio (player_id VARCHAR, born DOUBLE)")
    con.executemany(
        "INSERT INTO bio VALUES (?, ?)",
        [(p, float(rng.integers(1985, 2004))) for p in now_off],
    )

    def model():
        # 8 half lives of 4 days: the 2023 games age out in 2024's first weeks
        return PlayerRatings(
            half_life=4,
            ridge=50,
            season_ridge=100,
            level_half_life=2,
            prior_weight=0,
            tempo_half_life=1e9,
            tempo_ridge=1e-6,
            minutes="actual",
            aging_ridge=1,
        )

    daily = model()
    days = [date(2023, 11, 1) + timedelta(days=i) for i in range(0, 36, 3)]
    rows = []
    for day in days:
        history = History(con, day, 2024)
        daily.fit(history)
        if day == days[0]:
            window = daily.window
        assert daily.window is window and window is not None
        # 2023's rows: its season's column
        rows.append(
            int(window.rows.covariates[:, 1 + window.seasons.index(2023)].sum())
        )
        fresh = model()
        fresh.fit(history)
        assert daily.ratings is not None and fresh.ratings is not None
        players = sorted(now_off)
        got = np.array([daily.ratings.rating(p) for p in players])
        want = np.array([fresh.ratings.rating(p) for p in players])
        assert got == pytest.approx(want, abs=1e-6)
        assert daily.level == pytest.approx(fresh.level, abs=1e-6)
        assert daily.home == pytest.approx(fresh.home, abs=1e-6)
        assert daily.curve == pytest.approx(fresh.curve, abs=1e-6)
    # the last season's rows aged out, and this season's were added
    assert rows[0] > 0 and rows[-1] == 0
    assert len(window.rows.poss) > 0
