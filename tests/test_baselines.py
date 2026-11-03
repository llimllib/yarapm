from datetime import date, timedelta
from itertools import permutations

import duckdb
import numpy as np
import pytest

from yarapm.baselines import TeamRatings
from yarapm.harness import Game, History
from yarapm.tempo import Tempo

TEAMS = ["A", "B", "C", "D"]
OFFENSE = {"A": 6, "B": 2, "C": -3, "D": -5}
DEFENSE = {"A": -4, "B": 0, "C": 1, "D": 3}  # points allowed above average
PACE = {"A": 0.05, "B": 0.0, "C": -0.02, "D": -0.03}  # log effects
HOME_EDGE = 3  # per 100 possessions, for the home offense and against the away one


def test_tempo_recovers_team_effects():
    pairs = list(permutations(TEAMS, 2)) * 5
    team, opp = [p[0] for p in pairs], [p[1] for p in pairs]
    # every fifth game goes to overtime; possessions scale with the minutes
    minutes = np.array([53.0 if i % 5 == 0 else 48.0 for i in range(len(pairs))])
    poss = np.array([100 * np.exp(PACE[t] + PACE[o]) for t, o in pairs]) * minutes / 48
    tempo = Tempo(half_life=1e9, ridge=1e-6)
    tempo.fit(np.zeros(len(pairs)), team, opp, poss, minutes, np.zeros(len(pairs)))
    # effects are identified relative to the league's, so compare differences
    assert tempo.effect("A") - tempo.effect("D") == pytest.approx(0.08)
    length = (4 * 48 + 53) / 5 / 48
    assert tempo.predict("A", "B", False) == pytest.approx(
        100 * np.exp(0.05) * length, rel=1e-3
    )


def test_team_ratings_recover_offense_and_defense():
    con = duckdb.connect()
    rows = []
    start = date(2023, 11, 1)
    for i, (team, opp) in enumerate(permutations(TEAMS, 2)):
        for week in range(4):
            poss = 100 * np.exp(PACE[team] + PACE[opp])
            pts = poss * (110 + OFFENSE[team] + DEFENSE[opp] + HOME_EDGE) / 100
            opp_pts = poss * (110 + OFFENSE[opp] + DEFENSE[team] - HOME_EDGE) / 100
            day = start + timedelta(days=7 * week + i)
            rows.append(
                (
                    2024,
                    f"g{i}-{week}",
                    day,
                    "regular_season",
                    team,
                    opp,
                    True,
                    pts,
                    opp_pts,
                    poss,
                    poss,
                    48.0,
                )
            )
    con.execute("""CREATE TEMP TABLE results (season INTEGER, game_id VARCHAR,
        game_date DATE, game_type VARCHAR, team_id VARCHAR, opp_team_id VARCHAR,
        home BOOLEAN, pts DOUBLE, opp_pts DOUBLE, poss DOUBLE, opp_poss DOUBLE,
        min DOUBLE)""")
    con.executemany(
        "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )

    model = TeamRatings(
        half_life=1e9, ridge=1e-6, tempo_half_life=1e9, tempo_ridge=1e-6
    )
    model.fit(History(con, date(2024, 1, 1)))
    assert model.offense("A") - model.offense("D") == pytest.approx(11, abs=1e-3)
    assert model.defense("D") - model.defense("A") == pytest.approx(7, abs=1e-3)

    game = Game("x", 2024, date(2024, 1, 1), "regular_season", "A", "D", True)
    (prediction,) = model.predict([game])
    poss = 100 * np.exp(PACE["A"] + PACE["D"])
    assert prediction.poss == pytest.approx(poss, rel=1e-3)
    assert prediction.pts == pytest.approx(poss * (110 + 6 + 3 + 3) / 100, rel=1e-3)
    assert prediction.opp_pts == pytest.approx(poss * (110 - 5 - 4 - 3) / 100, rel=1e-3)
