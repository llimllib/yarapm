"""
Baseline models for the harness (yarapm.harness).

HomeCourt: every team scores the league average, plus or minus half the
home-court edge. The floor every other model has to clear.

TeamRatings: KenPom-style adjusted offense, defense and tempo per team. The
model the player ratings have to beat.
"""

from typing import Any

import numpy as np

from yarapm.fit import decay, index, weighted_ridge
from yarapm.harness import Game, History, Prediction
from yarapm.tempo import PLAYOFF_TYPES, Tempo

# games older than this many half lives (weights under 1/256) are left out
# of a fit
HALF_LIVES = 8


def recent(history: History, half_life: float) -> dict[str, np.ndarray]:
    """The results that still carry weight, with each game's age in days"""
    types = ", ".join(f"'{t}'" for t in PLAYOFF_TYPES)
    return (
        history.results.select(f"""
        date_diff('day', game_date, DATE '{history.cutoff}') AS age,
        team_id, opp_team_id, coalesce(home, false) AS home,
        game_type IN ({types}) AS playoffs,
        pts, opp_pts, poss, opp_poss, min
    """)
        .filter(f"age <= {HALF_LIVES * half_life}")
        .fetchnumpy()
    )


class HomeCourt:
    """
    League average points per team, weighting older games less, plus or
    minus half the home team's average margin over its own (longer) half
    life
    """

    name = "home"

    def __init__(self, half_life: float = 60, home_half_life: float = 730):
        self.half_life = half_life
        self.home_half_life = home_half_life
        self.average = 0.0
        self.edge = 0.0

    @property
    def settings(self) -> dict[str, Any]:
        return {"half_life": self.half_life, "home_half_life": self.home_half_life}

    def fit(self, history: History) -> None:
        r = recent(history, max(self.half_life, self.home_half_life))
        age = np.asarray(r["age"], dtype=float)
        pts = np.asarray(r["pts"], dtype=float)
        opp_pts = np.asarray(r["opp_pts"], dtype=float)
        w = decay(age, self.half_life)
        self.average = float(np.sum(w * (pts + opp_pts)) / (2 * np.sum(w)))
        home = np.asarray(r["home"], dtype=bool)
        w = decay(age[home], self.home_half_life)
        self.edge = float(np.sum(w * (pts[home] - opp_pts[home])) / np.sum(w))

    def predict(self, games: list[Game]) -> list[Prediction]:
        half = self.edge / 2
        return [
            Prediction(self.average + half, self.average - half)
            if g.home
            else Prediction(self.average, self.average)
            for g in games
        ]


class TeamRatings:
    """
    Points per 100 possessions for each team's offense in a game: the league
    average + the offense's rating + the defense's rating (points allowed,
    relative to average) ± half the home edge + a playoff term, fit by
    weighted ridge regression over both teams' offenses in every game, older
    games weighing less. Ratings are pulled toward average by `ridge`, in
    units of games at full weight. Expected score: efficiency × adjusted
    tempo (yarapm.tempo) / 100

    The defaults are the best of a grid on the tuning seasons (#5), weekly
    refits: efficiency by margin RMSE (half life 30-270 days, ridge 1-100)
    and tempo by possession RMSE (30-365 days, ridge 1-30). Both are flat
    near the best: half lives of 60-120 days and ridges of 1-10 are within
    0.05 points of margin RMSE
    """

    name = "team"

    def __init__(
        self,
        half_life: float = 90,
        ridge: float = 3,
        tempo_half_life: float = 60,
        tempo_ridge: float = 3,
    ):
        self.half_life = half_life
        self.ridge = ridge
        self.tempo = Tempo(tempo_half_life, tempo_ridge)
        self.teams: dict[str, int] = {}
        self.coef = np.zeros(3)

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "half_life": self.half_life,
            "ridge": self.ridge,
            "tempo_half_life": self.tempo.half_life,
            "tempo_ridge": self.tempo.ridge,
        }

    def fit(self, history: History) -> None:
        r = recent(history, max(self.half_life, self.tempo.half_life))
        age = np.asarray(r["age"], dtype=float)
        team, opp = list(r["team_id"]), list(r["opp_team_id"])
        playoffs = np.asarray(r["playoffs"], dtype=float)
        poss = np.asarray(r["poss"], dtype=float)
        opp_poss = np.asarray(r["opp_poss"], dtype=float)
        self.tempo.fit(
            age,
            team,
            opp,
            (poss + opp_poss) / 2,
            np.asarray(r["min"], dtype=float),
            playoffs,
        )

        # two rows per game, one for each offense: intercept, home (+1 for
        # the home offense, -1 for the away one), playoffs, offenses, defenses
        self.teams = index(team + opp)
        n, k = len(team), len(self.teams)
        X = np.zeros((2 * n, 3 + 2 * k))
        X[:, 0] = 1
        home = np.asarray(r["home"], dtype=float)
        X[:, 1] = np.r_[home, -home]
        X[:, 2] = np.r_[playoffs, playoffs]
        rows = np.arange(2 * n)
        offense = np.array([self.teams[t] for t in team + opp])
        defense = np.array([self.teams[t] for t in opp + team])
        X[rows, 3 + offense] = 1
        X[rows, 3 + k + defense] = 1
        y = 100 * np.r_[r["pts"] / poss, r["opp_pts"] / opp_poss]
        w = np.tile(decay(age, self.half_life), 2)
        penalty = np.r_[0, 0, 0, np.full(2 * k, self.ridge)]
        self.coef = weighted_ridge(X, y, w, penalty)

    def offense(self, team: str) -> float:
        i = self.teams.get(team)
        return 0.0 if i is None else float(self.coef[3 + i])

    def defense(self, team: str) -> float:
        i = self.teams.get(team)
        return 0.0 if i is None else float(self.coef[3 + len(self.teams) + i])

    def efficiency(self, team: str, opp: str, home: float, playoffs: bool) -> float:
        return float(
            self.coef[0]
            + home * self.coef[1]
            + playoffs * self.coef[2]
            + self.offense(team)
            + self.defense(opp)
        )

    def predict(self, games: list[Game]) -> list[Prediction]:
        predictions = []
        for g in games:
            playoffs = g.game_type in PLAYOFF_TYPES
            home = 1.0 if g.home else 0.0
            poss = self.tempo.predict(g.team_id, g.opp_team_id, playoffs)
            predictions.append(
                Prediction(
                    self.efficiency(g.team_id, g.opp_team_id, home, playoffs)
                    * poss
                    / 100,
                    self.efficiency(g.opp_team_id, g.team_id, -home, playoffs)
                    * poss
                    / 100,
                    poss,
                )
            )
        return predictions
