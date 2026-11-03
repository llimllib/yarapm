"""
Adjusted tempo, KenPom-style: a game's possessions per 48 minutes are the
league's tempo × each team's tempo relative to the league (and a factor for
the playoffs), fit on log possessions so the effects multiply. Each team's
effect is pulled toward the league's by `ridge`, and older games count less
(`half_life`, in days).

A game's possessions are the average of the two teams' (they differ by at
most 4), scaled to 48 minutes so overtime doesn't make a team look fast.
Predictions are scaled back up by the league's average game length, so they
include the average overtime.
"""

import numpy as np

from yarapm.fit import decay, index, weighted_ridge

PLAYOFF_TYPES = ("playoffs", "play_in")


class Tempo:
    def __init__(self, half_life: float = 90, ridge: float = 5):
        self.half_life = half_life
        self.ridge = ridge
        self.teams: dict[str, int] = {}
        self.coef = np.zeros(2)
        self.length = 1.0

    def fit(
        self,
        age: np.ndarray,
        team: list[str],
        opp: list[str],
        poss: np.ndarray,
        minutes: np.ndarray,
        playoffs: np.ndarray,
    ) -> None:
        """
        One row per game: its age in days, the two teams, the game's
        possessions (the two teams' average) and minutes, and whether it's a
        playoff game
        """
        self.teams = index(list(team) + list(opp))
        n, k = len(team), len(self.teams)
        X = np.zeros((n, k + 2))
        X[:, 0] = 1
        X[:, 1] = playoffs
        rows = np.arange(n)
        X[rows, 2 + np.array([self.teams[t] for t in team])] += 1
        X[rows, 2 + np.array([self.teams[t] for t in opp])] += 1
        w = decay(age, self.half_life)
        y = np.log(poss * 48 / minutes)
        self.coef = weighted_ridge(X, y, w, np.r_[0, 0, np.full(k, self.ridge)])
        self.length = float(np.sum(w * minutes / 48) / np.sum(w))

    def effect(self, team: str) -> float:
        i = self.teams.get(team)
        return 0.0 if i is None else float(self.coef[2 + i])

    def league(self) -> float:
        """The league's possessions per 48 minutes, in the regular season"""
        return float(np.exp(self.coef[0]))

    def predict(self, team: str, opp: str, playoffs: bool) -> float:
        """A game's expected possessions, for each team"""
        log = (
            self.coef[0]
            + playoffs * self.coef[1]
            + self.effect(team)
            + self.effect(opp)
        )
        return float(np.exp(log) * self.length)
