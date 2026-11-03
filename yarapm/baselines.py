"""
Baseline models for the harness (yarapm.harness).

HomeCourt: every team scores the league average, plus or minus half the
home-court edge. The floor every other model has to clear.
"""

from typing import Any

import numpy as np

from yarapm.harness import Game, History


def decay(ages: np.ndarray, half_life: float) -> np.ndarray:
    """Weights for games `ages` days old"""
    return 0.5 ** (ages / half_life)


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
        r = history.results.select(
            f"date_diff('day', game_date, DATE '{history.cutoff}') AS age, "
            "coalesce(home, false) AS home, pts, opp_pts"
        ).fetchnumpy()
        age = np.asarray(r["age"], dtype=float)
        pts = np.asarray(r["pts"], dtype=float)
        opp_pts = np.asarray(r["opp_pts"], dtype=float)
        w = decay(age, self.half_life)
        self.average = float(np.sum(w * (pts + opp_pts)) / (2 * np.sum(w)))
        home = np.asarray(r["home"], dtype=bool)
        w = decay(age[home], self.home_half_life)
        self.edge = float(np.sum(w * (pts[home] - opp_pts[home])) / np.sum(w))

    def predict(self, games: list[Game]) -> list[tuple[float, float]]:
        half = self.edge / 2
        return [
            (self.average + half, self.average - half)
            if g.home
            else (self.average, self.average)
            for g in games
        ]
