"""
PlayerRatings: each team's expected score from its players' ratings (#8).

Ratings are ridge RAPM (yarapm.rapm) on the segments before the fit date,
with older games weighted less (`half_life`, in days) and each player
pulled by `ridge` possessions toward their box-score prior
(yarapm.prior) × `prior_weight`: 0 is plain RAPM. Players without segments
are rated at their prior. The covariates are a league average, an offset
for each season pulled toward it by `season_ridge` possessions, home court
(+1 for the home offense, -1 for the away one) and a playoff term.

The league average used to predict is the points per 100 possessions left
over by everything else (ratings, home court and the playoff term), with
older games weighted less by its own, shorter half life,
`level_half_life`: scoring drifts within a season, by up to several points
a game after the trade deadline, and the season offsets lag it.

A team's offense in a game, per 100 possessions: the league average ± home
court + the playoff term + its players' offensive ratings −
the opponents' defensive ratings, each player weighted by their share of
the team's minutes × 5, so the weights add up to the five on the floor.
Expected score: that × adjusted tempo (yarapm.tempo) / 100.

Minutes come from yarapm.minutes, by `minutes`: "actual" (the minutes
played, a leak that measures rating error on its own), "known" (who plays
is known, minutes projected) or "unknown" (who plays is projected too),
with `minutes_half_life` and `start_bonus`.

The defaults are the best of a grid on the tuning seasons (#8), weekly
refits, by margin RMSE: half life 180-1460 days, ridge 1000-30000, prior
weight 0.7-1.6, box half life 180-730 and level half life 7-60. It's flat
near the best: half lives of 365-730 days, ridges of 3000-10000 and prior
weights of 1-1.3 are within 0.03 points, so the simplest were kept. The
level half life only moves totals.
"""

from typing import Any

import numpy as np

from yarapm import rapm
from yarapm.baselines import HALF_LIVES, recent
from yarapm.fit import decay
from yarapm.harness import Game, History, Prediction
from yarapm.minutes import ActualMinutes, ProjectedMinutes, expected
from yarapm.prior import BoxPrior, OpenRapmPrior
from yarapm.tempo import PLAYOFF_TYPES, Tempo

# the ridge of the plain RAPM the box-score prior is fit to, in possessions
TARGET_RIDGE = 3000


def lineups(
    history: History, half_life: float, where: str = "true"
) -> tuple[rapm.Lineups, list[int], np.ndarray]:
    """
    The segments matching `where` that still carry weight, with covariates:
    an intercept, one column per season, home and playoffs. Returns the
    seasons, and each row's age in days
    """
    types = ", ".join(f"'{t}'" for t in PLAYOFF_TYPES)
    t = (
        history.segments.select(f"""
        season, date_diff('day', game_date, DATE '{history.cutoff}') AS age,
        off_player_ids, def_player_ids, poss, pts,
        CASE WHEN off_home THEN 1 WHEN NOT off_home THEN -1 ELSE 0 END AS home,
        game_type IN ({types}) AS playoffs
    """)
        .filter(f"age <= {HALF_LIVES * half_life} AND ({where})")
        .to_arrow_table()
    )
    season = t["season"].to_numpy()
    age = t["age"].to_numpy().astype(float)
    seasons = sorted(set(season.tolist()))
    covariates = np.c_[
        np.ones(len(season)),
        season[:, None] == np.array(seasons)[None, :],
        t["home"].to_numpy(),
        t["playoffs"].to_numpy(),
    ]
    rows = rapm.Lineups.from_ids(
        t["off_player_ids"].combine_chunks(),
        t["def_player_ids"].combine_chunks(),
        t["poss"].to_numpy(),
        t["pts"].to_numpy(),
        decay(age, half_life),
        covariates,
    )
    return rows, seasons, age


def penalties(seasons: list[int], season_ridge: float) -> np.ndarray:
    """The covariates': only the season offsets are penalized"""
    return np.r_[0, np.full(len(seasons), season_ridge), 0, 0]


class PlayerRatings:
    name = "rapm"

    def __init__(
        self,
        half_life: float = 365,
        ridge: float = 3000,
        season_ridge: float = 20000,
        level_half_life: float = 7,
        prior_weight: float = 1,
        box_half_life: float = 365,
        tempo_half_life: float = 60,
        tempo_ridge: float = 3,
        minutes: str = "known",
        minutes_half_life: float = 5,
        start_bonus: float = 5,
    ):
        self.half_life = half_life
        self.ridge = ridge
        self.season_ridge = season_ridge
        self.level_half_life = level_half_life
        self.prior_weight = prior_weight
        self.box_half_life = box_half_life
        self.prior: BoxPrior | OpenRapmPrior = BoxPrior(box_half_life)
        self.tempo = Tempo(tempo_half_life, tempo_ridge)
        self.minutes = minutes
        self.projection: ActualMinutes | ProjectedMinutes = (
            ActualMinutes()
            if minutes == "actual"
            else ProjectedMinutes(minutes, minutes_half_life, start_bonus)
        )
        self.ratings: rapm.Fit | None = None
        self.level = 0.0
        self.home = 0.0
        self.playoffs = 0.0

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "half_life": self.half_life,
            "ridge": self.ridge,
            "season_ridge": self.season_ridge,
            "level_half_life": self.level_half_life,
            "prior_weight": self.prior_weight,
            "box_half_life": self.box_half_life,
            "tempo_half_life": self.tempo.half_life,
            "tempo_ridge": self.tempo.ridge,
            "minutes": self.minutes,
        } | (
            {}
            if isinstance(self.projection, ActualMinutes)
            else {
                "minutes_half_life": self.projection.half_life,
                "start_bonus": self.projection.start_bonus,
            }
        )

    def fit(self, history: History) -> None:
        self.projection.fit(history)
        rows, seasons, age = lineups(history, max(self.half_life, self.level_half_life))
        offense = defense = None
        if self.prior_weight:
            self.prior.fit(history, lambda: self.target(history))
            priors = np.array([self.prior.prior(p) for p in rows.players])
            offense = self.prior_weight * priors[:, 0]
            defense = self.prior_weight * priors[:, 1]
        self.ratings = rapm.fit(
            rows,
            self.ridge,
            penalties(seasons, self.season_ridge),
            offense,
            defense,
            start=self.ratings,
        )
        coef = self.ratings.coef
        self.home, self.playoffs = coef[-2], coef[-1]

        # the league average: what the ratings, home court and the playoff
        # term leave over, recent games weighing more
        levels = rows.covariates[:, : 1 + len(seasons)] @ coef[: 1 + len(seasons)]
        rest = self.ratings.predict(rows) - levels
        w = decay(age, self.level_half_life)
        self.level = float(
            100
            * np.sum(w * (rows.pts - rows.poss * rest / 100))
            / np.sum(w * rows.poss)
        )

        r = recent(history, self.tempo.half_life)
        self.tempo.fit(
            np.asarray(r["age"], dtype=float),
            list(r["team_id"]),
            list(r["opp_team_id"]),
            (np.asarray(r["poss"], dtype=float) + np.asarray(r["opp_poss"])) / 2,
            np.asarray(r["min"], dtype=float),
            np.asarray(r["playoffs"], dtype=float),
        )

    def target(self, history: History) -> tuple[rapm.Fit, list[int]]:
        """
        Plain RAPM over every season before this one, unweighted, which the
        box-score prior is fit to
        """
        where = "true" if history.season is None else f"season < {history.season}"
        rows, seasons, _ = lineups(history, 1e12, where)
        return rapm.fit(
            rows, TARGET_RIDGE, penalties(seasons, self.season_ridge)
        ), seasons

    def rating(self, player: str) -> tuple[float, float]:
        """A player's ratings, or their prior if they have no segments"""
        assert self.ratings is not None
        if player in self.ratings.players or not self.prior_weight:
            return self.ratings.rating(player)
        o, d = self.prior.prior(player)
        return self.prior_weight * o, self.prior_weight * d

    def strength(self, minutes: dict[str, float]) -> tuple[float, float]:
        """
        A team's offense and defense: its players' ratings weighted by their
        share of its minutes × 5
        """
        total = sum(minutes.values())
        offense = defense = 0.0
        if total == 0:
            return offense, defense
        for player, m in minutes.items():
            o, d = self.rating(player)
            offense += 5 * m / total * o
            defense += 5 * m / total * d
        return offense, defense

    def predict(self, games: list[Game]) -> list[Prediction]:
        minutes = self.projection.predict(games)

        predictions = []
        for g in games:
            playoffs = g.game_type in PLAYOFF_TYPES
            home = 1.0 if g.home else 0.0
            base = self.level + playoffs * self.playoffs
            off, dfn = self.strength(minutes.get((g.game_id, g.team_id), {}))
            opp_off, opp_dfn = self.strength(
                minutes.get((g.game_id, g.opp_team_id), {})
            )
            poss = self.tempo.predict(g.team_id, g.opp_team_id, playoffs)
            predictions.append(
                Prediction(
                    (base + home * self.home + off - opp_dfn) * poss / 100,
                    (base - home * self.home + opp_off - dfn) * poss / 100,
                    poss,
                    expected(minutes, g),
                )
            )
        return predictions


class OpenRapm(PlayerRatings):
    """
    OpenRAPM's ratings through the same prediction step (#9): the ridge
    holds every player at their OpenRAPM rating, so only the league
    average, home court and the playoff term are fit. `season` is "last"
    (fair: earlier games only) or "this" (fit on the whole season, so an
    optimistic ceiling)
    """

    name = "openrapm"

    def __init__(self, season: str = "last", **settings: Any):
        settings.setdefault("ridge", 1e9)
        settings.setdefault("prior_weight", 1)
        super().__init__(**settings)
        self.season = season
        self.prior = OpenRapmPrior(season)

    @property
    def settings(self) -> dict[str, Any]:
        settings = super().settings
        del settings["box_half_life"]
        return {"season": self.season} | settings
