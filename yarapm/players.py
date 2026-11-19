"""
PlayerRatings: each team's expected score from its players' ratings (#8).

Ratings are ridge RAPM (yarapm.rapm) on the segments before the fit date,
with older games weighted less (`half_life`, in days of the season, see
below) and each player pulled by `ridge` possessions toward their
box-score prior (yarapm.prior) × `prior_weight`: 0 is plain RAPM.
Players without segments are rated at their prior. The covariates are a league average, an offset
for each season pulled toward it by `season_ridge` possessions, home court
(+1 for the home offense, -1 for the away one), a playoff term and
back-to-backs (`rest`, below).

The league average used to predict is the points per 100 possessions left
over by everything else (ratings, home court, the playoff term and
rest), with older games weighted less by its own, shorter half life,
`level_half_life`: scoring drifts within a season, by up to several points
a game after the trade deadline, and the season offsets lag it.

A team's offense in a game, per 100 possessions: the league average ± home
court + the playoff term + the rest terms + its players' offensive
ratings − the opponents' defensive ratings, each player weighted by
their share of the team's minutes × 5, so the weights add up to the five on the floor.
Expected score: that × adjusted tempo (yarapm.tempo) / 100.

Minutes come from yarapm.minutes, by `minutes`: "actual" (the minutes
played, a leak that measures rating error on its own), "known" (who plays
is known, minutes projected) or "unknown" (who plays is projected too),
with `minutes_half_life` and `start_bonus`. With unknown availability,
`report` says whether the NBA's injury report (#52) sets the chance of
playing, and `injuries` whether ESPN's injury list (#45) does for the games
the report doesn't cover; both false is the pattern alone (#46).

A game's age counts only days in a season, regular season to Finals:
the breaks between seasons (about 128 days) are skipped, so the summer
doesn't pull every player toward their prior before a new season's first
game (#43). `offseason=True` counts calendar days, as before #43. On the
tuning seasons the two predict the same (daily margin RMSE 13.570 with
240 in-season days, 13.568 with 365 calendar days); in-season ageing was
chosen so that a player's rating doesn't drop over a summer with no
games. 240 days, about one season, keeps about the memory of 365
calendar days.

With `aging` (#48), players age: their rating in an earlier season was
their rating now less the change in an aging curve (offense and defense,
each a cubic in age, `age_basis`) between their age then and now. The
curve's six coefficients are covariates of the same fit, penalized by
`aging_ridge`, so it's learned from how players' ratings moved between
the seasons in the fit, and each rating is the player's as of this
season. Only how players age relative to each other is identified: a
change shared by everyone goes to the season offsets. On the tuning
seasons it lowers daily margin RMSE from 13.570 to 13.536 (known
availability), most early in the season (13.475 to 13.405), where the
ratings are mostly last season's. `aging_ridge` from 1000 to 100000
scores the same. Shifting the box-score prior along the curve as well
(by the average age of the box scores it's made of) was worse, 13.605
weekly against 13.549.

Early in a season, the ridge toward the prior is larger (#50): `ridge` ×
(1 + `loosen` × exp(−n / `loosen_games`)), n the teams' average games so
far this season. So until this season's games arrive, the evidence
carried over from earlier seasons counts for less, and a new season
starts with every player pulled harder toward their prior. Without it,
early predictions were spread too wide (the actual margin's slope on the
predicted was 0.83 in the first 20 games, against 0.91-1.00 later). On
the tuning seasons, daily, it lowers margin RMSE from 13.536 to 13.518
(known availability, 2.3 paired SEs), early from 13.405 to 13.341, and
the early slope rises to 0.95. A weekly grid of `loosen` 1-10 and
`loosen_games` 5-60 is flat near the best (3-5 and 10-30 are within
0.003). Scaling each team's strength toward the league's by 1 − a ×
exp(−n / n0), a over 0.1-0.4 and n0 over 5-40 games, was a tie at best
(−0.011 ± 0.007 weekly).

Rest (#21) is a covariate of the same fit: with `rest` = 1, a term for
the offense being on the second night of a back-to-back and one for the
defense being on it (each team's days off, from the schedule:
yarapm.harness.load_rest). Larger values add a term for each number of
days off from 2 to `rest`, measured from one day; 0 leaves rest out. The
fit puts a back-to-back at about −0.65 points per 100 possessions on
offense and +1.0 on defense, about 1.7 points of margin. On the tuning
seasons, daily, it lowers margin RMSE from 13.518 to 13.500 (known
availability, −0.019 ± 0.009 paired) and from 13.662 to 13.630 with
unknown availability (−0.033 ± 0.009: teams rest players on
back-to-backs, which the projected minutes don't foresee); in the games
with one team on a back-to-back, from 13.36 to 13.29 (known). Weekly,
terms for 2 and 3 days off as well were no better (−0.017 and −0.016,
against −0.020 for back-to-backs alone), their coefficients near zero.

The other defaults are the best of a grid on the tuning seasons (#8), weekly
refits, by margin RMSE: half life 180-1460 calendar days, ridge 1000-30000, prior
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


# the number of teams, for the average games played (`PlayerRatings.games`)
TEAMS = 30

# the aging curve's inputs: age clamped to AGE_RANGE, centered at AGE_CENTER
AGE_RANGE = (19, 38)
AGE_CENTER = 27


def age_basis(age: np.ndarray) -> np.ndarray:
    """
    The aging curve's basis, (..., 3): a cubic in age, so a player's rating
    changes by an amount that is quadratic in their age from one season to
    the next
    """
    a = np.clip(age, *AGE_RANGE) - AGE_CENTER
    return np.stack([a, a**2 / 10, a**3 / 100], axis=-1)


def born(history: History) -> dict[str, float]:
    """Each player's season of birth: the season they were 0 years old"""
    return {
        p: float(b)
        for p, b in history.bio.select("player_id, born").fetchall()
        if b is not None
    }


def aging_covariates(
    rows: rapm.Lineups, season: np.ndarray, born: dict[str, float], now: int
) -> np.ndarray:
    """
    Six columns, the aging curve's for offense then defense: in a row from
    an earlier season, each player was younger than they are in season
    `now`, and their rating then was their rating now less the change in
    the curve between the two ages. Players without an age don't age
    """
    b = np.array([born.get(p, np.nan) for p in rows.players])
    off = age_basis(now - b[rows.off]) - age_basis(season[:, None] - b[rows.off])
    dfn = age_basis(now - b[rows.dfn]) - age_basis(season[:, None] - b[rows.dfn])
    off = np.nan_to_num(off).sum(axis=1)
    dfn = np.nan_to_num(dfn).sum(axis=1)
    return np.c_[-off, dfn]


def rest_levels(cap: int) -> list[int]:
    """
    The days of rest that get a covariate, capped at `cap`: every level but
    1, the usual day off between games, which the others are measured from
    """
    return [0, *range(2, cap + 1)] if cap else []


def rest_covariates(off_rest: np.ndarray, def_rest: np.ndarray, cap: int) -> np.ndarray:
    """
    2 × len(`rest_levels`) columns, the offense's then the defense's: 1
    where the team had that many days off before the game, capped at `cap`.
    NaN, a team's first game of a season, counts as rested (`cap`)
    """
    levels = np.array(rest_levels(cap))

    def side(rest: np.ndarray) -> np.ndarray:
        r = np.minimum(np.nan_to_num(np.asarray(rest, dtype=float), nan=cap), cap)
        return (r[:, None] == levels[None, :]).astype(float)

    return np.c_[side(off_rest), side(def_rest)]


def lineups(
    history: History,
    half_life: float,
    where: str = "true",
    offseason: bool = True,
    aging: dict[str, float] | None = None,
    rest: int = 0,
) -> tuple[rapm.Lineups, list[int], np.ndarray, np.ndarray]:
    """
    The segments matching `where` that still carry weight, with covariates:
    an intercept, one column per season, home and playoffs, with `rest`,
    each team's days off capped at it (`rest_covariates`), and with
    `aging` (each player's season of birth), the aging curve's
    (`aging_covariates`). Returns the seasons, each row's age in days, and
    its age for weighting: the same, or with `offseason` false, without the
    breaks between seasons
    """
    types = ", ".join(f"'{t}'" for t in PLAYOFF_TYPES)
    segments = history.rested_segments if rest else history.segments
    t = (
        segments.select(f"""
        season, date_diff('day', game_date, DATE '{history.cutoff}') AS age,
        {history.age(offseason)} AS weight_age,
        off_player_ids, def_player_ids, poss, pts,
        CASE WHEN off_home THEN 1 WHEN NOT off_home THEN -1 ELSE 0 END AS home,
        game_type IN ({types}) AS playoffs
        {", off_rest, def_rest" if rest else ""}
    """)
        .filter(f"weight_age <= {HALF_LIVES * half_life} AND ({where})")
        .to_arrow_table()
    )
    season = t["season"].to_numpy()
    age = t["age"].to_numpy().astype(float)
    weight_age = t["weight_age"].to_numpy().astype(float)
    seasons = sorted(set(season.tolist()))
    covariates = np.c_[
        np.ones(len(season)),
        season[:, None] == np.array(seasons)[None, :],
        t["home"].to_numpy(),
        t["playoffs"].to_numpy(),
    ]
    if rest:
        covariates = np.c_[
            covariates,
            rest_covariates(
                t["off_rest"].to_numpy(zero_copy_only=False),
                t["def_rest"].to_numpy(zero_copy_only=False),
                rest,
            ),
        ]
    rows = rapm.Lineups.from_ids(
        t["off_player_ids"].combine_chunks(),
        t["def_player_ids"].combine_chunks(),
        t["poss"].to_numpy(),
        t["pts"].to_numpy(),
        decay(weight_age, half_life),
        covariates,
    )
    if aging is not None and len(season):
        now = history.season if history.season is not None else max(seasons)
        rows.covariates = np.c_[
            rows.covariates, aging_covariates(rows, season, aging, now)
        ]
    return rows, seasons, age, weight_age


def penalties(
    seasons: list[int],
    season_ridge: float,
    aging_ridge: float | None = None,
    rest: int = 0,
) -> np.ndarray:
    """
    The covariates': the season offsets are penalized, and with
    `aging_ridge`, the aging curve's. The rest terms aren't
    """
    aging = [] if aging_ridge is None else [aging_ridge] * 6
    return np.r_[
        0,
        np.full(len(seasons), season_ridge),
        0,
        0,
        np.zeros(2 * len(rest_levels(rest))),
        aging,
    ]


class PlayerRatings:
    name = "rapm"

    def __init__(
        self,
        half_life: float = 240,
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
        offseason: bool = False,
        box_offseason: bool = True,
        injuries: bool = True,
        report: bool = True,
        aging: bool = True,
        aging_ridge: float = 1e4,
        loosen: float = 3.0,
        loosen_games: float = 20.0,
        rest: int = 1,
    ):
        self.half_life = half_life
        self.offseason = offseason
        self.ridge = ridge
        self.season_ridge = season_ridge
        self.level_half_life = level_half_life
        self.prior_weight = prior_weight
        self.box_half_life = box_half_life
        self.prior: BoxPrior | OpenRapmPrior = BoxPrior(box_half_life, box_offseason)
        self.tempo = Tempo(tempo_half_life, tempo_ridge)
        self.minutes = minutes
        self.projection: ActualMinutes | ProjectedMinutes = (
            ActualMinutes()
            if minutes == "actual"
            else ProjectedMinutes(
                minutes, minutes_half_life, start_bonus, injuries, report
            )
        )
        self.aging = aging
        self.aging_ridge = aging_ridge
        self.loosen = loosen
        self.loosen_games = loosen_games
        self.rest = rest
        # the rest terms' coefficients, offense and defense, (2, levels)
        self.rest_effects = np.zeros((2, len(rest_levels(rest))))
        self.history: History | None = None
        # the aging curves' coefficients, offense and defense, (2, 3)
        self.curve: np.ndarray | None = None
        self.ratings: rapm.Fit | None = None
        self.level = 0.0
        self.home = 0.0
        self.playoffs = 0.0

    @property
    def settings(self) -> dict[str, Any]:
        return (
            {
                "half_life": self.half_life,
                "ridge": self.ridge,
                "season_ridge": self.season_ridge,
                "level_half_life": self.level_half_life,
                "prior_weight": self.prior_weight,
                "box_half_life": self.box_half_life,
                "tempo_half_life": self.tempo.half_life,
                "tempo_ridge": self.tempo.ridge,
                "minutes": self.minutes,
            }
            | (
                # only when false, so calendar-day runs from before #43
                # keep their ids
                {} if self.offseason else {"offseason": False}
            )
            | (
                {}
                if not isinstance(self.prior, BoxPrior) or self.prior.offseason
                else {"box_offseason": False}
            )
            | (
                {}
                if isinstance(self.projection, ActualMinutes)
                else {
                    "minutes_half_life": self.projection.half_life,
                    "start_bonus": self.projection.start_bonus,
                }
                # only when false, so runs from before #46 keep their ids
                | ({} if self.projection.use_injuries else {"injuries": False})
                # only when true, so runs from before #52 keep their ids
                | ({"report": True} if self.projection.use_report else {})
            )
            | ({"aging": True, "aging_ridge": self.aging_ridge} if self.aging else {})
            | (
                {"loosen": self.loosen, "loosen_games": self.loosen_games}
                if self.loosen
                else {}
            )
            | ({"rest": self.rest} if self.rest else {})
        )

    def fit(self, history: History) -> None:
        self.projection.fit(history)
        self.history = history
        ridge = self.ridge
        if self.loosen:
            ridge *= 1 + self.loosen * np.exp(-self.games(history) / self.loosen_games)
        births = born(history) if self.aging else None
        rows, seasons, age, _ = lineups(
            history,
            max(self.half_life, self.level_half_life),
            offseason=self.offseason,
            aging=births,
            rest=self.rest,
        )
        aging_ridge = self.aging_ridge if self.aging else None
        offense = defense = None
        if self.prior_weight:
            self.prior.fit(history, lambda: self.target(history))
            priors = np.array([self.prior.prior(p) for p in rows.players])
            offense = self.prior_weight * priors[:, 0]
            defense = self.prior_weight * priors[:, 1]
        self.ratings = rapm.fit(
            rows,
            ridge,
            penalties(seasons, self.season_ridge, aging_ridge, self.rest),
            offense,
            defense,
            start=self.ratings,
        )
        coef = self.ratings.coef
        self.home, self.playoffs = coef[1 + len(seasons)], coef[2 + len(seasons)]
        if self.aging:
            self.curve = coef[-6:].reshape(2, 3)
        if self.rest:
            n = 2 * len(rest_levels(self.rest))
            self.rest_effects = coef[3 + len(seasons) : 3 + len(seasons) + n].reshape(
                2, -1
            )

        # the league average: what the ratings, home court and the playoff
        # term leave over, recent games weighing more
        levels = rows.covariates[:, : 1 + len(seasons)] @ coef[: 1 + len(seasons)]
        others = self.ratings.predict(rows) - levels
        w = decay(age, self.level_half_life)
        self.level = float(
            100
            * np.sum(w * (rows.pts - rows.poss * others / 100))
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

    @staticmethod
    def games(history: History) -> float:
        """The teams' average number of games so far this season"""
        (n,) = (
            history.results.filter(f"season = {history.season}")
            .aggregate("count(*)")
            .fetchall()[0]
        )
        return 2 * n / TEAMS

    def target(self, history: History) -> tuple[rapm.Fit, list[int]]:
        """
        Plain RAPM over every season before this one, unweighted, which the
        box-score prior is fit to
        """
        where = "true" if history.season is None else f"season < {history.season}"
        rows, seasons, _, _ = lineups(history, 1e12, where)
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

    def rested(self, rest: int | None, opp_rest: int | None) -> float:
        """
        What a team's days off before a game and its opponent's add to its
        offense, per 100 possessions
        """
        if not self.rest:
            return 0.0
        cols = rest_covariates(
            np.array([np.nan if rest is None else rest]),
            np.array([np.nan if opp_rest is None else opp_rest]),
            self.rest,
        )[0]
        return float(cols @ self.rest_effects.ravel())

    def predict(self, games: list[Game]) -> list[Prediction]:
        minutes = self.projection.predict(games)
        days = (
            self.history.rest([g.game_id for g in games])
            if self.rest and self.history is not None
            else {}
        )

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
            rest = days.get((g.game_id, g.team_id), 1)
            opp_rest = days.get((g.game_id, g.opp_team_id), 1)
            rested = self.rested(rest, opp_rest)
            opp_rested = self.rested(opp_rest, rest)
            predictions.append(
                Prediction(
                    (base + home * self.home + rested + off - opp_dfn) * poss / 100,
                    (base - home * self.home + opp_rested + opp_off - dfn) * poss / 100,
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
        # OpenRAPM's ratings are its own: they don't age, and the ridge
        # holds them whatever its size
        settings.setdefault("aging", False)
        settings.setdefault("loosen", 0)
        # and are compared without rest (#21), as they were before it
        settings.setdefault("rest", 0)
        super().__init__(**settings)
        self.season = season
        self.prior = OpenRapmPrior(season)

    @property
    def settings(self) -> dict[str, Any]:
        settings = super().settings
        del settings["box_half_life"]
        return {"season": self.season} | settings
