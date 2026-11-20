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

The box-score prior is fit to plain RAPM over the `target_seasons` seasons
before this one (`target`; None for every one). With play-by-play from
1997, every one is up to 30 seasons, and that fit's rows and memory grew
every year (2.1M rows for 2026). On the tuning seasons, daily, 10 is a tie
with every season: margin RMSE 13.507 vs 13.509 (known availability,
−0.002 ± 0.005 paired) and 13.507 vs 13.507 (unknown), total RMSE
18.357 vs 18.380 and 18.412 vs 18.435. Weekly, 6 was a tie too (13.517 vs
13.518 with 10 and 13.520 with every one, #32).

The other defaults are the best of a grid on the tuning seasons (#8), weekly
refits, by margin RMSE: half life 180-1460 calendar days, ridge 1000-30000, prior
weight 0.7-1.6, box half life 180-730 and level half life 7-60. It's flat
near the best: half lives of 365-730 days, ridges of 3000-10000 and prior
weights of 1-1.3 are within 0.03 points, so the simplest were kept. The
level half life only moves totals.
"""

from datetime import date, timedelta
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


# the number of teams, for the average games played (`PlayerRatings.games`),
# when the season's aren't known (`History.teams`)
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
    # the change for each player and season, (players, seasons, 3), then
    # gathered for each row: far cheaper than the basis for every row
    seasons, s = np.unique(season, return_inverse=True)
    change = np.nan_to_num(
        age_basis(now - b)[:, None, :] - age_basis(seasons[None, :] - b[:, None])
    )
    off = change[rows.off, s[:, None]].sum(axis=1)
    dfn = change[rows.dfn, s[:, None]].sum(axis=1)
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
    seasons: list[int] | None = None,
    since: date | None = None,
) -> tuple[rapm.Lineups, list[int], np.ndarray, np.ndarray]:
    """
    The segments matching `where` that still carry weight, oldest first,
    with covariates: an intercept, one column per season (`seasons`, or
    those in the rows), home and playoffs, with `rest`, each team's days
    off capped at it (`rest_covariates`), and with `aging` (each player's
    season of birth), the aging curve's (`aging_covariates`). Returns the
    seasons, each row's age in days, and its age for weighting: the same,
    or with `offseason` false, without the breaks between seasons. `since`
    leaves out the games before it, read from fewer files than `where`
    """
    types = ", ".join(f"'{t}'" for t in PLAYOFF_TYPES)
    segments = history.rested_segments if rest else history.segments
    if since is not None:
        segments = segments.filter(f"game_date >= DATE '{since}'")
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
        .order("age DESC")
        .to_arrow_table()
    )
    season = t["season"].to_numpy()
    age = t["age"].to_numpy().astype(float)
    weight_age = t["weight_age"].to_numpy().astype(float)
    if seasons is None:
        seasons = sorted(set(season.tolist()))
    elif not set(season.tolist()) <= set(seasons):
        raise ValueError(f"rows from seasons not in {seasons}")
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


# room for this share more rows than a `Window` is built with, so that its
# buffers are rarely moved
WINDOW_ROOM = 0.25


class Window:
    """
    A season's fit's rows and normal equations, kept so that the next fit,
    on a later cutoff in the same season, only adds the games since and
    takes away the ones that have aged out, rather than building them
    again: every row's age for weighting grows by the same number of days
    between two cutoffs in a season, so every weight is scaled by the same
    factor. Building them was most of each day's fit.

    The covariates are fixed for the season: a column for each season in
    the rows when it's built, and this one, whose column is all zeros (and
    so its coefficient 0) until its first games. A season whose rows have
    all aged out keeps its column, all zeros. Players whose rows have
    aged out keep theirs too, rated at their prior.

    The rows, oldest first, are kept in buffers with room for more, and
    `rows`, `age` and `weight_age` are views of them: rows that age out
    move the start, and new ones are written after the end. Copying every
    row into new arrays each day instead grew the process's memory by
    about 100MB a day, as freed arrays of slightly different sizes weren't
    reused
    """

    def __init__(
        self,
        history: History,
        half_life: float,
        offseason: bool,
        aging: bool,
        rest: int,
    ):
        self.half_life = half_life
        self.offseason = offseason
        self.aging = aging
        self.rest = rest
        self.history = history
        rows, seasons, age, weight_age = lineups(
            history,
            half_life,
            offseason=offseason,
            aging=born(history) if aging else None,
            rest=rest,
        )
        if history.season is not None and history.season not in seasons:
            at = 1 + sum(s < history.season for s in seasons)
            rows.covariates = np.insert(rows.covariates, at, 0.0, axis=1)
            seasons = sorted([*seasons, history.season])
        self.seasons = seasons
        self.normal = rapm.Normal(rows.covariates.shape[1])
        # in order, so the rows' indices are the fit's
        self.normal.codes(rows.players)
        self.normal.add(
            rows.off,
            rows.dfn,
            rows.covariates,
            *rapm.weights(rows.poss, rows.pts, rows.weight),
        )
        self.buffers: dict[str, np.ndarray] = {}
        self.start = self.end = 0
        self.store(
            {
                "off": rows.off,
                "dfn": rows.dfn,
                "poss": rows.poss,
                "pts": rows.pts,
                "weight": rows.weight,
                "covariates": rows.covariates,
                "age": age,
                "weight_age": weight_age,
            }
        )

    def store(self, arrays: dict[str, np.ndarray]) -> None:
        """New buffers holding `arrays`, with room for more"""
        n = len(arrays["poss"])
        room = n + int(WINDOW_ROOM * n) + 1000
        self.buffers = {
            k: np.empty((room, *a.shape[1:]), dtype=a.dtype) for k, a in arrays.items()
        }
        for k, a in arrays.items():
            self.buffers[k][:n] = a
        self.start, self.end = 0, n
        self.views()

    def views(self) -> None:
        b = {k: v[self.start : self.end] for k, v in self.buffers.items()}
        self.rows = rapm.Lineups(
            players=list(self.normal.players),
            off=b["off"],
            dfn=b["dfn"],
            poss=b["poss"],
            pts=b["pts"],
            weight=b["weight"],
            covariates=b["covariates"],
        )
        self.age = b["age"]
        self.weight_age = b["weight_age"]

    def append(self, drop: int, arrays: dict[str, np.ndarray]) -> None:
        """Drop the oldest `drop` rows and add `arrays` after the newest"""
        start, end = self.start + drop, self.end
        n = len(arrays["poss"])
        room = len(self.buffers["poss"])
        if end + n > room:
            if end - start + n > room:
                self.store(
                    {
                        k: np.concatenate([v[start:end], arrays[k]])
                        for k, v in self.buffers.items()
                    }
                )
                return
            # move the rows to the front, a block at a time so the copies
            # don't overlap
            keep = end - start
            for k, v in self.buffers.items():
                for i in range(0, keep, start):
                    j = min(i + start, keep)
                    v[i:j] = v[start + i : start + j]
            start, end = 0, keep
        for k, v in self.buffers.items():
            v[end : end + n] = arrays[k]
        self.start, self.end = start, end + n
        self.views()

    def shift(self, history: History) -> int | None:
        """
        How many days every row's age for weighting grows from this
        window's cutoff to `history`'s, or None if it isn't the same for
        every row: probed at the oldest row's game day and the day before
        the cutoff
        """
        old = self.history
        oldest = old.cutoff - timedelta(days=int(self.age.max(initial=1)))
        probes = f"(DATE '{oldest}'), (DATE '{old.cutoff - timedelta(days=1)}')"
        shifts = {
            a - b
            for (a,), (b,) in zip(
                *(
                    h.con.sql(
                        f"SELECT {h.age(self.offseason)} "
                        f"FROM (VALUES {probes}) v(game_date)"
                    ).fetchall()
                    for h in (history, old)
                ),
                strict=True,
            )
        }
        return shifts.pop() if len(shifts) == 1 else None

    def reusable(self, history: History) -> bool:
        old = self.history
        return (
            history.season is not None
            and history.con is old.con
            and history.season == old.season
            and history.cutoff >= old.cutoff
        )

    def advance(self, history: History, shift: int) -> None:
        """
        Move the window to `history`'s cutoff, `shift` days on for
        weighting: scale every weight, take away the rows that have aged
        out, and add the games since
        """
        rows = self.rows
        days = (history.cutoff - self.history.cutoff).days
        factor = float(decay(np.array(shift), self.half_life))
        self.normal.scale(factor)
        rows.weight *= factor
        self.weight_age += shift
        self.age += days

        # the oldest rows, first, as they're ordered by age
        limit = HALF_LIVES * self.half_life
        old = int(np.count_nonzero(self.weight_age > limit))
        if not (self.weight_age[:old] > limit).all():
            raise ValueError("the window's rows aren't oldest first")
        if old:
            s = slice(0, old)
            w, rate = rapm.weights(rows.poss[s], rows.pts[s], rows.weight[s])
            self.normal.add(rows.off[s], rows.dfn[s], rows.covariates[s], -w, rate)

        new, _, age, weight_age = lineups(
            history,
            self.half_life,
            offseason=self.offseason,
            aging=born(history) if self.aging and days else None,
            rest=self.rest,
            seasons=self.seasons,
            since=self.history.cutoff,
        )
        if self.aging and not len(new.poss):
            # no games since: no aging columns either, but the same width
            new.covariates = np.zeros((0, rows.covariates.shape[1]))
        codes = self.normal.codes(new.players)
        off, dfn = codes[new.off], codes[new.dfn]
        self.normal.add(
            off, dfn, new.covariates, *rapm.weights(new.poss, new.pts, new.weight)
        )
        self.append(
            old,
            {
                "off": off,
                "dfn": dfn,
                "poss": new.poss,
                "pts": new.pts,
                "weight": new.weight,
                "covariates": new.covariates,
                "age": age,
                "weight_age": weight_age,
            },
        )
        self.history = history


# rows older than this many of the league average's half lives are left out
# of it: their weights are under 2^-64 of the newest's
LEVEL_HALF_LIVES = 64


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
        target_seasons: int | None = 10,
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
        self.target_seasons = target_seasons
        # the rest terms' coefficients, offense and defense, (2, levels)
        self.rest_effects = np.zeros((2, len(rest_levels(rest))))
        self.history: History | None = None
        self.window: Window | None = None
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
            | ({"target_seasons": self.target_seasons} if self.target_seasons else {})
        )

    def fit(self, history: History) -> None:
        self.projection.fit(history)
        self.history = history
        ridge = self.ridge
        if self.loosen:
            ridge *= 1 + self.loosen * np.exp(-self.games(history) / self.loosen_games)
        window = self.window
        shift = None
        if window is not None and window.reusable(history):
            shift = window.shift(history)
        if window is not None and shift is not None:
            window.advance(history, shift)
        else:
            window = self.window = Window(
                history,
                max(self.half_life, self.level_half_life),
                self.offseason,
                self.aging,
                self.rest,
            )
        rows, seasons = window.rows, window.seasons
        aging_ridge = self.aging_ridge if self.aging else None
        offense = defense = np.zeros(len(rows.players))
        if self.prior_weight:
            self.prior.fit(history, lambda: self.target(history))
            priors = np.array(
                [self.prior.prior(p) for p in rows.players], dtype=float
            ).reshape(-1, 2)
            offense = self.prior_weight * priors[:, 0]
            defense = self.prior_weight * priors[:, 1]
        self.ratings = window.normal.solve(
            ridge,
            penalties(seasons, self.season_ridge, aging_ridge, self.rest),
            offense,
            defense,
            start=self.ratings,
        )
        self.ratings.coef += rapm.loose_points(
            rows.covariates, rows.poss, rows.pts, rows.weight
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
        # term leave over, recent games weighing more. The rows are oldest
        # first, so the recent ones are the last
        newest = window.age[-1] if len(window.age) else 0.0
        oldest = newest + LEVEL_HALF_LIVES * self.level_half_life
        latest = slice(int(np.searchsorted(-window.age, -oldest, "left")), None)
        n = 1 + len(seasons)
        others = (
            rows.covariates[latest, n:] @ coef[n:]
            + self.ratings.offense[rows.off[latest]].sum(axis=1)
            - self.ratings.defense[rows.dfn[latest]].sum(axis=1)
        )
        w = decay(window.age[latest], self.level_half_life)
        self.level = float(
            100
            * np.sum(w * (rows.pts[latest] - rows.poss[latest] * others / 100))
            / np.sum(w * rows.poss[latest])
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
        teams = history.teams()
        return 2 * n / (len(teams) if teams else TEAMS)

    def target(self, history: History) -> tuple[rapm.Fit, list[int]]:
        """
        Plain RAPM over the `target_seasons` seasons before this one (every
        one, if None), unweighted, which the box-score prior is fit to
        """
        where = "true" if history.season is None else f"season < {history.season}"
        if history.season is not None and self.target_seasons:
            where += f" AND season >= {history.season - self.target_seasons}"
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
        settings.pop("target_seasons", None)
        return {"season": self.season} | settings
