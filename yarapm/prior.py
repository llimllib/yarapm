"""
The box-score prior (#7): each player's offensive and defensive rating as of
a date, estimated from their box scores, which RAPM (yarapm.rapm) pulls
toward.

Rates: per 100 of the player's possessions, from nba.player_game_logs back
to 2010, with older games weighted less (`half_life`, in days), each shrunk
toward the average of players of similar height (a stand-in for position,
which we don't have) by STABILIZE: lightly for rates that settle quickly
(rebounds) and more for slow ones (steals, three-point percentage).

Two linear formulas, refit at the start of each season on earlier seasons
only, so nothing from the season being predicted goes into them:

- The box formula: offense and defense, separately, from the rates, usage ×
  true shooting, minutes per game, height and draft position. Fit to
  plain RAPM (a zero prior) over every season with segments before this
  one, each player weighted by possessions. For 2021, that's 2019-20
- The rookie formula: from height and draft position alone, for players
  with no box scores. Fit to the box formula's estimate of each rookie
  season from 2011 on (2010 is the first season of box scores, so its
  rookies aren't known)

A player's prior blends the two by their possessions: the box formula with
the weight of their (decayed) possessions and the rookie formula with
ROOKIE_POSSESSIONS', so the first games of a season don't swing it.

STABILIZE and ROOKIE_POSSESSIONS are rough guesses, not tuned.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from yarapm import rapm
from yarapm.fit import weighted_ridge
from yarapm.harness import History

# per 100 possessions: SQL for the count, and the possessions it takes to
# pull a rate halfway to the average
RATES: dict[str, tuple[str, float]] = {
    "pts": ("pts", 500),
    "fg2a": ("fga - fg3a", 200),
    "fg3a": ("fg3a", 200),
    "fta": ("fta", 300),
    "ast": ("ast", 200),
    "tov": ("tov", 300),
    "oreb": ("oreb", 150),
    "dreb": ("dreb", 150),
    "stl": ("stl", 600),
    "blk": ("blk", 300),
    "pf": ("pf", 300),
}
# true shooting, in shot attempts (fga + 0.44 fta); three-point percentage,
# in attempts; minutes per game, in games
STABILIZE_TS = 300
STABILIZE_FG3 = 300
STABILIZE_MPG = 10

# heights, in inches, splitting guards, wings, forwards and centers
HEIGHT_BUCKETS = np.array([76, 79, 82])
DEFAULT_HEIGHT = 79
UNDRAFTED = 61

ROOKIE_POSSESSIONS = 500

# the formulas' penalty, in possessions, on standardized inputs
FORMULA_RIDGE = 1000


@dataclass
class Box:
    """Each player's box-score totals, possibly weighted"""

    players: list[str]
    poss: np.ndarray
    games: np.ndarray
    totals: dict[str, np.ndarray]


def box_totals(history: History, half_life: float, where: str = "true") -> Box:
    """Totals per player over the games matching `where`, older ones weighted less"""
    w = f"0.5 ** (date_diff('day', game_date, DATE '{history.cutoff}') / {half_life})"
    sums = ", ".join(
        f"sum({w} * ({expr}))::DOUBLE AS {name}"
        for name, expr in {
            **{k: v[0] for k, v in RATES.items()},
            "fga": "fga",
            "fg3m": "fg3m",
            "min": "min",
        }.items()
    )
    t = (
        history.player_games.filter(f"poss > 0 AND ({where})")
        .aggregate(
            f"""player_id, sum({w} * poss)::DOUBLE AS poss,
                sum({w})::DOUBLE AS games, {sums}""",
            "player_id",
        )
        .order("player_id")
        .fetchnumpy()
    )
    names = [*RATES, "fga", "fg3m", "min"]
    return Box(
        players=list(t["player_id"]),
        poss=np.asarray(t["poss"], dtype=float),
        games=np.asarray(t["games"], dtype=float),
        totals={k: np.asarray(t[k], dtype=float) for k in names},
    )


@dataclass
class Bio:
    height: dict[str, float] = field(default_factory=dict)
    draft: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, history: History) -> Bio:
        bio = cls()
        for player, height, draft in history.bio.fetchall():
            if height is not None:
                bio.height[player] = float(height)
            if draft is not None:
                bio.draft[player] = float(draft)
        return bio

    def features(self, players: list[str]) -> np.ndarray:
        """Height, the log of the draft pick, and whether undrafted"""
        height = np.array([self.height.get(p, DEFAULT_HEIGHT) for p in players])
        pick = np.array([max(self.draft.get(p, UNDRAFTED), 1) for p in players])
        undrafted = np.array([p not in self.draft for p in players], dtype=float)
        return np.c_[height, np.log(pick), undrafted]


def box_features(box: Box, bio: Bio) -> np.ndarray:
    """
    The box formula's inputs, one row per player: shrunk rates, usage ×
    true shooting, minutes per game, then `Bio.features`
    """
    t = box.totals
    tsa = t["fga"] + 0.44 * t["fta"]
    usage = 100 * (tsa + t["tov"]) / box.poss
    height = np.array([bio.height.get(p, DEFAULT_HEIGHT) for p in box.players])
    bucket = np.searchsorted(HEIGHT_BUCKETS, height, side="right")

    def shrink(num: np.ndarray, den: np.ndarray, k: float) -> np.ndarray:
        """num / den, pulled toward the height bucket's by k of den"""
        avg = np.zeros(len(HEIGHT_BUCKETS) + 1)
        for b in range(len(avg)):
            i = bucket == b
            avg[b] = num[i].sum() / den[i].sum() if den[i].sum() > 0 else 0
        return (num + k * avg[bucket]) / (den + k)

    columns = [100 * shrink(t[name], box.poss, k) for name, (_, k) in RATES.items()]
    ts = shrink(t["pts"], 2 * tsa, STABILIZE_TS)
    columns += [
        ts,
        shrink(t["fg3m"], t["fg3a"], STABILIZE_FG3),
        usage * (ts - np.average(ts, weights=box.poss)),
        shrink(t["min"], box.games, STABILIZE_MPG),
    ]
    return np.c_[np.column_stack(columns), bio.features(box.players)]


@dataclass
class Formula:
    """A weighted ridge on standardized inputs, for offense and defense"""

    mean: np.ndarray
    scale: np.ndarray
    offense: np.ndarray
    defense: np.ndarray

    @classmethod
    def fit(
        cls, X: np.ndarray, offense: np.ndarray, defense: np.ndarray, w: np.ndarray
    ) -> Formula:
        mean = np.average(X, axis=0, weights=w)
        scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w))
        scale[scale == 0] = 1
        Z = np.c_[np.ones(len(X)), (X - mean) / scale]
        penalty = np.r_[0, np.full(X.shape[1], FORMULA_RIDGE)]
        return cls(
            mean,
            scale,
            weighted_ridge(Z, offense, w, penalty),
            weighted_ridge(Z, defense, w, penalty),
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = np.c_[np.ones(len(X)), (X - self.mean) / self.scale]
        return Z @ self.offense, Z @ self.defense


class BoxPrior:
    """
    `fit(history)` refits the formulas when the season changes and
    computes everyone's prior as of the cutoff; `prior(players)` gives it
    """

    def __init__(self, half_life: float = 365):
        self.half_life = half_life
        self.season: int | None = None
        self.box: Formula | None = None
        self.rookie: Formula | None = None
        self.bio = Bio()
        self.known: dict[str, tuple[float, float]] = {}

    def fit(
        self, history: History, target: Callable[[], tuple[rapm.Fit, list[int]]]
    ) -> None:
        """
        `target` gives plain RAPM over every season before this one, and
        those seasons, for the box formula; it's only called when the
        season changes
        """
        if self.box is None or history.season != self.season:
            self.bio = Bio.load(history)
            self.train(history, *target())
            self.season = history.season
        assert self.box is not None and self.rookie is not None

        box = box_totals(history, self.half_life)
        offense, defense = self.box.predict(box_features(box, self.bio))
        rookie_o, rookie_d = self.rookie.predict(self.bio.features(box.players))
        share = box.poss / (box.poss + ROOKIE_POSSESSIONS)
        offense = share * offense + (1 - share) * rookie_o
        defense = share * defense + (1 - share) * rookie_d
        self.known = {
            p: (float(o), float(d)) for p, o, d in zip(box.players, offense, defense)
        }

    def train(self, history: History, target: rapm.Fit, seasons: list[int]) -> None:
        # the box scores of the seasons RAPM was fit on, unweighted
        box = box_totals(
            history, 1e12, f"season IN ({', '.join(str(s) for s in seasons)})"
        )
        has = np.array([p in target.players for p in box.players])
        idx = [target.players[p] for p in np.array(box.players)[has]]
        X = box_features(box, self.bio)[has]
        self.box = Formula.fit(
            X, target.offense[idx], target.defense[idx], box.poss[has]
        )

        # each rookie season's box prior, from the seasons after the first
        # season of box scores
        rookies = history.player_games.aggregate(
            "player_id, min(season) AS first", "player_id"
        )
        first_box = history.player_games.aggregate("min(season)").fetchall()[0][0]
        firsts = [
            (p, s)
            for p, s in rookies.fetchall()
            if s > first_box and (history.season is None or s < history.season)
        ]
        X_rookie, o_rookie, d_rookie, w_rookie = [], [], [], []
        for season in sorted({s for _, s in firsts}):
            names = {p for p, s in firsts if s == season}
            box = box_totals(history, 1e12, f"season = {season}")
            keep = np.array([p in names for p in box.players])
            if not keep.any():
                continue
            o, d = self.box.predict(box_features(box, self.bio)[keep])
            X_rookie.append(self.bio.features(list(np.array(box.players)[keep])))
            o_rookie.append(o)
            d_rookie.append(d)
            w_rookie.append(box.poss[keep])
        self.rookie = Formula.fit(
            np.concatenate(X_rookie),
            np.concatenate(o_rookie),
            np.concatenate(d_rookie),
            np.concatenate(w_rookie),
        )

    def prior(self, player: str) -> tuple[float, float]:
        """A player's prior: from their box scores, or the rookie formula"""
        if player in self.known:
            return self.known[player]
        assert self.rookie is not None
        o, d = self.rookie.predict(self.bio.features([player]))
        return float(o[0]), float(d[0])


# OpenRAPM players with fewer possessions than this in a season set its
# replacement level, the rating of players it doesn't have
REPLACEMENT_POSSESSIONS = 1000


class OpenRapmPrior:
    """
    OpenRAPM's ratings (#9), from last season or (leaking) this one, with
    the season's replacement level for players it doesn't have: the
    average of its players with few possessions
    """

    def __init__(self, which: str):
        if which not in ("last", "this"):
            raise ValueError(f"OpenRAPM season {which!r} is 'last' or 'this'")
        self.which = which
        self.season: int | None = None
        self.known: dict[str, tuple[float, float]] = {}
        self.replacement = (0.0, 0.0)

    def fit(
        self, history: History, target: Callable[[], tuple[rapm.Fit, list[int]]]
    ) -> None:
        if history.season is None:
            raise ValueError("OpenRAPM's ratings need the season being predicted")
        if self.season == history.season:
            return
        self.season = history.season
        season = history.season - (1 if self.which == "last" else 0)
        rows = history.openrapm(season).select(
            "player_id, possessions, o_rating, d_rating"
        )
        self.known = {p: (o, d) for p, _, o, d in rows.fetchall()}
        (o, d) = (
            rows.filter(f"possessions < {REPLACEMENT_POSSESSIONS}")
            .aggregate("coalesce(avg(o_rating), 0), coalesce(avg(d_rating), 0)")
            .fetchall()[0]
        )
        self.replacement = (float(o), float(d))

    def prior(self, player: str) -> tuple[float, float]:
        return self.known.get(player, self.replacement)
