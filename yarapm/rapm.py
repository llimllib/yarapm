"""
Ridge RAPM (#6): each player's offensive and defensive rating from the lineup
segments (yarapm.segments).

One row per segment and offense. The target is the offense's points per 100
possessions; the inputs are the five offensive players (+1 each), the five
defenders (-1 each, so a positive defensive rating is good), and covariates
such as a league average per season, home court and a playoff term. Rows
are weighted by possessions × a weight per row (the decay on the game's
age, say).

The ridge penalty pulls each player toward a prior: the prior's prediction
is subtracted from the target, the fit is pulled toward zero, and the prior
is added back. A zero prior gives plain RAPM. Penalties are in possessions
at full weight: a player with `ridge` possessions is pulled halfway from
their data toward their prior, roughly.

The normal equations are sparse and well conditioned by the penalty, so they
are solved by conjugate gradients, started from the previous solution when
refitting every game day. They're kept in `Normal`, which rows can be added
to and taken from, and whose weights can all be scaled at once, so a fit
refit every day can update them rather than build them again
(yarapm.players).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from scipy.sparse.linalg import cg

# relative tolerance of the conjugate gradient solve
TOLERANCE = 1e-8


@dataclass
class Lineups:
    """
    The rows to fit: `off` and `dfn` are (n, 5) indices into `players`,
    `covariates` is (n, c)
    """

    players: list[str]
    off: np.ndarray
    dfn: np.ndarray
    poss: np.ndarray
    pts: np.ndarray
    weight: np.ndarray
    covariates: np.ndarray

    @classmethod
    def from_ids(
        cls,
        off_ids: pa.Array | list[list[str]],
        def_ids: pa.Array | list[list[str]],
        poss: np.ndarray,
        pts: np.ndarray,
        weight: np.ndarray,
        covariates: np.ndarray,
    ) -> Lineups:
        """Rows from lists of five player ids per side"""
        n = len(poss)
        flat = pa.concat_arrays(
            [
                pa.array(off_ids, pa.list_(pa.string())).flatten(),
                pa.array(def_ids, pa.list_(pa.string())).flatten(),
            ]
        ).dictionary_encode()
        codes = np.asarray(flat.indices, dtype=np.int64)
        if len(codes) != 10 * n:
            raise ValueError("every row needs five players a side")
        covariates = np.asarray(covariates, dtype=float)
        # (n, c), and with no rows, (0, c) if c is known
        width = covariates.shape[1] if covariates.ndim == 2 else -1
        covariates = covariates.reshape(n, width if n or width >= 0 else 0)
        return cls(
            players=flat.dictionary.to_pylist(),
            off=codes[: 5 * n].reshape(n, 5),
            dfn=codes[5 * n :].reshape(n, 5),
            poss=np.asarray(poss, dtype=float),
            pts=np.asarray(pts, dtype=float),
            weight=np.asarray(weight, dtype=float),
            covariates=covariates,
        )


@dataclass
class Fit:
    """
    A fit's ratings, points per 100 possessions: `offense` (points added)
    and `defense` (points prevented) for each player, and the covariates'
    coefficients
    """

    players: dict[str, int]
    offense: np.ndarray
    defense: np.ndarray
    coef: np.ndarray

    def rating(self, player: str) -> tuple[float, float]:
        """(offense, defense), or zeros for a player the fit hasn't seen"""
        i = self.players.get(player)
        return (0.0, 0.0) if i is None else (self.offense[i], self.defense[i])

    def predict(self, rows: Lineups) -> np.ndarray:
        """Each row's expected points per 100 possessions"""
        index = np.array([self.players.get(p, -1) for p in rows.players])
        offense = np.r_[self.offense, 0][index]
        defense = np.r_[self.defense, 0][index]
        return (
            rows.covariates @ self.coef
            + offense[rows.off].sum(axis=1)
            - defense[rows.dfn].sum(axis=1)
        )


def design(rows: Lineups) -> sp.csr_matrix:
    """Covariates, then each player's offense column, then their defense"""
    n, c = rows.covariates.shape
    k = len(rows.players)
    cov_cols = np.tile(np.arange(c), n)
    cols = np.c_[cov_cols.reshape(n, c), c + rows.off, c + k + rows.dfn]
    vals = np.c_[rows.covariates, np.ones((n, 5)), -np.ones((n, 5))]
    return sp.csr_matrix(
        (vals.ravel(), cols.ravel(), np.arange(0, n * (c + 10) + 1, c + 10)),
        shape=(n, c + 2 * k),
    )


def players_design(off: np.ndarray, dfn: np.ndarray, k: int) -> sp.csr_matrix:
    """
    The players' columns of `design`, interleaved, for rows of `off` and
    `dfn` among `k` players: player i's offense is column 2i and their
    defense 2i + 1, so a new player's columns go at the end
    """
    n = len(off)
    cols = np.c_[2 * off, 2 * dfn + 1]
    vals = np.c_[np.ones((n, 5)), -np.ones((n, 5))]
    return sp.csr_matrix(
        (vals.ravel(), cols.ravel(), np.arange(0, n * 10 + 1, 10)),
        shape=(n, 2 * k),
    )


# rows at a time in `Normal.add`, to bound the copies it makes
CHUNK = 200_000


class Normal:
    """
    The normal equations of a fit with `covariates` covariates, without the
    prior or the penalty: X'WX and X'W·rate over the rows added and not
    taken away, by blocks. The covariates are dense, so their block is a
    dense product and only the players' block, ten entries a row, is a
    sparse one; the players' columns are interleaved (`players_design`), so
    players can be added. Building X'WX from `design`, with its covariates
    stored as sparse entries (zeros included), costs the square of the
    row's length.

    The prior enters at the solve: X'W(rate − prior) is X'W·rate less the
    players' columns of X'WX times the prior
    """

    def __init__(self, covariates: int):
        self.c = covariates
        # each player's index, in the order they were added
        self.players: dict[str, int] = {}
        self.cc = np.zeros((covariates, covariates))
        self.pc = np.zeros((0, covariates))
        self.pp = sp.csr_matrix((0, 0))
        self.bc = np.zeros(covariates)
        self.bp = np.zeros(0)

    def codes(self, players: list[str]) -> np.ndarray:
        """Each of `players`' index, adding the ones not yet seen"""
        out = np.empty(len(players), dtype=np.int64)
        for i, p in enumerate(players):
            j = self.players.get(p)
            if j is None:
                j = self.players[p] = len(self.players)
            out[i] = j
        k2 = 2 * len(self.players)
        if k2 > len(self.bp):
            grow = k2 - len(self.bp)
            self.pc = np.r_[self.pc, np.zeros((grow, self.c))]
            self.bp = np.r_[self.bp, np.zeros(grow)]
            self.pp.resize((k2, k2))
        return out

    def add(
        self,
        off: np.ndarray,
        dfn: np.ndarray,
        covariates: np.ndarray,
        w: np.ndarray,
        rate: np.ndarray,
    ) -> None:
        """
        Rows of `off` and `dfn`, indices from `codes`, with weights `w`
        (negative to take rows away) and points per 100 possessions `rate`.
        CHUNK rows at a time, so the weighted copies stay small
        """
        k = len(self.players)
        for i in range(0, len(w), CHUNK):
            s = slice(i, i + CHUNK)
            C = covariates[s]
            P = players_design(off[s], dfn[s], k)
            WC = C * w[s, None]
            self.cc += C.T @ WC
            self.pc += P.T @ WC
            self.pp = self.pp + (P.T @ P.multiply(w[s, None]).tocsr())
            wy = w[s] * rate[s]
            self.bc += C.T @ wy
            self.bp += P.T @ wy

    def scale(self, factor: float) -> None:
        """Every row's weight × `factor`"""
        self.cc *= factor
        self.pc *= factor
        self.pp = self.pp * factor
        self.bc *= factor
        self.bp *= factor

    def solve(
        self,
        ridge: float,
        covariate_penalty: np.ndarray,
        prior_offense: np.ndarray,
        prior_defense: np.ndarray,
        start: Fit | None = None,
    ) -> Fit:
        """
        `fit`'s solve, for the players in the order of `players`, without
        the points in rows without possessions
        """
        c = self.c
        k = len(self.players)
        prior = np.empty(2 * k)
        prior[0::2] = prior_offense
        prior[1::2] = prior_defense
        b = np.r_[self.bc - self.pc.T @ prior, self.bp - self.pp @ prior]
        penalty = np.r_[
            np.asarray(covariate_penalty, dtype=float), np.full(2 * k, ridge)
        ]
        # a tiny penalty everywhere, so a column of zeros gets a coefficient of 0
        A = sp.bmat(
            [
                [sp.csr_matrix(self.cc), sp.csr_matrix(self.pc.T)],
                [sp.csr_matrix(self.pc), self.pp],
            ]
        ).tocsr() + sp.diags(penalty + 1e-9)
        precondition = sp.diags(1 / A.diagonal())
        x0 = None
        if start is not None and len(start.coef) == c:
            # a player new to this fit starts at their prior
            x0 = np.r_[start.coef, np.zeros(2 * k)]
            for p, i in self.players.items():
                j = start.players.get(p)
                if j is not None:
                    x0[c + 2 * i] = start.offense[j] - prior_offense[i]
                    x0[c + 2 * i + 1] = start.defense[j] - prior_defense[i]
        x, info = cg(A, b, x0=x0, rtol=TOLERANCE, M=precondition, maxiter=10_000)
        if info != 0:
            raise RuntimeError(f"conjugate gradients didn't converge ({info})")
        return Fit(
            players=dict(self.players),
            offense=prior_offense + x[c::2],
            defense=prior_defense + x[c + 1 :: 2],
            coef=x[:c].copy(),
        )


def weights(
    poss: np.ndarray, pts: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rows' weights in the fit, possessions × `weight` (0 without
    possessions), and their points per 100 possessions
    """
    has = poss > 0
    w = np.where(has, poss * weight, 0.0)
    rate = np.divide(100 * pts, poss, out=np.zeros(len(w)), where=has)
    return w, rate


def loose_points(
    covariates: np.ndarray, poss: np.ndarray, pts: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """
    What the points in rows without possessions add to each covariate's
    coefficient: to every column that is all ones, as a rate on the
    possessions that are fit
    """
    out = np.zeros(covariates.shape[1])
    loose = np.flatnonzero(poss == 0)
    if len(loose) and len(loose) < len(poss):
        intercept = (covariates.min(axis=0) == 1) & (covariates.max(axis=0) == 1)
        # rows without possessions add nothing to the dot product
        out[intercept] = 100 * np.dot(weight[loose], pts[loose]) / np.dot(poss, weight)
    return out


def fit(
    rows: Lineups,
    ridge: float,
    covariate_penalty: np.ndarray,
    prior_offense: np.ndarray | None = None,
    prior_defense: np.ndarray | None = None,
    start: Fit | None = None,
) -> Fit:
    """
    Ratings minimizing the weighted squared error in points per 100
    possessions, with each player pulled toward their prior (arrays in the
    order of `rows.players`, zero if None) by `ridge`, and each covariate by
    its `covariate_penalty`. Rows without possessions carry no weight, so
    the points scored in them (free throws after a substitution, mostly)
    are added to every covariate column that is all ones, which should be
    the intercept: as a rate on the possessions that are fit.

    `start`, a previous fit with the same covariates, is where the solve
    starts, which makes refitting with a few more games fast
    """
    k = len(rows.players)
    prior_offense = np.zeros(k) if prior_offense is None else prior_offense
    prior_defense = np.zeros(k) if prior_defense is None else prior_defense
    # rows without possessions get no weight, rather than being copied out
    w, rate = weights(rows.poss, rows.pts, rows.weight)
    normal = Normal(rows.covariates.shape[1])
    # in order, so the players' indices are `rows`'
    normal.codes(rows.players)
    normal.add(rows.off, rows.dfn, rows.covariates, w, rate)
    out = normal.solve(ridge, covariate_penalty, prior_offense, prior_defense, start)
    out.coef += loose_points(rows.covariates, rows.poss, rows.pts, rows.weight)
    return out
