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
refitting every game day.
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
        return cls(
            players=flat.dictionary.to_pylist(),
            off=codes[: 5 * n].reshape(n, 5),
            dfn=codes[5 * n :].reshape(n, 5),
            poss=np.asarray(poss, dtype=float),
            pts=np.asarray(pts, dtype=float),
            weight=np.asarray(weight, dtype=float),
            covariates=np.asarray(covariates, dtype=float).reshape(n, -1),
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
    `design` without the covariates, for rows of `off` and `dfn` among `k`
    players: each player's offense, then their defense
    """
    n = len(off)
    cols = np.c_[off, k + dfn]
    vals = np.c_[np.ones((n, 5)), -np.ones((n, 5))]
    return sp.csr_matrix(
        (vals.ravel(), cols.ravel(), np.arange(0, n * 10 + 1, 10)),
        shape=(n, 2 * k),
    )


# rows at a time in `normal_equations`, to bound the copies it makes
CHUNK = 200_000


def normal_equations(
    rows: Lineups, w: np.ndarray, y: np.ndarray
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    X'WX and X'Wy for `design(rows)`, with weights `w`, by blocks: the
    covariates are dense, so their block is a dense product and only the
    players' block, ten entries a row, is a sparse one. Building X'WX from
    `design`, with its covariates stored as sparse entries (zeros included),
    costs the square of the row's length, and was most of the fit's time.
    CHUNK rows at a time, so the weighted copies stay small
    """
    c = rows.covariates.shape[1]
    k = len(rows.players)
    cc = np.zeros((c, c))
    pc = np.zeros((2 * k, c))
    pp = sp.csr_matrix((2 * k, 2 * k))
    b = np.zeros(c + 2 * k)
    for i in range(0, len(w), CHUNK):
        s = slice(i, i + CHUNK)
        C = rows.covariates[s]
        P = players_design(rows.off[s], rows.dfn[s], k)
        WC = C * w[s, None]
        cc += C.T @ WC
        pc += P.T @ WC
        pp = pp + (P.T @ P.multiply(w[s, None]).tocsr())
        wy = w[s] * y[s]
        b += np.r_[C.T @ wy, P.T @ wy]
    A = sp.bmat([[sp.csr_matrix(cc), sp.csr_matrix(pc.T)], [sp.csr_matrix(pc), pp]])
    return A.tocsr(), b


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
    has = rows.poss > 0
    w = np.where(has, rows.poss * rows.weight, 0.0)
    prior = prior_offense[rows.off].sum(axis=1) - prior_defense[rows.dfn].sum(axis=1)
    rate = np.divide(100 * rows.pts, rows.poss, out=np.zeros(len(w)), where=has)
    y = np.where(has, rate - prior, 0.0)

    c = rows.covariates.shape[1]
    penalty = np.r_[np.asarray(covariate_penalty, dtype=float), np.full(2 * k, ridge)]
    # a tiny penalty everywhere, so a column of zeros gets a coefficient of 0
    A, b = normal_equations(rows, w, y)
    A = A + sp.diags(penalty + 1e-9)
    precondition = sp.diags(1 / A.diagonal())
    x0 = None
    if start is not None and len(start.coef) == c:
        # a player new to this fit starts at their prior
        offense, defense = np.zeros(k), np.zeros(k)
        for i, p in enumerate(rows.players):
            j = start.players.get(p)
            if j is not None:
                offense[i] = start.offense[j] - prior_offense[i]
                defense[i] = start.defense[j] - prior_defense[i]
        x0 = np.r_[start.coef, offense, defense]
    x, info = cg(A, b, x0=x0, rtol=TOLERANCE, M=precondition, maxiter=10_000)
    if info != 0:
        raise RuntimeError(f"conjugate gradients didn't converge ({info})")

    coef = x[:c].copy()
    loose = rows.weight * (rows.poss == 0)
    if loose.any():
        intercept = np.all(rows.covariates == 1, axis=0)
        coef[intercept] += 100 * np.sum(loose * rows.pts) / np.sum(w)
    return Fit(
        players={p: i for i, p in enumerate(rows.players)},
        offense=prior_offense + x[c : c + k],
        defense=prior_defense + x[c + k :],
        coef=coef,
    )
