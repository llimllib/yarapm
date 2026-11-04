import numpy as np
import pytest

from yarapm import rapm

PLAYERS = 30
LEAGUE = 110.0
HOME = 1.5


def simulate(n=4000, seed=0):
    """
    Random lineups from a pool of players with known ratings, scoring their
    expected points exactly: (lineups, offense, defense)
    """
    rng = np.random.default_rng(seed)
    offense = rng.normal(0, 3, PLAYERS)
    defense = rng.normal(0, 2, PLAYERS)
    ids = [f"p{i:02}" for i in range(PLAYERS)]
    picks = np.array([rng.choice(PLAYERS, 10, replace=False) for _ in range(n)])
    off, dfn = picks[:, :5], picks[:, 5:]
    home = rng.choice([-1.0, 0.0, 1.0], n)
    poss = rng.integers(1, 12, n).astype(float)
    pts = (
        poss
        * (LEAGUE + HOME * home + offense[off].sum(axis=1) - defense[dfn].sum(axis=1))
        / 100
    )
    rows = rapm.Lineups.from_ids(
        [[ids[i] for i in r] for r in off],
        [[ids[i] for i in r] for r in dfn],
        poss,
        pts,
        np.ones(n),
        np.c_[np.ones(n), home],
    )
    # from_ids numbers players in order of appearance
    order = np.array([int(p[1:]) for p in rows.players])
    return rows, offense[order], defense[order]


def centered(x):
    return x - x.mean()


def test_recovers_ratings():
    rows, offense, defense = simulate()
    fit = rapm.fit(rows, ridge=1e-6, covariate_penalty=np.zeros(2))
    # ratings are identified up to a shift that the intercept absorbs
    assert centered(fit.offense) == pytest.approx(centered(offense), abs=1e-4)
    assert centered(fit.defense) == pytest.approx(centered(defense), abs=1e-4)
    assert fit.coef[1] == pytest.approx(HOME, abs=1e-4)
    assert fit.predict(rows) == pytest.approx(100 * rows.pts / rows.poss, abs=1e-4)


def test_ridge_pulls_toward_the_prior():
    rows, offense, defense = simulate()
    prior_offense = np.full(PLAYERS, 1.0)
    prior_defense = np.full(PLAYERS, -1.0)
    fit = rapm.fit(rows, 1e12, np.zeros(2), prior_offense, prior_defense)
    assert fit.offense == pytest.approx(prior_offense, abs=1e-4)
    assert fit.defense == pytest.approx(prior_defense, abs=1e-4)

    # a correct prior is already the answer, however strong the ridge
    fit = rapm.fit(rows, 1e4, np.zeros(2), offense, defense)
    assert fit.offense == pytest.approx(offense, abs=1e-4)
    assert fit.coef[0] == pytest.approx(LEAGUE, abs=1e-4)


def test_ridge_shrinks_plain_rapm_toward_zero():
    rows, _, _ = simulate()
    loose = rapm.fit(rows, 1e-6, np.zeros(2))
    tight = rapm.fit(rows, 1e5, np.zeros(2))
    assert np.abs(tight.offense).sum() < np.abs(centered(loose.offense)).sum() / 2


def test_warm_start_gives_the_same_fit():
    rows, _, _ = simulate()
    first = rapm.fit(rows, 100, np.zeros(2))
    again = rapm.fit(rows, 100, np.zeros(2), start=first)
    assert again.offense == pytest.approx(first.offense, abs=1e-6)
    assert again.coef == pytest.approx(first.coef, abs=1e-6)


def test_points_without_possessions_go_to_the_intercept():
    rows, _, _ = simulate()
    base = rapm.fit(rows, 1e-6, np.zeros(2))
    # two points on a free throw after a substitution, for each 100
    # possessions fit
    extra = rapm.Lineups.from_ids(
        [rows.players[:5]],
        [rows.players[5:10]],
        np.zeros(1),
        np.array([rows.poss.sum() / 50]),
        np.ones(1),
        np.array([[1.0, 0.0]]),
    )
    both = rapm.Lineups(
        rows.players,
        np.r_[rows.off, [[rows.players.index(p) for p in extra.players[:5]]]],
        np.r_[rows.dfn, [[rows.players.index(p) for p in extra.players[5:]]]],
        np.r_[rows.poss, extra.poss],
        np.r_[rows.pts, extra.pts],
        np.r_[rows.weight, extra.weight],
        np.r_[rows.covariates, extra.covariates],
    )
    fit = rapm.fit(both, 1e-6, np.zeros(2))
    assert fit.coef[0] == pytest.approx(base.coef[0] + 2, abs=1e-4)
    assert fit.offense == pytest.approx(base.offense, abs=1e-4)
