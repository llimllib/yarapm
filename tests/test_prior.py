import numpy as np
import pytest

from yarapm.prior import RATES, Bio, Box, Formula, box_features


def test_rates_shrink_toward_players_of_similar_height():
    players = ["big", "small", "center"]
    poss = np.array([1e9, 1.0, 1e9])
    totals = {name: poss * 0.1 for name in [*RATES, "fga", "fg3m", "min"]}
    # the two centers rebound on 20% of possessions, then 5% in one game
    totals["oreb"] = np.array([0.2e9, 0.05, 0.2e9])
    bio = Bio(height={"big": 84, "small": 84, "center": 85})
    features = box_features(Box(players, poss, np.ones(3), totals), bio)
    oreb = list(RATES).index("oreb")
    assert features[0, oreb] == pytest.approx(20)
    # one possession barely moves a rate from the bucket's
    assert features[1, oreb] == pytest.approx(20, abs=0.1)


def test_formula_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 3)) * [1, 10, 100] + [0, 5, 50]
    offense = 1 + X @ [2, 0.1, 0]
    defense = -1 + X @ [0, 0, 0.01]
    w = np.full(len(X), 1e6)
    formula = Formula.fit(X, offense, defense, w)
    o, d = formula.predict(X[:10])
    assert o == pytest.approx(offense[:10], abs=1e-3)
    assert d == pytest.approx(defense[:10], abs=1e-3)
