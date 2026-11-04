from datetime import UTC, date, datetime

import duckdb
import pytest

from yarapm.harness import Game, History, Prediction
from yarapm.predict import SIGMA, forecast, scheduled


def schedule(con, rows):
    """
    nba.schedule from (game_id, game_date, game_type, home_team_id,
    away_team_id, neutral_site)
    """
    con.execute("ATTACH ':memory:' AS nba")
    con.execute("""CREATE TABLE nba.schedule (season INTEGER, game_id VARCHAR,
        game_date DATE, start_time TIMESTAMPTZ, game_type VARCHAR,
        home_team_id VARCHAR, away_team_id VARCHAR, neutral_site BOOLEAN)""")
    con.executemany(
        """INSERT INTO nba.schedule VALUES
            (2027, ?, ?, ?::TIMESTAMPTZ, ?, ?, ?, ?)""",
        # 7pm Eastern
        [(r[0], r[1], f"{r[1]} 19:00:00-05", *r[2:]) for r in rows],
    )


def test_scheduled_games():
    con = duckdb.connect()
    schedule(
        con,
        [
            ("g1", date(2026, 11, 5), "regular_season", "H", "A", False),
            # listed with a home team at a neutral site: from the lower id's side
            ("g2", date(2026, 11, 5), "regular_season", "Z", "B", True),
            ("g3", date(2026, 11, 5), "preseason", "H", "A", False),
            # a Cup knockout game whose teams aren't set
            ("g4", date(2026, 11, 6), "regular_season", None, None, False),
            ("g5", date(2026, 11, 7), "regular_season", "H", "A", False),
        ],
    )
    games = scheduled(con, date(2026, 11, 5), 2)
    assert [g for g, _ in games] == [
        Game("g1", 2027, date(2026, 11, 5), "regular_season", "H", "A", True),
        Game("g2", 2027, date(2026, 11, 5), "regular_season", "B", "Z", None),
    ]
    assert games[0][1] == datetime(2026, 11, 6, 0, tzinfo=UTC)


class Fixed:
    """Predicts every game 110-100, and remembers what it was fit on"""

    name = "rapm"

    @property
    def settings(self) -> dict:
        return {}

    def fit(self, history: History) -> None:
        self.history = history

    def predict(self, games: list[Game]) -> list[Prediction]:
        return [Prediction(110.0, 100.0)] * len(games)


def test_forecast():
    con = duckdb.connect()
    games = [
        (Game("g1", 2027, date(2026, 11, 5), "regular_season", "H", "A", True), None)
    ]
    model = Fixed()
    (f,) = forecast(con, model, games, date(2026, 11, 4))
    assert (model.history.cutoff, model.history.season) == (date(2026, 11, 4), 2027)
    # a 10 point margin with the fixed spread
    assert SIGMA["rapm"] == 13.71
    assert f.win_prob == pytest.approx(0.7671, abs=1e-4)

    other = (
        Game("g2", 2028, date(2027, 10, 20), "regular_season", "H", "A", True),
        None,
    )
    with pytest.raises(ValueError, match="more than one season"):
        forecast(con, model, games + [other], date(2026, 11, 4))
