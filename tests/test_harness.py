from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from yarapm.baselines import HomeCourt
from yarapm.harness import EASTERN, Game, History, Prediction, fit_days, run, score


def utc(*args: int) -> datetime:
    """A time in UTC without a zone, as `History` has them"""
    return datetime(*args, tzinfo=UTC).replace(tzinfo=None)


def results(con, games):
    """
    Temp table `results` and view `all_segments` (one row per game) from
    (season, game_date, game_type, team_id, opp_team_id, home, pts, opp_pts)
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE results (season INTEGER,
        game_id VARCHAR, game_date DATE, game_type VARCHAR, team_id VARCHAR,
        opp_team_id VARCHAR, home BOOLEAN, pts INTEGER, opp_pts INTEGER,
        poss INTEGER, opp_poss INTEGER, min DOUBLE)""")
    con.executemany(
        "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 100, 100, 48)",
        [(g[0], f"g{i:04}", *g[1:]) for i, g in enumerate(games)],
    )
    con.execute(
        "CREATE OR REPLACE VIEW all_segments AS SELECT game_id, game_date FROM results"
    )


def season(year, start, days, teams=("A", "B", "C", "D")):
    """A game a day between rotating pairs of teams, home team scoring 110-100"""
    games = []
    for d in range(days):
        team, opp = teams[d % len(teams)], teams[(d + 1) % len(teams)]
        games.append(
            (
                year,
                start + timedelta(days=d),
                "regular_season",
                team,
                opp,
                True,
                110,
                100,
            )
        )
    return games


def test_fit_days():
    days = [date(2024, 10, d) for d in (22, 23, 28, 29, 30)]
    assert fit_days(days, "daily") == [[d] for d in days]
    assert fit_days(days, "weekly") == [days[:3], days[3:]]


class Spy:
    """Checks that a fit only sees games before the games it predicts"""

    name = "spy"

    @property
    def settings(self) -> dict:
        return {}

    def __init__(self):
        self.seen = date.min
        self.fits = 0

    def fit(self, history: History) -> None:
        self.fits += 1
        for table in (history.results, history.segments):
            (latest,) = table.aggregate("max(game_date)").fetchall()[0]
            self.seen = latest or date.min
            assert self.seen < history.cutoff

    def predict(self, games: list[Game]) -> list[Prediction]:
        assert all(self.seen < g.game_date for g in games)
        return [Prediction(100.0, 100.0)] * len(games)


@pytest.mark.parametrize("cadence", ["daily", "weekly"])
def test_no_leakage(tmp_path, cadence):
    con = duckdb.connect()
    results(
        con, season(2022, date(2021, 10, 19), 30) + season(2023, date(2022, 10, 18), 30)
    )
    spy = Spy()
    run(con, spy, [2023], cadence, tmp_path)
    assert spy.fits == (30 if cadence == "daily" else 5)


def test_holdout_needs_asking(tmp_path):
    con = duckdb.connect()
    results(con, season(2025, date(2024, 10, 22), 10))
    with pytest.raises(ValueError, match="held out"):
        run(con, Spy(), [2025], "weekly", tmp_path)
    run(con, Spy(), [2025], "weekly", tmp_path, allow_holdout=True)


def test_run_writes_predictions_and_metrics(tmp_path):
    con = duckdb.connect()
    results(
        con, season(2022, date(2021, 10, 19), 60) + season(2023, date(2022, 10, 18), 60)
    )
    rid = run(con, HomeCourt(), [2023], "weekly", tmp_path)
    predictions = con.sql(
        f"FROM read_parquet('{tmp_path}/runs/predictions/run={rid}/data.parquet')"
    )
    # 4 teams, one game a day: each team's 20th game is on day 40
    assert predictions.aggregate("phase, count(*)", "phase").order(
        "phase"
    ).fetchall() == [
        ("early", 40),
        ("mid", 20),
    ]
    # every home team wins by 10, so the home edge is 10 and the average 105
    assert predictions.aggregate(
        "round(min(pred_pts), 6), round(max(pred_pts), 6)"
    ).fetchall() == [(110, 110)]
    metrics = con.sql(
        f"SELECT phase, games, round(margin_rmse, 6), round(total_rmse, 6), settings "
        f"FROM read_parquet('{tmp_path}/runs/metrics/run={rid}/data.parquet')"
    ).fetchall()
    assert metrics[0] == ("all", 60, 0, 0, '{"half_life": 60, "home_half_life": 730}')


def test_score():
    con = duckdb.connect()
    predictions = con.sql("""
        SELECT * FROM (VALUES
            ('early', 110.0, 100.0, 120, 100, 100.0, 98.0),
            ('playoffs', 100.0, 110.0, 100, 100, NULL, 95.0))
        v(phase, pred_pts, pred_opp_pts, pts, opp_pts, pred_poss, poss)
    """)
    metrics = {m["phase"]: m for m in score(predictions)}
    assert set(metrics) == {"all", "early", "playoffs"}
    # margin errors -10 and -10; totals errors -10 and +10
    assert metrics["all"]["margin_rmse"] == 10
    assert metrics["all"]["margin_bias"] == -10
    assert metrics["all"]["total_bias"] == 0
    assert metrics["all"]["total_mae"] == 10
    assert metrics["early"]["games"] == 1
    assert metrics["early"]["poss_bias"] == 2
    # predicted margin of +10 with a spread of 10: 84% to win, and they won
    assert metrics["early"]["brier"] == pytest.approx((1 - 0.8413) ** 2, abs=1e-4)
    # no lines
    assert metrics["all"]["line_games"] == 0
    assert metrics["all"]["line_margin_rmse"] is None
    assert metrics["all"]["line_log_loss"] is None


def test_score_lines():
    con = duckdb.connect()
    predictions = con.sql("""
        SELECT * FROM (VALUES
            ('early', 110.0, 100.0, 120, 100, NULL, 98.0, 14.0, 215.0),
            ('early', 100.0, 110.0, 100, 106, NULL, 95.0, -12.0, 210.0),
            ('playoffs', 100.0, 110.0, 100, 100, NULL, 95.0, NULL, NULL))
        v(phase, pred_pts, pred_opp_pts, pts, opp_pts, pred_poss, poss,
            line_margin, line_total)
    """)
    metrics = {m["phase"]: m for m in score(predictions)}
    # line errors -6 and -6 (margin), -5 and +4 (total), on the games with one
    assert metrics["all"]["games"] == 3
    assert metrics["all"]["line_games"] == 2
    assert metrics["all"]["line_margin_rmse"] == 6
    assert metrics["all"]["line_total_rmse"] == pytest.approx((41 / 2) ** 0.5)
    # +14 and -12 with a spread of 6: both right, at 99.0% and 97.7%
    assert metrics["early"]["line_log_loss"] == pytest.approx(0.0164, abs=1e-4)
    assert metrics["playoffs"]["line_games"] == 0
    assert metrics["playoffs"]["line_margin_rmse"] is None


def test_age_without_offseason():
    con = duckdb.connect()
    # 2021 plays Jan 1-10, 2022 from Jun 1
    results(con, season(2021, date(2021, 1, 1), 10) + season(2022, date(2021, 6, 1), 5))

    def ages(history, offseason):
        return dict(
            history.results.select(
                f"game_date, {history.age(offseason)} AS age"
            ).fetchall()
        )

    # in the offseason: the break so far is skipped, as if 2021 ended yesterday
    h = History(con, date(2021, 3, 1), 2022)
    assert h.offseasons() == [(date(2021, 1, 10), 49)]
    assert ages(h, True)[date(2021, 1, 10)] == 50
    assert ages(h, False)[date(2021, 1, 10)] == 1
    assert ages(h, False)[date(2021, 1, 1)] == 10

    # in 2022: the whole break is skipped, in-season days still count
    h = History(con, date(2021, 6, 4), 2022)
    a = ages(h, False)
    assert a[date(2021, 6, 3)] == 1
    assert a[date(2021, 6, 1)] == 3
    assert a[date(2021, 1, 10)] == 4
    assert ages(h, True)[date(2021, 1, 10)] == 145

    # within the first season there's no break
    h = History(con, date(2021, 1, 5), 2021)
    assert h.offseasons() == []
    assert ages(h, False) == ages(h, True)


def test_as_of():
    con = duckdb.connect()
    # no schedule: the end of the cutoff day, US Eastern
    assert History(con, date(2026, 11, 16)).as_of == utc(2026, 11, 17, 5)
    con.execute("""CREATE TEMP TABLE schedule AS
        SELECT DATE '2026-11-16' AS game_date, t AS start_time
        FROM (VALUES (TIMESTAMP '2026-11-17 00:30'), (TIMESTAMP '2026-11-17 03:00')) v(t)""")
    # the first tip-off
    assert History(con, date(2026, 11, 16)).as_of == utc(2026, 11, 17, 0, 30)
    assert History(con, date(2026, 11, 18)).as_of == utc(2026, 11, 19, 5)
    # given: in UTC
    given = datetime(2026, 11, 16, 13, tzinfo=EASTERN)
    assert History(con, date(2026, 11, 16), as_of=given).as_of == utc(2026, 11, 16, 18)
