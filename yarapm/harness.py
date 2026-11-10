"""
The walk-forward harness: every model and experiment is judged by it.

    uv run python -m yarapm.harness --model home [--cadence weekly|daily]
        [--season 2021 ...] [--set name=value ...]

For each game day of the test seasons, a model is fit on the games before
that day and predicts that day's games: each team's expected score. With
`--cadence weekly`, it's fit on the first game day of each week of the
season and predicts the week's games, which is enough for tuning; final
numbers use `daily`.

What a model sees when it's fit is a `History`: the game results and
segments before the day it's fit on, and nothing else, so it can't use the
games it's predicting.

Tuning is done on TUNING_SEASONS. HOLDOUT_SEASONS are scored once, with
every setting fixed (#12): the harness won't run them without
`--holdout`.

Each run writes its predictions and metrics under out/runs/, tagged with the
model's settings, and replaces an earlier run with the same model, settings,
seasons and cadence:

    out/runs/predictions/run=<run_id>/data.parquet
    out/runs/metrics/run=<run_id>/data.parquet
    out/runs/minutes/run=<run_id>/data.parquet   if the model projects minutes

Metrics are computed for every game and for each phase of the season:

    early          regular season, either team in its first EARLY_GAMES games
    mid            the rest of the regular season (and the Cup final)
    post_deadline  regular season after the trade deadline
    playoffs       play-in and playoffs

Win probability is P(margin > 0) for a normal centered on the predicted
margin, with a spread of the run's margin RMSE; it's scored with log loss
and Brier score.

The closing betting line is scored beside every model, on the games that
have one (`line_games`), as a benchmark: `line_margin_rmse`,
`line_total_rmse` and `line_log_loss`, its win probability taken the same
way from its own RMSE. It isn't in `History`, so no model sees it. The
close knows the injury news that only a model with known availability
has.

A model that projects each player's minutes (`Prediction.minutes`) is also
scored on them: `minutes_mae` is the mean absolute error per player, over
everyone projected or who played (missing ones count as 0), against the
minutes they played scaled to 240 a team, since overtime can't be
projected.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import duckdb
import numpy as np
import pyarrow as pa
from scipy.special import ndtr

from yarapm.catalog import OUT, connect, read
from yarapm.games import load_results

TUNING_SEASONS = (2021, 2022, 2023, 2024)
HOLDOUT_SEASONS = (2025, 2026)

EARLY_GAMES = 20

TRADE_DEADLINES = {
    2010: date(2010, 2, 18),
    2011: date(2011, 2, 24),
    2012: date(2012, 3, 15),
    2013: date(2013, 2, 21),
    2014: date(2014, 2, 20),
    2015: date(2015, 2, 19),
    2016: date(2016, 2, 18),
    2017: date(2017, 2, 23),
    2018: date(2018, 2, 8),
    2019: date(2019, 2, 7),
    2020: date(2020, 2, 6),
    2021: date(2021, 3, 25),
    2022: date(2022, 2, 10),
    2023: date(2023, 2, 9),
    2024: date(2024, 2, 8),
    2025: date(2025, 2, 6),
    2026: date(2026, 2, 5),
}


@dataclass(frozen=True)
class Game:
    """A game to predict, from `team_id`'s side (the home team's, if any)"""

    game_id: str
    season: int
    game_date: date
    game_type: str
    team_id: str
    opp_team_id: str
    home: bool | None  # None at a neutral site


class Prediction(NamedTuple):
    """
    A game's expected score, its possessions and each player's minutes (both
    teams'), if the model has them
    """

    pts: float
    opp_pts: float
    poss: float | None = None
    minutes: dict[str, float] | None = None


class History:
    """
    What a model may use when it's fit: everything before `cutoff`, in
    `season`. Every table a model reads goes through here, filtered on
    `game_date`
    """

    def __init__(
        self, con: duckdb.DuckDBPyConnection, cutoff: date, season: int | None = None
    ):
        self.con = con
        self.cutoff = cutoff
        # None for every season in the data
        self.season = season

    def _before(self, table: str) -> duckdb.DuckDBPyRelation:
        return self.con.sql(
            f"SELECT * FROM {table} WHERE game_date < DATE '{self.cutoff}'"
        )

    @property
    def results(self) -> duckdb.DuckDBPyRelation:
        """`yarapm.games.load_results`' rows: one per game, with both scores"""
        return self._before("results")

    @property
    def segments(self) -> duckdb.DuckDBPyRelation:
        """`yarapm.segments`' rows"""
        return self._before("all_segments")

    @property
    def player_games(self) -> duckdb.DuckDBPyRelation:
        """nba.player_game_logs' rows for the games in `results`"""
        return self._before("player_games")

    @property
    def bio(self) -> duckdb.DuckDBPyRelation:
        """
        Each player's height and draft position, which don't change: one
        row per player, from their latest season in nba.player_season_stats.
        That can be after the cutoff, but both are known before a player's
        first game
        """
        return self.con.sql("FROM bio")

    def openrapm(self, season: int) -> duckdb.DuckDBPyRelation:
        """
        nba.openrapm's ratings for a season. OpenRAPM is fit on whole
        seasons, so this season's uses games after the cutoff: a leak,
        only for an optimistic baseline (#9)
        """
        return self.con.sql(f"FROM openrapm WHERE season = {int(season)}")

    def active(self, game_ids: list[str]) -> list[tuple[str, str, str]]:
        """
        (game_id, team_id, player_id) for every player who played in
        `game_ids`: a leak, for games on or after the cutoff, standing in
        for knowing who's available (#10). It flatters a projection
        slightly, since a player who's available but doesn't play is left
        out
        """
        return self.con.execute(
            """SELECT game_id, team_id, player_id FROM player_games
            WHERE game_id IN (SELECT unnest(?::VARCHAR[])) AND min > 0""",
            [game_ids],
        ).fetchall()

    def actual_minutes(self, game_ids: list[str]) -> list[tuple[str, str, str, float]]:
        """
        (game_id, team_id, player_id, min) for every player who played in
        `game_ids`. A deliberate leak, for games on or after the cutoff: it
        tells a model who played and for how long, to measure rating error
        on its own (#8) until minutes are projected (#10)
        """
        return self.con.execute(
            """SELECT game_id, team_id, player_id, min FROM player_games
            WHERE game_id IN (SELECT unnest(?::VARCHAR[])) AND min > 0""",
            [game_ids],
        ).fetchall()


class Model(Protocol):
    name: str

    @property
    def settings(self) -> dict[str, Any]: ...

    def fit(self, history: History) -> None: ...

    def predict(self, games: list[Game]) -> list[Prediction]:
        """Each game's prediction, in order"""
        ...


def fit_days(days: list[date], cadence: str) -> list[list[date]]:
    """
    Game days grouped by the fit that predicts them: one a day, or the game
    days in each week starting from the season's first
    """
    if cadence == "daily":
        return [[d] for d in days]
    if cadence != "weekly":
        raise ValueError(f"unknown cadence {cadence!r}")
    groups: list[list[date]] = []
    for d in days:
        if groups and d < groups[-1][0] + timedelta(days=7):
            groups[-1].append(d)
        else:
            groups.append([d])
    return groups


def run_id(model: Model, seasons: list[int], cadence: str) -> str:
    key = json.dumps([model.name, model.settings, sorted(seasons), cadence])
    return f"{model.name}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def run(
    con: duckdb.DuckDBPyConnection,
    model: Model,
    seasons: list[int],
    cadence: str = "weekly",
    out: Path = OUT,
    allow_holdout: bool = False,
) -> str:
    """
    Walk `model` forward through `seasons`, using the temp table `results`
    and the view `all_segments`, and write its predictions and metrics.
    Returns the run's id
    """
    held = sorted(set(seasons) & set(HOLDOUT_SEASONS))
    if held and not allow_holdout:
        raise ValueError(f"{held} are held out until the final evaluation (#12)")

    rows = []
    minutes = []
    for season in sorted(seasons):
        games = [
            Game(*r)
            for r in con.execute(
                """SELECT game_id, season, game_date, game_type, team_id,
                    opp_team_id, home FROM results WHERE season = ?
                ORDER BY game_date, game_id""",
                [season],
            ).fetchall()
        ]
        days = sorted({g.game_date for g in games})
        for group in fit_days(days, cadence):
            model.fit(History(con, group[0], season))
            todo = [g for g in games if g.game_date in group]
            predicted = [Prediction(*p) for p in model.predict(todo)]
            if len(predicted) != len(todo):
                raise ValueError(
                    f"{model.name} predicted {len(predicted)} of {len(todo)} games"
                )
            rows += [(g.game_id, *p[:3]) for g, p in zip(todo, predicted)]
            minutes += [
                (g.game_id, player, m)
                for g, p in zip(todo, predicted)
                if p.minutes is not None
                for player, m in p.minutes.items()
            ]

    rid = run_id(model, seasons, cadence)
    con.register(
        "predicted",
        pa.table(
            {
                "game_id": [r[0] for r in rows],
                "pred_pts": pa.array([r[1] for r in rows], pa.float64()),
                "pred_opp_pts": pa.array([r[2] for r in rows], pa.float64()),
                "pred_poss": pa.array([r[3] for r in rows], pa.float64()),
            }
        ),
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE predictions AS
        SELECT '{rid}' AS run_id, r.season, r.game_id, r.game_date, r.game_type,
            ph.phase, r.team_id, r.opp_team_id, r.home,
            p.pred_pts, p.pred_opp_pts, r.pts, r.opp_pts,
            p.pred_poss, (r.poss + r.opp_poss) / 2 AS poss,
            l.line_margin, l.line_total
        FROM predicted p
        JOIN results r USING (game_id)
        JOIN ({phases()}) ph USING (game_id)
        LEFT JOIN ({lines_query(con)}) l USING (game_id)
        ORDER BY r.game_date, r.game_id
    """)
    con.unregister("predicted")

    metrics = score(con.sql("FROM predictions"))
    outputs = [("predictions", "FROM predictions")]
    if minutes:
        minutes_errors(con, minutes)
        outputs.append(("minutes", "FROM minutes"))
        mae = dict(
            con.sql("""
                SELECT coalesce(phase, 'all'), avg(abs(pred_min - min))
                FROM minutes GROUP BY ROLLUP (phase)
            """).fetchall()
        )
        for m in metrics:
            m["minutes_mae"] = mae.get(m["phase"])
    created = datetime.now(UTC)
    for m in metrics:
        m |= {
            "run_id": rid,
            "model": model.name,
            "settings": json.dumps(model.settings, sort_keys=True),
            "seasons": sorted(seasons),
            "cadence": cadence,
            "created": created,
        }
    con.register("scored", pa.Table.from_pylist(metrics))
    for name, query in [
        *outputs,
        (
            "metrics",
            """SELECT run_id, model, settings, seasons, cadence, created,
                * EXCLUDE (run_id, model, settings, seasons, cadence, created)
            FROM scored""",
        ),
    ]:
        path = out / "runs" / name / f"run={rid}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    con.unregister("scored")
    return rid


def phases() -> str:
    """SQL for each game in `results`' phase of the season: (game_id, phase)"""
    deadlines = ", ".join(f"({s}, DATE '{d}')" for s, d in TRADE_DEADLINES.items())
    return f"""
        WITH numbered AS (
            -- each team's game number in the regular season
            SELECT game_id, team_id, row_number() OVER (
                PARTITION BY season, team_id ORDER BY game_date, game_id) AS n
            FROM (
                SELECT season, game_id, game_date, team_id FROM results
                WHERE game_type = 'regular_season'
                UNION ALL
                SELECT season, game_id, game_date, opp_team_id FROM results
                WHERE game_type = 'regular_season'
            )
        ), deadlines AS (
            SELECT * FROM (VALUES {deadlines}) v(season, deadline)
        )
        SELECT r.game_id,
            CASE
                WHEN r.game_type IN ('playoffs', 'play_in') THEN 'playoffs'
                WHEN r.game_type = 'regular_season'
                    AND least(t.n, o.n) <= {EARLY_GAMES} THEN 'early'
                WHEN r.game_type = 'regular_season'
                    AND r.game_date > d.deadline THEN 'post_deadline'
                ELSE 'mid'
            END AS phase
        FROM results r
        LEFT JOIN numbered t ON t.game_id = r.game_id AND t.team_id = r.team_id
        LEFT JOIN numbered o ON o.game_id = r.game_id AND o.team_id = r.opp_team_id
        LEFT JOIN deadlines d ON d.season = r.season
    """


def minutes_errors(
    con: duckdb.DuckDBPyConnection, minutes: list[tuple[str, str, float]]
) -> None:
    """
    Temp table `minutes`: each player's projected minutes in a game
    (`minutes`: game_id, player_id, minutes) and the minutes they played,
    scaled to 240 a team, for everyone in either, with each game's phase
    from `predictions`
    """
    con.register(
        "projected",
        pa.table(
            {
                "game_id": [m[0] for m in minutes],
                "player_id": [m[1] for m in minutes],
                "pred_min": pa.array([m[2] for m in minutes], pa.float64()),
            }
        ),
    )
    con.execute("""
        CREATE OR REPLACE TEMP TABLE minutes AS
        WITH played AS (
            SELECT game_id, player_id,
                240 * min / sum(min) OVER (PARTITION BY game_id, team_id) AS min
            FROM player_games
            WHERE min > 0 AND game_id IN (SELECT game_id FROM projected)
        )
        SELECT p.run_id, p.season, game_id, p.game_date, p.phase, player_id,
            coalesce(j.pred_min, 0) AS pred_min, coalesce(a.min, 0) AS min
        FROM projected j
        FULL JOIN played a USING (game_id, player_id)
        JOIN predictions p USING (game_id)
        ORDER BY p.game_date, game_id, player_id
    """)
    con.unregister("projected")


def lines_query(con: duckdb.DuckDBPyConnection) -> str:
    """
    SQL for each game's closing line from `results`' team's side, from the
    temp table `lines` if it's loaded: (game_id, line_margin, line_total)
    """
    loaded = con.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE temporary AND table_name = 'lines'"
    ).fetchall()[0][0]
    if not loaded:
        return """SELECT NULL::VARCHAR AS game_id, NULL::DOUBLE AS line_margin,
            NULL::DOUBLE AS line_total WHERE false"""
    return "FROM lines"


def load_lines(con: duckdb.DuckDBPyConnection) -> None:
    """
    Temp table `lines`: each game in `results`' closing spread as the
    margin it predicts for `team_id`, and its total. nba.betting_lines has
    a row per provider: prefer ESPN's books (DraftKings, then ESPN BET),
    then The Odds API's consensus, then sportsbookreview's archive, as its
    data dictionary advises. A line whose teams don't match the game's is
    dropped
    """
    seasons = ", ".join(
        str(s) for (s,) in con.execute("SELECT DISTINCT season FROM results").fetchall()
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lines AS
        WITH ranked AS (
            SELECT b.game_id,
                CASE r.team_id
                    WHEN b.home_team_id THEN -b.close_spread
                    WHEN b.away_team_id THEN b.close_spread
                END AS line_margin,
                b.close_total AS line_total,
                CASE b.provider_id WHEN '100' THEN 1 WHEN '58' THEN 2
                    WHEN 'odds_api_consensus' THEN 3 ELSE 4 END AS rank
            FROM nba.betting_lines b
            JOIN results r USING (season, game_id)
            WHERE b.season IN ({seasons})
                AND b.provider_id IN ('100', '58', 'odds_api_consensus', 'sbr')
                AND list_sort([r.team_id, r.opp_team_id])
                    = list_sort([b.home_team_id, b.away_team_id])
        )
        SELECT game_id,
            arg_min(line_margin, rank) FILTER (WHERE line_margin IS NOT NULL)
                AS line_margin,
            arg_min(line_total, rank) FILTER (WHERE line_total IS NOT NULL)
                AS line_total
        FROM ranked GROUP BY game_id
    """)


def score(predictions: duckdb.DuckDBPyRelation) -> list[dict[str, Any]]:
    """
    Metrics for every game ("all") and each phase, and the closing line's if
    `predictions` has `line_margin` and `line_total`
    """
    has_lines = {"line_margin", "line_total"} <= set(predictions.columns)
    cols = predictions.select(
        "phase, pred_pts, pred_opp_pts, pts, opp_pts, "
        "coalesce(pred_poss, 'nan'::DOUBLE) AS pred_poss, poss, "
        + (
            "coalesce(line_margin, 'nan'::DOUBLE) AS line_margin, "
            "coalesce(line_total, 'nan'::DOUBLE) AS line_total"
            if has_lines
            else "'nan'::DOUBLE AS line_margin, 'nan'::DOUBLE AS line_total"
        )
    ).fetchnumpy()
    pred_poss = np.asarray(cols["pred_poss"], dtype=float)
    poss = np.asarray(cols["poss"], dtype=float)
    phase = np.asarray(cols["phase"])
    pred = np.column_stack([cols["pred_pts"], cols["pred_opp_pts"]]).astype(float)
    actual = np.column_stack([cols["pts"], cols["opp_pts"]]).astype(float)
    pred_margin = pred[:, 0] - pred[:, 1]
    margin = actual[:, 0] - actual[:, 1]
    sigma = float(np.sqrt(np.mean((pred_margin - margin) ** 2)))
    win_prob = np.clip(ndtr(pred_margin / max(sigma, 1e-9)), 1e-6, 1 - 1e-6)
    won = margin > 0
    line_margin = np.asarray(cols["line_margin"], dtype=float)
    line_total = np.asarray(cols["line_total"], dtype=float)
    lined = ~np.isnan(line_margin)
    line_sigma = (
        float(np.sqrt(np.mean((line_margin[lined] - margin[lined]) ** 2)))
        if lined.any()
        else np.nan
    )
    line_prob = np.clip(ndtr(line_margin / max(line_sigma, 1e-9)), 1e-6, 1 - 1e-6)

    metrics = []
    for name in ["all", "early", "mid", "post_deadline", "playoffs"]:
        i = np.full(len(phase), True) if name == "all" else phase == name
        if not i.any():
            continue
        margin_err = pred_margin[i] - margin[i]
        total_err = pred[i].sum(axis=1) - actual[i].sum(axis=1)
        p, y = win_prob[i], won[i]
        # NaN for a model that doesn't predict possessions
        poss_err = pred_poss[i] - poss[i]
        j = i & lined
        line_err = line_margin[j] - margin[j]
        line_total_err = line_total[i] - actual[i].sum(axis=1)
        line_total_err = line_total_err[~np.isnan(line_total_err)]
        lp, ly = line_prob[j], won[j]
        metrics.append(
            {
                "phase": name,
                "games": int(i.sum()),
                "margin_bias": float(margin_err.mean()),
                "margin_rmse": float(np.sqrt((margin_err**2).mean())),
                "margin_mae": float(np.abs(margin_err).mean()),
                "total_bias": float(total_err.mean()),
                "total_rmse": float(np.sqrt((total_err**2).mean())),
                "total_mae": float(np.abs(total_err).mean()),
                "score_rmse": float(np.sqrt(((pred[i] - actual[i]) ** 2).mean())),
                "poss_bias": float(poss_err.mean()),
                "poss_rmse": float(np.sqrt((poss_err**2).mean())),
                "sigma": sigma,
                "log_loss": float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean()),
                "brier": float(((p - y) ** 2).mean()),
                "line_games": int(j.sum()),
                "line_margin_rmse": rmse(line_err),
                "line_total_rmse": rmse(line_total_err),
                "line_log_loss": (
                    float(-(ly * np.log(lp) + (~ly) * np.log(1 - lp)).mean())
                    if j.any()
                    else None
                ),
            }
        )
    return metrics


def rmse(errors: np.ndarray) -> float | None:
    return float(np.sqrt((errors**2).mean())) if len(errors) else None


def load(con: duckdb.DuckDBPyConnection, out: Path = OUT) -> None:
    """
    The tables `History` reads: `results`, `player_games`, `bio`,
    `openrapm`, and `all_segments` if built; and `lines`, for scoring
    """
    load_results(con)
    load_lines(con)
    # `started` is NULL for a game without rotations (before 2019)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE player_games AS
        WITH starters AS (
            SELECT DISTINCT game_id, player_id, true AS started
            FROM nba.game_rotations WHERE in_time = 0
        ), rotations AS (
            SELECT DISTINCT game_id FROM nba.game_rotations
        )
        SELECT l.*, CASE WHEN r.game_id IS NOT NULL
                THEN coalesce(s.started, false) END AS started
        FROM nba.player_game_logs l
        LEFT JOIN starters s USING (game_id, player_id)
        LEFT JOIN rotations r USING (game_id)
        WHERE l.game_id IN (SELECT game_id FROM results)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE bio AS
        SELECT player_id,
            arg_max(player_height_inches, season) AS height,
            arg_max(draft_number, season) AS draft_number
        FROM nba.player_season_stats
        WHERE season_type = 'regular_season'
        GROUP BY player_id
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE openrapm AS
        SELECT season, player_id, possessions, o_rating, d_rating FROM nba.openrapm
    """)
    if (out / "segments").is_dir():
        read(con, "segments", out).create_view("all_segments", replace=True)


def models() -> dict[str, type]:
    from yarapm.baselines import HomeCourt, TeamRatings
    from yarapm.players import OpenRapm, PlayerRatings

    return {
        "home": HomeCourt,
        "team": TeamRatings,
        "rapm": PlayerRatings,
        "openrapm": OpenRapm,
    }


def setting(text: str) -> tuple[str, Any]:
    name, _, value = text.partition("=")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def main(argv: list[str] | None = None) -> None:
    available = models()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--model", required=True, choices=sorted(available))
    parser.add_argument("--cadence", choices=["weekly", "daily"], default="weekly")
    parser.add_argument("--season", type=int, action="append")
    parser.add_argument("--set", type=setting, action="append", default=[])
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    con = connect()
    load(con, args.out)
    model = available[args.model](**dict(args.set))
    rid = run(
        con,
        model,
        args.season or list(TUNING_SEASONS),
        args.cadence,
        args.out,
        args.holdout,
    )
    print(rid, json.dumps(model.settings))
    path = f"{args.out}/runs/metrics/run={rid}/data.parquet"
    columns = [c for (c, *_) in con.sql(f"DESCRIBE FROM '{path}'").fetchall()]
    minutes_mae = (
        ", round(minutes_mae, 2) AS minutes_mae" if "minutes_mae" in columns else ""
    )
    print(
        con.sql(f"""
            SELECT phase, games, round(margin_rmse, 2) AS margin_rmse,
                round(margin_mae, 2) AS margin_mae, round(margin_bias, 2) AS margin_bias,
                round(total_rmse, 2) AS total_rmse, round(total_bias, 2) AS total_bias,
                round(score_rmse, 2) AS score_rmse, round(poss_rmse, 2) AS poss_rmse,
                round(log_loss, 4) AS log_loss,
                round(brier, 4) AS brier{minutes_mae}, line_games,
                round(line_margin_rmse, 2) AS line_margin_rmse,
                round(line_total_rmse, 2) AS line_total_rmse,
                round(line_log_loss, 4) AS line_log_loss
            FROM read_parquet('{path}')
        """)
    )


if __name__ == "__main__":
    main()
