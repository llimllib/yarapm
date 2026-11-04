"""
The datasets we publish (#27), under the `yarapm/` prefix of the
basketball-data bucket, with their own catalog:

    uv run python -m yarapm.publish build [--date 2026-11-05] [--no-segments]
    uv run python -m yarapm.publish catalog [--url https://basketball-data.billmill.org]
    uv run python -m yarapm.publish current
    uv run python -m yarapm.publish verify [--url https://basketball-data.billmill.org]

`build` predicts the rest of the season with the live `rapm` model
(yarapm.predict) and writes the datasets into out/publish/, a mirror of the
bucket's yarapm/ prefix, laid out like nba_data's, one file per season:

    yarapm/<dataset>/season=<season>/data.parquet
    yarapm/metadata.json

    ratings_daily  each player's ratings as of every game day of the
                   season and the build's date. Days already in the
                   mirror are kept and missing ones are fit, so the job
                   downloads the bucket's yarapm/ first
    predictions    the latest prediction for each of the season's games
                   from the date on, replaced every build
    minutes        each player's projected minutes and chance of playing
                   in those games
    prediction_log every prediction made before its game (#29), from the
                   live `rapm` and `team` models, for the games in the
                   next LOG_DAYS days that haven't started. A build adds a
                   row only when a game's prediction has changed since its
                   last one, and never changes a row
    minutes_log    `rapm`'s projected minutes for each prediction it logs
    live_metrics   the log scored: for each finished game, the last
                   prediction made before tip-off, with the harness's
                   metrics, by series (model, commit and settings) and
                   phase
    metadata       per dataset: seasons, when it was updated, and the
                   commit and settings that built it; also the date the
                   build predicted from and nba_data's latest update

Both logs and ratings_daily keep what earlier builds wrote, so a build
refuses to run if metadata.json lists a season of them that the mirror
doesn't have.

docs/DATA.md has the columns. Values are rounded (ratings, points and
minutes to two decimals, probabilities to four): full-precision doubles
compress badly. Before any file is written, the datasets are
checked (`validate`); a failed check raises, and the mirror keeps the last
good files.

`catalog` writes yarapm/yarapm.duckdb: a view over every season's files at
`--url` for each dataset, and the metadata table. DuckDB reads a view's
files when it's created, so they must be uploaded first. Without `--url`,
the views read the mirror's files, and the catalog goes to
out/yarapm.local.duckdb, outside the mirror, so it's never uploaded.

`current` exits 0 if the mirror's last build read nba_data's latest
update (its metadata.json's newest `updated`), so a scheduled run can skip
a build that would find nothing new, and 1 if not. `verify` attaches the
catalog at `--url` and reads every dataset in it.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from yarapm import segments
from yarapm.baselines import TeamRatings
from yarapm.catalog import OUT, connect
from yarapm.harness import History, load, phases, score
from yarapm.minutes import ProjectedMinutes
from yarapm.players import PlayerRatings
from yarapm.predict import EASTERN, Forecast, forecast, scheduled

PUBLISH = OUT / "publish"
PREFIX = "yarapm"
URL = "https://basketball-data.billmill.org"
DATASETS = ("ratings_daily", "predictions", "minutes")
LOGS = ("prediction_log", "minutes_log")
# a build logs the predictions for the games in this many days
LOG_DAYS = 7
# DuckDB 1.1.1, in Observable Framework's DuckDB-wasm, reads up to v1.0.0
STORAGE_VERSION = "v1.0.0"


def dataset_path(root: Path, dataset: str, season: int) -> Path:
    return root / PREFIX / dataset / f"season={season}" / "data.parquet"


def metadata_path(root: Path) -> Path:
    return root / PREFIX / "metadata.json"


def live_model() -> PlayerRatings:
    return PlayerRatings(minutes="unknown")


def commit() -> str:
    """The commit being run, with `-dirty` if tracked files have changed"""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        clean = (
            subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode
            == 0
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return sha if clean else f"{sha}-dirty"


def ratings(model: PlayerRatings, history: History) -> list[tuple]:
    """
    (player_id, team_id, o_rating, d_rating, o_prior, d_prior, possessions)
    as of the cutoff the model was fit on, for each player on a roster or
    with possessions this season. `team_id` is the roster's, if any
    """
    assert isinstance(model.projection, ProjectedMinutes)
    poss = dict(
        history.segments.filter(f"season = {history.season}")
        .select("unnest(off_player_ids) AS player_id, poss")
        .aggregate("player_id, sum(poss)::INTEGER", "player_id")
        .fetchall()
    )
    teams = {
        p: team for team, roster in model.projection.rosters.items() for p in roster
    }
    rows = []
    for p in sorted(set(poss) | set(teams)):
        o, d = model.rating(p)
        prior_o, prior_d = model.prior.prior(p)
        rows.append(
            (
                p,
                teams.get(p),
                o,
                d,
                model.prior_weight * prior_o,
                model.prior_weight * prior_d,
                poss.get(p, 0),
            )
        )
    return rows


def rating_days(
    con: duckdb.DuckDBPyConnection, season: int, day: date, have: set[date]
) -> list[date]:
    """
    The days to rate: every game day of the season before `day` that isn't
    in `have`, and `day`
    """
    days = {
        d
        for (d,) in con.execute(
            "SELECT DISTINCT game_date FROM results WHERE season = ? AND game_date < ?",
            [season, day],
        ).fetchall()
    }
    return sorted((days - have) | {day})


def register(con: duckdb.DuckDBPyConnection, name: str, columns: dict) -> None:
    """A temp table from a dict of pyarrow arrays or lists"""
    con.register("_rows", pa.table(columns))
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS FROM _rows")
    con.unregister("_rows")


def build(
    con: duckdb.DuckDBPyConnection,
    day: date,
    root: Path = PUBLISH,
    sha: str = "unknown",
    now: datetime | None = None,
) -> list[str]:
    """
    Fit and write the datasets into `root` for the season of the first game
    scheduled from `day`, as of `now` (when a prediction is logged). Needs
    the tables `yarapm.harness.load` makes. Returns the datasets written
    """
    now = now or datetime.now(UTC)
    games = scheduled(con, day, 366)
    if not games:
        print(f"no games scheduled from {day}")
        return []
    season = games[0][0].season
    check_mirror(root, season)
    games = [(g, start) for g, start in games if g.season == season]
    model = live_model()
    settings = json.dumps(model.settings, sort_keys=True)

    forecasts = forecast(con, model, games, day)
    assert isinstance(model.projection, ProjectedMinutes)
    minutes = model.projection.predict([g for g, _ in games])
    chances = {
        team: model.projection.playing_chances(team)
        for g, _ in games
        for team in (g.team_id, g.opp_team_id)
    }
    rated = {day: ratings(model, History(con, day, season))}

    existing = dataset_path(root, "ratings_daily", season)
    have: set[date] = set()
    if existing.exists():
        have = {
            d
            for (d,) in con.execute(
                f"SELECT DISTINCT as_of_date FROM '{existing}'"
            ).fetchall()
        }
    earlier = live_model()
    for d in rating_days(con, season, day, have):
        if d != day:
            earlier.fit(History(con, d, season))
            rated[d] = ratings(earlier, History(con, d, season))

    team = TeamRatings()
    window = [(g, start) for g, start in games if (g.game_date - day).days < LOG_DAYS]
    team_forecasts = forecast(con, team, window, day)

    predictions_table(con, forecasts, season, day)
    predictions_table(con, team_forecasts, season, day, "team_predictions")
    minutes_table(con, minutes, chances, day)
    ratings_table(con, rated, sha, settings, existing if have else None)
    log_tables(
        con,
        now,
        day,
        sha,
        {
            model.name: ("predictions", settings),
            team.name: ("team_predictions", json.dumps(team.settings, sort_keys=True)),
        },
        {log: dataset_path(root, log, season) for log in LOGS},
    )
    scored = live_metrics(con, season)
    validate(con, len(games))

    written = [*DATASETS, *LOGS] + (["live_metrics"] if scored else [])
    for dataset in written:
        write(con, f"FROM {dataset}", dataset_path(root, dataset, season))
    about = {"commit": sha, "model": model.name, "settings": model.settings}
    both = {"commit": sha, "model": f"{model.name}, {team.name}", "settings": None}
    write_metadata(
        root,
        {
            d: both if d in ("prediction_log", "live_metrics") else about
            for d in written
        },
        {
            "as_of_date": day.isoformat(),
            "source_updated": source_updated(con),
        },
    )
    return written


def check_mirror(root: Path, season: int) -> None:
    """
    Raise if the mirror lacks a season of a dataset that keeps what earlier
    builds wrote, which metadata.json says the bucket has
    """
    path = metadata_path(root)
    if not path.exists():
        return
    datasets = json.loads(path.read_text()).get("datasets", {})
    for dataset in ("ratings_daily", *LOGS):
        if dataset not in datasets:
            continue
        expected = datasets[dataset]["seasons"]
        found = seasons(root, dataset)
        if len(found) < expected or (
            datasets[dataset]["last_season"] == season and season not in found
        ):
            raise ValueError(
                f"{dataset} has {len(found)} of its {expected} seasons in {root}: "
                "download the bucket's yarapm/ first"
            )


def predictions_table(
    con: duckdb.DuckDBPyConnection,
    forecasts: list[Forecast],
    season: int,
    day: date,
    name: str = "predictions",
) -> None:
    """Temp table `name`, from the home team's side, as nba.schedule lists it"""
    register(
        con,
        "forecasts",
        {
            "game_id": [f.game.game_id for f in forecasts],
            "team_id": [f.game.team_id for f in forecasts],
            "pts": pa.array([f.prediction.pts for f in forecasts], pa.float64()),
            "opp_pts": pa.array(
                [f.prediction.opp_pts for f in forecasts], pa.float64()
            ),
            "poss": pa.array([f.prediction.poss for f in forecasts], pa.float64()),
            "win_prob": pa.array([f.win_prob for f in forecasts], pa.float64()),
        },
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        WITH f AS (
            SELECT f.*, s.*, f.team_id = s.home_team_id AS home
            FROM forecasts f
            JOIN nba.schedule s ON s.season = ? AND s.game_id = f.game_id
        )
        SELECT game_id, game_date, start_time, game_type,
            home_team_id, away_team_id, neutral_site,
            round(CASE WHEN home THEN pts ELSE opp_pts END, 2) AS home_pts,
            round(CASE WHEN home THEN opp_pts ELSE pts END, 2) AS away_pts,
            round(poss, 2) AS poss,
            round(CASE WHEN home THEN win_prob ELSE 1 - win_prob END, 4)
                AS home_win_prob,
            ?::DATE AS as_of_date
        FROM f
        ORDER BY game_date, start_time, game_id
        """,
        [season, day],
    )


def minutes_table(
    con: duckdb.DuckDBPyConnection,
    minutes: dict[tuple[str, str], dict[str, float]],
    chances: dict[str, dict[str, float]],
    day: date,
) -> None:
    """Temp table `minutes`: each player's projected minutes and chance of playing"""
    rows = [
        (game_id, team, player, m, chances[team].get(player, 0.0))
        for (game_id, team), players in minutes.items()
        for player, m in players.items()
    ]
    register(
        con,
        "projected",
        {
            "game_id": [r[0] for r in rows],
            "team_id": [r[1] for r in rows],
            "player_id": [r[2] for r in rows],
            "min": pa.array([r[3] for r in rows], pa.float64()),
            "play_prob": pa.array([r[4] for r in rows], pa.float64()),
        },
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE minutes AS
        SELECT m.game_id, m.team_id, m.player_id, round(m.min, 2) AS min,
            round(m.play_prob, 4) AS play_prob,
            ?::DATE AS as_of_date
        FROM projected m JOIN predictions p USING (game_id)
        ORDER BY p.game_date, p.start_time, m.game_id, m.team_id, m.min DESC,
            m.player_id
        """,
        [day],
    )


def ratings_table(
    con: duckdb.DuckDBPyConnection,
    rated: dict[date, list[tuple]],
    sha: str,
    settings: str,
    existing: Path | None,
) -> None:
    """
    Temp table `ratings_daily`: the days in `rated`, and the other days in
    `existing`, the season's file so far
    """
    rows = [(d, *r) for d, players in rated.items() for r in players]
    register(
        con,
        "rated",
        {
            "as_of_date": pa.array([r[0] for r in rows], pa.date32()),
            "player_id": [r[1] for r in rows],
            "team_id": pa.array([r[2] for r in rows], pa.string()),
            "o_rating": pa.array([r[3] for r in rows], pa.float64()),
            "d_rating": pa.array([r[4] for r in rows], pa.float64()),
            "o_prior": pa.array([r[5] for r in rows], pa.float64()),
            "d_prior": pa.array([r[6] for r in rows], pa.float64()),
            "possessions": pa.array([r[7] for r in rows], pa.int32()),
        },
    )
    kept = (
        f"""UNION ALL BY NAME
        SELECT * FROM read_parquet('{existing}')
        WHERE as_of_date NOT IN (SELECT as_of_date FROM rated)"""
        if existing
        else ""
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ratings_daily AS
        SELECT as_of_date, player_id, team_id, round(o_rating, 2) AS o_rating,
            round(d_rating, 2) AS d_rating,
            round(o_rating + d_rating, 2) AS rating, round(o_prior, 2) AS o_prior,
            round(d_prior, 2) AS d_prior, possessions,
            ? AS commit, ? AS settings
        FROM rated
        {kept}
        ORDER BY as_of_date, player_id
        """,
        [sha, settings],
    )


def log_tables(
    con: duckdb.DuckDBPyConnection,
    now: datetime,
    day: date,
    sha: str,
    models: dict[str, tuple[str, str]],
    existing: dict[str, Path],
) -> None:
    """
    Temp tables `prediction_log` and `minutes_log`: the `existing` files
    (by log), and the predictions in each model's table (`models`: name ->
    (table, settings)) for the games in the next LOG_DAYS days that haven't
    started by `now`, where they've changed since the game's last row.
    `minutes_log` gets the `minutes` of the `rapm` predictions logged
    """
    created = now.astimezone(UTC).isoformat()
    made = " UNION ALL ".join(
        f"""SELECT '{name}' AS model, '{sha}' AS commit, '{settings}' AS settings,
            '{created}'::TIMESTAMPTZ AS created, *
        FROM {table}
        WHERE game_date < DATE '{day}' + {LOG_DAYS}
            AND start_time > '{created}'::TIMESTAMPTZ"""
        for name, (table, settings) in models.items()
    )
    old = {
        log: f"read_parquet('{path}')" if path.exists() else f"(FROM {log}_empty)"
        for log, path in existing.items()
    }
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prediction_log_empty AS
        SELECT * FROM ({made}) LIMIT 0
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE minutes_log_empty AS
        SELECT NULL::VARCHAR AS model, NULL::TIMESTAMPTZ AS created,
            * EXCLUDE (as_of_date)
        FROM minutes LIMIT 0
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE logged AS
        WITH last AS (
            FROM {old["prediction_log"]}
            QUALIFY row_number() OVER (
                PARTITION BY model, game_id ORDER BY created DESC) = 1
        )
        SELECT n.* FROM ({made}) n
        LEFT JOIN last l USING (model, game_id)
        WHERE l.game_id IS NULL
            OR (n.home_pts, n.away_pts, n.poss, n.home_win_prob, n.commit,
                n.settings)
            IS DISTINCT FROM
                (l.home_pts, l.away_pts, l.poss, l.home_win_prob, l.commit,
                l.settings)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prediction_log AS
        SELECT * FROM {old["prediction_log"]}
        UNION ALL BY NAME FROM logged
        ORDER BY created, model, game_date, start_time, game_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE minutes_log AS
        SELECT * FROM {old["minutes_log"]}
        UNION ALL BY NAME
        SELECT l.model, l.created, m.* EXCLUDE (as_of_date)
        FROM minutes m JOIN logged l ON l.model = 'rapm' AND l.game_id = m.game_id
        ORDER BY created, model, game_id, team_id, min DESC, player_id
    """)


def live_metrics(con: duckdb.DuckDBPyConnection, season: int) -> int:
    """
    Temp table `live_metrics`: the season's finished games in
    `prediction_log`, each scored on its last prediction made before
    tip-off, by series and phase. Returns the number of games scored
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        WITH last AS (
            FROM prediction_log WHERE created < start_time
            QUALIFY row_number() OVER (
                PARTITION BY model, game_id ORDER BY created DESC) = 1
        )
        SELECT l.model, l.commit, l.settings, l.game_id, l.game_date, l.created,
            ph.phase, l.home_win_prob, l.home_pts AS pred_pts,
            l.away_pts AS pred_opp_pts,
            CASE WHEN r.team_id = l.home_team_id THEN r.pts ELSE r.opp_pts END
                AS pts,
            CASE WHEN r.team_id = l.home_team_id THEN r.opp_pts ELSE r.pts END
                AS opp_pts,
            l.poss AS pred_poss, (r.poss + r.opp_poss) / 2 AS poss
        FROM last l
        JOIN results r USING (game_id)
        JOIN ({phases()}) ph USING (game_id)
        WHERE r.season = {season}
    """)
    series = con.execute("""
        SELECT model, commit, settings, min(game_date), max(game_date)
        FROM scored GROUP BY ALL ORDER BY min(game_date), model
    """).fetchall()
    rows = []
    for model, sha, settings, first, last in series:
        where = f"model = '{model}' AND commit = '{sha}' AND settings = '{settings}'"
        published = {
            phase: (log_loss, brier)
            for phase, log_loss, brier in con.execute(f"""
                SELECT coalesce(phase, 'all'),
                    avg(-CASE WHEN pts > opp_pts THEN ln(p) ELSE ln(1 - p) END),
                    avg((p - (pts > opp_pts)::INTEGER) ** 2)
                FROM (
                    SELECT *, greatest(least(home_win_prob, 1 - 1e-6), 1e-6) AS p
                    FROM scored WHERE {where}
                )
                GROUP BY ROLLUP (phase)
            """).fetchall()
        }
        minutes = dict(
            con.execute(f"""
                WITH s AS (FROM scored WHERE {where}),
                projected AS (
                    SELECT m.game_id, m.player_id, m.min AS pred_min
                    FROM minutes_log m JOIN s USING (model, game_id, created)
                ), played AS (
                    SELECT game_id, player_id,
                        240 * min / sum(min) OVER (PARTITION BY game_id, team_id)
                            AS min
                    FROM player_games
                    WHERE min > 0 AND game_id IN (SELECT game_id FROM projected)
                )
                SELECT coalesce(s.phase, 'all'),
                    avg(abs(coalesce(pred_min, 0) - coalesce(a.min, 0)))
                FROM projected j
                FULL JOIN played a USING (game_id, player_id)
                JOIN s USING (game_id)
                GROUP BY ROLLUP (s.phase)
            """).fetchall()
        )
        for m in score(con.sql(f"FROM scored WHERE {where}")):
            log_loss, brier = published[m["phase"]]
            rows.append(
                {
                    "model": model,
                    "commit": sha,
                    "settings": settings,
                    "first_game_date": first,
                    "last_game_date": last,
                    **m,
                    "published_log_loss": log_loss,
                    "published_brier": brier,
                    "minutes_mae": minutes.get(m["phase"]),
                }
            )
    if rows:
        con.register("_rows", pa.Table.from_pylist(rows))
        con.execute("CREATE OR REPLACE TEMP TABLE live_metrics AS FROM _rows")
        con.unregister("_rows")
    (games,) = con.execute("SELECT count(DISTINCT game_id) FROM scored").fetchall()[0]
    return games


def validate(con: duckdb.DuckDBPyConnection, games: int) -> None:
    """Raise if a dataset looks wrong, before anything is written"""
    checks = {
        "every game is predicted once": f"""
            SELECT count(DISTINCT game_id) = {games} AND count(*) = {games}
            FROM predictions""",
        "predictions are plausible": """
            SELECT coalesce(bool_and(
                home_pts BETWEEN 70 AND 170 AND away_pts BETWEEN 70 AND 170
                AND poss BETWEEN 80 AND 120
                AND home_win_prob > 0 AND home_win_prob < 1
                AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL), true)
            FROM predictions""",
        "each team's minutes add up to 240": """
            SELECT coalesce(bool_and(abs(total - 240) < 0.1 AND players >= 5), true)
                AND count(*) = 2 * (SELECT count(*) FROM predictions)
            FROM (SELECT game_id, team_id, sum(min) AS total, count(*) AS players
                FROM minutes GROUP BY ALL)""",
        "minutes are plausible": """
            SELECT coalesce(bool_and(min BETWEEN 0 AND 48
                AND play_prob BETWEEN 0 AND 1), true) FROM minutes""",
        "a rating per player and day": """
            SELECT count(*) = count(DISTINCT (as_of_date, player_id))
                AND count(*) > 0
            FROM ratings_daily""",
        "a log row per model, game and time": """
            SELECT count(*) = count(DISTINCT (model, game_id, created))
            FROM prediction_log""",
        "logged predictions were made before their games": """
            SELECT coalesce(bool_and(created < start_time), true)
            FROM prediction_log""",
        "logged minutes belong to logged predictions": """
            SELECT count(*) = 0 FROM minutes_log m
            ANTI JOIN prediction_log p USING (model, game_id, created)""",
        "ratings are plausible": """
            SELECT coalesce(bool_and(abs(o_rating) < 30 AND abs(d_rating) < 30
                AND abs(o_prior) < 30 AND abs(d_prior) < 30
                AND possessions >= 0), true) FROM ratings_daily""",
    }
    failed = [
        name
        for name, query in checks.items()
        if not con.execute(query).fetchall()[0][0]
    ]
    if failed:
        raise ValueError(f"not publishing: {'; '.join(failed)}")


def write(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    """Write a query to parquet, moved into place once it's complete"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def source_updated(con: duckdb.DuckDBPyConnection) -> str:
    """When nba_data last updated a dataset, in UTC ISO 8601"""
    (updated,) = con.execute(
        "SELECT strftime(timezone('UTC', max(updated)), '%Y-%m-%dT%H:%M:%SZ') "
        "FROM nba.metadata"
    ).fetchall()[0]
    return updated


def seasons(root: Path, dataset: str) -> list[int]:
    return sorted(
        int(p.parent.name.removeprefix("season="))
        for p in (root / PREFIX / dataset).glob("season=*/data.parquet")
    )


def write_metadata(
    root: Path,
    written: dict[str, dict[str, Any]],
    run: dict[str, Any],
) -> None:
    """
    metadata.json: per dataset, its seasons and when it was updated, with
    what `written` has (commit, model, settings) for the ones just written
    and what an earlier build recorded for the rest; and `run`
    """
    path = metadata_path(root)
    before = json.loads(path.read_text()).get("datasets", {}) if path.exists() else {}
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    datasets = {}
    for dataset in sorted(set(before) | set(written)):
        found = seasons(root, dataset)
        if not found:
            continue
        entry = (
            {"updated": now, **written[dataset]}
            if dataset in written
            else {
                k: v
                for k, v in before[dataset].items()
                if k not in ("first_season", "last_season", "seasons")
            }
        )
        datasets[dataset] = {
            "first_season": found[0],
            "last_season": found[-1],
            "seasons": len(found),
            **entry,
        }
    text = json.dumps({"datasets": datasets, **run, "updated": now}, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def catalog(root: Path = PUBLISH, url: str | None = None) -> Path:
    """
    Write the catalog: a view per dataset over its files at `url` (or the
    mirror's, without one), and the metadata table. Returns its path
    """
    meta = json.loads(metadata_path(root).read_text())
    location = url.rstrip("/") if url else str(root.resolve())
    path = (
        root / PREFIX / f"{PREFIX}.duckdb"
        if url
        else root.parent / f"{PREFIX}.local.duckdb"
    )
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        with duckdb.connect() as con:
            # DuckDB's default 256KB blocks would make the file mostly empty
            con.execute(
                f"ATTACH '{tmp}' AS catalog "
                f"(BLOCK_SIZE 16384, STORAGE_VERSION '{STORAGE_VERSION}')"
            )
            con.execute("USE catalog")
            for dataset in meta["datasets"]:
                files = ", ".join(
                    f"'{location}/{PREFIX}/{dataset}/season={s}/data.parquet'"
                    for s in seasons(root, dataset)
                )
                con.execute(
                    f"""CREATE VIEW {dataset} AS SELECT * FROM read_parquet([{files}],
                        hive_partitioning = true, hive_types = {{'season': INTEGER}},
                        union_by_name = true)"""
                )
            con.execute("""
                CREATE TABLE metadata (
                    dataset VARCHAR PRIMARY KEY, first_season INTEGER,
                    last_season INTEGER, seasons INTEGER, updated TIMESTAMPTZ,
                    commit VARCHAR, model VARCHAR, settings VARCHAR
                )
            """)
            con.executemany(
                "INSERT INTO metadata VALUES (?, ?, ?, ?, ?::TIMESTAMPTZ, ?, ?, ?)",
                [
                    (
                        dataset,
                        d["first_season"],
                        d["last_season"],
                        d["seasons"],
                        d["updated"],
                        d.get("commit"),
                        d.get("model"),
                        json.dumps(d.get("settings"), sort_keys=True),
                    )
                    for dataset, d in sorted(meta["datasets"].items())
                ],
            )
            con.execute(
                f"COMMENT ON TABLE metadata IS 'also published as {PREFIX}/metadata.json'"
            )
            con.execute("USE memory")
            con.execute("DETACH catalog")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def current(root: Path = PUBLISH, url: str = URL) -> bool:
    """Whether the last build read nba_data's latest update"""
    path = metadata_path(root)
    if not path.exists():
        return False
    built = json.loads(path.read_text()).get("source_updated")
    with urllib.request.urlopen(f"{url}/nba/metadata.json", timeout=60) as r:
        nba = json.load(r)
    latest = max(d["updated"] for d in nba["datasets"].values())
    print(f"nba_data updated {latest}; the last build read {built}")
    # both are UTC ISO 8601 with a Z, so they compare as strings
    return built is not None and built >= latest


def verify(url: str = URL) -> None:
    """Attach the published catalog and read every dataset in it"""
    con = connect()
    con.execute(f"ATTACH '{url}/{PREFIX}/{PREFIX}.duckdb' AS {PREFIX} (READ_ONLY)")
    datasets = [
        d for (d,) in con.execute(f"SELECT dataset FROM {PREFIX}.metadata").fetchall()
    ]
    for dataset in datasets:
        (rows,) = con.execute(f"SELECT count(*) FROM {PREFIX}.{dataset}").fetchall()[0]
        print(f"{dataset}: {rows} rows")
        if not rows:
            raise ValueError(f"{dataset} is empty")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="fit and write the datasets")
    b.add_argument("--date", type=date.fromisoformat)
    b.add_argument(
        "--no-segments",
        action="store_true",
        help="use the segments already in out/ instead of rebuilding them",
    )
    c = commands.add_parser("catalog", help="write the catalog")
    c.add_argument("--url", help=f"where the files are, such as {URL}")
    u = commands.add_parser("current", help="exit 0 if nba_data hasn't updated")
    v = commands.add_parser("verify", help="read every published dataset")
    v.add_argument("--url", default=URL)
    for p in (b, c, u):
        p.add_argument("--root", type=Path, default=PUBLISH)
    args = parser.parse_args(argv)

    if args.command == "catalog":
        print(f"wrote {catalog(args.root, args.url)}")
        return
    if args.command == "current":
        sys.exit(0 if current(args.root) else 1)
    if args.command == "verify":
        verify(args.url)
        return

    day = args.date or datetime.now(EASTERN).date()
    con = connect()
    games = scheduled(con, day, 366)
    # every season, not just this one: the box-score prior is fit to RAPM
    # over the earlier seasons, and CI starts without any (~1 minute)
    if games and not args.no_segments:
        segments.main([])
    load(con)
    for dataset in build(con, day, args.root, commit()):
        print(f"wrote {dataset}")


if __name__ == "__main__":
    main()
