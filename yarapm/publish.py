"""
The datasets we publish (#27), under the `yarapm/` prefix of the
basketball-data bucket, with their own catalog:

    uv run python -m yarapm.publish build [--date 2026-11-05] [--no-segments]
    uv run python -m yarapm.publish catalog [--url https://basketball-data.billmill.org]

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
    metadata       per dataset: seasons, when it was updated, and the
                   commit and settings that built it; also the date the
                   build predicted from and nba_data's latest update

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
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from yarapm import segments
from yarapm.catalog import OUT, connect
from yarapm.harness import History, load
from yarapm.minutes import ProjectedMinutes
from yarapm.players import PlayerRatings
from yarapm.predict import EASTERN, Forecast, forecast, scheduled

PUBLISH = OUT / "publish"
PREFIX = "yarapm"
URL = "https://basketball-data.billmill.org"
DATASETS = ("ratings_daily", "predictions", "minutes")
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
) -> list[str]:
    """
    Fit and write the datasets into `root` for the season of the first game
    scheduled from `day`. Needs the tables `yarapm.harness.load` makes.
    Returns the datasets written
    """
    games = scheduled(con, day, 366)
    if not games:
        print(f"no games scheduled from {day}")
        return []
    season = games[0][0].season
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

    predictions_table(con, forecasts, season, day)
    minutes_table(con, minutes, chances, day)
    ratings_table(con, rated, sha, settings, existing if have else None)
    validate(con, len(games))

    for dataset in DATASETS:
        write(con, f"FROM {dataset}", dataset_path(root, dataset, season))
    write_metadata(
        root,
        DATASETS,
        {
            "commit": sha,
            "model": model.name,
            "settings": model.settings,
        },
        {
            "as_of_date": day.isoformat(),
            "source_updated": source_updated(con),
        },
    )
    return list(DATASETS)


def predictions_table(
    con: duckdb.DuckDBPyConnection, forecasts: list[Forecast], season: int, day: date
) -> None:
    """Temp table `predictions`, from the home team's side, as nba.schedule lists it"""
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
        """
        CREATE OR REPLACE TEMP TABLE predictions AS
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
    written: tuple[str, ...] | list[str],
    about: dict[str, Any],
    run: dict[str, Any],
) -> None:
    """
    metadata.json: per dataset, its seasons and when it was updated, with
    `about` (commit, model, settings) for the ones just written and what
    an earlier build recorded for the rest; and `run`
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
            {"updated": now, **about}
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
    for p in (b, c):
        p.add_argument("--root", type=Path, default=PUBLISH)
    args = parser.parse_args(argv)

    if args.command == "catalog":
        print(f"wrote {catalog(args.root, args.url)}")
        return

    day = args.date or datetime.now(EASTERN).date()
    con = connect()
    games = scheduled(con, day, 366)
    if games and not args.no_segments:
        segments.main(["--season", str(games[0][0].season)])
    load(con)
    for dataset in build(con, day, args.root, commit()):
        print(f"wrote {dataset}")


if __name__ == "__main__":
    main()
