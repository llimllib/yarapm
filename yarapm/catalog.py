"""
Read nba_data's catalog, and the tables we build from it.

connect() returns a DuckDB connection with the catalog attached as `nba`, so
source data is always `nba.<table>`:

    con = connect()
    con.sql("SELECT count(*) FROM nba.games WHERE season = 2026")

What we build is cached as parquet under out/, laid out like nba_data's
bucket, one file per season: out/<dataset>/season=<season>/data.parquet.
write_season() writes one and read() reads every season of a dataset.
"""

import os
import urllib.parse
from pathlib import Path

import duckdb

CATALOG_URL = "https://basketball-data.billmill.org/nba/nba.duckdb"
OUT = Path("out")


def proxy_settings(env: dict[str, str]) -> tuple[str, str, str] | None:
    """
    (host:port, username, password) from HTTPS_PROXY, or None without one.
    DuckDB reads the proxy variables itself but can't parse credentials in
    them, so a proxy with credentials has to be set explicitly.
    """
    url = env.get("HTTPS_PROXY") or env.get("https_proxy")
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    return (
        f"{parsed.hostname}:{parsed.port}",
        urllib.parse.unquote(parsed.username or ""),
        urllib.parse.unquote(parsed.password or ""),
    )


def connect(catalog: str = CATALOG_URL) -> duckdb.DuckDBPyConnection:
    """A connection with nba_data's catalog attached, read-only, as `nba`."""
    con = duckdb.connect()
    proxy = proxy_settings(dict(os.environ))
    if proxy:
        host, username, password = proxy
        con.execute("SET http_proxy = ?", [host])
        con.execute("SET http_proxy_username = ?", [username])
        con.execute("SET http_proxy_password = ?", [password])
    con.execute(f"ATTACH '{catalog}' AS nba (READ_ONLY)")
    return con


def season_path(dataset: str, season: int, out: Path = OUT) -> Path:
    return out / dataset / f"season={season}" / "data.parquet"


def write_season(
    con: duckdb.DuckDBPyConnection,
    query: str,
    dataset: str,
    season: int,
    out: Path = OUT,
) -> Path:
    """Write one season of a dataset from a query, replacing any earlier file."""
    path = season_path(dataset, season, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    return path


def read(
    con: duckdb.DuckDBPyConnection, dataset: str, out: Path = OUT
) -> duckdb.DuckDBPyRelation:
    """Every season of a dataset. `season` comes from the path, as an INTEGER."""
    return con.sql(
        f"""SELECT * FROM read_parquet('{out / dataset}/season=*/data.parquet',
            hive_partitioning = true, hive_types = {{'season': INTEGER}})"""
    )
