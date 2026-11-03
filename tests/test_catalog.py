import duckdb

from yarapm.catalog import proxy_settings, read, season_path, write_season


def test_proxy_settings_splits_out_credentials():
    env = {"HTTPS_PROXY": "http://us%2Fer:p%40ss@localhost:8080"}
    assert proxy_settings(env) == ("localhost:8080", "us/er", "p@ss")


def test_proxy_settings_lowercase_and_missing():
    assert proxy_settings({"https_proxy": "http://localhost:3128"}) == (
        "localhost:3128",
        "",
        "",
    )
    assert proxy_settings({}) is None


def test_write_and_read_seasons(tmp_path):
    con = duckdb.connect()
    for season in (2025, 2026):
        write_season(
            con, f"SELECT {season} AS season, 'x' AS v", "things", season, tmp_path
        )
    assert season_path("things", 2026, tmp_path).exists()

    rows = read(con, "things", tmp_path).order("season").fetchall()
    assert rows == [(2025, "x"), (2026, "x")]
    assert str(read(con, "things", tmp_path).types[0]) == "INTEGER"


def test_write_season_replaces(tmp_path):
    con = duckdb.connect()
    write_season(con, "SELECT 1 AS n", "things", 2026, tmp_path)
    write_season(con, "SELECT 2 AS n", "things", 2026, tmp_path)
    assert read(con, "things", tmp_path).fetchall() == [(2, 2026)]
