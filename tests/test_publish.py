import json
from datetime import date

import duckdb
import pytest

from yarapm.publish import (
    catalog,
    dataset_path,
    metadata_path,
    ratings_table,
    validate,
    write,
    write_metadata,
)


def rating(player, o=1.0, d=-1.0):
    return (player, "T", o, d, 0.5, -0.5, 100)


def test_ratings_keep_earlier_days(tmp_path):
    con = duckdb.connect()
    ratings_table(
        con,
        {date(2026, 11, 1): [rating("a")], date(2026, 11, 2): [rating("a")]},
        "old",
        "{}",
        None,
    )
    path = dataset_path(tmp_path, "ratings_daily", 2027)
    write(con, "FROM ratings_daily", path)

    # the 2nd is refit, and the 3rd added
    ratings_table(
        con,
        {date(2026, 11, 2): [rating("a", o=2)], date(2026, 11, 3): [rating("b")]},
        "new",
        "{}",
        path,
    )
    assert con.sql(
        "SELECT as_of_date, player_id, o_rating, rating, commit FROM ratings_daily"
    ).fetchall() == [
        (date(2026, 11, 1), "a", 1, 0, "old"),
        (date(2026, 11, 2), "a", 2, 1, "new"),
        (date(2026, 11, 3), "b", 1, 0, "new"),
    ]


def tables(con, home_pts=110.0, minutes=240.0, logged="2026-11-01 12:00:00+00"):
    con.execute(f"""CREATE OR REPLACE TABLE prediction_log AS
        SELECT model, 'g1' AS game_id, '{logged}'::TIMESTAMPTZ AS created,
            '2026-11-02 00:00:00+00'::TIMESTAMPTZ AS start_time
        FROM (VALUES ('rapm'), ('team')) v(model)""")
    con.execute("""CREATE OR REPLACE TABLE minutes_log AS
        SELECT 'rapm' AS model, 'g1' AS game_id,
            '2026-11-01 12:00:00+00'::TIMESTAMPTZ AS created""")
    con.execute(f"""CREATE OR REPLACE TABLE predictions AS SELECT 'g1' AS game_id,
        'H' AS home_team_id, 'A' AS away_team_id, '{home_pts}'::DOUBLE AS home_pts,
        105.0 AS away_pts, 100.0 AS poss, 0.6 AS home_win_prob""")
    con.execute(f"""CREATE OR REPLACE TABLE minutes AS
        SELECT 'g1' AS game_id, team_id, 'p' || i AS player_id,
            {minutes} / 8 AS min, 0.9 AS play_prob
        FROM (VALUES ('H'), ('A')) t(team_id), range(8) r(i)""")
    ratings_table(con, {date(2026, 11, 1): [rating("a")]}, "c", "{}", None)


def test_validate():
    con = duckdb.connect()
    tables(con)
    validate(con, 1)
    with pytest.raises(ValueError, match="every game is predicted once"):
        validate(con, 2)
    tables(con, home_pts=float("nan"))
    with pytest.raises(ValueError, match="predictions are plausible"):
        validate(con, 1)
    tables(con, minutes=200)
    with pytest.raises(ValueError, match="add up to 240"):
        validate(con, 1)
    tables(con, logged="2026-11-02 01:00:00+00")
    with pytest.raises(ValueError, match="made before their games"):
        validate(con, 1)
    tables(con)
    con.execute("INSERT INTO prediction_log SELECT * FROM prediction_log LIMIT 1")
    with pytest.raises(ValueError, match="a log row per model, game and time"):
        validate(con, 1)


def test_metadata_and_catalog(tmp_path):
    con = duckdb.connect()
    tables(con)
    for dataset, season in [
        ("predictions", 2027),
        ("ratings_daily", 2026),
        ("ratings_daily", 2027),
    ]:
        query = (
            "FROM ratings_daily" if dataset == "ratings_daily" else "FROM predictions"
        )
        write(con, query, dataset_path(tmp_path, dataset, season))
    about = {"commit": "abc", "model": "rapm", "settings": {"ridge": 3000}}
    write_metadata(tmp_path, {"predictions": about, "ratings_daily": about}, {"x": 1})
    # a later build that only writes predictions keeps what ratings_daily had
    write_metadata(tmp_path, {"predictions": about | {"commit": "def"}}, {"x": 2})

    meta = json.loads(metadata_path(tmp_path).read_text())
    assert meta["x"] == 2
    assert meta["datasets"]["predictions"]["commit"] == "def"
    ratings = meta["datasets"]["ratings_daily"]
    assert (ratings["commit"], ratings["first_season"], ratings["seasons"]) == (
        "abc",
        2026,
        2,
    )

    path = catalog(tmp_path)
    assert path == tmp_path.parent / "yarapm.local.duckdb"
    con = duckdb.connect()
    con.execute(f"ATTACH '{path}' AS yarapm (READ_ONLY)")
    assert con.sql(
        "SELECT season, count(*) FROM yarapm.ratings_daily GROUP BY ALL ORDER BY ALL"
    ).fetchall() == [(2026, 1), (2027, 1)]
    assert con.sql("SELECT home_pts FROM yarapm.predictions").fetchall() == [(110,)]
    assert con.sql(
        "SELECT dataset, commit, settings FROM yarapm.metadata ORDER BY ALL"
    ).fetchall() == [
        ("predictions", "def", '{"ridge": 3000}'),
        ("ratings_daily", "abc", '{"ridge": 3000}'),
    ]
