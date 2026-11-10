import duckdb

from yarapm.games import load_results


def test_load_results_skips_incomplete_box_scores():
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS nba")
    con.execute("""CREATE TABLE nba.games AS SELECT * FROM (VALUES
        (2027, 'g1', DATE '2026-11-08', 'regular_season', 'H', false),
        (2027, 'g2', DATE '2026-11-08', 'regular_season', 'H', false),
        (2027, 'g3', DATE '2026-11-08', 'regular_season', 'H', false))
        v(season, game_id, game_date, game_type, home_team_id, neutral_site)""")
    # g2's advanced box score isn't in yet: 0 possessions; g3's has none
    con.execute("""CREATE TABLE nba.team_game_logs AS SELECT * FROM (VALUES
        (2027, 'g1', 'H', 'A', 110, 100, 48.0),
        (2027, 'g1', 'A', 'H', 100, 101, 48.0),
        (2027, 'g2', 'H', 'A', 110, 0, 48.0),
        (2027, 'g2', 'A', 'H', 100, 0, 48.0),
        (2027, 'g3', 'H', 'A', 110, 100, 48.0),
        (2027, 'g3', 'A', 'H', 100, NULL, 48.0))
        v(season, game_id, team_id, opp_team_id, pts, poss, min)""")
    load_results(con)
    assert con.execute(
        "SELECT game_id, team_id, poss, opp_poss FROM results"
    ).fetchall() == [("g1", "H", 100, 101)]
