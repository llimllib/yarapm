"""
Which games we model, and their results.

Games are the regular season, play-in, playoffs and the Cup final: no
preseason or All-Star games. The 2020 bubble games, listed with home teams,
were played at a neutral site, and count as neutral.
"""

import duckdb

GAME_TYPES = ("regular_season", "play_in", "playoffs", "cup_final")

# games at the 2020 restart in Orlando were played with no crowd
BUBBLE_START = "2020-07-30"


def games_query(where: str = "true") -> str:
    """
    nba.games for the game types we model, with `neutral`, for the rows
    matching `where`
    """
    types = ", ".join(f"'{t}'" for t in GAME_TYPES)
    return f"""
        SELECT season, game_id, game_date, game_type, home_team_id,
            neutral_site OR (season = 2020 AND game_date >= DATE '{BUBBLE_START}')
                AS neutral
        FROM nba.games
        WHERE game_type IN ({types}) AND ({where})
    """


def schedule_query(where: str = "true") -> str:
    """
    nba.schedule's games of the types we model whose teams are set, for the
    rows matching `where`, as `Game`'s columns plus `start_time` (UTC): from the
    home team's side, or at a neutral site the team with the lower id's,
    with `home` NULL, as `load_results` has them
    """
    types = ", ".join(f"'{t}'" for t in GAME_TYPES)
    return f"""
        SELECT game_id, season, game_date, game_type,
            CASE WHEN neutral_site THEN least(home_team_id, away_team_id)
                ELSE home_team_id END AS team_id,
            CASE WHEN neutral_site THEN greatest(home_team_id, away_team_id)
                ELSE away_team_id END AS opp_team_id,
            CASE WHEN neutral_site THEN NULL ELSE true END AS home,
            -- in UTC, without a zone: TIMESTAMPTZ needs pytz in Python
            timezone('UTC', start_time) AS start_time
        FROM nba.schedule
        WHERE game_type IN ({types}) AND home_team_id IS NOT NULL
            AND away_team_id IS NOT NULL AND ({where})
    """


def load_results(con: duckdb.DuckDBPyConnection, where: str = "true") -> None:
    """
    Temp table `results`: one row per game, from the home team's side (at a
    neutral site, the team with the lower id), with each side's points and
    the NBA's possession count, and the game's length in minutes. `home` is
    NULL at a neutral site
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE results AS
        SELECT g.season, g.game_id, g.game_date, g.game_type,
            t.team_id, t.opp_team_id,
            CASE WHEN g.neutral OR g.home_team_id IS NULL THEN NULL
                 ELSE true END AS home,
            t.pts, o.pts AS opp_pts, t.poss, o.poss AS opp_poss, t.min
        FROM ({games_query(where)}) g
        JOIN nba.team_game_logs t USING (season, game_id)
        JOIN nba.team_game_logs o ON o.season = t.season AND o.game_id = t.game_id
            AND o.team_id = t.opp_team_id
        WHERE t.team_id = coalesce(
            CASE WHEN g.neutral THEN NULL ELSE g.home_team_id END,
            least(t.team_id, t.opp_team_id))
        ORDER BY g.game_date, g.game_id
    """)
