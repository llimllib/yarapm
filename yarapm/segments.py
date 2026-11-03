"""
Build `segments`: every stretch of a game with the same ten players on the
floor, one row for each team's offense in it, from nba_data's
`game_rotations` (who's on) and `game_events` (what happened).

    uv run python -m yarapm.segments [--season 2026 ...]

Segments run between the times in `game_rotations` where anyone checks in
or out. A play belongs to the segment it happens in. On the clock tick of a
substitution:

- a play listed before the substitution belongs to the lineup going out, and
  one listed after to the lineup coming in
- except free throws, which belong to the lineup on for the foul (the one
  going out) unless a new foul comes after the substitution, as
  `game_rotations` counts them

In 2025-26 that makes every player's segments add up to their
plus-minus in 96% of games. The rest are off by a point or two from free
throws on a substitution's tick that `game_rotations` gives to different
lineups for different players (a substitute shooting for an injured
player, or both teams subbing around a foul on the same tick), which no
split into segments can match. A game is kept if every player is within
PM_TOLERANCE, each team's points match its box score, and its possessions
are within POSS_TOLERANCE of the NBA's.

Possessions: a team's possession starts when it gets the ball, from its
first shot, free throw, turnover or rebound after the other team's (as in
nba_data's `poss`, see its data dictionary). It's credited to the segment
of its first shot, free throw or turnover, since a substitution often comes
between a rebound and the play that follows; one with no play (a rebound as
the period ends) is credited at the rebound. When a period ends on a make,
a turnover or the last of a trip's made free throws, the other team gets a
possession at the end of the period. Technical free throws don't start or
continue a possession.

Games: those in `yarapm.games` with `matches_box_score`.

Columns, for the offense in the segment:

    season, game_id, game_date, game_type
    seg                          the segment's number in the game, from 1
    start_time, end_time         seconds from the start, as in game_rotations
    off_team_id, def_team_id
    off_home                     NULL at a neutral site
    off_player_ids, def_player_ids   five each, sorted
    poss, pts
    fg2a, fg2m, fg3a, fg3m, fta, ftm, tov
    oreb, dreb                   rebounds of the offense's live misses, by the
                                 offense and the defense, team rebounds included
"""

import argparse
from pathlib import Path

import duckdb

from yarapm.catalog import OUT, connect, write_season
from yarapm.games import games_query

NAME = "segments"

# a player's segments can differ from their plus-minus in game_rotations
# by this much before the game is dropped: free throws on a substitution's
# tick can't always go to the same lineup as game_rotations puts them
PM_TOLERANCE = 2

# a team's possessions in a game can differ from the NBA's count by this
# much before the game is dropped
POSS_TOLERANCE = 3


def load_sources(con: duckdb.DuckDBPyConnection, season: int) -> None:
    """Temp tables `games`, `rotations`, `events` and `box` for a season."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE games AS
        {games_query(f"season = {season}")}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rotations AS
        SELECT game_id, team_id, player_id, in_time, out_time, plus_minus
        FROM nba.game_rotations
        WHERE season = {season} AND matches_box_score
            AND game_id IN (SELECT game_id FROM games)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT game_id, action_id, period, time, team_id, person_id,
            action_type, sub_type, shot_result, shot_value, pts
        FROM nba.game_events
        WHERE season = {season} AND game_id IN (SELECT game_id FROM rotations)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE box AS
        SELECT game_id, team_id, pts, poss
        FROM nba.team_game_logs
        WHERE season = {season} AND game_id IN (SELECT game_id FROM rotations)
    """)


def build(con: duckdb.DuckDBPyConnection) -> None:
    """
    From the temp tables `games`, `rotations`, `events` and `box`, build
    `segments` (the rows) and `game_checks` (one row per game, whether it
    adds up and why not)
    """
    # segments: between consecutive check-in and check-out times
    con.execute("""
        CREATE OR REPLACE TEMP TABLE bounds AS
        SELECT game_id, start_time, end_time,
            row_number() OVER (PARTITION BY game_id ORDER BY start_time) AS seg
        FROM (
            SELECT game_id, t AS start_time,
                lead(t) OVER (PARTITION BY game_id ORDER BY t) AS end_time
            FROM (SELECT DISTINCT game_id, unnest([in_time, out_time]) AS t
                  FROM rotations)
        )
        WHERE end_time IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE lineups AS
        SELECT b.game_id, b.seg, r.team_id,
            list(r.player_id ORDER BY r.player_id) AS player_ids
        FROM bounds b
        JOIN rotations r ON r.game_id = b.game_id
            AND r.in_time <= b.start_time AND r.out_time >= b.end_time
        GROUP BY ALL
    """)

    # each event's segment. `after` decides the side of a boundary an event
    # exactly on one goes to; it's nudged by a hundredth of a second, and
    # times are in tenths
    con.execute("""
        CREATE OR REPLACE TEMP TABLE timed AS
        WITH e AS (
            SELECT *,
                CASE WHEN period <= 4 THEN 720 * (period - 1)
                     ELSE 2880 + 300 * (period - 5) END AS period_start,
                CASE WHEN period <= 4 THEN 720 * period
                     ELSE 2880 + 300 * (period - 4) END AS period_end,
                min(action_id) FILTER (WHERE action_type = 'Substitution') OVER tick
                    AS first_sub
            FROM events
            WHERE time IS NOT NULL
            WINDOW tick AS (PARTITION BY game_id, period, time)
        ), f AS (
            SELECT *,
                -- a foul on the tick after the first substitution: the free
                -- throws after it are the new lineup's
                max(CASE WHEN action_type = 'Foul' AND action_id > first_sub
                         THEN action_id END) OVER (
                    PARTITION BY game_id, period, time ORDER BY action_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS new_foul
            FROM e
        )
        SELECT *,
            CASE
                WHEN time = period_start THEN true
                WHEN time = period_end THEN false
                WHEN first_sub IS NULL OR action_id < first_sub THEN false
                WHEN action_type = 'Free Throw' THEN new_foul IS NOT NULL
                ELSE true
            END AS after
        FROM f
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE placed AS
        SELECT t.*, b.seg
        FROM timed t
        JOIN bounds b ON b.game_id = t.game_id
            AND b.start_time < t.time + CASE WHEN t.after THEN 0.01 ELSE -0.01 END
            AND b.end_time > t.time + CASE WHEN t.after THEN 0.01 ELSE -0.01 END
    """)
    possessions(con)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE segments AS
        WITH stats AS (
            SELECT game_id, seg, team_id,
                sum(pts) AS pts,
                count(*) FILTER (WHERE action_type IN ('Made Shot', 'Missed Shot')
                                 AND shot_value = 2) AS fg2a,
                count(*) FILTER (WHERE action_type = 'Made Shot' AND shot_value = 2) AS fg2m,
                count(*) FILTER (WHERE action_type IN ('Made Shot', 'Missed Shot')
                                 AND shot_value = 3) AS fg3a,
                count(*) FILTER (WHERE action_type = 'Made Shot' AND shot_value = 3) AS fg3m,
                count(*) FILTER (WHERE action_type = 'Free Throw') AS fta,
                count(*) FILTER (WHERE action_type = 'Free Throw' AND pts > 0) AS ftm,
                count(*) FILTER (WHERE action_type = 'Turnover') AS tov
            FROM placed
            GROUP BY ALL
        ), rebs AS (
            -- rebounds of a team's misses, by the team (oreb) or its
            -- opponent (dreb), both on the missing team's offense row
            SELECT game_id, seg, miss_team AS team_id,
                count(*) FILTER (WHERE team_id = miss_team) AS oreb,
                count(*) FILTER (WHERE team_id <> miss_team) AS dreb
            FROM rebounds
            GROUP BY ALL
        ), poss AS (
            SELECT game_id, seg, team_id, count(*) AS poss
            FROM possession_credits
            GROUP BY ALL
        )
        SELECT g.game_id, g.game_date, g.game_type, b.seg, b.start_time, b.end_time,
            o.team_id AS off_team_id, d.team_id AS def_team_id,
            CASE WHEN g.neutral OR g.home_team_id IS NULL THEN NULL
                 ELSE o.team_id = g.home_team_id END AS off_home,
            o.player_ids AS off_player_ids, d.player_ids AS def_player_ids,
            coalesce(p.poss, 0)::INTEGER AS poss,
            coalesce(s.pts, 0)::INTEGER AS pts,
            coalesce(s.fg2a, 0)::INTEGER AS fg2a, coalesce(s.fg2m, 0)::INTEGER AS fg2m,
            coalesce(s.fg3a, 0)::INTEGER AS fg3a, coalesce(s.fg3m, 0)::INTEGER AS fg3m,
            coalesce(s.fta, 0)::INTEGER AS fta, coalesce(s.ftm, 0)::INTEGER AS ftm,
            coalesce(s.tov, 0)::INTEGER AS tov,
            coalesce(r.oreb, 0)::INTEGER AS oreb, coalesce(r.dreb, 0)::INTEGER AS dreb
        FROM bounds b
        JOIN games g USING (game_id)
        JOIN lineups o ON o.game_id = b.game_id AND o.seg = b.seg
        JOIN lineups d ON d.game_id = b.game_id AND d.seg = b.seg
            AND d.team_id <> o.team_id
        LEFT JOIN stats s ON s.game_id = b.game_id AND s.seg = b.seg
            AND s.team_id = o.team_id
        LEFT JOIN rebs r ON r.game_id = b.game_id AND r.seg = b.seg
            AND r.team_id = o.team_id
        LEFT JOIN poss p ON p.game_id = b.game_id AND p.seg = b.seg
            AND p.team_id = o.team_id
        ORDER BY b.game_id, b.seg, off_team_id
    """)
    checks(con)


def possessions(con: duckdb.DuckDBPyConnection) -> None:
    """Temp tables `possession_credits` and `rebounds`, from `placed`"""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE marked AS
        SELECT *,
            action_type IN ('Made Shot', 'Missed Shot', 'Turnover')
                OR (action_type = 'Free Throw' AND sub_type NOT LIKE '%Technical%')
                AS play,
            action_type = 'Missed Shot' OR (action_type = 'Free Throw'
                AND shot_result = 'Missed' AND sub_type NOT LIKE '%Technical%')
                AS miss,
            -- a free throw that ends the trip: the rebound of a miss is live,
            -- and a make gives the other team the ball
            action_type = 'Free Throw' AND sub_type NOT LIKE '%Technical%'
                AND regexp_extract(sub_type, '([0-9]) of ([0-9])', 1)
                    = regexp_extract(sub_type, '([0-9]) of ([0-9])', 2) AS last_ft
        FROM placed
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE rebounds AS
        SELECT * FROM (
            SELECT *,
                last_value(CASE WHEN miss THEN team_id END IGNORE NULLS) OVER w
                    AS miss_team,
                last_value(CASE WHEN miss THEN action_type = 'Missed Shot' OR last_ft
                           END IGNORE NULLS) OVER w AS miss_live
            FROM marked
            WINDOW w AS (PARTITION BY game_id, period ORDER BY action_id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        )
        WHERE action_type = 'Rebound' AND team_id IS NOT NULL AND miss_live
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE possession_credits AS
        WITH ind AS (
            -- events that say who has the ball
            SELECT game_id, period, action_id, seg, team_id, play,
                action_type, pts, last_ft
            FROM marked
            WHERE team_id IS NOT NULL AND play
            UNION ALL
            SELECT game_id, period, action_id, seg, team_id, false,
                action_type, pts, false
            FROM rebounds
        ), runs AS (
            SELECT *, sum(new::INTEGER) OVER (
                    PARTITION BY game_id, period ORDER BY action_id) AS n
            FROM (
                SELECT *, team_id IS DISTINCT FROM lag(team_id) OVER (
                    PARTITION BY game_id, period ORDER BY action_id) AS new
                FROM ind
            )
        ), credited AS (
            SELECT game_id, team_id,
                coalesce(arg_min(seg, action_id) FILTER (WHERE play),
                         arg_min(seg, action_id)) AS seg
            FROM runs
            GROUP BY game_id, period, n, team_id
        ), period_last AS (
            -- the other team gets the ball after a period ends on a make,
            -- a turnover or a trip's last made free throw
            SELECT game_id, period,
                arg_max(team_id, action_id) AS team_id,
                arg_max(action_type = 'Turnover'
                        OR (action_type = 'Made Shot')
                        OR (last_ft AND pts > 0), action_id) AS gives_ball,
                max(seg) AS seg
            FROM ind
            GROUP BY ALL
        )
        SELECT game_id, team_id, seg FROM credited
        UNION ALL
        SELECT p.game_id, s.team_id, b.seg
        FROM period_last p
        JOIN (SELECT DISTINCT game_id, team_id FROM rotations) s
            ON s.game_id = p.game_id AND s.team_id <> p.team_id
        -- the segment the period ends in
        JOIN bounds b ON b.game_id = p.game_id
            AND b.start_time < (CASE WHEN p.period <= 4 THEN 720 * p.period
                                     ELSE 2880 + 300 * (p.period - 4) END)
            AND b.end_time >= (CASE WHEN p.period <= 4 THEN 720 * p.period
                                    ELSE 2880 + 300 * (p.period - 4) END)
        WHERE p.gives_ball
    """)


def checks(con: duckdb.DuckDBPyConnection) -> None:
    """
    `game_checks`: per game, the most any player's segments are off from
    their plus-minus, whether each team's add up to its box score points,
    and the most a team's possessions are off from the NBA's
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE game_checks AS
        WITH margins AS (
            SELECT o.game_id, o.off_team_id AS team_id, o.pts - d.pts AS margin,
                unnest(o.off_player_ids) AS player_id
            FROM segments o
            JOIN segments d ON d.game_id = o.game_id AND d.seg = o.seg
                AND d.off_team_id = o.def_team_id
        ), players AS (
            SELECT game_id, team_id, player_id, sum(margin) AS plus_minus
            FROM margins GROUP BY ALL
        ), stints AS (
            SELECT game_id, team_id, player_id, sum(plus_minus) AS plus_minus
            FROM rotations GROUP BY ALL
        ), pm AS (
            SELECT game_id,
                max(abs(coalesce(p.plus_minus, 0) - s.plus_minus)) AS pm_diff
            FROM stints s
            LEFT JOIN players p USING (game_id, team_id, player_id)
            GROUP BY ALL
        ), teams AS (
            SELECT s.game_id, s.off_team_id AS team_id, sum(s.pts) AS pts,
                sum(s.poss) AS poss, any_value(b.pts) AS box_pts,
                any_value(b.poss) AS box_poss
            FROM segments s
            JOIN box b ON b.game_id = s.game_id AND b.team_id = s.off_team_id
            GROUP BY ALL
        )
        SELECT game_id, pm.pm_diff,
            bool_and(pts = box_pts) AS pts_match,
            max(abs(poss - box_poss)) AS poss_diff,
            pm.pm_diff <= {PM_TOLERANCE} AND bool_and(pts = box_pts)
                AND max(abs(poss - box_poss)) <= {POSS_TOLERANCE} AS ok
        FROM teams JOIN pm USING (game_id)
        GROUP BY game_id, pm.pm_diff
    """)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, action="append")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    con = connect()
    seasons = args.season or [
        s
        for (s,) in con.sql(
            "SELECT DISTINCT season FROM nba.game_rotations ORDER BY season"
        ).fetchall()
    ]
    for season in seasons:
        load_sources(con, season)
        build(con)
        report(con, season)
        write_season(
            con,
            f"""SELECT {season}::INTEGER AS season, s.* FROM segments s
                WHERE game_id IN (SELECT game_id FROM game_checks WHERE ok)""",
            NAME,
            season,
            args.out,
        )


def report(con: duckdb.DuckDBPyConnection, season: int) -> None:
    games, rotations_ok, kept, bad_pm, bad_pts, bad_poss = con.execute(
        """
        SELECT (SELECT count(*) FROM games),
            (SELECT count(DISTINCT game_id) FROM rotations),
            count(*) FILTER (WHERE ok),
            count(*) FILTER (WHERE pm_diff > ?),
            count(*) FILTER (WHERE NOT pts_match),
            count(*) FILTER (WHERE poss_diff > ?)
        FROM game_checks
    """,
        [PM_TOLERANCE, POSS_TOLERANCE],
    ).fetchall()[0]
    print(
        f"{season}: kept {kept} of {games} games; dropped "
        f"{games - rotations_ok} without matching rotations, "
        f"{rotations_ok - kept} failing checks (plus-minus {bad_pm}, "
        f"points {bad_pts}, possessions {bad_poss})"
    )


if __name__ == "__main__":
    main()
