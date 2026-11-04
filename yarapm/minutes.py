"""
Minutes for each player in a game, which the player model (yarapm.players)
weights ratings by. Three sources:

    actual   the minutes each player played: a leak, which measures rating
             error on its own (#8)
    known    who plays is known (`History.active`, a leak standing in for
             injury reports and starting lineups) and their minutes are
             projected (#10)
    unknown  who plays is projected too (#11)

Minutes when playing: a decayed average of a player's minutes in their
recent games, older games weighted less by `half_life` (in the player's
games, not days, so an injury doesn't age them), scaled to 240 a team so
overtime doesn't count. A player who started their last game and doesn't
usually start gets `start_bonus` × the difference more, and one who
usually starts and didn't, that much less. The player's team doesn't
matter, so a traded player brings their minutes with them. A player with no
games gets NEW_PLAYER_MINUTES.

Availability: the chance a player on the roster plays, looked up by the
number of the team's games since they last played and how many of its last
ROSTER_GAMES they played, from how often players in the same spot played,
over the seasons before this one. The roster is the players who played one
of the team's last ROSTER_GAMES games and haven't played for another team
since. Players who change teams in the offseason or in a trade aren't on
their new team's roster until they've played for it.

A team's minutes are scaled to 240, with no player above CAP (the excess
goes to the others in proportion). With unknown availability, each player
gets their chance of playing × their minutes when playing, then scaled.

The defaults are the best of a grid on the tuning seasons (#10), weekly
refits, known availability, by minutes MAE: half lives of 2-20 games and
start bonuses of 0-10. Margin RMSE moved by 0.03 at most over the grid.
"""

from collections import defaultdict

from yarapm.harness import Game, History

TEAM_MINUTES = 240
CAP = 42
NEW_PLAYER_MINUTES = 10
ROSTER_GAMES = 10
# games older than this many half lives, or days, are left out
HALF_LIVES = 8
LOOKBACK_DAYS = 365
# seasons of history the availability table is fit on
AVAILABILITY_SEASONS = 3

Minutes = dict[tuple[str, str], dict[str, float]]


def allocate(
    weights: dict[str, float], total: float = TEAM_MINUTES, cap: float = CAP
) -> dict[str, float]:
    """
    `weights` scaled to add up to `total`, none above `cap` (so with too
    few players, everyone gets `cap`)
    """
    minutes = {p: max(w, 0.0) for p, w in weights.items()}
    capped: set[str] = set()
    while True:
        free = [p for p in minutes if p not in capped]
        weight = sum(minutes[p] for p in free)
        if weight <= 0:
            return minutes
        scale = (total - cap * len(capped)) / weight
        for p in free:
            minutes[p] *= scale
        over = {p for p in free if minutes[p] > cap}
        if not over:
            return minutes
        for p in over:
            minutes[p] = cap
        capped |= over


def by_team(rows: list[tuple[str, str, str, float]]) -> Minutes:
    out: Minutes = defaultdict(dict)
    for game_id, team_id, player_id, m in rows:
        out[game_id, team_id][player_id] = m
    return out


class ActualMinutes:
    def fit(self, history: History) -> None:
        self.history = history

    def predict(self, games: list[Game]) -> Minutes:
        played = by_team(self.history.actual_minutes([g.game_id for g in games]))
        # scaled to 240, as the minutes are scored, but not capped
        return {k: allocate(v, cap=TEAM_MINUTES) for k, v in played.items()}


class ProjectedMinutes:
    def __init__(
        self, availability: str = "known", half_life: float = 5, start_bonus: float = 5
    ):
        if availability not in ("known", "unknown"):
            raise ValueError(f"availability {availability!r} is 'known' or 'unknown'")
        self.availability = availability
        self.half_life = half_life
        self.start_bonus = start_bonus
        self.playing: dict[str, float] = {}
        self.chances: dict[tuple[int, int], float] = {}
        self.chances_season: int | None = None
        self.rosters: dict[str, dict[str, tuple[int, int]]] = {}

    def fit(self, history: History) -> None:
        self.history = history
        self.playing = self.when_playing(history)
        if self.availability == "unknown":
            if not self.chances or history.season != self.chances_season:
                self.chances = availability(history)
                self.chances_season = history.season
            self.rosters = rosters(history)

    def when_playing(self, history: History) -> dict[str, float]:
        """Each player's projected minutes if they play"""
        h = self.half_life
        rows = history.con.execute(f"""
            WITH g AS (
                SELECT player_id, started,
                    {TEAM_MINUTES} * min
                        / sum(min) OVER (PARTITION BY game_id, team_id) AS min,
                    row_number() OVER (PARTITION BY player_id
                        ORDER BY game_date DESC, game_id DESC) - 1 AS k
                FROM ({history.player_games.sql_query()})
                WHERE min > 0
                    AND game_date >= DATE '{history.cutoff}' - {LOOKBACK_DAYS}
            )
            SELECT player_id,
                sum(0.5 ** (k / {h}) * min) / sum(0.5 ** (k / {h})),
                coalesce(sum(0.5 ** (k / {h}) * started::INTEGER)
                    / sum(0.5 ** (k / {h})) FILTER (WHERE started IS NOT NULL), 0),
                coalesce(arg_min(started, k)::INTEGER, 0)
            FROM g
            WHERE k < {HALF_LIVES * h}
            GROUP BY player_id
        """).fetchall()
        return {p: m + self.start_bonus * (last - share) for p, m, share, last in rows}

    def minutes(self, player: str) -> float:
        return self.playing.get(player, NEW_PLAYER_MINUTES)

    def predict(self, games: list[Game]) -> Minutes:
        if self.availability == "known":
            out: Minutes = defaultdict(dict)
            for game_id, team_id, player in self.history.active(
                [g.game_id for g in games]
            ):
                out[game_id, team_id][player] = self.minutes(player)
            return {k: allocate(v) for k, v in out.items()}

        out = {}
        for g in games:
            for team in (g.team_id, g.opp_team_id):
                out[g.game_id, team] = allocate(
                    {
                        p: chance * self.minutes(p)
                        for p, chance in self.playing_chances(team).items()
                    }
                )
        return out

    def chance(self, since: int, played: int) -> float:
        return self.chances.get((since, played), 0.0)

    def playing_chances(self, team: str) -> dict[str, float]:
        """With unknown availability, the chance each player on a roster plays"""
        return {p: self.chance(*spot) for p, spot in self.rosters.get(team, {}).items()}


def team_games(history: History) -> str:
    """SQL for each team's games before the cutoff, numbered from its first"""
    return f"""
        SELECT season, team_id, game_id, row_number() OVER (
            PARTITION BY team_id ORDER BY game_date, game_id) AS n
        FROM (
            SELECT season, team_id, game_id, game_date
            FROM ({history.results.sql_query()})
            UNION ALL
            SELECT season, opp_team_id, game_id, game_date
            FROM ({history.results.sql_query()})
        )
    """


def availability(history: History) -> dict[tuple[int, int], float]:
    """
    How often a player on the roster plays, by (the team's games since
    they last played for it, how many of its last ROSTER_GAMES they played),
    over the AVAILABILITY_SEASONS seasons before this one (or before the
    cutoff, with no season)
    """
    # a condition on a season. The spots are in those seasons too: this
    # season's first games would count last season's players as not playing
    seasons = (
        "IS NOT NULL"
        if history.season is None
        else f"BETWEEN {history.season - AVAILABILITY_SEASONS} AND {history.season - 1}"
    )
    rows = history.con.execute(f"""
        WITH tg AS ({team_games(history)}),
        played AS (
            SELECT p.player_id, tg.team_id, tg.n
            FROM ({history.player_games.sql_query()}) p
            JOIN tg USING (game_id, team_id)
            WHERE p.min > 0 AND p.season {seasons}
        ), spots AS (
            SELECT t.team_id, t.n, a.player_id,
                t.n - 1 - max(a.n) AS since, count(*) AS played
            FROM tg t
            JOIN played a ON a.team_id = t.team_id
                AND a.n BETWEEN t.n - {ROSTER_GAMES} AND t.n - 1
            WHERE t.season {seasons}
            GROUP BY t.team_id, t.n, a.player_id
        )
        SELECT since, s.played, count(*), count(a.n)
        FROM spots s
        LEFT JOIN played a USING (team_id, n, player_id)
        GROUP BY since, s.played
    """).fetchall()
    # a half a game's prior toward even, for spots seen rarely
    return {(since, n): (yes + 0.5) / (total + 1) for since, n, total, yes in rows}


def rosters(history: History) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Each team's roster as of the cutoff: players who played one of its last
    ROSTER_GAMES games and not for another team since, with (games since
    they last played for it, how many of those games they played)
    """
    rows = history.con.execute(f"""
        WITH tg AS ({team_games(history)}),
        last AS (SELECT team_id, max(n) AS n FROM tg GROUP BY team_id),
        played AS (
            SELECT p.player_id, p.game_date, tg.team_id, tg.n
            FROM ({history.player_games.sql_query()}) p
            JOIN tg USING (game_id, team_id)
            WHERE p.min > 0
        ), latest AS (
            SELECT player_id, arg_max(team_id, game_date) AS team_id
            FROM played GROUP BY player_id
        )
        SELECT a.team_id, a.player_id, l.n - max(a.n) AS since, count(*) AS played
        FROM played a
        JOIN last l ON l.team_id = a.team_id AND a.n > l.n - {ROSTER_GAMES}
        JOIN latest t ON t.player_id = a.player_id AND t.team_id = a.team_id
        GROUP BY a.team_id, a.player_id, l.n
    """).fetchall()
    out: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for team, player, since, played in rows:
        out[team][player] = (int(since), int(played))
    return out


def expected(minutes: Minutes, game: Game) -> dict[str, float]:
    """Both teams' minutes in a game, for `Prediction.minutes`"""
    return {
        **minutes.get((game.game_id, game.team_id), {}),
        **minutes.get((game.game_id, game.opp_team_id), {}),
    }
