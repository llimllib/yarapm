"""
Team ratings from player ratings (#31): a team's offense and defense are
its players' ratings weighted by their share of its minutes, as the player
model predicts with (`PlayerRatings.strength`), for two lineups:

    full   everyone plays: the 240 minutes go to the players with the
           most minutes when playing (`ProjectedMinutes.minutes`), each
           getting theirs, until they're used up (`fill`)
    today  the lineup the predictions use: each player on the model's
           roster at their chance of playing × their minutes when playing,
           scaled to 240

The full-strength lineup is the players who've played for the team this
season and not for another team since (`played_for`). The model's roster
can't be used: it's the players in the team's last 10 games, so early in a
season it has last April's fill-ins, who at full minutes would count as
starters. A player hurt since before the season isn't counted until he
plays. Before a team's first game of the season, its full-strength lineup
is its model roster.

Full strength fills the minutes in order rather than scaling everyone's
down: on a team with injuries, the players filling in have played big
minutes, and scaling the 15 or so who've played to 240 would cut a star's
33 minutes to 21. Today's lineup is scaled, as the predictions are, so a
healthy team's two lineups can differ a little from that alone.

Each lineup's ratings are relative to the average of the teams, so off +
def = net in both and they add up to 0 over the league.
"""

from collections.abc import Callable
from dataclasses import dataclass

from yarapm.harness import History
from yarapm.minutes import CAP, TEAM_MINUTES, ProjectedMinutes, allocate


@dataclass(frozen=True)
class Lineup:
    full: dict[str, float]  # player -> minutes at full strength
    today: dict[str, float]  # player -> expected minutes today
    chances: dict[str, float]  # player -> chance of playing, on the model roster


@dataclass(frozen=True)
class TeamRating:
    full_off: float
    full_def: float
    off: float
    dfn: float


def played_for(history: History) -> dict[str, set[str]]:
    """
    Each team's players this season, before the cutoff: those who've
    played for it, and not for another team since
    """
    rows = (
        history.player_games.filter(f"season = {history.season} AND min > 0")
        .aggregate("player_id, arg_max(team_id, game_date)", "player_id")
        .fetchall()
    )
    teams: dict[str, set[str]] = {}
    for player, team in rows:
        teams.setdefault(team, set()).add(player)
    return teams


def fill(
    minutes: dict[str, float], total: float = TEAM_MINUTES, cap: float = CAP
) -> dict[str, float]:
    """
    `total` minutes given out in order of `minutes`, most first, each player
    getting theirs (up to `cap`) until they run out, and the rest none. If
    they add up to less than `total`, they're scaled up instead (`allocate`)
    """
    capped = {p: min(max(m, 0.0), cap) for p, m in minutes.items()}
    if sum(capped.values()) <= total:
        return allocate(minutes, total, cap)
    out = {}
    left = total
    for p in sorted(capped, key=lambda p: (-capped[p], p)):
        out[p] = min(capped[p], left)
        left -= out[p]
    return out


def lineups(projection: ProjectedMinutes, history: History) -> dict[str, Lineup]:
    """
    Each team's lineups, from a projection fit with unknown availability on
    `history`
    """
    played = played_for(history)
    out = {}
    for team in sorted(set(projection.rosters) | set(played)):
        chances = projection.playing_chances(team)
        full = played.get(team) or set(chances)
        out[team] = Lineup(
            fill({p: projection.minutes(p) for p in full}),
            allocate({p: c * projection.minutes(p) for p, c in chances.items()}),
            chances,
        )
    return out


def team_ratings(
    lineups: dict[str, Lineup],
    strength: Callable[[dict[str, float]], tuple[float, float]],
) -> tuple[dict[str, TeamRating], TeamRating]:
    """
    Each team's ratings relative to the average team, from `strength`
    (minutes -> offense, defense), and the average team's raw ones
    """
    raw = {
        team: TeamRating(*strength(lineup.full), *strength(lineup.today))
        for team, lineup in lineups.items()
    }
    n = len(raw)
    mean = TeamRating(
        sum(r.full_off for r in raw.values()) / n,
        sum(r.full_def for r in raw.values()) / n,
        sum(r.off for r in raw.values()) / n,
        sum(r.dfn for r in raw.values()) / n,
    )
    centered = {
        team: TeamRating(
            r.full_off - mean.full_off,
            r.full_def - mean.full_def,
            r.off - mean.off,
            r.dfn - mean.dfn,
        )
        for team, r in raw.items()
    }
    return centered, mean
