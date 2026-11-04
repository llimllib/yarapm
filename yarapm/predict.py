"""
Predict games that haven't been played (#26).

    uv run python -m yarapm.predict [--date 2026-11-05] [--days 7]
        [--model rapm|team] [--no-segments]

Rebuilds the season's segments (yarapm.segments), fits the model on every
game before `--date`, as the harness does for a day, and predicts the games
in nba.schedule from `--date` through `--days` days: each team's expected
points, possessions, win probability and, for `rapm`, each player's
projected minutes. Games whose teams aren't set yet (Cup knockouts, playoff
series) are skipped. `--date` defaults to today in US Eastern time, the
schedule's.

The live models are LIVE_MODELS: `rapm` with unknown availability (#11),
since who plays in a future game isn't known, and the `team` baseline to
compare it to (#29), each with its default settings otherwise.

Win probability is P(margin > 0) for a normal centered on the predicted
margin, as the harness scores it, but with a fixed spread: a live
prediction can't know its own margin RMSE, so each model's is its daily
run's on the tuning seasons (SIGMA).
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
from scipy.special import ndtr

from yarapm import segments
from yarapm.baselines import TeamRatings
from yarapm.catalog import OUT, connect
from yarapm.games import schedule_query
from yarapm.harness import Game, History, Model, Prediction, load
from yarapm.players import PlayerRatings

EASTERN = ZoneInfo("America/New_York")

# margin RMSE of each model's daily run on 2021-2024: rapm-93ea3652 (minutes
# unknown) and team-2ac45b4a
SIGMA = {"rapm": 13.71, "team": 13.93}


def live_models() -> dict[str, Model]:
    return {"rapm": PlayerRatings(minutes="unknown"), "team": TeamRatings()}


@dataclass(frozen=True)
class Forecast:
    game: Game
    start_time: datetime | None
    prediction: Prediction
    win_prob: float  # `game.team_id`'s


def scheduled(con: duckdb.DuckDBPyConnection, start: date, days: int) -> list[tuple]:
    """(Game, start_time) for the scheduled games from `start` through `days` days"""
    rows = con.execute(
        f"""{schedule_query("game_date >= ? AND game_date < ?")}
        ORDER BY game_date, start_time, game_id""",
        [start, start + timedelta(days=days)],
    ).fetchall()
    return [(Game(*r[:7]), r[7] and r[7].replace(tzinfo=UTC)) for r in rows]


def forecast(
    con: duckdb.DuckDBPyConnection,
    model: Model,
    games: list[tuple[Game, datetime | None]],
    cutoff: date,
) -> list[Forecast]:
    """
    Fit `model` on the games before `cutoff`, in the games' season, and
    predict them. Needs the tables `yarapm.harness.load` makes
    """
    if not games:
        return []
    seasons = {g.season for g, _ in games}
    if len(seasons) > 1:
        raise ValueError(f"games from more than one season: {sorted(seasons)}")
    model.fit(History(con, cutoff, seasons.pop()))
    predicted = [Prediction(*p) for p in model.predict([g for g, _ in games])]
    if len(predicted) != len(games):
        raise ValueError(
            f"{model.name} predicted {len(predicted)} of {len(games)} games"
        )
    sigma = SIGMA[model.name]
    return [
        Forecast(g, start, p, float(ndtr((p.pts - p.opp_pts) / sigma)))
        for (g, start), p in zip(games, predicted)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--model", choices=sorted(SIGMA), default="rapm")
    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="use the segments already in --out instead of rebuilding them",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    day = args.date or datetime.now(EASTERN).date()

    con = connect()
    games = scheduled(con, day, args.days)
    if not games:
        print(f"no games scheduled from {day} through {args.days} days")
        return
    if not args.no_segments:
        segments.main(["--season", str(games[0][0].season), "--out", str(args.out)])
    load(con, args.out)
    forecasts = forecast(con, live_models()[args.model], games, day)
    print_forecasts(con, forecasts)


def print_forecasts(con: duckdb.DuckDBPyConnection, forecasts: list[Forecast]) -> None:
    abbrevs = dict(
        con.execute(
            "SELECT team_id, nba_abbrev FROM nba.team_seasons WHERE season = ?",
            [forecasts[0].game.season],
        ).fetchall()
    )
    for f in forecasts:
        g, p = f.game, f.prediction
        team, opp = (
            abbrevs.get(g.team_id, g.team_id),
            abbrevs.get(g.opp_team_id, g.opp_team_id),
        )
        matchup = f"{opp} @ {team}" if g.home else f"{team} vs {opp}"
        tip = f.start_time.astimezone(EASTERN).strftime("%H:%M") if f.start_time else ""
        poss = f"{p.poss:5.1f}" if p.poss is not None else "     "
        print(
            f"{g.game_date} {tip:>5} {matchup:<11} "
            f"{team} {p.pts:5.1f} {opp} {p.opp_pts:5.1f}  "
            f"poss {poss}  P({team}) {f.win_prob:.3f}"
        )


if __name__ == "__main__":
    main()
