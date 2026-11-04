# yarapm's data

What yarapm publishes: player ratings, predictions for the games not yet
played, projected minutes, and a log of every prediction scored as the
games finish, rebuilt every time
[nba_data](https://github.com/llimllib/nba_data) updates. The files are
under the `yarapm/` prefix of the `basketball-data` bucket, laid out like
nba_data's, one parquet file per season:

    https://basketball-data.billmill.org/yarapm/<dataset>/season=<season>/data.parquet

They're in their own catalog, apart from nba_data's, and join nba_data's
tables on its ids:

```sql
ATTACH 'https://basketball-data.billmill.org/yarapm/yarapm.duckdb' AS yarapm;
ATTACH 'https://basketball-data.billmill.org/nba/nba.duckdb' AS nba;

-- the best players as of today
SELECT p.name, r.o_rating, r.d_rating, r.rating, r.possessions
FROM yarapm.ratings_daily r JOIN nba.players p USING (player_id)
WHERE r.season = 2027
    AND r.as_of_date = (SELECT max(as_of_date) FROM yarapm.ratings_daily WHERE season = 2027)
ORDER BY r.rating DESC LIMIT 10;
```

Filter on `season` whenever you can: as with nba_data's views, the other
filters read every season's file.

Conventions are nba_data's: ids are strings, `season` is the year the season
ends (2027 is 2026-27), dates are US Eastern and times are `TIMESTAMPTZ`.
Values are rounded: ratings, points and minutes to two decimals,
probabilities to four.

**The model:** ridge RAPM on lineup segments, each player pulled toward a
box-score prior, with older games weighted less; a team's expected score is
its players' ratings weighted by projected minutes, times adjusted tempo,
with home court. Who plays isn't known ahead of time, so minutes are
projected with each player's chance of playing. The
[tracking issue](https://github.com/llimllib/yarapm/issues/24) has the
design and how it's been tested. Every dataset is built with the settings
and commit in `metadata`.

**Updates:** a build predicts from the day it runs (US Eastern), with every
game before that day. Since nba_data updates every four hours, the files
can change during the day as a night's games come in.

## Datasets

### `ratings_daily`

Each player's ratings as of each game day of the season, and the latest
build's day. Keyed by `season`, `as_of_date` and `player_id`. A day's
ratings use the games before it, so they're what a prediction on that day
used. Players are those on a team's roster as of the day (see `minutes`)
or with possessions this season.

| Column | Type | Description |
|--------|------|-------------|
| season, as_of_date, player_id | | Key |
| team_id | VARCHAR | The team whose roster they're on, as of the day: NULL if none. A player who's changed teams is on their new team's once they've played for it |
| o_rating, d_rating, rating | DOUBLE | Offensive, defensive and total rating: points per 100 possessions above average that the player adds on the floor, positive is good |
| o_prior, d_prior | DOUBLE | The box-score prior the ratings are pulled toward. A player with no possessions in the fit is rated at their prior |
| possessions | INTEGER | Offensive possessions on the floor this season before the day |
| commit, settings | VARCHAR | The yarapm commit and model settings (JSON) that rated the day |

A day is rated once, by the first build on or after it that's missing it;
the latest day is rated again by every build. Rows from an older model
stay as they were, with its commit and settings.

### `predictions`

The latest prediction for each game of the season from the build's day on,
replaced by every build. One row per game, from the home team's side as
nba_data's `schedule` lists it (at a neutral site too). Games whose teams
aren't set (Cup knockouts, unset playoff series) aren't predicted until
they are.

| Column | Type | Description |
|--------|------|-------------|
| season, game_id | | Key |
| game_date, start_time, game_type | | As in `schedule` |
| home_team_id, away_team_id, neutral_site | | As in `schedule` |
| home_pts, away_pts | DOUBLE | Each team's expected points |
| poss | DOUBLE | Expected possessions per team |
| home_win_prob | DOUBLE | The chance the home team wins: P(margin > 0) for a normal centered on the expected margin, with a standard deviation of 13.71 points, the model's error on 2021-2024 |
| as_of_date | DATE | The build's day: the prediction used the games before it |

```sql
-- the next week's predictions
SELECT p.start_time, a.nba_abbrev AS away, h.nba_abbrev AS home,
    p.away_pts, p.home_pts, p.home_win_prob
FROM yarapm.predictions p
JOIN nba.team_seasons h ON h.season = p.season AND h.team_id = p.home_team_id
JOIN nba.team_seasons a ON a.season = p.season AND a.team_id = p.away_team_id
WHERE p.season = 2027 AND p.game_date < current_date + 7
ORDER BY p.start_time;
```

### `minutes`

Each player's projected minutes in the games in `predictions`, which the
prediction weights their ratings by. Keyed by `season`, `game_id`,
`team_id` and `player_id`. A team's minutes add up to 240.

A team's players are its roster: those who played one of its last 10
games and haven't played for another team since. Each one's minutes are
their chance of playing × their minutes when they play (a weighted average
of their recent games), scaled to 240, with no one above 42.

| Column | Type | Description |
|--------|------|-------------|
| season, game_id, team_id, player_id | | Key |
| min | DOUBLE | Expected minutes, counting the chance they don't play |
| play_prob | DOUBLE | The chance they play, from how many of the team's last 10 games they played and how long ago their last one was |
| as_of_date | DATE | The build's day |

### `prediction_log`

Every prediction made for a game before it started, from the published
model (`rapm`) and, to compare it to, a team-rating baseline (`team`,
KenPom-style). Each build logs its predictions for the games in the next 7
days that haven't started, but only where a game's prediction has changed
since its last row (so usually once a day, when the games before the day
are added). Rows are never changed or removed: they're what was said
before the game. Keyed by `season`, `model`, `game_id` and `created`.

| Column | Type | Description |
|--------|------|-------------|
| season, model, game_id, created | | Key. `created` (TIMESTAMPTZ) is when the build ran |
| commit, settings | VARCHAR | The yarapm commit and the model's settings (JSON) |
| game_date ... as_of_date | | As in `predictions` |

`team` predictions use their own spread for `home_win_prob`, 13.93
points, its error on 2021-2024.

```sql
-- what was predicted for each finished game: the last row before tip-off
SELECT * FROM yarapm.prediction_log
WHERE season = 2027 AND model = 'rapm' AND created < start_time
QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY created DESC) = 1;
```

### `minutes_log`

The projected minutes behind each `rapm` prediction in `prediction_log`,
keyed by `season`, `model`, `game_id`, `created` and `player_id`, with
`team_id`, `min` and `play_prob` as in `minutes`.

### `live_metrics`

The log scored, rebuilt by every build: for each finished game, the last
prediction each model made before tip-off, against the result. One row per
series and phase. A series is a model with one commit and settings, so a
change to the model starts a new one; each is reported separately. Missing
until a logged game has finished.

| Column | Type | Description |
|--------|------|-------------|
| season, model, commit, settings, phase | | Key. `phase` is `all`, or `early` (either team in its first 20 games), `mid`, `post_deadline` or `playoffs` (with the play-in) |
| first_game_date, last_game_date | DATE | The series' scored games |
| games | BIGINT | Games scored |
| margin_rmse, margin_mae, margin_bias | DOUBLE | Error in the home team's margin, predicted − actual |
| total_rmse, total_mae, total_bias | DOUBLE | Error in the total |
| score_rmse | DOUBLE | Error in each team's points |
| poss_rmse, poss_bias | DOUBLE | Error in possessions per team |
| sigma | DOUBLE | The series' margin RMSE over all its games |
| log_loss, brier | DOUBLE | The win probabilities scored with `sigma` as the spread, as yarapm's test harness scores them, so they compare with its results |
| published_log_loss, published_brier | DOUBLE | The win probabilities as published, with the fixed spread |
| minutes_mae | DOUBLE | `rapm`'s mean absolute error in projected minutes per player, over everyone projected or who played, against minutes played scaled to 240 a team. NULL for `team` |

The `post_deadline` phase needs the season's trade deadline in
`yarapm.harness.TRADE_DEADLINES`; until it's added, those games count as
`mid`.

### `metadata`

A table in the catalog, one row per dataset, also published as
`yarapm/metadata.json` for clients without DuckDB.

| Column | Type | Description |
|--------|------|-------------|
| dataset | VARCHAR | Key |
| first_season, last_season, seasons | INTEGER | The seasons it has; `seasons` is a count |
| updated | TIMESTAMPTZ | When a build last wrote it |
| commit | VARCHAR | The yarapm commit that wrote it, with `-dirty` if it had uncommitted changes |
| model, settings | VARCHAR | The model and its settings (JSON). The logs and `live_metrics` have both models, so no settings: they're in each row |

`metadata.json` has the table's rows keyed by dataset, with times in UTC
ISO 8601, and three more keys: `as_of_date`, the latest build's day;
`source_updated`, nba_data's latest update as the build read it; and
`updated`, when the build ran.

```json
{
  "datasets": {
    "predictions": {"first_season": 2027, "last_season": 2027, "seasons": 1, "updated": "2026-11-04T12:22:47Z", "commit": "...", "model": "rapm", "settings": {"ridge": 3000, "...": "..."}}
  },
  "as_of_date": "2026-11-04",
  "source_updated": "2026-11-04T00:54:13Z",
  "updated": "2026-11-04T12:22:47Z"
}
```
