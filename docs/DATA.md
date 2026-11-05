# yarapm's data

What yarapm publishes: player and team ratings, predictions for the games
not yet played, projected minutes, and a log of every prediction scored as the
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

**History:** the daily datasets (`ratings_daily`, `team_ratings_daily`,
`team_players_daily`) go back to 2018 (2017-18), so a player's or team's
ratings can be followed across seasons. Every row is the current model's
view of its day, fit only on the games before it, as if it had been run
that day: when the model changes, every season is rebuilt with it, and a
season's files never mix model versions (`model_version`). The first
season rated is the first with two seasons of play-by-play before it,
since the box-score prior learns from earlier seasons; nba_data's
play-by-play starts in 2016 and is being backfilled toward 2010, and the
history will start earlier as it lands. The first seasons rest on less
data than later ones: their prior learned from fewer seasons, and their
ratings from fewer games before them.

## Datasets

### `ratings_daily`

Each player's ratings as of each game day of the season: for the current
season, also the latest build's day, and for a finished one, the day after
its last game (its final ratings, with the Finals). Keyed by `season`,
`as_of_date` and `player_id`. A day's ratings use the games before it, so
they're what a prediction on that day would use. Players are those on a
team's roster as of the day (see `minutes`) or with possessions this
season. There are no rows for the offseason: a player's ratings don't move
without games, until a new season's prior.

| Column | Type | Description |
|--------|------|-------------|
| season, as_of_date, player_id | | Key |
| team_id | VARCHAR | The team whose roster they're on, as of the day: NULL if none. A player who's changed teams is on their new team's once they've played for it |
| o_rating, d_rating, rating | DOUBLE | Offensive, defensive and total rating: points per 100 possessions above average that the player adds on the floor, positive is good |
| o_prior, d_prior | DOUBLE | The box-score prior the ratings are pulled toward. A player with no possessions in the fit is rated at their prior |
| possessions | INTEGER | Offensive possessions on the floor this season before the day |
| commit, settings | VARCHAR | The yarapm commit and model settings (JSON) that rated the day |
| model_version | VARCHAR | A hash of the code and locked dependencies that rated the day: the same for every row of a season, and of every daily dataset's season |

A build keeps the season's days if its files were built by its model
version, rating the missing ones and its own day again, and rates every
day if not; a backfill does the same for the finished seasons. The latest
day changes with every build until the season moves on.

### `team_ratings_daily`

Each team's rating as of the same days as `ratings_daily`, at full
strength and with that day's expected lineup. Keyed by
`season`, `as_of_date` and `team_id`. A team's rating is its players'
ratings (`ratings_daily`) weighted by their share of its minutes, as the
predictions weigh them; `team_players_daily` has the lineups.

- **Full strength:** everyone's available. The players are those who've
  played for the team this season and not for another team since (before
  its first game, its roster as in `minutes`). The 240 minutes go to the
  ones with the most minutes when they play, each getting theirs (up to
  42), until they're used up; the rest get none. A player hurt since before
  the season isn't counted until he plays.
- **Today:** the lineup the predictions use: each player's chance of
  playing × their minutes when playing, scaled to 240 (see `minutes`).

Each is relative to the average of the 30 teams that day. The two lineups
give out minutes differently (in order, and scaled), so even a healthy
team's numbers differ a little; compare a team's to the other teams' in the
same column rather than its full strength to its today.

| Column | Type | Description |
|--------|------|-------------|
| season, as_of_date, team_id | | Key |
| full_off, full_def, full_net | DOUBLE | Full strength: points per 100 possessions better than the average team on offense, defense and overall, positive is good. `full_off + full_def = full_net`, within rounding |
| off, def, net | DOUBLE | The same with today's lineup |
| league_average | DOUBLE | Points per 100 possessions the average team scores against the average team, with today's lineups: a team's offense today is `league_average + off` points per 100 against an average defense, on a neutral court |
| roster | VARCHAR | Where the full-strength players come from: `played`, this season's games. Will be `rosters_daily` once nba_data publishes rosters |
| commit, settings, model_version | VARCHAR | As in `ratings_daily` |

```sql
-- teams at full strength today, and how far each is from it
SELECT t.nba_abbrev, r.full_net, r.net, r.full_off, r.full_def
FROM yarapm.team_ratings_daily r
JOIN nba.team_seasons t ON t.season = r.season AND t.team_id = r.team_id
WHERE r.season = 2027
    AND r.as_of_date = (SELECT max(as_of_date) FROM yarapm.team_ratings_daily WHERE season = 2027)
ORDER BY r.full_net DESC;
```

### `team_players_daily`

The players in each team's lineups in `team_ratings_daily`, keyed by
`season`, `as_of_date`, `team_id` and `player_id`: everyone in either one.
Each team's `full_min` and `min` add up to 240.

| Column | Type | Description |
|--------|------|-------------|
| season, as_of_date, team_id, player_id | | Key |
| full_min | DOUBLE | Minutes at full strength. 0 for a player only in today's lineup (one from late last season with a small chance of playing, say) or left out at full strength by the players ahead of him |
| min, play_prob | DOUBLE | Today's expected minutes and chance of playing, as in `minutes`. 0 for a player who isn't on the team's roster as `minutes` has it |
| o_rating, d_rating | DOUBLE | As in `ratings_daily` that day |
| model_version | VARCHAR | As in `ratings_daily` |

```sql
-- who a team is missing today: big minutes at full strength, few today
SELECT p.name, m.full_min, m.min, m.play_prob, m.o_rating + m.d_rating AS rating
FROM yarapm.team_players_daily m JOIN nba.players p USING (player_id)
WHERE m.season = 2027 AND m.as_of_date = '2026-11-04' AND m.team_id = '1610612754'
ORDER BY m.full_min - m.min DESC;
```

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
| model_version | VARCHAR | The model version that last wrote it (see `ratings_daily`) |

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
