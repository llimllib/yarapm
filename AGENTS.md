# AGENTS.md - Project Guidelines for AI Assistants

## Project Overview
yarapm predicts the expected score of NBA games from player ratings (RAPM with a box-score prior, weighted by projected minutes). Every model change is judged by the walk-forward harness (#3) on the tuning seasons (2021-2024); 2025-2026 are held out until #12. It also publishes every player's and team's ratings as of each game day, across every season we can rate, so their progression can be charted (DARKO-style); this repo makes the data, not the charts.

Commands (tools come from `mise.toml`):
- `mise run test` / `mise run lint` (ruff format, ruff check, ty). CI runs both; keep them passing
- `uv run python -m yarapm.segments [--season 2026 ...]` builds `out/segments/` (every season with play-by-play by default, ~20s), printing how many games each season keeps and why the rest are dropped
- `uv run python -m yarapm.harness --model home|team|rapm|openrapm [--cadence weekly|daily] [--season 2021 ...] [--set name=value ...]` walks a model forward through the tuning seasons, writes `out/runs/{predictions,metrics}/run=<id>/`, and prints the metrics by phase. Models are registered in `yarapm.harness.models()`; a model only sees data through `History`, filtered to games before the fit date. Compare runs with `FROM read_parquet('out/runs/metrics/*/*.parquet', union_by_name = true)` (older runs may lack newer metrics)
- `uv run python -m yarapm.predict [--date 2026-11-05] [--days 7] [--model rapm|team] [--no-segments]` rebuilds every season's segments (the prior needs the earlier ones), fits a live model on the games before the date and predicts the games in `nba.schedule` from it
- `mise run book` builds the book (`book/`, a textbook on what yarapm does, published to https://llimllib.github.io/yarapm/ by `.github/workflows/book.yml`) into `out/book/`; `mise run book-figures [name ...]` redraws its SVGs from the data and prints every number the text quotes. Its numbers are a snapshot: when the model or the data changes, rerun the figures and check the text against what they print
- `uv run python -m yarapm.publish build [--date ...] [--no-segments]` writes the published datasets into `out/publish/`, a mirror of the bucket's `yarapm/`; `... publish catalog [--url https://basketball-data.billmill.org]` writes the catalog (without `--url`, `out/yarapm.local.duckdb` over the local files); `publish backfill [--season ...] [--force]` writes the daily datasets for the finished seasons (#34), skipping those already built by this model version (refitting all of them takes 45-75 minutes on GitHub Actions); `publish current` exits 0 if nba_data hasn't updated since the mirror's last build and the model version hasn't changed; `publish verify` reads every dataset in the published catalog. `docs/DATA.md` documents the datasets: keep it current
- `.github/workflows/publish.yml` runs the publish job (#28) when nba_data's update dispatches `nba-data-updated`, daily as a fallback (skipped with no new data), or by hand. It syncs the bucket's `yarapm/` into `out/publish/` first, builds, uploads, purges the CDN, writes the catalog and verifies it. After a build, its `backfill` job (#35) does the same with `publish backfill`, which only fits anything when the model version has changed or a season has finished; dispatch it by hand with `force` to refit every season. The backfill job times out at 120 minutes (`timeout-minutes`), and uploads nothing if it does. It needs the secrets nba_data uses: `DO_SPACES_ACCESS_KEY`, `DO_SPACES_SECRET`, `DO_CDN_API_KEY`, `DO_CDN_ENDPOINT_ID`. Don't upload a local `out/publish/` wholesale: its logs (`prediction_log`, `minutes_log`) aren't the live record. If a backfill can't finish in CI, it can be run locally instead: disable the workflow, check out the deployed commit with no local changes (the model version hashes the code), sync the bucket into `out/publish/`, run `publish backfill`, upload only the daily datasets' finished seasons, then do the workflow's catalog, purge and verify steps and re-enable it

`out/` is local and not committed. In a fresh checkout, run `yarapm.segments` before the harness: models only see segments if `out/segments/` exists. Code that isn't run with `-m` from the repo root (a scratch script in /tmp, say) needs `PYTHONPATH=.`, since the package isn't installed.

## Code
- `yarapm/catalog.py`: `connect()`, and `write_season()`/`read()` for `out/`
- `yarapm/games.py`: which games we model (regular season, play-in, playoffs, Cup final; the 2020 bubble is neutral) and `load_results()`, one row per game from the home side
- `yarapm/segments.py`: the lineup segments (#2); its docstring has the rules for placing plays and crediting possessions, and the columns
- `yarapm/harness.py`: `Game`, `Prediction`, `History`, the `Model` protocol, `run()` and the metrics. Phases: `early`, `mid`, `post_deadline`, `playoffs`. Every run also scores the closing betting line (`nba.betting_lines`, from 2012) on the same games as a benchmark (`line_margin_rmse`, `line_total_rmse`, `line_log_loss`); models never see it
- `yarapm/fit.py`: `decay()` (half-life weights) and `weighted_ridge()`, shared by the models
- `yarapm/tempo.py`, `yarapm/baselines.py`: adjusted tempo, and the `home` and `team` models
- `yarapm/rapm.py`: the sparse ridge RAPM solver (#6), pulling each player toward a prior
- `yarapm/prior.py`: the box-score prior (#7) and OpenRAPM as a prior (#9)
- `yarapm/players.py`: the `rapm` model (#8), ratings × minutes + tempo + home court, and `openrapm` (#9), OpenRAPM's ratings through the same step
- `yarapm/minutes.py`: minutes for those models, `--set minutes=actual|known|unknown`. `actual` is the minutes played (`History.actual_minutes`, a leak that isolates rating error); `known` projects minutes given who played (`History.active`, a leak standing in for injury reports, #10), the default; `unknown` projects who plays too (#11). Projected minutes are scored as `minutes_mae`
- `yarapm/predict.py`: predictions for games not yet played (#26), from `nba.schedule` (`yarapm.games.schedule_query`). The live models are `rapm` with unknown availability and `team`; win probability uses a fixed spread, each model's margin RMSE on the tuning seasons (`SIGMA`). A fresh fit for a day matches the harness's daily prediction for it
- `yarapm/strength.py`: team ratings at full strength and today (#31), from the player model's ratings and minutes; `publish` writes them as `team_ratings_daily` and `team_players_daily`
- `yarapm/publish.py`: the published datasets and their catalog (#27), checked by `validate()` before anything is written. `prediction_log` (#29) is append-only: never rewrite or backfill it, it's the live record. The daily datasets are the opposite (#33): the current model's view of every day, rebuilt whenever `model_version()` (a hash of `yarapm/*.py`, `uv.lock` and the first season of play-by-play, #37) changes, so a season's files never mix models. `live_metrics` scores it with `harness.score` and `harness.phases()`. Add each season's trade deadline to `harness.TRADE_DEADLINES` once it's known, or the live record has no `post_deadline` phase

To tune, call `harness.load(con)` once and loop `harness.run(con, Model(...), list(harness.TUNING_SEASONS), "weekly")` in one process, comparing on the `all` phase; confirm the chosen settings with a `daily` run. Record the grid and results in the issue, and put the chosen defaults in the model with a note on how they were picked.

The segments grow as nba_data backfills play-by-play (2012 on, so far), and the box-score prior is fit to every earlier season with segments, so results move when seasons are added. The results below used segments from 2019 on. Judge an experiment against a baseline run on the same segments, not against these tables (#32).

## Results so far
Daily refits, 2021-2024 (the scoreboard is kept in #24):

| model | margin RMSE | log loss | total RMSE | poss RMSE | minutes MAE |
|---|---|---|---|---|---|
| `home` (home court only) | 14.98 | 0.687 | 19.97 | | |
| `team` (KenPom-style) | 13.93 | 0.636 | 18.73 | 4.57 | |
| `openrapm` (last season's, unknown availability) | 13.99 | 0.635 | 18.62 | 4.57 | 6.24 |
| `openrapm` (last season's, known availability) | 13.85 | 0.625 | 18.57 | 4.57 | 4.73 |
| `rapm`, unknown availability | 13.71 | 0.624 | 18.43 | 4.57 | 6.24 |
| `rapm`, plain (`prior_weight=0`, actual minutes) | 13.71 | 0.620 | 18.40 | 4.57 | |
| `rapm` with the box-score prior, actual minutes | 13.62 | 0.613 | 18.40 | 4.57 | |
| `rapm` with the box-score prior, known availability | 13.58 | 0.613 | 18.35 | 4.57 | 4.73 |
| `openrapm` (this season's: leaks, a ceiling; actual minutes) | 13.06 | 0.591 | 17.96 | 4.57 | |
| closing line (5,132 games: knows injury news) | 13.40 | 0.609 | 18.00 | | |

The holdout, 2025-2026 (2,643 games), was scored once with every setting fixed (#12). Don't tune on it:

| model | margin RMSE | log loss | total RMSE | minutes MAE |
|---|---|---|---|---|
| `home` | 16.23 | 0.688 | 19.97 | |
| `openrapm`, last season's, unknown availability | 14.78 | 0.617 | 18.48 | 6.43 |
| `openrapm`, last season's, known availability | 14.62 | 0.606 | 18.34 | 4.72 |
| `team` | 14.48 | 0.608 | 18.48 | |
| `rapm`, unknown availability | 14.42 | 0.606 | 18.36 | 6.43 |
| `rapm`, known availability | 14.22 | 0.593 | 18.23 | 4.72 |
| `rapm`, actual minutes | 14.22 | 0.592 | 18.28 | |
| `openrapm`, this season's (a ceiling) | 13.69 | 0.570 | 17.82 | 4.72 |

## Data
Source data is nba_data's catalog (`https://basketball-data.billmill.org/nba/nba.duckdb`); its data dictionary is `docs/DATA_DICTIONARY.md` in https://github.com/llimllib/nba_data (often checked out at `../nba_data`). `yarapm.catalog.connect()` attaches it as `nba`, so source tables are always `nba.<table>`. Filter on `season` whenever you can: other filters read every season's file. What we build goes under `out/` (gitignored), one parquet file per season, via `write_season()` and `read()`.

### DuckDB behind a proxy
DuckDB reads `http_proxy`/`HTTPS_PROXY` itself but can't parse credentials in them (`Failed to parse http_proxy ... into a host and port`), which is the case in the agent sandbox. `connect()` handles it: it splits `HTTPS_PROXY` into host, username and password and sets them explicitly. Use `connect()` from Python rather than opening your own connection.

For the DuckDB CLI, unset the proxy variables and set the proxy in SQL:

```sh
P=$(python3 -c "import os,urllib.parse as u;x=u.urlparse(os.environ['HTTPS_PROXY']);print(f'{x.hostname}:{x.port}|{u.unquote(x.username)}|{u.unquote(x.password)}')")
IFS='|' read -r HOST USER PASS <<< "$P"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY duckdb -c "
  SET http_proxy='$HOST'; SET http_proxy_username='$USER'; SET http_proxy_password='$PASS';
  ATTACH 'https://basketball-data.billmill.org/nba/nba.duckdb' AS nba (READ_ONLY);
  SELECT count(*) FROM nba.games WHERE season = 2026"
```

The CLI prints markdown tables when `AI_AGENT` is set; pass `-no-agent` for its normal output. Queries outside the sandbox's domain allowlist fail with `403 blocked-by-allowlist`; ask the user to allow the domain.

### DuckDB 2.0
The Python API is stricter than 1.x:
- relations have no `fetchone()`: use `con.execute(...).fetchone()` or `rel.fetchall()[0]`
- a result from `con.execute()` must be fetched completely before the connection runs anything else (`ResourceInUseException`)
- `duckdb.typing` is gone: compare `str(rel.types[i])`

## GitHub issues
Plans and progress live in GitHub issues on `llimllib/yarapm`, not in docs. Use the `gh` CLI.

- **Tracking issue #24** has the design summary and a task list of every work issue by milestone. Keep it current when issues are added or closed
- **Milestones**, in build order (use the exact title with `gh issue create --milestone`): `1. Lineup segments`, `2. Walk-forward harness & team baseline`, `3. Player ratings`, `4. Projected minutes & final evaluation`, `5. Publishing`, `6. Rating history`
- **Experiments:** ideas to try later are issues labeled `experiment`, with no milestone. Each says what to try and what result would show it worked. When something comes up that's out of scope, open one rather than doing it. Report results in the issue, citing the harness run's settings, and judge them on the tuning seasons only
- Close work issues from the commit that finishes them (`Closes #N`), and check that CI passes
- Creating issues and pushing are visible to others: confirm with the user first unless they've asked for it

In the sandbox, `.git/config` isn't writable: pushes work, but `git push -u` can't record the upstream branch.
