# AGENTS.md - Project Guidelines for AI Assistants

## Project Overview
yarapm predicts the expected score of NBA games from player ratings (RAPM with a box-score prior, weighted by projected minutes). Every model change is judged by the walk-forward harness (#3) on the tuning seasons (2021-2024); 2025-2026 are held out until #12.

Commands (tools come from `mise.toml`):
- `mise run test` / `mise run lint` (ruff format, ruff check, ty). CI runs both; keep them passing
- `uv run python -m yarapm.segments [--season 2026 ...]` builds `out/segments/` (every season with play-by-play by default, ~20s), printing how many games each season keeps and why the rest are dropped
- `uv run python -m yarapm.harness --model home [--cadence weekly|daily] [--season 2021 ...] [--set name=value ...]` walks a model forward through the tuning seasons, writes `out/runs/{predictions,metrics}/run=<id>/`, and prints the metrics by phase. Models are registered in `yarapm.harness.models()`; a model only sees data through `History`, filtered to games before the fit date. Compare runs with `FROM read_parquet('out/runs/metrics/*/*.parquet')`

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

## GitHub issues
Plans and progress live in GitHub issues on `llimllib/yarapm`, not in docs. Use the `gh` CLI.

- **Tracking issue #24** has the design summary and a task list of every work issue by milestone. Keep it current when issues are added or closed
- **Milestones**, in build order (use the exact title with `gh issue create --milestone`): `1. Lineup segments`, `2. Walk-forward harness & team baseline`, `3. Player ratings`, `4. Projected minutes & final evaluation`
- **Experiments:** ideas to try later are issues labeled `experiment`, with no milestone. Each says what to try and what result would show it worked. When something comes up that's out of scope, open one rather than doing it. Report results in the issue, citing the harness run's settings, and judge them on the tuning seasons only
- Close work issues from the commit that finishes them (`Closes #N`), and check that CI passes
- Creating issues and pushing are visible to others: confirm with the user first unless they've asked for it

In the sandbox, `.git/config` isn't writable: pushes work, but `git push -u` can't record the upstream branch.
