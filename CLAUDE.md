# CLAUDE.md - Project Guidelines for AI Assistants

## Project Overview
yarapm predicts the expected score of NBA games from player ratings (RAPM with a box-score prior, weighted by projected minutes). Plans and progress live in GitHub issues, not docs: the tracking issue is #24, with work in milestones and ideas to try labeled `experiment`. Every model change is judged by the walk-forward harness (#3) on the tuning seasons (2021-2024); 2025-2026 are held out until #12.

Source data is nba_data's catalog (`https://basketball-data.billmill.org/nba/nba.duckdb`); its data dictionary is `docs/DATA_DICTIONARY.md` in https://github.com/llimllib/nba_data. `yarapm.catalog.connect()` attaches it as `nba`, so source tables are always `nba.<table>`. Filter on `season` whenever you can: other filters read every season's file. What we build goes under `out/` (gitignored), one parquet file per season, via `write_season()` and `read()`.

Commands (tools come from `mise.toml`):
- `mise run test` / `mise run lint` (ruff format, ruff check, ty). CI runs both; keep them passing
