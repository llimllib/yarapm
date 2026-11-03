# yarapm

Expected scores for NBA games, built from player ratings: RAPM with a
box-score prior, weighted by projected minutes. The plan and its progress
are in the [tracking issue](https://github.com/llimllib/yarapm/issues/24).

The data comes from [nba_data](https://github.com/llimllib/nba_data)'s
catalog, `https://basketball-data.billmill.org/nba/nba.duckdb`. What we build
from it is cached as parquet under `out/`.

## Development

Tools come from `mise.toml`:

- `mise run test`
- `mise run lint` (ruff format, ruff check, ty)
