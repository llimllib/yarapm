---
title: The Expected Score
pagetitle: Contents
---

This is a short textbook about one program: [yarapm](https://github.com/llimllib/yarapm),
which predicts the score of NBA games from ratings of the players in them.
Every morning it fits those ratings on every game played before that day
and publishes them, with a prediction for each game on the schedule. It
also keeps a record of every prediction it made before tip-off, so its
record can be checked as the games finish.

The aim is to explain exactly what the program computes and why, to a
reader who has taken calculus and a course in linear algebra and knows
what a basketball game is. Nothing about basketball analytics is assumed.
The mathematics is mostly regression: least squares, a penalty that pulls
estimates toward a guess, and weights that make old games count less. But
regression on basketball data has quirks that make it interesting.
Players who are always on the floor together can't be told apart, a
player new to the league has no data at all, and the league itself changes
under the model from one month to the next.

**How to read it.** Chapters build on each other, and each ends with
exercises. Some of them are queries you can run against the published
data, which is free to read with [DuckDB](https://duckdb.org). Refreshers
on mathematics past a first linear algebra course are in the margin, as
sidenotes, so a reader who knows them can skip them. Boxes headed *In the
code* say how the program does a step, and link to the code that does it.
They're for readers who want to check, change or reuse the program, and
the rest of the text doesn't depend on them.

**Numbers.** The ratings and predictions quoted are a snapshot, from
the published data of November 9, 2026 (model version `e0125ad35d45`),
and the code links point at the commit the text describes. Both move as
the data grows, so a rating looked up today may differ in the last digit.
The model is tuned on the 2021–2024 seasons and was tested once on
2025–2026, which it never saw while it was being tuned. Seasons are
named for the year they end: 2026 is the 2025–26 season.

**The data.** The ratings, predictions and the prediction log are in
[yarapm's DuckDB catalog](https://basketball-data.billmill.org/yarapm/yarapm.duckdb),
described in its [data
dictionary](https://github.com/llimllib/yarapm/blob/main/docs/DATA.md).
The game data comes from [nba_data's
catalog](https://basketball-data.billmill.org/nba/nba.duckdb). To query
both:

```sql
ATTACH 'https://basketball-data.billmill.org/yarapm/yarapm.duckdb' AS yarapm (READ_ONLY);
ATTACH 'https://basketball-data.billmill.org/nba/nba.duckdb' AS nba (READ_ONLY);
```
