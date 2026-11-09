---
title: Ratings Through Time
label: ch:history
---

Every morning the model is fit on every game before that day. Kept, the
fits make a record: each player's rating as of each day, back as far as
the data allows. This chapter is about that record, what it is and
isn't, and the engineering that keeps it consistent.

## What we would have said

::: {.definition #def:daily title="Daily ratings"}
A player's rating *as of day $t$* is their rating in the model fit on
the games before $t$. The published dataset `ratings_daily` has one row
per player and game day of each season, and one for the day after each
season's last game, with the offensive and defensive ratings, the
box-score prior and the possessions played so far that season. Each
day's ratings are relative to that day's average player: weighted by
their expected minutes in the day's lineups, they average 0.
:::

Because each day's fit only sees the past, a player's ratings over time
are what the model would have said about them that morning: the
predictions of @ch:evaluation were made from exactly these numbers.
That makes them different from season-end ratings, which are fit to the
whole season at once and describe it in hindsight. A daily rating in
November rests on a few weeks of this season and a discounted memory of
the last ones. Season-end ratings know how the season turned out.

The history starts in 2014. The box-score prior learns from earlier
seasons of plus-minus (@ch:prior), so a season can only be rated once
there are two seasons of play-by-play before it, and nba_data's
play-by-play currently starts in 2012. The first seasons rest on less
than the later ones: a prior learned from two seasons rather than ten,
and a fit with fewer years of games behind it.

## A career

![Paul George's ratings as of each day of each season since 2014: total,
offensive and defensive. Labels give the team George was on for most of
each season. Gaps are offseasons, when there are no
games.](figures/george.svg){#fig:george}

@fig:george follows Paul George through thirteen seasons and five teams.
In Indiana, George's rating ended each season between $+4.0$ and $+6.0$,
with defense rated as high as offense or higher. After the trade to
Oklahoma City it ended 2018 at $+4.2$ and 2019 at $+7.5$. Through five
seasons with the Clippers it stayed between $+5.8$ and $+7.5$, more and
more of it on offense. In Philadelphia, with 41 games played in 2025 and
49 in 2026, it fell to $+4.0$ and then $+3.5$, and this season, in
Boston, it started near $+3.6$.

The figure shows the model's habits as well as the player's. Within a
season the rating moves in small steps, a game's worth of evidence at a
time, and most at the start, when each game is a larger share of what's
new. Between seasons it can jump, because the box-score formula is refit
on opening night (@ch:prior). And the defensive line is the flatter one,
which is true of the ratings generally: among 2026's regulars, the
published defensive ratings have a standard deviation of 1.7 against
offense's 2.0. Defense leaves less of a trace in the box score and in the
scoring margin, so the data moves defensive ratings less far from their
priors.

## Teams at full strength, and tonight

The same ratings and minutes of @ch:score rate teams. Two ratings are
published for each team and day, both relative to the league's average
team:

- **Full strength**: the players who've played for the team this season,
  with 240 minutes given out in order of their minutes when playing, as
  if everyone were healthy.
- **Tonight**: the expected lineup the predictions use, with each
  player's chance of playing (@def:availability).

The gap between them is the team's injury report, in points per 100
possessions. @fig:warriors returns to the Warriors of @ch:question.

![Golden State's net rating in 2026 as of each day, relative to the
average team: at full strength (solid) and with the expected lineup
(dashed). Ticks along the bottom are games Stephen Curry
missed.](figures/warriors.svg){#fig:warriors}

At full strength the Warriors averaged $+6.0$ points per 100 possessions
over the season, a good team. With the lineup expected each day they
averaged $+3.3$, and the gap follows Curry's absences: on March 20, in
the middle of Curry's longest stretch out, they were $+7.9$ at full strength
and $+1.4$ as they would actually take the floor. A team rating fit to
results, @ch:question's baseline, sees one number somewhere in between
and can't tell which team is playing tonight.

## One model for all of history

The record has to be consistent to be worth charting. A model that
changed in March would draw a jump in every player's line in March, which
would look like something about the players. So every season of the
published history comes from the same model.

::: {.definition #def:version title="Model version"}
The *model version* is a hash of the program's code, its locked
dependencies, and the first season of play-by-play. Every row of the
daily datasets records the version that computed it, and every row of a
season has the same one.
:::

When the version changes, every season is refit from scratch, about
2,700 days of fits. The first season of play-by-play is in the hash
because of @ch:prior: adding earlier seasons changes what the box
formula learns, and so every rating after them, even though no code has
changed. This happened while this book was being written. On November 5,
2026, nba_data added play-by-play for 2012 through 2015, the version
changed, and the publishing job refit thirteen seasons, starting the
history in 2014 instead of 2018, without anyone doing anything.

::: code
The datasets are built by [`yarapm/publish.py`](code:yarapm/publish.py#L352):
`build` writes the current season every time nba_data updates, refitting
only the new days if the season's files were built by this version (at
about 0.2 to 0.9 seconds a day), and
[`backfill`](code:yarapm/publish.py#L476) refits finished seasons whose
files are stale, which takes about 45 minutes for all of them in GitHub
Actions. The version is [`model_version`](code:yarapm/publish.py#L150).
The full-strength lineups are in
[`yarapm/strength.py`](code:yarapm/strength.py#L88).

Each season is one Parquet file, sorted by player and written in row
groups of 20,480 rows. A reader filtering on one player can skip every
row group whose statistics say the player isn't in it, so one player's
career, over thirteen files, is about a megabyte over HTTP, little enough
to chart in a browser with DuckDB's WebAssembly build:

```sql
SELECT r.season, r.as_of_date, r.o_rating, r.d_rating, r.rating
FROM yarapm.ratings_daily r JOIN nba.players p USING (player_id)
WHERE p.name = 'Paul George'
ORDER BY r.as_of_date;
```
:::

## What isn't there yet

The record has no error bars. @prop:variance gave the variance of the
least squares estimate, and the same argument for ridge gives a
posterior variance of roughly $\sigma^2 (X^\top W X + \lambda I)^{-1}$
(@exr:posterior), so every rating could carry a band, narrow for a
veteran and wide for a rookie. Computing the diagonal of that inverse for
2,700 unknowns every day is the hard part; it's one of the project's
open experiments. Until then, read small differences between players, a
point or less, as noise.

## Exercises {.unnumbered}

::: {.exercise #exr:career}
Chart any player's career from `yarapm.ratings_daily` with the query
above. Find a player whose offensive and defensive ratings moved in
opposite directions over a few seasons.
:::

::: {.exercise #exr:injury-report}
From `yarapm.team_ratings_daily`, find the team and day in 2026 with the
largest gap between `full_net` and `net`. Using `yarapm.team_players_daily`,
which players account for it?
:::

::: {.exercise #exr:posterior}
Under the assumptions of @prop:bayes, show that the posterior covariance
of the ratings is $\sigma^2 (X^\top W X + \lambda I)^{-1}$. For the one
player of @thm:shrinkage, what is the posterior standard deviation with
$n$ possessions? With $\sigma = 110$ and $\lambda = 3{,}000$, how many
possessions does it take to get it below 1 point per 100?
:::

::: {.exercise #exr:hindsight}
`nba.openrapm` has season-end ratings fit to whole seasons. For 2026,
compare a few players' OpenRAPM ratings with their `ratings_daily`
ratings on the day after the season's last game, and on the first day of
the season. Which is closer to OpenRAPM, and why should it be?
:::
