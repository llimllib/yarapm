---
title: Time
label: ch:time
---

Players change. A 22-year-old gets better every season, a 35-year-old
gets worse, and anyone can come back from an injury a step slower.
@ch:ridge ended with the opposite problem: one season of data isn't
enough to tell Nikola Jokić from an average player with any confidence.
These pull in opposite directions. More seasons of data mean better
estimates of the player as they *were*, and worse estimates of the player
as they *are*. This chapter resolves the tension the standard way, by
weighting old games less, and then deals with a second thing that changes
over time: the league itself.

## Exponential decay

::: {.definition #def:decay title="Decay weights"}
On a fit day $t$, a segment played on day $t_s < t$ has *age* $a_s = t -
t_s$ days, and its weight in @eq:ridge is multiplied by
$$
w_s = 2^{-a_s / h} ,
$$ {#eq:decay}
where $h$ is the *half life*, in days. Segments older than $8h$, whose
weights are below $1/256$, are left out.
:::

So @eq:ridge becomes $\sum_s w_s n_s (r_s - x_s^\top\beta)^2$ plus the
penalty, and a possession from a year ago counts half as much as one from
last night. yarapm uses $h = 365$ days: last season counts about half,
the one before a quarter. @fig:decay shows the weights.

![(a) The weight of a game in the ratings, by its age: it halves every
365 days, and games over eight years old are dropped. (b) The weight of a
game in the league's level (@sec:level), which halves every
week.](figures/decay.svg){#fig:decay}

Why exponential, and not, say, the last two seasons equally? Two reasons.
The first is that it has no edge: a game doesn't go from counting fully
to not counting at all on one particular day, which would make ratings
jump when a big game falls out of the window. The second is that it's
the only weighting that doesn't depend on when you look.

::: {.proposition #prop:memoryless}
Moving the fit day forward by $\Delta$ days multiplies every existing
segment's weight by the same factor, $2^{-\Delta/h}$.
:::

::: proof
$2^{-(a + \Delta)/h} = 2^{-\Delta/h}\, 2^{-a/h}$.
:::

So between one day and the next, the relative weights of all old games
are unchanged, and the only things that change are that the new day's
games arrive at full weight and everything older counts a little less
compared with them, and with the penalty. That makes the day-to-day
change in the ratings small and smooth, which is what makes the warm
starts of @sec:cg work.

How much data does a half life of a year amount to? Integrating the
weights,
$$
\int_0^{8h} 2^{-a/h}\, da = \frac{h}{\ln 2}\left(1 - 2^{-8}\right) \approx 1.44\,h ,
$$
so with $h = 365$ the fit has the weight of about 525 days of games at
full strength: roughly a season and a half, spread over eight. For a
regular who plays 5,000 possessions a season, that's about 7,000
possessions of weight, against a penalty of 3,000 (@exr:effective).

::: {.example #ex:half-life}
The half life was chosen like the penalty, by walk-forward testing. With
$\lambda = 3{,}000$ and no box-score prior, predicting 2021–2024 with
weekly refits, the margin RMSE was 13.776 points with a half life of 180
days, 13.717 with 365, 13.706 with 730 and 13.715 with 1,460. Again the
minimum is broad. A half life of half a season is clearly too short,
throwing away data the model needs, and anything from one to four years
is about the same. With the box-score prior, 365 days did slightly better
than 730, and it's the default.
:::

## The league's level {#sec:level}

Players aren't the only things that change. @fig:league showed the
league's efficiency rising by nine points per 100 possessions over a
decade, and @fig:drift shows that it moves within a season too, by as
much as the gap between a good team and an average one.

![The league's points per 100 possessions in each month of the regular
season, in three seasons. In 2022 it rose almost ten points from October
to April; in 2024 it peaked in December and fell four points by April;
2026 was nearly flat.](figures/drift.svg){#fig:drift}

In 2022, scoring rose from 105.6 points per 100 possessions in October to
115.3 in April. In 2024 it went the other way after the new year, from
116.7 in December to 112.7 in April. A model whose sense of the league's
average lags by a few months will be off by several points a game in
total, even if every player is rated perfectly.

The regression of @ch:ridge has a season offset for each season, but it
can't follow this. An offset is one number for a whole season, and it's
penalized, so early in a season it barely moves. yarapm instead estimates
the level separately, after the ratings, with a much shorter memory.

::: {.definition #def:level title="The league's level"}
Given the fitted ratings and covariates, let $\rho_s$ be segment $s$'s
predicted rate with the intercept and season offsets left out: its
players' ratings, home court and the playoff term. The *level* on day
$t$ is the points per 100 possessions those don't explain, averaged with
weights $v_s = 2^{-a_s/7}$ that halve every week:
$$
\ell = 100\, \frac{\sum_s v_s\, (y_s - n_s\, \rho_s / 100)}{\sum_s v_s\, n_s} .
$$ {#eq:level}
:::

Predictions use $\ell$ in place of the intercept and season offset. Since
the ratings and the level are fit separately, the ratings still learn
from eight years of games, while the level follows the last few weeks.
This mattered for totals, not margins: the level is the same for both
teams in a game, so it cancels out of the margin. Tuning its half life on
total points (@ch:evaluation), the total RMSE was 18.66 with a 60-day
half life, 18.58 with 30, 18.51 with 14 and 18.50 with 7, and before this
estimate existed, totals after the trade deadline were off by 3 to 8
points a game in some seasons.

There's a subtlety at the start of a season. In late October, the
newest games are last spring's playoffs, months old, and with a
seven-day half life the weights of all of them are tiny. But
@eq:level is a ratio, so only their *relative* weights matter, and the
level on opening night is essentially the level of the last few weeks
of the previous season, with the playoff term taken out.

::: code
[`decay()`](code:yarapm/fit.py#L6) is one line. The cutoff at eight half
lives is applied in SQL, before any rows are read
([`lineups()`](code:yarapm/players.py#L69)), so a fit reads only the
segments that still matter: eight years of them, about 570,000 rows. The
level is computed right after the ratings, in
[`PlayerRatings.fit`](code:yarapm/players.py#L178), from the same rows
reweighted with the seven-day half life. Because @prop:memoryless makes
a day's change in the weights a single factor, yesterday's ratings are a
good starting point for today's solve, which is why each day's fit is
warm-started from the last.
:::

## Exercises {.unnumbered}

::: {.exercise #exr:effective}
A player has played 5,000 offensive possessions in each of the last
eight seasons, spread evenly through each one. Approximately what is the
total decayed weight of their possessions with $h = 365$? By
@thm:shrinkage, what share of their rating comes from the data rather
than the prior? What if they're in their second season?
:::

::: {.exercise #exr:memoryless-only}
Show that if a weight function $w(a)$, with $w(0) = 1$, has the property
of @prop:memoryless for every $a$ and $\Delta$, then $w(a) = c^a$ for some
constant $c$. (Hint: $w(a + \Delta) = w(a)\, w(\Delta)$.)
:::

::: {.exercise #exr:level-margin}
Show that the level $\ell$ cancels out of a game's predicted margin when
both teams have the same number of possessions. Why does a
seven-day half life hurt the ratings if it's used for them too?
:::

::: {.exercise #exr:monthly}
Using `nba.team_game_logs`, compute the league's points per 100
possessions by month for the 2027 season so far. Is it above or below
the level of April 2026, 116.8?
:::
