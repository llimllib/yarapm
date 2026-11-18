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
weighting old games less, then lets players age, and then deals with a
second thing that changes over time: the league itself.

## Exponential decay

::: {.definition #def:decay title="Decay weights"}
On a fit day $t$, a segment played on day $t_s < t$ has *age* $a_s$: the
days from $t_s$ to $t$, not counting the breaks between seasons. Its
weight in @eq:ridge is multiplied by
$$
w_s = 2^{-a_s / h} ,
$$ {#eq:decay}
where $h$ is the *half life*, in days. Segments older than $8h$, whose
weights are below $1/256$, are left out.
:::

A break runs from a season's last game, the end of the Finals, to the
next season's first, usually about 128 days. Leaving it out means the
clock only runs while games are played, so a season, from opening night
to the Finals, is about 237 days long.

So @eq:ridge becomes $\sum_s w_s n_s (r_s - x_s^\top\beta)^2$ plus the
penalty. yarapm uses $h = 240$ days, about one season: a game from a
season ago counts half as much as one from last night, and one from two
seasons ago a quarter. @fig:decay shows the weights.

![(a) The weight of a game in the ratings, by its age: it halves every
240 days of the season, about a season, and games over eight seasons old
are dropped. (b) The weight of a game in the league's level
(@sec:level), which halves every week of calendar
time.](figures/decay.svg){#fig:decay}

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

How much data does a half life of a season amount to? Integrating the
weights,
$$
\int_0^{8h} 2^{-a/h}\, da = \frac{h}{\ln 2}\left(1 - 2^{-8}\right) \approx 1.44\,h ,
$$
so with $h = 240$ the fit has the weight of about 345 days of games at
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
than 730.

Those half lives were in calendar days, and they had a side effect. The
summer between seasons is four months long, so on opening night every
game counted about 20% less than it had at the Finals, while the penalty
pulling each player toward their prior stayed the same. Every rating was
pulled toward its prior without a game being played, and most for
players with little data: Day'Ron Sharpe's rating fell by 0.4 to 0.5
points over each of the last two summers. Counting only days in the
season, with a half life of 240 days to keep about the same memory,
predicted 2021–2024 just as well (a daily margin RMSE of 13.570, against
13.568 with 365 calendar days), and the summer drop is gone. That's the
default.
:::

## Aging {#sec:aging}

Decay treats an old game as weaker evidence about a player, but still
as evidence about the same player. For a 21-year-old that's wrong in a
predictable direction: last season's games show a worse player than the
one playing tonight. For a 32-year-old they show a better one. Neither
half life helps. A shorter one forgets the old player sooner, but it
forgets the evidence about the player along with them.

The model noticed. Early in a season, when the ratings are mostly last
season's, its predicted margins were too spread out: regressing the
actual margins on the predicted ones, over 2021–2024's games played
before either team's 20th, gave a slope of 0.77, so a predicted 10-point win was on average a 7.7-point
one. The closing betting line, on the same games, had a slope of 0.92.

::: {.definition #def:aging title="Aging"}
Let $a_i(t)$ be player $i$'s age in season $t$, and let $A^o$ and $A^d$
be *aging curves*, each a cubic in age,
$$
A(a) = \gamma_1 (a - 27) + \gamma_2 \frac{(a - 27)^2}{10} + \gamma_3 \frac{(a - 27)^3}{100},
$$ {#eq:aging}
with ages clamped to 19–38. In a fit for season $T$, a segment $s$ from
season $t_s$ is modeled as
$$
\begin{aligned}
r_s = \mu & + \sum_{i \in O_s} \left(o_i - \Delta^o_{is}\right) \\
& - \sum_{j \in D_s} \left(d_j - \Delta^d_{js}\right) + \varepsilon_s ,
\end{aligned}
$$ {#eq:apm-aging}
where $\Delta_{is} = A(a_i(T)) - A(a_i(t_s))$ is how much the curve
says player $i$ has changed since the segment was played.
:::

So $o_i$ and $d_i$ are the player as of season $T$, and each old segment
is read as evidence about that player, after allowing for the years in
between. In segments from this season, $\Delta = 0$. Because
@eq:aging is linear in the $\gamma$s, the $\Delta$ terms sum to six more
columns of $X$, three for offense and three for defense, and the
$\gamma$s are covariates like home court (@sec:covariates). They're
fit along with the ratings, with a small penalty of their own, from how
players' plus-minus changed between seasons as they got older. Ages
come from nba_data's season stats, as a year of birth.

::: {.remark #rem:aging-level}
Only the curve's shape is learned. If every player's offense improved
by $c$ points a year, the five offensive players in a segment from $k$
seasons ago would all have been $c\,k$ worse than they are today, $5c\,k$
points per 100 in all, the same for every segment of that season. The
season offsets absorb a shift like that, so a constant added to the
yearly change at every age changes no prediction. What the data pins
down is how players of one age change compared with players of
another.
:::

![(a) The aging curves fit on the opening night of 2026: how much a
year changes a rating, by age, for offense (solid) and defense (dashed).
Only the shape matters (@rem:aging-level). Past 34 there are few players
left, and the cubic turns up. (b) The change in each regular's rating
(3,000 or more possessions in 2025) from the last day of 2025 to the
first day of 2026, by their age in 2026.](figures/aging.svg){#fig:aging}

@fig:aging shows the curve. From one season to the next, a 20-year-old
gains about 1.4 points per 100 possessions more than a 30-year-old, a
little more of it on offense than on defense, and the gain shrinks
steadily until the late twenties.
Panel (b) is what this does to ratings over a summer. The regulars aged
19 to 23 rose by 0.39 on average between the 2025 Finals and opening
night, those aged 28 to 31 fell by 0.23, and without aging both groups
moved by 0.02 or less. The gap between the two groups, 0.62, is about
half what the curve says, because each rating is still pulled toward a
box-score prior that doesn't age.

::: {.example #ex:aging}
On 2021–2024, with daily refits and who plays known, aging lowered the
margin RMSE from 13.570 to 13.536, and from 13.475 to 13.405 in the
games before either team's 20th. The early slope rose from 0.77 to 0.83, part
of the way to the line's 0.92. A difference of 0.034 is small. But the
two models predicted the same 5,133 games, and comparing their squared
errors game by game, the difference is about four standard errors from
zero, not noise. Two ways of going further did no better. Shifting the
box-score prior along the curve as well, by the age of the box scores it
was made from, was worse (13.605 against 13.549, weekly). Pulling each
older season's evidence a fixed fraction toward the prior, a regression
to the mean on top of the aging, gained 0.006, within the noise.
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
[`decay()`](code:yarapm/fit.py#L6) is one line. The ages, without the
breaks between seasons, come from
[`History.age()`](code:yarapm/harness.py#L204). The cutoff at eight half
lives is applied in SQL, before any rows are read
([`lineups()`](code:yarapm/players.py#L145)), so a fit reads only the
segments that still matter: eight seasons of them, about 570,000 rows.
The aging columns are
[`aging_covariates()`](code:yarapm/players.py#L104), from the cubic in
[`age_basis()`](code:yarapm/players.py#L85) and each player's year of
birth ([`born()`](code:yarapm/players.py#L95)). The
level is computed right after the ratings, in
[`PlayerRatings.fit`](code:yarapm/players.py#L268), from the same rows
reweighted with the seven-day half life. Because @prop:memoryless makes
a day's change in the weights a single factor, yesterday's ratings are a
good starting point for today's solve, which is why each day's fit is
warm-started from the last.
:::

## Exercises {.unnumbered}

::: {.exercise #exr:effective}
A player has played 5,000 offensive possessions in each of the last
eight seasons, spread evenly through each one. Approximately what is the
total decayed weight of their possessions with $h = 240$, if a season is
240 days long? By
@thm:shrinkage, what share of their rating comes from the data rather
than the prior? What if they're in their second season?
:::

::: {.exercise #exr:memoryless-only}
Show that if a weight function $w(a)$, with $w(0) = 1$, has the property
of @prop:memoryless for every $a$ and $\Delta$, then $w(a) = c^a$ for some
constant $c$. (Hint: $w(a + \Delta) = w(a)\, w(\Delta)$.)
:::

::: {.exercise #exr:aging-shape}
Show that adding a constant $c$ to the offensive curve's yearly change at
every age, $A^o(a) \to A^o(a) + c\,a$, changes the predicted rate of
every segment from season $t$ by the same amount. How much, and why can
the season offsets absorb it, but not a change that's larger for
younger players?
:::

::: {.exercise #exr:aging-rookie}
A player rated $+1$ today is 22. Their only games are from last season.
Using panel (a) of @fig:aging, about what would last season's games alone have rated
them, ignoring the prior? Why does panel (b) show a smaller change
for players that age than (a) does?
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
