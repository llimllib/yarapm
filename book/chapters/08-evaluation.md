---
title: Was It Any Good?
label: ch:evaluation
---

The model said the Spurs would win Game 5 by six, and they lost by four.
One game says almost nothing about a model; the question is how it does
over thousands. This chapter defines what "how it does" means, turns
predicted scores into win probabilities, and reports the results. It
also explains how the results were kept honest, which matters as much as
the numbers.

## Errors in points

For each game the model predicts a margin $\hat M = \hat S_A - \hat
S_B$, from the home team's side, and a total $\hat S_A + \hat S_B$. Its
error in a game is the predicted margin minus the actual one, and over a
set of $N$ games we report three summaries:

::: {.definition #def:errors title="Error summaries"}
For errors $e_1, \ldots, e_N$, the *bias* is $\frac1N \sum e_g$, the
*mean absolute error* is $\frac1N \sum |e_g|$, and the *root mean square
error* is $\sqrt{\frac1N \sum e_g^2}$.
:::

The RMSE is the headline number: it punishes large misses more than
small ones, and it's what the regression minimizes. Over the 5,133 games
of 2021–2024, predicted the way the model is used, from the games before
each day and guessing who will play, the margin RMSE is 13.71 points,
the mean absolute error 10.76 and the bias $-0.22$. For totals the RMSE
is 18.43.

## From a margin to a probability

A prediction of "Spurs by 6.3" isn't yet a statement about who will win.
To make it one we need the spread of the actual margin around the
predicted one, and the simplest assumption is that it's normal.

::: {.definition #def:win-prob title="Win probability"}
If the margin is $M \sim N(\hat M, \sigma^2)$, the home team wins with
probability
$$
\Pr(M > 0) = \Phi\!\left(\frac{\hat M}{\sigma}\right),
$$ {#eq:win-prob}
where $\Phi$ is the standard normal distribution function, and $\sigma$
is the model's margin RMSE on the tuning seasons, 13.71 points.^[$\Phi(z)
= \int_{-\infty}^z e^{-t^2/2}\,dt/\sqrt{2\pi}$ is the probability that a
standard normal is at most $z$; if $M \sim N(\mu, \sigma^2)$ then $(M -
\mu)/\sigma$ is standard normal, so $\Pr(M > 0) = \Pr\!\left(\frac{M -
\mu}{\sigma} > -\frac{\mu}{\sigma}\right) = \Phi(\mu/\sigma)$.]
:::

For Game 5, $\Phi(6.3/13.71) = \Phi(0.46) = 0.68$. Is the normal
assumption any good? @fig:errors compares the errors with a normal curve
of the same spread. They fit well: 29.9% of games miss by more than one
$\sigma$, against 31.7% for a normal, and 4.6% by more than two, against
4.55%.

![The distribution of the error in the predicted margin, 2021–2024, in
4-point bins, and a normal density with the same standard deviation,
13.71 points.](figures/errors.svg){#fig:errors}

## Scoring probabilities

A probability can't be right or wrong about one game, but it can be
scored. Two standard scores, for a predicted probability $q$ that the
home team wins and outcome $y = 1$ if it did, $0$ if not, are

::: {.definition #def:scores title="Log loss and Brier score"}
$$
\begin{aligned}
\text{log loss} &= -\frac1N \sum_g \left[y_g \ln q_g + (1 - y_g) \ln(1 - q_g)\right], \\
\text{Brier} &= \frac1N \sum_g (q_g - y_g)^2 .
\end{aligned}
$$
Lower is better. Saying 50% every game scores $\ln 2 \approx 0.693$
and $0.25$.
:::

Both are *proper*: a forecaster who believes the probability is $p$ gets
the best expected score by saying $p$, and can't gain by hedging or by
exaggerating (@exr:proper). Log loss punishes confident misses severely,
since $-\ln q$ grows without bound as $q \to 0$, which makes it sensitive
to whether $\sigma$ is right. With $\sigma$ too small, every prediction
is overconfident, and the losses on the upsets dominate.

The model's log loss on 2021–2024 is 0.624, and its Brier score 0.218.
The most direct check of the probabilities is *calibration*: of the
games it gave the home team a 70–80% chance, the home team should win
about 75%. @fig:calibration groups the games by predicted probability.
Every group is within its sampling error of the diagonal, and overall,
the favorite won 64.9% of the games, where the model expected 64.3%.

![Calibration on 2021–2024: the games grouped by the predicted chance
that the home team wins, in tenths, plotted at the group's average
prediction against the share of them it won, with two standard errors
either side. The dashed diagonal is perfect
calibration.](figures/calibration.svg){#fig:calibration}

## The scoreboard

Every change to the model is judged by these numbers, against baselines
that set the bar.

| model | margin RMSE | log loss | total RMSE |
|:--|--:|--:|--:|
| home court only | 14.98 | 0.687 | 19.97 |
| team ratings | 13.93 | 0.636 | 18.73 |
| last season's OpenRAPM, who plays guessed | 13.99 | 0.635 | 18.62 |
| player ratings, who plays guessed | **13.71** | **0.624** | **18.43** |
| player ratings, who plays known | 13.58 | 0.613 | 18.35 |
| this season's OpenRAPM (a ceiling) | 13.06 | 0.591 | 17.96 |

Table: Daily walk-forward results on the 2021–2024 seasons, 5,133
games. "Who plays known" uses who actually played, standing in for an
injury report. The OpenRAPM rows feed another site's published player
ratings through the same prediction step; the last uses ratings fit to
the whole season being predicted, so it has seen the games, and is an
upper bound rather than a fair comparison. {#tbl:scoreboard-full}

@tbl:scoreboard-full puts the work of the earlier chapters in order.
Ratings fit to the whole of the season being predicted, which no fair
model could have, would get to about 13.1; nothing that only sees the
past will. Between that ceiling and team ratings, yarapm's player model
with guessed availability closes about a quarter of the gap, and knowing
who plays closes 40%. OpenRAPM, a respected public set of
player ratings, does no better than team ratings when last season's are
used for this one, which is a reminder that ratings for *describing*
past seasons and ratings for *predicting* the next game are different
products.^[OpenRAPM's ratings are in nba_data's catalog as
`nba.openrapm`, one row per player and season.]

The numbers also vary by the phase of the season. Early, when either
team has played fewer than 20 games and rosters are new, the margin RMSE
is 13.59; in mid-season 13.61; after the trade deadline 13.74; in the
playoffs 14.53. The playoffs are where the model is weakest: it
underrates home teams by 2.4 points of margin and overpredicts totals by
1.4, both open questions in the project.

## Keeping it honest

A model tuned on enough settings will find some that fit the past by
luck. Three habits guard against fooling ourselves.

1. **Walk forward, always.** Every number in this book is from
   predictions made with data from before the game's day, by the same
   code that makes the live predictions (@sec:honesty).
2. **Tune on some seasons, test on others.** Settings were chosen on
   2021–2024, and 2025–2026 were held out until every setting was fixed,
   then scored once.
3. **Distrust small differences.** The grids in @tbl:ridge-grid and
   @ex:half-life are flat near their minima, and a difference of a few
   hundredths of a point is noise. Where settings are within 0.03 points
   of the best, the simplest was kept.

| model | margin RMSE | log loss |
|:--|--:|--:|
| home court only | 16.23 | 0.688 |
| team ratings | 14.48 | 0.608 |
| player ratings, who plays guessed | **14.42** | **0.606** |
| player ratings, who plays known | 14.22 | 0.593 |
| this season's OpenRAPM (a ceiling) | 13.69 | 0.570 |

Table: The held-out seasons, 2025–2026, 2,643 games, scored once. {#tbl:holdout}

@tbl:holdout is the result. Every model's errors are larger, because
those two seasons' games were less predictable: the standard deviation
of the margin itself was 16.0 and 16.5 points, against 13.7 to 15.7 in the
tuning seasons. But the order held, and the player model's margin RMSE
stayed ahead of team ratings, by less than in the tuning seasons.

The last check is the one that can't be gamed. Since November 2026,
every prediction the model publishes before a game is logged, with the
time it was made, and never changed. As games finish, the log is scored
with the same metrics, separately for each version of the model. That
record will be the real answer to this chapter's question; as this is
written, it has scored a handful of games.

::: code
The harness is [`yarapm/harness.py`](code:yarapm/harness.py#L231): for
each game day it fits the model on a `History` cut at that day and
predicts the day's games, then writes the predictions and
[`score()`](code:yarapm/harness.py#L413)'s metrics to Parquet, tagged
with a run id hashed from the model's settings. A daily run over four
seasons takes about seven minutes; for tuning, a weekly cadence, fitting
on each week's first game day, is fast enough to sweep grids and close
enough to the daily results to rank settings. The phases come from
[`phases()`](code:yarapm/harness.py#L342), which needs each season's
trade deadline. The live log is `prediction_log` in the published data,
scored into `live_metrics`.
:::

## Exercises {.unnumbered}

::: {.exercise #exr:win-prob}
What is the win probability of a 1-point favorite under @eq:win-prob? A
10-point favorite? What predicted margin makes a team a 90% favorite?
:::

::: {.exercise #exr:proper}
Show that log loss is proper: if the home team wins with probability
$p$, the expected log loss of forecasting $q$, $-p\ln q - (1-p)\ln(1-q)$,
is minimized at $q = p$. Do the same for the Brier score.
:::

::: {.exercise #exr:sigma}
Suppose the model used $\sigma = 10$ in @eq:win-prob instead of 13.71,
with the same predicted margins. Which way would @fig:calibration's points
move? Would the log loss go up or down? Why?
:::

::: {.exercise #exr:live}
Query `yarapm.live_metrics` for the current season. How many games has
the live log scored, and how do the `rapm` and `team` series compare so
far? How many games would you want before trusting the difference?
:::
