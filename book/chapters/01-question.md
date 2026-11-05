---
title: The Question
label: ch:question
---

On June 13, 2026, the New York Knicks won the NBA championship in San
Antonio, 94–90, taking the Finals four games to one. On the morning of
that game we knew every game played so far that season: who played in it,
for how long, and what happened while they were on the floor. How many
points should we have expected each team to score?

That is the whole question this book answers, for every game on every
day. Everything else, the ratings, the priors, the projected minutes, is
there to answer it well. We
start by making the question precise, then split it into two smaller ones
that are easier to answer, and end the chapter with a map of the rest of
the book.

## What we predict

::: {.definition #def:prediction title="Prediction"}
A *game* $g$ is played on a day $t_g$ between two teams, $A$ and $B$. A
*prediction* for $g$ is a pair of numbers $(\hat S_A, \hat S_B)$, the
points we expect each team to score, computed from the games played
before day $t_g$ and nothing else.
:::

From the pair we get the expected margin $\hat S_A - \hat S_B$ and total
$\hat S_A + \hat S_B$, and in @ch:evaluation a probability that $A$
wins.^[*Expected* is meant in the probabilist's sense: $\hat S_A$ is
our estimate of $\mathbb{E}[S_A]$, the mean of the score $S_A$ over
the games that could happen tonight. A refresher: for a random variable
$X$ taking values $x$ with probabilities $p(x)$, $\mathbb{E}[X] =
\sum_x x\,p(x)$; expectation is linear, so $\mathbb{E}[S_A - S_B] =
\mathbb{E}[S_A] - \mathbb{E}[S_B]$.] The final score is a long way from
certain. The margin of an NBA game varies by about 14 points around
anything a model can predict, so the goal isn't to be right but to be
less wrong, on average, than the alternatives.

The restriction to earlier games is the most important rule in the book.
A model that peeks at the game it predicts, even a little, looks better
than it is, and the program is built so that it can't (@sec:honesty).

## Possessions and efficiency

A basketball game is a sequence of possessions. One team has the ball
until it scores, misses and the other team rebounds, or turns it over;
then the other team has it. A miss that the shooting team rebounds itself
doesn't end anything: the possession goes on, and a team that misses
three times and scores on a putback has had one possession, worth two
points.

::: {.definition #def:possession title="Possession"}
A team's *possession* starts when it gains control of the ball and ends
when the other team does. Its *efficiency* $e$ in a game is its points per
100 possessions: a team scoring $S$ points on $P$ possessions has $e = 100
S / P$.
:::

Since possessions alternate, the two teams in a game have almost the same
number of them.^[Not exactly the same: whoever has the ball first in a
period may also have it last. Over the 2016–2027 seasons, the two teams'
counts in a game never differ by more than 4.] Call that number $P$, the
game's *pace*. Then the score splits into a product:

$$
S_A = \frac{P \, e_A}{100}, \qquad S_B = \frac{P \, e_B}{100}.
$$ {#eq:score}
In the Finals clincher, San Antonio had 90 possessions and scored 90
points, an efficiency of 100.0, and New York scored 94 on 91, or 103.3.
Both are far below the league's regular-season average of 114.5, and so
is the pace: about 90 possessions against the league's 100. Finals games
are slow, and Finals defenses are good.

This is an identity, not a model: any score can be written this way. Its
value is that the two factors depend on different things. The pace $P$
belongs to the game, since both teams share the same possessions, and it
depends on how fast both teams like to play. The efficiency $e_A$ depends
on how good $A$'s offense is and how good $B$'s defense is. So we can
predict them separately: pace from how fast the two teams' games have
been, and efficiency from the players on the floor. In 2026, Miami played
the fastest, at 104.1 possessions per 48 minutes, and Boston the slowest,
at 95.4; but Boston had the best offense, 119.5 points per 100
possessions, so the slow pace wasn't hurting it. The best defense was
Detroit's, allowing 108.8.

![Pace and efficiency of the league, 2016–2026: (a) possessions per 48
minutes, per team, and (b) points per 100 possessions. The pace has
varied by about four possessions a game. Efficiency rose from 105.6 to
114.5 points per 100 possessions in ten
seasons.](figures/league.svg){#fig:league}

@fig:league shows the two factors over the last decade. Pace has been
steady near 100 possessions a game. Efficiency has risen by nine points
per 100 possessions, and it moves within a season too, so a model needs a
sense of how many points the *league* scores right now, apart from how
good any one team is. We return to this in @ch:time.

Most of the book is about efficiency, since it's where the teams differ
most. A typical difference between two teams' efficiencies is several
points per 100 possessions, worth several points of margin, while their
paces differ by a few possessions, worth only a fraction of a point of
margin (@exr:pace).

## Why players, not teams

The simplest model of efficiency gives each team an offensive rating and a
defensive rating, and says that $A$'s efficiency against $B$ is the
league's average plus $A$'s offensive rating minus $B$'s defensive
one.^[This is the model behind Ken Pomeroy's college basketball
ratings, [kenpom.com](https://kenpom.com), and yarapm uses it as a
baseline: the `team` model.] Fit to every game so far, it does well. But
a team isn't a fixed thing. A star sits out with a sprained ankle, a
trade sends two starters away, a coach rests everyone on the second night
of a back-to-back. A team's rating is an average over the lineups it has
used, and tonight's may be different.

Golden State in 2026 is a clean example. Stephen Curry played 43 of the
Warriors' 82 regular-season games. They outscored their opponents by 4.0
points a game when Curry played and were outscored by 5.5 when Curry
didn't.^[A raw with-and-without split like this isn't a fair
measure of a player: other players were hurt at the same time, the
schedule differed, and 39 games is a small sample. @ch:regression is
about doing it properly. But the size of the swing is the point here.] A
team rating sees a team that's about even, $-0.5$ a game, and predicts
every game as if the Warriors were that team. They never were: on any
given night they were either a good team or a bad one, and whether Curry
was on the injury report said which.

So yarapm rates players instead, and builds each team from the players
expected to play tonight, weighted by how many minutes each will play. A
player's rating goes with them when they're traded, and a missing player
is simply left out. The price is that rating players is much harder than
rating teams: there are about 450 of them in a season against 30 teams,
and they play in groups of five.

| model | margin RMSE | log loss |
|:--|--:|--:|
| home court only | 14.98 | 0.687 |
| team ratings (`team`) | 13.93 | 0.636 |
| player ratings, who plays guessed | 13.71 | 0.624 |
| player ratings, who plays known | 13.58 | 0.613 |

Table: How far off each model's predicted margin is, in points (the
root mean square error, @ch:evaluation), and its log loss on win
probabilities, lower is better for both. Games of 2021–2024, each predicted
from the games before its day. {#tbl:scoreboard}

@tbl:scoreboard is the payoff. Predicting every game with only a
home-court edge misses the margin by about 15 points. Rating teams cuts
that to 13.9, and rating players to 13.7, or 13.6 when we know who will
play. These differences look small, but most of the 13.6 is the
randomness of a single game, which no model can remove. Even a model
allowed to cheat, by using player ratings fit to the whole season
including the games it predicts, only gets to 13.06. Against that floor,
the step from teams to players is a good share of the error that can be
removed at all.

## The plan of the book

@fig:pipeline shows how the pieces fit. The raw material is
play-by-play data: who was on the floor for each stretch of each game,
and what happened. @ch:segments cuts each game into *lineup segments*,
stretches with the same ten players on the floor. @ch:regression and
@ch:ridge fit player ratings to them by regression, @ch:time adds
the weighting that lets old games count less, and @ch:prior gives
every player a starting guess from box-score statistics. @ch:score
combines the ratings with projected minutes and pace into an expected
score, @ch:evaluation measures how good the predictions are, and
@ch:history follows the ratings day by day through ten seasons.

![The model. Data from nba_data on the left becomes lineup segments, a
box-score prior and tempo; segments and the prior give player ratings;
ratings, projected minutes and tempo give each team's expected score and
a win probability. The labels give the chapters.](figures/pipeline.svg){#fig:pipeline}

## The no-peeking rule {#sec:honesty}

Every model in this book is judged the same way. For each day of a
season, we fit the model on the games before that day, predict that
day's games, and compare the predictions to what happened. This is called
*walk-forward* testing, and it's the only kind of testing that tells us how
the model will do on games that haven't happened yet. Any setting of the
model, such as how fast old games are forgotten, is chosen by its walk-forward
results on the 2021–2024 seasons, and the 2025–2026 seasons were held back
and scored once, at the end, to check that the choices weren't tuned to
the noise in the four tuning seasons.

::: code
The data comes from [nba_data](https://github.com/llimllib/nba_data), a
separate project that publishes NBA data as Parquet files with a DuckDB
catalog. [`connect()`](code:yarapm/catalog.py#L42) attaches it as `nba`,
so every query reads `nba.<table>`, and what yarapm builds goes under
`out/`, one Parquet file per season.

The no-peeking rule is enforced by a class: a model is fit on a
[`History`](code:yarapm/harness.py#L114), whose every table is filtered to
games before the day being predicted, and that's all it can see. Which
games count at all (regular season, play-in, playoffs and the Cup final,
but not preseason or the All-Star game) is decided in
[`yarapm/games.py`](code:yarapm/games.py#L11), which also treats the 2020
games in the Orlando "bubble" as played on a neutral court.
:::

## Notation

We use these symbols throughout. Ratings are in points per 100
possessions, so a rating of $+2$ means two more points per 100
possessions than average.

| symbol | meaning |
|:--|:--|
| $i, j$ | players |
| $s$ | a lineup segment, a stretch of a game with the same ten players |
| $O_s, D_s$ | the five players on offense and on defense in segment $s$ |
| $n_s, y_s$ | the offense's possessions and points in segment $s$ |
| $o_i, d_i$ | player $i$'s offensive and defensive ratings |
| $\mu$ | the league's average points per 100 possessions |
| $P$ | a game's pace, possessions per team |
| $\lambda$ | a ridge penalty, in possessions (@ch:ridge) |

Table: Notation. {#tbl:notation}

## Exercises {.unnumbered}

::: {.exercise #exr:efficiency}
A team scores 112 points on 98 possessions, and its opponent scores 104
on 97. What are their efficiencies? If both had had exactly 98
possessions at those efficiencies, what would the margin have been?
:::

::: {.exercise #exr:pace}
Suppose both teams have $P$ possessions. Show from @eq:score that the
margin is $P(e_A - e_B)/100$ and the total is $P(e_A + e_B)/100$. If
$e_A - e_B = 6$, how much does the expected margin change when the pace
goes from 96 to 104? Which team should want a faster game, the better
one or the worse one?
:::

::: {.exercise #exr:alternate}
Why can't two teams' possession counts in a game differ by more than a
few? What would have to happen for them to differ by two in one
period?
:::

::: {.exercise #exr:query}
With DuckDB, attach nba_data's catalog,

```sql
ATTACH 'https://basketball-data.billmill.org/nba/nba.duckdb' AS nba (READ_ONLY);
```

and compute the league's points per 100 possessions in the 2027 season so
far from `nba.team_game_logs` (columns `season`, `pts` and `poss`). Is
it higher or lower than 2026's 114.5? (Filter on `season`: other filters
read every season's file.)
:::
