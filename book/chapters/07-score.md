---
title: From Ratings to a Score
label: ch:score
---

We have a rating for every player, a level for the league, and terms for
home court and the playoffs. To predict a game we need two more things:
how many minutes each player will play, and how many possessions the
game will have. This chapter adds both, and then predicts one game from
start to finish: Game 5 of the 2026 Finals, as the model saw it on the
morning it was played.

## A team is its minutes

A team's strength in a game is its players' ratings weighted by how much
each is expected to play. Five players are on the floor at a time, so
over a game a team's players play 240 minutes between them.

::: {.definition #def:strength title="Team strength"}
If player $i$ is expected to play $m_i$ minutes for a team, with $\sum_i
m_i = 240$, the team's offensive and defensive strengths are
$$
\begin{aligned}
O &= 5 \sum_i \frac{m_i}{240}\, o_i , \\
D &= 5 \sum_i \frac{m_i}{240}\, d_i .
\end{aligned}
$$ {#eq:strength}
:::

The factor of 5 makes this the average over the team's possessions of the
sum of the five ratings on the floor, which is what @eq:apm adds up. A
team that played the same five players all 48 minutes would have $O$
equal to the sum of their offensive ratings.

::: {.definition #def:expected-score title="Expected score"}
In a game between $A$, at home, and $B$, with $P$ expected possessions
each, the expected points are
$$
\begin{aligned}
\hat S_A &= \tfrac{P}{100}\left(\ell + \pi + \eta + O_A - D_B\right), \\
\hat S_B &= \tfrac{P}{100}\left(\ell + \pi - \eta + O_B - D_A\right),
\end{aligned}
$$ {#eq:expected-score}
where $\ell$ is the league's level (@def:level), $\pi$ the playoff term if
it's a playoff game, and $\eta$ the home-court term (@sec:covariates),
which is zero at a neutral site.
:::

This is @eq:score with each efficiency predicted from the model. All that
remains is $m_i$ and $P$.

## Minutes when playing

How long will a player play, if they play? Rotations change slowly, so
the best guess is recent history.

::: {.definition #def:when-playing title="Minutes when playing"}
A player's projected minutes when playing are an average of their minutes
in their recent games, with the $k$th most recent game weighted by
$2^{-k/5}$: a half life of five *games*. Each game's minutes are first
scaled so the team's add to 240, so overtime doesn't count. A player who
started their last game, and doesn't usually start, gets up to five
minutes more, and one who usually starts but didn't, up to five fewer. A
player with no games gets 10 minutes.
:::

The half life is counted in the player's own games, not in days, so a
player back from a month out with an injury is projected from the games
before it rather than from nothing. And the player's team doesn't enter:
a player traded yesterday brings their minutes along, which is usually
about right. The five-game half life and the five-minute start bonus
were tuned on how well they predicted minutes. Over the 2021–2024
seasons, the projections missed each player's minutes by 4.7 a game on
average when we knew who would play, and the predicted margins were
barely sensitive to the choice: within 0.03 points over the whole grid.

## Who plays

The harder question is who will play at all. The NBA publishes injury
reports, but nba_data doesn't have them yet, so the model guesses from
each player's recent pattern.

::: {.definition #def:availability title="Chance of playing"}
A team's *roster* is the players who played in one of its last 10 games
and haven't played for another team since. For each player on it, let
$g$ be the number of the team's games since they last played and $p$
the number of the last 10 they played in. The chance they play tonight
is the share of players in the same spot $(g, p)$ who played their
team's next game, over the previous three seasons:
$$
\Pr(\text{plays}) = \frac{\text{played} + \tfrac12}{\text{in that spot} + 1} .
$$
:::

The $\tfrac12$ and $1$ are one last application of @thm:shrinkage: a
spot seen rarely is pulled toward even odds. @fig:availability shows the
table as of the 2026 Finals. A player who has played all of their team's
last 10 games plays the next one 94% of the time. Missing just the last
game, with 9 of 10 played, drops it to 49%: a player who sat last night
is usually still hurt. And a player out for a week or more is a long
shot.

![The chance a player on a team's roster plays its next game, by the
number of team games since they last played, for four rates of recent
appearance. Fit on the 2023–2025 seasons, as of the 2026
Finals.](figures/availability.svg){#fig:availability}

A player's expected minutes are then their chance of playing times their
minutes when playing, and the team's are scaled to add to 240, with no
player above 42; minutes over the cap go to the others in proportion.

This is the weakest part of the model, and the scoreboard shows it.
When the harness is allowed to know who actually played, standing in
for an injury report, the margin RMSE on 2021–2024 is 13.58; guessing,
as it must for real predictions, it's 13.71. Knowing who plays is worth
more than the box-score prior.

## How many possessions

Pace is modeled the way @ch:question suggested, as a property of the
game that both teams contribute to, and it's fit on team games rather
than segments.

::: {.definition #def:tempo title="Adjusted tempo"}
For a game between teams $A$ and $B$ with $P$ possessions in 48 minutes,
$$
\log P = c + \pi_P + t_A + t_B + \varepsilon,
$$
where $c$ is the league's tempo, $\pi_P$ a playoff term and $t_A$, $t_B$
the teams' effects, each pulled toward zero by a small ridge penalty, and
games older than about two months count half (a half life of 60 days).
The predicted possessions are $e^{c + \pi_P + t_A + t_B}$, times the
league's average game length in units of 48 minutes, to allow for
overtime.
:::

The logarithm makes the effects multiply: a team 3% faster than average
playing a team 2% slower gives a game about 1% faster than average. In
June 2026 the league's tempo was 100.0 possessions per 48 minutes, and
the playoff term was $-2.7$%: playoff games are slower.

## Game 5, predicted

Here is everything at once, for the game in @fig:timeline: San Antonio at
home against New York, fit on every game through June 12.

| | Spurs | Knicks |
|:--|--:|--:|
| league level, $\ell$ | 105.38 | 105.38 |
| playoff term, $\pi$ | $-3.93$ | $-3.93$ |
| home court, $\pm\eta$ | $+0.93$ | $-0.93$ |
| own offense, $O$ | $+14.22$ | $+13.73$ |
| opponent's defense, $-D$ | $-0.84$ | $-5.06$ |
| points per 100 possessions | 115.76 | 109.19 |
| expected possessions, $P$ | 96.2 | 96.2 |
| **expected points** | **111.3** | **105.0** |
| actual points | 90 | 94 |

Table: The model's prediction of Game 5 of the 2026 Finals, made from
games before June 13, 2026, built up term by term from
@eq:expected-score. {#tbl:game5}

The Spurs' offense rates slightly better than the Knicks', and their
defense much better: Wembanyama alone, projected for 37 minutes with a
defensive rating of $+6.3$, contributes $5 \times \tfrac{37.1}{240} \times
6.3 \approx 4.9$ points per 100 to it. With home court, the model made
San Antonio a 6.3-point favorite, and a 68% chance to win (@ch:evaluation
explains the conversion).

::: {.remark #rem:terms}
The individual terms in @tbl:game5 are less stable than their sum. While
this chapter was being written, nba_data added four earlier seasons of
play-by-play, and the box-score prior learned from them. Refit, the
league level for this game fell from 107.8 to 105.4, while both teams'
offensive strengths rose by about 0.6 and their defensive strengths fell
by about 1.8, which together make up the 2.4. The predicted scores
changed by a hundredth of a point. This is @prop:singular at work:
shifting the ratings one way and the level the other predicts the same
scores, and the data only pins down the combination. Here the prior,
not the data, decided where the level sits.
:::

The Knicks won by four. The game was slower than predicted, about 90
possessions to 96, and both offenses far worse: the Spurs scored 100 per
100 possessions against a prediction of 116. A 68% favorite loses about
one time in three, and this was one of those times. The question
@ch:evaluation takes up is whether, over thousands of games, the
favorites lose as often as the model says they should, and no more.

The projected minutes are worth a look too. For the Knicks, Brunson was
projected for 36 minutes and played 41. OG Anunoby, who had missed two
games of the second round in May and so had played 8 of the Knicks' last
10, was given an 87% chance of playing rather than 94%, and played 33
minutes. In
a Finals elimination game, starters play more than their season pattern
says, which the model doesn't know. Playoff rotations are one of the
open experiments in the project.

::: code
The expected score, @eq:expected-score, is
[`PlayerRatings.predict`](code:yarapm/players.py#L229), and team
strength is [`strength`](code:yarapm/players.py#L214), which takes
each team's projected minutes and normalizes them, so it works with any
total. Minutes are [`ProjectedMinutes`](code:yarapm/minutes.py#L97):
minutes when playing are one SQL query with a window function numbering
each player's games ([`when_playing`](code:yarapm/minutes.py#L120)), the
roster and the $(g, p)$ spots come from
[`rosters`](code:yarapm/minutes.py#L230) and
[`availability`](code:yarapm/minutes.py#L191), and
[`allocate`](code:yarapm/minutes.py#L55) scales a team's minutes to 240
under the cap, iterating because capping one player changes everyone
else's share. Tempo is [`Tempo`](code:yarapm/tempo.py#L21), a small dense
weighted ridge regression with one column per team.
:::

## Exercises {.unnumbered}

::: {.exercise #exr:strength}
A team plays five players 48 minutes each, with offensive ratings $+4$,
$+2$, $+1$, $0$ and $-1$. What is $O$? If the second player is out and a
player rated $-2$ takes their minutes, how many points per game does the
team's expected margin fall, at 100 possessions?
:::

::: {.exercise #exr:wemby}
Using @tbl:game5, how much of the Spurs' 6.3-point expected margin came
from home court? From the difference in defense? If Wembanyama had been
ruled out that morning, and the 37 projected minutes had gone to players
with an average defensive rating of 0 and offensive rating of $+1$, what
would the predicted margin have been?
:::

::: {.exercise #exr:tempo-multiply}
Show that under @def:tempo, the predicted possessions of $A$ against $B$
are the league's tempo times $(1 + a)(1 + b)$ approximately, where $a$
and $b$ are the teams' percentages above average, written as fractions.
What's the error of the approximation for $a = b = 0.05$?
:::

::: {.exercise #exr:minutes-query}
`yarapm.minutes` has each player's projected minutes and chance of
playing for every upcoming game. Pick tonight's games and find the
players with the lowest `play_prob` who are still projected for more
than 10 minutes. What does that say about the model's view of them?
:::
