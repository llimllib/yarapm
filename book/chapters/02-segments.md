---
title: Lineup Segments
label: ch:segments
---

The box score of the Finals clincher tells us that Jalen Brunson scored
45 points in 41 minutes, and that the Knicks outscored the Spurs by 10
with Brunson on the floor. Since they won by 4, they lost the 7 minutes
Brunson sat by 6. That plus-minus is a start, but it can't say how much of the
+10 was Brunson's doing and how much was the teammates', or the Spurs'
players who happened to be on the floor at the same time. To separate them we
need something finer: the stretches of a game during which the same ten
players were on the floor, and what each team scored in them. This
chapter builds those stretches from the NBA's play-by-play.

## Who was on the floor

nba_data gives us two tables for every game. The *rotations* list each
player's stints, as the times they checked in and out. The *events* are the
play-by-play: every shot, free throw, rebound, turnover, foul and
substitution, with the clock time and the team and player involved.

::: {.definition #def:segment title="Segment"}
Sort the times at which any player checks in or out of a game. A
*segment* is the stretch between two consecutive times: no one enters or
leaves during it, so the same five players are on the floor for each
team throughout. A segment gives two rows of data, one for each team's
offense: its five players, the five defenders, and its possessions $n_s$
and points $y_s$ in the segment.
:::

![Game 5 of the 2026 Finals, New York at San Antonio. Top: the Knicks'
margin at the end of each segment; ticks under it mark the segment
boundaries. Below: when each player was on the floor, Knicks shaded and
Spurs hatched, in order of minutes played. Every boundary is a time
someone checked in or out.](figures/timeline.svg){#fig:timeline}

@fig:timeline shows the Finals clincher from @ch:question cut this way.
San Antonio led by 10 after the first quarter and by 16 early in the
second. The Knicks got within 5 by halftime, trailed by 7 after three
quarters, tied the game with four and a half minutes left and won by
four. Every one of the 49 tick marks under the margin is a substitution,
and so a segment boundary. Near the end they come in a burst: the last 26
seconds were six segments, as both coaches swapped offense for defense at
every stoppage.

| | time | offense | lineup | poss. | pts. |
|-:|:---------|:----|:--------------------------------|--:|--:|
| 1 | 0:00–4:11 | NYK | Brunson, Bridges, Hart, Anunoby, Towns | 7 | 5 |
| | | SAS | Fox, Castle, Vassell, Champagnie, Wembanyama | 7 | 6 |
| 2 | 4:11–6:09 | NYK | the same five | 4 | 3 |
| | | SAS | Castle, Vassell, Harper, Johnson, Wembanyama | 3 | 5 |
| 3 | 6:09–7:09 | NYK | Shamet for Anunoby | 2 | 0 |
| | | SAS | Fox, Vassell, Harper, Johnson, Kornet | 3 | 2 |

Table: The first three segments of the game, with times from the
opening tip. Each segment is two rows, one for each team's offense. The
Spurs changed two players at 4:11 and two more at 6:09, so a segment
can be as short as a minute. {#tbl:first-segments}

## The substitution problem

Placing an event in a segment is easy, except at the boundaries. The
play-by-play records time to a tenth of a second, and substitutions
happen during stoppages, at the same clock time as the foul or timeout
that stopped play. So which lineup gets the free throws after a
foul, if a substitution is recorded on the same tick?

The NBA answers this for its own plus-minus statistic, and we follow it,
because it lets us check our work against the official numbers
(@sec:checks).

::: {.definition #def:placement title="Placing events on a substitution's tick"}
An event at the same clock time as a substitution goes to the lineup
going out if it's listed before the substitution in the play-by-play,
and to the lineup coming in if it's listed after. The exception is free
throws: they go to the lineup that was on the floor for the foul, the
one going out, unless a new foul comes after the substitution.
:::

The free throw rule matters more than it looks. Late in close games,
teams foul on purpose and substitute between free throws, so the last
minute of a close game is full of ticks with both a substitution and a
free throw. Getting them wrong would credit points to players who were on
the bench.

## Counting possessions

The other thing a segment needs is its number of possessions, the $n_s$
that will turn its points into a rate.

::: {.definition #def:possession-count title="Possessions in a segment"}
A team's possession starts with its first shot, free throw, turnover or
rebound after the other team's, so an offensive rebound continues a
possession rather than starting one. A possession is credited to the
segment of its first shot, free throw or turnover, or of its rebound if
it has none. When a period ends on a make, a turnover or the last of a
trip's made free throws, the other team is credited with a possession at
the end of the period. Technical free throws neither start nor continue
a possession.
:::

Why credit the first *play*, and not the rebound that started the
possession? Because coaches substitute in exactly that gap: after a
defensive rebound, before the next shot. The lineup that runs the
possession is the one that should be credited with it.^[This is a
modeling choice, not a fact about basketball. One could split a
possession between the two lineups, or credit it to the rebounding
lineup. Crediting the first play matches what the offense did with the
ball, and the possession counts this gives come within 3 of the NBA's
own counts for each team in nearly every game.]

![How many possessions an offense has in a segment, 2026 regular season
and playoffs. Half the segments have two possessions or fewer, and 7.7%
have none: points scored in those are free throws on a substitution's
tick.](figures/segment-possessions.svg){#fig:segment-possessions}

Segments are short. In 2026 a game averaged 32 of them, the median one
lasted 71 seconds, and @fig:segment-possessions shows that half of them
gave an offense two possessions or fewer. Some have none at all, but can
still have points: a player fouled before a substitution shoots their
free throws after it, and by @def:placement they belong to the segment that
began with the substitution, which has had no possession of its own.
@ch:ridge deals with these.

## Checking the work {#sec:checks}

A pipeline like this is easy to get subtly wrong, so every game is
checked against three totals the NBA publishes, and dropped if it fails
any of them.

1. Each team's points over its segments must equal its box score points.
2. Each team's possessions must be within 3 of the NBA's count.
3. Each player's plus-minus, the margin over the segments they were on
   the floor for, must be within 2 of the official one.

The third check is the strong one: it tests the placement of every
point of every game, player by player. In 2025–26, 96% of games match
every player's plus-minus exactly. The rest are off by a point or two
from free throws on a substitution's tick that the NBA's own numbers
give to different lineups for different players, which no division into
segments can match. Of 1,322 games in 2026, 1,321 passed; the
other was dropped because its rotation data didn't match its box score.

::: code
The segments are built in SQL, one season at a time, in
[`yarapm/segments.py`](code:yarapm/segments.py#L101). Segment boundaries
are the distinct check-in and check-out times
([`bounds`](code:yarapm/segments.py#L109)), and each segment's lineups
are the players whose stints cover it. @def:placement becomes a boolean,
[`after`](code:yarapm/segments.py#L134), computed with window functions
over each clock tick: an event is placed by nudging its time a hundredth
of a second forward or back, into the segment on that side. Possessions
are runs of consecutive events by the same team
([`possession_credits`](code:yarapm/segments.py#L266)), and
[`checks`](code:yarapm/segments.py#L319) computes the three tests. A
season takes a couple of seconds, and the result is one Parquet file per
season under `out/segments/`, about 80,000 rows each.
:::

## What we have

A season of segments is about 80,000 rows: 85,194 in 2026. Each row is a
small experiment. Five offensive players and five defenders shared the
floor for a few possessions, and the offense scored some number of
points. In 2026 those rows involved 19,985 different five-man units, so
almost no lineup appears often enough to be rated on its own. What does
appear often is each *player*, in many different lineups. The next
chapter turns that into a regression.

## Exercises {.unnumbered}

::: {.exercise #exr:segment-rate}
In segment 1 of @tbl:first-segments, what were each team's points per 100
possessions? Suppose the points a team scores on one possession are 0, 1,
2 or 3 with probabilities 0.50, 0.05, 0.30 and 0.15. What are the mean and
standard deviation of points per possession, and of points per 100
possessions over a 7-possession segment? What does this say about
reading anything into one segment?
:::

::: {.exercise #exr:plus-minus}
Show that the plus-minus of a team's players in a game, summed over all
of them, is five times the team's margin. Is the same true of the sums of
their segment margins, whatever the substitution rules?
:::

::: {.exercise #exr:credit}
A team rebounds a miss, substitutes during the timeout that follows, and
scores on the first shot after it. Which segment gets the possession, and
which gets the points? What would go wrong in the ratings if possessions
were credited to the segment of the rebound instead?
:::

::: {.exercise #exr:starters}
Using `nba.game_rotations` (columns `season`, `game_id`, `team_id`,
`player_id`, `in_time`), find how many different starting lineups each
team used in 2026. Starters are the players with `in_time = 0`. Which
team used the most?
:::
