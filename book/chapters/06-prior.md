---
title: The Box-Score Prior
label: ch:prior
---

The ridge penalty pulls every rating toward a prior, and @ch:ridge used
the laziest prior there is: everyone is average. That's right for a player
nobody has seen, and badly wrong for Nikola Jokić, whose 2026 rating it
cut from $+14.0$ to $+5.2$. Basketball has better information than that
about every player who has played: the box score. Points, assists,
rebounds, steals and blocks don't measure what a player does for the
team's scoring margin, but they're correlated with it, and they settle
much faster than plus-minus does. This chapter turns them into a prior.

## Rates, steadied

Each player's box scores are turned into rates per 100 of their
possessions on the floor: points, two- and three-point attempts, free
throw attempts, assists, turnovers, offensive and defensive rebounds,
steals, blocks and fouls. Older games count less, with the same one-year
half life as the ratings.

A rate from a few games is noisy, and how noisy depends on the stat. A
player's rebounding rate settles in a few hundred possessions, but steals
are rare events, and two lucky games can double a rate. So each rate is
first shrunk toward the average of players of similar height, by an
amount that depends on how fast that stat settles.

::: {.definition #def:stabilize title="A steadied rate"}
For a player with count $c$ of some stat in $n$ possessions, and $\bar
c$ the stat's rate per possession among players in the same height
group, the steadied rate is
$$
\frac{c + k\,\bar c}{n + k} ,
$$
where $k$, in possessions, is how much evidence it takes to move the rate
halfway from the group's to the player's own.
:::

This is @thm:shrinkage again, with the group average as the prior and
$k$ as the penalty.^[Shrinking toward a group average like this is
called *empirical Bayes*: the prior is estimated from the data itself.
The classic example, Efron and Morris's 1975 study of batting averages,
is about baseball.] Height stands in for position, which the data
doesn't have: the groups are under 6′4″, 6′4″ to 6′7″, 6′7″ to 6′10″ and
taller, roughly guards, wings, forwards and centers. The values of $k$
run from 150 possessions for rebounds to 600 for steals. They're rough
guesses, not tuned, and they matter little, since the formula below is
fit on the steadied rates and adjusts to whatever they are.

To the eleven rates the prior adds seven more inputs: true shooting
percentage (points per shooting attempt, counting free throws), three-point
percentage, usage times true shooting above average (how much a player
shoots, times how well), minutes per game, height, the logarithm of the
player's draft pick, and whether they were drafted at all. Eighteen
numbers per player.

## A formula fit to plus-minus

::: {.definition #def:box-formula title="The box formula"}
Fit plain RAPM, with $\lambda = 3{,}000$ and no prior, to every season
of segments *before* the one being predicted, with no decay. Then fit two
linear functions of the eighteen standardized inputs, one to the players'
offensive ratings and one to their defensive ratings, by ridge
regression weighted by each player's possessions. These are the *box
formulas*; a player's prior is the formula applied to their current
inputs.
:::

The formulas are refit once a season, on opening night. Fitting only to
earlier seasons is essential: a formula fit on this season's ratings
would have seen the games it predicts, and the walk-forward testing of
@sec:honesty would flatter it. For 2026, the formula learns from RAPM on
the fourteen seasons 2012–2025, about 1,750 players.

![The 2026 box formulas' coefficients, in points per 100 possessions per
standard deviation of each input: (a) for the offensive rating, (b) for
the defensive rating. Shaded bars are positive, hatched negative.
Inputs are correlated, so read them together, not one at a
time.](figures/box-formula.svg){#fig:box-formula}

@fig:box-formula shows what the 2026 formula learned, and it reads like a
scouting report. Offense goes up with points, assists, offensive
rebounds and high-usage efficient shooting, and down with turnovers.
Defense goes up with steals, blocks and defensive rebounds, and with
height. Points *lower* the defensive prior by 2.6 per standard
deviation: other things equal, a high scorer is a worse defender, either
because offense is where their energy goes or because good defenders who
also score get credit for it on offense instead.

Not every bar can be read on its own. True shooting has a negative
coefficient on offense, which looks wrong until you notice that usage
times true shooting has a large positive one: the two together say that
efficient shooting is worth a lot for players who shoot often, and much
less for those who rarely do. Correlated inputs share credit in ways that
make single coefficients misleading, the same problem @ch:regression had
with Braun and Jokić, on a smaller scale.

## Rookies

A player with no box scores still needs a prior. For them there's a
second, smaller formula.

::: {.definition #def:rookie-formula title="The rookie formula"}
Fit the offensive and defensive priors by ridge regression on height,
the log of the draft pick and whether the player was undrafted alone,
using the box formula's estimate of each earlier season's rookies, from
2011 on, weighted by their possessions.
:::

It learns what a scout would expect: in 2026 its defensive prior rises by
0.38 points per 100 for each standard deviation of height (3.3 inches),
and its offensive prior falls by 0.31 for each standard deviation of the
log of the draft pick. On opening night of 2024, Victor Wembanyama, a
7′4″ first overall pick, had a prior of $+0.6$ on offense and $+0.9$ on
defense, $+1.5$ in total: about as good a prior as the formula gives
anyone, and far from where Wembanyama would end up.

A player's prior blends the two formulas by how much the box formula
has to go on: with $n$ decayed possessions, the prior is $n/(n + 500)$
of the box formula and the rest of the rookie formula. So a rookie's
prior moves from the rookie formula toward their box scores over their
first few games, rather than jumping on the strength of one.

![Victor Wembanyama's rookie season, 2024: total rating (solid) and
box-score prior (dashed), as of each day. Early on the rating is the
prior, give or take; as possessions accumulate, it pulls away. Wembanyama
ended the regular season with 4,400 possessions, a prior of $+2.5$ and a
rating of $+4.3$.](figures/wembanyama.svg){#fig:wembanyama}

@fig:wembanyama follows Wembanyama through that season, and shows the
prior and the penalty working together. Through November, with a few
hundred possessions, the rating can't stray far from the prior, and both
dip as the early box scores came in modest. From January on, the prior
rises as the box scores improve, and the rating rises faster:
the plus-minus data saw something the box score didn't. By April the
data, about 4,400 possessions against the penalty's 3,000, has the larger
say (@thm:shrinkage), and the rating ends nearly two points above the
prior.

Two seasons later, the same model's final 2026 rating for Wembanyama was $+10.3$,
second in the league behind Jokić.

## What the prior is worth

The prior replaces $m = 0$ in @eq:ridge. On the 2021–2024 seasons with
actual minutes and daily refits, it improved the margin RMSE from 13.71
to 13.62 points, and the log loss from 0.620 to 0.613, one of the larger
improvements of any single piece of the model.
The prior can also be scaled, using $c\,m$ in place of $m$, and the
scale was tuned along with the penalty: anything from 1.0 to 1.3 did
about equally well, and yarapm uses 1.

The rookie numbers are a good test of the whole idea. In 2026, the first
overall pick, Cooper Flagg, started from a prior of $+1.0$ and finished
the season at $+1.3$. The second pick, Dylan Harper, whom we met coming
off the Spurs' bench in the Finals in @tbl:first-segments, started from
$+0.2$ and finished at $+5.1$. The prior was a sensible starting point
for both, and the games moved each where they belonged.

::: code
The prior is [`BoxPrior`](code:yarapm/prior.py#L196) in
`yarapm/prior.py`. Its [`train`](code:yarapm/prior.py#L234) method fits
the two formulas when the season changes, from
[`PlayerRatings.target`](code:yarapm/players.py#L195), the plain RAPM fit
over every earlier season. [`box_features`](code:yarapm/prior.py#L136)
computes the eighteen inputs, with the steadying constants in
[`RATES`](code:yarapm/prior.py#L42). Box scores go back to 2010, two
years further than play-by-play, so the rookie formula has more seasons
to learn from than the box formula does. Because the box formula learns
from every earlier season with segments, the ratings of every season
change when nba_data adds an earlier season of play-by-play, and yarapm
rebuilds its published history when it does (@ch:history).
:::

## Exercises {.unnumbered}

::: {.exercise #exr:steals}
A guard has 9 steals in 300 possessions; guards average 1.5 steals per
100 possessions; $k = 600$ for steals. What is the steadied rate, per 100
possessions? After how many possessions at the same raw rate would the
steadied rate reach 2.5?
:::

::: {.exercise #exr:leak}
Suppose the box formula were fit to RAPM of the season being predicted,
including games after the fit day. Explain why the walk-forward margin
RMSE would improve, and why the improvement would be fake.
:::

::: {.exercise #exr:usage}
In standardized units, the offensive formula has $-0.55$ for true
shooting and $+1.04$ for usage times true shooting above average.
Consider two players one standard deviation above average in true
shooting, one who shoots rarely and one who shoots often. Sketch why the
formula rates the second's efficiency higher. What would happen to the
two coefficients if usage times shooting were dropped?
:::

::: {.exercise #exr:rookie-prior}
From `yarapm.ratings_daily`, find the 2026 rookies (`nba.players` has a
`first_season` column) with at least 3,000 `possessions` by the end of
the season, and compare their final `o_prior + d_prior` with their final
`rating`. Which outplayed their box scores the most?
:::
