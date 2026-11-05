---
title: Regression on Lineups
label: ch:regression
---

We have about 80,000 small experiments a season: five players on offense,
five on defense, a few possessions, some points. We want one offensive and
one defensive number per player that explains them all at once. This is a
regression, and setting it up is straightforward. Solving it is not,
because basketball lineups are a poorly designed experiment. Coaches play
their best players together, and nobody randomizes who shares the floor.
This chapter sets up the regression, solves it, and shows exactly how and
why the answer goes wrong.

## The model

::: {.definition #def:rapm title="Adjusted plus-minus"}
For a segment's offense $s$, let $r_s = 100\,y_s / n_s$ be its points per
100 possessions. The *adjusted plus-minus* model is
$$
r_s = \mu + \sum_{i \in O_s} o_i - \sum_{j \in D_s} d_j + \varepsilon_s,
\qquad \mathbb{E}[\varepsilon_s] = 0, \quad \operatorname{Var}(\varepsilon_s) = \frac{\sigma^2}{n_s},
$$ {#eq:apm}
where $\mu$ is the league's average, $o_i$ is how many points per 100
possessions player $i$ adds on offense, and $d_j$ how many player $j$
takes away on defense.
:::

The defensive ratings enter with a minus sign so that a positive rating
is good on both ends, and a player's total rating is $o_i + d_i$. The
units are worth pausing on. A rating of $+5$ means that with this player
on the floor instead of an average one, their team scores, or stops, five
more points per 100 possessions: about five points a game, if they played
every minute. The variance assumption says that a segment's noise
shrinks with its possessions, as the noise of an average does.^[If the
points on each possession are independent with variance $\tau^2$, their
mean over $n$ possessions has variance $\tau^2/n$, so 100 times it has
variance $100^2\tau^2/n$; here $\sigma = 100\,\tau$. Exercise
@exr:segment-rate puts $\tau$ near 1.1 points, so $\sigma \approx 110$.]

Take the first row of @tbl:first-segments: the Knicks' opening five
scored 5 points on 7 possessions against the Spurs' starters, so
$$
\begin{aligned}
71.4 = \mu &+ o_\text{Brunson} + o_\text{Bridges} + o_\text{Hart} + o_\text{Anunoby} + o_\text{Towns} \\
&- d_\text{Fox} - d_\text{Castle} - d_\text{Vassell} - d_\text{Champagnie} - d_\text{Wembanyama} + \varepsilon_s .
\end{aligned}
$$
Seven possessions is nothing, and $\varepsilon_s$ is huge: its standard
deviation is around $110/\sqrt{7} \approx 42$ points per 100. But there
are 85,000 such equations in a season, and each player appears in
thousands of them, with different teammates and opponents each time.

## Weighted least squares

Stack the unknowns into one vector, $\beta = (\mu, o_1, \ldots, o_k, d_1,
\ldots, d_k)$ for $k$ players, and write each segment's offense as a
row $x_s \in \mathbb{R}^{1+2k}$: a 1 for the intercept, $+1$ in the
offensive column of each of its five offensive players, and $-1$ in the
defensive column of each of the five defenders. Every row has exactly
eleven nonzero entries. Then @eq:apm is $r_s = x_s^\top \beta +
\varepsilon_s$.

Since segments with more possessions are less noisy, we weight each one
by its possessions, and choose $\beta$ to minimize
$$
L(\beta) = \sum_s n_s \left( r_s - x_s^\top \beta \right)^2 = (r - X\beta)^\top W (r - X\beta),
$$ {#eq:wls}
where $X$ has the rows $x_s$, $r$ the rates, and $W = \operatorname{diag}(n_s)$.^[Weighting
by $n_s$ is exactly right when $\operatorname{Var}(\varepsilon_s) = \sigma^2/n_s$:
it's the same as dividing each equation by its standard deviation, which
makes the noise of every row equal, and then ordinary least squares is
the best linear unbiased estimator (the Gauss–Markov theorem). Rows with
$n_s = 0$ get no weight at all.] Notice what the weights do to the units:
$n_s r_s = 100\, y_s$, so a segment counts in proportion to its
possessions, and every possession of the season counts once.

$L$ is a quadratic in $\beta$, so its minimum is where its gradient
vanishes. Differentiating,
$$
\nabla L(\beta) = -2 X^\top W (r - X\beta) = 0,
$$
so the least squares estimate solves the *normal equations*
$$
X^\top W X\, \hat\beta = X^\top W r .
$$ {#eq:normal}

The matrix $A = X^\top W X$ has a concrete meaning, and it's worth
reading entry by entry. Its diagonal entry for $o_i$ is $\sum_s n_s
(x_s)_i^2$, the possessions player $i$ played on offense. The entry for
$o_i$ and $o_j$ is the possessions $i$ and $j$ played *together* on
offense. The entry for $o_i$ and $d_j$ is minus the possessions $i$
played on offense *against* $j$. So $A$ is the season's table of who
played with and against whom, counted in possessions, and $X^\top W r$
is the table of points scored by and against each player. In 2026 it
has 582 players, so $A$ is $1{,}165 \times 1{,}165$, built from 78,608
rows with at least one possession.

## The first problem: shifting ratings {#sec:shift}

::: {.proposition #prop:singular}
$A = X^\top W X$ is singular. Adding a constant $c$ to every offensive
rating and subtracting $5c$ from $\mu$ changes no prediction, and neither
does adding $c$ to every defensive rating and adding $5c$ to $\mu$.
:::

::: proof
Every row has exactly five offensive $+1$s and an intercept of 1, so for
$v = (-5, 1, \ldots, 1, 0, \ldots, 0)$, with ones in the offensive
columns, $x_s^\top v = -5 + 5 = 0$ for every $s$. Then $Xv = 0$, so $Av =
X^\top W X v = 0$. The same argument works for $(5, 0, \ldots, 0, 1,
\ldots, 1)$ and the five $-1$s of the defenders.
:::

So the data can't say what an "average" player is: only differences
between players are identified, and something extra has to fix the
level. That's easy to supply, and the next chapter's penalty does it
automatically. Numerically, the two zero eigenvalues are there: in 2026
the smallest two eigenvalues of $A$ are $-4 \times 10^{-13}$ and $2
\times 10^{-11}$, zero up to rounding.^[A refresher: for a symmetric
matrix $A$ there's an orthonormal basis of eigenvectors $v$, with
$Av = \lambda v$ and real $\lambda$. Here $A$ is also *positive
semidefinite*, $v^\top A v = \sum_s n_s (x_s^\top v)^2 \ge 0$, so every
$\lambda \ge 0$, and $A$ is invertible exactly when none is zero. See
[3Blue1Brown's video on eigenvectors](https://www.youtube.com/watch?v=PFDu9oVAE-g)
for the geometry.]

## The real problem: players who play together {#sec:collinear}

Shifting is a technicality. The real trouble is *nearly* zero eigenvalues,
and basketball produces them by design. In 2026, Christian Braun played
3,318 offensive possessions for Denver, and 89% of them were with Nikola
Jokić on the floor. Whatever Denver did in those possessions, the data
can attribute it to Jokić or to Braun almost interchangeably; only the
other 11% tell them apart.

::: {.example #ex:twins}
Suppose players 1 and 2 are always on the floor together, for $n$
possessions. Then the part of $A$ for their offensive ratings is
$$
\begin{pmatrix} n & n \\ n & n \end{pmatrix},
$$
with eigenvalues $2n$, for the direction $(1, 1)$, and $0$, for $(1,
-1)$. The data pins down $o_1 + o_2$ as well as it pins down any one
player, and says nothing about $o_1 - o_2$. If they share 89% of their
possessions instead, the second eigenvalue is no longer zero, but it's
small: it grows only with the possessions they *don't* share.
:::

How small the small eigenvalues are matters because they set how noisy the
estimate is.

::: {.proposition #prop:variance}
Under @eq:apm, the least squares estimate has
$\operatorname{Var}(\hat\beta) = \sigma^2 A^{-1}$ (when $A$ is invertible). In the
direction of a unit eigenvector $v$ of $A$ with eigenvalue $\lambda$,
$\operatorname{Var}(v^\top\hat\beta) = \sigma^2 / \lambda$.
:::

::: proof
$\hat\beta = A^{-1} X^\top W r$, and $\operatorname{Var}(r) = \sigma^2 W^{-1}$, so
$\operatorname{Var}(\hat\beta) = A^{-1} X^\top W (\sigma^2 W^{-1}) W X A^{-1}
= \sigma^2 A^{-1} A A^{-1} = \sigma^2 A^{-1}$. For a unit
eigenvector, $A^{-1} v = v/\lambda$, so $v^\top (\sigma^2 A^{-1}) v =
\sigma^2/\lambda$.
:::

The eigenvalues are measured in possessions. A direction with eigenvalue
$\lambda$ is estimated about as well as a rating from $\lambda$
possessions of data, with a standard deviation near $110/\sqrt{\lambda}$
points per 100. @fig:eigenvalues shows them for 2026. Apart from the two
zeros, they run from 3.8 to 302,415, a ratio of about 80,000, the
*condition number* of the problem.^[The condition number $\kappa =
\lambda_{\max}/\lambda_{\min}$ bounds how much a relative change in the
right-hand side can change the solution: $\|\delta\beta\|/\|\beta\| \le
\kappa\, \|\delta b\|/\|b\|$. With $\kappa$ near $10^5$, small noise in
the points can make large changes in the ratings.] There are 111
directions estimated from fewer than 100 possessions' worth of
information, and 514, nearly half, from fewer than 1,000.

![The eigenvalues of $A = X^\top W X$ for 2026, sorted, on a log
scale. Two are zero (@prop:singular). Nearly half are below 1,000
possessions, the scale on which one player's rating is reliable. The
dashed line is the penalty of @ch:ridge, which every player's eigenvalue
is raised by.](figures/eigenvalues.svg){#fig:eigenvalues}

## What least squares says

Fix the level by adding a negligible penalty, one possession's worth,
which by @ch:ridge's reasoning barely moves anything else, and solve
@eq:normal for 2026. @fig:least-squares plots each player's total rating
against their possessions.

![Each player's total rating $o_i + d_i$ from weighted least squares on
the 2026 season, against the offensive possessions they played. Filled
points at the edges are off the scale. The spread is enormous for players
who barely played, and narrows as possessions grow, roughly as $1/\sqrt{n}$.](figures/least-squares.svg){#fig:least-squares}

It's a funnel. Players with over 4,000 possessions have ratings with a
standard deviation of 5.6, and a basketball fan would recognize the top
of the list: Nikola Jokić is $+14.0$, Victor Wembanyama $+13.8$, Shai
Gilgeous-Alexander $+12.9$. But players with under 500 possessions have a
standard deviation of 25, and the extremes are absurd. Colby Jones, who
played 15 offensive possessions all season, is rated $+89.9$: taken at
face value, any lineup including Jones would outscore its opponents by 90
points per 100 possessions. Noa Essengue, 14 possessions, is $-116.8$. Nine players are
beyond $\pm 40$.

These aren't bugs. With 15 possessions, @prop:variance says the
standard deviation of a rating is around $110/\sqrt{15} \approx 28$, so
$\pm 90$ is a three-sigma result among hundreds of players who barely
played, which is about what we should expect. Least squares is doing
exactly what it promises, an unbiased estimate, and the estimate is
useless because it's unbiased. A player who played 15 possessions is
almost certainly close to average, and the data can't overrule that with
15 possessions. The next chapter writes "almost certainly close to
average" into the objective.

::: code
The design matrix is sparse: every row has eleven nonzero entries (plus
one per extra covariate, @ch:ridge), out of more than a thousand columns.
[`design()`](code:yarapm/rapm.py#L111) builds it in compressed sparse row
form directly from arrays of player indices, without a loop: the column
indices of row $s$ are its covariates, then its five offensive players,
then its five defenders offset by $k$. The players are numbered by
[`Lineups.from_ids`](code:yarapm/rapm.py#L50), which dictionary-encodes
the segments' lists of player ids with Arrow. Forming $A = X^\top W X$
is then a sparse matrix product ([line 165](code:yarapm/rapm.py#L165)).
:::

## Exercises {.unnumbered}

::: {.exercise #exr:row}
Write out the nonzero entries of the row $x_s$ for the Spurs' offense in
segment 1 of @tbl:first-segments, and its weight and target $r_s$.
:::

::: {.exercise #exr:psd}
Show that $A = X^\top W X$ is positive semidefinite, and that its
diagonal entry for $d_j$ is the possessions player $j$ played on
defense. What is the entry for $\mu$ and $\mu$?
:::

::: {.exercise #exr:overlap}
Two players share $m$ of their possessions; player 1 plays $a$ more
without player 2, and player 2 plays $b$ more without player 1.
Ignoring everyone else, the part of $A$ for their offensive ratings is
$$
\begin{pmatrix} m+a & m \\ m & m+b \end{pmatrix}.
$$
Find its smallest
eigenvalue when $a = b$. For Braun and Jokić, with $m = 2{,}963$, $a =
355$ and $b = 2{,}359$, how much information is there about the
difference between them?
:::

::: {.exercise #exr:rookie-variance}
Using @prop:variance, estimate the standard deviation of the least
squares rating of a player with 15 possessions, and of one with 5,000,
ignoring the overlap with teammates. Compare with the
spread in @fig:least-squares.
:::
