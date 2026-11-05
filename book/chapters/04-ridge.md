---
title: Ridge Regression
label: ch:ridge
---

Least squares believes the data completely, and @ch:regression showed
where that leads: a player with 15 possessions rated $+90$. What we want
is a rule that believes the data in proportion to how much of it there
is, trusting 5,000 possessions of Nikola Jokić and shrugging off 15 of
Colby Jones. The rule is one extra term in the objective, and this
chapter is about what that term does and how to choose it.

## A penalty

::: {.definition #def:ridge title="Ridge regression"}
For a penalty $\lambda > 0$, in possessions, and a vector $m$ of prior
ratings, the *ridge* estimate minimizes
$$
L_\lambda(\beta) = (r - X\beta)^\top W (r - X\beta) + \lambda \sum_i \left[(o_i - m^o_i)^2 + (d_i - m^d_i)^2\right].
$$ {#eq:ridge}
:::

The penalty charges for every point of rating a player is given away
from their prior. For now take the prior to be zero, an average player;
@ch:prior replaces it with a guess from box-score statistics. The
intercept isn't penalized: nothing pulls the league's average toward
zero.

The effect is clearest with one player. Suppose everyone else's
ratings are known, so the only unknown is player $i$'s offensive rating
$o$, and let $\bar r$ be the possession-weighted average of what's left of
the segments' rates after everyone else is accounted for, over the $n$
possessions player $i$ played on offense.

::: {.theorem #thm:shrinkage title="Shrinkage"}
With one unknown rating, the ridge estimate is
$$
\hat o = \frac{n}{n + \lambda}\,\bar r + \frac{\lambda}{n + \lambda}\, m ,
$$ {#eq:shrinkage}
a weighted average of the data's estimate and the prior, weighted by
possessions: the data counts as $n$ possessions and the prior as
$\lambda$.
:::

::: proof
The objective is $\sum_s n_s (\rho_s - o)^2 + \lambda (o - m)^2$, where
$\rho_s$ is segment $s$'s residual rate. Its derivative is $-2\sum_s
n_s(\rho_s - o) + 2\lambda(o - m)$, which vanishes when $(n + \lambda) o
= \sum_s n_s \rho_s + \lambda m = n \bar r + \lambda m$.
:::

So $\lambda$ has units: it's the number of possessions of data that
it takes to move a rating halfway from the prior to what the data says.
@fig:shrinkage shows the weight on the data for three penalties. With
yarapm's $\lambda = 3{,}000$, Colby Jones's 15 possessions get weight
$15/3{,}015 = 0.005$, and Jokić's 5,322 get $0.64$.

![The weight @eq:shrinkage puts on the data, $n/(n+\lambda)$, against
possessions $n$, for three penalties. The dotted line is where data and
prior count equally, at $n = \lambda$.](figures/shrinkage.svg){#fig:shrinkage}

## What the penalty does to the matrix

With all the players at once, setting the gradient of @eq:ridge to zero
gives normal equations with one change:
$$
(X^\top W X + \lambda I)\,\hat\beta = X^\top W r + \lambda m ,
$$ {#eq:ridge-normal}
where the identity is zero in the intercept's position. Adding $\lambda
I$ adds $\lambda$ to every eigenvalue of the players' part of
$A$.^[If $Av = \alpha v$, then $(A + \lambda I) v = (\alpha + \lambda)
v$: same eigenvectors, eigenvalues shifted. A symmetric matrix with
every eigenvalue positive is *positive definite*, $v^\top A v > 0$ for
every $v \ne 0$, which makes it invertible and makes $L_\lambda$ a
strictly convex bowl with a single minimum.] That disposes of both
problems from @ch:regression at once. The two zero eigenvalues of
@prop:singular become $\lambda$, so the solution is unique: among all the
ways of shifting ratings, the penalty picks the one closest to the
prior. And the smallest eigenvalues, which @prop:variance said were the
noisy directions, rise from 3.8 to about 3,000, the dashed line in
@fig:eigenvalues. The condition number falls from 80,000 to about 100.

@fig:valley shows this for the pair from @sec:collinear. Least squares
sees a long valley: almost any split of Denver's success between Jokić
and Braun fits nearly as well, as long as the sum is right. The penalty
tilts the valley toward the origin and rounds it into a bowl, and the
minimum moves to where the data and the penalty balance. Jokić, who has
far more possessions without Braun than Braun has without him, keeps
more of his rating.

![Contours of the objective for Jokić's and Braun's 2026 ratings, the
rest held fixed, using the possessions they played together and apart.
(a) Least squares: a long valley, along which their ratings trade off.
(b) With $\lambda = 3{,}000$: a rounder bowl, and a minimum pulled toward
zero.](figures/valley.svg){#fig:valley}

## Ridge as a prior belief

There's a second way to read @eq:ridge, and it explains why the
penalty is a sensible thing to add rather than a trick.

::: {.proposition #prop:bayes title="Ridge is a posterior mode"}
Suppose the noise in @eq:apm is normal, and that before seeing the data
we believe each rating is normal around its prior with variance
$\tau^2$: $o_i \sim N(m^o_i, \tau^2)$. Then the most probable ratings
given the data are the ridge estimate with $\lambda = \sigma^2/\tau^2$.
:::

::: proof
The log of the posterior density is, up to constants, the log
likelihood plus the log prior:
$$
-\frac{1}{2\sigma^2} (r - X\beta)^\top W (r - X\beta) - \frac{1}{2\tau^2} \sum_i \left[(o_i - m^o_i)^2 + (d_i - m^d_i)^2\right],
$$
which is $-L_\lambda(\beta)/2\sigma^2$ for $\lambda = \sigma^2/\tau^2$.
Maximizing one minimizes the other.
:::

With $\sigma \approx 110$ points per 100 (@ch:regression) and $\lambda =
3{,}000$, this says that before any games, we believe a player's
offensive rating is within $\tau = 110/\sqrt{3000} \approx 2$ points per
100 of the prior, give or take one standard deviation, and the same for
defense. That's a strong belief, and it's roughly right: among the 112
players with at least 4,000 possessions in 2026, the published offensive
ratings have a standard deviation of 2.0 and the defensive ones 1.7.

## Choosing $\lambda$

A larger $\lambda$ trusts the prior more. Too small, and @fig:least-squares's
funnel comes back; too large, and every player looks average, including
the ones who aren't. Rather than reason about it, yarapm tries several
and keeps the one that predicts best, by the walk-forward testing of
@sec:honesty.

| | $\lambda = 1{,}000$ | $2{,}000$ | $3{,}000$ | $5{,}000$ | $10{,}000$ |
|:--|--:|--:|--:|--:|--:|
| half life 365 days | 13.749 | **13.702** | 13.717 | 13.787 | 13.962 |
| half life 730 days | 13.761 | 13.704 | 13.706 | 13.752 | 13.898 |

Table: Margin RMSE, in points, of RAPM with no box-score prior, refit
weekly and predicting the 2021–2024 seasons with actual minutes, by
penalty and by how fast old games are forgotten (@ch:time). It's flat
between 2,000 and 3,000 possessions and worse on either side. With the
box-score prior of @ch:prior the best is again near 3,000, which is the
default. {#tbl:ridge-grid}

@tbl:ridge-grid is typical of tuning results: a broad, shallow minimum.
The difference between 2,000 and 3,000 is a hundredth of a point,
far less than the noise in four seasons of games, so the choice within
that range doesn't matter much. What matters is the order of magnitude:
a few thousand possessions, about half of a starter's season.

@fig:ridge shows the 2026 ratings again, now with $\lambda = 3{,}000$.
The funnel is gone. Every player with under 500 possessions is within a
point or two of zero, and the spread grows with possessions, because
only players with a lot of data can earn a rating far from the prior.

![The ratings of @fig:least-squares, fit with $\lambda = 3{,}000$ on the
2026 season alone. Players who barely played are pinned near zero; the
ratings range from $-5.0$ to $+7.4$.](figures/ridge.svg){#fig:ridge}

But the penalty has a cost you can see at the top of the figure. Jokić's
rating falls from $+14.0$ to $+5.2$, Wembanyama's from $+13.8$ to $+7.4$.
Shrinking toward zero is right for Colby Jones and wrong for Nikola
Jokić, whom no one believed was average before the season started. The
cure is not a smaller penalty, which brings the funnel back, but more
data and a better prior: more seasons of games (@ch:time) and a starting
guess from each player's statistics (@ch:prior). The published 2026
rating for Jokić, with both, is $+11.4$.

## The other unknowns {#sec:covariates}

The ratings aren't the only unknowns. Each row also carries covariates,
with their own penalties:

- **An intercept**, the league's average, unpenalized.
- **A season offset** for each season in the fit, penalized by 20,000
  possessions toward zero, which lets the league's scoring level differ
  from season to season (@fig:league).
- **Home court**: $+1$ in the rows of the home team's offense and $-1$
  in the away team's, $0$ at a neutral site. Unpenalized. In the April
  2026 fits, it's 0.92 points per 100 possessions each way, so home court
  is worth about $2 \times 0.92 = 1.8$ points per 100, or close to two
  points a game.
- **Playoffs**: $1$ in playoff and play-in rows. Unpenalized, and
  $-3.65$ points per 100 in the same fits: playoff offense is less
  efficient, as the Finals clincher of @ch:question suggested.

One more detail. Segments with no possessions (@fig:segment-possessions)
get no weight, but they can contain points: free throws after a
substitution. Dropping them would lose two-thirds of a point a game. Instead their
points are spread over everything else, as a rate on the possessions
that were fit, and added to the intercept.

## Solving it {#sec:cg}

@eq:ridge-normal is a linear system with a symmetric positive definite
matrix, and it's big: a real fit, with eight years of games (@ch:time),
has about 1,300 players and 2,700 unknowns. yarapm solves it with
*conjugate gradients*. It never factors the matrix; it only multiplies
it by vectors.^[Conjugate gradients minimizes $f(\beta) = \tfrac12
\beta^\top A \beta - b^\top\beta$, whose gradient is $A\beta - b$, over a
growing sequence of subspaces, each step adding the direction $A$ times
the last. In exact arithmetic it's done in at most as many steps as there
are unknowns, and after $j$ steps its error has shrunk by at least
$2\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^j$, so a smaller
condition number $\kappa$ means fewer steps. Shewchuk's [*An
Introduction to the Conjugate Gradient Method Without the Agonizing
Pain*](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
is the classic explanation.] Two things make it fast. The penalty keeps
the condition number small, and the system is *preconditioned* by its
diagonal: dividing each equation by its diagonal entry puts players with
5,000 and 500 possessions on the same scale.

The other is that yarapm refits every day, and today's answer is very
close to yesterday's. Starting the solver from yesterday's ratings, a
*warm start*, cuts the work: in April 2026 a fit from scratch took 29
iterations, and the next days' warm-started fits took 25 and 20.

::: code
The solve is [`rapm.fit`](code:yarapm/rapm.py#L124). It forms $A$ as a
sparse matrix ([line 165](code:yarapm/rapm.py#L165)), preconditions with
its diagonal, and calls SciPy's `cg` with a relative tolerance of
$10^{-8}$ and the previous fit as the starting point. The zero-possession
points are added to the intercept at [line 183](code:yarapm/rapm.py#L183).

Most of a day's fit is forming $A$, not solving with it. On February 28,
2025, the fit had 566,721 rows and 2,692 unknowns, and $A$ was 17%
nonzero. Forming it took 0.6 seconds; each multiplication by it took
1.2 milliseconds, so 25 iterations cost 0.03 seconds. Two ways to cut the
0.6 seconds are open questions for the project. One never forms $A$,
computing $Av = X^\top (W (Xv))$ instead, at 11.6 milliseconds a product.
The other notices that a day's decay (@ch:time) multiplies every old row's
weight by the same factor, so today's $A$ is yesterday's times a
constant plus the new day's rows, and could be updated rather than
rebuilt.
:::

## Exercises {.unnumbered}

::: {.exercise #exr:shrink-numbers}
A rookie plays 1,200 offensive possessions, and the residual of the
segments they played in, after his teammates and opponents are accounted
for, averages $+6$ points per 100. With a prior of $-1$ and $\lambda =
3{,}000$, what is the ridge estimate? How many possessions would he need
for the estimate to reach $+4$?
:::

::: {.exercise #exr:ridge-normal}
Derive @eq:ridge-normal by differentiating @eq:ridge. Then show that the
substitution $\beta = m + \delta$ turns it into ridge regression toward
zero for $\delta$, with $r$ replaced by $r - Xm$.
:::

::: {.exercise #exr:twins-ridge}
For the twins of @ex:twins, who share all $n$ possessions, find the
ridge estimates of $o_1$ and $o_2$ with priors of zero, if their
combined least squares estimate $o_1 + o_2$ is $s$. How is $s$ split
between them, and why?
:::

::: {.exercise #exr:tau}
The published 2026 ratings (`yarapm.ratings_daily`, the last
`as_of_date` of season 2026) include players with every number of
possessions. Compute the standard deviation of `o_rating` for players
with at least 4,000 `possessions`. How does it compare with the $\tau
\approx 2$ implied by $\lambda = 3{,}000$? Why should the two differ?
:::
