# The book: what yarapm does, as a math textbook

A document on GitHub Pages (https://llimllib.github.io/yarapm/) that explains
the repository to a reader who has had calculus and linear algebra, in the
style of a math textbook.

## Reader and scope

- Assumes calculus and a linear algebra course. Anything past the basics
  (eigenvalues, condition numbers, positive definiteness, conjugate
  gradients, the normal distribution) gets a margin sidenote with a short
  refresher and a link
- The modeling, end to end. Engineering isn't given chapters, but it isn't
  avoided either: "In the code" asides say how the repo does each step and
  why, linking to the file and line at a pinned commit

## Look

Adapted from a page the user liked (schema-browser-performance.html): a
paper-coloured page with a shadow, Latin Modern type, justified paragraphs
with indents, a running head, small-caps chapter numbers, booktabs tables,
definitions with a dark-blue rule, boxed propositions, exercises. Figures
are SVG in ink on paper: hatching and dot patterns instead of colour, a few
warm greys, italic axis labels, panel labels like "(a) before".

## Build

- `book/chapters/NN-name.md`: pandoc Markdown, one file per chapter
- `book/style.css`: the stylesheet, with margin sidenotes
- `book/filters/environments.lua`: fenced divs (`::: definition`,
  `theorem`, `example`, `remark`, `exercise`, `code`) numbered per chapter
  ("Definition 3.2"), with cross-references
- `book/filters/sidenotes.lua`: footnotes as margin notes, inline on narrow
  screens
- Math: pandoc `--katex`
- `mise run book` writes `out/book/`: an index (title page and contents)
  and a page per chapter with a running head and previous/next links
- `.github/workflows/book.yml`: builds on pushes to `main` that touch
  `book/`, and deploys with `actions/deploy-pages`. CI builds it too

## Figures

`book/figures.py` writes `book/figures/*.svg` with a small SVG helper
(axes, ticks, lines, bars, patterns), from closed forms, the published
catalog, or a harness run on the tuning seasons. It prints every number the
text quotes. The SVGs are committed, so the build needs no network.

## Chapters

1. The question: expected score as possessions × points per possession, and
   why players rather than teams
2. Lineup segments
3. Regression on lineups: weighted least squares and why it's ill-posed
4. Ridge regression: shrinkage toward a prior, the Bayesian reading,
   conjugate gradients
5. Time: exponential decay, half-lives, the league level's drift
6. The box-score prior
7. From ratings to a score: team strength, minutes, availability, tempo,
   home court
8. Was it any good? Win probability, scoring rules, the walk-forward
   harness, the scoreboard
9. Ratings through time: the daily datasets, full-strength team ratings,
   model versions and backfills

Each chapter ends with exercises, some of them queries against the
published data.
