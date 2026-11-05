"""
The book's figures, as SVG in ink on paper, into book/figures/.

    mise run book-figures [name ...]

Each figure is a function registered with @figure. Some are closed forms,
some read nba_data's catalog or yarapm's published one, and some need the
segments in out/ (`yarapm.segments`) or a harness run. Every number the
text quotes from a figure is printed, so a rerun can be checked against
the text. The SVGs are committed: the book's build doesn't run this.
"""

import itertools
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGURES = ROOT / "book" / "figures"

INK = "#1a1a1a"
FAINT = "#6b6b6b"
PAPER = "#fbfaf5"
GREY = "#d9d6ca"
DARK_GREY = "#b9b5a6"
BAND = "#ebe8dd"
ACCENT = "#1f3a68"

PATTERNS = f"""<defs>
<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="6" height="6" fill="{PAPER}"/><line x1="0" y1="0" x2="0" y2="6" stroke="{INK}" stroke-width="1.4"/></pattern>
<pattern id="dots" width="5" height="5" patternUnits="userSpaceOnUse">
<rect width="5" height="5" fill="{PAPER}"/><circle cx="2.5" cy="2.5" r=".9" fill="{INK}"/></pattern>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 1 L 9 5 L 0 9 z" fill="{INK}"/></marker>
</defs>"""


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


class Svg:
    """An SVG drawing, `width` wide and `height` high in its own units"""

    def __init__(self, width: float, height: float, label: str):
        self.width = width
        self.height = height
        self.label = label
        self.parts: list[str] = []

    def add(self, part: str) -> None:
        self.parts.append(part)

    def line(
        self,
        x1,
        y1,
        x2,
        y2,
        stroke=INK,
        width=1.0,
        dash: str | None = None,
        arrow=False,
    ):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}" '
            f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
        )

    def rect(self, x, y, w, h, fill=PAPER, stroke=INK, width=1.0, rx=0):
        self.add(
            f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def circle(self, x, y, r, fill=INK, stroke="none"):
        self.add(
            f'<circle cx="{num(x)}" cy="{num(y)}" r="{r}" fill="{fill}" stroke="{stroke}"/>'
        )

    def text(
        self, x, y, text, size=11, anchor="start", italic=False, fill=INK, rotate=None
    ):
        style = ' font-style="italic"' if italic else ""
        transform = f' transform="rotate({rotate} {num(x)} {num(y)})"' if rotate else ""
        self.add(
            f'<text x="{num(x)}" y="{num(y)}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}"{style}{transform}>{text}</text>'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        stroke=INK,
        width=1.4,
        dash=None,
        fill="none",
    ):
        pts = " ".join(f"{num(x)},{num(y)}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"{extra}/>'
        )

    def path(self, d: str, stroke=INK, width=1.2, fill="none", arrow=False, dash=None):
        extra = ' marker-end="url(#arrow)"' if arrow else ""
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>'
        )

    def render(self) -> str:
        return (
            f'<svg viewBox="0 0 {num(self.width)} {num(self.height)}" width="100%" '
            f'role="img" aria-label="{esc(self.label)}" xmlns="http://www.w3.org/2000/svg">\n'
            + PATTERNS
            + "\n"
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


class Axes:
    """A plot area on an Svg: maps data to the drawing, and draws its axes"""

    def __init__(self, svg: Svg, x: float, y: float, w: float, h: float, xlim, ylim):
        self.svg = svg
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xlim, self.ylim = xlim, ylim

    def px(self, x: float) -> float:
        lo, hi = self.xlim
        return self.x + (x - lo) / (hi - lo) * self.w

    def py(self, y: float) -> float:
        lo, hi = self.ylim
        return self.y + self.h - (y - lo) / (hi - lo) * self.h

    def frame(self, xticks, yticks, xlabel="", ylabel="", xfmt=num, yfmt=num):
        s = self.svg
        s.line(self.x, self.y, self.x, self.y + self.h)
        s.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h)
        for t in xticks:
            x = self.px(t)
            s.line(x, self.y + self.h, x, self.y + self.h + 4)
            s.text(x, self.y + self.h + 16, xfmt(t), anchor="middle")
        for t in yticks:
            y = self.py(t)
            s.line(self.x - 4, y, self.x, y)
            s.text(self.x - 7, y + 4, yfmt(t), anchor="end")
        if xlabel:
            s.text(
                self.x + self.w / 2,
                self.y + self.h + 34,
                xlabel,
                italic=True,
                anchor="middle",
            )
        if ylabel:
            s.text(
                self.x - 40,
                self.y + self.h / 2,
                ylabel,
                italic=True,
                anchor="middle",
                rotate=-90,
            )

    def line(self, xs, ys, **kw):
        self.svg.polyline([(self.px(a), self.py(b)) for a, b in zip(xs, ys)], **kw)

    def points(self, xs, ys, r=2.6, fill=INK, stroke="none"):
        for a, b in zip(xs, ys):
            self.svg.circle(self.px(a), self.py(b), r, fill=fill, stroke=stroke)


REGISTRY: dict[str, Callable[[], Svg]] = {}


def figure(name: str):
    def register(fn: Callable[[], Svg]) -> Callable[[], Svg]:
        REGISTRY[name] = fn
        return fn

    return register


def connect():
    from yarapm.catalog import connect as nba

    return nba()


# chapter 1


@figure("pipeline")
def pipeline() -> Svg:
    """How the pieces fit: the book's chapters as a flow"""
    s = Svg(
        640,
        320,
        "Diagram of the model: segments and box scores feed player ratings, which with minutes and tempo give each team's expected score",
    )

    def box(x, y, w, h, title, sub, ch="", fill=PAPER):
        s.rect(x, y, w, h, fill=fill)
        s.text(x + w / 2, y + 19, title, size=12.5, anchor="middle")
        if sub:
            s.text(
                x + w / 2,
                y + 34,
                sub,
                size=10,
                anchor="middle",
                italic=True,
                fill=FAINT,
            )
        if ch:
            s.text(x + w, y + h + 12, ch, size=9.5, anchor="end", fill=FAINT)

    s.text(20, 18, "from nba_data", size=10.5, italic=True, fill=FAINT)
    box(20, 28, 140, 46, "play-by-play", "who was on, what happened")
    box(20, 104, 140, 46, "box scores", "per player, per game")
    box(20, 180, 140, 46, "game results", "scores, possessions")
    box(20, 256, 140, 46, "schedule", "the games to predict")

    box(
        210,
        28,
        140,
        46,
        "lineup segments",
        "ten players, a stretch",
        "ch. 2",
        fill=BAND,
    )
    box(210, 104, 140, 46, "box-score prior", "a rating from stats", "ch. 6")
    box(210, 180, 140, 46, "tempo", "possessions per game", "ch. 7")

    # the ratings: the heart of it, framed in hatching
    s.rect(398, 50, 114, 74, fill="url(#hatch)")
    s.rect(406, 58, 98, 58, fill=PAPER)
    s.text(455, 83, "player", size=13, anchor="middle")
    s.text(455, 100, "ratings", size=13, anchor="middle")
    s.text(512, 136, "ch. 3–5", size=9.5, anchor="end", fill=FAINT)
    box(398, 166, 114, 46, "minutes", "who plays, how long", "ch. 7")

    s.rect(555, 100, 75, 104, fill=GREY)
    s.text(592.5, 136, "expected", size=12.5, anchor="middle")
    s.text(592.5, 152, "score", size=12.5, anchor="middle")
    s.text(592.5, 172, "and a win", size=10.5, anchor="middle", italic=True)
    s.text(592.5, 185, "chance", size=10.5, anchor="middle", italic=True)
    s.text(630, 94, "ch. 7–8", size=9.5, anchor="end", fill=FAINT)

    s.line(160, 51, 207, 51, arrow=True)
    s.line(160, 127, 207, 127, arrow=True)
    s.line(160, 203, 207, 203, arrow=True)
    s.path("M 350 51 L 396 76", arrow=True)
    s.path("M 350 127 L 396 100", arrow=True)
    s.path("M 160 140 C 210 166, 320 164, 396 183", arrow=True)
    s.path("M 512 90 L 553 130", arrow=True)
    s.path("M 512 189 L 553 172", arrow=True)
    s.path("M 350 213 C 430 252, 560 250, 578 206", arrow=True)
    s.path("M 160 279 C 420 290, 610 280, 610 206", arrow=True)
    return s


@figure("league")
def league() -> Svg:
    """Possessions per game and points per 100 possessions, by season"""
    from yarapm.games import load_results

    con = connect()
    load_results(con, "season BETWEEN 2016 AND 2026")
    rows = con.execute("""
        SELECT season, avg((poss + opp_poss) / 2 * 48 / min),
            100 * sum(pts + opp_pts) / sum(poss + opp_poss)
        FROM results GROUP BY season ORDER BY season
    """).fetchall()
    seasons = [r[0] for r in rows]
    pace = [r[1] for r in rows]
    rtg = [r[2] for r in rows]
    for season, p, r in rows:
        print(
            f"league {season}: {p:.1f} possessions per 48 minutes, {r:.1f} points per 100"
        )

    s = Svg(
        640,
        230,
        "Two line charts by season, 2016 to 2026: possessions per 48 minutes rise from 97 to 100, points per 100 possessions from 106 to 115",
    )
    a = Axes(s, 60, 22, 240, 150, (2015.5, 2026.5), (95, 103))
    a.frame(
        [2016, 2018, 2020, 2022, 2024, 2026],
        [95, 97, 99, 101, 103],
        "season",
        "possessions per 48 min",
    )
    a.line(seasons, pace)
    a.points(seasons, pace)
    s.text(300, 14, "(a) pace", italic=True, anchor="end", size=12)
    b = Axes(s, 380, 22, 240, 150, (2015.5, 2026.5), (104, 116))
    b.frame(
        [2016, 2018, 2020, 2022, 2024, 2026],
        [104, 107, 110, 113, 116],
        "season",
        "points per 100 possessions",
    )
    b.line(seasons, rtg)
    b.points(seasons, rtg, fill=PAPER, stroke=INK)
    s.text(620, 14, "(b) efficiency", italic=True, anchor="end", size=12)
    return s


# chapter 2

FINALS = "0042500405"  # 2026 Finals, Game 5: New York 94, San Antonio 90


@figure("timeline")
def timeline() -> Svg:
    """Who was on the floor in Game 5 of the 2026 Finals, and the margin"""
    from yarapm.catalog import read

    con = connect()
    stints = con.execute(f"""
        SELECT t.nba_abbrev, p.name, r.in_time / 60, r.out_time / 60
        FROM nba.game_rotations r
        JOIN nba.players p USING (player_id)
        JOIN nba.team_seasons t ON t.season = 2026 AND t.team_id = r.team_id
        WHERE r.season = 2026 AND r.game_id = '{FINALS}'
    """).fetchall()
    read(con, "segments", ROOT / "out").create_view("seg")
    margin = con.execute(f"""
        SELECT end_time / 60, sum(CASE WHEN t.nba_abbrev = 'NYK' THEN pts ELSE -pts END)
            OVER (ORDER BY seg)
        FROM (SELECT seg, end_time, off_team_id, pts FROM seg
              WHERE season = 2026 AND game_id = '{FINALS}')
        JOIN nba.team_seasons t ON t.season = 2026 AND t.team_id = off_team_id
        ORDER BY seg
    """).fetchall()
    # one point per segment end, the margin after both teams' rows
    by_time: dict[float, float] = {}
    for t, m in margin:
        by_time[t] = m
    times = [0.0, *by_time]
    margins = [0.0, *by_time.values()]
    bounds = sorted({t for _, _, a, b in stints for t in (a, b)})
    print(f"timeline: {len(bounds) - 1} segments, final margin {margins[-1]:+.0f}")

    minutes: dict[tuple[str, str], float] = {}
    for team, name, a, b in stints:
        minutes[team, name] = minutes.get((team, name), 0) + b - a
    rows = {
        team: sorted(
            (n for t, n in minutes if t == team), key=lambda n: -minutes[team, n]
        )
        for team in ("NYK", "SAS")
    }
    s = Svg(
        640,
        470,
        "The 2026 Finals clincher: the Knicks' lead over the game, and when each player was on the floor",
    )
    lo = 5 * math.floor((min(margins) - 1) / 5)
    hi = 5 * math.ceil((max(margins) + 1) / 5)
    print(f"timeline: margin from {min(margins):+.0f} to {max(margins):+.0f}")
    m = Axes(s, 130, 20, 490, 90, (0, 48), (lo, hi))
    s.text(20, m.py(hi) + 10, "NYK ahead", size=10, italic=True, fill=FAINT)
    s.text(20, m.py(lo), "SAS ahead", size=10, italic=True, fill=FAINT)
    s.line(m.x, m.py(0), m.x + m.w, m.py(0), stroke=FAINT, width=0.6)
    for q in (12, 24, 36):
        s.line(m.px(q), 14, m.px(q), 452, stroke=FAINT, width=0.6, dash="2 3")
    # the margin is a step function: it changes within segments, but we
    # only know it at their ends
    m.line(times, margins, width=1.3)
    for tick in range(lo, hi + 1, 5):
        s.line(m.x - 4, m.py(tick), m.x, m.py(tick))
        s.text(
            m.x - 7,
            m.py(tick) + 4,
            f"{tick:+d}" if tick else "0",
            anchor="end",
            size=10,
        )

    y = 132
    row = 14.5
    for team, fill in (("NYK", DARK_GREY), ("SAS", "url(#hatch)")):
        s.text(20, y + 9, team, size=11, italic=True)
        for name in rows[team]:
            last = name.split(" ", 1)[-1]
            s.text(m.x - 8, y + 10, esc(last), size=10, anchor="end")
            for t, n, a, b in stints:
                if t == team and n == name:
                    s.rect(
                        m.px(a), y + 2, m.px(b) - m.px(a), row - 4, fill=fill, width=0.6
                    )
            y += row
        y += 10
    for q, label in ((6, "1st"), (18, "2nd"), (30, "3rd"), (42, "4th")):
        s.text(
            m.px(q),
            464,
            f"{label} quarter",
            size=10,
            anchor="middle",
            italic=True,
            fill=FAINT,
        )
    for b in bounds:
        s.line(m.px(b), 124, m.px(b), 128, stroke=INK, width=0.6)
    return s


@figure("segment-possessions")
def segment_possessions() -> Svg:
    """How many possessions an offense has in a segment, 2026"""
    from yarapm.catalog import read

    con = connect()
    read(con, "segments", ROOT / "out").create_view("seg")
    rows = con.execute("""
        SELECT least(poss, 12), count(*) FROM seg WHERE season = 2026
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    total = sum(n for _, n in rows)
    for p, n in rows:
        print(f"segment possessions {p}{'+' if p == 12 else ''}: {n / total:.1%}")
    s = Svg(
        640,
        200,
        "Bar chart: most segments are short; a quarter have one possession, and few have more than six",
    )
    a = Axes(s, 70, 15, 540, 140, (-0.6, 12.6), (0, 0.25))
    a.frame(
        [],
        [0, 0.05, 0.10, 0.15, 0.20, 0.25],
        "possessions in the segment, for one offense",
        "share of segments",
        yfmt=lambda v: f"{v:.0%}",
    )
    for p, n in rows:
        x0 = a.px(p - 0.4)
        s.rect(
            x0,
            a.py(n / total),
            a.px(p + 0.4) - x0,
            a.py(0) - a.py(n / total),
            fill=GREY,
            width=0.8,
        )
        s.text(a.px(p), a.py(0) + 16, f"{p}+" if p == 12 else str(p), anchor="middle")
    return s


# chapters 3 and 4

_FITS: dict = {}


def season_fits(season: int = 2026):
    """
    Plain RAPM on one season's segments, intercept only, with a ridge of 1
    possession (nearly least squares) and of 3000 (yarapm's): each
    player's total rating and possessions, by fit
    """
    if season in _FITS:
        return _FITS[season]
    import numpy as np

    from yarapm import rapm
    from yarapm.catalog import read

    con = connect()
    read(con, "segments", ROOT / "out").create_view("seg")
    t = con.sql(
        f"SELECT off_player_ids, def_player_ids, poss, pts FROM seg WHERE season = {season}"
    ).to_arrow_table()
    rows = rapm.Lineups.from_ids(
        t["off_player_ids"].combine_chunks(),
        t["def_player_ids"].combine_chunks(),
        t["poss"].to_numpy(),
        t["pts"].to_numpy(),
        np.ones(len(t)),
        np.ones((len(t), 1)),
    )
    k = len(rows.players)
    poss = np.zeros(k)
    np.add.at(poss, rows.off.ravel(), np.repeat(rows.poss, 5))
    has = rows.poss > 0
    sub = rapm.Lineups(
        rows.players,
        rows.off[has],
        rows.dfn[has],
        rows.poss[has],
        rows.pts[has],
        rows.weight[has],
        rows.covariates[has],
    )
    X = rapm.design(sub)
    w = sub.poss
    y = 100 * sub.pts / sub.poss
    A = (X.T @ X.multiply(w[:, None])).toarray()
    b = X.T @ (w * y)
    out = {}
    for ridge in (1.0, 3000.0):
        penalty = np.r_[1e-9, np.full(2 * k, ridge)]
        x = np.linalg.solve(A + np.diag(penalty), b)
        out[ridge] = x[1 : 1 + k] + x[1 + k :]
    _FITS[season] = (rows.players, poss, out)
    return _FITS[season]


def rating_scatter(svg: Svg, a: Axes, poss, rating, label: str):
    a.frame(
        [0, 2000, 4000, 6000, 8000],
        [-40, -20, 0, 20, 40],
        "offensive possessions played",
        "rating, points per 100",
        xfmt=lambda v: f"{v:,.0f}",
        yfmt=lambda v: f"{v:+.0f}" if v else "0",
    )
    svg.line(a.x, a.py(0), a.x + a.w, a.py(0), stroke=FAINT, width=0.6)
    lo, hi = a.ylim
    for p, r in zip(poss, rating):
        if lo <= r <= hi:
            svg.circle(a.px(p), a.py(r), 1.8, fill="none", stroke=INK)
        else:
            y = a.py(hi) + 3 if r > hi else a.py(lo) - 3
            svg.circle(a.px(p), y, 1.8, fill=INK)
    svg.text(a.x + a.w, a.y + 4, label, italic=True, anchor="end", size=12)


@figure("least-squares")
def least_squares() -> Svg:
    """Nearly unpenalized ratings against possessions, 2026"""
    import numpy as np

    players, poss, fits = season_fits()
    r = fits[1.0]
    few, many = poss < 500, poss >= 4000
    print(
        f"least squares: {len(players)} players; sd of rating {np.std(r[few]):.1f} under 500 possessions, {np.std(r[many]):.1f} over 4000"
    )
    print(
        f"least squares: {(np.abs(r) > 40).sum()} players beyond ±40, max {r.max():+.1f}, min {r.min():+.1f}"
    )
    s = Svg(
        640,
        260,
        "Scatter plot: nearly unpenalized ratings fan out wildly for players with few possessions",
    )
    a = Axes(s, 70, 15, 540, 200, (0, 8000), (-40, 40))
    rating_scatter(s, a, poss, r, "λ = 1 possession")
    return s


@figure("eigenvalues")
def eigenvalues() -> Svg:
    """The eigenvalues of XᵀWX for 2026, on a log scale"""
    import numpy as np

    from yarapm import rapm
    from yarapm.catalog import read

    con = connect()
    read(con, "segments", ROOT / "out").create_view("seg")
    t = con.sql(
        "SELECT off_player_ids, def_player_ids, poss, pts FROM seg WHERE season = 2026 AND poss > 0"
    ).to_arrow_table()
    rows = rapm.Lineups.from_ids(
        t["off_player_ids"].combine_chunks(),
        t["def_player_ids"].combine_chunks(),
        t["poss"].to_numpy(),
        t["pts"].to_numpy(),
        np.ones(len(t)),
        np.ones((len(t), 1)),
    )
    X = rapm.design(rows)
    A = (X.T @ X.multiply(rows.poss[:, None])).toarray()
    ev = np.sort(np.linalg.eigvalsh(A))
    print(
        f"eigenvalues: {len(ev)}; {(ev < 1e-6).sum()} zero, then {ev[2]:.2f}; largest {ev[-1]:,.0f}; "
        f"{(ev < 100).sum()} under 100, {(ev < 1000).sum()} under 1000; condition without the zeros {ev[-1] / ev[2]:,.0f}"
    )
    s = Svg(
        640,
        240,
        "The eigenvalues of the normal equations, sorted: two are zero, and they range over five orders of magnitude",
    )
    a = Axes(s, 70, 15, 540, 175, (0, len(ev)), (0, 6))
    a.frame(
        [0, 200, 400, 600, 800, 1000],
        [0, 1, 2, 3, 4, 5, 6],
        "eigenvalues in increasing order",
        "eigenvalue",
        xfmt=lambda v: f"{v:,.0f}",
        yfmt=lambda v: ["1", "10", "100", "1,000", "10⁴", "10⁵", "10⁶"][int(v)],
    )
    pts = [(i, math.log10(e)) for i, e in enumerate(ev) if e > 1e-6]
    a.line([p[0] for p in pts], [p[1] for p in pts], width=1.4)
    s.circle(a.px(0), a.py(0) - 3, 2.4)
    s.circle(a.px(1), a.py(0) - 3, 2.4)
    s.text(a.px(8), a.py(0) - 8, "two zeros", size=10.5, italic=True)
    s.line(
        a.x,
        a.py(math.log10(3000)),
        a.x + a.w,
        a.py(math.log10(3000)),
        stroke=FAINT,
        width=0.8,
        dash="4 3",
    )
    s.text(
        a.x + a.w,
        a.py(math.log10(3000)) - 5,
        "λ = 3000",
        size=10.5,
        italic=True,
        anchor="end",
        fill=FAINT,
    )
    return s


@figure("ridge")
def ridge() -> Svg:
    """Ridge ratings against possessions, 2026, on the axes of least-squares"""
    import numpy as np

    _, poss, fits = season_fits()
    r = fits[3000.0]
    few, many = poss < 500, poss >= 4000
    print(
        f"ridge: sd of rating {np.std(r[few]):.2f} under 500 possessions, {np.std(r[many]):.2f} over 4000; range {r.min():+.1f} to {r.max():+.1f}"
    )
    s = Svg(
        640,
        260,
        "Scatter plot: with a ridge penalty, ratings of players with few possessions collapse to zero",
    )
    a = Axes(s, 70, 15, 540, 200, (0, 8000), (-40, 40))
    rating_scatter(s, a, poss, r, "λ = 3000 possessions")
    return s


@figure("shrinkage")
def shrinkage() -> Svg:
    """The share of the data's estimate kept, n / (n + λ)"""
    s = Svg(
        640,
        230,
        "Curves of n over n plus lambda for three penalties: the larger the penalty, the more possessions it takes to trust the data",
    )
    a = Axes(s, 70, 15, 540, 165, (0, 20000), (0, 1))
    a.frame(
        [0, 5000, 10000, 15000, 20000],
        [0, 0.25, 0.5, 0.75, 1],
        "possessions of data, n",
        "weight on the data",
        xfmt=lambda v: f"{v:,.0f}",
        yfmt=lambda v: f"{v:g}",
    )
    xs = [i * 100 for i in range(201)]
    for lam, dash, label_x in (
        (1000, None, 2600),
        (3000, "6 3", 7800),
        (10000, "2 3", 15500),
    ):
        ys = [x / (x + lam) for x in xs]
        a.line(xs, ys, dash=dash)
        y = label_x / (label_x + lam)
        s.text(a.px(label_x) + 4, a.py(y) + 16, f"λ = {lam:,}", size=10.5, italic=True)
    s.line(a.x, a.py(0.5), a.x + a.w, a.py(0.5), stroke=FAINT, width=0.6, dash="1 3")
    return s


@figure("valley")
def valley() -> Svg:
    """
    Contours of the least squares objective for two players who mostly play
    together, and of the ridge objective
    """
    import numpy as np

    m, a_, b_ = 2963, 355, 2359  # Braun and Jokić, 2026
    A = np.array([[m + b_, m], [m, m + a_]])  # (Jokić, Braun)
    est = np.array([14.0, 2.6])
    lam = 3000
    ridge_est = np.linalg.solve(A + lam * np.eye(2), A @ est)
    print(
        f"valley: eigenvalues {np.linalg.eigvalsh(A).round(0)}, ridge estimate {ridge_est.round(2)}"
    )
    s = Svg(
        640,
        300,
        "Two contour plots: least squares has a long thin valley along which two teammates' ratings trade off; the ridge penalty rounds it into a bowl near zero",
    )

    def panel(x0, M, center, title, mark):
        ax = Axes(s, x0, 22, 230, 230, (-10, 25), (-10, 25))
        ax.frame(
            [-10, 0, 10, 20],
            [-10, 0, 10, 20],
            "Jokić, points per 100",
            "Braun, points per 100",
            xfmt=lambda v: f"{v:+.0f}" if v else "0",
            yfmt=lambda v: f"{v:+.0f}" if v else "0",
        )
        s.line(ax.px(0), ax.py(-10), ax.px(0), ax.py(25), stroke=FAINT, width=0.5)
        s.line(ax.px(-10), ax.py(0), ax.px(25), ax.py(0), stroke=FAINT, width=0.5)
        vals, vecs = np.linalg.eigh(M)
        for level in (2e4, 8e4, 2e5, 4e5):
            pts = []
            for t in np.linspace(0, 2 * np.pi, 120):
                u = vecs @ (np.array([np.cos(t), np.sin(t)]) * np.sqrt(level / vals))
                p = center + u
                pts.append((ax.px(p[0]), ax.py(p[1])))
            s.add(
                f'<clipPath id="c{x0}"><rect x="{num(ax.x)}" y="{num(ax.y)}" '
                f'width="{num(ax.w)}" height="{num(ax.h)}"/></clipPath>'
            )
            s.add(f'<g clip-path="url(#c{x0})">')
            s.polyline(pts + [pts[0]], width=1)
            s.add("</g>")
        s.circle(ax.px(center[0]), ax.py(center[1]), 3)
        s.text(ax.px(center[0]) + 6, ax.py(center[1]) - 6, mark, size=10.5, italic=True)
        s.text(x0 + 230, 14, title, size=12, italic=True, anchor="end")

    panel(75, A, est, "(a) least squares", "estimate")
    panel(395, A + lam * np.eye(2), ridge_est, "(b) ridge, λ = 3000", "estimate")
    return s


# chapter 5


@figure("decay")
def decay_weights() -> Svg:
    """The weight of a game by its age: ratings (365 days) and the league level (7 days)"""
    s = Svg(
        640,
        230,
        "Two decay curves: a game's weight halves every year for the ratings and every week for the league's level",
    )
    a = Axes(s, 60, 22, 300, 150, (0, 8 * 365), (0, 1))
    a.frame(
        [0, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920],
        [0, 0.25, 0.5, 0.75, 1],
        "age of the game, years",
        "weight",
        xfmt=lambda v: f"{v / 365:.0f}",
        yfmt=lambda v: f"{v:g}",
    )
    xs = list(range(0, 8 * 365 + 1, 10))
    a.line(xs, [0.5 ** (x / 365) for x in xs])
    for yr in range(1, 8):
        s.circle(a.px(yr * 365), a.py(0.5**yr), 2.2)
    s.text(a.px(365) + 6, a.py(0.5) - 6, "½ after a year", size=10.5, italic=True)
    s.text(
        360, 14, "(a) ratings: half life 365 days", italic=True, anchor="end", size=12
    )
    b = Axes(s, 410, 22, 210, 150, (0, 56), (0, 1))
    b.frame(
        [0, 7, 14, 21, 28, 35, 42, 49, 56],
        [0, 0.25, 0.5, 0.75, 1],
        "age of the game, days",
        "",
        yfmt=lambda v: f"{v:g}",
    )
    xs = [x / 2 for x in range(113)]
    b.line(xs, [0.5 ** (x / 7) for x in xs])
    s.text(620, 14, "(b) league level: 7 days", italic=True, anchor="end", size=12)
    eff = 365 / math.log(2) * (1 - 0.5**8)
    print(f"decay: effective length of the 365-day window {eff:.0f} days")
    return s


@figure("drift")
def drift() -> Svg:
    """The league's points per 100 possessions by month, three seasons"""
    from yarapm.games import load_results

    con = connect()
    load_results(con, "season IN (2022, 2024, 2026)")
    rows = con.execute("""
        SELECT season, month(game_date) AS m, 100 * sum(pts + opp_pts) / sum(poss + opp_poss)
        FROM results WHERE game_type = 'regular_season'
        GROUP BY season, m
    """).fetchall()
    order = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    s = Svg(
        640,
        240,
        "Line chart of league efficiency by month in 2022, 2024 and 2026: it rose ten points through 2022, peaked in December 2024 and fell four by April",
    )
    a = Axes(s, 70, 15, 470, 170, (-0.3, 6.3), (104, 118))
    a.frame([], [104, 107, 110, 113, 116], "", "points per 100 possessions")
    for i in order.values():
        s.line(a.px(i), a.py(104), a.px(i), a.py(104) + 4)
        s.text(
            a.px(i),
            a.py(104) + 16,
            ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"][i],
            anchor="middle",
        )
    styles = {2022: (None, INK), 2024: ("6 3", INK), 2026: ("2 3", INK)}
    for season, (dash, fill) in styles.items():
        pts = sorted((order[m], e) for se, m, e in rows if se == season and m in order)
        print(f"drift {season}: " + ", ".join(f"{e:.1f}" for _, e in pts))
        a.line([p[0] for p in pts], [p[1] for p in pts], dash=dash)
        a.points(
            [p[0] for p in pts],
            [p[1] for p in pts],
            r=2.2,
            fill=PAPER if season == 2024 else INK,
            stroke=INK,
        )
        last = pts[-1]
        s.text(a.px(last[0]) + 8, a.py(last[1]) + 4, str(season), size=11, italic=True)
    return s


# chapter 6


@figure("box-formula")
def box_formula() -> Svg:
    """The box formula's coefficients on standardized inputs, fit for 2026"""
    from yarapm.harness import History, load
    from yarapm.players import PlayerRatings
    from yarapm.prior import RATES, BoxPrior

    con = connect()
    load(con, ROOT / "out")
    opener = con.execute(
        "SELECT min(game_date) FROM results WHERE season = 2026"
    ).fetchall()[0][0]
    m = PlayerRatings(minutes="unknown")
    m.fit(History(con, opener, 2026))
    prior = m.prior
    assert isinstance(prior, BoxPrior) and prior.box is not None
    labels = {
        "pts": "points",
        "fg2a": "2-pt attempts",
        "fg3a": "3-pt attempts",
        "fta": "free throw att.",
        "ast": "assists",
        "tov": "turnovers",
        "oreb": "off. rebounds",
        "dreb": "def. rebounds",
        "stl": "steals",
        "blk": "blocks",
        "pf": "fouls",
    }
    names = [labels[r] for r in RATES] + [
        "true shooting",
        "3-pt %",
        "usage × shooting",
        "minutes per game",
        "height",
        "log draft pick",
        "undrafted",
    ]
    off, dfn = prior.box.offense[1:], prior.box.defense[1:]
    for n, o, d in zip(names, off, dfn):
        print(f"box formula {n}: offense {o:+.2f}, defense {d:+.2f}")
    s = Svg(
        640,
        20 + 17 * len(names) + 40,
        "Bar charts of the box formula's coefficients for offense and defense",
    )
    for col, (vals, title) in enumerate(((off, "(a) offense"), (dfn, "(b) defense"))):
        x0 = 230 + col * 210
        a = Axes(s, x0, 20, 170, 17 * len(names), (-1.5, 1.5), (0, len(names)))
        s.line(a.px(0), a.y, a.px(0), a.y + a.h)
        for i, v in enumerate(vals):
            y = a.y + 17 * i + 3
            s.rect(
                min(a.px(0), a.px(v)),
                y,
                abs(a.px(v) - a.px(0)),
                11,
                fill=GREY if v > 0 else "url(#hatch)",
                width=0.7,
            )
        for t in (-1, 0, 1):
            s.line(a.px(t), a.y + a.h, a.px(t), a.y + a.h + 4)
            s.text(a.px(t), a.y + a.h + 16, f"{t:+d}" if t else "0", anchor="middle")
        s.text(x0 + 170, 12, title, italic=True, anchor="end", size=12)
    for i, n in enumerate(names):
        s.text(215, 20 + 17 * i + 12, esc(n), size=10.5, anchor="end")
    s.text(
        435,
        20 + 17 * len(names) + 34,
        "points per 100 possessions, per standard deviation of the input",
        italic=True,
        anchor="middle",
        size=10.5,
    )
    return s


@figure("wembanyama")
def wembanyama() -> Svg:
    """Victor Wembanyama's rating and prior through his rookie season, 2024"""
    from datetime import date

    con = connect()
    con.execute(
        "ATTACH 'https://basketball-data.billmill.org/yarapm/yarapm.duckdb' AS yarapm (READ_ONLY)"
    )
    rows = con.execute("""
        SELECT r.as_of_date, r.o_rating + r.d_rating, r.o_prior + r.d_prior, r.possessions
        FROM yarapm.ratings_daily r JOIN nba.players p USING (player_id)
        WHERE r.season = 2024 AND p.name = 'Victor Wembanyama' ORDER BY 1
    """).fetchall()
    start = date(2023, 10, 24)
    days = [(d - start).days for d, *_ in rows]
    print(
        f"wembanyama 2024: first {rows[0][0]} rating {rows[0][1]:+.2f}, last {rows[-1][0]} rating {rows[-1][1]:+.2f} prior {rows[-1][2]:+.2f}, {rows[-1][3]} possessions"
    )
    s = Svg(
        640,
        240,
        "Line chart: Wembanyama's rating rose from about +1.5 to +4.4 through his rookie season, pulling away from his box-score prior",
    )
    a = Axes(s, 60, 15, 510, 170, (0, 240), (-1, 5))
    months = [
        ("Nov", date(2023, 11, 1)),
        ("Dec", date(2023, 12, 1)),
        ("Jan", date(2024, 1, 1)),
        ("Feb", date(2024, 2, 1)),
        ("Mar", date(2024, 3, 1)),
        ("Apr", date(2024, 4, 1)),
        ("May", date(2024, 5, 1)),
        ("Jun", date(2024, 6, 1)),
    ]
    a.frame(
        [],
        [-1, 0, 1, 2, 3, 4, 5],
        "",
        "points per 100 possessions",
        yfmt=lambda v: f"{v:+.0f}" if v else "0",
    )
    for name, d in months:
        x = a.px((d - start).days)
        s.line(x, a.y + a.h, x, a.y + a.h + 4)
        s.text(x, a.y + a.h + 16, name, anchor="middle")
    s.line(a.x, a.py(0), a.x + a.w, a.py(0), stroke=FAINT, width=0.6)
    a.line(days, [r[1] for r in rows], width=1.6)
    a.line(days, [r[2] for r in rows], dash="5 3", width=1.2)
    s.text(a.px(days[-1]) + 4, a.py(rows[-1][1]) + 4, "rating", size=11, italic=True)
    s.text(a.px(days[-1]) + 4, a.py(rows[-1][2]) + 4, "prior", size=11, italic=True)
    return s


# chapter 7


@figure("availability")
def availability_curves() -> Svg:
    """The chance a rostered player plays, by games since they last played"""
    from datetime import date

    from yarapm.harness import History, load
    from yarapm.minutes import availability

    con = connect()
    load(con, ROOT / "out")
    chances = availability(History(con, date(2026, 6, 13), 2026))
    s = Svg(
        640,
        240,
        "Line chart: a player who played every recent game plays 94% of the time; one who missed the last game, about 40 to 50%",
    )
    a = Axes(s, 70, 15, 470, 170, (-0.3, 9.3), (0, 1))
    a.frame(
        list(range(10)),
        [0, 0.25, 0.5, 0.75, 1],
        "team games since the player last played",
        "chance of playing",
        xfmt=lambda v: f"{v:.0f}",
        yfmt=lambda v: f"{v:g}",
    )
    for played, dash in ((9, None), (7, "6 3"), (4, "3 3"), (1, "1 2")):
        pts = sorted(
            (since, c)
            for (since, n), c in chances.items()
            if n == played and since <= 9
        )
        print(
            f"availability, played {played} of the last 10: "
            + ", ".join(f"{since}: {c:.3f}" for since, c in pts)
        )
        a.line([p[0] for p in pts], [p[1] for p in pts], dash=dash)
        a.points([p[0] for p in pts], [p[1] for p in pts], r=2)
        last = pts[-1]
        s.text(
            a.px(last[0]) + 8,
            a.py(last[1]) + 4,
            f"played {played} of 10",
            size=10.5,
            italic=True,
        )
    return s


# chapter 8

# the daily `rapm` run with unknown availability on 2021-2024, the
# scoreboard's (yarapm.harness --model rapm --cadence daily --set
# minutes=unknown), with segments from 2019 on
RUN = (
    ROOT
    / "out"
    / "x2019"
    / "runs"
    / "predictions"
    / "run=rapm-93ea3652"
    / "data.parquet"
)


def run_errors():
    import duckdb
    import numpy as np

    p = duckdb.execute(
        f"SELECT pred_pts - pred_opp_pts AS pm, pts - opp_pts AS m FROM '{RUN}'"
    ).fetchnumpy()
    pm, m = np.asarray(p["pm"], dtype=float), np.asarray(p["m"], dtype=float)
    return pm, m, float(np.sqrt(np.mean((pm - m) ** 2)))


@figure("errors")
def errors() -> Svg:
    """The distribution of margin errors against a normal with the same spread"""
    import numpy as np

    pm, m, sigma = run_errors()
    err = pm - m
    print(
        f"errors: {len(err)} games, sigma {sigma:.2f}; beyond 1 sigma {np.mean(np.abs(err) > sigma):.3f}, beyond 2 {np.mean(np.abs(err) > 2 * sigma):.3f}"
    )
    s = Svg(
        640,
        230,
        "Histogram of the error in predicted margin, close to a normal curve with the same standard deviation",
    )
    a = Axes(s, 70, 15, 540, 170, (-50, 50), (0, 0.032))
    a.frame(
        [-50, -25, 0, 25, 50],
        [0, 0.01, 0.02, 0.03],
        "predicted margin − actual margin, points",
        "share of games per point",
        xfmt=lambda v: f"{v:+.0f}" if v else "0",
        yfmt=lambda v: f"{v:g}",
    )
    edges = np.arange(-50, 51, 4)
    counts, _ = np.histogram(err, edges)
    for lo, c in zip(edges[:-1].tolist(), counts):
        h = c / len(err) / 4
        s.rect(
            a.px(lo),
            a.py(h),
            a.px(lo + 4) - a.px(lo),
            a.py(0) - a.py(h),
            fill=GREY,
            width=0.7,
        )
    xs = np.linspace(-50, 50, 201)
    a.line(
        xs,
        np.exp(-(xs**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi)),
        width=1.5,
    )
    s.text(a.px(20), a.py(0.022), f"normal, σ = {sigma:.2f}", italic=True, size=11)
    return s


@figure("calibration")
def calibration() -> Svg:
    """Predicted win probability against how often the team won"""
    import numpy as np
    from scipy.special import ndtr

    pm, m, sigma = run_errors()
    prob = ndtr(pm / sigma)
    won = m > 0
    s = Svg(
        640,
        300,
        "Calibration chart: in each band of predicted probability, the home team won about as often as predicted",
    )
    a = Axes(s, 170, 15, 260, 240, (0, 1), (0, 1))
    a.frame(
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0.25, 0.5, 0.75, 1],
        "predicted chance the home team wins",
        "share it won",
        xfmt=lambda v: f"{v:g}",
        yfmt=lambda v: f"{v:g}",
    )
    s.line(a.px(0), a.py(0), a.px(1), a.py(1), stroke=FAINT, width=0.8, dash="3 3")
    edges = np.linspace(0, 1, 11)
    for lo, hi in itertools.pairwise(edges):
        i = (prob >= lo) & (prob < hi)
        if i.sum() < 20:
            continue
        x, y, n = prob[i].mean(), won[i].mean(), i.sum()
        se = math.sqrt(y * (1 - y) / n)
        s.line(a.px(x), a.py(y - 2 * se), a.px(x), a.py(y + 2 * se), width=0.8)
        a.points([x], [y], r=2.8)
        print(
            f"calibration {lo:.1f}-{hi:.1f}: {n} games, predicted {x:.3f}, won {y:.3f}"
        )
    return s


# chapter 9


def published():
    con = connect()
    con.execute(
        "ATTACH 'https://basketball-data.billmill.org/yarapm/yarapm.duckdb' AS yarapm (READ_ONLY)"
    )
    return con


@figure("george")
def george() -> Svg:
    """Paul George's ratings as of each day, 2018 to now"""
    from datetime import date

    con = published()
    rows = con.execute("""
        SELECT r.season, r.as_of_date, r.o_rating, r.d_rating, r.rating, t.nba_abbrev
        FROM yarapm.ratings_daily r
        JOIN nba.players p USING (player_id)
        LEFT JOIN nba.team_seasons t ON t.season = r.season AND t.team_id = r.team_id
        WHERE p.name = 'Paul George' ORDER BY r.as_of_date
    """).fetchall()
    version = con.execute(
        "SELECT model_version FROM yarapm.metadata WHERE dataset = 'ratings_daily'"
    ).fetchall()[0][0]
    print(f"george: {len(rows)} days, model version {version}")
    for season in sorted({r[0] for r in rows}):
        last = [r for r in rows if r[0] == season][-1]
        teams = sorted({r[5] for r in rows if r[0] == season and r[5]})
        print(
            f"george {season}: final {last[4]:+.2f} (offense {last[2]:+.2f}, defense {last[3]:+.2f}), {', '.join(teams)}"
        )
    start, end = date(2013, 10, 1), date(2026, 12, 1)
    span = (end - start).days
    s = Svg(
        640,
        270,
        "Line chart of Paul George's rating from 2014 to 2027: four to six in Indiana, near +7 from 2019 to 2024, under +4 since",
    )
    a = Axes(s, 60, 25, 560, 180, (0, span), (-2, 10))
    a.frame(
        [],
        [-2, 0, 2, 4, 6, 8, 10],
        "",
        "points per 100 possessions",
        yfmt=lambda v: f"{v:+.0f}" if v else "0",
    )
    s.line(a.x, a.py(0), a.x + a.w, a.py(0), stroke=FAINT, width=0.6)
    for season in sorted({r[0] for r in rows}):
        pts = [r for r in rows if r[0] == season]
        xs = [(r[1] - start).days for r in pts]
        a.line(xs, [r[4] for r in pts], width=1.6)
        a.line(xs, [r[2] for r in pts], width=0.9, dash="4 2")
        a.line(xs, [r[3] for r in pts], width=0.9, dash="1 2")
        mid = (xs[0] + xs[-1]) / 2
        s.text(
            a.px(mid), a.y + a.h + 16, f"{season % 100:02d}", anchor="middle", size=10.5
        )
        team = sorted(
            {r[5] for r in pts if r[5]}, key=lambda t: -sum(1 for r in pts if r[5] == t)
        )
        if team:
            s.text(
                a.px(mid),
                a.y - 6,
                team[0],
                anchor="middle",
                size=9,
                italic=True,
                fill=FAINT,
            )
    s.text(a.x + a.w, a.y + a.h + 34, "season, '14 to '27", italic=True, anchor="end")
    lx = a.x + 8
    s.line(lx, 258, lx + 20, 258, width=1.6)
    s.text(lx + 25, 262, "total", size=10.5)
    s.line(lx + 70, 258, lx + 90, 258, width=0.9, dash="4 2")
    s.text(lx + 95, 262, "offense", size=10.5)
    s.line(lx + 155, 258, lx + 175, 258, width=0.9, dash="1 2")
    s.text(lx + 180, 262, "defense", size=10.5)
    return s


@figure("warriors")
def warriors() -> Svg:
    """Golden State's rating at full strength and on the day, 2026, with Curry's games"""
    from datetime import date

    con = published()
    rows = con.execute("""
        SELECT r.as_of_date, r.full_net, r.net
        FROM yarapm.team_ratings_daily r JOIN nba.team_seasons t ON t.season = r.season AND t.team_id = r.team_id
        WHERE r.season = 2026 AND t.nba_abbrev = 'GSW' ORDER BY 1
    """).fetchall()
    played = {
        d
        for (d,) in con.execute("""
        SELECT l.game_date FROM nba.player_game_logs l JOIN nba.players p USING (player_id)
        WHERE l.season = 2026 AND p.name = 'Stephen Curry' AND l.min > 0
    """).fetchall()
    }
    games = [
        d
        for (d,) in con.execute("""
        SELECT DISTINCT game_date FROM nba.team_game_logs l JOIN nba.team_seasons t ON t.season = l.season AND t.team_id = l.team_id
        WHERE l.season = 2026 AND t.nba_abbrev = 'GSW' ORDER BY 1
    """).fetchall()
    ]
    missed = [d for d in games if d not in played]
    start = date(2025, 10, 15)
    span = (max(r[0] for r in rows) - start).days + 5
    print(
        f"warriors: {len(rows)} days; Curry played {len(played)}, missed {len(missed)} of {len(games)}; largest gap full minus today {max(f - n for d, f, n in rows):+.2f}"
    )
    s = Svg(
        640,
        250,
        "Two lines over the 2026 season: Golden State's full-strength rating and its rating with the expected lineup, which drops on the days Curry was out",
    )
    lo = 2 * math.floor(min(min(r[1], r[2]) for r in rows) / 2) - 2
    hi = 2 * math.ceil(max(max(r[1], r[2]) for r in rows) / 2)
    a = Axes(s, 60, 15, 560, 180, (0, span), (lo, hi))
    a.frame(
        [],
        list(range(lo, hi + 1, 2)),
        "",
        "net rating, points per 100",
        yfmt=lambda v: f"{v:+.0f}" if v else "0",
    )
    s.line(a.x, a.py(0), a.x + a.w, a.py(0), stroke=FAINT, width=0.6)
    for d in missed:
        x = a.px((d - start).days)
        s.line(x, a.y + a.h - 6, x, a.y + a.h, width=1.2)
    xs = [(r[0] - start).days for r in rows]
    a.line(xs, [r[1] for r in rows], width=1.6)
    a.line(xs, [r[2] for r in rows], width=1.0, dash="4 2")
    for name, d in [
        ("Nov", date(2025, 11, 1)),
        ("Dec", date(2025, 12, 1)),
        ("Jan", date(2026, 1, 1)),
        ("Feb", date(2026, 2, 1)),
        ("Mar", date(2026, 3, 1)),
        ("Apr", date(2026, 4, 1)),
        ("May", date(2026, 5, 1)),
    ]:
        if (d - start).days <= span:
            x = a.px((d - start).days)
            s.line(x, a.y + a.h, x, a.y + a.h + 4)
            s.text(x, a.y + a.h + 16, name, anchor="middle")
    lx = a.x + 8
    s.line(lx, 240, lx + 20, 240, width=1.6)
    s.text(lx + 25, 244, "full strength", size=10.5)
    s.line(lx + 115, 240, lx + 135, 240, width=1.0, dash="4 2")
    s.text(lx + 140, 244, "expected lineup", size=10.5)
    s.line(lx + 250, 234, lx + 250, 240, width=1.2)
    s.text(lx + 256, 244, "a game Curry missed", size=10.5)
    return s


def main(names: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names or list(REGISTRY):
        svg = REGISTRY[name]()
        (FIGURES / f"{name}.svg").write_text(svg.render())
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
