"""
The book's figures, as SVG in ink on paper, into book/figures/.

    mise run book-figures [name ...]

Each figure is a function registered with @figure. Some are closed forms,
some read nba_data's catalog or yarapm's published one, and some need the
segments in out/ (`yarapm.segments`) or a harness run. Every number the
text quotes from a figure is printed, so a rerun can be checked against
the text. The SVGs are committed: the book's build doesn't run this.
"""

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


def main(names: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names or list(REGISTRY):
        svg = REGISTRY[name]()
        (FIGURES / f"{name}.svg").write_text(svg.render())
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
