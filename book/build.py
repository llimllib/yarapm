"""
Build the book, book/chapters/*.md, into out/book/ with pandoc.

    mise run book

Two passes: the first numbers each chapter's sections, environments,
equations, figures and tables (filters/environments.lua) and collects
their labels; the second renders every chapter with all of them, so a
reference can point into another chapter. book/index.md is the title page,
followed by the contents.
"""

import html
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

BOOK = Path(__file__).resolve().parent
OUT = BOOK.parent / "out" / "book"
# KaTeX's script and stylesheet, which also has the book's text font.
# BOOK_KATEX_URL overrides it, to preview without the network
KATEX = os.environ.get(
    "BOOK_KATEX_URL", "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/"
)


def pandoc(source: Path, output: Path, *args: str) -> None:
    subprocess.run(
        [
            "pandoc",
            str(source.relative_to(BOOK)),
            "--from=markdown",
            "--to=html5",
            "--metadata-file=book.yaml",
            # the text's font comes with KaTeX's stylesheet, so every page needs it
            f"--variable=katex:{KATEX}",
            "--lua-filter=filters/environments.lua",
            "--lua-filter=filters/sidenotes.lua",
            f"--output={output}",
            *args,
        ],
        check=True,
        # figures are read relative to book/
        cwd=BOOK,
    )


def contents(chapters: list[dict[str, Any]]) -> str:
    """The table of contents, as HTML"""
    items = []
    for c in chapters:
        sections = "".join(
            f'<li class="toc-sub"><a href="{c["file"]}#{s["id"]}">'
            f'<span class="num">{s["number"]}</span> {html.escape(s["title"])}</a></li>'
            for s in c["sections"]
        )
        items.append(
            f'<li class="toc-sec"><a href="{c["file"]}">'
            f'<span class="num">{c["number"]}</span> {html.escape(c["title"])}</a></li>'
            f"{sections}"
        )
    return (
        '<nav class="toc book-toc"><p class="toc-head">Contents</p>'
        f"<ol>{''.join(items)}</ol></nav>"
    )


def main() -> None:
    sources = sorted((BOOK / "chapters").glob("[0-9][0-9]-*.md"))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        chapters: list[dict[str, Any]] = []
        labels: dict[str, dict] = {}
        for source in sources:
            number = int(source.name[:2])
            file = source.with_suffix(".html").name
            found = Path(tmp) / f"{number}.json"
            pandoc(
                source,
                Path(tmp) / "pass1.html",
                f"--metadata=chapter:{number}",
                f"--metadata=file:{file}",
                f"--metadata=collect:{found}",
            )
            collected = json.loads(found.read_text())
            labels |= collected["labels"]
            chapters.append(
                {
                    "number": number,
                    "file": file,
                    "title": collected["title"],
                    "sections": collected["sections"],
                }
            )
        path = Path(tmp) / "labels.json"
        path.write_text(json.dumps(labels))

        for i, (source, c) in enumerate(zip(sources, chapters)):
            pager = []
            if i > 0:
                prev = chapters[i - 1]
                pager += [
                    f"--variable=prev-href:{prev['file']}",
                    f"--variable=prev-title:{prev['title']}",
                ]
            if i + 1 < len(chapters):
                nxt = chapters[i + 1]
                pager += [
                    f"--variable=next-href:{nxt['file']}",
                    f"--variable=next-title:{nxt['title']}",
                ]
            pandoc(
                source,
                OUT / c["file"],
                "--standalone",
                "--template=template.html",
                f"--math-method=katex:{KATEX}",
                "--toc",
                "--toc-depth=2",
                f"--metadata=chapter:{c['number']}",
                f"--metadata=file:{c['file']}",
                f"--metadata=labels:{path}",
                *pager,
            )
        first = chapters[0] if chapters else None
        pandoc(
            BOOK / "index.md",
            OUT / "index.html",
            "--standalone",
            "--template=template.html",
            f"--math-method=katex:{KATEX}",
            "--metadata=file:index.html",
            f"--metadata=labels:{path}",
            f"--variable=contents:{contents(chapters)}",
            *(
                [
                    f"--variable=next-href:{first['file']}",
                    f"--variable=next-title:{first['title']}",
                ]
                if first
                else []
            ),
        )
    shutil.copy(BOOK / "style.css", OUT / "style.css")
    print(f"wrote {OUT} ({len(chapters)} chapters)")


if __name__ == "__main__":
    main()
