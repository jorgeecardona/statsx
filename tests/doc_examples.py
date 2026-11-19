"""Run the Python examples in the docs and compare their output with the text block below them.

Usage:
    python tests/doc_examples.py          # check every page, exit 1 on a mismatch
    python tests/doc_examples.py --write  # replace each output block with the real output

An example is a ```python block followed by a ```text block. The blocks of one page run in order
in one namespace, so a later block can use the names of an earlier one. A ```python block with no
```text block after it is an error, unless the line before it is ``<!-- not run -->``.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

_DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(
    [_DOCS.parent / "README.md", *_DOCS.glob("methods/*.md"), *_DOCS.glob("recipes/*.md")]
)
PATTERN = re.compile(r"```python\n(?P<code>.*?)```\n\n```text\n(?P<out>.*?)```", re.DOTALL)
BLOCK = re.compile(
    r"(?P<before>[^\n]*\n)?```python\n(?P<code>.*?)```(?P<after>\n\n```text\n)?", re.DOTALL
)
NOT_RUN = "<!-- not run -->"


def run(code: str, namespace: dict[str, object] | None = None) -> str:
    """Return what ``code`` prints, running it in ``namespace`` (a fresh one by default)."""
    buffer = io.StringIO()
    scope = {"__name__": "__doc_example__"} if namespace is None else namespace
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<doc example>", "exec"), scope)  # noqa: S102
    return "\n".join(line.rstrip() for line in buffer.getvalue().rstrip().splitlines()) + "\n"


def check(page: Path, *, write: bool) -> list[str]:
    """Return one message per example whose output differs; rewrite the page when ``write``."""
    text = page.read_text()
    problems: list[str] = []
    namespace: dict[str, object] = {"__name__": "__doc_example__"}

    def replace(match: re.Match[str]) -> str:
        got = run(match["code"], namespace)
        if got != match["out"]:
            problems.append(
                f"{page.name}: output differs for example starting {match['code'][:60]!r}"
            )
        return f"```python\n{match['code']}```\n\n```text\n{got}```" if write else match[0]

    problems += [
        f"{page.name}: python block with no output block, starting {m['code'][:60]!r}"
        for m in BLOCK.finditer(text)
        if m["after"] is None and NOT_RUN not in (m["before"] or "")
    ]
    new = PATTERN.sub(replace, text)
    if write and new != text:
        page.write_text(new)
    return problems


if __name__ == "__main__":
    found: list[str] = []
    for md in PAGES:
        found += check(md, write="--write" in sys.argv)
    sys.stdout.write("\n".join(found) or "all doc examples match")
    sys.stdout.write("\n")
    sys.exit(1 if found and "--write" not in sys.argv else 0)
