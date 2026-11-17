"""Run the Python examples in the docs and compare their output with the text block below them.

Usage:
    python tests/doc_examples.py          # check every page, exit 1 on a mismatch
    python tests/doc_examples.py --write  # replace each output block with the real output

An example is a ```python block followed by a ```text block. The code runs in a fresh namespace.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

PAGES = sorted(Path(__file__).resolve().parent.parent.glob("docs/methods/*.md"))
PATTERN = re.compile(r"```python\n(?P<code>.*?)```\n\n```text\n(?P<out>.*?)```", re.DOTALL)


def run(code: str) -> str:
    """Return what ``code`` prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<doc example>", "exec"), {"__name__": "__doc_example__"})  # noqa: S102
    return "\n".join(line.rstrip() for line in buffer.getvalue().rstrip().splitlines()) + "\n"


def check(page: Path, *, write: bool) -> list[str]:
    """Return one message per example whose output differs; rewrite the page when ``write``."""
    text = page.read_text()
    problems: list[str] = []

    def replace(match: re.Match[str]) -> str:
        got = run(match["code"])
        if got != match["out"]:
            problems.append(
                f"{page.name}: output differs for example starting {match['code'][:60]!r}"
            )
        return f"```python\n{match['code']}```\n\n```text\n{got}```" if write else match[0]

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
