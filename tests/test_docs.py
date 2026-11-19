import re
from pathlib import Path

import pytest

import doc_examples


@pytest.mark.parametrize("page", doc_examples.PAGES, ids=lambda p: p.name)
def test_every_example_in_the_methods_pages_prints_what_the_page_shows(page: Path) -> None:
    assert doc_examples.check(page, write=False) == []


def test_a_python_block_without_an_output_block_is_an_error_unless_marked_not_run(
    tmp_path: Path,
) -> None:
    page = tmp_path / "page.md"
    page.write_text("text\n\n```python\nprint(1)\n```\n")
    assert len(doc_examples.check(page, write=False)) == 1
    page.write_text("<!-- not run -->\n```python\nprint(1)\n```\n")
    assert doc_examples.check(page, write=False) == []


def test_the_blocks_of_one_page_share_one_namespace() -> None:
    first = doc_examples.run("x = 2\nprint(x)", namespace=(scope := {"__name__": "t"}))
    second = doc_examples.run("print(x + 1)", namespace=scope)
    assert (first, second) == ("2\n", "3\n")


_NOT_A_METHOD = {"__init__", "_ranks", "errors", "result"}
_SRC = Path(__file__).resolve().parent.parent / "src" / "statsx"


def test_every_public_module_is_in_the_choose_page_and_in_the_api_reference() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    choose, api = (docs / "choose.md").read_text(), (docs / "api.md").read_text()
    modules = sorted(p.stem for p in _SRC.glob("*.py") if p.stem not in _NOT_A_METHOD)
    assert [m for m in modules if f"`statsx.{m}`" not in choose] == []
    assert [m for m in modules if f"::: statsx.{m}" not in api] == []


def test_every_link_in_the_choose_page_points_to_a_file_or_an_api_entry() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    text = (docs / "choose.md").read_text()
    api = (docs / "api.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        path, _, anchor = target.partition("#")
        assert (docs / path).exists(), target
        if anchor:
            assert f"## {anchor.replace('statsx', 'statsx.', 1)}" in api, target


_ASSUMPTION_HEADINGS = (
    "## What it assumes\n",
    "## What the library checks and does not check\n",
)


@pytest.mark.parametrize(
    "page", sorted((Path(__file__).resolve().parent.parent / "docs" / "methods").glob("*.md"))
)
def test_every_methods_page_has_the_two_assumption_headings(page: Path) -> None:
    text = page.read_text()
    assert [h for h in _ASSUMPTION_HEADINGS if h not in text] == []
    assert text.index(_ASSUMPTION_HEADINGS[0]) < text.index(_ASSUMPTION_HEADINGS[1])


def test_every_module_in_the_choose_page_links_to_a_methods_page() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    text = (docs / "choose.md").read_text()
    assert "api.md#" not in text


def test_no_module_docstring_starts_with_a_need_id_or_an_issue_number() -> None:
    first_lines = [p.read_text().split("\n", 1)[0] for p in sorted(_SRC.glob("*.py"))]
    pattern = re.compile(r"\((?:[A-Z]{2}-\d+|GitLab issue #\d+)")
    assert [line for line in first_lines if pattern.search(line)] == []


def test_the_changelog_has_one_section_per_version_and_one_heading_per_kind() -> None:
    text = (Path(__file__).resolve().parent.parent / "CHANGELOG.md").read_text()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    assert sections[0].startswith("[Unreleased]")
    for section in sections[1:]:
        assert re.match(r"\[\d+\.\d+\.\d+\] - \d{4}-\d\d-\d\d\n", section), section[:40]
    for section in sections:
        kinds = re.findall(r"^### (.+)$", section, flags=re.MULTILINE)
        assert len(kinds) == len(set(kinds)), section[:40]
