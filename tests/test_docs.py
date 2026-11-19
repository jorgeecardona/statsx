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
