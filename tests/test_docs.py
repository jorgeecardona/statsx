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
