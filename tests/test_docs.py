from pathlib import Path

import pytest

import doc_examples


@pytest.mark.parametrize("page", doc_examples.PAGES, ids=lambda p: p.name)
def test_every_example_in_the_methods_pages_prints_what_the_page_shows(page: Path) -> None:
    assert doc_examples.check(page, write=False) == []
