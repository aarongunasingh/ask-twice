from pathlib import Path

import pytest

from asktwice.questions import load_questions


@pytest.fixture(autouse=True)
def all_candidate_products(monkeypatch):
    # The fixtures cover every candidate product; the Day 0 selection is for the real data only.
    monkeypatch.setattr("asktwice.data.SELECTED_PRODUCTS", None)


@pytest.fixture
def questions():
    return load_questions()


@pytest.fixture
def tmp_cache(tmp_path: Path):
    from asktwice.answer import AnswerCache

    return AnswerCache(tmp_path / "answers.sqlite")
