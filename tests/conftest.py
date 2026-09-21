from pathlib import Path

import pytest

from asktwice.questions import load_questions


@pytest.fixture
def questions():
    return load_questions()


@pytest.fixture
def tmp_cache(tmp_path: Path):
    from asktwice.answer import AnswerCache

    return AnswerCache(tmp_path / "answers.sqlite")
