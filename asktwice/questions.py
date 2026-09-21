"""Load and hash the frozen question set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from asktwice.config import N_NOUL, N_SCORE, ROOT

DEFAULT_PATH = ROOT / "questions.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    jev: str
    nli_hypothesis: str | None = None
    levels: tuple[str, ...] = ()
    nli_hypotheses: tuple[str, ...] = ()
    criteria: tuple[str, str] | None = None  # Noul only: what (yes, no) mean, for boundary cases

    def to_canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "jev": self.jev,
            "nli_hypothesis": self.nli_hypothesis,
            "levels": list(self.levels),
            "nli_hypotheses": list(self.nli_hypotheses),
            "criteria": list(self.criteria) if self.criteria else None,
        }

    def jev_question(self) -> dict[str, Any]:
        if self.type == "score":
            return {"type": "score", "instructions": self.jev, "criteria": list(self.levels)}
        out: dict[str, Any] = {"type": "noul", "instructions": self.jev}
        if self.criteria:
            out["criteria"] = {"true": self.criteria[0], "false": self.criteria[1]}
        return out


@dataclass(frozen=True)
class QuestionSet:
    status: str
    direct: Question
    questions: tuple[Question, ...]

    @property
    def noul(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.type == "noul")

    @property
    def score(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.type == "score")


def _parse_one(raw: dict[str, Any]) -> Question:
    qtype = raw["type"]
    if qtype not in {"noul", "score"}:
        raise ValueError(f"unknown question type: {qtype}")
    levels = tuple(raw.get("levels") or ())
    hyps = tuple(raw.get("nli_hypotheses") or ())
    if qtype == "score":
        if not (2 <= len(levels) <= 10):
            raise ValueError(f"{raw.get('id')}: Score needs 2–10 levels")
        if hyps and len(hyps) != len(levels):
            raise ValueError(f"{raw.get('id')}: nli_hypotheses must match levels")
        if not hyps:
            hyps = levels
    criteria = None
    if raw.get("criteria"):
        if qtype != "noul":
            raise ValueError(f"{raw.get('id')}: criteria is for noul; score uses levels")
        c = raw["criteria"]  # YAML reads an unquoted `true:` key as a boolean
        yes, no = c.get("true", c.get(True)), c.get("false", c.get(False))
        if not (yes and no):
            raise ValueError(f"{raw.get('id')}: noul criteria needs both true and false")
        criteria = (str(yes), str(no))
    return Question(
        id=str(raw["id"]),
        type=qtype,
        jev=str(raw["jev"]),
        nli_hypothesis=raw.get("nli_hypothesis"),
        levels=levels,
        nli_hypotheses=hyps,
        criteria=criteria,
    )


def load_questions(path: Path | str | None = None) -> QuestionSet:
    p = Path(path) if path is not None else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    direct = _parse_one(raw["direct"])
    questions = tuple(_parse_one(q) for q in raw["questions"])
    ids = [direct.id, *[q.id for q in questions]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    missing = [q.id for q in questions if q.type == "noul" and not q.nli_hypothesis]
    if missing:
        raise ValueError(f"noul questions need an nli_hypothesis: {missing}")
    spec = QuestionSet(status=str(raw.get("status", "")), direct=direct, questions=questions)
    if spec.status == "frozen" and (len(spec.noul), len(spec.score)) != (N_NOUL, N_SCORE):
        raise ValueError(f"a frozen set needs {N_NOUL} noul + {N_SCORE} score questions")
    return spec


def question_set_hash(spec: QuestionSet) -> str:
    payload = {
        "direct": spec.direct.to_canonical(),
        "questions": {q.id: q.to_canonical() for q in spec.questions},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def jev_payload(spec: QuestionSet) -> dict[str, Any]:
    return {q.id: q.jev_question() for q in (*spec.questions, spec.direct)}
