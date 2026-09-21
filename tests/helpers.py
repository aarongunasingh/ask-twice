from __future__ import annotations

import csv
import zlib
from pathlib import Path

import numpy as np

from asktwice.config import BGE_DIMS, JEV_MODEL, REQUIRED_CSV_COLUMNS

PRODUCTS = (
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
)
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
CLS, SEP, PAD = 1, 2, 0


class FakeTok:
    """The slice of the HuggingFace tokenizer contract the pipeline uses."""

    pad_token_id, cls_token_id, sep_token_id = PAD, CLS, SEP

    def encode(self, text, add_special_tokens=False):
        return [3 + zlib.crc32(w.encode()) % 997 for w in str(text).split()]


def fake_nli_forward(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deterministic logits that depend on the tokens."""
    s = (ids * mask).sum(axis=1) % 11 / 11.0
    return np.stack([4 * s - 2, np.zeros_like(s), 2 - 4 * s], axis=1)


def fake_bge_forward(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = (ids * mask).sum(axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True)
    return np.sin(s * np.linspace(0.01, 1.0, BGE_DIMS))


def fake_jev_call(spec):
    """Jev-shaped responses; one narrative triggers a 422."""
    from asktwice.answer import JevHttpError

    def call(state, questions, model):
        if state.endswith("complaint 7."):
            raise JevHttpError(422, "input too long")
        rng = np.random.default_rng(zlib.crc32(state.encode()))
        answers = {}
        for key, q in questions.items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": float(rng.random())}
            else:
                p = rng.random(len(q["criteria"]))
                p /= p.sum()
                answers[key] = {"type": "score", "probabilities": {str(i): float(v) for i, v in enumerate(p)}}
        return {"model": JEV_MODEL, "answers": answers, "usage": {"input_tokens": 40 + len(state.split()), "output_tokens": 1}}

    return call


class FakeProc:
    def __init__(self, code=0, stdout=""):
        self.returncode = code
        self.stdout = stdout
        self.stderr = ""


def fake_git(*, after_tag=True, dirty=False):
    def run(args, **kwargs):
        if args[:2] == ["git", "merge-base"]:
            return FakeProc(0 if after_tag else 1)
        if args[:2] == ["git", "rev-parse"]:
            return FakeProc(0, "abc123\n")
        if args[:2] == ["git", "status"]:
            return FakeProc(0, " M asktwice/data.py\n" if dirty else "")
        return FakeProc(0, "")

    return run


def write_cfpb_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = [
        "Company public response",
        "ZIP code",
        "Consumer consent provided?",
        "Date sent to company",
        "Timely response?",
        "Consumer disputed?",
    ]
    fieldnames = list(REQUIRED_CSV_COLUMNS) + [c for c in extras if c not in REQUIRED_CSV_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in fieldnames}
            full.update(r)
            w.writerow(full)
    return path


def row(
    cid: str,
    date: str,
    product: str,
    narrative: str,
    response: str = "Closed with explanation",
    company: str = "Bank A",
    **kwargs,
) -> dict:
    return {
        "Complaint ID": cid,
        "Date received": date,
        "Product": product,
        "Sub-product": kwargs.get("sub_product", "Checking account"),
        "Issue": kwargs.get("issue", "Managing an account"),
        "Sub-issue": kwargs.get("sub_issue", "Deposits and withdrawals"),
        "Consumer complaint narrative": narrative,
        "Company": company,
        "State": kwargs.get("state", "CA"),
        "Tags": kwargs.get("tags", ""),
        "Submitted via": kwargs.get("submitted_via", "Web"),
        "Company response to consumer": response,
        "ZIP code": "94107",
        "Company public response": "",
        "Consumer consent provided?": "Consent provided",
        "Date sent to company": date,
        "Timely response?": "Yes",
        "Consumer disputed?": "No",
    }
