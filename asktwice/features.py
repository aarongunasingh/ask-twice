"""Answer encoding, tabular encoding, shared row mask, dev structural check."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from asktwice.answer import encode_noul_jev, encode_score_jev, nli_score_from_chunk_logits
from asktwice.config import BROKEN_STD, COMPANY_TOP_K, FAILURE_RATE_STOP, PREREG_TAG, SPLITS
from asktwice.gitio import head_after_tag
from asktwice.questions import QuestionSet

CAT_FIELDS = ("product", "sub_product", "issue", "sub_issue", "company", "state", "submitted_via", "tags")
OTHER = "other"
ANSWERERS = ("jev", "nli", "bge")


class TooManyFailures(RuntimeError):
    pass


def check_failure_rate(n_fail: int, n_total: int, answerer: str, limit: float = FAILURE_RATE_STOP) -> None:
    if n_total and n_fail / n_total > limit:
        raise TooManyFailures(
            f"{answerer} failed on {n_fail}/{n_total} test rows ({n_fail / n_total:.2%} > {limit:.0%})"
        )


def shared_mask(*ok_arrays: np.ndarray) -> np.ndarray:
    out = np.ones(len(ok_arrays[0]), dtype=bool)
    for arr in ok_arrays:
        out &= arr.astype(bool)
    return out


def month_of_year(value) -> int:
    if hasattr(value, "month"):
        return int(value.month)
    text = str(value)
    if len(text) >= 7 and text[4] == "-":
        return int(text[5:7])
    raise ValueError(f"cannot read month from {value!r}")


def _norm(value) -> str:
    return "" if value is None else str(value)


@dataclass
class TabularEncoder:
    company_keep: tuple[str, ...]
    vocabs: dict[str, dict[str, int]]

    def transform(self, rows: list[dict]) -> np.ndarray:
        x = np.zeros((len(rows), len(CAT_FIELDS) + 1), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, f in enumerate(CAT_FIELDS):
                raw = _norm(row.get(f))
                if f == "company" and raw not in self.company_keep:
                    raw = OTHER
                x[i, j] = self.vocabs[f].get(raw, self.vocabs[f][OTHER])
            x[i, len(CAT_FIELDS)] = month_of_year(row["date_received"])
        return x

    def max_codes(self) -> dict[str, int]:
        return {k: max(v.values()) for k, v in self.vocabs.items()}


def fit_tabular_encoder(train_rows: list[dict], k: int = COMPANY_TOP_K) -> TabularEncoder:
    counts: dict[str, int] = {}
    for row in train_rows:
        c = _norm(row.get("company"))
        counts[c] = counts.get(c, 0) + 1
    keep = tuple(sorted(counts, key=lambda x: (-counts[x], x))[:k])
    vocabs: dict[str, dict[str, int]] = {}
    for f in CAT_FIELDS:
        values = {_norm(r.get(f)) for r in train_rows}
        if f == "company":
            values = {v if v in keep else OTHER for v in values}
        vocabs[f] = {v: i for i, v in enumerate(sorted(values | {OTHER}))}
    return TabularEncoder(company_keep=keep, vocabs=vocabs)


def encode_jev_row(payload: dict, spec: QuestionSet) -> np.ndarray:
    return np.asarray(
        [encode_noul_jev(payload, q.id) if q.type == "noul" else encode_score_jev(payload, q.id) for q in spec.questions],
        dtype=np.float64,
    )


def encode_nli_row(row: dict, spec: QuestionSet) -> np.ndarray:
    """One exported nli_answers.parquet row -> 20 features, same encodings as Jev."""
    return np.asarray(
        [
            float(row[q.id])
            if q.type == "noul"
            else nli_score_from_chunk_logits(np.array([row[f"{q.id}@{i}"] for i in range(len(q.levels))], dtype=float))
            for q in spec.questions
        ],
        dtype=np.float64,
    )


@dataclass
class Built:
    """Everything the arms and stats need, after the shared row mask."""

    rows: dict[str, list[dict]]
    y: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    texts: dict[str, list[str]]
    single_window: np.ndarray  # test rows whose narrative fits one NLI window
    failures: dict[str, dict[str, int]]
    jev_input_tokens: float  # mean per row
    nli_seconds: float  # mean per row
    bge_seconds: float  # mean per row
    jev_test_payloads: list[dict] = field(default_factory=list)
    nli_test_rows: list[dict] = field(default_factory=list)


EmbedFn = Callable[[dict[str, list[str]]], tuple[dict[str, np.ndarray], float]]


def build(frozen: Path | str, spec: QuestionSet, y_test_by_id: dict[str, int], embed: EmbedFn) -> Built:
    frozen = Path(frozen)
    rows = {s: pq.read_table(frozen / f"{s}.parquet").to_pylist() for s in SPLITS}
    jev = {r["complaint_id"]: r for r in pq.read_table(frozen / "jev_answers.parquet").to_pylist()}
    nli = {r["complaint_id"]: r for r in pq.read_table(frozen / "nli_answers.parquet").to_pylist()}
    bge, bge_seconds = embed({s: [r["narrative"] for r in rows[s]] for s in SPLITS})
    enc = fit_tabular_encoder(rows["train"])  # company top-50 from the whole train pool

    out = Built(
        rows={}, y={}, features={}, texts={}, single_window=np.array([], dtype=bool),
        failures={a: {} for a in ANSWERERS}, jev_input_tokens=0.0, nli_seconds=0.0, bge_seconds=bge_seconds,
    )
    tokens, seconds = [], []
    for s in SPLITS:
        ids = [r["complaint_id"] for r in rows[s]]
        jrec = [jev.get(i) for i in ids]
        nrec = [nli.get(i) for i in ids]
        ok = {
            "jev": np.array([bool(r and r["payload"]) for r in jrec]),
            "nli": np.array([bool(r and r["ok"]) for r in nrec]),
            "bge": np.isfinite(bge[s]).all(axis=1),
        }
        for a in ANSWERERS:
            out.failures[a][s] = int((~ok[a]).sum())
            if s == "test":
                check_failure_rate(out.failures[a][s], len(ids), a)
        keep = np.flatnonzero(shared_mask(*ok.values()))
        kept = [rows[s][i] for i in keep]
        payloads = [json.loads(jrec[i]["payload"]) for i in keep]
        tokens += [jrec[i]["input_tokens"] for i in keep if jrec[i]["input_tokens"] is not None]
        seconds += [float(nrec[i]["seconds"]) for i in keep]

        sfx = "" if s == "train" else f"_{s}"
        out.rows[s] = kept
        out.texts[s] = [r["narrative"] for r in kept]
        out.y[s] = np.array(
            [y_test_by_id[r["complaint_id"]] for r in kept] if s == "test" else [r["y"] for r in kept], dtype=int
        )
        out.features["tab" + sfx] = enc.transform(kept)
        out.features["jev" + sfx] = np.stack([encode_jev_row(p, spec) for p in payloads])
        out.features["nli" + sfx] = np.stack([encode_nli_row(nrec[i], spec) for i in keep])
        out.features["bge" + sfx] = bge[s][keep]
        if s == "test":
            out.features["jev_direct_test"] = np.array([encode_noul_jev(p, spec.direct.id) for p in payloads])
            out.single_window = np.array([int(nrec[i]["max_chunks"]) == 1 for i in keep])
            out.jev_test_payloads = payloads
            out.nli_test_rows = [nrec[i] for i in keep]
    out.jev_input_tokens = float(np.mean(tokens)) if tokens else 0.0
    out.nli_seconds = float(np.mean(seconds)) if seconds else 0.0
    return out


def check_dev(frozen: Path | str, spec: QuestionSet, *, run=None) -> list[dict]:
    """Per question and answerer, the spread of encoded dev answers. Near-constant output marks
    a question structurally broken (plan: fixable as prereg-v2, before the first test prediction).
    Deliberately shows nothing about labels: an amendment must not be tuned to the outcome."""
    from asktwice.stats import GuardError

    if not head_after_tag(PREREG_TAG, run=run):
        raise GuardError(f"check_dev refused: answerer output is not looked at before {PREREG_TAG}")
    frozen = Path(frozen)
    ids = pq.read_table(frozen / "dev.parquet", columns=["complaint_id"]).column("complaint_id").to_pylist()
    jev = {r["complaint_id"]: r for r in pq.read_table(frozen / "jev_answers.parquet").to_pylist()}
    nli = {r["complaint_id"]: r for r in pq.read_table(frozen / "nli_answers.parquet").to_pylist()}
    cols = {
        "jev": [encode_jev_row(json.loads(jev[i]["payload"]), spec) for i in ids if jev.get(i) and jev[i]["payload"]],
        "nli": [encode_nli_row(nli[i], spec) for i in ids if nli.get(i) and nli[i]["ok"]],
    }
    out = []
    for k, q in enumerate(spec.questions):
        row: dict = {"question": q.id, "type": q.type, "broken": []}
        for a, rows in cols.items():
            x = np.array([r[k] for r in rows])
            row[a] = {"n": len(x), "mean": float(x.mean()) if len(x) else None, "std": float(x.std()) if len(x) else None}
            if not len(x) or x.std() < BROKEN_STD:
                row["broken"].append(a)
        out.append(row)
    return out
