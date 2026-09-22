from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from asktwice.answer import AnswerCache, export_answers, run_jev, run_nli, split_rows
from asktwice.config import BGE_DIMS, HANDLABEL_QUESTIONS
from asktwice.data import filter_candidates, sample_handlabels, split_tables, write_splits
from asktwice.features import TooManyFailures
from asktwice.questions import load_questions
from asktwice.report import run_report
from asktwice.stats import calibrate_margin
from tests.helpers import LABEL2ID, PRODUCTS, FakeTok, fake_git, fake_jev_call, fake_nli_forward, row, write_cfpb_csv


def _narrative(i: int) -> str:
    return (
        f"On March {1 + i % 28} I called the bank about a {20 + i} dollar fee. "
        f"They kept the charge. I want a refund for complaint {i}."
    )


def _fixture_csv(path: Path) -> Path:
    rows = [
        row("drop-progress", "2021-01-01", PRODUCTS[0], _narrative(0), "In progress"),
        row("drop-credit", "2021-01-01", "Credit reporting or other personal consumer reports", _narrative(1), "Closed with explanation"),
    ]
    n = 0
    for split, start, count in (("train", 2019, 120), ("dev", 2023, 40), ("test", 2024, 40)):
        for i in range(count):
            n += 1
            year = start + (i % 4 if split == "train" else 0)
            rows.append(
                row(
                    f"{split}-{i}",
                    f"{year}-{1 + i % 12:02d}-15",
                    PRODUCTS[i % 3],
                    _narrative(n),
                    "Closed with monetary relief" if i % 5 == 0 else "Closed with explanation",
                    company=f"Bank {i % 8}",
                )
            )
    return write_cfpb_csv(path, rows)


def _fake_embed(texts):
    rng = np.random.default_rng(0)
    return {s: rng.normal(size=(len(v), BGE_DIMS)).astype(np.float32) for s, v in texts.items()}, 0.01


def _pipeline(tmp_path: Path) -> Path:
    spec = load_questions()
    cand = filter_candidates(_fixture_csv(tmp_path / "complaints.csv"), tmp_path / "candidates.parquet")
    splits = split_tables(pq.read_table(cand), sample=False)
    assert {k: v.num_rows for k, v in splits.items()} == {"train": 120, "dev": 40, "test": 40}
    frozen = tmp_path / "frozen"
    write_splits(splits, frozen)

    cache = AnswerCache(tmp_path / "answers.sqlite")
    rows = split_rows(frozen)
    run_jev(rows, cache, spec, call=fake_jev_call(spec), workers=4)
    run_nli(rows, cache, spec, tokenizer=FakeTok(), forward=fake_nli_forward, label2id=LABEL2ID, group=50)
    assert export_answers(frozen, cache, spec) == {"jev": 0, "nli": 0}

    labels = sample_handlabels(frozen, spec)
    with labels.open(encoding="utf-8") as f:
        judged = list(csv.DictReader(f))
    for i, r in enumerate(judged):
        r["label"] = str(i % 2)
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(judged[0]))
        w.writeheader()
        w.writerows(judged)
    return frozen


def test_fixture_end_to_end(tmp_path):
    frozen = _pipeline(tmp_path)

    margin = calibrate_margin(frozen, n_labels=40, n_boot=20)
    assert margin["pseudo_test_rows"] == 30 and margin["half_width"] >= 0

    out = tmp_path / "results"
    summary = run_report(frozen, out, embed=_fake_embed, run=fake_git(), label_counts=(50,), n_boot=20)
    for name in ("results.csv", "headline.json", "chart.png", "scoring_log.csv"):
        assert (out / name).stat().st_size > 0
    assert summary["headline"]["verdict"] in {"Jev better", "NLI better", "equivalent", "inconclusive"}
    assert summary["failures"]["jev"] == {"train": 1, "dev": 0, "test": 0}  # the fixture's one 422
    assert summary["rows_used"] == {"train": 119, "dev": 40, "test": 40}
    assert summary["answer_quality"]["status"] == "ok"
    assert summary["answer_quality"]["judgments"] == 40 * min(HANDLABEL_QUESTIONS, len(load_questions().noul))
    assert summary["single_window"]["rows"] == 40
    assert summary["arm4_variant"]["chosen"] in {"raw", "stacked"}
    assert summary["scoring_log"]["dirty"] == "false"
    arms = {r["arm"] for r in csv.DictReader((out / "results.csv").open(encoding="utf-8"))}
    assert {"tabular", "bge_tab", "nli20_tab", "jev20_tab", "jev_direct"} <= arms
    assert json.loads((out / "headline.json").read_text(encoding="utf-8"))["costs"]["jev_usd_per_10k_1x"] > 0


def test_report_stops_on_test_failures(tmp_path):
    frozen = _pipeline(tmp_path)
    t = pq.read_table(frozen / "jev_answers.parquet").to_pylist()
    for r in t:
        if r["complaint_id"] == "test-3":
            r["payload"], r["error"] = None, "422:too long"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(t), frozen / "jev_answers.parquet")
    with pytest.raises(TooManyFailures):
        run_report(frozen, tmp_path / "results", embed=_fake_embed, run=fake_git(), label_counts=(50,), n_boot=5)


def test_dev_check_flags_constant_questions(tmp_path):
    import pyarrow as pa

    from asktwice.features import check_dev
    from asktwice.stats import GuardError

    spec = load_questions()
    frozen = _pipeline(tmp_path)
    with pytest.raises(GuardError):
        check_dev(frozen, spec, run=fake_git(after_tag=False))  # blind until the tag
    assert all(not r["broken"] for r in check_dev(frozen, spec, run=fake_git()))

    q = spec.noul[0].id
    t = pq.read_table(frozen / "nli_answers.parquet")
    t = t.set_column(t.column_names.index(q), q, pa.array(np.full(t.num_rows, 0.5, dtype=np.float16)))
    pq.write_table(t, frozen / "nli_answers.parquet")
    rows = {r["question"]: r for r in check_dev(frozen, spec, run=fake_git())}
    assert rows[q]["broken"] == ["nli"]
    assert set(rows[q]) == {"question", "type", "broken", "jev", "nli"}  # no label information
    assert set(rows[q]["nli"]) == {"n", "mean", "std"}
