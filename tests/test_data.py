from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from asktwice.data import (
    drop_test_neardups,
    filter_candidates,
    hash_ids,
    product_relief_rates,
    sample_handlabels,
    split_tables,
    write_splits,
)
from tests.helpers import PRODUCTS, row, write_cfpb_csv


def _tiny_csv(tmp_path):
    rows = [
        row("1", "2020-06-01", PRODUCTS[0], "I was charged a thirty dollar overdraft fee.", "Closed with monetary relief"),
        row("2", "2021-06-01", PRODUCTS[1], "The card company raised my APR without notice.", "Closed with explanation"),
        row("3", "2023-03-01", PRODUCTS[2], "The wire never arrived and I want my money back.", "Closed with non-monetary relief"),
        row("4", "2025-02-01", PRODUCTS[0], "They froze my checking account after a small dispute.", "Closed with monetary relief"),
        row("5", "2021-01-01", PRODUCTS[0], "In progress should drop.", "In progress"),
        row("6", "2021-01-01", PRODUCTS[0], "Untimely should drop.", "Untimely response"),
        row("7", "2026-07-01", PRODUCTS[0], "After the cutoff should drop.", "Closed with explanation"),
        row("8", "2021-01-01", "Credit reporting or other personal consumer reports", "Template credit report complaint.", "Closed with explanation"),
        row("9", "2021-01-01", PRODUCTS[0], "", "Closed with explanation"),
        row("10", "2024-05-01", PRODUCTS[1], "I was charged a thirty dollar overdraft fee.", "Closed with explanation"),
    ]
    return write_cfpb_csv(tmp_path / "complaints.csv", rows)


def _splits(tmp_path):
    return split_tables(pq.read_table(filter_candidates(_tiny_csv(tmp_path), tmp_path / "c.parquet")), sample=False)


def test_filter_rules(tmp_path):
    table = pq.read_table(filter_candidates(_tiny_csv(tmp_path), tmp_path / "candidates.parquet"))
    assert set(table.column("complaint_id").to_pylist()) == {"1", "2", "3", "4", "10"}
    y = dict(zip(table.column("complaint_id").to_pylist(), table.column("y").to_pylist()))
    assert y["1"] == 1 and y["2"] == 0
    assert "response" not in table.column_names  # the response text is the label


def test_split_dates_and_unique_ids(tmp_path):
    splits = _splits(tmp_path)
    dates = {k: v.column("date_received").to_pylist() for k, v in splits.items()}
    assert max(dates["train"]) < min(dates["dev"]) < min(dates["test"])
    ids = [i for v in splits.values() for i in v.column("complaint_id").to_pylist()]
    assert len(ids) == len(set(ids))


def test_test_neardup_of_train_removed(tmp_path):
    assert "10" not in _splits(tmp_path)["test"].column("complaint_id").to_pylist()


def test_drop_test_neardups_direct():
    keep = drop_test_neardups(
        ["the bank took twenty dollars from my checking account yesterday"],
        ["unrelated dev narrative about a mortgage payoff"],
        [
            "the bank took twenty dollars from my checking account yesterday",
            "a completely different story about a lost debit card in spain",
        ],
    )
    assert list(keep) == [False, True]


def test_test_labels_own_file_and_stable_hashes(tmp_path):
    splits = _splits(tmp_path)
    hashes = write_splits(splits, tmp_path / "frozen")
    test = pq.read_table(tmp_path / "frozen" / "test.parquet")
    assert "y" not in test.column_names and "response" not in test.column_names
    labels = pq.read_table(tmp_path / "frozen" / "test_labels.parquet")
    assert set(labels.column_names) == {"complaint_id", "y"}
    ids = splits["train"].column("complaint_id").to_pylist()
    assert hashes["train"] == hash_ids(ids) == hash_ids(list(reversed(ids)))


def test_renamed_column_fails_loudly(tmp_path):
    path = _tiny_csv(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("Complaint ID", "complaint_id"), encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        filter_candidates(path, tmp_path / "x.parquet")


def test_product_rates_use_train_pool_only(tmp_path):
    rates = product_relief_rates(filter_candidates(_tiny_csv(tmp_path), tmp_path / "c.parquet"))
    assert sum(r["n"] for r in rates) == 2  # rows 1 and 2; dev and test years excluded
    assert {r["product"]: r["relief_rate"] for r in rates} == {PRODUCTS[0]: 1.0, PRODUCTS[1]: 0.0}


def test_handlabels_never_overwritten(tmp_path, questions):
    write_splits(_splits(tmp_path), tmp_path / "frozen")
    out = sample_handlabels(tmp_path / "frozen", questions)
    assert out.read_text(encoding="utf-8").startswith("complaint_id,question_id,question,narrative,label")
    with pytest.raises(FileExistsError):
        sample_handlabels(tmp_path / "frozen", questions)


def test_filter_reads_a_glob_of_exports(tmp_path):
    lines = _tiny_csv(tmp_path).read_text(encoding="utf-8").splitlines(keepends=True)
    (tmp_path / "ccdb").mkdir()
    (tmp_path / "ccdb" / "export_1.csv").write_text("".join(lines[:6]), encoding="utf-8")
    (tmp_path / "ccdb" / "export_2.csv").write_text(lines[0] + "".join(lines[6:]), encoding="utf-8")
    table = pq.read_table(filter_candidates(tmp_path / "ccdb" / "*.csv", tmp_path / "c.parquet"))
    assert set(table.column("complaint_id").to_pylist()) == {"1", "2", "3", "4", "10"}

    (tmp_path / "ccdb" / "export_3.csv").write_text(lines[0].replace("Tags", "Tag list") + lines[1], encoding="utf-8")
    with pytest.raises(ValueError, match="header differs"):
        filter_candidates(tmp_path / "ccdb" / "*.csv", tmp_path / "c.parquet")
