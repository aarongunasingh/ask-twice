"""DuckDB filter, MinHash dedup, time split, split hashes, hand-label sample."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from asktwice.config import (
    CSV_COLUMNS,
    DATA_DIR,
    DEDUP_JACCARD,
    DEDUP_SHINGLE_N,
    DEV_END,
    DEV_START,
    FROZEN_DIR,
    HANDLABEL_QUESTIONS,
    HANDLABEL_ROWS,
    HANDLABEL_SEED,
    MINHASH_PERM,
    PRODUCT_CANDIDATES,
    RECEIVED_CUTOFF,
    RELIEF_RATE_HI,
    RELIEF_RATE_LO,
    RELIEF_RESPONSE,
    REQUIRED_CSV_COLUMNS,
    RESPONSE_KEEP,
    SAMPLE_DEV,
    SAMPLE_TEST,
    SAMPLE_TRAIN,
    SELECTED_PRODUCTS,
    SPLIT_SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)
from asktwice.questions import QuestionSet, load_questions

DATE_SQL = """
COALESCE(
  TRY_STRPTIME("Date received", '%Y-%m-%d'),
  TRY_STRPTIME("Date received", '%m/%d/%Y'),
  TRY_STRPTIME("Date received", '%m/%d/%y')
)::DATE
"""


def _sql_list(values: tuple[str, ...] | list[str]) -> str:
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + ")"


def csv_headers(csv_path: Path | str) -> list[str]:
    con = duckdb.connect()
    rel = con.execute("SELECT * FROM read_csv(?, header=true, sample_size=1) LIMIT 0", [str(csv_path)])
    names = [c[0] for c in rel.description]
    con.close()
    return names


def filter_candidates(
    csv_path: Path | str,
    out_parquet: Path | str,
    *,
    products: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Typed DuckDB read. Never pandas. Writes candidates.parquet.
    csv_path may be a glob (the narratives archive ships as 21 exports)."""
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(csv_path))) or [str(csv_path)]
    names = csv_headers(files[0])
    # Explicit columns map by position, so every file must carry the same header.
    for f in files[1:]:
        if csv_headers(f) != names:
            raise ValueError(f"CFPB CSV header differs from {files[0]}: {f}")
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in names]
    if missing:
        raise ValueError(f"CFPB CSV missing columns: {missing}")
    keep = products or SELECTED_PRODUCTS or PRODUCT_CANDIDATES
    cols = ", ".join("'" + k.replace("'", "''") + "': '" + CSV_COLUMNS.get(k, "VARCHAR") + "'" for k in names)
    # The response column itself is not kept: it is the label.
    sql = f"""
    SELECT
      "Complaint ID"::VARCHAR AS complaint_id,
      {DATE_SQL} AS date_received,
      Product AS product,
      "Sub-product" AS sub_product,
      Issue AS issue,
      "Sub-issue" AS sub_issue,
      "Consumer complaint narrative" AS narrative,
      Company AS company,
      State AS state,
      Tags AS tags,
      "Submitted via" AS submitted_via,
      ("Company response to consumer" = '{RELIEF_RESPONSE}')::INTEGER AS y
    FROM read_csv(?, header = true, delim = ',', quote = '"', columns = {{{cols}}})
    WHERE "Consumer complaint narrative" IS NOT NULL
      AND length(trim("Consumer complaint narrative")) > 0
      AND "Company response to consumer" IN {_sql_list(RESPONSE_KEEP)}
      AND {DATE_SQL} >= DATE '{TRAIN_START}'
      AND {DATE_SQL} <= DATE '{RECEIVED_CUTOFF}'
      AND NOT regexp_matches(Product, '(?i)credit report')
      AND Product IN {_sql_list(tuple(keep))}
    """
    con = duckdb.connect()
    con.execute(f"COPY ({sql}) TO '{out.as_posix().replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET)", [files])
    con.close()
    return out


def product_relief_rates(candidates: Path | str) -> list[dict]:
    """Relief rate by product on the train pool only, so product choice never sees test years."""
    con = duckdb.connect()
    rel = con.execute(
        f"""
        SELECT product, count(*) AS n, avg(y) AS relief_rate
        FROM read_parquet(?)
        WHERE date_received BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
        GROUP BY product
        ORDER BY n DESC
        """,
        [str(candidates)],
    ).fetchall()
    con.close()
    return [
        {"product": p, "n": int(n), "relief_rate": float(r), "in_band": RELIEF_RATE_LO <= r <= RELIEF_RATE_HI}
        for p, n, r in rel
    ]


def shingles(text: str, n: int = DEDUP_SHINGLE_N) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _minhash(tokens: set[str]) -> MinHash:
    m = MinHash(num_perm=MINHASH_PERM)
    m.update_batch([t.encode("utf-8") for t in tokens])
    return m


def drop_test_neardups(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    *,
    threshold: float = DEDUP_JACCARD,
) -> np.ndarray:
    """True for test rows that are NOT a near-duplicate of train or dev."""
    refs = [*train_texts, *dev_texts]
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_PERM)
    for i, text in enumerate(refs):
        lsh.insert(i, _minhash(shingles(text)))
    keep = np.ones(len(test_texts), dtype=bool)
    for j, text in enumerate(test_texts):
        s = shingles(text)
        # LSH is approximate: confirm hits with exact Jaccard. Ref shingles are
        # recomputed per hit rather than held in memory for the whole pool.
        keep[j] = not any(jaccard(s, shingles(refs[i])) >= threshold for i in lsh.query(_minhash(s)))
    return keep


def hash_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for i in sorted(ids):
        h.update(str(i).encode())
        h.update(b"\n")
    return h.hexdigest()


def _take(table: pa.Table, idx: np.ndarray) -> pa.Table:
    return table.take(pa.array(np.asarray(idx, dtype=np.int64)))


def _in_dates(table: pa.Table, start: str, end: str) -> np.ndarray:
    dates = np.array(table.column("date_received").to_pylist(), dtype="datetime64[D]")
    return np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))


def split_tables(candidates: pa.Table, *, sample: bool = True, seed: int = SPLIT_SEED) -> dict[str, pa.Table]:
    train = _take(candidates, _in_dates(candidates, TRAIN_START, TRAIN_END))
    dev = _take(candidates, _in_dates(candidates, DEV_START, DEV_END))
    test = _take(candidates, _in_dates(candidates, TEST_START, TEST_END))

    keep = drop_test_neardups(
        [t or "" for t in train.column("narrative").to_pylist()],
        [t or "" for t in dev.column("narrative").to_pylist()],
        [t or "" for t in test.column("narrative").to_pylist()],
    )
    test = _take(test, np.flatnonzero(keep))

    splits = {"train": train, "dev": dev, "test": test}
    if sample:
        # Plain random samples: the test draw never looks at test labels.
        rng = np.random.default_rng(seed)
        for name, n in (("train", SAMPLE_TRAIN), ("dev", SAMPLE_DEV), ("test", SAMPLE_TEST)):
            tbl = splits[name]
            if len(tbl) > n:
                splits[name] = _take(tbl, np.sort(rng.choice(len(tbl), size=n, replace=False)))

    ids = {k: v.column("complaint_id").to_pylist() for k, v in splits.items()}
    for name, lst in ids.items():
        if len(lst) != len(set(lst)):
            raise ValueError(f"duplicate complaint_id in {name}")
    if sum(len(set(v)) for v in ids.values()) != len(set().union(*map(set, ids.values()))):
        raise ValueError("complaint_id appears in two splits")
    return splits


def write_splits(splits: dict[str, pa.Table], out_dir: Path | str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: hash_ids([str(i) for i in tbl.column("complaint_id").to_pylist()]) for name, tbl in splits.items()}
    for name, tbl in splits.items():
        if name == "test":
            pq.write_table(tbl.select(["complaint_id", "y"]), out / "test_labels.parquet", compression="zstd")
            tbl = tbl.drop_columns(["y"])
        pq.write_table(tbl, out / f"{name}.parquet", compression="zstd")
    (out / "splits.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return hashes


def sample_handlabels(frozen: Path | str, spec: QuestionSet, *, seed: int = HANDLABEL_SEED) -> Path:
    """100 test rows x 5 Noul questions to hand-label after the tag, before any answerer runs."""
    out = Path(frozen) / "handlabels.csv"
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite hand labels")
    test = pq.read_table(Path(frozen) / "test.parquet", columns=["complaint_id", "narrative"]).to_pylist()
    rng = np.random.default_rng(seed)
    rows = [test[i] for i in np.sort(rng.choice(len(test), size=min(HANDLABEL_ROWS, len(test)), replace=False))]
    noul = spec.noul
    qs = [noul[i] for i in np.sort(rng.choice(len(noul), size=min(HANDLABEL_QUESTIONS, len(noul)), replace=False))]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["complaint_id", "question_id", "question", "narrative", "label"])
        for r in rows:
            for q in qs:
                w.writerow([r["complaint_id"], q.id, q.jev, r["narrative"], ""])
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asktwice.data")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--csv", required=True)
    f.add_argument("--out", default=str(DATA_DIR / "candidates.parquet"))
    r = sub.add_parser("rates")
    r.add_argument("--candidates", default=str(DATA_DIR / "candidates.parquet"))
    s = sub.add_parser("split")
    s.add_argument("--candidates", default=str(DATA_DIR / "candidates.parquet"))
    s.add_argument("--out", default=str(FROZEN_DIR))
    s.add_argument("--no-sample", action="store_true")
    h = sub.add_parser("handlabels")
    h.add_argument("--frozen", default=str(FROZEN_DIR))
    args = p.parse_args(argv)
    if args.cmd == "filter":
        print(filter_candidates(args.csv, args.out))
    elif args.cmd == "rates":
        for row in product_relief_rates(args.candidates):
            flag = "KEEP" if row["in_band"] else "skip"
            print(f"{flag}\t{row['n']:8d}\t{row['relief_rate']:.3f}\t{row['product']}")
    elif args.cmd == "split":
        splits = split_tables(pq.read_table(args.candidates), sample=not args.no_sample)
        for k, v in write_splits(splits, args.out).items():
            print(k, len(splits[k]), v)
    elif args.cmd == "handlabels":
        print(sample_handlabels(args.frozen, load_questions()))


if __name__ == "__main__":
    main()
