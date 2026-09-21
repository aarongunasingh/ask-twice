"""Bootstrap CIs, paired deltas, verdicts, scoring log, test-label guard, margin calibration."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from asktwice.config import (
    EQUIVALENCE_MARGIN,
    FROZEN_DIR,
    MARGIN_HALF_WIDTH_MAX,
    N_BOOT,
    PREREG_TAG,
    PSEUDO_TEST_YEAR,
)
from asktwice.gitio import head_after_tag, snapshot


class GuardError(RuntimeError):
    pass


def load_test_labels(path: Path | str, *, cwd: Path | None = None, run=None) -> dict[str, int]:
    """complaint_id -> y. The only reader of test_labels.parquet; refuses before the prereg tag."""
    if not head_after_tag(PREREG_TAG, cwd=cwd, run=run):
        raise GuardError(f"load_test_labels() refused: HEAD is not after {PREREG_TAG}")
    t = pq.read_table(path)
    return dict(zip(t.column("complaint_id").to_pylist(), map(int, t.column("y").to_pylist())))


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))


def seed_mean(metric, y: np.ndarray, scores: np.ndarray) -> float:
    """Point estimate: the metric per seed, averaged. scores shape (n_seeds, n_test)."""
    return float(np.mean([metric(y, s) for s in np.atleast_2d(scores)]))


def make_resamples(
    n_test: int,
    n_seeds: int,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    rows = rng.integers(0, n_test, size=(n_boot, n_test))
    seeds = rng.integers(0, max(n_seeds, 1), size=n_boot)
    return rows, seeds


def _safe_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float("nan") if y.min() == y.max() else pr_auc(y, scores)


def bootstrap_scores(y: np.ndarray, scores: np.ndarray, rows: np.ndarray, seed_idx: np.ndarray) -> np.ndarray:
    """PR-AUC per resample; each resample draws test rows and one seed. scores shape (n_seeds, n_test)."""
    scores = np.atleast_2d(scores)
    return np.array([_safe_pr_auc(y[r], scores[s % len(scores), r]) for r, s in zip(rows, seed_idx)])


def ci95(samples: np.ndarray) -> tuple[float, float, float]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(np.mean(finite)), float(lo), float(hi)


def paired_delta(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, rows: np.ndarray, seed_idx: np.ndarray) -> np.ndarray:
    return bootstrap_scores(y, scores_a, rows, seed_idx) - bootstrap_scores(y, scores_b, rows, seed_idx)


def verdict(lo: float, hi: float, margin: float = EQUIVALENCE_MARGIN) -> str:
    """Plan table order: a significant difference is reported before equivalence."""
    if lo > 0:
        return "Jev better"
    if hi < 0:
        return "NLI better"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    return "inconclusive"


def stops_beating(lo_by_n: dict[int, float]) -> int | str | None:
    ordered = sorted(lo_by_n.items())
    beating = [n for n, lo in ordered if lo > 0]
    if not beating:
        return "never beats"
    for n, lo in ordered:
        if n > beating[0] and lo <= 0:
            return n
    return None


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    which = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[which == b].mean() - p[which == b].mean()) * (which == b).mean() for b in np.unique(which)))


LOG_FIELDS = ("utc", "commit", "dirty", "diff_hash")


def append_scoring_log(path: Path | str, *, cwd: Path | None = None, run=None) -> dict:
    snap = snapshot(cwd=cwd, run=run)
    row = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": snap["commit"],
        "dirty": str(snap["dirty"]).lower(),
        "diff_hash": snap["diff_hash"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def calibrate_margin(
    frozen: Path | str,
    *,
    n_labels: int = 1_000,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> dict:
    """Plan: 2022 train-pool rows as a pseudo-test; paired bootstrap of arm 1 vs tabular +
    a TF-IDF score at 1k labels; the delta CI half-width sizes the equivalence margin."""
    from asktwice.arms import ARMS_BY_ID, TFIDF_TAB, fit_eval, schedule
    from asktwice.features import fit_tabular_encoder

    frozen = Path(frozen)
    train = pq.read_table(frozen / "train.parquet").to_pylist()
    dev = pq.read_table(frozen / "dev.parquet").to_pylist()
    part = {
        "train": [r for r in train if r["date_received"].year != PSEUDO_TEST_YEAR],
        "dev": dev,
        "test": [r for r in train if r["date_received"].year == PSEUDO_TEST_YEAR],
    }
    enc = fit_tabular_encoder(part["train"])
    features = {"tab" + ("" if k == "train" else f"_{k}"): enc.transform(v) for k, v in part.items()}
    texts = {k: [r["narrative"] for r in v] for k, v in part.items()}
    y = {k: np.array([r["y"] for r in v], dtype=int) for k, v in part.items()}

    n, seeds = schedule(n_labels, len(y["train"]))
    preds = {
        arm.id: np.stack(
            [
                fit_eval(arm, n, s, y_train=y["train"], y_dev=y["dev"], features=features, texts=texts)["pred"]
                for s in range(seeds)
            ]
        )
        for arm in (ARMS_BY_ID[1], TFIDF_TAB)
    }
    rows, seed_idx = make_resamples(len(y["test"]), seeds, n_boot, rng)
    mu, lo, hi = ci95(paired_delta(y["test"], preds[TFIDF_TAB.id], preds[1], rows, seed_idx))
    half = (hi - lo) / 2
    return {
        "pseudo_test_rows": len(y["test"]),
        "labels": n,
        "seeds": seeds,
        "delta": mu,
        "lo": lo,
        "hi": hi,
        "half_width": half,
        "within_limit": half <= MARGIN_HALF_WIDTH_MAX,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asktwice.stats")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="size the equivalence margin before the tag")
    c.add_argument("--frozen", default=str(FROZEN_DIR))
    args = p.parse_args(argv)
    out = calibrate_margin(args.frozen)
    print(json.dumps(out, indent=2))
    if not out["within_limit"]:
        print(f"half-width > {MARGIN_HALF_WIDTH_MAX}: grow the test set or widen the margin, and record why in the tag")


if __name__ == "__main__":
    main()
