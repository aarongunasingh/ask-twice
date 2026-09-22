"""One command: committed splits + answers -> results.csv, headline.json, chart.png, scoring log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from asktwice.arms import ARMS, ARMS_BY_ID, fit_eval, schedule, select_arm4_variant
from asktwice.config import (
    DATA_DIR,
    EQUIVALENCE_MARGIN,
    FRONTIER_PRICE_PER_M_INPUT,
    FROZEN_DIR,
    JEV_MODEL,
    JEV_PRICE_PER_M_INPUT,
    LABEL_COUNTS,
    N_BOOT,
    RESULTS_DIR,
    SPLITS,
)
from asktwice.features import Built, EmbedFn, build
from asktwice.questions import load_questions
from asktwice.stats import (
    append_scoring_log,
    bootstrap_scores,
    ci95,
    expected_calibration_error,
    load_test_labels,
    make_resamples,
    paired_delta,
    pr_auc,
    roc_auc,
    seed_mean,
    stops_beating,
    verdict,
)

HEADLINE_N = 1_000


def cost_per_10k(mean_input_tokens: float, price_per_m: float) -> float:
    return 10_000 * mean_input_tokens * price_per_m / 1_000_000


def bge_embed(cache: Path = DATA_DIR / "bge.npz") -> EmbedFn:
    """Real bge-small, cached locally (not committed) and keyed by the exact texts."""

    def embed(texts: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], float]:
        key = hashlib.sha256("\x00".join(t for s in SPLITS for t in texts[s]).encode()).hexdigest()
        if cache.exists():
            z = np.load(cache)
            if str(z["key"]) == key:
                return {s: z[s] for s in SPLITS}, float(z["seconds"])
        from asktwice.answer import embed_texts, load_bge

        tok, fwd = load_bge()
        t0 = time.perf_counter()
        out = {s: embed_texts(texts[s], tokenizer=tok, forward=fwd) for s in SPLITS}
        seconds = (time.perf_counter() - t0) / sum(len(v) for v in texts.values())
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=key, seconds=seconds, **out)
        return out, seconds

    return embed


def run_grid(built: Built, label_counts: tuple[int, ...], arm4_variant: str) -> tuple[dict, dict]:
    """preds[(arm_id, n)] -> (seeds, n_test); plan[n] -> (labels drawn, seeds)."""
    preds: dict[tuple[int, int], np.ndarray] = {}
    plan = {n: schedule(n, len(built.y["train"])) for n in label_counts}
    for n, (drawn, seeds) in plan.items():
        for arm in ARMS:
            preds[(arm.id, n)] = np.stack(
                [
                    fit_eval(
                        arm,
                        drawn,
                        s,
                        y_train=built.y["train"],
                        y_dev=built.y["dev"],
                        features=built.features,
                        texts=built.texts,
                        arm4_variant=arm4_variant,
                    )["pred"]
                    for s in range(seeds)
                ]
            )
    return preds, plan


def _delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, boot_a: np.ndarray, boot_b: np.ndarray) -> dict:
    _, lo, hi = ci95(boot_a - boot_b)
    return {"delta": seed_mean(pr_auc, y, a) - seed_mean(pr_auc, y, b), "lo": lo, "hi": hi}


def subset_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray, n_boot: int, rng) -> dict:
    """Paired bootstrap restricted to a subset of test rows (its own resample matrix)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0 or y[idx].min() == y[idx].max():
        return {"rows": int(len(idx)), "delta": None, "lo": None, "hi": None, "verdict": "too few rows"}
    a, b = np.atleast_2d(a)[:, idx], np.atleast_2d(b)[:, idx]
    rows, seed_idx = make_resamples(len(idx), max(len(a), len(b)), n_boot, rng)
    _, lo, hi = ci95(paired_delta(y[idx], a, b, rows, seed_idx))
    d = seed_mean(pr_auc, y[idx], a) - seed_mean(pr_auc, y[idx], b)
    return {"rows": int(len(idx)), "delta": d, "lo": lo, "hi": hi, "verdict": verdict(lo, hi)}


def subset_pr_auc(y: np.ndarray, scores: np.ndarray, mask: np.ndarray, n_boot: int, rng) -> dict:
    idx = np.flatnonzero(mask)
    if len(idx) == 0 or y[idx].min() == y[idx].max():
        return {"rows": int(len(idx)), "pr_auc": None, "lo": None, "hi": None}
    s = np.atleast_2d(scores)[:, idx]
    rows, seed_idx = make_resamples(len(idx), len(s), n_boot, rng)
    _, lo, hi = ci95(bootstrap_scores(y[idx], s, rows, seed_idx))
    return {"rows": int(len(idx)), "pr_auc": seed_mean(pr_auc, y[idx], s), "lo": lo, "hi": hi}


def half_year(d) -> str:
    return f"{d.year}H{1 if d.month <= 6 else 2}"


def answer_quality(frozen: Path, built: Built) -> dict:
    """Per-question accuracy (Jev, NLI) and Jev ECE on the hand-labelled 100 x 5."""
    path = frozen / "handlabels.csv"
    if not path.exists():
        return {"status": "no handlabels.csv"}
    with path.open(encoding="utf-8-sig") as f:  # Excel's "CSV UTF-8" adds a BOM
        labels = list(csv.DictReader(f))
    blank = sum(not r["label"].strip() for r in labels)
    if blank:
        return {"status": f"{blank} of {len(labels)} judgments unlabelled"}
    pos = {r["complaint_id"]: i for i, r in enumerate(built.rows["test"])}
    rows = [r for r in labels if r["complaint_id"] in pos]
    y = np.array([int(r["label"]) for r in rows])
    p_jev = np.array([built.jev_test_payloads[pos[r["complaint_id"]]]["answers"][r["question_id"]]["noul"] for r in rows])
    p_nli = np.array([float(built.nli_test_rows[pos[r["complaint_id"]]][r["question_id"]]) for r in rows])
    qids = [r["question_id"] for r in rows]
    per_q = {}
    for q in sorted(set(qids)):
        m = np.array([x == q for x in qids])
        per_q[q] = {
            "n": int(m.sum()),
            "jev_acc": float(((p_jev[m] >= 0.5) == y[m]).mean()),
            "nli_acc": float(((p_nli[m] >= 0.5) == y[m]).mean()),
        }
    return {"status": "ok", "judgments": len(rows), "per_question": per_q, "jev_ece": expected_calibration_error(y, p_jev)}


def render_chart(results: list[dict], headline: dict, path: Path | str, *, costs: dict[str, str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = costs or {}
    styles = {0: ":", 1: "--", 4: "-", 8: "-", 9: "-"}
    labels = {0: "base rate", 1: "tabular only", 4: "bge + tabular", 8: "NLI-20 + tabular", 9: "Jev-20 + tabular"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm_id, ls in styles.items():
        pts = sorted((r for r in results if r["arm_id"] == arm_id), key=lambda r: r["n_labels"])
        if not pts:
            continue
        xs = [r["n_labels"] for r in pts]
        name = labels[arm_id]
        if costs.get(ARMS_BY_ID[arm_id].name):
            name = f"{name} ({costs[ARMS_BY_ID[arm_id].name]})"
        (line,) = ax.plot(xs, [r["pr_auc"] for r in pts], linestyle=ls, marker="o", label=name)
        ax.fill_between(xs, [r["pr_auc_lo"] for r in pts], [r["pr_auc_hi"] for r in pts], color=line.get_color(), alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks([300, 1000, 3000, 10000], ["300", "1k", "3k", "10k"])
    ax.set_xlabel("training labels")
    ax.set_ylabel("test PR-AUC (95% CI)")
    ax.set_title(
        f"Jev − NLI at {headline['n_labels']} labels: {headline['delta']:+.3f} "
        f"[{headline['lo']:+.3f}, {headline['hi']:+.3f}] → {headline['verdict']}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


RESULT_FIELDS = (
    "arm_id", "arm", "n_labels", "labels_drawn", "seeds", "pr_auc", "pr_auc_lo", "pr_auc_hi",
    "roc_auc", "lift_vs_tabular", "lift_lo", "lift_hi", "chart",
)


def run_report(
    frozen: Path | str = FROZEN_DIR,
    out: Path | str = RESULTS_DIR,
    *,
    embed: EmbedFn | None = None,
    run=None,
    label_counts: tuple[int, ...] = LABEL_COUNTS,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    frozen, out = Path(frozen), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_questions()
    y_test_by_id = load_test_labels(frozen / "test_labels.parquet", run=run)  # guard
    log = append_scoring_log(out / "scoring_log.csv", run=run)
    built = build(frozen, spec, y_test_by_id, embed or bge_embed())
    y = built.y["test"]
    rng = np.random.default_rng(seed)

    n_select = HEADLINE_N if HEADLINE_N in label_counts else label_counts[0]
    arm4_variant, arm4_dev = select_arm4_variant(
        n_select, y_train=built.y["train"], y_dev=built.y["dev"], features=built.features
    )
    preds, plan = run_grid(built, label_counts, arm4_variant)

    # One resample matrix shared by every arm and label count: all comparisons are paired.
    rows, seed_idx = make_resamples(len(y), max(len(v) for v in preds.values()), n_boot, rng)
    boots = {k: bootstrap_scores(y, v, rows, seed_idx) for k, v in preds.items()}

    results = []
    for arm in ARMS:
        for n in label_counts:
            k = (arm.id, n)
            _, lo, hi = ci95(boots[k])
            lift = _delta(y, preds[k], preds[(1, n)], boots[k], boots[(1, n)])
            results.append(
                {
                    "arm_id": arm.id,
                    "arm": arm.name,
                    "n_labels": n,
                    "labels_drawn": plan[n][0],
                    "seeds": plan[n][1],
                    "pr_auc": seed_mean(pr_auc, y, preds[k]),
                    "pr_auc_lo": lo,
                    "pr_auc_hi": hi,
                    "roc_auc": seed_mean(roc_auc, y, preds[k]),
                    "lift_vs_tabular": lift["delta"],
                    "lift_lo": lift["lo"],
                    "lift_hi": lift["hi"],
                    "chart": arm.chart,
                }
            )

    n_head = n_select
    jev, nli = preds[(9, n_head)], preds[(8, n_head)]
    head = _delta(y, jev, nli, boots[(9, n_head)], boots[(8, n_head)])
    headline = {"n_labels": n_head, **head, "verdict": verdict(head["lo"], head["hi"]), "margin": EQUIVALENCE_MARGIN}

    vs_bge = {n: _delta(y, preds[(9, n)], preds[(4, n)], boots[(9, n)], boots[(4, n)]) for n in label_counts}
    dates = [r["date_received"] for r in built.rows["test"]]
    halves = sorted(set(map(half_year, dates)))
    jev_direct = preds[(7, n_head)]

    tokens = built.jev_input_tokens
    compute = {
        "nli_compute_min_per_10k": built.nli_seconds * 10_000 / 60,
        "bge_compute_min_per_10k": built.bge_seconds * 10_000 / 60,
        "mean_jev_input_tokens": tokens,
    }
    # TypeSafe MCA 14.1 treats pricing as confidential, so dollar figures go to a gitignored file.
    usd = {
        "jev_usd_per_10k_1x": cost_per_10k(tokens, JEV_PRICE_PER_M_INPUT),
        "jev_usd_per_10k_10x": cost_per_10k(tokens, 10 * JEV_PRICE_PER_M_INPUT),
        "frontier_usd_per_10k_arithmetic": cost_per_10k(tokens, FRONTIER_PRICE_PER_M_INPUT),
    }
    legend = {
        "tabular": "$0",
        "bge_tab": f"{compute['bge_compute_min_per_10k']:.0f} compute-min/10k",
        "nli20_tab": f"{compute['nli_compute_min_per_10k']:.0f} compute-min/10k",
    }

    summary = {
        "jev_model": JEV_MODEL,
        "headline": headline,
        "single_window": subset_delta(y, jev, nli, built.single_window, n_boot, rng),
        "stops_beating_bge_tab": {
            "first_n_not_beating": stops_beating({n: d["lo"] for n, d in vs_bge.items()}),
            "by_n": {str(n): d for n, d in vs_bge.items()},
        },
        "arm4_variant": {"chosen": arm4_variant, "dev_pr_auc": arm4_dev, "at_n": n_select},
        "contamination_descriptive": {
            "jev_direct_pr_auc_by_half_year": {
                h: subset_pr_auc(y, jev_direct, np.array([half_year(d) == h for d in dates]), n_boot, rng) for h in halves
            },
            "headline_delta_2026_only": subset_delta(y, jev, nli, np.array([d.year == 2026 for d in dates]), n_boot, rng),
        },
        "compute": compute,
        "failures": built.failures,
        "rows_used": {s: len(built.rows[s]) for s in SPLITS},
        "answer_quality": answer_quality(frozen, built),
        "scoring_log": log,
    }

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)
    (out / "headline.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "costs_private.json").write_text(json.dumps(usd, indent=2) + "\n", encoding="utf-8")
    render_chart(results, headline, out / "chart.png", costs=legend)
    return {"results": results, **summary}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asktwice.report")
    p.add_argument("--frozen", default=str(FROZEN_DIR))
    p.add_argument("--out", default=str(RESULTS_DIR))
    args = p.parse_args(argv)
    summary = run_report(args.frozen, args.out)
    h = summary["headline"]
    print(f"Jev - NLI at {h['n_labels']}: {h['delta']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}] -> {h['verdict']}")  # ASCII: Windows consoles are cp1252
    print(f"wrote {args.out}/results.csv, headline.json, chart.png")


if __name__ == "__main__":
    main()
