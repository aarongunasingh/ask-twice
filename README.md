# Ask Twice

[![tests](https://github.com/aarongunasingh/ask-twice/actions/workflows/ci.yml/badge.svg)](https://github.com/aarongunasingh/ask-twice/actions/workflows/ci.yml)

Independent comparison: when a fixed set of human-written questions about a consumer-complaint narrative becomes features for a tabular model, does [TypeSafe](https://typesafe.ai) Jev's answering beat a free local NLI model on the same questions? And at how many training labels does any of it beat sentence embeddings?

This project is **not affiliated with TypeSafe AI**.

The experiment is specified in [`PLAN.md`](PLAN.md). Frozen choices are tagged `prereg-v1` before any test-set scoring.

## Status

Pipeline built end to end and tested on a synthetic fixture. Not yet tagged. `questions.yaml` is a throwaway Day 0 set.

## Setup

```bash
uv sync --extra jev --extra nli   # dev tools + Jev SDK + NLI/bge (torch, transformers)
uv run pytest
```

`uv sync` replaces the environment each time, so pass every extra you need in one call. Plain `uv sync` is enough for the tests: CI runs it on every push, with no API key or model download.

## Runbook

`data/` (raw CSV, candidates, SQLite cache) is never committed. `frozen/` (splits, hashes, exported answers, hand labels) and `results/` are.

**Day 0: go / no-go**

Download the exports from the [CFPB narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) (21 zips, ~1.35 GB) and unzip the CSVs into `data/ccdb/`.

```bash
uv run python -m asktwice.data filter --csv "data/ccdb/*.csv"       # -> data/candidates.parquet
uv run python -m asktwice.data rates                                 # train-pool relief rate by product
uv run python -m asktwice.answer probe                               # 20 longest narratives vs Jev; no answer values
```

Set `SELECTED_PRODUCTS` (2–3 products at 5–30%) and, if any probe hits 422, `JEV_MAX_CHARS` in `asktwice/config.py`, then re-run `filter`.

**Day 1: freeze**

```bash
uv run python -m asktwice.data split           # -> frozen/{train,dev,test}.parquet, test_labels.parquet, splits.json
uv run python -m asktwice.answer run nli --limit 500   # NLI throughput on train-pool rows (Colab GPU); needs the split
uv run python -m asktwice.stats calibrate      # equivalence-margin CI half-width on a 2022 pseudo-test
```

Write the 15 Noul + 5 Score questions and their NLI hypotheses into `questions.yaml` from ~150 dev narratives, blind, with `status: frozen`. Commit and push the `prereg-v1` tag. Then:

```bash
uv run python -m asktwice.data handlabels      # -> frozen/handlabels.csv; fill the label column (0/1) by hand
```

**Day 2: answer**

```bash
uv run python -m asktwice.answer run jev
uv run python -m asktwice.answer run nli --backup /content/drive/MyDrive/asktwice/answers.sqlite   # Colab
uv run python -m asktwice.answer export        # -> frozen/jev_answers.parquet, frozen/nli_answers.parquet
uv run python -m asktwice.answer check-dev     # flag questions with near-constant dev answers (prereg-v2 candidates)
```

Both runs resume from `data/answers.sqlite`. On Colab the cache stays on local disk; `--backup` copies it to Drive every 5 minutes and restores it on a fresh runtime.

**Day 3: score**

```bash
uv run python -m asktwice.report
```

This regenerates every number from `frozen/` with no API key. It refuses to run before the tag, appends to `results/scoring_log.csv`, and writes `results/results.csv`, `results/headline.json` and `results/chart.png`. The bge embeddings are recomputed on CPU and cached in `data/bge.npz`.

## Terms

TypeSafe's [Master Customer Agreement](https://typesafe.ai/legal/mca) (19 Sep 2026) assigns Output to the customer and forbids using Output to distill or imitate Jev. This experiment trains LightGBM on **monetary relief**, using Jev answers as features, the same pattern as TypeSafe's own feature-discovery cookbook. `frozen/jev_answers.parquet` holds raw JSON; it is only committed if that still looks allowed at tag time.

## Layout

```
asktwice/data.py      DuckDB filter, MinHashLSH dedup, time split, hashes, hand-label sample
asktwice/answer.py    Jev / NLI / bge, SQLite cache + Drive backup, run / export / probe
asktwice/features.py  encodings, shared row mask, 1% failure stop
asktwice/arms.py      ARMS table, fit_eval, arm-4 variant choice
asktwice/stats.py     bootstrap, verdicts, scoring log, prereg guard, margin calibration
asktwice/report.py    one command: results.csv, headline.json, chart.png
```
