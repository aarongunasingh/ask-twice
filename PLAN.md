# Ask Twice: Plan

**Status:** draft, written 2026-09-21. Everything marked *frozen* is committed under the git tag `prereg-v1` before any test-set scoring (see [Pre-registration](#pre-registration)).

Ask Twice asks the same 20 plain-English questions about every consumer complaint twice: once to TypeSafe AI's Jev, a typed-decision model, and once to a free open NLI model. Both sets of answers feed the same LightGBM model, which predicts whether the complaint ended with the consumer getting money back.

**The question:** when a fixed set of human-written questions about a text becomes features for a tabular model, does Jev's answering beat a free local model answering the same questions? And at how many training labels does any of it beat plain sentence embeddings?

This project is independent and not affiliated with TypeSafe AI.

---

## Why this comparison

- **Jev vs classic text classifiers is already covered.** Within a week of Jev's early-access launch (2026-09-15), several independent evaluations compared it with TF-IDF, BERT-family and zero-shot baselines on standard datasets (see References).
- **"Jev answers as features for gradient boosting" already exists.** TypeSafe's own feature-discovery cookbook feeds Jev's answers into CatBoost (2,000 wine reviews). The method itself is QA-Emb (Benara et al., 2024). The cookbook compares against a mean-score baseline and a word-count model only. It has no embeddings, TF-IDF, or other-answerer baselines, and lists adding them as a next step.
- **What's missing is attribution.** If Jev-answered features beat embeddings, is that Jev, or the analyst who wrote good questions? Holding the questions fixed and swapping the answerer separates the two.

---

## Data

| | |
|---|---|
| Source | CFPB Consumer Complaint Database with consumer narratives. Narrative publication ended 2026-08-14, so the archive is frozen. |
| Loading | The [narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) ships as 21 zipped CCDB exports by date range (~1.35 GB zipped; exports 2–19 cover 2019 to 2026-06-15). One DuckDB query with explicit column types reads them all as a glob, after checking that every file has the same header, and writes `candidates.parquet`; it never goes through pandas. |
| Target | `Company response to consumer == "Closed with monetary relief"` → 1, else 0 |
| Keep | Rows with a narrative and a final response in {Closed with monetary relief, Closed with non-monetary relief, Closed with explanation} |
| Drop | "In progress" and "Untimely response"; complaints received after 2026-06-15 (60 days before the freeze, responses may not be final); credit reporting (template-driven, relief near 0%) |
| Products | 2–3 products with a relief base rate of 5–30%, chosen on Day 0 (candidates: checking/savings, credit card, money transfer) |
| Excluded fields | Fields set at or after the company response: Timely response?, Consumer disputed?, Company public response, Date sent to company |
| Dedup | `datasketch` MinHashLSH, Jaccard ≥ 0.8 on 5-word shingles. Any test row that is a near-duplicate of a train or dev row is removed. |
| Sample | ~27k rows |

**Time split** *(frozen)*:

| Split | Received | Size | Use |
|---|---|---|---|
| Train pool | 2019–2022 | ≥ 20k after filtering | Label-count subsets are drawn from here |
| Dev | 2023 | ~2k | Writing questions; choosing between pre-registered model options |
| Test | 2024-01 to 2026-06-15 | ~5k, or more (see margin calibration) | Scored once, after the tag |

- **Why the pool must be ≥ 20k:** five 10k-label draws then overlap about 50%. If the pool comes in under 20k, the 10k point becomes "full pool, single fit", and its CI bootstraps test rows only.
- **Test labels** are written to a separate `test_labels.parquet`. Only `stats.load_test_labels()` reads it.
- **Split hashes** are committed.

---

## Questions *(frozen)*

- **Mix:** 15 yes/no questions (Jev "Noul") and 5 ordered-scale questions (Jev "Score", 2–10 levels), written from ~150 dev narratives.
- **Written blind:** no answerer output of any kind is looked at before the tag, so neither answerer gets questions tuned to it.
- **Rules:**
  - No counting, arithmetic, date or number-magnitude questions, and nothing a regex could find (see the [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) page).
  - No negated or multi-hop wording. Phrase each question so that a high value means yes.
  - No question may restate Product / Sub-product / Issue / Sub-issue; those are tabular features.
  - Each question is a concrete proposition, and each Score level describes a concrete situation, not an adjective.
  - A Noul may carry `criteria` (what yes and no mean) where the boundary is subtle. Jev's docs recommend this for literal-reading failures. The NLI hypothesis states the same condition.
- **NLI wording is frozen too:** each entry in `questions.yaml` carries the question text sent to Jev, the NLI hypothesis (Noul), or one hypothesis per level (Score).

---

## Answerers and features

### Jev
- **Client:** official `typesafe` Python SDK, `POST https://api.typesafe.ai/v1/systemone`.
- **Version:** `model: "jev-1.13.0"` pinned. Every response's `model` field is checked, and a mismatch aborts the run.
- **Why not Pydantic AI:** it maps a Noul question to `bool`, which would discard the probabilities.
- **One call per row:** all 20 questions, plus the direct question used by arm 7.
- **Errors:** 429 (rate limit) and 529 (overloaded) are retried inside the SDK with exponential backoff, honouring `Retry-After`. The retry policy is widened for a bulk run: 8 retries, capped at 60 s. A 422 means the request failed validation, not that the narrative was bad. It is recorded and not retried, and more than 10 of them at over 1% of rows stops the run.
- **Limits** ([models page](https://docs.typesafe.ai/models.md), checked 2026-09-21): 1,200 requests/min, 250k tokens/s, $0.042 per M input tokens, output tokens free.
- **Stored:** the full JSON response; `usage.input_tokens` is summed for the cost figures.
- **Long narratives:** the published context limit is 32k tokens for `state` plus the longest question (64k for the whole request). CFPB narratives are far shorter. Day 0 still probes the 20 longest narratives. If any fail, a head-truncation rule is frozen at the tag, and the share of truncated rows is reported.

### NLI (the free control)
- **Model:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, called directly through `AutoModelForSequenceClassification`, with length-sorted batches and fp16 on GPU.
- **Why not the HuggingFace pipeline:** the zero-shot pipeline's `multi_label` mode softmaxes entailment against contradiction only and drops neutral. A narrative that never mentions a topic can then score around 0.5 instead of near 0. This project uses the full 3-way softmax, reading the entailment index from `model.config.label2id`.
- **Chunking:** the 512-token limit covers premise *and* hypothesis. Chunk length = 512 − hypothesis tokens − 3 special tokens, with 64-token overlap.

### Embeddings
- **Model:** `BAAI/bge-small-en-v1.5` (384 dims), 512-token chunks, mean-pooled.

### Feature encoding *(frozen, identical for both answerers)*

| Question type | Jev | NLI |
|---|---|---|
| Noul | the `noul` probability | P(entailment), 3-way softmax; max over chunks |
| Score | expected level, Σ level × P(level), from the `probabilities` map | per level: max entailment logit over chunks → softmax across levels → expected level |

- **Tabular fields:** product, sub-product, issue, sub-issue, company (top 50 by frequency in the train pool, else "other"), state, submitted_via, tags, received month-of-year.
  - Year-month is excluded: under a time split, every test value would be outside the training range.
- **Shared row mask:** a row is used by any arm only if every answerer (Jev, NLI, bge) produced a valid answer for it. The same mask applies to all arms.
  - Failure counts are reported per answerer.
  - More than 1% test-row failures for any answerer stops the run.

---

## Arms *(frozen)*

| # | Arm | Head | Purpose |
|---|---|---|---|
| 0 | Base rate | none | floor |
| 1 | **Tabular only** | LightGBM | the bar every text arm must clear |
| 2 | TF-IDF text only | LogReg | classic baseline |
| 3 | bge text only | LogReg | embeddings with the head they suit |
| 4 | **bge + tabular** | LightGBM, raw 384 dims or stacked (out-of-fold LogReg-on-bge score as one feature) | strongest non-question arm |
| 5 | NLI-20 text only | LogReg | answer quality without tabular help |
| 6 | Jev-20 text only | LogReg | answer quality without tabular help |
| 7 | Jev direct | none: one Jev Noul, "Will this complaint end with monetary relief?", used as the score | does splitting into questions matter? |
| 8 | **NLI-20 + tabular** | LightGBM | the control that makes attribution possible |
| 9 | **Jev-20 + tabular** | LightGBM | the arm under test |

**Bold** arms appear on the chart. Arms are defined as one `ARMS` table run by a single `fit_eval(arm, n_labels, seed)`, so every arm shares the same sampling and fitting code.

**Training schedule:**

| | Setting |
|---|---|
| Label counts | {300, 1k, 3k, 10k}, stratified draws from the train pool |
| Seeds | 10 for N ≤ 1k; 5 above |
| LightGBM, N < 1k | fixed: num_leaves=7, min_child_samples=5, 200 trees, learning_rate=0.05 |
| LightGBM, N ≥ 1k | tuned on dev from a grid committed at the tag |
| LogReg (arms 2, 3, 5, 6) | C ∈ {0.01, 0.1, 1, 10} tuned on dev for N ≥ 1k; C = 1 below |
| Arm 2 n-grams | {(1,1), (1,2)} |
| Arm 4 variant | raw vs stacked, chosen once on dev at 1k labels and used at every label count |

---

## Metrics and statistical tests *(frozen)*

### Primary metric
- Test **PR-AUC** per arm per label count.
- 95% CI from 1,000 bootstrap resamples that draw test rows *and* seeds together.
- One resample-index matrix is generated once and shared by every arm, which makes all comparisons paired.

### Headline test
Δ = PR-AUC(Jev-20 + tab) − PR-AUC(NLI-20 + tab) at 1k labels, paired bootstrap:

| Verdict | Rule |
|---|---|
| Jev better | CI lower bound > 0 |
| NLI better | CI upper bound < 0 |
| Equivalent | whole CI inside the equivalence margin |
| Inconclusive | none of the above |

Equivalence is never claimed from a non-significant difference.

### Equivalence margin
- Starts at ±0.02 PR-AUC and is calibrated before the tag.
- Use a 2022 train-pool slice as a pseudo-test. Run the same paired bootstrap on arm 1 vs tabular + a TF-IDF score at 1k labels, and read off the CI half-width.
- If it is above ~0.015, grow the test set or widen the margin. The choice and the reason are recorded in the tag.

### Robustness and secondary measures
- **Single-window delta:** the headline test restricted to test rows whose narrative fits one NLI window, so both answerers see the whole text.
- **"Stops beating":** the smallest label count at which the paired-CI lower bound for Jev-20 + tab minus bge + tab is ≤ 0. If it is never above 0, report "never beats".
- **Secondary:**
  - ROC-AUC
  - lift over arm 1
  - $ per 10k rows: Jev at 1× and 10× list price ($0.042/M input tokens today); a frontier LLM answering the same questions costed arithmetically (tokens × list price, not run); NLI and bge as compute time.
- **Answer quality:**
  - After the tag and before any answerer runs, 100 test rows × 5 Noul questions (500 judgments, drawn with a seed recorded in the tag) are hand-labeled.
  - Report per-question accuracy for Jev and NLI, and Jev's expected calibration error on those 500.
- **Not measured:** latency. This is batch feature extraction, and early-access gateway latency is noise.

### Contamination
Jev (released 2026) may have seen CFPB rows, outcomes included, during pretraining. DeBERTa-v3 (2021) and bge-small (2023) cannot have seen 2024–26 rows. Any statistical test of memorization here would have almost no power, so the plan reports descriptively and states the risk in the write-up:
- Arm 7 PR-AUC for each test half-year.
- The headline delta on 2026 rows only, labeled descriptive.
- Jev's training cutoff, if TypeSafe publishes it.

---

## The chart

- **x:** training labels, log scale (300 · 1k · 3k · 10k).
- **y:** test PR-AUC with 95% CI bands.
- **Lines:**
  - tabular only (dashed)
  - bge + tabular
  - NLI-20 + tabular
  - Jev-20 + tabular
  - base rate (dotted)
- **Legend:** $ per 10k rows for each arm.
- **Headline number:** the Jev − NLI delta at 1k labels, with its CI and verdict.

Every verdict is a reportable result, including "inconclusive", "Jev's edge disappears on single-window rows", and "no question arm beats tabular only".

---

## Pre-registration

1. **Day 0 checks use throwaway questions** on train-pool rows. Only schema, token counts, latency and throughput are recorded, never answer values.
2. **Freeze and tag.** Questions, NLI hypotheses, splits and hashes, encodings, truncation rule, grids and fixed parameters, statistical tests, equivalence margin and primary metric are committed and tagged `prereg-v1`.
3. **Amendments.** A question that is structurally broken for either answerer (e.g. constant output on dev) may be fixed only before the first test prediction. The same change applies to both answerers, and it is tagged `prereg-v2` with a `CHANGELOG` entry.
4. **Guard.** `load_test_labels()` refuses unless HEAD comes after the prereg tag.
5. **Scoring log.** Every test scoring run appends commit, dirty-tree flag and diff hash to `scoring_log.csv`. The published run must show `dirty=false`.

### Implementation choices *(frozen with the tag)*

Settled while building the pipeline. The plan left them open.

- **Verdict precedence:** the rows of the headline table are checked in order. A CI that excludes 0 is reported as "Jev better" / "NLI better" even if it also sits inside the margin.
- **Point estimates:** PR-AUC and ROC-AUC are computed per seed and then averaged. The CI comes from the bootstrap.
- **Sampling:** train, dev and test are plain random samples, so drawing the test sample never reads test labels. The train pool is sampled at 20.5k, which gives headroom for answer failures above the 20k floor.
- **Fallback when the pool is under 2n:** any label count whose pool is under 2n becomes "full pool, single fit".
- **Tabular encoding:** categoricals are integer codes (alphabetical vocabulary from the train pool; unseen values map to "other"). They are not LightGBM native categoricals, which need about 100 rows per group.
- **TF-IDF:** `max_features=20000`, `min_df=2`.
- **Stacked arm 4:** 5-fold out-of-fold LogReg score on the training draw. The variant is picked by mean dev PR-AUC over the 1k-label seeds.
- **Product base rates:** computed on the train pool only.
- **NLI input:** `[CLS] premise-chunk [SEP] hypothesis [SEP]`. Single-window rows are those where every hypothesis needs exactly one chunk.
- **Hand-label accuracy:** answers are thresholded at 0.5. ECE uses 10 equal-width bins.
- **Structurally broken question:** after the tag, `answer check-dev` flags any question whose encoded dev answers have a standard deviation below 0.01 for either answerer. It shows no label information, so an amendment cannot be tuned to the outcome.

---

## Architecture

```
 CFPB complaints.csv (8GB+)
        |  DuckDB filter, typed columns
        v
 candidates.parquet --MinHashLSH dedup--> time split + hashes --> test_labels.parquet (guarded)
        |
        v
 asktwice/answer.py   ask_jev() | nli_scores() | embed_bge()      <-- questions.yaml (frozen)
        |             all write to ONE SQLite cache
        |             PK (answerer, model_version, question_set_hash, text_sha256)
        v
 asktwice/features.py answers -> matrices; shared row mask; >1% failures = stop
        v
 asktwice/arms.py     ARMS x {300, 1k, 3k, 10k} x seeds -> fit_eval() -> test predictions
        v
 asktwice/stats.py    shared resample matrix -> CIs, paired deltas, verdicts, scoring log
        v
 asktwice/report.py   chart.png, results.csv, headline; exports cache to Parquet
```

**Answer cache:**
- stdlib `sqlite3` on local disk, committed after each answer, so an interrupted run resumes where it stopped.
- The full JSON response is stored.
- On Colab, the database stays on local disk and is copied to Google Drive with `Connection.backup()` every ~5 minutes and at the end of each batch, then restored on restart. It never lives on the Drive mount directly: SQLite on a network filesystem can corrupt on disconnect.

| Codepath | Realistic failure | Handling |
|---|---|---|
| `ask_jev` | `jev-latest` updated mid-run | version pinned and asserted, run aborts |
| `ask_jev` | 422 (request failed validation) | recorded; systematic 422s stop the run; shared mask |
| NLI on Colab | runtime disconnect | local SQLite + Drive backup; resume |
| `nli_scores` | label order differs by model | index read from `label2id` |
| `features` | an answerer fails on > 1% of test | run stops |
| `data` | CFPB column renamed | typed DuckDB read fails loudly |
| `stats` | test scored before the tag | guard refuses |

---

## Tests and CI

- **Runner:** pytest via `uv`.
- **CI:** GitHub Actions on every push (`astral-sh/setup-uv` → `uv run pytest`), with a README badge.
- **No secrets needed:** the end-to-end test uses a 200-row synthetic fixture with fake cached answers.

| File | Covers |
|---|---|
| `test_data.py` | filter rules; train < dev < test dates; no complaint ID in two splits; test near-duplicates of train removed; test labels in their own file; stable split hashes |
| `test_answer.py` | cache key deterministic and independent of question order; resume skips cached rows; 422 recorded, not retried; version mismatch aborts; backup/restore round trip; swapped `label2id` still yields P(entailment); chunk leaves room for the hypothesis; Score chunk rule |
| `test_features.py` | expected-level maths; a row missing in any answerer is dropped from all arms; > 1% failures raises; test-only company maps to "other"; no test feature value outside the training range |
| `test_arms.py` | n = 300 at a 5% base rate still draws positives; same seed gives the same draw; fixed parameters below 1k; arm-4 variant chosen once |
| `test_stats.py` | planted delta recovered; identical predictions give "equivalent"; small noisy delta gives "inconclusive"; "never beats"; resample matrix shared across arms; guard refuses before the tag; dirty tree logged |
| `test_e2e.py` | fixture → chart + `results.csv`, no network |

---

## Reproducibility

- **One command:** `uv run python -m asktwice.report` regenerates every number and the chart from committed answers, with no API key needed.
- **Committed:**
  - `questions.yaml` + `CHANGELOG`
  - `frozen/`: split hashes, the three splits (zstd Parquet; the narratives and tabular fields the report needs), `test_labels.parquet`, `handlabels.csv`
  - Jev answers (zstd Parquet)
  - NLI logits (float16 Parquet)
  - `results/`: `results.csv`, `headline.json`, `chart.png`, `scoring_log.csv`
- **Not committed:** bge embeddings, which are recomputed in a few CPU minutes.
- **GitHub Release:** the full SQLite cache, with its sha256 in the README.
- **Reported:** the write-up names the Jev version that answered, run dates, prices at 1× and 10×, and failure counts per answerer.
- **Terms caveat:** redistributing raw Jev outputs depends on TypeSafe's early-access terms, which are checked on Day 0. If they don't allow it, the repo commits derived features only.

---

## Roadmap

Five work sessions. Each ends in a check that must pass before the next one starts.

| Session | Work | Gate |
|---|---|---|
| **Day 0** go / no-go | ✓ repo created · T1 DuckDB filter, confirm the narrative ↔ ID ↔ response join, base rate by product · T12 read TypeSafe terms; Jev key; throwaway smoke test; probe the 20 longest narratives · NLI throughput on Colab | Join works; 2–3 products at 5–30% relief; terms allow publishing; Jev and NLI both run |
| **Parallel** from Day 0 | T10 invariant test suite + fixture · T11 CI | `uv run pytest` green |
| **Day 1** freeze | T2 dedup, split, hashes, `test_labels.parquet` · write 20 questions + NLI hypotheses (blind) · T9 calibrate the margin · commit and tag **`prereg-v1`** · then hand-label 100 × 5 | Tag pushed before any answerer run |
| **Day 2** answer | T3 SQLite cache + Drive backup · T4 Jev answerer · T5 NLI answerer · run Jev, NLI, bge · check dev for broken questions (`prereg-v2` if needed) | < 1% failures per answerer |
| **Day 3** score | T6 features + shared mask · T7 ARMS + `fit_eval` · T8 bootstrap, verdicts, single-window delta, contamination slices, scoring log · render the chart | Published run logged `dirty=false`, after the tag |
| **Day 4** ship | README with the chart on top, Release asset, write-up | — |

---

## Open questions

1. **CFPB join.** Does the narratives archive still link narrative, Complaint ID and company response? If not: a dated pre-2026-08-14 mirror; if none exists, a different dataset.
2. **Jev access and terms.** Direct API key availability; whether publishing results and sharing raw outputs are allowed.
3. **NLI throughput.** ~1.1M premise–hypothesis pairs. Free Colab GPU by default; CPU overnight as the fallback.
4. **Jev training cutoff.** Not published; it would sharpen the descriptive contamination slice.
5. **CI width.** "Inconclusive" is a likely verdict; the margin calibration keeps it honest rather than guaranteed.

---

## Out of scope

- **A small instruct LLM as the control answerer** (same wording, full context, logprobs). Worth revisiting if the single-window delta shows Jev's edge is all context length.
- **Gateway adapters** (Vercel AI Gateway, OpenRouter): only if the direct API key doesn't arrive.
- **Latency benchmarks.**
- **Running a frontier LLM on the same questions:** costed arithmetically only.
- **A formal memorization test:** too little statistical power; contamination is reported descriptively instead.
- **Future work:** "frozen questions, moving world". Keep the 20 questions fixed and score each quarter through 2026 to show how each question's link to the outcome drifts over time.

---

## References

- TypeSafe AI, *Introducing System One Models & Jev* — https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe API reference — https://docs.typesafe.ai/api.md
- TypeSafe cookbook, *Autoresearch feature discovery* — https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery
- Benara et al., *Crafting Interpretable Embeddings by Asking LLMs Questions* (QA-Emb), 2024 — https://arxiv.org/abs/2405.16714
- Independent Jev evaluations: https://github.com/priorbench/jev · https://github.com/zhuyansen/jev-zeroshot-vs-bert · https://github.com/AbdelStark/jev-benchmarks · https://github.com/fstandhartinger/jevbench
- CFPB, *Cease discretionary publication of complaint narratives* (2026-08-14) — https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/
- CFPB narratives archive — https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
- HuggingFace zero-shot pipeline source (multi-label scoring) — https://huggingface.co/transformers/v4.11.3/_modules/transformers/pipelines/zero_shot_classification.html
