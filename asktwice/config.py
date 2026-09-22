"""Frozen experiment constants. Values here are tagged at prereg-v1."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"  # raw CSV, candidates, SQLite cache, bge cache (not committed)
FROZEN_DIR = ROOT / "frozen"  # splits, split hashes, exported answers, hand labels (committed)
RESULTS_DIR = ROOT / "results"  # results.csv, headline.json, chart.png, scoring_log.csv (committed)
SPLITS = ("train", "dev", "test")

PREREG_TAG = "prereg-v1"

JEV_MODEL = "jev-1.13.0"
JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_PRICE_PER_M_INPUT = 0.042
FRONTIER_PRICE_PER_M_INPUT = 2.50

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_DIMS = 384

MAX_SEQ = 512
SPECIAL_TOKENS = 3
CHUNK_OVERLAP = 64

RESPONSE_KEEP = (
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with explanation",
)
RELIEF_RESPONSE = "Closed with monetary relief"
CREDIT_REPORTING_RE = r"credit report"
RECEIVED_CUTOFF = "2026-06-15"
TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
DEV_START = "2023-01-01"
DEV_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = RECEIVED_CUTOFF

PRODUCT_CANDIDATES = (
    "Checking or savings account",
    "Bank account or service",
    "Credit card",
    "Credit card or prepaid card",
    "Money transfer, virtual currency, or money service",
    "Money transfer",
    "Money transfers",
    "Virtual currency",
)
# Frozen on Day 0 after base-rate check. None = keep PRODUCT_CANDIDATES.
SELECTED_PRODUCTS: tuple[str, ...] | None = (
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
)

RELIEF_RATE_LO = 0.05
RELIEF_RATE_HI = 0.30

# 20k + headroom: up to 2.5% answer failures still leave >= 20k, so 10k keeps 5 draws.
SAMPLE_TRAIN = 20_500
SAMPLE_DEV = 2_000
# Grown from 5k on Day 1: calibration half-width 0.0228 on 7,404 pseudo-test rows; fit h^2 = 3.564/n + 9e-6
# puts h <= 0.015 at ~16.5k rows.
SAMPLE_TEST = 16_500
SPLIT_SEED = 0

DEDUP_JACCARD = 0.8
DEDUP_SHINGLE_N = 5
MINHASH_PERM = 128
COMPANY_TOP_K = 50

N_NOUL = 15
N_SCORE = 5

LABEL_COUNTS = (300, 1_000, 3_000, 10_000)
N_SEEDS_SMALL = 10
N_SEEDS_LARGE = 5
N_BOOT = 1_000
EQUIVALENCE_MARGIN = 0.02
MARGIN_HALF_WIDTH_MAX = 0.015
PSEUDO_TEST_YEAR = 2022
FAILURE_RATE_STOP = 0.01
BROKEN_STD = 0.01  # dev answers this flat mark a question structurally broken (prereg-v2 candidate)

HANDLABEL_ROWS = 100
HANDLABEL_QUESTIONS = 5
HANDLABEL_SEED = 20260921

JEV_WORKERS = 8
NLI_BATCH = 64
NLI_GROUP = 256  # narratives per cache write
BGE_BATCH = 64
BACKUP_SECONDS = 300

LGBM_SMALL = {
    "num_leaves": 7,
    "min_child_samples": 5,
    "n_estimators": 200,
    "learning_rate": 0.05,
}
LGBM_GRID = (
    {"num_leaves": 15, "min_child_samples": 20, "n_estimators": 200, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 20, "n_estimators": 200, "learning_rate": 0.05},
    {"num_leaves": 15, "min_child_samples": 50, "n_estimators": 400, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 50, "n_estimators": 400, "learning_rate": 0.1},
)
LOGREG_C_GRID = (0.01, 0.1, 1.0, 10.0)
TFIDF_NGRAMS = ((1, 1), (1, 2))
TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF = 2
STACK_FOLDS = 5
# Tabular categoricals enter LightGBM as integer codes (alphabetical vocab from the
# train pool, unseen -> "other"). Native categorical splits need ~100 rows per
# group, which the 300-label point does not have.

# Head-truncation for Jev, frozen after the Day 0 long-narrative probe. The published limit is
# 32k tokens for state + longest question (docs.typesafe.ai/models), so None is the expected value.
JEV_MAX_CHARS: int | None = None

CSV_COLUMNS = {
    "Date received": "VARCHAR",
    "Product": "VARCHAR",
    "Sub-product": "VARCHAR",
    "Issue": "VARCHAR",
    "Sub-issue": "VARCHAR",
    "Consumer complaint narrative": "VARCHAR",
    "Company public response": "VARCHAR",
    "Company": "VARCHAR",
    "State": "VARCHAR",
    "ZIP code": "VARCHAR",
    "Tags": "VARCHAR",
    "Consumer consent provided?": "VARCHAR",
    "Submitted via": "VARCHAR",
    "Date sent to company": "VARCHAR",
    "Company response to consumer": "VARCHAR",
    "Timely response?": "VARCHAR",
    "Consumer disputed?": "VARCHAR",
    "Complaint ID": "VARCHAR",
}

REQUIRED_CSV_COLUMNS = (
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Tags",
    "Submitted via",
    "Company response to consumer",
    "Complaint ID",
)

EXCLUDED_FEATURE_FIELDS = (
    "Timely response?",
    "Consumer disputed?",
    "Company public response",
    "Date sent to company",
)
