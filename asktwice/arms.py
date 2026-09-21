"""ARMS table and fit_eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from asktwice.config import (
    LGBM_GRID,
    LGBM_SMALL,
    LOGREG_C_GRID,
    N_SEEDS_LARGE,
    N_SEEDS_SMALL,
    STACK_FOLDS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAMS,
)

ArmHead = Literal["none", "lgbm", "logreg"]


@dataclass(frozen=True)
class Arm:
    id: int
    name: str
    head: ArmHead
    features: str
    chart: bool = False


ARMS = (
    Arm(0, "base_rate", "none", "none", chart=True),
    Arm(1, "tabular", "lgbm", "tab", chart=True),
    Arm(2, "tfidf", "logreg", "tfidf"),
    Arm(3, "bge", "logreg", "bge"),
    Arm(4, "bge_tab", "lgbm", "bge_tab", chart=True),
    Arm(5, "nli20", "logreg", "nli"),
    Arm(6, "jev20", "logreg", "jev"),
    Arm(7, "jev_direct", "none", "jev_direct"),
    Arm(8, "nli20_tab", "lgbm", "nli_tab", chart=True),
    Arm(9, "jev20_tab", "lgbm", "jev_tab", chart=True),
)
ARMS_BY_ID = {a.id: a for a in ARMS}
# Margin calibration only (plan: arm 1 vs tabular + a TF-IDF score). Not a reported arm.
TFIDF_TAB = Arm(-1, "tfidf_score_tab", "lgbm", "tfidf_tab")

_PARTS = {
    "tab": ("tab",),
    "bge": ("bge",),
    "nli": ("nli",),
    "jev": ("jev",),
    "bge_tab": ("tab", "bge"),
    "nli_tab": ("tab", "nli"),
    "jev_tab": ("tab", "jev"),
}


def n_seeds_for(n_labels: int) -> int:
    return N_SEEDS_SMALL if n_labels <= 1_000 else N_SEEDS_LARGE


def schedule(n_labels: int, pool: int) -> tuple[int, int]:
    """(labels drawn, seeds). Pool under 2n: the point becomes 'full pool, single fit'."""
    if 2 * n_labels > pool:
        return pool, 1
    return n_labels, n_seeds_for(n_labels)


def lgbm_params(n_labels: int) -> dict[str, Any] | None:
    return dict(LGBM_SMALL) if n_labels < 1_000 else None


def stratified_draw(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if n > len(y):
        raise ValueError(f"cannot draw {n} from {len(y)}")
    n_pos = int(round(n * float(y.mean()))) if len(y) else 0
    n_pos = max(1 if len(pos) else 0, n_pos)
    n_pos = min(n_pos, len(pos), n - (1 if len(neg) else 0))
    n_neg = min(n - n_pos, len(neg))
    n_pos = n - n_neg
    idx = np.concatenate([rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(idx)
    return idx


def choose_arm4_variant(dev_prauc_raw: float, dev_prauc_stacked: float) -> str:
    return "stacked" if dev_prauc_stacked >= dev_prauc_raw else "raw"


def fit_lgbm(x: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **params)
    model.fit(x, y)
    return model


def _dev_score(model: Any, x_dev: Any, y_dev: np.ndarray) -> float:
    return float(average_precision_score(y_dev, model.predict_proba(x_dev)[:, 1]))


def tune_lgbm(x_tr: np.ndarray, y_tr: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray) -> tuple[dict[str, Any], Any]:
    fits = [(dict(p), fit_lgbm(x_tr, y_tr, p)) for p in LGBM_GRID]
    return max(fits, key=lambda f: _dev_score(f[1], x_dev, y_dev))


def fit_logreg(x: Any, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000).fit(x, y)


def tune_logreg(x_tr: Any, y_tr: np.ndarray, x_dev: Any, y_dev: np.ndarray) -> tuple[float, LogisticRegression]:
    fits = [(C, fit_logreg(x_tr, y_tr, C)) for C in LOGREG_C_GRID]
    return max(fits, key=lambda f: _dev_score(f[1], x_dev, y_dev))


def stack_score(x_tr: Any, y: np.ndarray, others: list[Any], C: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Out-of-fold LogReg score for the training rows; full-fit score for dev/test."""
    folds = StratifiedKFold(min(STACK_FOLDS, int(np.bincount(y).min())), shuffle=True, random_state=0)
    oof = cross_val_predict(LogisticRegression(C=C, max_iter=2000), x_tr, y, cv=folds, method="predict_proba")
    clf = fit_logreg(x_tr, y, C)
    return oof[:, 1:2], [clf.predict_proba(x)[:, 1:2] for x in others]


def _tfidf(texts: dict[str, list[str]], idx: np.ndarray, ngram: tuple[int, int]) -> tuple[Any, Any, Any]:
    vec = TfidfVectorizer(ngram_range=ngram, min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES)
    x_tr = vec.fit_transform([texts["train"][i] for i in idx])
    return x_tr, vec.transform(texts["dev"]), vec.transform(texts["test"])


def fit_eval(
    arm: Arm,
    n_labels: int,
    seed: int,
    *,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    features: dict[str, np.ndarray],
    texts: dict[str, list[str]] | None = None,
    arm4_variant: str = "raw",
) -> dict[str, Any]:
    idx = stratified_draw(y_train, n_labels, seed)
    y = y_train[idx]
    tuned = n_labels >= 1_000
    params: dict[str, Any] = {}

    def result(model: Any, x_dev: Any, x_te: Any) -> dict[str, Any]:
        return {
            "arm": arm.id,
            "n_labels": n_labels,
            "seed": seed,
            "pred": model.predict_proba(x_te)[:, 1],
            "dev_pred": model.predict_proba(x_dev)[:, 1],
            "params": params,
        }

    if arm.features in ("none", "jev_direct"):
        n_test = len(features["tab_test"])
        pred = np.full(n_test, float(y.mean())) if arm.features == "none" else features["jev_direct_test"]
        return {"arm": arm.id, "n_labels": n_labels, "seed": seed, "pred": pred, "dev_pred": None, "params": params}

    if arm.features == "tfidf":
        fits = []
        for ng in TFIDF_NGRAMS if tuned else TFIDF_NGRAMS[:1]:
            x_tr, x_dev, x_te = _tfidf(texts, idx, ng)
            C, clf = tune_logreg(x_tr, y, x_dev, y_dev) if tuned else (1.0, fit_logreg(x_tr, y))
            fits.append((_dev_score(clf, x_dev, y_dev), ng, C, clf, x_dev, x_te))
        _, ng, C, clf, x_dev, x_te = max(fits, key=lambda f: f[0])
        params.update(ngram=ng, C=C)
        return result(clf, x_dev, x_te)

    if arm.features == "tfidf_tab" or (arm.features == "bge_tab" and arm4_variant == "stacked"):
        if arm.features == "tfidf_tab":
            s_tr, s_dev, s_te = _tfidf(texts, idx, TFIDF_NGRAMS[0])
        else:
            s_tr, s_dev, s_te = features["bge"][idx], features["bge_dev"], features["bge_test"]
        C = tune_logreg(s_tr, y, s_dev, y_dev)[0] if tuned else 1.0
        oof, (score_dev, score_te) = stack_score(s_tr, y, [s_dev, s_te], C)
        params["stack_C"] = C
        x_tr = np.hstack([features["tab"][idx], oof])
        x_dev = np.hstack([features["tab_dev"], score_dev])
        x_te = np.hstack([features["tab_test"], score_te])
    else:
        parts = _PARTS[arm.features]
        x_tr = np.hstack([features[p][idx] for p in parts])
        x_dev = np.hstack([features[p + "_dev"] for p in parts])
        x_te = np.hstack([features[p + "_test"] for p in parts])

    if arm.head == "lgbm":
        fixed = lgbm_params(n_labels)
        p, model = (fixed, fit_lgbm(x_tr, y, fixed)) if fixed else tune_lgbm(x_tr, y, x_dev, y_dev)
        params.update(p)
    else:
        C, model = tune_logreg(x_tr, y, x_dev, y_dev) if tuned else (1.0, fit_logreg(x_tr, y))
        params["C"] = C
    return result(model, x_dev, x_te)


def select_arm4_variant(
    n_labels: int,
    *,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    features: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    """Plan: raw vs stacked, chosen once on dev at 1k labels (mean dev PR-AUC over that point's seeds)."""
    n, seeds = schedule(n_labels, len(y_train))
    dev = {}
    for v in ("raw", "stacked"):
        fits = [
            fit_eval(ARMS_BY_ID[4], n, s, y_train=y_train, y_dev=y_dev, features=features, arm4_variant=v)
            for s in range(seeds)
        ]
        dev[v] = float(np.mean([average_precision_score(y_dev, f["dev_pred"]) for f in fits]))
    return choose_arm4_variant(dev["raw"], dev["stacked"]), dev
