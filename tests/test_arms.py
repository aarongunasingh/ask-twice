from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from asktwice.arms import choose_arm4_variant, fit_logreg, lgbm_params, schedule, select_arm4_variant, stack_score, stratified_draw
from asktwice.config import BGE_DIMS, LGBM_SMALL


def test_n300_at_5_percent_still_draws_positives():
    y = np.zeros(2000, dtype=int)
    y[:100] = 1
    idx = stratified_draw(y, 300, seed=7)
    assert idx.shape == (300,)
    assert y[idx].sum() >= 1 and y[idx].min() == 0


def test_same_seed_same_draw():
    y = np.array([0] * 190 + [1] * 10)
    assert np.array_equal(stratified_draw(y, 50, seed=42), stratified_draw(y, 50, seed=42))
    assert not np.array_equal(stratified_draw(y, 50, seed=42), stratified_draw(y, 50, seed=43))


def test_fixed_params_below_1k():
    assert lgbm_params(300) == dict(LGBM_SMALL)
    assert lgbm_params(999) == dict(LGBM_SMALL)
    assert lgbm_params(1000) is None


def test_pool_under_2n_becomes_full_pool_single_fit():
    assert schedule(300, 20_000) == (300, 10)
    assert schedule(10_000, 20_100) == (10_000, 5)
    assert schedule(10_000, 19_000) == (19_000, 1)


def test_stacked_score_is_out_of_fold():
    # 384 noise dims, 300 rows: an in-sample LogReg score separates train almost perfectly.
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    x = rng.normal(size=(300, BGE_DIMS))
    in_sample = fit_logreg(x, y).predict_proba(x)[:, 1]
    oof, _ = stack_score(x, y, [x[:5]], C=1.0)
    assert roc_auc_score(y, in_sample) > 0.95
    assert roc_auc_score(y, oof[:, 0]) < 0.75


def test_arm4_variant_chosen_once_on_dev():
    rng = np.random.default_rng(1)
    n_tr, n_dev = 400, 200
    y_tr = (rng.random(n_tr) < 0.2).astype(int)
    y_dev = (rng.random(n_dev) < 0.2).astype(int)
    feats = {
        "tab": rng.normal(size=(n_tr, 9)),
        "tab_dev": rng.normal(size=(n_dev, 9)),
        "tab_test": rng.normal(size=(50, 9)),
        "bge": rng.normal(size=(n_tr, 8)) + y_tr[:, None],
        "bge_dev": rng.normal(size=(n_dev, 8)) + y_dev[:, None],
        "bge_test": rng.normal(size=(50, 8)),
    }
    variant, dev = select_arm4_variant(100, y_train=y_tr, y_dev=y_dev, features=feats)
    assert variant == choose_arm4_variant(dev["raw"], dev["stacked"])
    assert choose_arm4_variant(0.40, 0.40) == "stacked"
