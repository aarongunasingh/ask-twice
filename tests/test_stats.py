from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asktwice.stats import (
    GuardError,
    append_scoring_log,
    ci95,
    expected_calibration_error,
    load_test_labels,
    make_resamples,
    paired_delta,
    stops_beating,
    verdict,
)
from tests.helpers import fake_git


def test_planted_delta_recovered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    good = y.astype(float) + rng.normal(0, 0.05, size=400)
    bad = rng.random(400)
    rows, seeds = make_resamples(400, 1, n_boot=200, rng=np.random.default_rng(1))
    mu, lo, _ = ci95(paired_delta(y, good[None, :], bad[None, :], rows, seeds))
    assert lo > 0 and mu > 0


def test_identical_predictions_equivalent():
    y = np.array([0, 1, 0, 1] * 50)
    s = np.linspace(0, 1, len(y))
    rows, seeds = make_resamples(len(y), 1, n_boot=100, rng=np.random.default_rng(0))
    _, lo, hi = ci95(paired_delta(y, s[None, :], s[None, :], rows, seeds))
    assert verdict(lo, hi, margin=0.02) == "equivalent"


def test_significant_difference_reported_before_equivalence():
    assert verdict(0.005, 0.015, margin=0.02) == "Jev better"
    assert verdict(-0.015, -0.005, margin=0.02) == "NLI better"


def test_small_noisy_delta_inconclusive():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=80)
    a = rng.random(80)
    b = a + rng.normal(0, 0.5, size=80)
    rows, seeds = make_resamples(80, 1, n_boot=80, rng=np.random.default_rng(3))
    _, lo, hi = ci95(paired_delta(y, a[None, :], b[None, :], rows, seeds))
    assert verdict(lo, hi, margin=0.02) == "inconclusive"


def test_never_beats():
    assert stops_beating({300: -0.01, 1000: -0.02, 3000: -0.01, 10000: 0.0}) == "never beats"
    assert stops_beating({300: 0.04, 1000: 0.02, 3000: -0.01, 10000: -0.02}) == 3000
    assert stops_beating({300: 0.04, 1000: 0.03, 3000: 0.02, 10000: 0.01}) is None


def test_bootstrap_draws_seeds_with_rows():
    rows, seeds = make_resamples(20, 3, n_boot=500, rng=np.random.default_rng(0))
    assert set(seeds) == {0, 1, 2}
    y = np.array([0, 1] * 10)
    good = np.tile(y.astype(float), (3, 1))
    good[1] = 1 - good[1]  # seed 1 is perfectly wrong
    d = paired_delta(y, good, good, rows, seeds)
    assert np.allclose(d[np.isfinite(d)], 0)  # same rows and same seed on both sides


def test_ece_zero_when_calibrated():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0
    assert expected_calibration_error(y, np.array([0.9, 0.9, 0.1, 0.1])) > 0.5


def test_guard_refuses_before_tag_and_returns_ids_after(tmp_path):
    labels = tmp_path / "test_labels.parquet"
    pq.write_table(pa.table({"complaint_id": ["7", "3"], "y": [1, 0]}), labels)
    with pytest.raises(GuardError):
        load_test_labels(labels, run=fake_git(after_tag=False))
    assert load_test_labels(labels, run=fake_git()) == {"7": 1, "3": 0}


def test_dirty_tree_logged_with_stable_header(tmp_path):
    path = tmp_path / "scoring_log.csv"
    assert append_scoring_log(path, run=fake_git(dirty=True))["dirty"] == "true"
    assert append_scoring_log(path, run=fake_git())["dirty"] == "false"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "utc,commit,dirty,diff_hash" and len(lines) == 3
