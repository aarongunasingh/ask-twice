from __future__ import annotations

import numpy as np
import pytest

from asktwice.answer import expected_level
from asktwice.features import (
    TooManyFailures,
    check_failure_rate,
    encode_jev_row,
    encode_nli_row,
    fit_tabular_encoder,
    shared_mask,
)


def test_nli_row_uses_same_encodings(questions):
    row = {}
    for q in questions.questions:
        if q.type == "noul":
            row[q.id] = 0.25
        else:
            row.update({f"{q.id}@{i}": 0.0 for i in range(len(q.levels))})
    vec = encode_nli_row(row, questions)
    assert vec[0] == pytest.approx(0.25)
    k = len(questions.score[0].levels)
    assert vec[[q.type == "score" for q in questions.questions].index(True)] == pytest.approx((k - 1) / 2)


def test_expected_level_math():
    assert expected_level({"0": 0.2, "1": 0.5, "2": 0.3}) == pytest.approx(1.1)


def test_shared_mask_drops_row_missing_any_answerer():
    jev = np.array([True, True, False, True])
    nli = np.array([True, False, True, True])
    bge = np.array([True, True, True, True])
    assert list(shared_mask(jev, nli, bge)) == [True, False, False, True]


def test_failure_rate_over_one_percent_raises():
    check_failure_rate(0, 100, "jev")
    with pytest.raises(TooManyFailures):
        check_failure_rate(2, 100, "jev")


def test_test_only_company_maps_to_other():
    train = [
        {
            "product": "Credit card",
            "sub_product": "General-purpose",
            "issue": "Fees",
            "sub_issue": "",
            "company": "Bank A",
            "state": "CA",
            "submitted_via": "Web",
            "tags": "",
            "date_received": "2020-01-15",
        }
        for _ in range(3)
    ]
    enc = fit_tabular_encoder(train, k=1)
    test = [{**train[0], "company": "Unseen Bank Z", "date_received": "2024-06-01"}]
    x_tr = enc.transform(train)
    x_te = enc.transform(test)
    company_idx = 4
    other_code = enc.vocabs["company"]["other"]
    assert x_te[0, company_idx] == other_code
    assert x_te[0, company_idx] in set(x_tr[:, company_idx]) | {other_code}
    for field, vocab in enc.vocabs.items():
        assert max(vocab.values()) == enc.max_codes()[field]


def test_no_test_feature_outside_training_range():
    train = [
        {
            "product": "Credit card",
            "sub_product": "A",
            "issue": "Fees",
            "sub_issue": "x",
            "company": "Bank A",
            "state": "CA",
            "submitted_via": "Web",
            "tags": "",
            "date_received": "2020-01-15",
        }
    ]
    enc = fit_tabular_encoder(train)
    test = [{**train[0], "state": "ZZ", "date_received": "2024-12-01"}]
    x_te = enc.transform(test)
    for j, field in enumerate(("product", "sub_product", "issue", "sub_issue", "company", "state", "submitted_via", "tags")):
        assert 0 <= x_te[0, j] <= enc.max_codes()[field]
    assert 1 <= x_te[0, 8] <= 12


def test_jev_encode_uses_probabilities(questions):
    payload = {"answers": {}}
    for q in questions.questions:
        if q.type == "noul":
            payload["answers"][q.id] = {"type": "noul", "noul": 0.25}
        else:
            payload["answers"][q.id] = {
                "type": "score",
                "probabilities": {str(i): (1.0 if i == 1 else 0.0) for i in range(len(q.levels))},
            }
    vec = encode_jev_row(payload, questions)
    assert vec.shape == (len(questions.questions),)
    assert vec[0] == pytest.approx(0.25)
