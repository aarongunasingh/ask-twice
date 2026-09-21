from __future__ import annotations

import numpy as np
import pytest

from asktwice.answer import (
    AnswerCache,
    JevHttpError,
    ModelVersionError,
    ask_jev,
    chunk_token_ids,
    embed_texts,
    entailment_index,
    nli_noul_from_chunk_probs,
    nli_score_from_chunk_logits,
    nli_scores,
    open_cache,
    restore_cache,
    run_jev,
    run_sorted,
    text_sha256,
)
from asktwice.config import BGE_DIMS, JEV_MODEL, MAX_SEQ, SPECIAL_TOKENS
from asktwice.questions import QuestionSet, question_set_hash as qhash
from tests.helpers import CLS, LABEL2ID, SEP, FakeTok


def _ok_result(spec):
    answers = {}
    for q in spec.questions:
        if q.type == "noul":
            answers[q.id] = {"type": "noul", "noul": 0.7}
        else:
            k = len(q.levels)
            answers[q.id] = {"type": "score", "probabilities": {str(i): 1.0 / k for i in range(k)}}
    answers[spec.direct.id] = {"type": "noul", "noul": 0.2}
    return {"model": JEV_MODEL, "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}}


def test_cache_key_independent_of_question_order(questions):
    swapped = QuestionSet(status=questions.status, direct=questions.direct, questions=tuple(reversed(questions.questions)))
    assert qhash(questions) == qhash(swapped)
    assert text_sha256("abc") == text_sha256("abc")


def test_resume_skips_cached_rows(tmp_cache, questions):
    calls = []

    def call(state, payload, model):
        calls.append(state)
        return _ok_result(questions)

    rows = [
        {"complaint_id": "1", "narrative": "hello"},
        {"complaint_id": "2", "narrative": "hello"},  # same text: answered once
        {"complaint_id": "3", "narrative": "bye"},
    ]
    assert run_jev(rows, tmp_cache, questions, call=call, workers=2) == 2
    assert run_jev(rows, tmp_cache, questions, call=call, workers=2) == 0
    assert sorted(calls) == ["bye", "hello"]
    assert tmp_cache.get("jev", JEV_MODEL, qhash(questions), text_sha256("bye"))["input_tokens"] == 10


def test_422_recorded_not_retried(tmp_cache, questions):
    calls = {"n": 0}

    def call(state, payload, model):
        calls["n"] += 1
        raise JevHttpError(422, "too long")

    run_jev([{"complaint_id": "1", "narrative": "long"}], tmp_cache, questions, call=call, workers=1)
    rec = tmp_cache.get("jev", JEV_MODEL, qhash(questions), text_sha256("long"))
    assert rec["error"].startswith("422")
    assert rec["payload"] is None
    assert calls["n"] == 1


def test_version_mismatch_aborts(tmp_cache, questions):
    def call(state, payload, model):
        return {**_ok_result(questions), "model": "jev-9.9.9"}

    with pytest.raises(ModelVersionError):
        run_jev([{"complaint_id": "1", "narrative": "hi"}], tmp_cache, questions, call=call, workers=1)
    assert tmp_cache.get("jev", JEV_MODEL, qhash(questions), text_sha256("hi")) is None


def test_sdk_422_is_recorded(tmp_cache, questions):
    class SdkError(Exception):  # typesafe_sdk exceptions carry .status (retries happen inside the SDK)
        status = 422

    def call(state, payload, model):
        raise SdkError("malformed")

    with pytest.raises(JevHttpError) as err:
        ask_jev("hi", questions, call=call)
    assert err.value.status_code == 422
    run_jev([{"complaint_id": "1", "narrative": "hi"}], tmp_cache, questions, call=call, workers=1)
    assert tmp_cache.get("jev", JEV_MODEL, qhash(questions), text_sha256("hi"))["error"].startswith("422")


def test_systematic_422_stops_the_run(tmp_cache, questions):
    def call(state, payload, model):
        raise JevHttpError(422, "questions.x.criteria: field required")

    rows = [{"complaint_id": str(i), "narrative": f"row {i}"} for i in range(50)]
    with pytest.raises(JevHttpError, match="rows rejected"):
        run_jev(rows, tmp_cache, questions, call=call, workers=1)


def test_backup_restore_roundtrip(tmp_path, questions):
    src = AnswerCache(tmp_path / "a.sqlite")
    key = (JEV_MODEL, qhash(questions), text_sha256("z"))
    src.put(answerer="jev", model_version=key[0], question_set_hash=key[1], text_hash=key[2], complaint_id="9", payload={"model": JEV_MODEL})
    src.backup(tmp_path / "drive" / "b.sqlite")
    src.close()
    restored = restore_cache(tmp_path / "drive" / "b.sqlite", tmp_path / "c.sqlite")
    assert restored.get("jev", *key)["payload"]["model"] == JEV_MODEL
    # a fresh runtime (no local db) restores from the backup automatically
    fresh = open_cache(tmp_path / "new.sqlite", tmp_path / "drive" / "b.sqlite")
    assert fresh.get("jev", *key) is not None


def test_nli_pairs_carry_special_tokens_and_fit(questions):
    seen = []

    def forward(ids, mask):
        seen.append((ids, mask))
        return np.zeros((len(ids), 3))

    long_text = " ".join(f"w{i}" for i in range(1200))
    out = nli_scores([long_text, "short text"], questions, tokenizer=FakeTok(), forward=forward, label2id=LABEL2ID, batch_size=16)
    for ids, mask in seen:
        for seq, m in zip(ids, mask):
            seq = seq[m.astype(bool)]
            assert seq[0] == CLS and seq[-1] == SEP and list(seq).count(SEP) == 2
            assert len(seq) <= MAX_SEQ
    widths = [ids.shape[1] for ids, _ in seen]
    assert widths == sorted(widths)  # length-sorted batches
    assert out[0]["max_chunks"] > 1 and out[1]["max_chunks"] == 1
    score_q = questions.score[0]
    assert len(out[0]["answers"][score_q.id]["level_logits"]) == len(score_q.levels)


def test_swapped_label2id_still_entailment(questions):
    label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
    assert entailment_index(label2id) == 2

    def forward(ids, mask):
        return np.tile([0.0, 0.0, 5.0], (len(ids), 1))

    out = nli_scores(["the bank refunded the fee"], questions, tokenizer=FakeTok(), forward=forward, label2id=label2id)
    assert out[0]["answers"][questions.noul[0].id]["p_entail"] > 0.9


def test_run_sorted_keeps_input_order():
    seqs = [[5, 5, 5], [5], [5, 5]]
    out = run_sorted(seqs, lambda ids, mask: mask.sum(axis=1, keepdims=True), pad_id=0, batch_size=2)
    assert out[:, 0].tolist() == [3, 1, 2]


def test_embed_texts_mean_over_chunks():
    def forward(ids, mask):  # every dim = sequence length
        return np.repeat(mask.sum(axis=1, keepdims=True).astype(float), BGE_DIMS, axis=1)

    vec = embed_texts(["a " * 1000, "b c"], tokenizer=FakeTok(), forward=forward)
    assert vec.shape == (2, BGE_DIMS)
    assert vec[1, 0] == 4  # [CLS] b c [SEP]
    assert 4 < vec[0, 0] <= MAX_SEQ


def test_chunk_leaves_room_for_hypothesis():
    premise = list(range(600))
    chunks = chunk_token_ids(premise, 20)
    assert all(len(c) <= MAX_SEQ - 20 - SPECIAL_TOKENS for c in chunks)
    assert chunks[0][0] == 0 and chunks[-1][-1] == 599
    assert chunk_token_ids(list(range(489)), 20) == [list(range(489))]  # exactly one window


def test_score_chunk_rule_longer_hypothesis_shorter_premise():
    premise = list(range(800))
    assert max(map(len, chunk_token_ids(premise, 5))) > max(map(len, chunk_token_ids(premise, 40)))


def test_nli_pool_math():
    assert nli_noul_from_chunk_probs(np.array([0.1, 0.8, 0.2])) == 0.8
    assert abs(nli_score_from_chunk_logits(np.zeros(3)) - 1.0) < 1e-9  # uniform over 3 levels


def test_noul_criteria_reach_the_payload(tmp_path):
    from asktwice.questions import jev_payload, load_questions

    path = tmp_path / "q.yaml"
    path.write_text(
        """
status: throwaway-day0
direct:
  id: d
  type: noul
  jev: Will this end with relief?
questions:
  - id: repeat
    type: noul
    jev: Has the consumer already contacted the company about this?
    nli_hypothesis: The consumer already contacted the company.
    criteria:
      true: Mentions a prior call, letter, or dispute with the company
      false: No sign of earlier contact
""",
        encoding="utf-8",
    )
    spec = load_questions(path)
    assert jev_payload(spec)["repeat"]["criteria"] == {
        "true": "Mentions a prior call, letter, or dispute with the company",
        "false": "No sign of earlier contact",
    }
    assert "criteria" not in jev_payload(spec)["d"]
