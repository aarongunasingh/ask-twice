"""Real transformers/torch plumbing on tiny random models (no downloads). Skipped without the nli extra."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from asktwice.answer import _pad, embed_texts, nli_scores, torch_forward, with_special  # noqa: E402
from asktwice.config import BGE_DIMS  # noqa: E402
from tests.helpers import LABEL2ID, FakeTok  # noqa: E402


def test_special_tokens_match_real_tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join("[PAD] [UNK] [CLS] [SEP] [MASK] the bank charged a fee asks refund".split()))
    tok = transformers.BertTokenizer(vocab_file=str(vocab))
    a = tok.encode("the bank charged a fee", add_special_tokens=False)
    b = tok.encode("asks refund", add_special_tokens=False)
    assert with_special(tok, a, b) == tok("the bank charged a fee", "asks refund")["input_ids"]
    assert with_special(tok, a) == tok("the bank charged a fee")["input_ids"]


def test_torch_forward_nli_and_bge(questions):
    torch.manual_seed(0)
    cfg = transformers.DebertaV2Config(
        vocab_size=1000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        num_labels=3, label2id=LABEL2ID, id2label={v: k for k, v in LABEL2ID.items()},
    )
    fwd = torch_forward(transformers.DebertaV2ForSequenceClassification(cfg), pool=False)
    tok = FakeTok()
    texts = [" ".join(f"w{i}" for i in range(1500)), "short narrative about a fee"]
    out = nli_scores(texts, questions, tokenizer=tok, forward=fwd, label2id=LABEL2ID, batch_size=8)
    assert [o["max_chunks"] for o in out] == [4, 1]
    seq = with_special(tok, tok.encode("short narrative"), tok.encode("a hypothesis"))
    assert np.allclose(fwd(*_pad([seq], 0))[0], fwd(*_pad([seq, seq + list(range(3, 300))], 0))[0], atol=1e-4)

    bert = transformers.BertModel(
        transformers.BertConfig(vocab_size=1000, hidden_size=BGE_DIMS, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
    )
    vec = embed_texts(texts, tokenizer=tok, forward=torch_forward(bert, pool=True))
    assert vec.shape == (2, BGE_DIMS) and np.isfinite(vec).all()
