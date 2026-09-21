"""Jev / NLI / bge answerers, the SQLite cache, and the answer runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from asktwice.config import (
    BACKUP_SECONDS,
    BGE_BATCH,
    BGE_DIMS,
    BGE_MODEL,
    CHUNK_OVERLAP,
    DATA_DIR,
    FAILURE_RATE_STOP,
    FROZEN_DIR,
    JEV_MAX_CHARS,
    JEV_MODEL,
    JEV_WORKERS,
    MAX_SEQ,
    NLI_BATCH,
    NLI_GROUP,
    NLI_MODEL,
    SPECIAL_TOKENS,
    SPLITS,
    TRAIN_END,
    TRAIN_START,
)
from asktwice.questions import Question, QuestionSet, jev_payload, load_questions, question_set_hash

CallFn = Callable[[str, dict[str, Any], str], dict[str, Any]]
# (input_ids, attention_mask), both int64 (batch, width) -> one output row per sequence
Forward = Callable[[np.ndarray, np.ndarray], np.ndarray]


class ModelVersionError(RuntimeError):
    pass


class JevHttpError(RuntimeError):
    def __init__(self, status_code: int, body: str = "") -> None:
        super().__init__(f"Jev HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_head(text: str, max_chars: int | None = JEV_MAX_CHARS) -> str:
    return text if max_chars is None else text[:max_chars]


def expected_level(probabilities: dict[str, float]) -> float:
    return sum(int(k) * float(v) for k, v in probabilities.items())


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def entailment_index(label2id: dict[str, int]) -> int:
    for key, idx in label2id.items():
        if str(key).lower() in {"entailment", "entails"}:
            return int(idx)
    raise KeyError(f"no entailment label in {label2id}")


def chunk_token_ids(
    premise_ids: list[int],
    hyp_len: int,
    *,
    max_len: int = MAX_SEQ,
    special: int = SPECIAL_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[list[int]]:
    room = max_len - hyp_len - special
    if room <= overlap:
        raise ValueError(f"hypothesis length {hyp_len} leaves no room in {max_len}")
    chunks = [premise_ids[: room]]
    start = room - overlap
    while start + overlap < len(premise_ids):
        chunks.append(premise_ids[start : start + room])
        start += room - overlap
    return chunks


def nli_noul_from_chunk_probs(p_entail: np.ndarray) -> float:
    return float(np.max(p_entail))


def nli_score_from_chunk_logits(level_max_logits: np.ndarray) -> float:
    p = softmax(np.asarray(level_max_logits, dtype=float))
    return float(np.arange(len(p)) @ p)


class AnswerCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
              answerer TEXT NOT NULL,
              model_version TEXT NOT NULL,
              question_set_hash TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              complaint_id TEXT,
              payload TEXT,
              error TEXT,
              input_tokens INTEGER,
              created_at TEXT NOT NULL,
              PRIMARY KEY (answerer, model_version, question_set_hash, text_sha256)
            )
            """
        )
        self.conn.commit()

    def get(self, answerer: str, model_version: str, question_set_hash: str, text_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT payload, error, input_tokens, complaint_id FROM answers
            WHERE answerer=? AND model_version=? AND question_set_hash=? AND text_sha256=?
            """,
            (answerer, model_version, question_set_hash, text_hash),
        ).fetchone()
        if row is None:
            return None
        payload, error, tokens, cid = row
        return {"payload": json.loads(payload) if payload else None, "error": error, "input_tokens": tokens, "complaint_id": cid}

    def put(
        self,
        *,
        answerer: str,
        model_version: str,
        question_set_hash: str,
        text_hash: str,
        complaint_id: str | None,
        payload: dict[str, Any] | None,
        error: str | None = None,
        input_tokens: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO answers
            (answerer, model_version, question_set_hash, text_sha256, complaint_id,
             payload, error, input_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                answerer,
                model_version,
                question_set_hash,
                text_hash,
                complaint_id,
                json.dumps(payload) if payload is not None else None,
                error,
                input_tokens,
            ),
        )
        self.conn.commit()

    def backup(self, dest: Path | str) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        other = sqlite3.connect(dest)
        self.conn.backup(other)
        other.close()

    def close(self) -> None:
        self.conn.close()


def restore_cache(backup: Path | str, dest: Path | str) -> AnswerCache:
    src = sqlite3.connect(backup)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    other = sqlite3.connect(dest_path)
    src.backup(other)
    other.close()
    src.close()
    return AnswerCache(dest_path)


class Backup:
    """Copies the local cache to e.g. Google Drive every BACKUP_SECONDS and on flush()."""

    def __init__(self, cache: AnswerCache, dest: Path | str | None, every: float = BACKUP_SECONDS) -> None:
        self.cache, self.dest, self.every = cache, dest, every
        self.last = time.monotonic()

    def tick(self) -> None:
        if self.dest and time.monotonic() - self.last >= self.every:
            self.flush()

    def flush(self) -> None:
        if self.dest:
            self.cache.backup(self.dest)
            self.last = time.monotonic()


def open_cache(path: Path | str, backup: Path | str | None = None) -> AnswerCache:
    """Local SQLite only; on a fresh runtime, restore from the backup copy first."""
    if backup and not Path(path).exists() and Path(backup).exists():
        return restore_cache(backup, path)
    return AnswerCache(path)


# ---------------------------------------------------------------- Jev


def _http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def serialize_jev_result(result: Any) -> dict[str, Any]:
    return result if isinstance(result, dict) else result.model_dump(mode="json")


def make_jev_call() -> CallFn:
    from typesafe_sdk import Noul, RetryPolicy, Score, TypeSafeClient

    # The SDK retries 408/429/5xx (529 = overloaded) with backoff and honours Retry-After.
    # Its default (2 retries, 5 s cap) is sized for interactive calls, not an hours-long bulk run.
    client = TypeSafeClient(retry=RetryPolicy(max_retries=8, backoff_max=60.0))

    def call(state: str, questions: dict[str, Any], model: str) -> dict[str, Any]:
        qs = {
            k: Noul(instructions=q["instructions"], criteria=q.get("criteria"))
            if q["type"] == "noul"
            else Score(instructions=q["instructions"], criteria=q["criteria"])
            for k, q in questions.items()
        }
        return serialize_jev_result(client.system_one(state, qs, model=model))

    return call


def ask_jev(text: str, spec: QuestionSet, *, call: CallFn, model: str = JEV_MODEL) -> dict[str, Any]:
    try:
        result = call(truncate_head(text), jev_payload(spec), model)
    except Exception as exc:
        status = _http_status(exc)
        if status is None:
            raise
        raise JevHttpError(status, str(exc)) from exc
    if result.get("model") != model:
        raise ModelVersionError(f"model mismatch: {result.get('model')!r} != {model!r}")
    return result


def encode_noul_jev(result: dict[str, Any], question_id: str) -> float:
    return float(result["answers"][question_id]["noul"])


def encode_score_jev(result: dict[str, Any], question_id: str) -> float:
    return expected_level(result["answers"][question_id]["probabilities"])


def _todo(rows: list[dict], cache: AnswerCache, answerer: str, model: str, qhash: str) -> list[dict]:
    """Uncached rows, one per distinct narrative (the cache is keyed by text)."""
    seen: dict[str, dict] = {}
    for r in rows:
        h = text_sha256(r["narrative"])
        if h not in seen and cache.get(answerer, model, qhash, h) is None:
            seen[h] = r
    return list(seen.values())


def run_jev(
    rows: list[dict],
    cache: AnswerCache,
    spec: QuestionSet,
    *,
    call: CallFn,
    workers: int = JEV_WORKERS,
    backup: Backup | None = None,
) -> int:
    qhash = question_set_hash(spec)
    todo = _todo(rows, cache, "jev", JEV_MODEL, qhash)

    def work(r: dict) -> tuple[dict, dict | None, str | None]:
        try:
            return r, ask_jev(r["narrative"], spec, call=call), None
        except JevHttpError as exc:
            if exc.status_code != 422:
                raise
            return r, None, f"422:{exc.body}"

    pool = ThreadPoolExecutor(max_workers=workers)
    t0 = time.perf_counter()
    n422 = 0
    try:
        for done, (r, payload, error) in enumerate(pool.map(work, todo), 1):
            if done % 100 == 0:
                print(f"jev {done}/{len(todo)}  {done / (time.perf_counter() - t0):.1f} rows/s", flush=True)
            # 422 is request validation (a malformed question), not narrative content:
            # a systematic 422 must stop the run, not fill the cache with failures.
            n422 += error is not None
            if n422 > 10 and n422 > FAILURE_RATE_STOP * done:
                raise JevHttpError(422, f"{n422}/{done} rows rejected; last: {error}")
            tokens = ((payload or {}).get("usage") or {}).get("input_tokens")
            cache.put(
                answerer="jev",
                model_version=JEV_MODEL,
                question_set_hash=qhash,
                text_hash=text_sha256(r["narrative"]),
                complaint_id=r["complaint_id"],
                payload=payload,
                error=error,
                input_tokens=tokens,
            )
            if backup:
                backup.tick()
    finally:
        pool.shutdown(cancel_futures=True)
        if backup:
            backup.flush()
    return len(todo)


# ---------------------------------------------------------------- NLI / bge


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((len(seqs), max(len(s) for s in seqs)), pad_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return ids, mask


def run_sorted(seqs: list[list[int]], forward: Forward, pad_id: int, batch_size: int) -> np.ndarray:
    """Length-sorted, padded batches; outputs come back in input order."""
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out: list[np.ndarray | None] = [None] * len(seqs)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        res = forward(*_pad([seqs[i] for i in idx], pad_id))
        for k, i in enumerate(idx):
            out[i] = res[k]
    return np.stack(out)  # type: ignore[arg-type]


def with_special(tokenizer: Any, a: list[int], b: list[int] | None = None) -> list[int]:
    """[CLS] a [SEP] (b [SEP]): the input format of both DeBERTa-v3 and BERT/bge.
    Built by hand: transformers v5 removed build_inputs_with_special_tokens."""
    out = [tokenizer.cls_token_id, *a, tokenizer.sep_token_id]
    return out if b is None else [*out, *b, tokenizer.sep_token_id]


def _hypotheses(spec: QuestionSet) -> list[tuple[Question, int, str]]:
    out = []
    for q in spec.questions:
        if q.type == "noul":
            out.append((q, 0, q.nli_hypothesis))
        else:
            out.extend((q, i, h) for i, h in enumerate(q.nli_hypotheses))
    return out


def nli_scores(
    texts: list[str],
    spec: QuestionSet,
    *,
    tokenizer: Any,
    forward: Forward,
    label2id: dict[str, int],
    batch_size: int = NLI_BATCH,
) -> list[dict[str, Any]]:
    """3-way softmax; entailment index from label2id. Noul: max P(entail) over chunks.
    Score: per level, max entailment logit over chunks."""
    ent = entailment_index(label2id)
    hyps = [(q, lvl, tokenizer.encode(h, add_special_tokens=False)) for q, lvl, h in _hypotheses(spec)]
    seqs: list[list[int]] = []
    owner: list[tuple[int, int]] = []  # (text index, hypothesis index)
    n_chunks = np.zeros(len(texts), dtype=int)
    for t, text in enumerate(texts):
        premise = tokenizer.encode(text, add_special_tokens=False)
        for k, (_, _, hyp_ids) in enumerate(hyps):
            chunks = chunk_token_ids(premise, len(hyp_ids))
            n_chunks[t] = max(n_chunks[t], len(chunks))
            for chunk in chunks:
                seqs.append(with_special(tokenizer, chunk, hyp_ids))
                owner.append((t, k))
    logits = run_sorted(seqs, forward, tokenizer.pad_token_id, batch_size)
    p_ent = np.full((len(texts), len(hyps)), -np.inf)
    l_ent = np.full((len(texts), len(hyps)), -np.inf)
    probs = softmax(logits)[:, ent]
    for (t, k), p, l in zip(owner, probs, logits[:, ent]):
        p_ent[t, k] = max(p_ent[t, k], p)
        l_ent[t, k] = max(l_ent[t, k], l)

    out = []
    for t in range(len(texts)):
        answers: dict[str, Any] = {}
        for k, (q, lvl, _) in enumerate(hyps):
            if q.type == "noul":
                answers[q.id] = {"type": "noul", "p_entail": float(p_ent[t, k])}
            else:
                answers.setdefault(q.id, {"type": "score", "level_logits": []})["level_logits"].append(float(l_ent[t, k]))
        out.append({"model": NLI_MODEL, "max_chunks": int(n_chunks[t]), "answers": answers})
    return out


def embed_texts(texts: list[str], *, tokenizer: Any, forward: Forward, batch_size: int = BGE_BATCH) -> np.ndarray:
    """bge: 512-token chunks, token-mean per chunk, mean over chunks."""
    seqs: list[list[int]] = []
    owner: list[int] = []
    for t, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        for chunk in chunk_token_ids(ids, 0, special=2):  # [CLS] x [SEP]
            seqs.append(with_special(tokenizer, chunk))
            owner.append(t)
    vecs = run_sorted(seqs, forward, tokenizer.pad_token_id, batch_size)
    if vecs.shape[1] != BGE_DIMS:
        raise ValueError(f"bge dim {vecs.shape[1]} != {BGE_DIMS}")
    out = np.zeros((len(texts), BGE_DIMS), dtype=np.float64)
    np.add.at(out, owner, vecs)
    return (out / np.bincount(owner, minlength=len(texts))[:, None]).astype(np.float32)


def torch_forward(model: Any, *, pool: bool) -> Forward:
    """GPU + fp16 autocast when available. pool=False -> logits, pool=True -> masked token mean."""
    import contextlib

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def forward(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        i = torch.from_numpy(ids).to(device)
        m = torch.from_numpy(mask).to(device)
        fp16 = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), fp16:
            out = model(input_ids=i, attention_mask=m)
        if not pool:
            return out.logits.float().cpu().numpy()
        mf = m.unsqueeze(-1).float()
        return ((out.last_hidden_state.float() * mf).sum(1) / mf.sum(1)).cpu().numpy()

    return forward


def load_nli() -> tuple[Any, Forward, dict[str, int]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    return AutoTokenizer.from_pretrained(NLI_MODEL), torch_forward(model, pool=False), dict(model.config.label2id)


def load_bge() -> tuple[Any, Forward]:
    from transformers import AutoModel, AutoTokenizer

    return AutoTokenizer.from_pretrained(BGE_MODEL), torch_forward(AutoModel.from_pretrained(BGE_MODEL), pool=True)


def run_nli(
    rows: list[dict],
    cache: AnswerCache,
    spec: QuestionSet,
    *,
    tokenizer: Any,
    forward: Forward,
    label2id: dict[str, int],
    group: int = NLI_GROUP,
    backup: Backup | None = None,
) -> int:
    qhash = question_set_hash(spec)
    todo = _todo(rows, cache, "nli", NLI_MODEL, qhash)
    try:
        for start in range(0, len(todo), group):
            batch = todo[start : start + group]
            t0 = time.perf_counter()
            payloads = nli_scores([r["narrative"] for r in batch], spec, tokenizer=tokenizer, forward=forward, label2id=label2id)
            per_row = (time.perf_counter() - t0) / len(batch)
            for r, payload in zip(batch, payloads):
                payload["seconds"] = per_row
                cache.put(
                    answerer="nli",
                    model_version=NLI_MODEL,
                    question_set_hash=qhash,
                    text_hash=text_sha256(r["narrative"]),
                    complaint_id=r["complaint_id"],
                    payload=payload,
                )
            print(f"nli {start + len(batch)}/{len(todo)}  {1 / per_row:.1f} rows/s", flush=True)
            if backup:
                backup.tick()
    finally:
        if backup:
            backup.flush()
    return len(todo)


# ---------------------------------------------------------------- export


def split_rows(frozen: Path | str) -> list[dict]:
    rows = []
    for s in SPLITS:
        rows += pq.read_table(Path(frozen) / f"{s}.parquet", columns=["complaint_id", "narrative"]).to_pylist()
    return rows


def nli_columns(spec: QuestionSet) -> list[str]:
    return [q.id if q.type == "noul" else f"{q.id}@{lvl}" for q, lvl, _ in _hypotheses(spec)]


def export_answers(frozen: Path | str, cache: AnswerCache, spec: QuestionSet) -> dict[str, int]:
    """Cache -> frozen/jev_answers.parquet (zstd JSON) and frozen/nli_answers.parquet (float16)."""
    frozen = Path(frozen)
    qhash = question_set_hash(spec)
    rows = split_rows(frozen)
    jev, cols = [], nli_columns(spec)
    nli: dict[str, list] = {"complaint_id": [], "ok": [], "max_chunks": [], "seconds": [], **{c: [] for c in cols}}
    missing = {"jev": 0, "nli": 0}
    for r in rows:
        h = text_sha256(r["narrative"])
        j = cache.get("jev", JEV_MODEL, qhash, h)
        missing["jev"] += j is None
        jev.append(
            {
                "complaint_id": r["complaint_id"],
                "payload": json.dumps(j["payload"]) if j and j["payload"] else None,
                "error": j["error"] if j else "missing",
                "input_tokens": j["input_tokens"] if j else None,
            }
        )
        n = cache.get("nli", NLI_MODEL, qhash, h)
        missing["nli"] += n is None
        p = (n or {}).get("payload") or {}
        nli["complaint_id"].append(r["complaint_id"])
        nli["ok"].append(bool(p))
        nli["max_chunks"].append(p.get("max_chunks", 0))
        nli["seconds"].append(p.get("seconds", 0.0))
        flat: list[float] = []
        for q in spec.questions:
            a = p.get("answers", {}).get(q.id, {})
            flat += [a.get("p_entail", 0.0)] if q.type == "noul" else a.get("level_logits", [0.0] * len(q.levels))
        for c, v in zip(cols, flat):
            nli[c].append(v)
    pq.write_table(pa.Table.from_pylist(jev), frozen / "jev_answers.parquet", compression="zstd")
    table = pa.table(
        {
            k: pa.array(np.asarray(v, dtype=np.float16)) if k in cols else pa.array(v)
            for k, v in nli.items()
        }
    )
    pq.write_table(table, frozen / "nli_answers.parquet", compression="zstd")
    return missing


# ---------------------------------------------------------------- CLI


PROBE = QuestionSet(
    status="probe",
    direct=Question(id="probe", type="noul", jev="Is this text written in English?"),
    questions=(),
)


def probe_long(candidates: Path | str, n: int, call: CallFn) -> None:
    """Day 0: send the n longest train-pool narratives with a throwaway question.
    Prints length, status, tokens and latency only, never answer values."""
    import duckdb

    rows = duckdb.connect().execute(
        f"""
        SELECT complaint_id, narrative FROM read_parquet(?)
        WHERE date_received BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
        ORDER BY length(narrative) DESC LIMIT {int(n)}
        """,
        [str(candidates)],
    ).fetchall()
    for cid, text in rows:
        t0 = time.perf_counter()
        try:
            out = ask_jev(text, PROBE, call=call)
            status, tokens = "ok", out.get("usage", {}).get("input_tokens")
        except JevHttpError as exc:
            status, tokens = str(exc.status_code), None
        print(f"{cid}\tchars={len(text)}\tstatus={status}\ttokens={tokens}\t{time.perf_counter() - t0:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asktwice.answer")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="answer every split row; resumes from the cache")
    r.add_argument("answerer", choices=["jev", "nli"])
    r.add_argument("--frozen", default=str(FROZEN_DIR))
    r.add_argument("--cache", default=str(DATA_DIR / "answers.sqlite"))
    r.add_argument("--backup", help="e.g. /content/drive/MyDrive/asktwice/answers.sqlite")
    r.add_argument("--limit", type=int, help="first N rows only (throughput checks)")
    r.add_argument("--workers", type=int, default=JEV_WORKERS)
    e = sub.add_parser("export", help="cache -> frozen/*_answers.parquet")
    e.add_argument("--frozen", default=str(FROZEN_DIR))
    e.add_argument("--cache", default=str(DATA_DIR / "answers.sqlite"))
    pr = sub.add_parser("probe", help="Day 0: longest narratives vs Jev, no answer values")
    pr.add_argument("--candidates", default=str(DATA_DIR / "candidates.parquet"))
    pr.add_argument("-n", type=int, default=20)
    c = sub.add_parser("check-dev", help="after export: flag questions whose dev answers are near-constant")
    c.add_argument("--frozen", default=str(FROZEN_DIR))
    args = p.parse_args(argv)

    if args.cmd == "probe":
        probe_long(args.candidates, args.n, make_jev_call())
        return
    spec = load_questions()
    if args.cmd == "export":
        print("missing:", export_answers(args.frozen, AnswerCache(args.cache), spec))
        return
    if args.cmd == "check-dev":
        from asktwice.features import check_dev

        def fmt(a: dict) -> str:
            return f"{a['mean']:6.3f} ± {a['std']:.3f} (n={a['n']})" if a["n"] else "no answers"

        rows = check_dev(args.frozen, spec)
        print(f"{'question':28} {'type':6} {'jev':26} {'nli':26}")
        for r in rows:
            flag = "  BROKEN: " + ", ".join(r["broken"]) if r["broken"] else ""
            print(f"{r['question']:28} {r['type']:6} {fmt(r['jev']):26} {fmt(r['nli']):26}{flag}")
        broken = [r["question"] for r in rows if r["broken"]]
        print(
            f"\n{len(broken)} broken: {', '.join(broken)}. Fix for both answerers, tag prereg-v2, add a CHANGELOG entry, "
            "and only before the first `asktwice.report` run."
            if broken
            else "\nNo structurally broken questions."
        )
        return
    cache = open_cache(args.cache, args.backup)
    backup = Backup(cache, args.backup)
    rows = split_rows(args.frozen)[: args.limit]
    if args.answerer == "jev":
        n = run_jev(rows, cache, spec, call=make_jev_call(), workers=args.workers, backup=backup)
    else:
        tok, fwd, label2id = load_nli()
        n = run_nli(rows, cache, spec, tokenizer=tok, forward=fwd, label2id=label2id, backup=backup)
    print(f"{args.answerer}: answered {n} new rows")


if __name__ == "__main__":
    main()
