"""Git helpers for the prereg guard and scoring log."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Callable

from asktwice.config import PREREG_TAG

Run = Callable[..., subprocess.CompletedProcess]


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    run: Run | None = None,
) -> subprocess.CompletedProcess:
    fn = run or subprocess.run
    return fn(args, cwd=cwd, capture_output=True, text=True)


def head_after_tag(
    tag: str = PREREG_TAG,
    *,
    cwd: Path | None = None,
    run: Run | None = None,
) -> bool:
    r = _run(["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=cwd, run=run)
    return r.returncode == 0


def snapshot(
    *,
    cwd: Path | None = None,
    run: Run | None = None,
) -> dict[str, str | bool]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=cwd, run=run)
    commit = head.stdout.strip() if head.returncode == 0 else ""
    status = _run(["git", "status", "--porcelain"], cwd=cwd, run=run)
    dirty = bool((status.stdout or "").strip())
    diff = _run(["git", "diff", "HEAD"], cwd=cwd, run=run)
    diff_hash = hashlib.sha256((diff.stdout or "").encode()).hexdigest()
    return {"commit": commit, "dirty": dirty, "diff_hash": diff_hash}
