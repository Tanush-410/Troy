"""Provenance of the frozen files (CLAUDE.md rule 2), recorded in every run.

For each frozen file: the last commit that touched it, its sha256, and whether
the working copy differs from that commit. A run with a dirty frozen file must
not count as a result.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# The harm oracle's rule files join this list in milestone 3.
FROZEN_FILES: tuple[str, ...] = ("policy/permissions.py",)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def frozen_provenance(files: tuple[str, ...] = FROZEN_FILES) -> dict[str, Any]:
    out: dict[str, Any] = {"head": _git("rev-parse", "HEAD"), "files": {}}
    for rel in files:
        out["files"][rel] = {
            "last_commit": _git("log", "-1", "--format=%H", "--", rel) or None,
            "sha256": hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest(),
            "dirty": bool(_git("status", "--porcelain", "--", rel)),
        }
    out["any_dirty"] = any(f["dirty"] or f["last_commit"] is None for f in out["files"].values())
    return out
