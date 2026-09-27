"""CODEOWNERS must list exactly the frozen files, so every change to one needs owner review."""

from pathlib import Path

from experiments.provenance import FROZEN_FILES, REPO_ROOT


def test_codeowners_lists_every_frozen_file():
    lines = (REPO_ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
    owned = {ln.split()[0].lstrip("/") for ln in lines if ln.strip() and not ln.startswith("#")}
    assert owned == set(FROZEN_FILES)
    assert all((REPO_ROOT / f).is_file() for f in FROZEN_FILES)
