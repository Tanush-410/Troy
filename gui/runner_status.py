from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunnerStatus:
    active: bool
    reason: str | None = None
    source: Path | None = None

    def __bool__(self) -> bool:
        return self.active

    @property
    def is_active(self) -> bool:
        return self.active


_LOCK_NAMES = (
    ".runner.lock",
    "runner.lock",
    ".run.lock",
    "run.lock",
    ".pilot.lock",
    "pilot.lock",
    ".experiment.lock",
    "experiment.lock",
)

_PROGRESS_NAMES = (
    ".runner.progress",
    "runner.progress",
    ".runner.progress.json",
    "runner.progress.json",
    ".runner_progress",
    "runner_progress",
    ".runner_progress.json",
    "runner_progress.json",
    ".run.progress",
    "run.progress",
    ".run.progress.json",
    "run.progress.json",
    ".run_progress",
    "run_progress",
    ".run_progress.json",
    "run_progress.json",
    ".pilot.progress",
    "pilot.progress",
    ".pilot.progress.json",
    "pilot.progress.json",
    ".pilot_progress",
    "pilot_progress",
    ".pilot_progress.json",
    "pilot_progress.json",
    "progress.json",
    ".progress",
    "progress",
)

LOCK_NAMES = _LOCK_NAMES
PROGRESS_NAMES = _PROGRESS_NAMES


def _present(path: Path) -> bool:
    try:
        return os.path.lexists(path)
    except OSError:
        return False


def _status(active: bool, reason: str, source: Path | None = None) -> RunnerStatus:
    return RunnerStatus(active=active, reason=reason, source=source)


def _indicator(root: Path, names: tuple[str, ...], kind: str) -> RunnerStatus | None:
    for name in names:
        path = root / name
        if _present(path):
            return _status(True, f"explicit runner {kind} file present: {path}", path)
    return None


def _regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file()
    except OSError:
        return False


def _completed_records(path: Path, run_dir: Path) -> int | None:
    if not _present(path):
        return 0
    if not _regular_file(path):
        return None
    try:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(run_dir.resolve(strict=False)):
            return None
        with path.open(encoding="utf-8") as handle:
            count = 0
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    count += 1
            return count
    except (OSError, UnicodeError):
        return None


def _pilot_incomplete(meta_path: Path, run_dir: Path) -> RunnerStatus | None:
    if not _regular_file(meta_path):
        return _status(True, f"pilot metadata is not a readable regular file: {meta_path}", meta_path)
    try:
        with meta_path.open(encoding="utf-8") as handle:
            metadata: Any = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return _status(True, f"pilot metadata could not be read: {meta_path}", meta_path)
    if not isinstance(metadata, dict) or "episodes_per_cell" not in metadata:
        return _status(True, f"pilot metadata has no usable episodes_per_cell: {meta_path}", meta_path)
    episodes_per_cell = metadata["episodes_per_cell"]
    if isinstance(episodes_per_cell, bool) or not isinstance(episodes_per_cell, (int, float)):
        return _status(True, f"pilot metadata has an invalid episodes_per_cell: {meta_path}", meta_path)
    try:
        numeric = float(episodes_per_cell)
        integer_value = int(episodes_per_cell)
    except (OverflowError, TypeError, ValueError):
        return _status(True, f"pilot metadata has an invalid episodes_per_cell: {meta_path}", meta_path)
    if not math.isfinite(numeric) or integer_value != episodes_per_cell or episodes_per_cell < 0:
        return _status(True, f"pilot metadata has an invalid episodes_per_cell: {meta_path}", meta_path)
    expected = 24 * int(episodes_per_cell)
    completed = _completed_records(run_dir / "episodes.jsonl", run_dir)
    if completed is None:
        return _status(True, f"pilot episode records could not be read: {run_dir}", meta_path)
    if completed < expected:
        return _status(
            True,
            f"pilot run is active or incomplete: {completed} of {expected} episode records in {run_dir}",
            meta_path,
        )
    return None


def detect_active_runner(project_root: Path) -> RunnerStatus:
    root = Path(project_root).expanduser().resolve(strict=False)
    for name in _LOCK_NAMES:
        path = root / name
        if _present(path):
            return _status(True, f"explicit runner lock file present: {path}", path)
    for name in _PROGRESS_NAMES:
        path = root / name
        if _present(path):
            return _status(True, f"explicit runner progress file present: {path}", path)

    logs = root / "logs"
    if not _present(logs):
        return _status(False, "no conventional runner indicators found")

    try:
        if logs.is_symlink():
            return _status(True, f"logs path is a symlink and cannot be safely inspected: {logs}", logs)
        if not logs.is_dir():
            return _status(True, f"logs path is not a directory: {logs}", logs)
    except OSError as exc:
        return _status(True, f"logs path could not be inspected: {exc}", logs)

    found = _indicator(logs, _LOCK_NAMES, "lock")
    if found is not None:
        return found
    found = _indicator(logs, _PROGRESS_NAMES, "progress")
    if found is not None:
        return found

    try:
        entries = sorted(logs.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return _status(True, f"logs directory could not be inspected: {exc}", logs)

    for entry in entries:
        if entry.name.casefold() in {"demo", "transcripts"}:
            continue
        try:
            if entry.is_symlink():
                return _status(True, f"run path is a symlink and cannot be safely inspected: {entry}", entry)
            if not entry.is_dir():
                continue
        except OSError as exc:
            return _status(True, f"run path could not be inspected: {exc}", entry)

        found = _indicator(entry, _LOCK_NAMES, "lock")
        if found is not None:
            return found
        found = _indicator(entry, _PROGRESS_NAMES, "progress")
        if found is not None:
            return found

        tmp = entry / "tmp"
        if _present(tmp):
            return _status(True, f"in-flight runner temporary directory present: {tmp}", tmp)

        try:
            metadata_files = sorted(
                (item for item in entry.iterdir()
                 if item.name.startswith("run_meta") and item.name.endswith(".json")),
                key=lambda item: item.name,
            )
        except OSError as exc:
            return _status(True, f"run directory could not be inspected: {exc}", entry)
        for meta_path in metadata_files:
            found = _pilot_incomplete(meta_path, entry)
            if found is not None:
                return found

    return _status(False, "no conventional runner indicators found")


runner_status = detect_active_runner
