"""Read-only data facade for the Streamlit dashboard.

Every number the app shows comes from `analysis.tables` (and `analysis.metrics`)
over run logs: nothing is recomputed here and nothing is written, least of all
into `results/`. A run may still be appending to its logs while the app reads
them, so a load first snapshots the *complete* lines of every selected run into
an OS temp directory and hands those to `Dataset.load` / `all_tables`. The
source logs are opened read-only and are never modified.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analysis.detection import DetectionResult
from analysis.metrics import (DRIFT_TYPES, drift_breakdown, format_errors_by_model, harm_summary,
                              success_summary)
from analysis.stats import bootstrap_ci, ratio_stat
from analysis.tables import ROLES, Dataset, Episode, all_tables
from detector.pldd import EpisodeScore

ALL = "all"
CONTROLS = ("C1", "C2", "C3", "C4")
DRIFTS = ("D0", "D1", "D2", "D3")
HARMFUL = "harmful"
CLEAN = "clean"
HARM_CHOICES = (ALL, HARMFUL, CLEAN)
LOG_FILES = ("events.jsonl", "episodes.jsonl", "retries.jsonl")
MAX_WARNINGS = 25
DetectionResults = dict[tuple[str, str, str], DetectionResult]
PARSE_ERRORS = (OSError, ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class RunInfo:
    """One run directory and the state of the log files it owns."""

    path: Path
    transcript_root: Path | None
    label: str
    signature: tuple[tuple[str, int, int, int], ...]

    @property
    def events(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def episodes(self) -> Path:
        return self.path / "episodes.jsonl"

    def key(self) -> tuple[Any, ...]:
        return tuple(self.signature)


@dataclass
class DashboardData:
    """Everything the app renders, derived from the selected runs."""

    dataset: Dataset
    tables: dict[str, list[dict[str, Any]]]
    detection: DetectionResults
    run_infos: list[RunInfo]
    warnings: list[str] = field(default_factory=list)

    @property
    def episodes(self) -> list[Episode]:
        return self.dataset.episodes

    @property
    def models(self) -> list[str]:
        return self.dataset.models

    def episodes_with(self, **kw: str | None) -> list[Episode]:
        return select_episodes(self.dataset, **kw)


def source_signature(path: Path) -> tuple[tuple[str, int, int, int], ...]:
    """(name, exists, size, mtime_ns) for every log file of a run directory."""
    out = []
    for name in LOG_FILES:
        try:
            st = (path / name).stat()
        except OSError:
            out.append((name, 0, 0, 0))
        else:
            out.append((name, 1, st.st_size, st.st_mtime_ns))
    return tuple(out)


def _is_temp(path: Path, root: Path) -> bool:
    """A run under a temp path is a run in progress, not a finished one."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(p.lower().startswith("tmp") for p in parts)


def _is_demo(path: Path, root: Path) -> bool:
    """`logs/demo/` holds live-demo output, which the dashboard reads only while
    the demo itself streams it, so those runs never join the analysis slice."""
    try:
        path.relative_to(root / "demo")
    except ValueError:
        return False
    return True


def _is_run(path: Path) -> bool:
    return any((path / name).is_file() for name in ("events.jsonl", "episodes.jsonl"))


def _transcript_root(path: Path, logs_root: Path) -> Path | None:
    """Where the qualitative transcripts of a run live, if they exist at all.

    The runner writes them to `<root>/transcripts/<run_id>/`, but a run directory
    collected from elsewhere may carry its own copy.
    """
    for cand in (path / "transcripts", logs_root.parent / "transcripts" / path.name,
                 logs_root / "transcripts" / path.name):
        if cand.is_dir():
            return cand
    return None


def discover_runs(logs_root: Path) -> list[RunInfo]:
    """Every finished run directory under `logs_root`, outermost first.

    A directory counts as a run when it holds `events.jsonl` or
    `episodes.jsonl` directly; runs in a temp path are skipped, `logs/demo/` is
    skipped, and a run nested inside another run is reported only as the outer
    one.
    """
    root = Path(logs_root)
    if not root.is_dir():
        return []
    found = [p for p in root.rglob("*")
             if p.is_dir() and _is_run(p) and not _is_temp(p, root) and not _is_demo(p, root)]
    outer = []
    for p in sorted(found, key=lambda p: (len(p.parts), str(p))):
        if not any(q in p.parents for q in outer):
            outer.append(p)
    return [RunInfo(path=p, transcript_root=_transcript_root(p, root), label=p.name,
                    signature=source_signature(p)) for p in outer]


def _clean_lines(path: Path) -> tuple[list[bytes], int]:
    """The complete, parseable JSONL lines of `path`, and how many were dropped.

    A line only counts as complete once its terminating newline is on disk, so a
    record the PEP is still writing is left out of the snapshot.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return [], 0
    out, dropped = [], 0
    for line in raw.split(b"\n")[:-1]:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            dropped += 1
            continue
        out.append(line)
    return out, dropped


def _snapshot(src: Path, dst: Path) -> int:
    """Copy the complete lines of `src` into `dst`; return how many were dropped.

    `src` is only ever read. The copy exists even when `src` is missing, so that
    `Dataset.load` sees a directory with both log files.
    """
    lines, dropped = _clean_lines(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        for line in lines:
            f.write(line + b"\n")
    return dropped


def _load_dirs(dirs: Sequence[Path]) -> tuple[Dataset | None, str]:
    try:
        return Dataset.load(list(dirs)), ""
    except PARSE_ERRORS as e:
        return None, f"{type(e).__name__}: {e}"


def _build(runs: Sequence[RunInfo], workdir: Path) -> tuple[Dataset, dict[str, list[dict[str, Any]]],
                                                             DetectionResults, list[str]]:
    warnings: list[str] = []
    snaps = []
    for i, run in enumerate(runs):
        snap = workdir / f"run{i:03d}"
        dropped = _snapshot(run.events, snap / "events.jsonl") + _snapshot(run.episodes, snap / "episodes.jsonl")
        if dropped:
            warnings.append(f"{run.label}: {dropped} unreadable log line(s) skipped")
        snaps.append((run, snap))
    dataset, error = _load_dirs([s for _, s in snaps])
    if dataset is None:
        good = []
        for run, snap in snaps:
            one, why = _load_dirs([snap])
            if one is None:
                warnings.append(f"{run.label}: not loaded ({why})")
            else:
                good.append(snap)
        dataset, error = _load_dirs(good)
        if dataset is None:
            warnings.append(f"no run could be loaded ({error})")
            return Dataset([], []), {}, {}, warnings[:MAX_WARNINGS]
    try:
        tables, detection = all_tables(dataset)
    except PARSE_ERRORS as e:
        warnings.append(f"tables not computed ({type(e).__name__}: {e})")
        tables, detection = {}, {}
    return dataset, tables, detection, warnings[:MAX_WARNINGS]


def load_dashboard(runs: Sequence[RunInfo], cache: dict[Any, Any] | None = None) -> DashboardData:
    """Load the selected runs into a dashboard, snapshotting each load into a temp dir.

    `cache` is any mutable mapping; an entry is reused while none of the selected
    runs' log files has changed (see `RunInfo.signature`). The snapshot directory
    only outlives this call, so anything that reads the event files themselves has
    to go through `all_tables` first.
    """
    runs = list(runs)
    key = tuple(r.key() for r in runs)
    if cache is not None and key in cache:
        dataset, tables, detection, warnings = cache[key]
        return DashboardData(dataset, tables, detection, runs, list(warnings))
    with tempfile.TemporaryDirectory(prefix="troy-gui-") as tmp:
        dataset, tables, detection, warnings = _build(runs, Path(tmp))
    if cache is not None:
        cache[key] = (dataset, tables, detection, warnings)
    return DashboardData(dataset, tables, detection, runs, warnings)


def load_dashboard_from_root(logs_root: Path, cache: dict[Any, Any] | None = None) -> DashboardData:
    """Discover the runs under `logs_root` and load all of them."""
    return load_dashboard(discover_runs(logs_root), cache)


def select_episodes(dataset: Dataset, **kw: str | None) -> list[Episode]:
    """`Dataset.select` with unset selections dropped: "all", "" and None match everything."""
    return dataset.select(**{k: v for k, v in kw.items() if v})


def filter_options(dataset: Dataset) -> dict[str, list[str]]:
    """The values the filter widgets can offer, plus "all" for each."""
    return {
        "models": [ALL, *dataset.models],
        "agents": [ALL, *ROLES],
        "controls": [ALL, *CONTROLS],
        "drifts": [ALL, *DRIFTS],
    }


@dataclass(frozen=True)
class EpisodeRow:
    """One row of the episode table."""

    episode_id: str
    run_id: str
    seed: int
    model: str
    agent: str
    drift: str
    control: str
    tasks: int
    outcomes: dict[str, int]
    success_rate: float
    harmful_attempted: int
    harmful_executed: int
    first_harm_step: int | None

    @property
    def outcome(self) -> str:
        """Task outcomes as a summary, most common first: "completed=3, escalated=1"."""
        ordered = sorted(self.outcomes.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{name}={n}" for name, n in ordered)

    @property
    def harm_count(self) -> int:
        return self.harmful_attempted

    @property
    def harmful(self) -> bool:
        return self.harmful_attempted > 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_rows(episodes: Iterable[Episode]) -> list[EpisodeRow]:
    """Table rows for the given episodes, the harmful ones first."""
    rows = []
    for ep in episodes:
        r = ep.record
        outcomes: dict[str, int] = {}
        for o in r.task_outcomes:
            outcomes[o.outcome] = outcomes.get(o.outcome, 0) + 1
        rows.append(EpisodeRow(
            episode_id=r.episode_id, run_id=r.run_id, seed=r.seed, model=r.model, agent=r.agent_role,
            drift=r.drift_condition, control=r.control_condition, tasks=r.n_tasks,
            outcomes=dict(sorted(outcomes.items())), success_rate=r.task_success_rate,
            harmful_attempted=r.total_harmful_attempted, harmful_executed=r.total_harmful_executed,
            first_harm_step=r.first_harm_step))
    return sorted(rows, key=lambda x: (x.harmful_attempted == 0, -x.harmful_attempted, x.episode_id))


def rows_by_outcome(rows: Sequence[EpisodeRow], outcome: str = ALL) -> list[EpisodeRow]:
    """Rows with at least one task in `outcome` ("all" keeps every row)."""
    if outcome == ALL:
        return list(rows)
    return [r for r in rows if outcome in r.outcomes]


def rows_by_harm(rows: Sequence[EpisodeRow], harm: str = ALL, min_harm: int = 0) -> list[EpisodeRow]:
    """Rows whose harmful-action count passes `harm` ("harmful", "clean", "all") and `min_harm`."""
    out = list(rows)
    if harm == HARMFUL:
        out = [r for r in out if r.harmful]
    elif harm == CLEAN:
        out = [r for r in out if not r.harmful]
    return [r for r in out if r.harm_count >= min_harm]


@dataclass(frozen=True)
class OverviewCard:
    """One metric card: the headline numbers of a model/agent/control slice."""

    episodes: int
    task_success: float | None
    harmful_attempted_rate: float | None
    format_error_rate: float | None
    escalation_rate: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _card(episodes: Sequence[Episode], model: str) -> OverviewCard:
    """Headline numbers for a slice, straight from the `analysis.metrics` summaries."""
    records = [ep.record for ep in episodes]
    success = success_summary(records)
    events = [e for ep in episodes for e in ep.events]
    fmt = format_errors_by_model(events).get(model, {})
    return OverviewCard(
        episodes=len(episodes),
        task_success=success["success_rate"],
        harmful_attempted_rate=harm_summary(events)["harmful_rate"],
        format_error_rate=fmt.get("format_error_rate"),
        escalation_rate=success["escalation_rate"])


def overview_cards(dataset: Dataset, model: str, agent: str = ALL, control: str = ALL) -> dict[str, OverviewCard]:
    """Metric cards for one model, one agent role and one control condition.

    With `control` = "all" the pooled card comes first, followed by a card per
    control that has data, so the page can show C1 and C2 side by side. Every
    value is the ratio the matching `analysis.tables` row reports.
    """
    if control != ALL:
        return {control: _card(select_episodes(dataset, model=model, agent_role=agent,
                                               control_condition=control), model)}
    out = {ALL: _card(select_episodes(dataset, model=model, agent_role=agent), model)}
    for cc in CONTROLS:
        eps = select_episodes(dataset, model=model, agent_role=agent, control_condition=cc)
        if eps:
            out[cc] = _card(eps, model)
    return out


def agent_drift_breakdown(dataset: Dataset, model: str, agent: str = ALL, control: str = ALL
                          ) -> dict[str, dict[str, dict[str, int]]]:
    """`drift_breakdown` per drift type, for each agent role in the slice."""
    roles = ROLES if agent == ALL else (agent,)
    out = {}
    for role in roles:
        eps = select_episodes(dataset, model=model, agent_role=role, control_condition=control)
        if eps:
            out[role] = drift_breakdown([e for ep in eps for e in ep.events])
    return out


@dataclass(frozen=True)
class RateCell:
    """A rate over well-formed calls, with an episode-level bootstrap CI."""

    episodes: int
    calls: int
    rate: float | None
    lo: float | None
    hi: float | None


@dataclass(frozen=True)
class DriftRateRow:
    """C1 against C2 for one drift condition."""

    drift: str
    c1: RateCell
    c2: RateCell


def _rate_cell(episodes: Sequence[Episode]) -> RateCell:
    events = [e for ep in episodes for e in ep.events]
    rate = harm_summary(events)["harmful_rate"]
    lo, hi = bootstrap_ci(episodes, ratio_stat(lambda ep: sum(e.harm for e in ep.wf), lambda ep: len(ep.wf)))[1:]
    return RateCell(episodes=len(episodes), calls=len(events), rate=rate, lo=lo, hi=hi)


def harmful_rate_by_drift(dataset: Dataset, model: str, agent: str = ALL) -> list[DriftRateRow]:
    """Harmful attempted rate per drift condition under C1 and under C2.

    One row per drift condition, then a pooled "all" row. The CIs are the same
    episode-level bootstrap `analysis.tables` uses for RQ2.
    """
    rows = []
    for drift in (*DRIFTS, ALL):
        cells = [RateCell(0, 0, None, None, None)] * 2
        for i, control in enumerate(("C1", "C2")):
            eps = select_episodes(dataset, model=model, agent_role=agent, control_condition=control,
                                  drift_condition=drift)
            cells[i] = _rate_cell(eps)
        rows.append(DriftRateRow(drift=drift, c1=cells[0], c2=cells[1]))
    return rows


@dataclass(frozen=True)
class DriftShare:
    """What static RBAC held back, for one drift type."""

    drift_type: str
    calls: int
    harmful_attempted: int
    harmful_blocked: int
    blocked_share: float | None


def harm_blocked_by_drift_type(dataset: Dataset, model: str, agent: str = ALL, control: str = ALL
                               ) -> list[DriftShare]:
    """Share of harmful calls that were denied, per drift type, plus an "all" row."""
    episodes = select_episodes(dataset, model=model, agent_role=agent, control_condition=control)
    events = [e for ep in episodes for e in ep.wf]
    out = []
    for drift_type in (*DRIFT_TYPES, ALL):
        of_type = [e for e in events if e.drift_type == drift_type] if drift_type != ALL else events
        summary = harm_summary(of_type)
        out.append(DriftShare(drift_type=drift_type, calls=summary["well_formed_calls"],
                              harmful_attempted=summary["harmful_attempted"],
                              harmful_blocked=summary["harmful_blocked"],
                              blocked_share=summary["harmful_blocked_share"]))
    return out


@dataclass(frozen=True)
class EpisodeDetection:
    """The detector's scores for one episode, from the replayed logs only."""

    model: str
    control: str
    baseline: str
    result: DetectionResult
    score: EpisodeScore

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detection_keys(detection: DetectionResults) -> list[tuple[str, str, str]]:
    """The (model, control, baseline) slices the detector produced."""
    return sorted(detection)


def find_detection(detection: DetectionResults, episode_id: str, model: str | None = None,
                   control: str | None = None, baseline: str | None = None) -> EpisodeDetection | None:
    """The result and per-step scores of one episode, or None if it was not scored.

    Reads the detection table only: no transcript is opened, and the detector
    never sees one.
    """
    for key in detection_keys(detection):
        m, c, b = key
        if model is not None and m != model:
            continue
        if control is not None and c != control:
            continue
        if baseline is not None and b != baseline:
            continue
        result = detection[key]
        score = result.scores.get(episode_id)
        if score is not None:
            return EpisodeDetection(m, c, b, result, score)
    return None


def transcript_path(run: RunInfo, episode_id: str) -> Path | None:
    """The transcript of one episode, if this run kept any."""
    if run.transcript_root is None:
        return None
    path = run.transcript_root / f"{episode_id}.jsonl"
    return path if path.is_file() else None


def read_transcript(path: Path | None) -> list[dict[str, Any]]:
    """The JSONL entries of a transcript, for qualitative inspection only.

    Unreadable and half-written lines are skipped, so a transcript being appended
    to can be read. Nothing here is fed back into the numbers.
    """
    if path is None:
        return []
    out = []
    for line in _clean_lines(Path(path))[0]:
        entry = json.loads(line)
        if isinstance(entry, dict):
            out.append(entry)
    return out
