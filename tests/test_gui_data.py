"""The dashboard data facade: run discovery, snapshot loading, filters, and the
overview numbers, which have to agree with analysis.run_all exactly."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import stats
from analysis.metrics import DRIFT_TYPES
from analysis.run_all import run
from analysis.tables import by_key
from experiments.scripted_dataset import generate
from gui import data
from gui.data import CLEAN, HARMFUL, DashboardData, OverviewCard
from gui.filters import EpisodeFilters, GlobalFilters

MODEL = "scripted"
ROLES = ("support", "listing", "price_intel", "all")


def read(out: Path, name: str) -> list[dict[str, str]]:
    with (out / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def cell(row: dict[str, str], key: str) -> float:
    """A CSV number; run_all writes an empty cell where there is no data."""
    assert row[key] != "", f"{key} has no data in the table"
    return float(row[key])


def snapshot_of(path: Path) -> dict[str, tuple[int, bytes]]:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(path.iterdir())}


def write_log(path: Path, records: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    path.joinpath("episodes.jsonl").write_text("")


def subset_run(src: Path, dst: Path, n: int = 2) -> Path:
    """Copy the first `n` episodes of a run into a new run directory."""
    episodes = [json.loads(line) for line in (src / "episodes.jsonl").read_text().splitlines() if line.strip()]
    keep = {e["episode_id"] for e in episodes[:n]}
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "events.jsonl"):
        lines = [line for line in (src / name).read_text().splitlines()
                 if line.strip() and json.loads(line)["episode_id"] in keep]
        (dst / name).write_text("".join(line + "\n" for line in lines))
    return dst


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    saved, stats.N_BOOT = stats.N_BOOT, 200  # keep the test fast; the real pipeline uses 2000
    try:
        root = tmp_path_factory.mktemp("gui_data")
        paths, records = generate(root, "ds", 6)
        out = root / "results"
        index = run([paths.logs], out)
        yield SimpleNamespace(root=root, paths=paths, records=records, out=out, index=index)
    finally:
        stats.N_BOOT = saved


@pytest.fixture(scope="module")
def dashboard(scripted):
    return data.load_dashboard_from_root(scripted.root)


# ------------------------------------------------------------------ discovery


def test_discover_runs_finds_every_run_and_skips_temp_paths(tmp_path):
    root = tmp_path / "logs"
    write_log(root / "a", [{"action": "read_ticket"}])
    (root / "b").mkdir()
    (root / "b" / "episodes.jsonl").write_text("")
    (root / "not_a_run").mkdir()
    (root / "not_a_run" / "notes.txt").write_text("hello")
    write_log(root / "tmp_in_progress", [{"action": "read_ticket"}])
    write_log(root / "demo" / "gui-scripted-1", [{"action": "read_ticket"}])
    write_log(root / "nested" / "outer", [{"action": "read_ticket"}])
    write_log(root / "nested" / "outer" / "inner", [{"action": "read_ticket"}])
    (tmp_path / "transcripts" / "a").mkdir(parents=True)

    runs = data.discover_runs(root)
    assert [r.path for r in runs] == [root / "a", root / "b", root / "nested" / "outer"]
    assert [r.label for r in runs] == ["a", "b", "outer"]
    assert runs[0].transcript_root == tmp_path / "transcripts" / "a"
    assert runs[1].transcript_root is None
    assert data.discover_runs(tmp_path / "nope") == []


def test_discover_runs_never_returns_live_demo_output(tmp_path):
    root = tmp_path / "logs"
    write_log(root / "a", [{"action": "read_ticket"}])
    write_log(root / "demo" / "gui-scripted-1", [{"action": "read_ticket"}])
    write_log(root / "demo" / "gui-groq-1" / "transcripts", [{"action": "read_ticket"}])

    assert [r.path for r in data.discover_runs(root)] == [root / "a"]


def test_signature_follows_the_log_files(tmp_path):
    root = tmp_path / "logs"
    write_log(root / "a", [{"action": "read_ticket"}])
    found = data.discover_runs(root)[0]
    stat = found.events.stat()
    assert found.signature[0] == ("events.jsonl", 1, stat.st_size, stat.st_mtime_ns)
    assert [exists for _, exists, _, _ in found.signature] == [1, 1, 0]
    before = found.key()
    found.events.write_text(found.events.read_text() + json.dumps({"action": "reply_customer"}) + "\n")
    assert data.discover_runs(root)[0].key() != before


def test_dashboard_data_is_empty_without_runs():
    loaded = data.load_dashboard([])
    assert isinstance(loaded, DashboardData) and loaded.episodes == [] and loaded.detection == {}
    assert loaded.tables and all(rows == [] for rows in loaded.tables.values())
    assert loaded.warnings == [] and loaded.models == []
    assert data.filter_options(loaded.dataset)["models"] == ["all"]
    assert data.episode_rows(loaded.episodes) == []


# ------------------------------------------------------------------- loading


def test_loading_ignores_a_half_written_line_and_leaves_the_source_alone(scripted, tmp_path):
    root = tmp_path / "logs"
    subset_run(scripted.paths.logs, root / "ds")
    for name, tail in (("events.jsonl", '{"run_id": "ds", "episode_id": "trunc'),
                       ("episodes.jsonl", '{"run_id": "ds"')):
        with (root / "ds" / name).open("a") as f:
            f.write(tail)

    before = snapshot_of(root / "ds")
    clean = data.load_dashboard(data.discover_runs(root))
    torn = data.load_dashboard(data.discover_runs(root))

    assert snapshot_of(root / "ds") == before
    assert sorted(p.name for p in (root / "ds").iterdir()) == ["episodes.jsonl", "events.jsonl"]
    assert len(torn.episodes) == len(clean.episodes) == 2
    assert sum(len(ep.events) for ep in torn.episodes) == sum(len(ep.events) for ep in clean.episodes)
    assert torn.warnings == [] and clean.warnings == []
    assert all("troy-gui-" in str(p) and Path(p).is_relative_to(Path(tempfile.gettempdir()))
               for p in torn.dataset.event_paths)


def test_loading_survives_missing_empty_and_malformed_logs(tmp_path):
    root = tmp_path / "logs"
    write_log(root / "empty", [])
    (root / "events_only").mkdir()
    (root / "events_only" / "events.jsonl").write_text("")
    write_log(root / "garbage", [])
    (root / "garbage" / "events.jsonl").write_bytes(b"this is not json\n{]}\n")
    (root / "garbage" / "episodes.jsonl").write_bytes(b"\n")
    (root / "unreadable").mkdir()
    (root / "unreadable" / "events.jsonl").write_bytes(b'{"run_id": "x"\n')

    runs = data.discover_runs(root)
    assert [r.label for r in runs] == ["empty", "events_only", "garbage", "unreadable"]
    loaded = data.load_dashboard(runs)
    assert loaded.episodes == []
    assert loaded.warnings == ["garbage: 2 unreadable log line(s) skipped",
                               "unreadable: 1 unreadable log line(s) skipped"]


def test_one_broken_run_does_not_take_the_others_down(scripted, tmp_path):
    root = tmp_path / "logs"
    subset_run(scripted.paths.logs, root / "good")
    subset_run(scripted.paths.logs, root / "broken").joinpath("episodes.jsonl").write_text(
        json.dumps({"run_id": "ds"}) + "\n")

    loaded = data.load_dashboard(data.discover_runs(root))
    assert [ep.record.episode_id for ep in loaded.episodes] == \
        [json.loads(line)["episode_id"] for line in
         (root / "good" / "episodes.jsonl").read_text().splitlines()]
    assert loaded.tables and len(loaded.warnings) == 1 and "broken" in loaded.warnings[0]


def test_a_loaded_dashboard_is_cached_until_a_log_changes(scripted, tmp_path):
    root = tmp_path / "logs"
    subset_run(scripted.paths.logs, root / "ds")
    cache: dict = {}

    first = data.load_dashboard(data.discover_runs(root), cache)
    assert data.load_dashboard(data.discover_runs(root), cache).dataset is first.dataset
    assert len(cache) == 1

    with (root / "ds" / "episodes.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": "ds"}) + "\n")
    assert data.load_dashboard(data.discover_runs(root), cache).dataset is not first.dataset
    assert len(cache) == 2


def test_the_dashboard_never_touches_the_pipeline_output(scripted):
    out, root = scripted.out, scripted.root
    before, listing = snapshot_of(out), sorted(p.name for p in root.iterdir())
    loaded = data.load_dashboard_from_root(root)
    assert loaded.episodes
    assert snapshot_of(out) == before
    assert sorted(p.name for p in root.iterdir()) == listing


# -------------------------------------------------------------------- metrics


def test_overview_cards_match_the_run_all_tables(scripted, dashboard):
    rq1, rq2 = read(scripted.out, "rq1_drift_types"), read(scripted.out, "rq2_controls")
    formats, escalation = read(scripted.out, "format_errors"), read(scripted.out, "context_overflow_escalation")

    for agent in ROLES:
        cards = data.overview_cards(dashboard.dataset, MODEL, agent=agent)
        assert list(cards) == ["all", "C1", "C2"]
        c1, c2 = (by_key(rq1, agent=agent, control=c) for c in ("C1", "C2"))
        rates = by_key(rq2, agent=agent)
        assert cards["C1"].episodes == int(c1["episodes"])
        assert cards["C2"].episodes == int(c2["episodes"])
        assert cards["all"].episodes == int(c1["episodes"]) + int(c2["episodes"])
        for control in ("c1", "c2"):
            assert cards[control.upper()].task_success == cell(rates, f"task_success_rate_{control}")
            assert cards[control.upper()].harmful_attempted_rate == cell(rates, f"harm_attempted_rate_{control}")
        assert cards["all"].format_error_rate == cell(by_key(formats, agent=agent), "format_error_rate")
        assert cards["all"].escalation_rate == cell(by_key(escalation, agent=agent, drift="all"),
                                                    "escalation_rate")
        num = sum(int(c["harmful_attempted"]) for c in (c1, c2))
        assert cards["all"].harmful_attempted_rate == num / sum(int(c["calls"]) for c in (c1, c2))
        assert data.overview_cards(dashboard.dataset, MODEL, agent=agent, control="C1") == {"C1": cards["C1"]}


def test_overview_cards_survive_an_empty_slice(dashboard):
    empty = OverviewCard(0, None, None, None, None)
    assert data.overview_cards(dashboard.dataset, "no-such-model") == {"all": empty}
    assert data.overview_cards(dashboard.dataset, MODEL, agent="support", control="C3") == {"C3": empty}


def test_agent_drift_breakdown_counts_the_log(dashboard, scripted):
    events = [json.loads(line) for line in scripted.paths.events.read_text().splitlines() if line.strip()]
    breakdown = data.agent_drift_breakdown(dashboard.dataset, MODEL, control="C1")
    assert sorted(breakdown) == ["listing", "price_intel", "support"]
    logged = [e for e in events if e["control_condition"] == "C1" and not e["format_error"]]
    for role, rows in breakdown.items():
        assert set(rows) == set(DRIFT_TYPES)
        assert all(set(t) == {"calls", "harmful", "harmful_executed", "denied"} for t in rows.values())
        assert all(0 <= t["denied"] <= t["calls"] and t["harmful"] <= t["calls"] for t in rows.values())
        assert sum(t["calls"] for t in rows.values()) == sum(1 for e in logged if e["agent_role"] == role)
    rq1 = by_key(read(scripted.out, "rq1_drift_types"), agent="all", control="C1")
    assert sum(t["calls"] for role in breakdown.values() for t in role.values()) == int(rq1["calls"])
    assert int(rq1["type_I"]) == sum(breakdown[r]["I"]["calls"] for r in breakdown)
    assert int(rq1["type_II_harmful"]) == sum(breakdown[r]["II"]["harmful"] for r in breakdown)


def test_harmful_rate_by_drift_matches_rq2(dashboard, scripted):
    rows = data.harmful_rate_by_drift(dashboard.dataset, MODEL)
    assert [r.drift for r in rows] == ["D0", "D1", "D2", "D3", "all"]
    rates = by_key(read(scripted.out, "rq2_controls"), agent="all")
    pooled = rows[-1]
    assert pooled.c1.rate == cell(rates, "harm_attempted_rate_c1")
    assert pooled.c1.lo == cell(rates, "harm_attempted_rate_c1_lo")
    assert pooled.c1.hi == cell(rates, "harm_attempted_rate_c1_hi")
    assert pooled.c2.rate == cell(rates, "harm_attempted_rate_c2")
    for row in rows:
        for side in (row.c1, row.c2):
            assert side.episodes in (0, 6, 18, 72)
            assert side.calls >= side.episodes
            assert side.rate is None or side.lo - 1e-12 <= side.rate <= side.hi + 1e-12
    assert rows[0].c1.episodes == rows[0].c2.episodes == 18
    assert sum(r.c1.episodes for r in rows[:-1]) == 72
    assert data.harmful_rate_by_drift(dashboard.dataset, MODEL, agent="support")[0].c1.episodes == 6


def test_harm_blocked_by_drift_type_matches_rq1(dashboard, scripted):
    rq1 = read(scripted.out, "rq1_drift_types")
    for control in ("C1", "C2"):
        shares = data.harm_blocked_by_drift_type(dashboard.dataset, MODEL, control=control)
        assert [s.drift_type for s in shares] == [*DRIFT_TYPES, "all"]
        row = by_key(rq1, agent="all", control=control)
        pooled = shares[-1]
        assert pooled.blocked_share == cell(row, "harmful_blocked_share")
        assert pooled.calls == int(row["calls"])
        assert pooled.harmful_attempted == int(row["harmful_attempted"])
        by_type = {s.drift_type: s for s in shares}
        assert sum(s.calls for t, s in by_type.items() if t != "all") == pooled.calls
        for drift_type, share in by_type.items():
            if drift_type == "all":
                continue
            assert share.blocked_share == (share.harmful_blocked / share.harmful_attempted
                                           if share.harmful_attempted else None)
        breakdown = data.agent_drift_breakdown(dashboard.dataset, MODEL, control=control)["support"]
        assert by_type["I"].calls == breakdown["I"]["calls"]
        assert by_type["I"].harmful_blocked <= breakdown["I"]["denied"]
    assert data.harm_blocked_by_drift_type(dashboard.dataset, "no-such-model")[0].blocked_share is None


# ------------------------------------------------------------------ detection


def test_find_detection_returns_the_scores_of_one_episode(dashboard, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the detection view must not read transcripts")

    monkeypatch.setattr(data, "read_transcript", forbidden)
    monkeypatch.setattr(data, "transcript_path", forbidden)
    episode = next(ep for ep in dashboard.episodes if ep.record.total_harmful_attempted > 0)
    found = data.find_detection(dashboard.detection, episode.record.episode_id)
    assert found is not None and found.score.episode_id == episode.record.episode_id
    assert found.model == MODEL and found.control in ("C1", "C2") and found.baseline in ("all_d0", "harm_free_d0")
    assert found.score.steps and [s.step for s in found.score.steps] == sorted(s.step for s in found.score.steps)
    assert data.find_detection(dashboard.detection, episode.record.episode_id,
                               control="C1", baseline="all_d0").control == "C1"
    assert data.find_detection(dashboard.detection, "no-such-episode") is None
    assert data.find_detection(dashboard.detection, episode.record.episode_id, model="other") is None
    assert data.find_detection({}, episode.record.episode_id) is None
    assert data.detection_keys(dashboard.detection) == sorted(dashboard.detection)


def test_read_transcript_skips_half_written_lines(scripted, tmp_path):
    run_info = data.discover_runs(scripted.paths.logs.parent)[0]
    episode = json.loads(scripted.paths.episodes.read_text().splitlines()[0])["episode_id"]
    path = data.transcript_path(run_info, episode)
    assert path is not None and path.is_file()
    entries = data.read_transcript(path)
    assert entries and all(isinstance(e, dict) for e in entries)

    torn = tmp_path / "t.jsonl"
    torn.write_text(f"{json.dumps(entries[0])}\n{json.dumps(entries[-1])}\n" + '{"role": "us')
    assert data.read_transcript(torn) == [entries[0], entries[-1]]
    assert data.read_transcript(tmp_path / "missing.jsonl") == []
    assert data.read_transcript(None) == []
    assert data.transcript_path(run_info, "no-such-episode") is None


# -------------------------------------------------------------------- filters


def test_global_filters_select_the_dataset(dashboard):
    ds = dashboard.dataset
    assert GlobalFilters().episodes(ds) == ds.episodes
    assert GlobalFilters(control="", model="").episodes(ds) == ds.episodes
    assert GlobalFilters(agent="listing").episodes(ds) == data.select_episodes(ds, agent_role="listing")

    chosen = GlobalFilters(agent="support", control="C1", drift="D2")
    episodes = chosen.episodes(ds)
    assert len(episodes) == 6 and all(chosen.matches(ep) for ep in episodes)
    assert chosen.active() == {"agent_role": "support", "control_condition": "C1", "drift_condition": "D2"}
    assert all(ep not in episodes for ep in GlobalFilters(model="other").episodes(ds))
    assert GlobalFilters(model="other").episodes(ds) == []
    assert GlobalFilters(run="other").episodes(ds) == []
    assert GlobalFilters(run=ds.episodes[0].record.run_id).episodes(ds)
    assert dashboard.episodes_with(model=MODEL, agent_role="support") == data.select_episodes(
        ds, model=MODEL, agent_role="support")
    assert data.select_episodes(ds, model=None, agent="") == ds.episodes


def test_episode_rows_and_the_row_filters(dashboard):
    rows = data.episode_rows(dashboard.episodes)
    assert len(rows) == len(dashboard.episodes)
    assert all(r.as_dict()["episode_id"] == r.episode_id for r in rows)
    assert all(sum(r.outcomes.values()) == r.tasks and set(r.outcomes) <= {"completed", "escalated", "failed"}
               for r in rows)
    assert all(f"{name}={n}" in r.outcome for r in rows for name, n in r.outcomes.items())
    assert [r.harmful_attempted for r in rows] == sorted((r.harmful_attempted for r in rows), reverse=True)
    assert any(r.harmful for r in rows) and any(not r.harmful for r in rows)

    assert data.rows_by_outcome(rows) == rows
    completed = data.rows_by_outcome(rows, "completed")
    assert completed and all("completed" in r.outcomes for r in completed)
    assert data.rows_by_outcome(rows, "escalated") == [r for r in rows if "escalated" in r.outcomes]
    assert data.rows_by_harm(rows) == rows
    assert data.rows_by_harm(rows, harm=HARMFUL) == [r for r in rows if r.harmful]
    assert data.rows_by_harm(rows, harm=CLEAN) == [r for r in rows if not r.harmful]
    assert data.rows_by_harm(rows, min_harm=2) == [r for r in rows if r.harm_count >= 2]
    assert data.rows_by_harm(rows, harm=CLEAN, min_harm=1) == []


def test_episode_filters_narrow_the_rows_only(dashboard):
    rows = data.episode_rows(dashboard.episodes)
    assert EpisodeFilters().rows(rows) == rows
    harmful = EpisodeFilters(harm=HARMFUL).rows(rows)
    assert harmful and all(r.harmful for r in harmful)
    assert EpisodeFilters(harm=CLEAN).rows(rows) == [r for r in rows if not r.harmful]
    assert EpisodeFilters(min_harm=1).rows(rows) == harmful
    assert len(EpisodeFilters(model=MODEL, control="C1").rows(rows)) == 72
    assert len(EpisodeFilters(agent="listing", drift="D3").rows(rows)) == 12
    assert EpisodeFilters(control="C9").rows(rows) == []
    assert EpisodeFilters(outcome="escalated").rows(rows) == [r for r in rows if "escalated" in r.outcomes]
    assert EpisodeFilters(harm=HARMFUL, min_harm=99).rows(rows) == []

    episode = rows[0]
    assert EpisodeFilters(search=episode.episode_id.lower()).matches(episode)
    assert EpisodeFilters(search=episode.agent).matches(episode)
    assert EpisodeFilters(search=episode.run_id).matches(episode)
    assert not EpisodeFilters(search="zzz").matches(episode)
    assert not EpisodeFilters(search=episode.episode_id, model="other").matches(episode)
    assert EpisodeFilters(harm=CLEAN).matches(episode) is not episode.harmful
    assert EpisodeFilters(control=episode.control, drift=episode.drift, agent=episode.agent).matches(episode)


def test_filter_options_offer_every_choice(dashboard):
    options = data.filter_options(dashboard.dataset)
    assert options["models"] == ["all", MODEL]
    assert options["agents"] == ["all", "support", "listing", "price_intel"]
    assert options["drifts"] == ["all", "D0", "D1", "D2", "D3"]
    assert options["controls"] == ["all", "C1", "C2", "C3", "C4"]
    assert EpisodeFilters(agent="support").active() == {"agent": "support"}
    assert EpisodeFilters().active() == {} and GlobalFilters().active() == {}
