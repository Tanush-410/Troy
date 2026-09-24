"""Resumable runner: paired seeds, merging, resume, and no cross-episode interference."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agents.providers.base import ModelTurn, Usage
from experiments import pilot
from experiments.runner import RunPaths
from scenarios.generator import ScenarioConfig


class EndTurnSession:
    def add_user(self, text): pass
    def step(self): return ModelTurn(text="done", tool_calls=[], stop_reason="end_turn", usage=Usage(10, 2))
    def add_tool_results(self, results): pass


class FakeProvider:
    def __init__(self, cfg): self.config = cfg
    def new_session(self, system, tools): return EndTurnSession()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(pilot, "make_provider", FakeProvider)


def test_plan_is_complete_and_paired():
    p = pilot.plan("m", 3)
    assert len(p) == 3 * 4 * 2 * 3 == len({x[4] for x in p})
    seeds = {}
    for role, drift, control, k, _ in p:
        seeds.setdefault((role, drift, k), set()).add(pilot.episode_seed(role, drift, k))
    assert all(len(s) == 1 for s in seeds.values())  # C1 and C2 share the seed
    assert len({pilot.episode_seed(r, d, k) for r, d, _, k, _ in p}) == 3 * 4 * 3


def test_parallelism_follows_tpm():
    assert pilot.parallel_for_tpm(8_000) == 1
    assert pilot.parallel_for_tpm(250_000) == 1
    assert pilot.parallel_for_tpm(1_000_000) == 5


def run_and_merge(root, run_id, p, merger):
    role, drift, control, k, eid = p
    pilot.run_one(pilot.PRESETS["qwen3"], run_id, role, drift, control, k, eid, root, {"frozen": True},
                  ScenarioConfig())
    merger.merge(RunPaths(logs=root / "logs" / run_id / "tmp" / eid / "logs",
                          transcripts=root / "logs" / run_id / "tmp" / eid / "transcripts"))


def test_parallel_episodes_merge_without_interference(tmp_path, fake):
    paths = RunPaths.for_run(tmp_path, "r")
    paths.logs.mkdir(parents=True)
    merger = pilot.Merger(paths)
    todo = pilot.plan("m", 1)[:8]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p: run_and_merge(tmp_path, "r", p, merger), todo))
    assert pilot.completed_ids(paths) == {p[4] for p in todo}
    lines = paths.episodes.read_text().splitlines()
    assert len(lines) == 8 and all(json.loads(x)["provenance"] == {"frozen": True} for x in lines)
    assert not any((paths.logs / "tmp").iterdir())  # every per-episode tmp dir removed
    assert len(list(paths.transcripts.glob("*.jsonl"))) == 8


def test_resume_skips_finished_and_discards_partial(tmp_path, fake):
    paths = RunPaths.for_run(tmp_path, "r")
    paths.logs.mkdir(parents=True)
    merger = pilot.Merger(paths)
    todo = pilot.plan("m", 1)[:3]
    run_and_merge(tmp_path, "r", todo[0], merger)
    # a crashed episode leaves a tmp dir with events but no merged record
    partial = paths.logs / "tmp" / todo[1][4] / "logs"
    partial.mkdir(parents=True)
    (partial / "events.jsonl").write_text('{"partial": true}\n')
    remaining = [p for p in todo if p[4] not in pilot.completed_ids(paths)]
    assert [p[4] for p in remaining] == [todo[1][4], todo[2][4]]
    merged = paths.events.read_text() if paths.events.exists() else ""
    assert "partial" not in merged  # partial work never reached the run's logs
