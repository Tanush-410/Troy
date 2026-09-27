import json
import shutil

import pytest

from gui import demo
from gui.demo import DemoManager, DemoPathError, DemoPaths, DemoRunnerActiveError, DemoSpec
from gui.runner_status import detect_active_runner
from scenarios.generator import ScenarioConfig


def test_demo_paths_are_confined(tmp_path):
    paths = DemoPaths(tmp_path, "safe-run")
    base = (tmp_path / "logs" / "demo").resolve()
    assert paths.logs.resolve().is_relative_to(base)
    assert paths.transcripts.resolve().is_relative_to(base)
    with pytest.raises(DemoPathError):
        DemoPaths(tmp_path, "../outside")
    with pytest.raises(DemoPathError):
        DemoPaths(tmp_path, "nested/run")


def test_demo_spec_accepts_every_demo_cell():
    for role in ("support", "listing", "price_intel"):
        for drift in ("D0", "D1", "D2", "D3"):
            for control in ("C1", "C2"):
                spec = DemoSpec(role=role, drift=drift, control=control, scenario=ScenarioConfig())
                assert spec.role == role
                assert spec.drift == drift
                assert spec.control == control


def test_runner_detector_finds_explicit_lock_and_progress(tmp_path):
    lock = tmp_path / ".runner.lock"
    lock.write_text("busy", encoding="utf-8")
    status = detect_active_runner(tmp_path)
    assert status.active
    assert status.source == lock
    assert "lock" in (status.reason or "")
    lock.unlink()

    progress = tmp_path / "runner_progress.json"
    progress.write_text("{}", encoding="utf-8")
    status = detect_active_runner(tmp_path)
    assert status.active
    assert status.source == progress
    progress.unlink()

    (tmp_path / "uv.lock").write_text("not a runner lock", encoding="utf-8")
    assert not detect_active_runner(tmp_path).active


def test_runner_detector_finds_tmp_and_incomplete_pilot_metadata(tmp_path):
    run = tmp_path / "logs" / "pilot"
    (run / "tmp").mkdir(parents=True)
    status = detect_active_runner(tmp_path)
    assert status.active
    assert status.source == run / "tmp"

    shutil.rmtree(run / "tmp")
    (run / "run_meta_20260101T000000.json").write_text(
        json.dumps({"episodes_per_cell": 1}), encoding="utf-8"
    )
    (run / "episodes.jsonl").write_text("", encoding="utf-8")
    status = detect_active_runner(tmp_path)
    assert status.active
    assert "24" in (status.reason or "")

    (run / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_id": str(i)}) + "\n" for i in range(24)), encoding="utf-8"
    )
    assert not detect_active_runner(tmp_path).active


def test_runner_detector_ignores_demo_outputs(tmp_path):
    demo_run = tmp_path / "logs" / "demo" / "job"
    (demo_run / "tmp").mkdir(parents=True)
    (demo_run / "run_meta_20260101T000000.json").write_text(
        json.dumps({"episodes_per_cell": 1}), encoding="utf-8"
    )
    (demo_run / ".runner.lock").write_text("demo-only", encoding="utf-8")
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "not-read.jsonl").write_text("secret", encoding="utf-8")
    assert not detect_active_runner(tmp_path).active


def test_scripted_demo_runs_and_confines_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "frozen_provenance", lambda: {"any_dirty": False})
    spec = DemoSpec(mode="scripted", role="support", drift="D0", control="C1", seed=17, run_id="scripted")
    manager = DemoManager(tmp_path)
    job = manager.start(spec)
    job.wait(20)
    assert job.status == "completed", job.error
    assert job.result is not None
    assert job.actual_cost_usd is None
    assert job.paths.events.exists()
    assert list(job.paths.transcripts.glob("*.jsonl"))
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    assert all(path.relative_to(tmp_path).parts[:2] == ("logs", "demo") for path in files)

    with job.paths.events.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')
    assert demo.read_live_events(job.paths.events)


def test_scripted_demo_is_allowed_while_runner_lock_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "frozen_provenance", lambda: {"any_dirty": False})
    (tmp_path / ".runner.lock").write_text("busy", encoding="utf-8")
    job = DemoManager(tmp_path).start(DemoSpec(run_id="allowed"))
    job.wait(20)
    assert job.status == "completed", job.error


def test_real_model_is_refused_before_provider_construction(tmp_path, monkeypatch):
    (tmp_path / ".runner.lock").write_text("busy", encoding="utf-8")
    called = []

    def fake_provider(config):
        called.append(config)
        raise AssertionError("provider must not be constructed while a runner is active")

    monkeypatch.setattr(demo, "make_provider", fake_provider)
    spec = DemoSpec(mode="qwen3", run_id="real-model")
    with pytest.raises(DemoRunnerActiveError) as error:
        demo.start_demo(spec, tmp_path)
    assert "lock" in str(error.value)
    assert not called


def test_groq_requires_cost_estimate_and_confirmation(tmp_path):
    job = DemoManager(tmp_path).start(DemoSpec(mode="groq", run_id="groq"))
    assert job.status == "error"
    assert job.reason is None or "cost" in job.error.lower()
    assert not (tmp_path / "logs" / "demo" / "groq" / "events.jsonl").exists()
