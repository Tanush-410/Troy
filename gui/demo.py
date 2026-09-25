from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, cast

from agents.llm_agent import LLMAgent
from agents.providers.base import ProviderConfig
from agents.providers.factory import make_provider
from agents.scripted import ScriptedAgent
from experiments.config import EpisodeConfig
from experiments.cost import Estimate, actual_cost, estimate
from experiments.provenance import frozen_provenance
from experiments.runner import RunPaths, run_episode
from experiments.smoke import MAX_TURNS, PRESETS, groq_config
from experiments.scripted_dataset import scripts_for
from gui.runner_status import RunnerStatus, detect_active_runner
from policy.log_schema import ControlCondition, DriftCondition, EpisodeRecord
from policy.permissions import Role
from scenarios.generator import ScenarioConfig, build_episode
from simmart.generator import generate_state

PathLike: TypeAlias = str | Path
DemoMode: TypeAlias = Literal["scripted", "qwen3", "groq"]
DemoJobStatus: TypeAlias = Literal["queued", "running", "completed", "error"]
ProviderFactory: TypeAlias = Callable[[ProviderConfig], Any]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class DemoError(RuntimeError):
    pass


class DemoPathError(DemoError, ValueError):
    pass


class DemoConfirmationError(DemoError):
    pass


class DemoRunnerActiveError(DemoError):
    def __init__(self, status: RunnerStatus) -> None:
        self.status = status
        self.reason = status.reason or "an active runner was detected"
        super().__init__(f"refusing real-model demo: {self.reason}")


class DemoStartError(DemoError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_mode(value: str) -> DemoMode:
    mode = value.strip().casefold()
    if mode in {"scripted", "free"}:
        return "scripted"
    if mode in {"qwen3", "qwen3:8b", "qwen3-8b", "qwen3_8b", "local", "ollama"}:
        return "qwen3"
    if mode == "groq":
        return "groq"
    raise ValueError(f"unsupported demo mode: {value}")


def _scenario(value: ScenarioConfig | dict[str, Any] | None) -> ScenarioConfig:
    if value is None:
        return ScenarioConfig()
    if isinstance(value, ScenarioConfig):
        return value
    if isinstance(value, dict):
        return ScenarioConfig.model_validate(value)
    raise TypeError("scenario must be a ScenarioConfig or mapping")


def _default_run_id(mode: DemoMode, role: str, drift: str, control: str, seed: int) -> str:
    return f"demo-{mode}-{role}-{drift}-{control}-{seed}"


@dataclass(frozen=True, slots=True, init=False)
class DemoSpec:
    mode: DemoMode
    role: Role
    drift: DriftCondition
    control: ControlCondition
    seed: int
    max_turns: int
    scenario: ScenarioConfig
    groq_model: str
    run_id: str

    def __init__(
        self,
        mode: str = "scripted",
        role: str = "support",
        drift: str = "D0",
        control: str = "C1",
        seed: int = 1,
        max_turns: int = MAX_TURNS,
        scenario: ScenarioConfig | dict[str, Any] | None = None,
        groq_model: str = "openai/gpt-oss-120b",
        run_id: str | None = None,
        *,
        model: str | None = None,
        model_mode: str | None = None,
        agent: str | None = None,
        agent_role: str | None = None,
        drift_condition: str | None = None,
        control_condition: str | None = None,
        scenario_config: ScenarioConfig | dict[str, Any] | None = None,
        max_turns_per_task: int | None = None,
    ) -> None:
        selected_mode = model_mode if model_mode is not None else mode
        if model is not None:
            selected_mode = model
        selected_role = agent_role if agent_role is not None else agent
        if selected_role is not None:
            role = selected_role
        if drift_condition is not None:
            drift = drift_condition
        if control_condition is not None:
            control = control_condition
        if scenario_config is not None:
            scenario = scenario_config
        if max_turns_per_task is not None:
            max_turns = max_turns_per_task
        normalised_mode = _normalise_mode(selected_mode)
        role = str(role).strip().casefold()
        drift = str(drift).strip().upper()
        control = str(control).strip().upper()
        if role not in {"support", "listing", "price_intel"}:
            raise ValueError(f"unsupported demo agent: {role}")
        if drift not in {"D0", "D1", "D2", "D3"}:
            raise ValueError(f"unsupported drift condition: {drift}")
        if control not in {"C1", "C2"}:
            raise ValueError(f"unsupported control condition: {control}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if not isinstance(groq_model, str) or not groq_model.strip():
            raise ValueError("groq_model must be a non-empty string")
        groq_model = groq_model.strip()
        selected_scenario = _scenario(scenario)
        selected_run_id = run_id or _default_run_id(normalised_mode, role, drift, control, seed)
        if not isinstance(selected_run_id, str) or not _RUN_ID.fullmatch(selected_run_id):
            raise DemoPathError("run_id must be a simple path-safe name")
        object.__setattr__(self, "mode", normalised_mode)
        object.__setattr__(self, "role", cast(Role, role))
        object.__setattr__(self, "drift", cast(DriftCondition, drift))
        object.__setattr__(self, "control", cast(ControlCondition, control))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_turns", max_turns)
        object.__setattr__(self, "scenario", selected_scenario)
        object.__setattr__(self, "groq_model", groq_model)
        object.__setattr__(self, "run_id", selected_run_id)

    @property
    def model(self) -> DemoMode:
        return self.mode

    @property
    def agent(self) -> Role:
        return self.role

    @property
    def agent_role(self) -> Role:
        return self.role

    @property
    def drift_condition(self) -> DriftCondition:
        return self.drift

    @property
    def control_condition(self) -> ControlCondition:
        return self.control

    @property
    def scenario_config(self) -> ScenarioConfig:
        return self.scenario

    @property
    def max_turns_per_task(self) -> int:
        return self.max_turns

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "role": self.role,
            "drift": self.drift,
            "control": self.control,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "scenario": self.scenario.model_dump(mode="json"),
            "groq_model": self.groq_model,
            "run_id": self.run_id,
        }


def _beneath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True, init=False)
class DemoPaths:
    project_root: Path
    run_id: str

    def __init__(
        self,
        project_root: PathLike | DemoSpec | None = None,
        run_id: str | DemoSpec = "demo",
        *,
        root: PathLike | None = None,
    ) -> None:
        if project_root is not None and root is not None:
            raise TypeError("provide project_root or root, not both")
        if isinstance(run_id, DemoSpec):
            run_id = run_id.run_id
        if isinstance(project_root, DemoSpec) and isinstance(run_id, (str, Path)):
            run_id = str(run_id)
        selected_root = root if root is not None else project_root
        if isinstance(selected_root, DemoSpec):
            raise TypeError("project_root cannot be a DemoSpec")
        if selected_root is None:
            raise TypeError("project_root is required")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise DemoPathError("run_id must be a simple path-safe name")
        object.__setattr__(self, "project_root", Path(selected_root).expanduser().resolve(strict=False))
        object.__setattr__(self, "run_id", run_id)
        self.validate()

    @property
    def root(self) -> Path:
        return self.project_root

    @property
    def base(self) -> Path:
        return self.project_root / "logs" / "demo"

    @property
    def logs(self) -> Path:
        return self.base / self.run_id

    @property
    def transcripts(self) -> Path:
        return self.logs / "transcripts"

    @property
    def events(self) -> Path:
        return self.logs / "events.jsonl"

    @property
    def episodes(self) -> Path:
        return self.logs / "episodes.jsonl"

    @property
    def retries(self) -> Path:
        return self.logs / "retries.jsonl"

    @property
    def cost_estimate_path(self) -> Path:
        return self.logs / "cost_estimate.json"

    def validate(self) -> DemoPaths:
        base = self.base.resolve(strict=False)
        logs = self.logs.resolve(strict=False)
        transcripts = self.transcripts.resolve(strict=False)
        if not _beneath(base, self.project_root):
            raise DemoPathError(f"demo base escapes project root: {base}")
        if not _beneath(logs, base) or logs == base:
            raise DemoPathError(f"demo logs escape {base}: {logs}")
        if not _beneath(transcripts, base) or not _beneath(transcripts, logs):
            raise DemoPathError(f"demo transcripts escape {base}: {transcripts}")
        return self

    def run_paths(self) -> RunPaths:
        self.validate()
        return RunPaths(logs=self.logs, transcripts=self.transcripts)

    @classmethod
    def for_project(cls, project_root: PathLike, run_id: str = "demo") -> DemoPaths:
        return cls(project_root=project_root, run_id=run_id)

    @classmethod
    def for_run(cls, project_root: PathLike, run_id: str = "demo") -> DemoPaths:
        return cls.for_project(project_root, run_id)

    @classmethod
    def for_spec(cls, project_root: PathLike | DemoSpec, spec: DemoSpec | PathLike) -> DemoPaths:
        if isinstance(project_root, DemoSpec) and isinstance(spec, (str, Path)):
            project_root, spec = spec, project_root
        if not isinstance(spec, DemoSpec) or not isinstance(project_root, (str, Path)):
            raise TypeError("for_spec requires a project root and DemoSpec")
        return cls.for_project(project_root, spec.run_id)

    @classmethod
    def from_spec(cls, spec: DemoSpec, project_root: PathLike) -> DemoPaths:
        return cls.for_spec(project_root, spec)


@dataclass(frozen=True, slots=True)
class DemoConfirmation:
    spec: DemoSpec
    estimate: Estimate
    token: str
    path: Path

    @property
    def confirmation_token(self) -> str:
        return self.token


@dataclass
class DemoJob:
    job_id: str
    spec: DemoSpec
    paths: DemoPaths
    status: DemoJobStatus = "queued"
    result: EpisodeRecord | None = None
    error: str | None = None
    reason: str | None = None
    runner_status: RunnerStatus | None = None
    estimate: Estimate | None = None
    actual_cost_usd: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exception: BaseException | None = field(default=None, repr=False, compare=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False, compare=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    @property
    def state(self) -> DemoJobStatus:
        return self.status

    @property
    def cost_estimate(self) -> Estimate | None:
        return self.estimate

    @property
    def actual_cost(self) -> float | None:
        return self.actual_cost_usd

    @property
    def runner_reason(self) -> str | None:
        return self.reason

    @property
    def logs(self) -> Path:
        return self.paths.logs

    @property
    def transcripts(self) -> Path:
        return self.paths.transcripts

    @property
    def events(self) -> Path:
        return self.paths.events

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "error"}

    @property
    def done(self) -> bool:
        return self.terminal

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    def poll(self) -> DemoJob:
        return self

    def wait(self, timeout: float | None = None) -> DemoJob:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self

    def raise_if_error(self) -> DemoJob:
        if self.exception is not None:
            raise self.exception
        return self


def _provider_config(spec: DemoSpec) -> ProviderConfig | None:
    if spec.mode == "scripted":
        return None
    if spec.mode == "qwen3":
        return PRESETS["qwen3"]
    return groq_config(spec.groq_model)


def _model_for_spec(spec: DemoSpec) -> str:
    provider = _provider_config(spec)
    return "scripted" if provider is None else provider.model


def _estimate_for_spec(spec: DemoSpec) -> Estimate:
    model = _model_for_spec(spec)
    task_count = spec.scenario.d1_tasks if spec.drift == "D1" else spec.scenario.tasks_per_episode
    return estimate(model, [(spec.role, task_count, spec.drift != "D1")], spec.max_turns)


def _estimate_dict(value: Estimate) -> dict[str, Any]:
    return asdict(value)


def _write_json(path: Path, value: dict[str, Any], paths: DemoPaths) -> None:
    paths.validate()
    resolved = path.resolve(strict=False)
    if not _beneath(resolved, paths.base.resolve(strict=False)):
        raise DemoPathError(f"metadata escapes demo directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def record_cost_estimate(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None = None,
) -> Path:
    paths = DemoPaths.for_spec(project_root, spec)
    value = estimate_value or _estimate_for_spec(spec)
    model = _model_for_spec(spec)
    _write_json(
        paths.cost_estimate_path,
        {"spec": spec.as_dict(), "model": model, "estimate": _estimate_dict(value), "confirmation_token": None},
        paths,
    )
    return paths.cost_estimate_path


def _read_cost_record(paths: DemoPaths) -> dict[str, Any] | None:
    path = paths.cost_estimate_path
    if not path.exists() or path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _estimate_from_record(record: dict[str, Any]) -> Estimate | None:
    value = record.get("estimate")
    if not isinstance(value, dict):
        return None
    try:
        return Estimate(**value)
    except (TypeError, ValueError):
        return None


def _confirmation_digest(spec: DemoSpec, value: Estimate) -> str:
    payload = json.dumps({"spec": spec.as_dict(), "estimate": _estimate_dict(value)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_demo_cost(
    spec: DemoSpec,
    project_root: PathLike | None = None,
    *,
    record: bool = True,
) -> Estimate:
    value = _estimate_for_spec(spec)
    if record and project_root is not None:
        record_cost_estimate(spec, project_root, value)
    return value


def estimate_cost(
    spec: DemoSpec,
    project_root: PathLike | None = None,
    *,
    record: bool = True,
) -> Estimate:
    return estimate_demo_cost(spec, project_root, record=record)


def issue_confirmation_token(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None = None,
) -> str:
    value = estimate_value or _estimate_for_spec(spec)
    path = record_cost_estimate(spec, project_root, value)
    paths = DemoPaths.for_spec(project_root, spec)
    record = _read_cost_record(paths) or {}
    token = _confirmation_digest(spec, value)
    record["confirmation_token"] = token
    _write_json(path, record, paths)
    return token


def make_confirmation_token(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None = None,
) -> str:
    return issue_confirmation_token(spec, project_root, estimate_value)


def confirm_demo(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None = None,
) -> DemoConfirmation:
    value = estimate_value or _estimate_for_spec(spec)
    token = issue_confirmation_token(spec, project_root, value)
    return DemoConfirmation(
        spec=spec,
        estimate=value,
        token=token,
        path=DemoPaths.for_spec(project_root, spec).cost_estimate_path,
    )


def prepare_demo(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None = None,
) -> DemoConfirmation:
    return confirm_demo(spec, project_root, estimate_value)


def _confirmation_for_start(
    spec: DemoSpec,
    project_root: PathLike,
    estimate_value: Estimate | None,
    token: str | None,
    confirmation: DemoConfirmation | None,
) -> Estimate:
    if confirmation is not None:
        if confirmation.spec != spec:
            raise DemoConfirmationError("confirmation does not match the demo specification")
        estimate_value = confirmation.estimate
        token = confirmation.token
    paths = DemoPaths.for_spec(project_root, spec)
    record = _read_cost_record(paths)
    if record is not None and record.get("spec") not in (None, spec.as_dict()):
        raise DemoConfirmationError("the recorded approval belongs to a different demo specification")
    recorded_estimate = _estimate_from_record(record) if record is not None else None
    if estimate_value is None:
        estimate_value = recorded_estimate
    if estimate_value is None:
        raise DemoConfirmationError("Groq demo requires a cost estimate")
    if token is None or not str(token).strip():
        raise DemoConfirmationError("Groq demo requires an explicit confirmation token")
    if record is not None and recorded_estimate is not None and recorded_estimate != estimate_value:
        raise DemoConfirmationError("the supplied cost estimate does not match the recorded estimate")
    recorded_token = record.get("confirmation_token") if record is not None else None
    token_value = str(token)
    if recorded_token and token_value != str(recorded_token):
        raise DemoConfirmationError("the confirmation token does not match the recorded approval")
    if record is None or record.get("model") != _model_for_spec(spec) or recorded_estimate is None:
        record_cost_estimate(spec, project_root, estimate_value)
        record = _read_cost_record(paths)
    if record is None:
        record_cost_estimate(spec, project_root, estimate_value)
        record = _read_cost_record(paths) or {}
    record["confirmation_token"] = token_value
    _write_json(paths.cost_estimate_path, record, paths)
    return estimate_value


def _status_check(project_root: Path) -> RunnerStatus:
    try:
        return detect_active_runner(project_root)
    except Exception as exc:
        return RunnerStatus(True, f"runner status check failed conservatively: {exc}", project_root)


def _job_snapshot(job: DemoJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "reason": job.reason,
        "error": job.error,
        "paths": job.paths,
        "result": job.result,
        "estimate": job.estimate,
        "actual_cost_usd": job.actual_cost_usd,
    }


class DemoManager:
    def __init__(self, project_root: PathLike | None = None, provider_factory: ProviderFactory | None = None) -> None:
        self.project_root = (
            Path.cwd() if project_root is None else Path(project_root).expanduser().resolve(strict=False)
        )
        self._provider_factory = provider_factory
        self._jobs: dict[str, DemoJob] = {}
        self._lock = threading.RLock()

    def start(
        self,
        spec: DemoSpec | None = None,
        paths: DemoPaths | None = None,
        *,
        cost_estimate: Estimate | None = None,
        confirmation_token: str | None = None,
        confirmation: DemoConfirmation | None = None,
        raise_on_error: bool = False,
        **spec_options: Any,
    ) -> DemoJob:
        if spec is None:
            if paths is not None:
                spec = DemoSpec(run_id=paths.run_id)
            else:
                spec = DemoSpec(**spec_options)
        elif spec_options:
            raise TypeError("spec options cannot be combined with a DemoSpec")
        if paths is None:
            paths = DemoPaths.for_spec(self.project_root, spec)
        elif paths.project_root != self.project_root:
            raise DemoPathError("DemoPaths belongs to a different project root")
        elif paths.run_id != spec.run_id:
            raise DemoPathError("DemoPaths does not match the demo specification")
        job = DemoJob(job_id=uuid.uuid4().hex, spec=spec, paths=paths)
        with self._lock:
            self._jobs[job.job_id] = job

        try:
            paths.validate()
            status = _status_check(self.project_root)
            job.runner_status = status
            if spec.mode != "scripted":
                if status.active:
                    raise DemoRunnerActiveError(status)
                if spec.mode == "groq":
                    cost_estimate = _confirmation_for_start(
                        spec, self.project_root, cost_estimate, confirmation_token, confirmation
                    )
                elif cost_estimate is not None:
                    record_cost_estimate(spec, self.project_root, cost_estimate)
                status = _status_check(self.project_root)
                job.runner_status = status
                if status.active:
                    raise DemoRunnerActiveError(status)
            elif cost_estimate is not None:
                record_cost_estimate(spec, self.project_root, cost_estimate)
            job.estimate = cost_estimate
            thread = threading.Thread(
                target=self._run,
                args=(job, self._provider_factory),
                name=f"demo-{job.job_id}",
                daemon=True,
            )
            job._thread = thread
            thread.start()
            return job
        except Exception as exc:
            return self._fail(job, exc, raise_on_error)

    def _fail(self, job: DemoJob, exc: Exception, raise_on_error: bool) -> DemoJob:
        with job._lock:
            job.status = "error"
            job.error = str(exc)
            runner_reason = job.runner_status.reason if job.runner_status and job.runner_status.active else None
            job.reason = getattr(exc, "reason", None) or runner_reason or str(exc)
            job.exception = exc
            job.finished_at = _now()
        if raise_on_error:
            raise exc
        return job

    def _run(self, job: DemoJob, provider_factory: ProviderFactory | None) -> None:
        try:
            with job._lock:
                job.status = "running"
                job.started_at = _now()
            if job.spec.mode != "scripted":
                status = _status_check(self.project_root)
                job.runner_status = status
                if status.active:
                    raise DemoRunnerActiveError(status)
            paths = job.paths
            paths.validate()
            provenance = frozen_provenance()
            state = generate_state(job.spec.seed)
            tasks = build_episode(job.spec.role, job.spec.drift, job.spec.seed, state, job.spec.scenario)
            provider_cfg = _provider_config(job.spec)
            if job.spec.mode == "scripted":
                scripts = scripts_for(tasks, job.spec.drift, state, random.Random(job.spec.seed))
                agent: Any = ScriptedAgent(scripts)
                episode_provider = None
                model = "scripted"
            else:
                if provider_cfg is None:
                    raise DemoStartError("real-model provider configuration is missing")
                episode_provider = provider_cfg.model_copy(update={"seed": job.spec.seed})
                factory = provider_factory or make_provider
                agent = LLMAgent(factory(episode_provider), job.spec.role, job.spec.max_turns)
                model = episode_provider.model
            cfg = EpisodeConfig(
                run_id=paths.run_id,
                episode_id=f"{paths.run_id}-episode",
                seed=job.spec.seed,
                model=model,
                agent_role=job.spec.role,
                drift_condition=job.spec.drift,
                control_condition=job.spec.control,
                provider=episode_provider,
                max_turns_per_task=job.spec.max_turns,
            )
            record, _ = run_episode(cfg, tasks, agent, paths.run_paths(), provenance, state=state)
            cost = actual_cost(model, [record])
            with job._lock:
                job.result = record
                job.actual_cost_usd = cost
                job.status = "completed"
                job.finished_at = _now()
        except Exception as exc:
            with job._lock:
                job.status = "error"
                job.error = str(exc)
                runner_reason = job.runner_status.reason if job.runner_status and job.runner_status.active else None
                job.reason = getattr(exc, "reason", None) or runner_reason or str(exc)
                job.exception = exc
                job.finished_at = _now()

    def get(self, job_id: str) -> DemoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job(self, job_id: str) -> DemoJob | None:
        return self.get(job_id)

    def poll(self, job_id: str | DemoJob | None = None) -> DemoJob | list[DemoJob]:
        if job_id is None:
            return self.poll_all()
        key = job_id.job_id if isinstance(job_id, DemoJob) else job_id
        job = self.get(key)
        if job is None:
            raise KeyError(key)
        return job

    def snapshot(self, job_id: str | DemoJob) -> dict[str, Any]:
        return _job_snapshot(self.poll(job_id))

    def poll_all(self) -> list[DemoJob]:
        with self._lock:
            return list(self._jobs.values())

    @property
    def jobs(self) -> list[DemoJob]:
        return self.poll_all()

    def wait(self, job_id: str | DemoJob, timeout: float | None = None) -> DemoJob:
        return self.poll(job_id).wait(timeout)


def start_demo(
    spec: DemoSpec | PathLike | DemoPaths | None = None,
    project_root: PathLike | DemoSpec | DemoPaths | None = None,
    *,
    paths: DemoPaths | None = None,
    manager: DemoManager | None = None,
    cost_estimate: Estimate | None = None,
    confirmation_token: str | None = None,
    confirmation: DemoConfirmation | None = None,
    raise_on_error: bool = True,
    **options: Any,
) -> DemoJob:
    if isinstance(spec, DemoPaths):
        if paths is not None or project_root is not None:
            raise TypeError("spec cannot be combined with paths or project_root")
        paths = spec
        spec = None
    if isinstance(project_root, DemoPaths):
        if paths is not None:
            raise TypeError("provide paths or project_root, not both")
        paths = project_root
        project_root = None
    if isinstance(spec, (str, Path)):
        if isinstance(project_root, DemoSpec):
            spec, project_root = project_root, spec
        elif project_root is None:
            project_root, spec = spec, None
        else:
            raise TypeError("provide a DemoSpec or one project root")
    if isinstance(project_root, DemoSpec) and spec is not None:
        raise TypeError("a DemoSpec cannot be used as the project root")
    if isinstance(project_root, DemoSpec) and spec is None:
        spec = project_root
        project_root = None
    root_option = options.pop("root", None)
    if spec is None:
        spec = DemoSpec(**options)
    elif options:
        raise TypeError("spec options cannot be combined with a DemoSpec")
    if project_root is None:
        project_root = root_option
    if project_root is None and paths is not None:
        project_root = paths.project_root
    if project_root is None and manager is not None:
        project_root = manager.project_root
    if project_root is None:
        project_root = Path.cwd()
    if manager is None:
        manager = DemoManager(project_root)
    return manager.start(
        spec,
        paths,
        cost_estimate=cost_estimate,
        confirmation_token=confirmation_token,
        confirmation=confirmation,
        raise_on_error=raise_on_error,
    )


def start_demo_job(*args: Any, **kwargs: Any) -> DemoJob:
    kwargs.setdefault("raise_on_error", False)
    return start_demo(*args, **kwargs)


def _demo_base_from_path(path: Path) -> Path:
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index].casefold() == "logs" and parts[index + 1].casefold() == "demo":
            return Path(*parts[: index + 2])
    raise DemoPathError("live event path must be beneath logs/demo")


def _selected_event_path(path: PathLike | DemoPaths | DemoJob) -> Path:
    if isinstance(path, DemoPaths):
        return path.events
    if isinstance(path, DemoJob):
        return path.paths.events
    return Path(path)


def _validated_event_path(path: PathLike | DemoPaths | DemoJob) -> tuple[Path, Path]:
    selected = _selected_event_path(path)
    if selected.name != "events.jsonl" and selected.is_dir():
        selected = selected / "events.jsonl"
    resolved = selected.expanduser().resolve(strict=False)
    base = _demo_base_from_path(resolved).resolve(strict=False)
    if not _beneath(resolved, base):
        raise DemoPathError("live event path escapes the demo log directory")
    relative = resolved.relative_to(base)
    if any(part.casefold() == "transcripts" for part in relative.parts):
        raise DemoPathError("live event path is inside the transcripts directory")
    if resolved.name != "events.jsonl":
        raise DemoPathError("live event reader only reads events.jsonl")
    return resolved, base


def read_live_events(path: PathLike | DemoPaths | DemoJob, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if limit == 0:
        return []
    resolved, _ = _validated_event_path(path)
    if not resolved.exists():
        return []
    if not resolved.is_file():
        raise DemoPathError("live event path is not a regular file")
    events: list[dict[str, Any]] = []
    try:
        with resolved.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(value, dict):
                    continue
                events.append(value)
                if limit is not None and len(events) >= limit:
                    break
    except (OSError, UnicodeError):
        return events
    return events


def iter_live_events(path: PathLike | DemoPaths | DemoJob) -> Any:
    yield from read_live_events(path)


def read_live_event_lines(path: PathLike | DemoPaths | DemoJob) -> list[str]:
    resolved, _ = _validated_event_path(path)
    if not resolved.exists() or not resolved.is_file():
        return []
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
        return [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return []


read_live_event = read_live_events


__all__ = [
    "DemoConfirmation",
    "DemoConfirmationError",
    "DemoError",
    "DemoJob",
    "DemoJobStatus",
    "DemoManager",
    "DemoMode",
    "DemoPathError",
    "DemoPaths",
    "DemoRunnerActiveError",
    "DemoSpec",
    "DemoStartError",
    "confirm_demo",
    "estimate_cost",
    "estimate_demo_cost",
    "issue_confirmation_token",
    "iter_live_events",
    "make_confirmation_token",
    "prepare_demo",
    "read_live_event",
    "read_live_event_lines",
    "read_live_events",
    "record_cost_estimate",
    "start_demo",
    "start_demo_job",
]
