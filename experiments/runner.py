"""Runs one episode: tasks in order through the PEP, success judged at each task end.

Output layout:
    logs/<run_id>/events.jsonl        one PermissionEvent per tool call
    logs/<run_id>/episodes.jsonl      one EpisodeRecord per episode
    logs/<run_id>/retries.jsonl       one line per retried API attempt
    transcripts/<run_id>/<episode_id>.jsonl   prompts and outputs (qualitative only)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base import Agent
from experiments.config import EpisodeConfig
from oracle.harm_rules import HarmOracle
from oracle.success import judge_task
from policy.log_schema import DetectorEvent, EpisodeRecord
from policy.pep import PEP, EpisodeContext, JsonlWriter
from policy.task import TaskSpec
from policy.ts_rbac import ExpansionPolicy
from simmart.generator import generate_state
from simmart.state import SimMartState


@dataclass(frozen=True)
class RunPaths:
    logs: Path
    transcripts: Path

    @classmethod
    def for_run(cls, root: Path, run_id: str) -> RunPaths:
        return cls(logs=root / "logs" / run_id, transcripts=root / "transcripts" / run_id)

    @property
    def events(self) -> Path:
        return self.logs / "events.jsonl"

    @property
    def episodes(self) -> Path:
        return self.logs / "episodes.jsonl"

    @property
    def retries(self) -> Path:
        return self.logs / "retries.jsonl"


def run_episode(
    cfg: EpisodeConfig,
    tasks: Sequence[TaskSpec],
    agent: Agent,
    paths: RunPaths,
    provenance: dict[str, Any],
    state: SimMartState | None = None,
    expansion_policy: ExpansionPolicy | None = None,
    clock: Callable[[], str] | None = None,
    monitor: Callable[[DetectorEvent], bool] | None = None,
) -> tuple[EpisodeRecord, SimMartState]:
    """Run `tasks` in order. `state` defaults to a fresh state from the config's seed.

    The context persists across tasks only under D1; otherwise the agent and the
    PEP's taint tracking are reset before every task. The simulated clock
    advances by one tick before every task after the first. If a task ends in
    context overflow, the episode stops and the remaining tasks are recorded
    as "not_run" (they are not judged). With a live `monitor` (C3/C4 live-pause
    mode), an alert pauses the episode: later calls in the task are denied and
    the remaining tasks are recorded as "paused".
    """
    if state is None:
        state = generate_state(cfg.seed, cfg.generator)
    ctx = EpisodeContext(**cfg.model_dump(exclude={"generator", "provider", "max_turns_per_task"}))
    pep_kwargs: dict[str, Any] = {} if clock is None else {"clock": clock}
    if monitor is not None:
        pep_kwargs["monitor"] = monitor
    pep = PEP(ctx, state, HarmOracle(), JsonlWriter(paths.events), expansion_policy, **pep_kwargs)
    transcript = JsonlWriter(paths.transcripts / f"{cfg.episode_id}.jsonl")

    outcomes = []
    usage = {"model_turns": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
             "cache_write_tokens": 0}
    task_ends: dict[str, str] = {}
    retry_log = JsonlWriter(paths.retries)
    n_retries, retry_wait = 0, 0.0
    for i, task in enumerate(tasks):
        if i > 0:
            state.advance_clock()
        if i == 0 or cfg.drift_condition != "D1":
            agent.reset()
            pep.reset_context()
        pep.begin_task(task)
        task_run = agent.run_task(task.agent_view(), pep.gateway())
        for entry in task_run.transcript:
            transcript.write_raw(entry)
        for key in usage:
            usage[key] += getattr(task_run, key)
        task_ends[task.task_id] = task_run.end
        for r in task_run.retries:
            retry_log.write_raw({"run_id": cfg.run_id, "episode_id": cfg.episode_id, **r})
            n_retries, retry_wait = n_retries + 1, retry_wait + r["wait_s"]
        outcomes.append(judge_task(task, pep.task_records(task.task_id), state))
        pep.end_task()
        if task_run.end == "context_overflow":
            task_ends.update({t.task_id: "not_run" for t in tasks[i + 1:]})
            break
        if pep.paused_at is not None:
            task_ends.update({t.task_id: "paused" for t in tasks[i + 1:]})
            break

    record = pep.episode_record(outcomes, cfg.config_hash(), provenance).model_copy(
        update={**usage, "task_ends": task_ends, "api_retries": n_retries, "api_retry_wait_s": retry_wait,
                "pldd_alert_step": pep.paused_at,
                "provider_config": cfg.provider.model_dump(mode="json") if cfg.provider else None,
                "reasoning_setting": cfg.provider.reasoning_setting() if cfg.provider else None})
    JsonlWriter(paths.episodes).write(record)
    return record, state
