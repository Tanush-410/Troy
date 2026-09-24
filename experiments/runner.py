"""Runs one episode: tasks in order through the PEP, success judged at each task end.

Output layout:
    logs/<run_id>/events.jsonl        one PermissionEvent per tool call
    logs/<run_id>/episodes.jsonl      one EpisodeRecord per episode
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
from policy.log_schema import EpisodeRecord
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


def run_episode(
    cfg: EpisodeConfig,
    tasks: Sequence[TaskSpec],
    agent: Agent,
    paths: RunPaths,
    provenance: dict[str, Any],
    state: SimMartState | None = None,
    expansion_policy: ExpansionPolicy | None = None,
    clock: Callable[[], str] | None = None,
) -> tuple[EpisodeRecord, SimMartState]:
    """Run `tasks` in order. `state` defaults to a fresh state from the config's seed.

    The context persists across tasks only under D1; otherwise the agent and the
    PEP's taint tracking are reset before every task. The simulated clock
    advances by one tick before every task after the first.
    """
    if state is None:
        state = generate_state(cfg.seed, cfg.generator)
    ctx = EpisodeContext(**cfg.model_dump(exclude={"generator"}))
    pep_kwargs: dict[str, Any] = {} if clock is None else {"clock": clock}
    pep = PEP(ctx, state, HarmOracle(), JsonlWriter(paths.events), expansion_policy, **pep_kwargs)
    transcript = JsonlWriter(paths.transcripts / f"{cfg.episode_id}.jsonl")

    outcomes = []
    for i, task in enumerate(tasks):
        if i > 0:
            state.advance_clock()
        if i == 0 or cfg.drift_condition != "D1":
            agent.reset()
            pep.reset_context()
        pep.begin_task(task)
        for entry in agent.run_task(task.agent_view(), pep.gateway()):
            transcript.write_raw(entry)
        outcomes.append(judge_task(task, pep.task_records(task.task_id), state))
        pep.end_task()

    record = pep.episode_record(outcomes, cfg.config_hash(), provenance)
    JsonlWriter(paths.episodes).write(record)
    return record, state
