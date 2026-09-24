"""A zero-cost agent that replays a fixed action script.

Each task's script is a list of Steps. A step's params may be a callable that
receives the agent-visible replies so far in the task, so a script can use a
value it just read (a new listing id, a fetched price) the way a model would.
`expect_drift` / `expect_rules` record the label the script author intends;
the agent ignores them, and tests compare them to what the PEP logged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agents.base import TaskRun, TranscriptEntry
from oracle.drift import DriftType
from policy.pep import AgentGateway
from policy.task import AgentTask

Params = dict[str, Any] | Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class Step:
    action: str
    params: Params
    expect_drift: DriftType = "none"
    expect_rules: tuple[str, ...] = ()


@dataclass
class ScriptedAgent:
    scripts: dict[str, Sequence[Step]]
    resets: int = field(default=0, init=False)

    def reset(self) -> None:
        self.resets += 1

    def run_task(self, task: AgentTask, gateway: AgentGateway) -> TaskRun:
        replies: list[dict[str, Any]] = []
        transcript: list[TranscriptEntry] = [{"role": "user", "task_id": task.task_id, "content": task.instruction}]
        for step in self.scripts[task.task_id]:
            params = step.params(replies) if callable(step.params) else step.params
            reply = gateway.call(step.action, params)
            replies.append(reply)
            transcript.append({"role": "assistant", "task_id": task.task_id,
                               "tool_call": {"name": step.action, "params": params}})
            transcript.append({"role": "tool", "task_id": task.task_id, "content": reply})
        return TaskRun(transcript=transcript)
