"""The interface every agent implements.

An agent sees only an AgentTask (task id and instruction text) and acts only
through the AgentGateway. It never receives the TaskSpec, the PEP, or the logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from policy.pep import AgentGateway
from policy.task import AgentTask

TranscriptEntry = dict[str, Any]


@dataclass
class TaskRun:
    """What one task cost and how it ended, from the agent's side."""

    transcript: list[TranscriptEntry] = field(default_factory=list)
    model_turns: int = 0
    input_tokens: int = 0  # all input, cached or not
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    end: str = "done"  # done | step_limit | max_tokens | refusal | context_overflow


class Agent(Protocol):
    def reset(self) -> None:
        """Start a fresh context (called before every task except under D1)."""
        ...

    def run_task(self, task: AgentTask, gateway: AgentGateway) -> TaskRun:
        """Work on one task through the gateway."""
        ...
