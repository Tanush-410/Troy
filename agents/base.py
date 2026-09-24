"""The interface every agent implements.

An agent sees only an AgentTask (task id and instruction text) and acts only
through the AgentGateway. It never receives the TaskSpec, the PEP, or the logs.
"""

from __future__ import annotations

from typing import Any, Protocol

from policy.pep import AgentGateway
from policy.task import AgentTask

TranscriptEntry = dict[str, Any]


class Agent(Protocol):
    def reset(self) -> None:
        """Start a fresh context (called before every task except under D1)."""
        ...

    def run_task(self, task: AgentTask, gateway: AgentGateway) -> list[TranscriptEntry]:
        """Work on one task through the gateway; return transcript entries."""
        ...
