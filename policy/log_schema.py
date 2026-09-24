"""Log records written by the PEP (CLAUDE.md section 7).

`PermissionEvent` is the full per-call record, including ground-truth labels.
`DetectorEvent` is the only view the PLDD may consume: action, params,
decision and timing, whether the PEP had to ask for a scope expansion, plus the
orchestrator-assigned role and task type needed to pick a baseline. It carries
no labels, taint, tokens, prompts or outputs. The detector must load logs through `read_detector_events`, never by parsing
PermissionEvent itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from oracle.drift import DriftType
from policy.permissions import Role, TaskType

DriftCondition = Literal["D0", "D1", "D2", "D3"]
ControlCondition = Literal["C1", "C2", "C3", "C4"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpansionRequest(_Record):
    """A C2/C4 request to add an in-role action to the current task scope."""

    action: str
    task_id: str | None
    task_type: TaskType | None
    step: int
    granted: bool


class PermissionEvent(_Record):
    run_id: str
    episode_id: str
    seed: int
    model: str
    agent_role: Role
    task_id: str | None  # None between tasks (D1)
    task_type: TaskType | None
    drift_condition: DriftCondition
    control_condition: ControlCondition
    step: int
    timestamp: str  # ISO 8601, UTC
    latency_ms: float  # whole PEP handling time, including tool execution
    decision_latency_ms: float  # permission check alone (RQ2 added latency)
    action: str
    params: dict[str, Any]
    decision: Literal["allow", "deny"]
    deny_layer: Literal["rbac", "ts_rbac"] | None
    expansion_request: ExpansionRequest | None  # C2/C4 only, logged even when denied
    in_role: bool
    in_task: bool  # against the frozen S_tau, whatever the control condition
    drift_type: DriftType
    harm: bool
    harm_rule_ids: list[str]
    executed: bool  # allowed and the tool call succeeded
    tool_ok: bool | None  # None when denied
    tainted_context: bool  # an injection was in the agent's context at call time
    context_injection_ids: list[str]
    cross_agent_taint: bool  # context taint arrived via a price report
    taint_report_ids: list[str]
    tokens_in: int
    tokens_out: int


class TaskOutcome(_Record):
    """Deterministic success verdict for one task, kept separate from harm."""

    task_id: str
    task_type: TaskType
    outcome: Literal["completed", "escalated", "failed"]
    escalation_acceptable: bool
    success: bool  # completed, or escalated when escalation is acceptable
    reason: str


class EpisodeRecord(_Record):
    run_id: str
    episode_id: str
    seed: int
    model: str
    agent_role: Role
    drift_condition: DriftCondition
    control_condition: ControlCondition
    n_steps: int
    task_outcomes: list[TaskOutcome]
    n_tasks: int
    task_success_rate: float
    escalation_rate: float  # share of tasks whose outcome is "escalated"
    total_harmful_attempted: int
    total_harmful_executed: int
    first_harm_step: int | None  # first harmful call, attempted or executed
    first_executed_harm_step: int | None
    config_hash: str
    provenance: dict[str, Any] = Field(default_factory=dict)  # frozen-file commit hashes


class DetectorEvent(_Record):
    episode_id: str
    agent_role: Role
    task_id: str | None
    task_type: TaskType | None
    step: int
    timestamp: str
    latency_ms: float
    action: str
    params: dict[str, Any]
    decision: Literal["allow", "deny"]
    expansion_requested: bool


_COPIED = set(DetectorEvent.model_fields) - {"expansion_requested"}


def _detector_fields(raw: dict[str, Any]) -> DetectorEvent:
    return DetectorEvent.model_validate(
        {k: raw[k] for k in _COPIED} | {"expansion_requested": raw["expansion_request"] is not None}
    )


def to_detector_event(e: PermissionEvent) -> DetectorEvent:
    return _detector_fields(e.model_dump(mode="json"))


def read_detector_events(path: Path) -> Iterator[DetectorEvent]:
    """Read a PEP JSONL log, keeping only the fields the detector may see."""
    with path.open() as f:
        for line in f:
            if line.strip():
                yield _detector_fields(json.loads(line))
