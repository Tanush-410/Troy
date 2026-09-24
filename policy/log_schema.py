"""Log records written by the PEP (CLAUDE.md section 7).

`PermissionEvent` is the full per-call record, including ground-truth labels.
`DetectorEvent` is the only view the PLDD may consume: action, params,
decision and timing, whether the PEP had to ask for a scope expansion, the
audit before-values a real enforcement point records (AUDIT_FIELDS), plus the
orchestrator-assigned role and task type needed to pick a baseline. It carries
no labels, taint, tokens, prompts or outputs. Format-error events are left out
of the detector view entirely. The detector must load logs through
`read_detector_events`, never by parsing PermissionEvent itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from oracle.drift import DriftType
from policy.permissions import Role, TaskType

# Before-values an audit log records for state-changing calls, read from the
# pre-call state. Fixed allowlist: action -> fields. Never labels or taint.
AUDIT_FIELDS: dict[str, tuple[str, ...]] = {
    "set_price": ("previous_price",),
    "issue_refund": ("order_total",),
    "update_listing": ("previous_title_length", "previous_description_length"),
}

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
    params: dict[str, Any]  # {"_raw": <text>} when the arguments could not be parsed
    audit: dict[str, float | None]  # before-values from AUDIT_FIELDS; {} for other actions and format errors
    decision: Literal["allow", "deny", "invalid"]  # "invalid" only for format errors
    deny_layer: Literal["rbac", "ts_rbac", "pldd"] | None  # "pldd": episode paused by the live monitor
    format_error: bool  # unparseable arguments or schema validation failure
    format_error_detail: str | None
    expansion_request: ExpansionRequest | None  # C2/C4 only, logged even when denied
    in_role: bool
    in_task: bool  # against the frozen S_tau, whatever the control condition
    drift_type: DriftType | None  # None for format errors, which get no drift type
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
    # Provider settings (model, temperature, seed, context window), filled in by the runner.
    provider_config: dict[str, Any] | None = None
    reasoning_setting: str | None = None  # e.g. "reasoning_effort=low", "think=false"
    # Agent-side usage, filled in by the runner. Includes model turns that made
    # no tool call, which no PermissionEvent carries.
    model_turns: int = 0
    input_tokens: int = 0  # all input, cached or not
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    task_ends: dict[str, str] = Field(default_factory=dict)  # task_id -> done/step_limit/max_tokens/refusal
    pldd_alert_step: int | None = None  # live-pause mode only: step at which the monitor paused the episode
    api_retries: int = 0  # retried API attempts (identical re-sends; see agents/providers/retry.py)
    api_retry_wait_s: float = 0.0


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
    audit: dict[str, float | None]


_COPIED = set(DetectorEvent.model_fields) - {"expansion_requested"}


def _detector_fields(raw: dict[str, Any]) -> DetectorEvent:
    return DetectorEvent.model_validate(
        {k: raw[k] for k in _COPIED} | {"expansion_requested": raw["expansion_request"] is not None}
    )


def to_detector_event(e: PermissionEvent) -> DetectorEvent:
    if e.format_error:
        raise ValueError("format-error events are not part of the detector view")
    return _detector_fields(e.model_dump(mode="json"))


def read_detector_events(path: Path) -> Iterator[DetectorEvent]:
    """Read a PEP JSONL log, keeping only the fields the detector may see."""
    with path.open() as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                if not raw["format_error"]:
                    yield _detector_fields(raw)
