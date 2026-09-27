"""A human-driven session against the real PEP.

This is a demonstration surface, not an experiment harness. A person typing
free-form messages has no seed, so CLAUDE.md rule 6 (reproduce a run exactly from
its config and seed) cannot hold for it, and `pi*_tau` -- the reference policy the
drift score is measured against -- needs the same agent on the same task type with
no drift induction. Everything produced here is therefore tagged
``interactive_ui: True`` and the analysis readers refuse to load it. See
docs/ui-and-roles.md section 5.4.

What is *not* relaxed: the PEP is still the only path to the environment, still
the only writer of the permission log, and the agent still cannot see or change its
own permissions. Nothing here bypasses RBAC.

The one real difference from `agents.llm_agent.LLMAgent` is the loop shape. That
class runs a task to completion in a single blocking call, which cannot interleave
a human. The `Session` protocol is resumable, so this drives it a turn at a time
and yields after every observable event.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from agents.prompts import system_prompt
from agents.providers.base import ContextOverflow, ProviderConfig, Session, ToolOutcome
from agents.providers.factory import make_provider
from oracle.harm_rules import HarmOracle
from policy.log_schema import ControlCondition, DriftCondition, PermissionEvent
from policy.pep import PEP, EpisodeContext
from policy.permissions import ROLES, Role, TaskType
from policy.task import TaskSpec
from scenarios.generator import ROLE_TASK_TYPES
from simmart.generator import generate_state
from simmart.state import SimMartState
from tools import TOOLS, all_tool_schemas

__all__ = [
    "PROVENANCE",
    "UIEventSink",
    "InteractiveSession",
    "catalog",
    "role_task_types",
]

#: Stamped onto anything this module persists. The analysis pipeline rejects it.
PROVENANCE: dict[str, Any] = {"interactive_ui": True, "experimental": False}


class UIEventSink:
    """Duck-typed PEP event sink: keeps events in memory instead of on disk.

    `PEP` only ever calls `.write(record)`, so no subclassing is needed. Keeping
    them in memory is what lets the UI stream the log as it is written rather than
    tailing a file.
    """

    def __init__(self) -> None:
        self.events: list[PermissionEvent] = []

    def write(self, record: PermissionEvent) -> None:
        self.events.append(record)

    def write_raw(self, obj: dict[str, Any]) -> None:  # pragma: no cover - parity with JsonlWriter
        return None


def role_task_types(role: Role) -> tuple[TaskType, ...]:
    return ROLE_TASK_TYPES[role]


def catalog() -> dict[str, Any]:
    """Everything the frontend needs to render pickers, derived from the frozen tables.

    Derived, never hardcoded, so a change to `ROLE_PERMISSIONS` or `TASK_SCOPES`
    shows up in the UI without touching the frontend.
    """
    from policy.permissions import ROLE_PERMISSIONS, TASK_SCOPES, UNGRANTED_TOOLS, task_scope

    return {
        "roles": list(ROLES),
        "task_types": {r: list(role_task_types(r)) for r in ROLES},  # type: ignore[misc]
        "permissions": {r: sorted(ROLE_PERMISSIONS[r]) for r in ROLES},
        "ungranted": sorted(UNGRANTED_TOOLS),
        "scopes": {t: sorted(task_scope(t)) for t in TASK_SCOPES},
        "tools": {
            name: {
                "description": spec.description,
                "parameters": sorted(spec.args_model.model_fields),
                "mutates": spec.mutates,
            }
            for name, spec in sorted(TOOLS.items())
        },
    }


class InteractiveSession:
    """One conversation between a human and one agent, under one control condition.

    The role, control condition and provider are baked into the `EpisodeContext`
    and the `ProviderConfig`, both of which are frozen models, so changing any of
    them means starting a new session. The server does that for you.
    """

    def __init__(
        self,
        *,
        session_id: str,
        role: Role,
        provider_cfg: ProviderConfig,
        control: ControlCondition = "C1",
        drift: DriftCondition = "D0",
        task_type: TaskType | None = None,
        state: SimMartState | None = None,
        seed: int = 7,
        max_turns: int = 12,
    ) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self.session_id = session_id
        self.role = role
        self.control = control
        self.drift = drift
        self.seed = seed
        self.max_turns = max_turns
        self.provider_cfg = provider_cfg
        self.task_type: TaskType = task_type or role_task_types(role)[0]

        self.state = state if state is not None else generate_state(seed)
        self.sink = UIEventSink()
        ctx = EpisodeContext(
            run_id=f"ui-{session_id}",
            episode_id=session_id,
            seed=seed,
            model=provider_cfg.model,
            agent_role=role,
            drift_condition=drift,
            control_condition=control,
        )
        self.pep = PEP(ctx, self.state, HarmOracle(), self.sink)
        self.gateway = self.pep.gateway()
        self.model: Session = make_provider(provider_cfg).new_session(
            system_prompt(role), all_tool_schemas()
        )
        self.transcript: list[dict[str, Any]] = []
        self._messages = 0

    # ------------------------------------------------------------------ driving

    def send(self, text: str) -> Iterator[dict[str, Any]]:
        """Run the agent on one human message, yielding UI events as they happen.

        Each human message is one task, so under C2/C4 the task scope is re-derived
        per message and the log gets a `task_id` per message. Under C1 the task type
        does not affect any decision; it is still recorded because the schema wants
        it and because C2 needs it.
        """
        self._messages += 1
        task_id = f"ui-{self._messages:03d}"
        task = TaskSpec(task_id=task_id, task_type=self.task_type, instruction=text)
        self.pep.begin_task(task)
        usage = {"input": 0, "output": 0, "turns": 0}
        try:
            self._say({"type": "user", "text": text, "task_id": task_id,
                       "task_type": self.task_type})
            self.model.add_user(text)
            yield from self._run_loop(task_id, usage)
        finally:
            self.pep.end_task()

    def _run_loop(self, task_id: str, usage: dict[str, int]) -> Iterator[dict[str, Any]]:
        for _ in range(self.max_turns):
            try:
                turn = self.model.step()
            except ContextOverflow as e:
                self._say({"type": "end", "reason": "context_overflow", "detail": str(e),
                           "usage": usage})
                yield {"type": "end", "reason": "context_overflow", "detail": str(e), "usage": usage}
                return
            usage["turns"] += 1
            usage["input"] += turn.usage.total_input
            usage["output"] += turn.usage.output_tokens
            if turn.text:
                self._say({"type": "assistant", "text": turn.text, "task_id": task_id})
                yield {"type": "assistant", "text": turn.text}

            if not turn.tool_calls:
                reason = {"end_turn": "done", "max_tokens": "max_tokens",
                          "refusal": "refusal"}.get(turn.raw_stop_reason or "", "done")
                self._say({"type": "end", "reason": reason, "usage": usage})
                yield {"type": "end", "reason": reason, "usage": usage}
                return

            outcomes: list[ToolOutcome] = []
            for i, call in enumerate(turn.tool_calls):
                # A turn's tokens are attributed to its first tool call, as in
                # agents/llm_agent.py, so the log's token columns stay comparable.
                t_in = turn.usage.total_input if i == 0 else 0
                t_out = turn.usage.output_tokens if i == 0 else 0
                reply = self.gateway.call(call.name, call.arguments, t_in, t_out,
                                          parse_error=call.parse_error)
                event = self.pep.events[-1]
                payload = {
                    "type": "decision",
                    "task_id": task_id,
                    "call": {"id": call.id, "name": call.name},
                    "event": event.model_dump(mode="json"),
                    "result": reply,
                    "latency_ms": event.latency_ms,
                }
                self._say(payload)
                yield payload
                outcomes.append(ToolOutcome(
                    call_id=call.id,
                    content=json.dumps(reply, sort_keys=True, default=str),
                    is_error=not reply["ok"],
                ))
            self.model.add_tool_results(outcomes)

        self._say({"type": "end", "reason": "turn_limit", "usage": usage})
        yield {"type": "end", "reason": "turn_limit", "usage": usage}

    def _say(self, item: dict[str, Any]) -> None:
        self.transcript.append(item)

    # ----------------------------------------------------------------- reporting

    def log_rows(self) -> list[dict[str, Any]]:
        """The permission log for this session, as the UI's table shows it."""
        return [e.model_dump(mode="json") for e in self.pep.events]

    def summary(self) -> dict[str, Any]:
        """Counts the UI shows in its header. No analysis claims are made from this."""
        events = self.pep.events
        drift: dict[str, int] = {}
        for e in events:
            key = e.drift_type or "none"
            drift[key] = drift.get(key, 0) + 1
        return {
            "session_id": self.session_id,
            "role": self.role,
            "control": self.control,
            "drift_condition": self.drift,
            "task_type": self.task_type,
            "model": self.provider_cfg.model,
            "seed": self.seed,
            "provenance": PROVENANCE,
            "calls": len(events),
            "denied": sum(1 for e in events if e.decision == "deny"),
            "harmful": sum(1 for e in events if e.harm),
            "executed_harmful": sum(1 for e in events if e.harm and e.executed),
            "drift_counts": drift,
            "paused_at": self.pep.paused_at,
        }
