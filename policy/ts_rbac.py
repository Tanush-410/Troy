"""Task-scoped RBAC (C2/C4).

A scope grants S_tau for one task and expires at task end. Between tasks there
is no scope, so only the always-allowed actions pass. An in-role action outside
S_tau may be granted for the rest of the task only if the orchestrator's
expansion policy approves it; an out-of-role action can never be granted.
"""

from __future__ import annotations

from typing import Protocol

from policy.permissions import ALWAYS_ALLOWED, Role, task_scope
from policy.rbac import in_role
from policy.task import TaskSpec


class ExpansionPolicy(Protocol):
    def approve(self, task: TaskSpec, action: str) -> bool: ...


class DenyAllExpansions:
    """Default orchestrator: never widens a task scope."""

    def approve(self, task: TaskSpec, action: str) -> bool:
        return False


class TaskScopedRBAC:
    def __init__(self, role: Role, expansion_policy: ExpansionPolicy | None = None) -> None:
        self.role = role
        self.expansion_policy = expansion_policy or DenyAllExpansions()
        self.task: TaskSpec | None = None
        self._granted: frozenset[str] = ALWAYS_ALLOWED

    def begin_task(self, task: TaskSpec) -> None:
        if task.role != self.role:
            raise ValueError(f"task {task.task_id} ({task.task_type}) is not a {self.role} task")
        self.task = task
        self._granted = task_scope(task.task_type)

    def end_task(self) -> None:
        self.task = None
        self._granted = ALWAYS_ALLOWED

    def allows(self, action: str) -> bool:
        return action in self._granted

    def request_expansion(self, action: str) -> bool:
        """Ask the orchestrator to add an in-role action to the current task scope."""
        if self.task is None or not in_role(self.role, action):
            return False
        if self.expansion_policy.approve(self.task, action):
            self._granted = self._granted | {action}
            return True
        return False
