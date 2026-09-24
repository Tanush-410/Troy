"""Task specification handed to an agent by the orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from policy.permissions import TASK_ROLE, Role, TaskType


class TaskSpec(BaseModel):
    """One task. `*_ids` / `skus` name the entities the task is about; the harm
    oracle uses them (e.g. L2: editing a listing not named in the task)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task_type: TaskType
    instruction: str
    ticket_id: str | None = None
    listing_ids: tuple[str, ...] = ()
    skus: tuple[str, ...] = ()
    report_ids: tuple[str, ...] = ()
    # Injections carried by the task prompt itself (e.g. supplier text in a
    # create_listing task under D3). Empty means the prompt is clean.
    prompt_injection_ids: tuple[str, ...] = Field(default=())

    @property
    def role(self) -> Role:
        return TASK_ROLE[self.task_type]
