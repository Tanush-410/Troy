"""Task specification handed to an agent by the orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from policy.permissions import TASK_ROLE, Role, TaskType


class AgentTask(BaseModel):
    """The only part of a task the agent sees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    instruction: str


class TaskSpec(BaseModel):
    """One task. `*_ids` / `skus` name the entities the task is about; the harm
    oracle uses them (e.g. L2: editing a listing not named in the task).

    The goal fields (`expected_fact` ... `fix_fragment`) and
    `escalation_acceptable` are for the success checkers only and are never
    shown to the agent; the instruction must state whatever the agent needs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task_type: TaskType
    instruction: str
    ticket_id: str | None = None
    order_id: str | None = None
    listing_ids: tuple[str, ...] = ()
    skus: tuple[str, ...] = ()
    report_ids: tuple[str, ...] = ()
    # Injections carried by the task prompt itself (e.g. supplier text in a
    # create_listing task under D3). Empty means the prompt is clean.
    prompt_injection_ids: tuple[str, ...] = Field(default=())

    # --- success-checker inputs (hidden from the agent)
    # True for D2 tasks that conflict with role guidelines and D3 tasks whose
    # injection makes the request unsafe: escalating then counts as success.
    escalation_acceptable: bool = False
    expected_fact: str | None = None  # answer_query
    refund_amount: float | None = None  # process_refund
    new_address: str | None = None  # update_address
    fix_fragment: str | None = None  # fix_listing_content

    @property
    def role(self) -> Role:
        return TASK_ROLE[self.task_type]

    def agent_view(self) -> AgentTask:
        return AgentTask(task_id=self.task_id, instruction=self.instruction)
