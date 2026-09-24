"""Tool registry, result type, and raw (unenforced) dispatch.

Tools here do no permission checks: that is the PEP's job (milestone 2). They
also do not enforce the business limits the harm oracle checks (refund caps,
price floors, fetch counts), because enforcing them here would make that harm
unmeasurable. They only reject calls that are impossible, such as unknown IDs
or malformed arguments; a rejected call leaves state unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from simmart.state import SimMartState


class ToolResult(BaseModel):
    """Outcome of one tool execution.

    Only `ok`, `data` and `error` are shown to the agent. `tainted` and
    `injection_ids` are metadata for the PEP's taint tracking (harm rule A1).
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    tainted: bool = False
    injection_ids: list[str] = Field(default_factory=list)

    def agent_view(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


class ToolError(Exception):
    """Raised by a tool implementation for an impossible call."""


ToolFn = Callable[[SimMartState, Any], ToolResult]


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn
    mutates: bool

    def json_schema(self) -> dict[str, Any]:
        """OpenAI-style function schema, the format both provider types accept."""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOLS: dict[str, ToolSpec] = {}


def tool(name: str, args_model: type[BaseModel], description: str, mutates: bool) -> Callable[[ToolFn], ToolFn]:
    def register(fn: ToolFn) -> ToolFn:
        if name in TOOLS:
            raise ValueError(f"duplicate tool {name}")
        TOOLS[name] = ToolSpec(
            name=name, description=description, args_model=args_model, fn=fn, mutates=mutates
        )
        return fn

    return register


def all_tool_schemas() -> list[dict[str, Any]]:
    """Schemas for every tool. Every agent sees all of them, whatever its role."""
    return [TOOLS[n].json_schema() for n in sorted(TOOLS)]


def execute(state: SimMartState, name: str, raw_args: dict[str, Any]) -> ToolResult:
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(ok=False, error=f"unknown tool: {name}")
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as e:
        return ToolResult(ok=False, error=f"invalid arguments: {e.errors(include_url=False)}")
    # Validate fully before mutating: implementations raise ToolError only
    # before their first write, so a failed call never leaves partial changes.
    try:
        return spec.fn(state, args)
    except ToolError as e:
        return ToolResult(ok=False, error=str(e))
