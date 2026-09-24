"""The LLM agent loop, provider-neutral.

Each model turn's tool calls go through the AgentGateway (so the PEP decides,
judges and logs every one), and the agent-visible replies go back to the model
as tool results. A task ends when the model answers without a tool call, stops
for max_tokens or refusal, hits the per-task turn limit, or the next request
would overflow the context window (never truncated silently).
"""

from __future__ import annotations

import json
from dataclasses import asdict

from agents.base import TaskRun
from agents.prompts import system_prompt
from agents.providers.base import ContextOverflow, Provider, Session, ToolOutcome
from policy.pep import AgentGateway
from policy.permissions import Role
from policy.task import AgentTask
from tools import all_tool_schemas

END_REASONS = {"end_turn": "done", "max_tokens": "max_tokens", "refusal": "refusal"}


class LLMAgent:
    def __init__(self, provider: Provider, role: Role, max_turns_per_task: int = 20) -> None:
        self._provider = provider
        self._system = system_prompt(role)
        self._tools = all_tool_schemas()  # every tool, whatever the role
        self._max_turns = max_turns_per_task
        self._session: Session | None = None
        self._fresh = False

    def reset(self) -> None:
        self._session = self._provider.new_session(self._system, self._tools)
        self._fresh = True

    def run_task(self, task: AgentTask, gateway: AgentGateway) -> TaskRun:
        if self._session is None:
            self.reset()
        assert self._session is not None
        run = TaskRun()
        if self._fresh:
            run.transcript.append({"role": "system", "content": self._system})
            self._fresh = False
        run.transcript.append({"role": "user", "task_id": task.task_id, "content": task.instruction})
        self._session.add_user(task.instruction)

        for _ in range(self._max_turns):
            try:
                turn = self._session.step()
            except ContextOverflow as e:
                run.end = "context_overflow"
                run.transcript.append({"role": "harness", "task_id": task.task_id, "context_overflow": str(e)})
                return run
            run.model_turns += 1
            run.input_tokens += turn.usage.total_input
            run.output_tokens += turn.usage.output_tokens
            run.cache_read_tokens += turn.usage.cache_read_tokens
            run.cache_write_tokens += turn.usage.cache_write_tokens
            run.transcript.append({
                "role": "assistant", "task_id": task.task_id, "text": turn.text,
                "tool_calls": [asdict(c) for c in turn.tool_calls],
                "stop_reason": turn.raw_stop_reason, "usage": asdict(turn.usage),
            })
            if not turn.tool_calls:
                run.end = END_REASONS.get(turn.stop_reason, "done")
                return run

            outcomes = []
            for i, call in enumerate(turn.tool_calls):
                # A turn's tokens are attributed to its first tool call.
                t_in, t_out = (turn.usage.total_input, turn.usage.output_tokens) if i == 0 else (0, 0)
                reply = gateway.call(call.name, call.arguments, t_in, t_out)
                content = json.dumps(reply, sort_keys=True, default=str)
                outcomes.append(ToolOutcome(call_id=call.id, content=content, is_error=not reply["ok"]))
                run.transcript.append({"role": "tool", "task_id": task.task_id, "call_id": call.id,
                                       "content": reply})
            self._session.add_tool_results(outcomes)

        run.end = "step_limit"
        return run
