"""Claude via the official Anthropic SDK (manual tool-use loop)."""

from __future__ import annotations

import os
from typing import Any

import anthropic

import json

from agents.providers.base import (
    ContextGuard,
    ModelTurn,
    ProviderConfig,
    StopReason,
    ToolCall,
    ToolOutcome,
    Usage,
)

_STOP: dict[str, StopReason] = {
    "tool_use": "tool_use", "end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "refusal",
}


def to_anthropic_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": s["function"]["name"], "description": s["function"]["description"],
         "input_schema": s["function"]["parameters"]}
        for s in schemas
    ]


class AnthropicSession:
    def __init__(self, client: Any, config: ProviderConfig, system: str, tools: list[dict[str, Any]]) -> None:
        self._client = client
        self._config = config
        self._system = system
        self._tools = to_anthropic_tools(tools)
        self.messages: list[dict[str, Any]] = []
        self._guard = ContextGuard(config.context_window, config.max_tokens)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._guard.add(text)

    def step(self) -> ModelTurn:
        self._guard.check(len(self._system) + len(json.dumps(self._tools))
                          + len(json.dumps(self.messages, default=str)))
        extra: dict[str, Any] = {}
        if self._config.temperature is not None:
            # Removed from SDK 1.x signatures; Haiku 4.5 still honours it.
            extra["extra_body"] = {"temperature": self._config.temperature}
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=self._system,
            tools=self._tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},  # system + tools + history prefix is stable
            **extra,
        )
        # Append the full content so tool_use ids line up with our tool_results.
        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(id=b.id, name=b.name, arguments=b.input) for b in response.content if b.type == "tool_use"]
        text = "\n".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
        )
        self._guard.record(usage.total_input, usage.output_tokens)
        return ModelTurn(
            text=text,
            tool_calls=calls,
            stop_reason=_STOP.get(response.stop_reason or "", "other"),
            raw_stop_reason=response.stop_reason,
            usage=usage,
        )

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        # All results for one turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]})
        for r in results:
            self._guard.add(r.content)


class AnthropicProvider:
    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            key = os.environ[config.api_key_env] if config.api_key_env else None
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self._client = client

    def new_session(self, system: str, tools: list[dict[str, Any]]) -> AnthropicSession:
        return AnthropicSession(self._client, self.config, system, tools)
