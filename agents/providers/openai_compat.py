"""Open-weight models through an OpenAI-compatible endpoint (e.g. vLLM).

Not for Ollama: its OpenAI-compatible endpoint ignores num_ctx and truncates
silently; use agents/providers/ollama_provider.py.
"""

from __future__ import annotations

import os
import re
from typing import Any

import openai

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

_STOP: dict[str, StopReason] = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatSession:
    def __init__(self, client: Any, config: ProviderConfig, system: str, tools: list[dict[str, Any]]) -> None:
        self._client = client
        self._config = config
        self._tools = tools  # already OpenAI function schemas
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._guard = ContextGuard(config.context_window, config.max_tokens)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._guard.add(text)

    def step(self) -> ModelTurn:
        self._guard.check(len(json.dumps(self.messages)) + len(json.dumps(self._tools)))
        kwargs: dict[str, Any] = {}
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.seed is not None:
            kwargs["seed"] = self._config.seed
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=self.messages,
            tools=self._tools,
            max_tokens=self._config.max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        msg = choice.message
        raw_calls = msg.tool_calls or []
        # Keep the assistant turn exactly as returned so call ids line up.
        self.messages.append({
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in raw_calls
            ]} if raw_calls else {}),
        })
        calls = [
            # Raw argument text: the PEP parses it, so malformed JSON is logged as a format error.
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "")
            for c in raw_calls
        ]
        u = response.usage
        usage = Usage(input_tokens=u.prompt_tokens if u else 0, output_tokens=u.completion_tokens if u else 0)
        self._guard.record(usage.input_tokens, usage.output_tokens)
        return ModelTurn(
            text=_THINK.sub("", msg.content or "").strip(),
            tool_calls=calls,
            stop_reason="tool_use" if calls else _STOP.get(choice.finish_reason or "", "other"),
            raw_stop_reason=choice.finish_reason,
            usage=usage,
        )

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
            self._guard.add(r.content)


class OpenAICompatProvider:
    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            # Ollama ignores the key but the SDK requires one.
            key = os.environ.get(config.api_key_env, "") if config.api_key_env else "unused"
            client = openai.OpenAI(base_url=config.base_url, api_key=key)
        self._client = client

    def new_session(self, system: str, tools: list[dict[str, Any]]) -> OpenAICompatSession:
        return OpenAICompatSession(self._client, self.config, system, tools)

