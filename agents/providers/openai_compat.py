"""Open-weight models through an OpenAI-compatible endpoint (Groq, vLLM).

Not for Ollama: its OpenAI-compatible endpoint ignores num_ctx and truncates
silently; use agents/providers/ollama_provider.py.

The SDK's built-in retries are off; 429/5xx are retried by
agents/providers/retry.py so every retry is logged. A tool call the server
could not parse (Groq returns 400 `tool_use_failed`) becomes a ToolCall with
`parse_error` set, so the PEP logs it as a format error and the model is told.
"""

from __future__ import annotations

import os
import re
from typing import Any

import openai

import json
import time

from agents.providers.retry import call_with_retries
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
_NAME = re.compile(r'(?:"name"\s*:\s*"|<function=)([A-Za-z_][A-Za-z0-9_]*)')


def tool_use_failure(err: openai.BadRequestError) -> str | None:
    """The failed generation text if this 400 is Groq's tool_use_failed, else None."""
    body = err.body if isinstance(err.body, dict) else {}
    body = body.get("error", body) if isinstance(body.get("error"), dict) else body
    if body.get("code") != "tool_use_failed":
        return None
    return str(body.get("failed_generation") or "")


class OpenAICompatSession:
    def __init__(self, client: Any, config: ProviderConfig, system: str, tools: list[dict[str, Any]]) -> None:
        self._client = client
        self._config = config
        self._tools = tools  # already OpenAI function schemas
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._guard = ContextGuard(config.context_window, config.max_tokens)
        self._failed_ids: set[str] = set()
        self._n_failed = 0

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
        if self._config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._config.reasoning_effort
        retries: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        try:
            response = call_with_retries(lambda: self._client.chat.completions.create(
                model=self._config.model,
                messages=self.messages,
                tools=self._tools,
                max_tokens=self._config.max_tokens,
                **kwargs,
            ), self._config.retry, retries)
        except openai.BadRequestError as e:
            failed = tool_use_failure(e)
            if failed is None:
                raise
            return self._failed_tool_call(failed, retries)
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
        details = getattr(u, "prompt_tokens_details", None) if u else None
        cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
        usage = Usage(input_tokens=(u.prompt_tokens - cached) if u else 0, cache_read_tokens=cached,
                      output_tokens=u.completion_tokens if u else 0)
        self._guard.record(usage.total_input, usage.output_tokens)
        return ModelTurn(
            text=_THINK.sub("", msg.content or "").strip(),
            tool_calls=calls,
            stop_reason="tool_use" if calls else _STOP.get(choice.finish_reason or "", "other"),
            raw_stop_reason=choice.finish_reason,
            usage=usage,
            retries=retries,
            latency_s=time.perf_counter() - t0 - sum(r["wait_s"] for r in retries),
        )

    def _failed_tool_call(self, failed: str, retries: list[dict[str, Any]]) -> ModelTurn:
        # The server returned no usage for the failed generation, so tokens are unknown (0).
        self._n_failed += 1
        call_id = f"failed_{self._n_failed}"
        self._failed_ids.add(call_id)
        self.messages.append({"role": "assistant", "content": failed})
        self._guard.add(failed)
        m = _NAME.search(failed)
        call = ToolCall(id=call_id, name=m.group(1) if m else "", arguments=failed,
                        parse_error="server returned tool_use_failed")
        return ModelTurn(text="", tool_calls=[call], stop_reason="tool_use", raw_stop_reason="tool_use_failed",
                         usage=Usage(), retries=retries)

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        for r in results:
            if r.call_id in self._failed_ids:
                # No tool_call id exists on the server side; tell the model in a user turn.
                self.messages.append({"role": "user", "content": f"Your tool call failed: {r.content}"})
            else:
                self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
            self._guard.add(r.content)


class OpenAICompatProvider:
    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            # Ollama ignores the key but the SDK requires one.
            key = os.environ.get(config.api_key_env, "") if config.api_key_env else "unused"
            client = openai.OpenAI(base_url=config.base_url, api_key=key, max_retries=0)
        self._client = client

    def new_session(self, system: str, tools: list[dict[str, Any]]) -> OpenAICompatSession:
        return OpenAICompatSession(self._client, self.config, system, tools)

