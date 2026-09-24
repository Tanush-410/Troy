"""Ollama via its native /api/chat endpoint.

The OpenAI-compatible endpoint ignores num_ctx and the server default window
is about 2k tokens, which silently truncates our prompts (measured on Ollama
0.33.1). The native endpoint honours num_ctx, seed and temperature per request.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any

from agents.providers.base import (
    ContextGuard,
    ModelTurn,
    ProviderConfig,
    ToolCall,
    ToolOutcome,
    Usage,
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Tool calls the server failed to parse stay in the text as <tool_call> blocks.
_TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

DEFAULT_URL = "http://localhost:11434"


def _post(url: str, body: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def unparsed_tool_calls(text: str) -> list[tuple[str, str]]:
    """(name or "", raw body) for <tool_call> blocks left unparsed in the text."""
    out = []
    for body in _TOOL_CALL.findall(text):
        m = _NAME.search(body)
        out.append((m.group(1) if m else "", body.strip()))
    return out


class OllamaSession:
    def __init__(self, post: Any, config: ProviderConfig, system: str, tools: list[dict[str, Any]]) -> None:
        self._post = post
        self._config = config
        self._tools = tools
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._guard = ContextGuard(config.context_window, config.max_tokens)
        self._n = 0

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._guard.add(text)

    def step(self) -> ModelTurn:
        self._guard.check(len(json.dumps(self.messages)) + len(json.dumps(self._tools)))
        options: dict[str, Any] = {"num_ctx": self._config.context_window, "num_predict": self._config.max_tokens}
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.seed is not None:
            options["seed"] = self._config.seed
        t0 = time.perf_counter()
        r = self._post(f"{self._config.base_url or DEFAULT_URL}/api/chat", {
            "model": self._config.model, "messages": self.messages, "tools": self._tools,
            "stream": False, "options": options,
            **({"think": self._config.think} if self._config.think is not None else {}),
        })
        prompt, output = r.get("prompt_eval_count", 0), r.get("eval_count", 0)
        self._guard.record(prompt, output)
        msg = r["message"]
        content = msg.get("content") or ""
        self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})

        calls = []
        for c in msg.get("tool_calls") or []:
            self._n += 1
            fn = c["function"]
            calls.append(ToolCall(id=f"call_{self._n}", name=fn["name"], arguments=fn.get("arguments", {})))
        for name, raw in unparsed_tool_calls(content):
            # The server left this call unparsed; the PEP logs it as a format error.
            self._n += 1
            calls.append(ToolCall(id=f"call_{self._n}", name=name, arguments=raw,
                                  parse_error="server did not parse <tool_call> block"))
        text = _TOOL_CALL.sub("", _THINK.sub("", content)).strip()
        done = r.get("done_reason")
        return ModelTurn(
            text=text, tool_calls=calls, raw_stop_reason=done,
            stop_reason="tool_use" if calls else ("max_tokens" if done == "length" else "end_turn"),
            usage=Usage(input_tokens=prompt, output_tokens=output),
            latency_s=time.perf_counter() - t0,
        )

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        # The native API matches tool results to calls by order.
        for r in results:
            self.messages.append({"role": "tool", "content": r.content})
            self._guard.add(r.content)


class OllamaProvider:
    def __init__(self, config: ProviderConfig, post: Any = _post) -> None:
        self.config = config
        self._post = post

    def new_session(self, system: str, tools: list[dict[str, Any]]) -> OllamaSession:
        return OllamaSession(self._post, self.config, system, tools)
