"""Provider-neutral types for one model turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agents.providers.retry import RetryPolicy

StopReason = Literal["tool_use", "end_turn", "max_tokens", "refusal", "other"]


class ProviderConfig(BaseModel):
    """Everything needed to build a provider. Secrets come from the environment only.

    `context_window` is the hard limit the agent loop must never exceed; for
    Ollama it is also sent as `num_ctx` on every request, because the server's
    default window is small and it truncates silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["anthropic", "openai_compat", "ollama"]  # Groq is openai_compat
    model: str
    context_window: int
    max_tokens: int = 4096  # per model turn
    temperature: float | None = None  # None = the provider's default
    seed: int | None = None  # honoured by ollama and some openai_compat backends
    base_url: str | None = None  # openai_compat / ollama server URL
    api_key_env: str | None = None  # env var holding the key; None = SDK default
    think: bool | None = None  # ollama: reasoning mode for models that have one (qwen3)
    reasoning_effort: str | None = None  # openai_compat: for reasoning models that take it (gpt-oss)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # A dict when the provider parsed the arguments, the raw text otherwise;
    # the PEP parses and validates either form and logs failures as format errors.
    arguments: dict[str, Any] | str
    # Set when the provider or server already failed to parse the call (e.g.
    # Groq's tool_use_failed); the PEP then logs a format error directly.
    parse_error: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage
    raw_stop_reason: str | None = None
    retries: list[dict[str, Any]] = field(default_factory=list)  # one entry per retried attempt


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    content: str
    is_error: bool


class ContextOverflow(Exception):
    """The next request would not fit in the context window (or the server truncated one)."""


class Session(Protocol):
    """One conversation (context window) with a model.

    `step` raises ContextOverflow rather than send a request that would not fit.
    """

    def add_user(self, text: str) -> None: ...

    def step(self) -> ModelTurn: ...

    def add_tool_results(self, results: list[ToolOutcome]) -> None: ...


class Provider(Protocol):
    config: ProviderConfig

    def new_session(self, system: str, tools: list[dict[str, Any]]) -> Session:
        """`tools` are OpenAI-style function schemas (tools.all_tool_schemas())."""
        ...


@dataclass
class UsageTotals:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reasons: dict[str, int] = field(default_factory=dict)

    def add(self, turn: ModelTurn) -> None:
        self.turns += 1
        self.input_tokens += turn.usage.input_tokens
        self.output_tokens += turn.usage.output_tokens
        self.cache_read_tokens += turn.usage.cache_read_tokens
        self.cache_write_tokens += turn.usage.cache_write_tokens
        self.stop_reasons[turn.stop_reason] = self.stop_reasons.get(turn.stop_reason, 0) + 1


class ContextGuard:
    """Refuses a request that might not fit, instead of letting a server truncate it.

    The server's reported prompt size for the previous turn is exact; only the
    content appended since then is estimated, conservatively (3 chars/token).
    A prompt that shrinks between turns means the server truncated it anyway.
    """

    CHARS_PER_TOKEN = 3.0

    def __init__(self, window: int, max_tokens: int) -> None:
        self.window = window
        self.max_tokens = max_tokens
        self.last_prompt: int | None = None
        self.last_output = 0
        self.pending_chars = 0

    def add(self, text: str) -> None:
        self.pending_chars += len(text)

    def check(self, full_prompt_chars: int) -> None:
        if self.last_prompt is None:
            est = full_prompt_chars / self.CHARS_PER_TOKEN
        else:
            est = self.last_prompt + self.last_output + self.pending_chars / self.CHARS_PER_TOKEN
        if est + self.max_tokens > self.window:
            raise ContextOverflow(
                f"next request ~{est:.0f} tokens + max_tokens {self.max_tokens} > window {self.window}")

    def record(self, prompt_tokens: int, output_tokens: int) -> None:
        if self.last_prompt is not None and prompt_tokens < self.last_prompt:
            raise ContextOverflow(f"prompt shrank from {self.last_prompt} to {prompt_tokens}: server truncated")
        self.last_prompt, self.last_output, self.pending_chars = prompt_tokens, output_tokens, 0
