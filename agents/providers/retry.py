"""Retrying rate-limited and transient API failures without changing results.

A retry re-sends the identical request (same messages, same parameters), so it
cannot change what the model is asked. Every retry is recorded. When attempts
run out, RetriesExhausted is raised and the run stops; a task is never skipped.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = 10
    base_delay_s: float = 2.0
    max_delay_s: float = 120.0


class RetriesExhausted(RuntimeError):
    pass


class RequestTooLarge(RuntimeError):
    """The request exceeds a per-request token limit; waiting cannot fix it."""


def _retry_after(err: openai.APIStatusError) -> float | None:
    value = err.response.headers.get("retry-after") if err.response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def call_with_retries(
    fn: Callable[[], T], policy: RetryPolicy, log: list[dict[str, Any]],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, retrying 429 / 5xx / connection errors with exponential backoff.

    Each retry appends {attempt, status, wait_s, error} to `log`.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except openai.APIStatusError as e:
            if e.status_code == 413 or "request too large" in str(e).lower():
                raise RequestTooLarge(str(e)) from e
            if e.status_code != 429 and e.status_code < 500:
                raise
            status, hinted, message = e.status_code, _retry_after(e), str(e)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            status, hinted, message = None, None, str(e)
        if attempt == policy.max_attempts:
            raise RetriesExhausted(f"gave up after {attempt} attempts (last status {status})")
        backoff = min(policy.max_delay_s, policy.base_delay_s * 2 ** (attempt - 1))
        wait = max(backoff, hinted or 0.0)
        log.append({"attempt": attempt, "status": status, "wait_s": wait,
                    "retry_after": hinted, "error": message[:300]})
        sleep(wait)
    raise AssertionError("unreachable")
