"""Token and cost estimates before a run, and actual cost after one.

The estimate is a heuristic (about 4 characters per token for prompts and JSON
schemas); its assumptions are printed with it. Actual cost is computed from the
usage recorded in episode records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agents.prompts import system_prompt
from policy.log_schema import EpisodeRecord
from policy.permissions import Role
from tools import all_tool_schemas

CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class Price:
    input: float  # USD per million tokens
    output: float
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.10


PRICES: dict[str, Price] = {
    "claude-haiku-4-5-20251001": Price(input=1.00, output=5.00),
    "claude-haiku-4-5": Price(input=1.00, output=5.00),
    # Groq, from console.groq.com/docs/model/openai/gpt-oss-120b (2026-09-24);
    # prompt caching is automatic with a 50% discount on cached input.
    "openai/gpt-oss-120b": Price(input=0.15, output=0.60, cache_write_mult=1.0, cache_read_mult=0.5),
}



@dataclass(frozen=True)
class Assumptions:
    turns_per_task: int = 6  # model turns per task, typical
    tokens_added_per_turn: int = 350  # tool call + tool result appended to history
    output_tokens_per_turn: int = 150


@dataclass(frozen=True)
class Estimate:
    episodes: int
    tasks: int
    model_turns: int
    input_tokens: int
    output_tokens: int
    usd_no_cache: float | None  # None for unpriced (local) models
    usd_upper_bound: float | None  # every task hits the turn limit, no caching


def prefix_tokens(role: Role) -> int:
    text = system_prompt(role) + json.dumps(all_tool_schemas())
    return round(len(text) / CHARS_PER_TOKEN)


def _episode_tokens(role: Role, n_tasks: int, turns: int, fresh_context_per_task: bool,
                    a: Assumptions) -> tuple[int, int]:
    prefix = prefix_tokens(role)
    inp = out = 0
    history = 0
    for _ in range(n_tasks):
        if fresh_context_per_task:
            history = 0
        history += 100  # the task instruction
        for _ in range(turns):
            inp += prefix + history
            out += a.output_tokens_per_turn
            history += a.tokens_added_per_turn
    return inp, out


def estimate(model: str, plan: list[tuple[Role, int, bool]], max_turns: int,
             a: Assumptions = Assumptions()) -> Estimate:
    """`plan` is one (role, n_tasks, fresh_context_per_task) per episode."""
    inp = out = ub_in = ub_out = 0
    for role, n, fresh in plan:
        i, o = _episode_tokens(role, n, a.turns_per_task, fresh, a)
        inp, out = inp + i, out + o
        i, o = _episode_tokens(role, n, max_turns, fresh, a)
        ub_in, ub_out = ub_in + i, ub_out + o
    price = PRICES.get(model)

    def usd(i: int, o: int) -> float | None:
        return None if price is None else (i * price.input + o * price.output) / 1e6

    return Estimate(
        episodes=len(plan), tasks=sum(n for _, n, _ in plan), model_turns=sum(n for _, n, _ in plan) * a.turns_per_task,
        input_tokens=inp, output_tokens=out, usd_no_cache=usd(inp, out), usd_upper_bound=usd(ub_in, ub_out),
    )


def peak_context(role: Role, n_tasks: int, turns_per_task: int, a: Assumptions = Assumptions()) -> int:
    """Largest prompt in an episode whose context persists across all tasks (D1)."""
    return prefix_tokens(role) + n_tasks * (100 + turns_per_task * a.tokens_added_per_turn)


def max_tasks_that_fit(role: Role, window: int, max_tokens: int, turns_per_task: int,
                       a: Assumptions = Assumptions()) -> int:
    per_task = 100 + turns_per_task * a.tokens_added_per_turn
    return max(0, (window - max_tokens - prefix_tokens(role)) // per_task)


def actual_cost(model: str, records: list[EpisodeRecord]) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    total = 0.0
    for r in records:
        uncached = r.input_tokens - r.cache_read_tokens - r.cache_write_tokens
        total += (uncached * price.input
                  + r.cache_write_tokens * price.input * price.cache_write_mult
                  + r.cache_read_tokens * price.input * price.cache_read_mult
                  + r.output_tokens * price.output) / 1e6
    return total
