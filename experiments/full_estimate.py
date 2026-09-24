"""Full-experiment token, cost and time estimate from measured smoke-test logs.

Everything per-task is measured from a smoke run (per-turn usage in the
transcripts, wall-clock from event timestamps); the experiment design and the
rate limits are stated assumptions, printed with the result.
    uv run python -m experiments.full_estimate logs/<groq run> logs/<qwen run>
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments.cost import PRICES

ROLES = 3
LIVE_CONTROLS = 2  # C1, C2 (C3/C4 are offline replays)
NON_D1_DRIFTS = 3  # D0, D2, D3
TASKS_NON_D1 = 5  # assumption until milestone 6 fixes episode length
TASKS_D1 = 10
INSTRUCTION_TOKENS = 100  # approx. tokens a task instruction adds to a D1 context


@dataclass
class Measured:
    model: str
    tasks: int
    turns_per_task: float
    input_per_task: float  # all input tokens over a fresh-context task
    output_per_task: float
    cached_share: float  # cached / all input
    growth_per_task: float  # tokens a finished task leaves in the context
    first_prompt: float  # prefix + instruction on a fresh context
    max_prompt: int
    sec_per_turn: float
    retries: int
    retry_wait_s: float


def measure(run_dir: Path) -> Measured:
    transcripts = run_dir.parent.parent / "transcripts" / run_dir.name
    per_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in sorted(transcripts.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get("role") == "assistant":
                per_task[(f.stem, e["task_id"])].append(e["usage"])
    turns, inp, out, cached, growth, first, prompts = [], [], [], [], [], [], []
    for usages in per_task.values():
        totals = [u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"] for u in usages]
        turns.append(len(usages))
        inp.append(sum(totals))
        out.append(sum(u["output_tokens"] for u in usages))
        cached.append(sum(u["cache_read_tokens"] for u in usages))
        first.append(totals[0])
        growth.append(totals[-1] + usages[-1]["output_tokens"] - totals[0] + INSTRUCTION_TOKENS)
        prompts.extend(totals)

    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    latencies = _turn_latencies(run_dir, transcripts)
    return Measured(
        model=episodes[0]["model"], tasks=len(per_task),
        turns_per_task=statistics.mean(turns), input_per_task=statistics.mean(inp),
        output_per_task=statistics.mean(out), cached_share=sum(cached) / sum(inp) if sum(inp) else 0.0,
        growth_per_task=statistics.mean(growth), first_prompt=statistics.mean(first), max_prompt=max(prompts),
        sec_per_turn=statistics.mean(latencies) if latencies else 0.0,  # totals need the mean
        retries=sum(ep["api_retries"] for ep in episodes),
        retry_wait_s=sum(ep["api_retry_wait_s"] for ep in episodes),
    )


def _turn_latencies(run_dir: Path, transcripts: Path) -> list[float]:
    """Model latency per turn, excluding rate-limit waits.

    Uses the recorded `latency_s` when present. Otherwise (older logs) it is the
    gap between the last event of one tool-calling turn and the first event of
    the next tool-calling turn in the same task, minus the later turn's retry waits.
    """
    events: dict[str, list[datetime]] = defaultdict(list)
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        events[e["episode_id"]].append(datetime.fromisoformat(e["timestamp"]))
    out: list[float] = []
    for f in sorted(transcripts.glob("*.jsonl")):
        turns = [json.loads(line) for line in f.read_text().splitlines()]
        turns = [t for t in turns if t.get("role") == "assistant"]
        recorded = [t["latency_s"] for t in turns if t.get("latency_s") is not None]
        if recorded:
            out.extend(recorded)
            continue
        ts = iter(events.get(f.stem, []))
        prev_last, prev_task = None, None
        for t in turns:
            n = len(t["tool_calls"])
            stamps = [next(ts) for _ in range(n)]
            wait = sum(r["wait_s"] for r in t.get("retries", []))
            if n and prev_last is not None and t["task_id"] == prev_task:
                out.append((stamps[0] - prev_last).total_seconds() - wait)
            prev_last, prev_task = (stamps[-1], t["task_id"]) if n else (None, None)
    return [x for x in out if x >= 0]


@dataclass
class Plan:
    input_tokens: float
    output_tokens: float
    cached_tokens: float
    requests: float
    d1_peak_prompt: float


def plan(m: Measured, episodes_per_cell: int) -> Plan:
    n_other = ROLES * NON_D1_DRIFTS * LIVE_CONTROLS * episodes_per_cell
    n_d1 = ROLES * LIVE_CONTROLS * episodes_per_cell
    other_in = n_other * TASKS_NON_D1 * m.input_per_task
    # D1: task k also re-reads the k earlier tasks' context on each of its turns.
    pairs = TASKS_D1 * (TASKS_D1 - 1) / 2
    d1_in = n_d1 * (TASKS_D1 * m.input_per_task + m.turns_per_task * m.growth_per_task * pairs)
    tasks = n_other * TASKS_NON_D1 + n_d1 * TASKS_D1
    inp = other_in + d1_in
    return Plan(
        input_tokens=inp, output_tokens=tasks * m.output_per_task, cached_tokens=inp * m.cached_share,
        requests=tasks * m.turns_per_task,
        d1_peak_prompt=m.first_prompt + (TASKS_D1 - 1) * m.growth_per_task
        + (m.max_prompt - m.first_prompt),
    )


def dollars(model: str, p: Plan) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    uncached = p.input_tokens - p.cached_tokens
    return (uncached * price.input + p.cached_tokens * price.input * price.cache_read_mult
            + p.output_tokens * price.output) / 1e6


def days_rate_limited(p: Plan, tpm: float, rpd: float | None, tpd: float | None) -> float:
    tokens = p.input_tokens + p.output_tokens
    limits = [tokens / tpm / 60 / 24]
    if rpd:
        limits.append(p.requests / rpd)
    if tpd:
        limits.append(tokens / tpd)
    return max(limits)


def report(groq_dir: Path, qwen_dir: Path) -> None:
    g, q = measure(groq_dir), measure(qwen_dir)
    for m in (g, q):
        print(f"measured {m.model} over {m.tasks} tasks: {m.turns_per_task:.1f} turns/task, "
              f"{m.input_per_task:,.0f} in + {m.output_per_task:,.0f} out tokens/task, "
              f"cached {m.cached_share:.0%}, context growth {m.growth_per_task:,.0f}/task, "
              f"max prompt {m.max_prompt:,}, {m.sec_per_turn:.1f} s/turn, retries {m.retries}")
    print(f"\nassumed design: {ROLES} roles x (3 drifts x {TASKS_NON_D1} tasks + D1 x {TASKS_D1} tasks) "
          f"x {LIVE_CONTROLS} live controls, per model")
    for n in (3, 30, 20):
        label = "PILOT" if n == 3 else "FULL"
        print(f"\n=== {label}: {n} episodes per cell ({ROLES * 4 * LIVE_CONTROLS * n} episodes per model) ===")
        gp, qp = plan(g, n), plan(q, n)
        cost = dollars(g.model, gp)
        print(f"{g.model}: {gp.input_tokens / 1e6:,.1f}M in ({gp.cached_tokens / 1e6:,.1f}M cached) + "
              f"{gp.output_tokens / 1e6:,.2f}M out, {gp.requests:,.0f} requests, "
              f"D1 peak prompt ~{gp.d1_peak_prompt:,.0f} tokens")
        print(f"  cost (paid Developer tier): ${cost:,.2f}" if cost is not None else "  cost: unpriced")
        free = days_rate_limited(gp, tpm=8_000, rpd=1_000, tpd=200_000)
        too_big = [x for x, v in (("D1", gp.d1_peak_prompt), ("non-D1", g.max_prompt)) if v > 8_000]
        print(f"  free tier (8K TPM, 1K RPD, 200K TPD): {free:,.1f} days of rate-limited time"
              + (f"; INFEASIBLE for {', '.join(too_big)}: a single request exceeds 8K TPM" if too_big else ""))
        for tpm, parallel in ((250_000, 8), (1_000_000, 16)):
            limit_days = days_rate_limited(gp, tpm=tpm, rpd=None, tpd=None)
            latency_days = gp.requests * g.sec_per_turn / parallel / 86_400
            print(f"  paid, assuming {tpm / 1000:,.0f}K TPM and {parallel} episodes in parallel: "
                  f"~{max(limit_days, latency_days) * 24:,.1f} hours")
        local_days = qp.requests * q.sec_per_turn / 86_400
        print(f"{q.model}: {qp.input_tokens / 1e6:,.1f}M in + {qp.output_tokens / 1e6:,.2f}M out, "
              f"{qp.requests:,.0f} requests, D1 peak prompt ~{qp.d1_peak_prompt:,.0f} tokens "
              f"({'fits' if qp.d1_peak_prompt < 32_768 - 4_096 else 'DOES NOT FIT'} in num_ctx 32,768)")
        print(f"  cost $0 (local); time at the measured {q.sec_per_turn:.1f} s/turn, one episode at a time: "
              f"~{local_days:,.1f} days")


if __name__ == "__main__":
    report(Path(sys.argv[1]), Path(sys.argv[2]))
