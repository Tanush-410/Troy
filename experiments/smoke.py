"""Milestone 5 smoke test: one D0 / C1 episode per agent role with a real model.

Prints a token and cost estimate and exits unless --yes is given:
    uv run python -m experiments.smoke --model haiku           # estimate only
    uv run python -m experiments.smoke --model haiku --yes     # run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from agents.llm_agent import LLMAgent
from agents.providers.base import ProviderConfig
from agents.providers.factory import make_provider
from analysis.metrics import summarize
from experiments.config import EpisodeConfig
from experiments.cost import estimate, actual_cost
from experiments.provenance import frozen_provenance
from experiments.runner import RunPaths, run_episode
from policy.permissions import Role
from policy.task import TaskSpec
from scenarios import basic
from simmart.generator import generate_state
from simmart.state import SimMartState

SEED = 101
MAX_TURNS = 12
TEMPERATURE = 0.7  # fixed for every model (see docs/decisions.md)
QWEN3_NUM_CTX = 32_768  # qwen3:8b's native window, set explicitly on every request

PRESETS: dict[str, ProviderConfig] = {
    "haiku": ProviderConfig(kind="anthropic", model="claude-haiku-4-5-20251001", context_window=200_000,
                            temperature=TEMPERATURE, api_key_env="ANTHROPIC_API_KEY"),
    "qwen3": ProviderConfig(kind="ollama", model="qwen3:8b", context_window=QWEN3_NUM_CTX,
                            temperature=TEMPERATURE, seed=SEED, base_url="http://localhost:11434",
                            think=False),  # parity: Haiku runs without extended thinking
}


def support_tasks(s: SimMartState) -> list[TaskSpec]:
    return [basic.answer_query("su-1", s, basic.tickets_of_kind(s, "query")[0]),
            basic.process_refund("su-2", s, basic.tickets_of_kind(s, "refund")[0])]


def listing_tasks(s: SimMartState) -> list[TaskSpec]:
    lid, rid = basic.reprice_candidate(s)
    other = next(x for x in sorted(s.listings) if x != lid)
    return [basic.fix_listing_content("ls-1", s, other, "machine washable"),
            basic.reprice_listing("ls-2", s, lid, rid)]


def price_intel_tasks(s: SimMartState) -> list[TaskSpec]:
    skus = [sku for sku in sorted(s.products) if len(basic.carriers(s, sku)) >= 2]
    return [basic.competitor_scan("pi-1", s, skus[0]), basic.history_analysis("pi-2", s, skus[1])]


PLAN: list[tuple[Role, Callable[[SimMartState], list[TaskSpec]]]] = [
    ("support", support_tasks), ("listing", listing_tasks), ("price_intel", price_intel_tasks),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--yes", action="store_true", help="actually run (spends API money for paid models)")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    provider_cfg = PRESETS[args.model]

    est = estimate(provider_cfg.model, [(role, 2, True) for role, _ in PLAN], MAX_TURNS)
    print(f"Smoke test: {provider_cfg.model}, {est.episodes} episodes, {est.tasks} tasks, D0/C1")
    print(f"  estimated model turns: ~{est.model_turns} (limit {MAX_TURNS}/task)")
    print(f"  estimated tokens: ~{est.input_tokens:,} in / ~{est.output_tokens:,} out")
    if est.usd_no_cache is None:
        print("  estimated cost: $0 (local model)")
    else:
        print(f"  estimated cost: ~${est.usd_no_cache:.3f} (no caching); "
              f"upper bound ${est.usd_upper_bound:.3f} (every task hits the turn limit)")
    if not args.yes:
        print("Not run. Re-run with --yes to execute.")
        return

    prov = frozen_provenance()
    if prov["any_dirty"]:
        print("warning: frozen files have uncommitted changes (acceptable for a smoke test)", file=sys.stderr)
    run_id = f"smoke-{args.model}"
    paths = RunPaths.for_run(args.root, run_id)
    provider = make_provider(provider_cfg)
    records = []
    for role, build in PLAN:
        state = generate_state(SEED)
        tasks = build(state)
        cfg = EpisodeConfig(run_id=run_id, episode_id=f"{role}-1", seed=SEED, model=provider_cfg.model,
                            agent_role=role, drift_condition="D0", control_condition="C1",
                            provider=provider_cfg, max_turns_per_task=MAX_TURNS)
        record, _ = run_episode(cfg, tasks, LLMAgent(provider, role, MAX_TURNS), paths, prov, state=state)
        records.append(record)
        print(f"{role}: " + ", ".join(f"{o.task_id}={o.outcome}" for o in record.task_outcomes)
              + f" | steps={record.n_steps} turns={record.model_turns} ends={record.task_ends}"
              + f" | harmful attempted={record.total_harmful_attempted}")

    summary = summarize(paths.events, paths.episodes)
    print(json.dumps(summary, indent=2))
    cost = actual_cost(provider_cfg.model, records)
    tokens_in = sum(r.input_tokens for r in records)
    tokens_out = sum(r.output_tokens for r in records)
    print(f"actual tokens: {tokens_in:,} in / {tokens_out:,} out; "
          + ("cost: $0 (local)" if cost is None else f"cost: ${cost:.4f}"))
    print(f"logs: {paths.logs}  transcripts: {paths.transcripts}")


if __name__ == "__main__":
    main()
