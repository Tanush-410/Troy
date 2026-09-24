"""Resumable experiment runner (pilot and full run).

    uv run python -m experiments.pilot --model qwen3 --episodes-per-cell 3 --run-id pilot-qwen3 --yes
    uv run python -m experiments.pilot --model groq --episodes-per-cell 3 --run-id pilot-groq --max-usd 2 --yes

Design:
- Cells: 3 roles x D0-D3 x live controls C1, C2; `episodes_per_cell` episodes each.
- Paired seeds: episode k of (role, drift) gets the same seed, state and tasks under
  C1 and C2 and for every model, so comparisons differ only in the control/model.
- Resumable: each episode runs in logs/<run>/tmp/<episode_id>/ and is merged into
  the run's events/episodes/retries files only when complete. On restart,
  finished episodes are skipped and partial ones are discarded and rerun.
- Frozen: refuses to start if any frozen file differs from its last commit, and
  records the commit hash and frozen-file provenance in run_meta.json and on
  every episode record.
- Groq: parallel episodes are chosen from the account's tokens-per-minute limit
  (read from response headers at start), and --max-usd stops the run between
  episodes before the budget would be exceeded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from agents.llm_agent import LLMAgent
from agents.providers.base import ProviderConfig
from agents.providers.factory import make_provider
from experiments.config import EpisodeConfig
from experiments.cost import actual_cost
from experiments.provenance import frozen_provenance
from experiments.runner import RunPaths, run_episode
from experiments.smoke import MAX_TURNS, PRESETS, groq_config
from policy.log_schema import EpisodeRecord
from scenarios.generator import ScenarioConfig, build_episode
from simmart.generator import generate_state

ROLES = ("support", "listing", "price_intel")
DRIFTS = ("D0", "D1", "D2", "D3")
CONTROLS = ("C1", "C2")
BASE_SEED = 20_000
GROQ_MODEL = "openai/gpt-oss-120b"
TPM_SAFETY = 0.5  # use at most half the account's tokens-per-minute limit
TOKENS_PER_STREAM_MINUTE = 85_000  # measured in the smoke test: ~1.8k tokens per ~1.3 s turn


def episode_seed(role: str, drift: str, k: int) -> int:
    return BASE_SEED + ROLES.index(role) * 1000 + DRIFTS.index(drift) * 100 + k


def plan(model_slug: str, n: int) -> list[tuple[str, str, str, int, str]]:
    """(role, drift, control, k, episode_id) for every episode of the run."""
    return [(r, d, c, k, f"{model_slug}-{r}-{d}-{c}-{k}")
            for r in ROLES for d in DRIFTS for c in CONTROLS for k in range(n)]


def groq_tpm() -> int:
    import openai

    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                           max_retries=0)
    raw = client.chat.completions.with_raw_response.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": "OK"}], max_tokens=4)
    return int(raw.headers.get("x-ratelimit-limit-tokens", "0"))


def parallel_for_tpm(tpm: int) -> int:
    return max(1, math.floor(TPM_SAFETY * tpm / TOKENS_PER_STREAM_MINUTE))


def completed_ids(paths: RunPaths) -> set[str]:
    if not paths.episodes.exists():
        return set()
    return {json.loads(line)["episode_id"] for line in paths.episodes.read_text().splitlines() if line.strip()}


class Merger:
    """Appends a finished episode's files to the run's files, one episode at a time."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.lock = threading.Lock()

    def merge(self, tmp: RunPaths) -> None:
        with self.lock:
            for name in ("events.jsonl", "retries.jsonl"):
                src = tmp.logs / name
                if src.exists():
                    with (self.paths.logs / name).open("a") as out:
                        out.write(src.read_text())
            self.paths.transcripts.mkdir(parents=True, exist_ok=True)
            for f in tmp.transcripts.glob("*.jsonl"):
                shutil.move(str(f), self.paths.transcripts / f.name)
            # The episode record goes last: its presence marks the episode as complete.
            with self.paths.episodes.open("a") as out:
                out.write((tmp.logs / "episodes.jsonl").read_text())
            shutil.rmtree(tmp.logs.parent, ignore_errors=True)  # this episode's tmp dir only


def run_one(provider_cfg: ProviderConfig, run_id: str, role: str, drift: str, control: str, k: int,
            episode_id: str, root: Path, prov: dict, scenario: ScenarioConfig) -> EpisodeRecord:
    seed = episode_seed(role, drift, k)
    state = generate_state(seed)
    tasks = build_episode(role, drift, seed, state, scenario)  # type: ignore[arg-type]
    tmp = RunPaths(logs=root / "logs" / run_id / "tmp" / episode_id / "logs",
                   transcripts=root / "logs" / run_id / "tmp" / episode_id / "transcripts")
    cfg = EpisodeConfig(run_id=run_id, episode_id=episode_id, seed=seed, model=provider_cfg.model,
                        agent_role=role, drift_condition=drift, control_condition=control,  # type: ignore[arg-type]
                        provider=provider_cfg.model_copy(update={"seed": seed}), max_turns_per_task=MAX_TURNS)
    agent = LLMAgent(make_provider(cfg.provider), role, MAX_TURNS)  # type: ignore[arg-type]
    rec, _ = run_episode(cfg, tasks, agent, tmp, prov, state=state)
    return rec


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen3", "groq"], required=True)
    ap.add_argument("--episodes-per-cell", type=int, default=3)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--max-usd", type=float, default=None, help="stop before spending more (paid models)")
    ap.add_argument("--parallel", type=int, default=None, help="override the parallelism derived from TPM")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    prov = frozen_provenance()
    if prov["any_dirty"]:
        sys.exit("refusing to run: a frozen file differs from its last commit\n" + json.dumps(prov, indent=2))
    provider_cfg = PRESETS["qwen3"] if args.model == "qwen3" else groq_config(GROQ_MODEL)
    slug = "qwen3" if args.model == "qwen3" else "gptoss120b"
    paths = RunPaths.for_run(args.root, args.run_id)
    paths.logs.mkdir(parents=True, exist_ok=True)
    scenario = ScenarioConfig()

    if args.model == "groq":
        if args.max_usd is None:
            sys.exit("--max-usd is required for paid models")
        tpm = groq_tpm()
        parallel = args.parallel or parallel_for_tpm(tpm)
        if tpm < 16_000:
            sys.exit(f"Groq reports {tpm} tokens/minute (free tier). D1 requests exceed that; not running.")
    else:
        tpm, parallel = None, 1  # one local model instance

    todo = [p for p in plan(slug, args.episodes_per_cell) if p[4] not in completed_ids(paths)]
    shutil.rmtree(paths.logs / "tmp", ignore_errors=True)  # partial episodes from a previous attempt
    meta = {
        "run_id": args.run_id, "started": datetime.now(timezone.utc).isoformat(), "git_head": git_head(),
        "frozen_provenance": prov, "provider": provider_cfg.model_dump(mode="json"),
        "scenario": scenario.model_dump(), "episodes_per_cell": args.episodes_per_cell, "max_turns": MAX_TURNS,
        "tpm_limit": tpm, "parallel": parallel, "max_usd": args.max_usd, "pending_at_start": len(todo),
    }
    (paths.logs / f"run_meta_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"{args.run_id}: {len(todo)} episodes pending, parallel={parallel}, tpm={tpm}, commit {meta['git_head'][:8]}")
    if not args.yes:
        print("Not run. Re-run with --yes.")
        return

    merger = Merger(paths)
    spent_lock = threading.Lock()
    done_records: list[EpisodeRecord] = []
    stop = threading.Event()

    def task(p):
        if stop.is_set():
            return None
        role, drift, control, k, eid = p
        rec = run_one(provider_cfg, args.run_id, role, drift, control, k, eid, args.root, prov, scenario)
        merger.merge(RunPaths(logs=paths.logs / "tmp" / eid / "logs", transcripts=paths.logs / "tmp" / eid / "transcripts"))
        with spent_lock:
            done_records.append(rec)
            spent = actual_cost(provider_cfg.model, done_records) or 0.0
            if args.max_usd is not None and spent > 0.9 * args.max_usd:
                stop.set()
        outcome = "".join({"completed": "C", "escalated": "E", "failed": "F"}[o.outcome] for o in rec.task_outcomes)
        print(f"  {eid}: {outcome} steps={rec.n_steps} harm={rec.total_harmful_attempted} "
              f"retries={rec.api_retries} ends={sorted(set(rec.task_ends.values()))}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for f in as_completed([pool.submit(task, p) for p in todo]):
            f.result()  # re-raise failures: never skip an episode silently
    spent = actual_cost(provider_cfg.model, done_records)
    remaining = len([p for p in plan(slug, args.episodes_per_cell) if p[4] not in completed_ids(paths)])
    print(f"done: {len(done_records)} episodes this session; {remaining} remaining; "
          + ("cost $0 (local)" if spent is None else f"cost ${spent:.2f}")
          + ("; STOPPED at 90% of --max-usd" if stop.is_set() else ""))


if __name__ == "__main__":
    main()
