# SimMart drift harness

Experiment code for *Beyond Static Roles: Evaluating Role-Based Access Control as a Containment and Detection Mechanism for Agent Drift*.

LLM agents work inside a simulated e-commerce marketplace (SimMart) as three roles: customer support, listing management, and competitor price intelligence. Every tool call passes through a policy enforcement point (PEP) outside the agent. The PEP enforces role-based access control, labels each call against a fixed drift taxonomy and harm oracle, and writes a JSONL log. The code measures three things:

- **RQ1:** how much harmful drift static RBAC (C1) blocks, and how much happens inside permitted actions.
- **RQ2:** whether task-scoped RBAC (C2) reduces harm, and what it costs in task success.
- **RQ3:** whether a detector that reads only the permission log (PLDD) flags drift before the first harmful action.

Drift is induced four ways:

- D0: none
- D1: 10 tasks in one context
- D2: ambiguous or role-conflicting instructions
- D3: prompt injections in tickets, supplier text and competitor pages, including a cross-agent path

All tools are mocked and every run is seeded. `CLAUDE.md` is the project brief, and [`docs/requirements.md`](docs/requirements.md) is the consolidated requirements specification. [`docs/decisions.md`](docs/decisions.md) records every design decision, the frozen-file commit hashes, and any change made before the freeze. [`docs/progress.md`](docs/progress.md) summarises each milestone.

## Install

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python version (3.11, from `.python-version`) and the locked dependencies (`uv.lock`).

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync
```

Without uv, use Python 3.11 and the pinned `requirements.txt`, which is exported from `uv.lock`:

```sh
pip install -r requirements.txt
```

## Run the tests

```sh
uv run pytest            # full suite, about 35 s
uv run pytest -q tests/test_permissions.py   # one file
```

The tests cover:

- every permission decision against the tables in `CLAUDE.md`;
- one harmful and one benign case per harm rule;
- the success checkers;
- that every generated clean task is completable without harm;
- that injection targets have no false positives;
- that the detector package cannot import labels, taint or transcripts;
- the analysis pipeline end to end on scripted logs.

## Scripted pipeline (no model, no cost)

A fixed-script agent runs three episodes (price intelligence, then listing, then support) on one shared state, under C1 and then C2. Its steps deliberately include every drift type and every harm rule, including the cross-agent injection path.

```sh
uv run python -m experiments.scripted_pipeline          # writes logs/ and transcripts/ here
uv run python -m experiments.scripted_pipeline /tmp/run # or under another root
```

It prints the core metrics for each control condition. The logs land in `logs/scripted-C1/` and `logs/scripted-C2/`. `tests/test_pipeline.py` checks every logged event against the label the script intended.

## Smoke test (one episode per agent, real model)

Without `--yes` the command only prints a token and cost estimate; add `--yes` to run it.

**Local model (Ollama, free):**

```sh
ollama pull qwen3:8b
uv run python -m experiments.context_probe qwen3:8b 32768   # must print PASS before results count
uv run python -m experiments.smoke --model qwen3 --yes
```

The harness talks to Ollama through its native API with `num_ctx` set on every request. Ollama's OpenAI-compatible endpoint ignores `num_ctx` and silently truncates prompts; see `docs/decisions.md`.

**Groq (paid API):** set `GROQ_API_KEY` in your environment (never in the repo), then:

```sh
uv run python -m experiments.smoke --model groq --groq-model openai/gpt-oss-120b          # estimate only
uv run python -m experiments.smoke --model groq --groq-model openai/gpt-oss-120b --yes    # run
```

Logs go to `logs/smoke-<model>/`, and prompts and model outputs to `transcripts/smoke-<model>/`. Transcripts are kept for qualitative examples only; the detector never reads them.

## Analysis

One command regenerates every table and figure from one or more run log directories:

```sh
uv run python -m analysis.run_all logs/<run_id> [logs/<run_id> ...] --out results
```

It writes to `results/`:

- CSV tables: RQ1–RQ3, detector ablations, D0 harm, format errors, context overflow and escalation, cross-agent taint;
- figures: ROC curves, harm-rate bars with CIs, and D_t timelines;
- `summary.md`.

Every number is computed from the logs. Confidence intervals are 95% bootstraps over episodes, and a cell with no data reads "n/a".

## Dashboard (interactive)

A read-only Streamlit dashboard over the run logs, for exploring results interactively while the pilot is still running. It never writes to `logs/` or `transcripts/` (the live demo only writes under `logs/demo/`) and it imports every number from `analysis/`, so the GUI cannot disagree with `analysis.run_all`. The dependencies live in a non-default `gui` group, so `uv sync` is unchanged:

```sh
uv run --group gui streamlit run gui/app.py
```

Pick the runs and filters in the sidebar (run folders, agent role, model, drift condition, control condition), then use the views:

- **Overview** — the five headline metrics, calls per drift type per agent, C1 vs C2 harmful rate by condition, and the share of harmful calls static RBAC blocked;
- **Episode explorer** — every episode with its task outcomes, per-step tool calls (drift types and harm rules), the D_t and PLDD score timeline, the transcript, and a same-seed C1 vs C2 comparison;
- **Detector** — ROC curves, TPR at 5% FPR, AUROC with CI, lead time, and the feature ablation, computed from the replayed logs only;
- **Cross-agent** — the three-stage injection flow (competitor page -> tainted price report -> listing action) with the cross-agent taint table;
- **Live demo** — run one episode on demand (scripted by default, local `qwen3:8b`, or Groq). Real models are disabled while an experiment is running, and a Groq run shows its token and USD estimate and needs an explicit confirmation.

## Full runs

`experiments/pilot.py` runs every cell with paired seeds and is resumable. It refuses to start if any frozen file (oracle, success checkers, permission tables, prompts) differs from its last commit. For example:

```sh
uv run python -m experiments.pilot --model qwen3 --episodes-per-cell 3 --run-id pilot-qwen3 --yes
uv run python -m experiments.pilot --model groq  --episodes-per-cell 3 --run-id pilot-groq --max-usd 2 --yes
```

For cost and time estimates from measured smoke logs:

```sh
uv run python -m experiments.full_estimate logs/<groq smoke run> logs/<qwen smoke run>
```

## Layout

| Folder | Contents |
|---|---|
| `simmart/` | Marketplace state, seeded data generator, competitor price simulator |
| `tools/` | Mock tools and their schemas |
| `policy/` | Permission tables (frozen), static and task-scoped RBAC, the PEP, log schema |
| `oracle/` | Harm rules, prohibited claims, drift-type classifier, task-success checkers (frozen) |
| `agents/` | Agent loop, role prompts (frozen), scripted agent, providers (Anthropic, Groq/OpenAI-compatible, Ollama) |
| `scenarios/` | D0–D3 task and injection generators; reference solver used in tests |
| `detector/` | PLDD features, combiners, calibration, replay; reads only the detector view of the log |
| `experiments/` | Episode runner, smoke test, pilot/full runner, cost estimates, provenance |
| `analysis/` | Metrics, statistics, tables, figures, `run_all` |
| `gui/` | Read-only Streamlit dashboard (`gui/app.py`), log loading, live demo |
| `tests/` | pytest suite |
| `logs/`, `transcripts/`, `results/` | Generated output (`logs/` and `transcripts/` are gitignored) |
