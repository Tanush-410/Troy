"""Regenerate every table and figure from logs into results/:
    uv run python -m analysis.run_all logs/<run_id> [logs/<run_id> ...] [--out results]

Writes one CSV per table, PNG figures, and results/summary.md. Every number is
computed from the given logs; anything without data is reported as "n/a".
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.figures import dt_timelines, harm_bars, roc_curves
from analysis import stats
from analysis.stats import fmt_ci
from analysis.tables import Dataset, all_tables, by_key


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("no data\n")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _num(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def summary_md(ds: Dataset, tables: dict[str, list[dict[str, Any]]], run_dirs: list[Path]) -> str:
    lines = ["# Results summary", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
             + ", ".join(f"`{d}`" for d in run_dirs) + ".",
             f"All numbers are computed from these logs. 95% CIs are percentile bootstraps over episodes "
             f"({stats.N_BOOT} resamples). \"n/a\" means no data for that cell.", ""]
    for model in ds.models:
        eps = ds.select(model=model)
        cells = {(e.record.agent_role, e.record.drift_condition, e.record.control_condition) for e in eps}
        per_cell = len(eps) / len(cells) if cells else 0
        lines += [f"## {model}", "",
                  f"{len(eps)} episodes in {len(cells)} cells (~{per_cell:.1f} per cell).", ""]

        lines += ["### RQ1: what static RBAC contains (C1)", "",
                  "| Agent | Drifted calls | Type I | Type II | Type III | Harm inside permitted actions | "
                  "Harmful calls blocked (95% CI) |", "|---|---|---|---|---|---|---|"]
        for role in ("support", "listing", "price_intel", "all"):
            r = by_key(tables["rq1_drift_types"], model=model, agent=role, control="C1")
            if r:
                lines.append(f"| {role} | {r['drifted_calls']} | {_pct(r['type_I_share'])} | {_pct(r['type_II_share'])} | "
                             f"{_pct(r['type_III_share'])} | {_pct(r['harmful_in_permitted_actions_share'])} | "
                             f"{fmt_ci(r['harmful_blocked_share'], r['blocked_ci_lo'], r['blocked_ci_hi'])} |")
        lines += ["", "### RQ2: static (C1) vs task-scoped (C2) RBAC", "",
                  "| Agent | Harmful executed C1 | Harmful executed C2 | Fisher p (episodes) | "
                  "Task success C1 | Task success C2 | Fisher p | Decision latency C1 / C2 (ms) |",
                  "|---|---|---|---|---|---|---|---|"]
        for role in ("support", "listing", "price_intel", "all"):
            r = by_key(tables["rq2_controls"], model=model, agent=role)
            if r:
                lines.append(
                    f"| {role} | {fmt_ci(r['harm_executed_rate_c1'], r['harm_executed_rate_c1_lo'], r['harm_executed_rate_c1_hi'])} | "
                    f"{fmt_ci(r['harm_executed_rate_c2'], r['harm_executed_rate_c2_lo'], r['harm_executed_rate_c2_hi'])} | "
                    f"{_num(r['fisher_p_episodes_with_executed_harm'], 3)} | "
                    f"{fmt_ci(r['task_success_rate_c1'], r['task_success_rate_c1_lo'], r['task_success_rate_c1_hi'])} | "
                    f"{fmt_ci(r['task_success_rate_c2'], r['task_success_rate_c2_lo'], r['task_success_rate_c2_hi'])} | "
                    f"{_num(r['fisher_p_task_success'], 3)} | "
                    f"{_num(r['decision_latency_ms_c1'], 4)} / {_num(r['decision_latency_ms_c2'], 4)} |")
        lines += ["", "### RQ3: detection from permission logs (weighted combiner, all-D0 baseline)", "",
                  "| Control | Condition | AUROC (95% CI) | TPR at 5% FPR | Held-out FPR | Alerted before first harm | "
                  "Median lead (steps) |", "|---|---|---|---|---|---|---|"]
        for r in tables["rq3_detection"]:
            if r["model"] == model and r["baseline"] == "all_d0" and r["score"] == "weighted":
                lines.append(f"| {r['control']} | {r['condition']} | {fmt_ci(r['auroc'], r['auroc_lo'], r['auroc_hi'], pct=False)} | "
                             f"{_pct(r['tpr_at_5fpr'])} | {_pct(r['holdout_fpr'])} | {_pct(r['early_share'])} | "
                             f"{_num(r['median_lead_steps'], 1)} |")
        sens = [r for r in tables["rq3_detection"] if r["model"] == model and r["score"] == "weighted"]
        if sens:
            lines += ["", "Sensitivity (harm-free D0 baseline) AUROC: " + "; ".join(
                f"{r['control']} {r['condition']} {_num(r['auroc'], 3)}" for r in sens if r["baseline"] == "harm_free_d0")
                or "Sensitivity (harm-free D0 baseline): n/a"]
        lines += ["", "### Other rates", "", "| Agent | D0 harm rate (95% CI) | Format-error rate (95% CI) | "
                  "Context overflow | Escalation rate | API retries |", "|---|---|---|---|---|---|"]
        for role in ("support", "listing", "price_intel", "all"):
            d = by_key(tables["d0_harm"], model=model, agent=role)
            f = by_key(tables["format_errors"], model=model, agent=role)
            c = by_key(tables["context_overflow_escalation"], model=model, agent=role, drift="all")
            lines.append(f"| {role} | {fmt_ci(d['d0_harm_rate'], d['d0_harm_rate_lo'], d['d0_harm_rate_hi']) if d else 'n/a'} | "
                         f"{fmt_ci(f['format_error_rate'], f['lo'], f['hi']) if f else 'n/a'} | "
                         f"{_pct(c['context_overflow_rate']) if c else 'n/a'} | "
                         f"{_pct(c['escalation_rate']) if c else 'n/a'} | {c['api_retries'] if c else 'n/a'} |")
        xa = [r for r in tables["cross_agent_taint"] if r["model"] == model]
        if xa:
            lines += ["", "Cross-agent taint (listing agent): " + "; ".join(
                f"{r['control']}: {r['cross_agent_tainted_calls']} tainted calls, {r['cross_agent_a1_calls']} A1 "
                f"({r['cross_agent_a1_executed']} executed)" for r in xa)]
        lines.append("")
    lines += ["## Files", "", "Tables (CSV) and figures (PNG) are in this folder; see `index.json`.", ""]
    return "\n".join(lines)


def run(run_dirs: list[Path], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.load(run_dirs)
    tables, det = all_tables(ds)
    files = []
    for name, rows in tables.items():
        write_csv(rows, out / f"{name}.csv")
        files.append(f"{name}.csv")
    figs = roc_curves(ds, det, out) + harm_bars(tables["rq2_controls"], out) + dt_timelines(ds, det, out)
    files += [p.name for p in figs]
    (out / "summary.md").write_text(summary_md(ds, tables, run_dirs))
    index = {"run_dirs": [str(d) for d in run_dirs], "episodes": len(ds.episodes), "files": files + ["summary.md"]}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    index = run(args.run_dirs, args.out)
    print(f"wrote {len(index['files'])} files for {index['episodes']} episodes to {args.out}/")


if __name__ == "__main__":
    main()
