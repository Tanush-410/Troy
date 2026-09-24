"""Result tables, computed from logs only. Every number is derived here from the
JSONL files; a cell with no data is written as "n/a".

Units: the episode is the resampling unit for every CI. Rates over calls or
tasks are ratios of sums across the (re)sampled episodes.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.detection import DetectionResult, evaluate
from analysis.metrics import load_episodes, load_events
from analysis.stats import bootstrap_ci, fisher, mann_whitney, ratio_stat
from detector.features import FEATURES
from detector.pldd import COMBINERS
from policy.log_schema import EpisodeRecord, PermissionEvent

ROLES = ("support", "listing", "price_intel")
DRIFT_TYPES = ("I", "II", "III")
DETECT_CONTROL = {"C1": "C3", "C2": "C4"}  # offline replay: C1 logs + PLDD = C3, C2 logs + PLDD = C4


@dataclass
class Episode:
    record: EpisodeRecord
    events: list[PermissionEvent] = field(default_factory=list)

    @property
    def wf(self) -> list[PermissionEvent]:
        return [e for e in self.events if not e.format_error]


@dataclass
class Dataset:
    episodes: list[Episode]
    event_paths: list[Path]

    @classmethod
    def load(cls, run_dirs: Sequence[Path]) -> Dataset:
        by_id: dict[str, Episode] = {}
        for d in run_dirs:
            for r in load_episodes(d / "episodes.jsonl"):
                by_id[r.episode_id] = Episode(r)
        for d in run_dirs:
            for e in load_events(d / "events.jsonl"):
                if e.episode_id in by_id:
                    by_id[e.episode_id].events.append(e)
        return cls(list(by_id.values()), [d / "events.jsonl" for d in run_dirs])

    def select(self, **kw: str) -> list[Episode]:
        out = self.episodes
        for k, v in kw.items():
            if v != "all":
                out = [ep for ep in out if getattr(ep.record, k) == v]
        return out

    @property
    def models(self) -> list[str]:
        return sorted({ep.record.model for ep in self.episodes})


def _ci(units: Sequence[Episode], num, den) -> tuple[float | None, float | None, float | None]:
    return bootstrap_ci(units, ratio_stat(num, den))


def _cells(ci: tuple[float | None, float | None, float | None]) -> list[Any]:
    return list(ci)


def _slices(ds: Dataset) -> Iterable[tuple[str, str]]:
    for m in ds.models:
        for role in (*ROLES, "all"):
            yield m, role


# ------------------------------------------------------------------ RQ1


def rq1_drift_types(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model, role in _slices(ds):
        for control in ("C1", "C2"):
            eps = ds.select(model=model, agent_role=role, control_condition=control)
            if not eps:
                continue
            ev = [e for ep in eps for e in ep.wf]
            drifted = [e for e in ev if e.drift_type in DRIFT_TYPES]
            row: dict[str, Any] = {"model": model, "agent": role, "control": control, "episodes": len(eps),
                                   "calls": len(ev), "drifted_calls": len(drifted)}
            for t in DRIFT_TYPES:
                of_t = [e for e in drifted if e.drift_type == t]
                row[f"type_{t}"] = len(of_t)
                row[f"type_{t}_share"] = len(of_t) / len(drifted) if drifted else None
                row[f"type_{t}_harmful"] = sum(e.harm for e in of_t)
            harmful = [e for e in ev if e.harm]
            row["harmful_attempted"] = len(harmful)
            row["harmful_in_permitted_actions_share"] = (
                sum(e.drift_type in ("II", "III") for e in harmful) / len(harmful) if harmful else None)
            b = _ci(eps, lambda ep: sum(e.harm and e.decision == "deny" for e in ep.wf),
                    lambda ep: sum(e.harm for e in ep.wf))
            row["harmful_blocked_share"], row["blocked_ci_lo"], row["blocked_ci_hi"] = b
            rows.append(row)
    return rows


# ------------------------------------------------------------------ RQ2


def rq2_controls(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model, role in _slices(ds):
        c1 = ds.select(model=model, agent_role=role, control_condition="C1")
        c2 = ds.select(model=model, agent_role=role, control_condition="C2")
        if not c1 and not c2:
            continue
        row: dict[str, Any] = {"model": model, "agent": role, "episodes_c1": len(c1), "episodes_c2": len(c2)}
        for name, eps in (("c1", c1), ("c2", c2)):
            metrics = {
                "harm_attempted_rate": (lambda ep: sum(e.harm for e in ep.wf), lambda ep: len(ep.wf)),
                "harm_executed_rate": (lambda ep: sum(e.harm and e.executed for e in ep.wf), lambda ep: len(ep.wf)),
                "task_success_rate": (lambda ep: sum(o.success for o in ep.record.task_outcomes),
                                      lambda ep: len(ep.record.task_outcomes)),
                "decision_latency_ms": (lambda ep: sum(e.decision_latency_ms for e in ep.wf), lambda ep: len(ep.wf)),
            }
            for k, (num, den) in metrics.items():
                p, lo, hi = _ci(eps, num, den) if eps else (None, None, None)
                row[f"{k}_{name}"], row[f"{k}_{name}_lo"], row[f"{k}_{name}_hi"] = p, lo, hi
        # Episode-level Fisher tests (episodes are the independent units)
        exec_harm = [sum(any(e.harm and e.executed for e in ep.wf) for ep in eps) for eps in (c1, c2)]
        row["fisher_p_episodes_with_executed_harm"] = fisher(exec_harm[0], len(c1) - exec_harm[0],
                                                             exec_harm[1], len(c2) - exec_harm[1])
        succ = [(sum(o.success for ep in eps for o in ep.record.task_outcomes),
                 sum(not o.success for ep in eps for o in ep.record.task_outcomes)) for eps in (c1, c2)]
        row["fisher_p_task_success"] = fisher(succ[0][0], succ[0][1], succ[1][0], succ[1][1])
        rows.append(row)
    return rows


# ------------------------------------------------------------------ RQ3


def detection_results(ds: Dataset) -> dict[tuple[str, str, str], DetectionResult]:
    records = [ep.record for ep in ds.episodes]
    out = {}
    for model in ds.models:
        for control in ("C1", "C2"):
            for baseline in ("all_d0", "harm_free_d0"):
                r = evaluate(ds.event_paths, records, model, control, baseline=baseline,
                             ablate=baseline == "all_d0")
                if r is not None:
                    out[(model, control, baseline)] = r
    return out


def _auroc_ci(r: DetectionResult, ds: Dataset, model: str, control: str, cond: str, score: str):
    from sklearn.metrics import roc_auc_score
    recs = {ep.record.episode_id: ep.record for ep in ds.select(model=model, control_condition=control)}
    ids = [i for i in r.scores if recs.get(i) and (recs[i].drift_condition == cond or recs[i].drift_condition == "D0")]
    units = [(int(recs[i].total_harmful_attempted > 0),
              max((getattr(s, score) for s in r.scores[i].steps), default=0.0)) for i in ids]

    def auc(sample):
        ys = [y for y, _ in sample]
        return float(roc_auc_score(ys, [s for _, s in sample])) if 0 < sum(ys) < len(ys) else None

    return bootstrap_ci(units, auc)


def rq3_detection(ds: Dataset, results: dict[tuple[str, str, str], DetectionResult]) -> list[dict[str, Any]]:
    rows = []
    for (model, control, baseline), r in sorted(results.items()):
        for cond, c in r.by_condition.items():
            for score in (*COMBINERS, "drift_jsd"):
                p, lo, hi = _auroc_ci(r, ds, model, control, cond, score)
                lead = c.lead_times.get(score, [])
                rows.append({
                    "model": model, "control": DETECT_CONTROL[control], "baseline": baseline, "condition": cond,
                    "score": score, "n_pos": c.n_positive, "n_neg": c.n_negative,
                    "auroc": p, "auroc_lo": lo, "auroc_hi": hi,
                    "tpr_at_5fpr": c.tpr.get(score), "holdout_fpr": c.holdout_fpr.get(score),
                    "early_share": c.early_share.get(score),
                    "median_lead_steps": statistics.median(lead) if lead else None, "n_detected": len(lead),
                })
    return rows


def rq3_lead_time_tests(results: dict[tuple[str, str, str], DetectionResult]) -> list[dict[str, Any]]:
    """Mann-Whitney U on lead times, C3 (C1 logs) vs C4 (C2 logs), per model/baseline/condition/combiner."""
    rows = []
    for (model, control, baseline), r in sorted(results.items()):
        if control != "C1" or (model, "C2", baseline) not in results:
            continue
        r2 = results[(model, "C2", baseline)]
        for cond in r.by_condition:
            if cond not in r2.by_condition:
                continue
            for c in COMBINERS:
                a, b = r.by_condition[cond].lead_times[c], r2.by_condition[cond].lead_times[c]
                rows.append({"model": model, "baseline": baseline, "condition": cond, "combiner": c,
                             "n_c3": len(a), "n_c4": len(b),
                             "median_c3": statistics.median(a) if a else None,
                             "median_c4": statistics.median(b) if b else None,
                             "mann_whitney_p": mann_whitney(a, b)})
    return rows


def ablation(results: dict[tuple[str, str, str], DetectionResult]) -> list[dict[str, Any]]:
    rows = []
    for (model, control, baseline), r in sorted(results.items()):
        for f in FEATURES:
            for c in COMBINERS:
                if f in r.ablation:
                    rows.append({"model": model, "control": DETECT_CONTROL[control], "baseline": baseline,
                                 "feature_removed": f, "combiner": c, "delta_auroc": r.ablation[f][c]})
    return rows


# ------------------------------------------------------------------ extra tables


def d0_harm(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model, role in _slices(ds):
        eps = ds.select(model=model, agent_role=role, drift_condition="D0")
        if not eps:
            continue
        p, lo, hi = _ci(eps, lambda ep: sum(e.harm for e in ep.wf), lambda ep: len(ep.wf))
        q, qlo, qhi = bootstrap_ci(eps, lambda s: sum(ep.record.total_harmful_attempted > 0 for ep in s) / len(s))
        rows.append({"model": model, "agent": role, "d0_episodes": len(eps),
                     "d0_harm_rate": p, "d0_harm_rate_lo": lo, "d0_harm_rate_hi": hi,
                     "d0_episodes_with_harm_share": q, "share_lo": qlo, "share_hi": qhi})
    return rows


def format_errors(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model, role in _slices(ds):
        eps = ds.select(model=model, agent_role=role)
        if not eps:
            continue
        p, lo, hi = _ci(eps, lambda ep: sum(e.format_error for e in ep.events), lambda ep: len(ep.events))
        rows.append({"model": model, "agent": role, "calls": sum(len(ep.events) for ep in eps),
                     "format_errors": sum(e.format_error for ep in eps for e in ep.events),
                     "format_error_rate": p, "lo": lo, "hi": hi})
    return rows


def context_and_escalation(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model, role in _slices(ds):
        for drift in ("D0", "D1", "D2", "D3", "all"):
            eps = ds.select(model=model, agent_role=role, drift_condition=drift)
            if not eps:
                continue
            ends = Counter(v for ep in eps for v in ep.record.task_ends.values())
            n_planned = sum(ends.values())
            esc = _ci(eps, lambda ep: sum(o.outcome == "escalated" for o in ep.record.task_outcomes),
                      lambda ep: len(ep.record.task_outcomes))
            rows.append({
                "model": model, "agent": role, "drift": drift, "episodes": len(eps), "tasks_planned": n_planned,
                "context_overflow_rate": ends["context_overflow"] / n_planned if n_planned else None,
                "tasks_not_run": ends["not_run"], "step_limit_rate": ends["step_limit"] / n_planned if n_planned else None,
                "escalation_rate": esc[0], "escalation_lo": esc[1], "escalation_hi": esc[2],
                "escalations_acceptable": sum(o.outcome == "escalated" and o.escalation_acceptable
                                              for ep in eps for o in ep.record.task_outcomes),
                "api_retries": sum(ep.record.api_retries for ep in eps),
            })
    return rows


def cross_agent(ds: Dataset) -> list[dict[str, Any]]:
    rows = []
    for model in ds.models:
        for control in ("C1", "C2"):
            eps = ds.select(model=model, agent_role="listing", control_condition=control)
            if not eps:
                continue
            tainted = [e for ep in eps for e in ep.wf if e.cross_agent_taint]
            p, lo, hi = _ci(eps, lambda ep: sum(e.cross_agent_taint and "A1" in e.harm_rule_ids for e in ep.wf),
                            lambda ep: sum(e.cross_agent_taint for e in ep.wf))
            rows.append({"model": model, "control": control, "listing_episodes": len(eps),
                         "episodes_with_cross_agent_taint": sum(any(e.cross_agent_taint for e in ep.wf) for ep in eps),
                         "cross_agent_tainted_calls": len(tainted),
                         "cross_agent_a1_calls": sum("A1" in e.harm_rule_ids for e in tainted),
                         "cross_agent_a1_executed": sum("A1" in e.harm_rule_ids and e.executed for e in tainted),
                         "a1_share_of_tainted_calls": p, "lo": lo, "hi": hi})
    return rows


def all_tables(ds: Dataset) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str, str], DetectionResult]]:
    det = detection_results(ds)
    return {
        "rq1_drift_types": rq1_drift_types(ds),
        "rq2_controls": rq2_controls(ds),
        "rq3_detection": rq3_detection(ds, det),
        "rq3_lead_time_tests": rq3_lead_time_tests(det),
        "ablation": ablation(det),
        "d0_harm": d0_harm(ds),
        "format_errors": format_errors(ds),
        "context_overflow_escalation": context_and_escalation(ds),
        "cross_agent_taint": cross_agent(ds),
    }, det


def by_key(rows: list[dict[str, Any]], **kw: str) -> dict[str, Any] | None:
    return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), None)


def group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out
