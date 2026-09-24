"""Figures, drawn only from computed results (never from hand-entered numbers)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from analysis.detection import DetectionResult  # noqa: E402
from analysis.tables import DETECT_CONTROL, ROLES, Dataset  # noqa: E402

COLORS = {"D1": "#4C72B0", "D2": "#DD8452", "D3": "#55A868", "C1": "#8172B3", "C2": "#C44E52"}


def roc_curves(ds: Dataset, results: dict[tuple[str, str, str], DetectionResult], out: Path) -> list[Path]:
    paths = []
    for (model, control, baseline), r in sorted(results.items()):
        recs = {ep.record.episode_id: ep.record for ep in ds.select(model=model, control_condition=control)}
        fig, ax = plt.subplots(figsize=(5, 5))
        for cond in r.by_condition:
            ids = [i for i in r.scores if recs[i].drift_condition in (cond, "D0")]
            y = [int(recs[i].total_harmful_attempted > 0) for i in ids]
            if not 0 < sum(y) < len(y):
                continue
            for combiner, style in (("weighted", "-"), ("iforest", "--")):
                s = [max((getattr(st, combiner) for st in r.scores[i].steps), default=0.0) for i in ids]
                fpr, tpr, _ = roc_curve(y, s)
                auc = r.by_condition[cond].auroc[combiner]
                ax.plot(fpr, tpr, style, color=COLORS[cond], label=f"{cond} {combiner} (AUROC {auc:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="grey", linewidth=0.8)
        ax.axvline(0.05, color="grey", linewidth=0.6)
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
               title=f"{model} {DETECT_CONTROL[control]} ({baseline})")
        ax.legend(fontsize=7, loc="lower right")
        p = out / f"roc_{_slug(model)}_{DETECT_CONTROL[control]}_{baseline}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def harm_bars(rq2_rows: list[dict], out: Path) -> list[Path]:
    paths = []
    for model in sorted({r["model"] for r in rq2_rows}):
        rows = {r["agent"]: r for r in rq2_rows if r["model"] == model}
        agents = [a for a in (*ROLES, "all") if a in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, metric, title in ((axes[0], "harm_attempted_rate", "Harmful calls attempted"),
                                  (axes[1], "harm_executed_rate", "Harmful calls executed")):
            width = 0.38
            for k, ctl in enumerate(("c1", "c2")):
                xs, ys, err_lo, err_hi = [], [], [], []
                for i, a in enumerate(agents):
                    v = rows[a].get(f"{metric}_{ctl}")
                    if v is None:
                        continue
                    lo, hi = rows[a].get(f"{metric}_{ctl}_lo"), rows[a].get(f"{metric}_{ctl}_hi")
                    xs.append(i + (k - 0.5) * width)
                    ys.append(100 * v)
                    err_lo.append(100 * (v - lo) if lo is not None else 0)
                    err_hi.append(100 * (hi - v) if hi is not None else 0)
                ax.bar(xs, ys, width, yerr=[err_lo, err_hi], capsize=3, color=COLORS[ctl.upper()],
                       label=ctl.upper())
            ax.set_xticks(range(len(agents)), agents)
            ax.set_title(title)
            ax.set_ylabel("% of well-formed calls (95% CI by episode)")
        axes[0].legend()
        fig.suptitle(model)
        fig.tight_layout()
        p = out / f"harm_rates_{_slug(model)}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def dt_timelines(ds: Dataset, results: dict[tuple[str, str, str], DetectionResult], out: Path) -> list[Path]:
    """D_t and the weighted score over steps for one representative episode per drift condition
    (the detected positive episode with the median lead time), with alert and first-harm steps marked."""
    paths = []
    for (model, control, baseline), r in sorted(results.items()):
        if baseline != "all_d0":
            continue
        recs = {ep.record.episode_id: ep.record for ep in ds.select(model=model, control_condition=control)}
        picks = []
        for cond in ("D1", "D2", "D3"):
            cands = sorted(
                (recs[i].first_harm_step - s.alert_step["weighted"], i) for i, s in r.scores.items()
                if recs[i].drift_condition == cond and s.alert_step["weighted"] is not None
                and recs[i].first_harm_step is not None)
            if cands:
                picks.append((cond, cands[len(cands) // 2][1]))
        if not picks:
            continue
        fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.6 * len(picks)), squeeze=False)
        for ax, (cond, eid) in zip(axes[:, 0], picks):
            steps = r.scores[eid].steps
            xs = [s.step for s in steps]
            ax.plot(xs, [s.drift_jsd for s in steps], color=COLORS[cond], label="D_t (JSD)")
            ax2 = ax.twinx()
            ax2.plot(xs, [s.weighted for s in steps], color="black", linewidth=0.8, label="PLDD weighted score")
            ax2.axhline(r.thresholds["weighted"], color="black", linestyle=":", linewidth=0.8)
            ax.axvline(r.scores[eid].alert_step["weighted"], color="red", linestyle="--", label="alert")
            ax.axvline(recs[eid].first_harm_step, color="orange", linestyle="-", label="first harm")
            ax.set(title=f"{cond}: {eid}", xlabel="step", ylabel="D_t")
            ax2.set_ylabel("PLDD score (dotted: threshold)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        fig.tight_layout()
        p = out / f"dt_{_slug(model)}_{DETECT_CONTROL[control]}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")
