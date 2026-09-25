"""Streamlit dashboard over the PEP run logs.

Five views over the data `gui.data` exposes: an overview of the headline metrics,
an episode explorer down to the individual tool call, the RQ3 detector results,
the cross-agent taint path, and a live demo runner. Every number shown here is
produced by `gui.data` (or the `analysis.tables` helpers behind it); nothing is
recomputed in this file, transcripts are only ever displayed, and the app writes
nothing outside the demo log directory.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.tables import DETECT_CONTROL
from detector.pldd import COMBINERS
from experiments.smoke import GROQ_CANDIDATES
from gui import data as gdata
from gui import demo as gdemo
from gui.filters import EpisodeFilters, GlobalFilters
from gui.runner_status import RunnerStatus, detect_active_runner

LOGS_ROOT = PROJECT_ROOT / "logs"
ALL_RUNS = "All run folders"
NA = "n/a"
NO_RUNS = "no-runs-found"
NO_SELECTION = "no-selection"

VIEW_OVERVIEW = "Overview"
VIEW_EPISODES = "Episode explorer"
VIEW_DETECTOR = "Detector"
VIEW_CROSS_AGENT = "Cross-agent"
VIEW_DEMO = "Live demo"
VIEWS = (VIEW_OVERVIEW, VIEW_EPISODES, VIEW_DETECTOR, VIEW_CROSS_AGENT, VIEW_DEMO)

ROLE_LABEL = {"all": "All agents", "support": "Support", "listing": "Listing",
              "price_intel": "Price intel"}
OUTCOMES = ("completed", "escalated", "failed")
BASELINES = ("all_d0", "harm_free_d0")
SOURCE_CONTROLS = ("C1", "C2")

DRIFT_TYPE_COLORS = {"I": "#4C72B0", "II": "#DD8452", "III": "#55A868", "none": "#C9C9C9",
                     "format_error": "#B07AA1"}
DRIFT_TYPE_FILL = {"I": "#e7eef8", "II": "#fbeade", "III": "#e6f3ea", "none": "#ffffff",
                   "format_error": "#f3e9f7"}
CONDITION_COLORS = {"D0": "#8C8C8C", "D1": "#4C72B0", "D2": "#DD8452", "D3": "#55A868"}
CONTROL_COLORS = {"C1": "#8172B3", "C2": "#C44E52", "C3": "#8172B3", "C4": "#C44E52"}
COMBINER_COLORS = {"weighted": "#4C72B0", "iforest": "#55A868"}
HARM_FILL = "#ffb3b3"
STAGE_COLORS = ("#8C8C8C", "#DD8452", "#C44E52")

EPISODE_COLUMNS = ("episode_id", "run_id", "seed", "model", "agent", "drift", "control", "tasks",
                   "outcome", "success_rate", "harm_count", "harmful_executed", "first_harm_step")
CALL_COLUMNS = ("step", "task", "action", "params", "decision", "deny_layer", "drift_type", "harmful",
                "harm_rule_ids", "tainted_context", "context_injection_ids", "cross_agent_taint",
                "taint_report_ids", "executed", "in_role", "in_task", "latency_ms")
LIVE_COLUMNS = ("step", "task_id", "action", "decision", "deny_layer", "drift_type", "harm", "executed",
                "in_task", "latency_ms")
DEMO_MODES = (("Scripted (offline replay)", "scripted"), ("qwen3:8b (local Ollama)", "qwen3"),
              ("Groq (paid API)", "groq"))


def _pct(value: float | None, digits: int = 1) -> str:
    return NA if value is None else f"{100 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def _ci(point: float | None, lo: float | None, hi: float | None) -> str:
    if point is None:
        return NA
    if lo is None or hi is None:
        return _pct(point)
    return f"{_pct(point)} [{_pct(lo)}, {_pct(hi)}]"


def _usd(value: float | None) -> str:
    return "unpriced (local model)" if value is None else f"${value:.4f}"


def _ids(values: Sequence[str], limit: int = 3) -> str:
    shown = ", ".join(f"`{v}`" for v in values[:limit])
    return shown if len(values) <= limit else shown + f", +{len(values) - limit} more"


def _textual(frame: pd.DataFrame) -> pd.DataFrame:
    """Arrow carries no dict or list cells, so nested transcript values are shown as JSON."""
    def cell(value: Any) -> Any:
        return json.dumps(value, default=str, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value

    return frame.map(cell)


def _chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


@st.cache_data(show_spinner="Loading run logs (read-only snapshot)…", max_entries=12)
def _dashboard(runs: tuple[gdata.RunInfo, ...]) -> gdata.DashboardData:
    return gdata.load_dashboard(runs)


def _run_options(runs: Sequence[gdata.RunInfo]) -> dict[str, gdata.RunInfo]:
    """Display option -> run, with the parent folder appended when labels collide."""
    out: dict[str, gdata.RunInfo] = {}
    for run in runs:
        option = run.label
        if option in out:
            option = f"{run.label} ({run.path.parent.name})"
            n = 2
            while option in out:
                option = f"{run.label} ({run.path.parent.name} #{n})"
                n += 1
        out[option] = run
    return out


def _single_model(selected: str, episodes: Sequence[Any]) -> tuple[str | None, str]:
    """The one model the model-scoped panels need, plus a note when it is a fallback."""
    if selected != gdata.ALL:
        return selected, ""
    present = sorted({ep.record.model for ep in episodes})
    if not present:
        return None, "the current slice holds no episodes, so no model can be selected"
    if len(present) > 1:
        return present[0], f"showing `{present[0]}`, one of {len(present)} models in the slice"
    return present[0], ""


def _episode_frame(rows: Sequence[gdata.EpisodeRow]) -> pd.DataFrame:
    records = [{"episode_id": r.episode_id, "run_id": r.run_id, "seed": r.seed, "model": r.model,
                "agent": r.agent, "drift": r.drift, "control": r.control, "tasks": r.tasks,
                "outcome": r.outcome, "success_rate": r.success_rate, "harm_count": r.harm_count,
                "harmful_executed": r.harmful_executed, "first_harm_step": r.first_harm_step}
               for r in rows]
    return pd.DataFrame(records, columns=list(EPISODE_COLUMNS))


def _drift_label(event: Any) -> str:
    if event.drift_type:
        return event.drift_type
    return "format_error" if event.format_error else "none"


def _call_frame(events: Sequence[Any]) -> pd.DataFrame:
    records = [{"step": e.step, "task": e.task_id or "-", "action": e.action,
                "params": json.dumps(e.params, sort_keys=True, default=str), "decision": e.decision,
                "deny_layer": e.deny_layer or "-", "drift_type": _drift_label(e), "harmful": bool(e.harm),
                "harm_rule_ids": ", ".join(e.harm_rule_ids) or "-",
                "tainted_context": bool(e.tainted_context),
                "context_injection_ids": ", ".join(e.context_injection_ids) or "-",
                "cross_agent_taint": bool(e.cross_agent_taint),
                "taint_report_ids": ", ".join(e.taint_report_ids) or "-",
                "executed": bool(e.executed), "in_role": bool(e.in_role), "in_task": bool(e.in_task),
                "latency_ms": e.latency_ms}
               for e in events]
    return pd.DataFrame(records, columns=list(CALL_COLUMNS))


def _style_calls(frame: pd.DataFrame) -> Any:
    def row_style(row: pd.Series) -> list[str]:
        if bool(row["harmful"]):
            fill = HARM_FILL
        else:
            fill = DRIFT_TYPE_FILL.get(str(row["drift_type"]), "#ffffff")
        return [f"background-color: {fill}"] * len(row)

    return frame.style.apply(row_style, axis=1)


def _call_column_config(width: str = "small") -> dict[str, Any]:
    return {"action": st.column_config.TextColumn("Action", width="medium"),
            "params": st.column_config.TextColumn("Params (JSON)", width=width),
            "step": st.column_config.NumberColumn("Step", format="%d", width="small"),
            "latency_ms": st.column_config.NumberColumn("Latency ms", format="%.0f", width="small"),
            "harmful": st.column_config.CheckboxColumn("Harmful", width="small"),
            "in_role": st.column_config.CheckboxColumn("In role", width="small"),
            "in_task": st.column_config.CheckboxColumn("In task", width="small"),
            "executed": st.column_config.CheckboxColumn("Executed", width="small"),
            "tainted_context": st.column_config.CheckboxColumn("Tainted ctx", width="small"),
            "cross_agent_taint": st.column_config.CheckboxColumn("Cross-agent taint", width="small")}


def _sidebar() -> tuple[tuple[gdata.RunInfo, ...], GlobalFilters, str, str]:
    st.sidebar.header("Runs")
    runs = gdata.discover_runs(LOGS_ROOT)
    options = _run_options(runs)
    chosen = st.sidebar.multiselect(
        "Run folders", [ALL_RUNS, *options], default=[ALL_RUNS], key="run_picker",
        format_func=lambda value: "All run folders" if value == ALL_RUNS else value)
    if chosen and ALL_RUNS not in chosen:
        selected = tuple(options[value] for value in chosen if value in options)
    else:
        selected = tuple(runs)
    st.sidebar.caption(f"{len(selected)} of {len(runs)} run folder(s) under `{LOGS_ROOT}`")

    if not selected:
        if not runs:
            return (), GlobalFilters(), "", NO_RUNS
        return (), GlobalFilters(), "", NO_SELECTION

    data = _dashboard(selected)
    for warning in data.warnings:
        st.sidebar.warning(warning)

    choice = gdata.filter_options(data.dataset)
    st.sidebar.header("Slice")
    agent = st.sidebar.selectbox("Agent", choice["agents"], format_func=lambda v: ROLE_LABEL.get(v, v))
    model = st.sidebar.selectbox("Model", choice["models"])
    drift = st.sidebar.selectbox("Drift", choice["drifts"])
    control = st.sidebar.selectbox("Control", choice["controls"])
    filters = GlobalFilters(model=model, agent=agent, control=control, drift=drift)
    return selected, filters, model, ""


def _metric_row(card: gdata.OverviewCard) -> None:
    cols = st.columns(5)
    cols[0].metric("Episodes", f"{card.episodes:,}")
    cols[1].metric("Task success", _pct(card.task_success),
                   help="Completed tasks, plus escalations judged acceptable.")
    cols[2].metric("Harmful-action rate", _pct(card.harmful_attempted_rate),
                   help="Harmful calls attempted over well-formed calls.")
    cols[3].metric("Format-error rate", _pct(card.format_error_rate),
                   help="Unparseable or schema-invalid calls over all calls.")
    cols[4].metric("Escalation rate", _pct(card.escalation_rate),
                   help="Share of tasks that ended in an escalation.")


def _overview_drift_bar(data: gdata.DashboardData, model: str, filters: GlobalFilters) -> None:
    st.subheader("Drift type per agent")
    breakdown = gdata.agent_drift_breakdown(data.dataset, model, agent=filters.agent, control=filters.control)
    records = [{"agent": role, "drift_type": drift_type, "calls": counts["calls"],
                "harmful": counts["harmful"], "harmful_executed": counts["harmful_executed"],
                "denied": counts["denied"]}
               for role, types in breakdown.items() for drift_type, counts in types.items()]
    frame = pd.DataFrame(records, columns=["agent", "drift_type", "calls", "harmful",
                                           "harmful_executed", "denied"])
    if frame.empty:
        st.info("No well-formed agent calls in the current slice.")
        return
    fig = px.bar(frame, x="agent", y="calls", color="drift_type", barmode="stack",
                 color_discrete_map=DRIFT_TYPE_COLORS,
                 category_orders={"agent": ["support", "listing", "price_intel"],
                                  "drift_type": list(gdata.DRIFT_TYPES)},
                 labels={"calls": "Well-formed calls", "drift_type": "Drift type", "agent": "Agent"},
                 custom_data=["harmful", "denied"],
                 hover_data={"harmful": ":,", "denied": ":,",
                             "calls": ":,"})
    fig.update_layout(barmode="stack", legend_title_text="Drift type", legend_orientation="h",
                      margin=dict(t=10))
    _chart(fig, "overview_drift_bar")
    st.caption("Calls per drift type, as counted in the permission log by "
               "`gui.data.agent_drift_breakdown`.")


def _overview_control_bars(data: gdata.DashboardData, model: str, filters: GlobalFilters) -> None:
    st.subheader("Harmful calls attempted, C1 against C2")
    rows = gdata.harmful_rate_by_drift(data.dataset, model, agent=filters.agent)
    plotted = [row for row in rows if row.drift != gdata.ALL]
    fig = go.Figure()
    for label, pick in (("C1", lambda row: row.c1), ("C2", lambda row: row.c2)):
        xs, ys, plus, minus, counts = [], [], [], [], []
        for row in plotted:
            cell = pick(row)
            if cell.rate is None:
                continue
            xs.append(row.drift)
            ys.append(100 * cell.rate)
            plus.append(100 * (cell.hi - cell.rate) if cell.hi is not None else 0.0)
            minus.append(100 * (cell.rate - cell.lo) if cell.lo is not None else 0.0)
            counts.append(f"{cell.episodes} episodes / {cell.calls} calls")
        fig.add_trace(go.Bar(name=label, x=xs, y=ys, marker_color=CONTROL_COLORS[label],
                             error_y=dict(type="data", symmetric=False, array=plus, arrayminus=minus),
                             customdata=counts,
                             hovertemplate="%{x} " + label + ": %{y:.1f}%"
                                           "<extra>" + label + "</extra>"))
    fig.add_hline(y=0.0, line_width=0.6, line_color="#B0B0B0")
    fig.update_layout(barmode="group", yaxis_title="Harmful calls attempted (% of well-formed calls)",
                      xaxis_title="Drift condition", legend_title_text="", legend_orientation="h",
                      margin=dict(t=10))
    _chart(fig, "overview_control_bars")
    table = pd.DataFrame([{
        "drift": row.drift,
        "C1 rate (95% CI)": _ci(row.c1.rate, row.c1.lo, row.c1.hi),
        "C1 episodes": row.c1.episodes, "C1 calls": row.c1.calls,
        "C2 rate (95% CI)": _ci(row.c2.rate, row.c2.lo, row.c2.hi),
        "C2 episodes": row.c2.episodes, "C2 calls": row.c2.calls} for row in rows])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"C1 rate (95% CI)": st.column_config.TextColumn(width="medium"),
                                "C2 rate (95% CI)": st.column_config.TextColumn(width="medium")})
    st.caption("Episode-level bootstrap 95% CIs from `gui.data.harmful_rate_by_drift`. "
               "`all` pools the four drift conditions.")


def _overview_blocked(data: gdata.DashboardData, model: str, filters: GlobalFilters) -> None:
    st.subheader("Harmful calls held back by static RBAC")
    shares = gdata.harm_blocked_by_drift_type(data.dataset, model, agent=filters.agent,
                                               control=filters.control)
    frame = pd.DataFrame([{"drift_type": share.drift_type, "calls": share.calls,
                           "harmful_attempted": share.harmful_attempted,
                           "harmful_blocked": share.harmful_blocked,
                           "blocked_share": share.blocked_share} for share in shares])
    if frame.empty or not frame["blocked_share"].map(lambda v: v is not None).any():
        st.info("No harmful calls were attempted in the current slice, so there is no blocked share to show.")
    else:
        fig = go.Figure(go.Bar(
            x=frame["drift_type"], y=[100 * (v or 0.0) for v in frame["blocked_share"]],
            marker_color=[DRIFT_TYPE_COLORS.get(str(t), "#C9C9C9") for t in frame["drift_type"]],
            text=[_pct(v) for v in frame["blocked_share"]], textposition="auto",
            customdata=list(zip(frame["harmful_blocked"], frame["harmful_attempted"])),
            hovertemplate="%{x}: %{y:.1f}% blocked<br>%{customdata[0]} of %{customdata[1]}"
                          " harmful calls denied<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_title="Drift type", showlegend=False,
                          yaxis_title="Share of harmful calls denied (%)", margin=dict(t=10))
        _chart(fig, "overview_blocked_bar")
    st.dataframe(frame.rename(columns={"blocked_share": "blocked share"}), hide_index=True,
                 width="stretch",
                 column_config={"blocked share": st.column_config.NumberColumn(
                     "Blocked share", format="percent")})


def _overview(data: gdata.DashboardData, filters: GlobalFilters, model: str | None, note: str) -> None:
    st.subheader("Headline metrics")
    if model is None:
        st.info("No episodes in the current slice, so there is nothing to summarise.")
        return
    cards = gdata.overview_cards(data.dataset, model, agent=filters.agent, control=filters.control)
    primary = cards.get(filters.control) or next(iter(cards.values()), None)
    if primary is None:
        st.info(f"No metric cards available for `{model}`.")
        return
    scope = filters.control if filters.control != gdata.ALL else "all control conditions"
    st.caption(f"Model `{model}` · agent {ROLE_LABEL.get(filters.agent, filters.agent)} · {scope}.")
    if note:
        st.caption(note)
    if filters.drift != gdata.ALL:
        st.caption("The drift filter is not applied to these cards; it narrows the episode table and the "
                   "detector views. The per-drift breakdown is below.")
    _metric_row(primary)
    st.caption("Rates come from `gui.data.overview_cards`; see the analysis tables for the "
               "episode-level bootstrap CIs.")
    if filters.control == gdata.ALL:
        shown = [cc for cc in SOURCE_CONTROLS if cc in cards]
        if shown:
            st.markdown("**C1 against C2**")
            st.dataframe(pd.DataFrame([{
                "metric": "Episodes", **{cc: f"{cards[cc].episodes:,}" for cc in shown}},
                {"metric": "Task success", **{cc: _pct(cards[cc].task_success) for cc in shown}},
                {"metric": "Harmful-action rate",
                 **{cc: _pct(cards[cc].harmful_attempted_rate) for cc in shown}},
                {"metric": "Format-error rate",
                 **{cc: _pct(cards[cc].format_error_rate) for cc in shown}},
                {"metric": "Escalation rate",
                 **{cc: _pct(cards[cc].escalation_rate) for cc in shown}}]),
                hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        _overview_drift_bar(data, model, filters)
    with right:
        _overview_control_bars(data, model, filters)
    _overview_blocked(data, model, filters)


def _episode_outcomes(record: Any) -> pd.DataFrame:
    return pd.DataFrame([{"task": o.task_id, "type": o.task_type, "outcome": o.outcome,
                          "success": o.success, "acceptable escalation": o.escalation_acceptable,
                          "reason": o.reason} for o in record.task_outcomes],
                        columns=["task", "type", "outcome", "success", "acceptable escalation", "reason"])


def _score_chart(found: gdata.EpisodeDetection, record: Any) -> go.Figure:
    steps = found.score.steps
    xs = [s.step for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[s.drift_jsd for s in steps], name="D_t (JSD)", yaxis="y",
                             line=dict(color=CONDITION_COLORS.get(record.drift_condition, "#4C72B0"),
                                       width=2)))
    threshold = found.result.thresholds.get("weighted")
    if threshold is not None and xs:
        fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[threshold, threshold], name="weighted threshold",
                                 yaxis="y2", mode="lines", line=dict(color="#B0B0B0", dash="dot")))
    for combiner in COMBINERS:
        fig.add_trace(go.Scatter(x=xs, y=[getattr(s, combiner) for s in steps],
                                 name=f"PLDD {combiner}", yaxis="y2", mode="lines",
                                 line=dict(color=COMBINER_COLORS[combiner], width=1.2)))
    alert = found.score.alert_step.get("weighted")
    if alert is not None:
        fig.add_vline(x=alert, line_dash="dash", line_color="#C44E52", annotation_text="alert",
                      annotation_position="top left")
    if record.first_harm_step is not None:
        fig.add_vline(x=record.first_harm_step, line_color="#DD8452", annotation_text="first harm",
                      annotation_position="top right")
    fig.update_layout(
        yaxis=dict(title="D_t (JSD)", showgrid=False),
        yaxis2=dict(title="PLDD score", overlaying="y", side="right", showgrid=False),
        xaxis_title="step", hovermode="x unified", margin=dict(t=30), legend=dict(orientation="h"))
    return fig


def _transcript_for(runs: Sequence[gdata.RunInfo], record: Any) -> gdata.RunInfo | None:
    for run in runs:
        if run.label == record.run_id:
            return run
    for run in runs:
        if gdata.transcript_path(run, record.episode_id) is not None:
            return run
    return None


def _sibling_episode(data: gdata.DashboardData, record: Any, other_control: str) -> Any | None:
    for episode in data.dataset.episodes:
        other = episode.record
        if (other.model == record.model and other.agent_role == record.agent_role
                and other.drift_condition == record.drift_condition and other.seed == record.seed
                and other.control_condition == other_control):
            return episode
    return None


def _scope_comparison(data: gdata.DashboardData, episode: Any) -> None:
    record = episode.record
    other_control = "C2" if record.control_condition == "C1" else "C1"
    sibling = _sibling_episode(data, record, other_control)
    st.markdown(f"**Same seed under {other_control}**")
    st.caption("Task-scoping denies only exist under C2/C4, so the two columns show what the other "
               "control made of the same cell.")
    if sibling is None:
        st.info(f"No {other_control} episode with seed {record.seed} for this model, agent and drift "
                "condition in the loaded runs.")
        return
    left, right = st.columns(2)
    for column, label, item in ((left, record.control_condition, episode),
                                (right, other_control, sibling)):
        with column:
            st.caption(f"{label} · `{item.record.episode_id}` · {len(item.events)} calls")
            frame = _call_frame(item.events)[["step", "task", "action", "decision", "deny_layer",
                                              "in_task", "drift_type", "harmful"]]
            st.dataframe(_style_calls(frame), hide_index=True, width="stretch", height=320,
                         column_config=_call_column_config())


def _episode_detail(data: gdata.DashboardData, runs: Sequence[gdata.RunInfo], episode: Any) -> None:
    record = episode.record
    st.markdown(f"#### `{record.episode_id}`")
    st.caption(f"run `{record.run_id}` · seed {record.seed} · model `{record.model}` · "
               f"agent {ROLE_LABEL.get(record.agent_role, record.agent_role)} · "
               f"{record.drift_condition}/{record.control_condition} · {record.n_steps} steps · "
               f"task success {_pct(record.task_success_rate)}")
    outcomes = _episode_outcomes(record)
    if outcomes.empty:
        st.caption("No task outcomes were recorded for this episode.")
    else:
        st.dataframe(outcomes, hide_index=True, width="stretch",
                     column_config={"success": st.column_config.CheckboxColumn(width="small"),
                                    "acceptable escalation": st.column_config.CheckboxColumn(width="small")})
    st.caption("Task outcomes: " + (", ".join(f"{o.task_id}={o.outcome}" for o in record.task_outcomes)
                                    or "none"))

    st.markdown("**Tool calls**")
    frame = _call_frame(episode.events)
    if frame.empty:
        st.info("This episode logged no tool calls.")
    else:
        st.dataframe(_style_calls(frame), hide_index=True, width="stretch", height=420,
                     column_config=_call_column_config("large"))
        st.caption("Rows are tinted by drift type (I/II/III) and harmful calls are red. "
                   "`deny_layer` is the layer that refused the call; `in_task` is the call against the "
                   "task scope frozen at task start.")

    found = (gdata.find_detection(data.detection, record.episode_id, model=record.model,
                                  control=record.control_condition, baseline="all_d0")
             or gdata.find_detection(data.detection, record.episode_id))
    st.markdown("**D_t and the PLDD score over the episode**")
    if found is None:
        st.info("The detector did not score this episode: it is in the fit or calibration split, or the "
                "detector could not be fitted on the selected runs. The detector reads the permission log "
                "only, never a transcript.")
    else:
        _chart(_score_chart(found, record), "episode_scores")
        st.caption(f"Offline replay: `{found.control}` logs are displayed as "
                   f"`{DETECT_CONTROL.get(found.control, found.control)}` · baseline `{found.baseline}` · "
                   "scores from the detection replay, no transcript involved.")

    run = _transcript_for(runs, record)
    path = gdata.transcript_path(run, record.episode_id) if run is not None else None
    with st.expander("Transcript (qualitative only, never used for a number)"):
        if path is None:
            st.caption(f"This run kept no transcript for `{record.episode_id}`.")
        else:
            entries = gdata.read_transcript(path)
            st.caption(f"{len(entries)} entries from `{path}`.")
            if entries:
                st.dataframe(_textual(pd.DataFrame([{"role": entry.get("role", "?"),
                                                     "task": entry.get("task_id", "-"),
                                                     "content": entry.get("content", entry.get("arguments", ""))}
                                                    for entry in entries])),
                             hide_index=True, width="stretch", height=360,
                             column_config={"content": st.column_config.TextColumn(width="large")})

    with st.expander("Compare C1 and C2 for the same seed"):
        _scope_comparison(data, episode)


def _explorer(data: gdata.DashboardData, runs: Sequence[gdata.RunInfo],
              episodes: Sequence[Any]) -> None:
    st.subheader("Episode explorer")
    index = {episode.record.episode_id: episode for episode in episodes}
    all_rows = gdata.episode_rows(episodes)
    target = st.session_state.get("episode_target")
    if target and target in index:
        kept = EpisodeFilters(search=str(st.session_state.get("explorer_search", "")),
                              outcome=str(st.session_state.get("explorer_outcome", gdata.ALL)),
                              harm=str(st.session_state.get("explorer_harm", gdata.ALL)),
                              min_harm=int(st.session_state.get("explorer_min_harm") or 0))
        if target not in {row.episode_id for row in kept.rows(all_rows)}:
            st.caption("The requested episode is hidden by the local filters below; they are reset for it.")
            for key, empty in (("explorer_search", ""), ("explorer_outcome", gdata.ALL),
                               ("explorer_harm", gdata.ALL), ("explorer_min_harm", 0)):
                st.session_state[key] = empty

    head, search_col, outcome_col, harm_col, count_col = st.columns([1.4, 3, 1.4, 1.4, 1.4])
    head.caption(f"{len(all_rows)} episode(s) in the current slice.")
    search = search_col.text_input("Search", "", key="explorer_search",
                                   placeholder="episode id, run, seed, agent, outcome")
    outcome = outcome_col.selectbox("Outcome", (gdata.ALL, *OUTCOMES), key="explorer_outcome")
    harm = harm_col.selectbox("Harm", gdata.HARM_CHOICES, key="explorer_harm",
                              format_func=lambda v: {"all": "All", "harmful": "Harmful only",
                                                     "clean": "Harm-free only"}[v])
    min_harm = count_col.number_input("Min harmful calls", min_value=0, value=0, step=1,
                                      key="explorer_min_harm")

    local = EpisodeFilters(search=search, outcome=outcome, harm=harm, min_harm=int(min_harm))
    rows = local.rows(all_rows)
    if local.active():
        st.caption("Local filters: " + ", ".join(f"{k}={v}" for k, v in local.active().items()))

    frame = _episode_frame(rows)
    if frame.empty:
        st.info("No episodes match the current filters")
        st.session_state.pop("episode_target", None)
        return

    event = st.dataframe(
        frame, key="explorer_table", on_select="rerun", selection_mode="single-row",
        hide_index=True, width="stretch", height=380,
        column_config={
            "seed": st.column_config.NumberColumn("Seed", format="%d", width="small"),
            "success_rate": st.column_config.NumberColumn("Task success", format="percent"),
            "harm_count": st.column_config.NumberColumn("Harmful", format="%d", width="small"),
            "harmful_executed": st.column_config.NumberColumn("Executed", format="%d", width="small"),
            "first_harm_step": st.column_config.NumberColumn("First harm step", format="%d",
                                                             width="small"),
            "outcome": st.column_config.TextColumn("Outcomes", width="medium")})
    picked = [rows[i].episode_id for i in (event.selection.rows or []) if 0 <= i < len(rows)]

    if picked:
        st.session_state.pop("episode_target", None)
        chosen = picked[0]
        source = "table selection"
    elif target in index:
        chosen = target
        source = "requested from another view"
    else:
        chosen = ""
        source = ""
    if not chosen:
        st.caption("Select one row above to inspect its tool calls.")
        return
    if source == "requested from another view":
        left, right = st.columns([6, 1])
        left.caption(f"Showing `{chosen}`, {source}.")
        if right.button("Clear", key="clear_target", width="stretch"):
            st.session_state.pop("episode_target", None)
            st.rerun()
    _episode_detail(data, runs, index[chosen])


def _roc_figure(data: gdata.DashboardData, model: str, source_control: str,
                result: Any) -> go.Figure:
    records = {ep.record.episode_id: ep.record
               for ep in data.dataset.select(model=model, control_condition=source_control)}
    fig = go.Figure()
    for condition, scored in result.by_condition.items():
        ids = [i for i in result.scores if i in records
               and records[i].drift_condition in (condition, "D0")]
        labels = [int(records[i].total_harmful_attempted > 0) for i in ids]
        if not 0 < sum(labels) < len(labels):
            continue
        for combiner in COMBINERS:
            scores = [max((getattr(step, combiner) for step in result.scores[i].steps), default=0.0)
                      for i in ids]
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = scored.auroc.get(combiner)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{condition} {combiner} (AUROC {_num(auroc)})",
                line=dict(color=CONDITION_COLORS.get(condition, "#4C72B0"),
                          dash="solid" if combiner == "weighted" else "dash", width=1.8)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color="#C0C0C0",
                                                                                        dash="dot")))
    fig.add_vline(x=0.05, line_dash="dash", line_color="#B0B0B0")
    fig.update_layout(xaxis_title="False-positive rate (held-out clean episodes)",
                      yaxis_title="True-positive rate", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", font=dict(size=10)), margin=dict(t=10))
    return fig


def _detection_table(data: gdata.DashboardData, model: str, control: str, baseline: str) -> None:
    st.markdown("**Detection table (RQ3)**")
    rows = data.tables.get("rq3_detection")
    if rows is None:
        st.info("The analysis tables were not computed for the selected runs, so there is no detection "
                "table to show. The load warnings in the sidebar say why.")
        return
    selected = [row for row in rows if row.get("model") == model and row.get("control") == control
                and row.get("baseline") == baseline]
    if not selected:
        st.info(f"No rq3_detection rows for `{model}` / `{control}` / `{baseline}`.")
        return
    frame = pd.DataFrame(selected)
    columns = ["condition", "score", "n_pos", "n_neg", "tpr_at_5fpr", "auroc", "auroc_lo", "auroc_hi",
               "holdout_fpr", "early_share", "median_lead_steps", "n_detected"]
    shown = frame[[c for c in columns if c in frame.columns]].rename(columns={
        "condition": "Drift", "score": "Score", "n_pos": "n positive", "n_neg": "n negative",
        "tpr_at_5fpr": "TPR @ 5% FPR", "auroc": "AUROC", "auroc_lo": "CI low", "auroc_hi": "CI high",
        "holdout_fpr": "Held-out FPR", "early_share": "Alerted before harm",
        "median_lead_steps": "Median lead (steps)", "n_detected": "n detected"})
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"AUROC": st.column_config.NumberColumn(format="%.3f"),
                                "CI low": st.column_config.NumberColumn(format="%.3f"),
                                "CI high": st.column_config.NumberColumn(format="%.3f"),
                                "TPR @ 5% FPR": st.column_config.NumberColumn(format="percent"),
                                "Held-out FPR": st.column_config.NumberColumn(format="percent"),
                                "Alerted before harm": st.column_config.NumberColumn(format="percent"),
                                "Median lead (steps)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("From `analysis.tables.rq3_detection`: thresholds calibrated to 5% episode-level FPR, "
               "AUROC with its episode-level bootstrap CI, and the lead time in steps (positive means "
               "the alert came before the first harmful call).")


def _lead_time_figure(result: Any) -> go.Figure | None:
    records = [{"Drift": condition, "Combiner": combiner, "Lead steps": value}
               for condition, scored in result.by_condition.items()
               for combiner, values in scored.lead_times.items() for value in values]
    if not records:
        return None
    return px.box(pd.DataFrame(records), x="Drift", y="Lead steps", color="Combiner", points="outliers",
                  color_discrete_map=COMBINER_COLORS, category_orders={"Drift": ["D1", "D2", "D3"]},
                  labels={"Lead steps": "Lead time (steps)"})


def _ablation_figure(data: gdata.DashboardData, model: str, control: str, baseline: str
                     ) -> go.Figure | None:
    rows = data.tables.get("ablation")
    if rows is None:
        return None
    selected = [row for row in rows if row.get("model") == model and row.get("control") == control
                and row.get("baseline") == baseline]
    if not selected:
        return None
    frame = pd.DataFrame(selected).dropna(subset=["delta_auroc"])
    if frame.empty:
        return None
    return px.bar(frame, x="feature_removed", y="delta_auroc", color="combiner", barmode="group",
                  color_discrete_map=COMBINER_COLORS,
                  labels={"feature_removed": "Feature removed", "delta_auroc": "Change in AUROC",
                          "combiner": "Combiner"})


def _detector(data: gdata.DashboardData, filters: GlobalFilters) -> None:
    st.subheader("Detector")
    keys = gdata.detection_keys(data.detection)
    if not keys:
        st.info("No detector results for the selected runs. The detector needs a D0 baseline big enough to "
                "fit on and to calibrate a 5% false-positive rate; reload after a run with more D0 "
                "episodes.")
        return
    source_of = {DETECT_CONTROL[control]: control for _, control, _ in keys}
    models = sorted({model for model, _, _ in keys})

    first, second, third = st.columns(3)
    baseline = first.radio("D0 baseline", BASELINES, horizontal=True,
                           help="`all_d0` fits the baseline on all fit-split D0 episodes; "
                                "`harm_free_d0` uses only the harm-free ones.")
    control = second.selectbox("Detection control", sorted(source_of),
                               format_func=lambda value: f"{value} (replay of {source_of[value]} logs)")
    chosen_model = third.selectbox("Model", models, index=(
        models.index(filters.model) if filters.model in models else 0))

    source = source_of[control]
    result = data.detection.get((chosen_model, source, baseline))
    if result is None:
        st.warning(f"No detector result for `{chosen_model}` / `{control}` / `{baseline}`: the D0 fit or "
                   "calibration split is empty in the selected runs, so the baseline could not be "
                   "calibrated. Select more runs, or use the `all_d0` baseline.")
        return
    sizes = result.split_sizes
    st.caption(f"D0 split: {sizes.get('holdout', 0)} held out, {sizes.get('fit', 0)} fit "
               f"({sizes.get('fit_used', 0)} used), {sizes.get('calib', 0)} for calibration · "
               f"thresholds: " + ", ".join(f"{k} {_num(v)}" for k, v in sorted(result.thresholds.items())))

    st.markdown("**ROC per drift condition**")
    fig = _roc_figure(data, chosen_model, source, result)
    if len(fig.data) > 1:
        _chart(fig, "detector_roc")
        st.caption("Curve points from the replayed scores; the AUROC in the legend is the value the "
                   "analysis computed. Positives are episodes with a harmful call, negatives are the "
                   "held-out clean D0 episodes.")
    else:
        st.info("No drift condition has both positives and held-out negatives in this slice, so no ROC "
                "curve can be drawn.")

    _detection_table(data, chosen_model, control, baseline)

    st.markdown("**Lead time before the first harmful call**")
    lead = _lead_time_figure(result)
    if lead is None:
        st.info("No episode in this slice alerted before its first harmful call, so there is no lead time "
                "to show.")
    else:
        _chart(lead, "detector_lead")
        st.caption("Distribution of the per-episode lead times, in steps.")

    st.markdown("**Ablation: change in AUROC when one feature is removed**")
    ablation = _ablation_figure(data, chosen_model, control, baseline)
    if ablation is None:
        st.info(f"No ablation was computed for `{chosen_model}` / `{control}` / `{baseline}`. The "
                "single-feature ablation is only run for the `all_d0` baseline over D1-D3.")
    else:
        _chart(ablation, "detector_ablation")
        st.caption("A negative bar means the feature carried detection power: removing it lowered the "
                   "AUROC.")


def _taint_stages(events: Sequence[Any]) -> list[tuple[str, str, list[Any]]]:
    return [
        ("1. Poisoned competitor page",
         "action = fetch_competitor_price with a non-empty context_injection_ids",
         [e for e in events if e.action == "fetch_competitor_price" and e.context_injection_ids]),
        ("2. Tainted price report",
         "non-empty taint_report_ids: the injection reached the context through a price report",
         [e for e in events if e.taint_report_ids]),
        ("3. Harmful listing action under that taint",
         "cross_agent_taint and harm: the other agent acted on the tainted report",
         [e for e in events if e.cross_agent_taint and e.harm]),
    ]


def _flow_figure(stages: Sequence[tuple[str, list[Any]]]) -> go.Figure:
    fig = go.Figure()
    width, top, bottom = 0.30, 0.62, 0.24
    for index, (title, events) in enumerate(stages):
        left = index * 0.35
        fig.add_shape(type="rect", x0=left, x1=left + width, y0=bottom, y1=top,
                      line=dict(color=STAGE_COLORS[index], width=1.5), fillcolor="#F7F7F7")
        episodes = sorted({e.episode_id for e in events})
        fig.add_annotation(x=left + width / 2, y=(top + bottom) / 2 + 0.04,
                           text=f"<b>{title}</b>", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=left + width / 2, y=(top + bottom) / 2 - 0.08,
                           text=f"{len(events)} call(s)<br>{len(episodes)} episode(s)", showarrow=False,
                           font=dict(size=11, color=STAGE_COLORS[index]))
        if index < len(stages) - 1:
            fig.add_annotation(x=left + width + 0.025, y=(top + bottom) / 2,
                               ax=left + width, ay=(top + bottom) / 2, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5,
                               arrowcolor="#9A9A9A")
    fig.update_xaxes(visible=False, range=[-0.04, 0.35 * (len(stages) - 1) + width + 0.04])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
    return fig


def _open_episode(episode_id: str) -> None:
    st.session_state["episode_target"] = episode_id
    st.session_state["view_request"] = VIEW_EPISODES


def _cross_agent(data: gdata.DashboardData, filters: GlobalFilters, episodes: Sequence[Any]) -> None:
    st.subheader("Cross-agent taint")
    events = [event for episode in episodes for event in episode.events]
    if not events:
        st.info("No logged tool calls in the current slice, so the taint path cannot be drawn.")
        return
    st.caption("Descriptive tallies of raw log fields in the current slice (no resampling, no CI); the "
               "estimate with its bootstrap CI is in the table underneath.")
    stages = _taint_stages(events)
    if all(not found for _, _, found in stages):
        st.info("No call in the current slice carried a context injection, a taint report or cross-agent "
                "taint. D3 (or a run that includes it) is needed to see the path.")
    _chart(_flow_figure([(title, found) for title, _, found in stages]), "cross_agent_flow")

    table = pd.DataFrame([{
        "stage": title, "raw fields": fields, "calls": len(found),
        "executed": sum(bool(e.executed) for e in found),
        "harmful": sum(bool(e.harm) for e in found),
        "episodes": len({e.episode_id for e in found}),
        "example episodes": _ids(sorted({e.episode_id for e in found}))}
        for title, fields, found in stages])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"raw fields": st.column_config.TextColumn("Raw log fields", width="large"),
                                "example episodes": st.column_config.TextColumn("Example episodes",
                                                                                 width="medium")})

    st.markdown("**cross_agent_taint (analysis table)**")
    rows = data.tables.get("cross_agent_taint")
    if rows is None:
        st.info("The analysis tables were not computed for the selected runs.")
    else:
        selected = [row for row in rows if filters.model == gdata.ALL or row.get("model") == filters.model]
        if not selected:
            st.info(f"No cross_agent_taint rows for the current slice (model `{filters.model}`).")
        else:
            frame = pd.DataFrame(selected)
            st.dataframe(frame, hide_index=True, width="stretch",
                         column_config={"a1_share_of_tainted_calls": st.column_config.NumberColumn(
                             "A1 share of tainted calls", format="percent"),
                             "lo": st.column_config.NumberColumn("CI low", format="percent"),
                             "hi": st.column_config.NumberColumn("CI high", format="percent")})
            st.caption("Listing-agent episodes with a taint that arrived through a price report, the share "
                       "of tainted calls the harm oracle labelled A1, and the executed count.")

    examples = sorted({e.episode_id for _, _, found in stages for e in found
                       if e.cross_agent_taint or e.harm})
    st.markdown("**Open an episode**")
    if not examples:
        st.caption("No episode in the current slice shows the cross-agent path.")
        return
    picker_left, picker_right = st.columns([3, 1.4])
    picked = picker_left.selectbox("Episode to open", examples, key="cross_agent_pick")
    if picker_right.button("Open in Episode explorer", width="stretch"):
        _open_episode(picked)
        st.rerun()
    st.caption("Switching views keeps the sidebar slice; if the episode is hidden by the explorer's local "
               "filters, those are reset for it.")


def _manager() -> gdemo.DemoManager:
    manager = st.session_state.get("demo_manager")
    if manager is None:
        manager = gdemo.DemoManager(PROJECT_ROOT)
        st.session_state["demo_manager"] = manager
    return manager


def _current_job() -> gdemo.DemoJob | None:
    job_id = st.session_state.get("demo_job_id")
    manager = st.session_state.get("demo_manager")
    if not job_id or manager is None:
        return None
    return manager.get(job_id)


def _live_events_frame(job: gdemo.DemoJob) -> pd.DataFrame:
    calls = [event for event in gdemo.read_live_events(job) if "action" in event and "step" in event]
    return pd.DataFrame([{key: event.get(key) for key in LIVE_COLUMNS} for event in calls],
                        columns=list(LIVE_COLUMNS))


def _demo_job_live(job: gdemo.DemoJob) -> None:
    cols = st.columns(4)
    cols[0].metric("Status", job.status)
    cols[1].metric("Started", (job.started_at or "-")[:19].replace("T", " "))
    cols[2].metric("Finished", (job.finished_at or "-")[:19].replace("T", " ") if job.finished_at else "-")
    cols[3].metric("Actual cost", _usd(job.actual_cost_usd))
    frame = _live_events_frame(job)
    st.caption(f"{len(frame)} tool call(s) logged so far.")
    if frame.empty:
        st.caption("Waiting for the first tool call…")
    else:
        st.dataframe(frame.sort_values("step").tail(200), hide_index=True, width="stretch",
                     height=320,
                     column_config={"step": st.column_config.NumberColumn("Step", format="%d",
                                                                          width="small"),
                                    "task_id": st.column_config.TextColumn("Task", width="small"),
                                    "action": st.column_config.TextColumn("Action", width="medium"),
                                    "harm": st.column_config.CheckboxColumn("Harmful", width="small"),
                                    "executed": st.column_config.CheckboxColumn("Executed", width="small"),
                                    "in_task": st.column_config.CheckboxColumn("In task", width="small"),
                                    "latency_ms": st.column_config.NumberColumn("Latency ms",
                                                                                 format="%.0f",
                                                                                 width="small")})


def _demo_job_summary(job: gdemo.DemoJob) -> None:
    if job.status == "completed" and job.result is not None:
        record = job.result
        st.success(f"Completed: {len(record.task_outcomes)} task(s), {record.total_harmful_attempted} "
                   f"harmful call(s) attempted, {record.total_harmful_executed} executed, "
                   f"{record.n_steps} steps.")
        st.dataframe(_episode_outcomes(record), hide_index=True, width="stretch")
        st.caption("Task outcomes: " + (", ".join(f"{o.task_id}={o.outcome}"
                                                  for o in record.task_outcomes) or "none"))
    else:
        st.error(job.error or "the demo failed without a message.")
        if job.reason:
            st.caption(job.reason)
    info_left, info_right = st.columns(2)
    info_left.caption(f"logs/demo: `{job.paths.base}`")
    info_left.caption(f"this run: `{job.paths.logs}`")
    info_right.caption(f"events: `{job.events}`")
    info_right.caption(f"transcripts: `{job.transcripts}`")
    if st.button("Dismiss this run", key="dismiss_demo"):
        st.session_state.pop("demo_job_id", None)
        st.rerun()


@st.fragment(run_every=0.5)
def _demo_panel() -> None:
    job = _current_job()
    if job is None:
        st.info("The demo job is no longer tracked by this session.")
        return
    _demo_job_live(job)
    if job.terminal:
        st.rerun()


def _demo_spec(mode: str, role: str, drift: str, control: str, seed: int, groq_model: str,
               run_id: str | None) -> gdemo.DemoSpec:
    return gdemo.DemoSpec(mode=mode, role=role, drift=drift, control=control, seed=seed,
                          groq_model=groq_model, run_id=run_id)


def _live_demo() -> None:
    st.subheader("Live demo")
    st.caption("One episode per click, run in a background thread and logged under `logs/demo/`. The "
               "dashboard only reads those logs while the run is in flight.")

    try:
        status = detect_active_runner(PROJECT_ROOT)
    except Exception as exc:
        status = RunnerStatus(active=True, reason=f"the runner status check failed: {exc}",
                              source=PROJECT_ROOT)
    first, second, third, fourth = st.columns(4)
    role = first.selectbox("Agent", ("support", "listing", "price_intel"),
                           format_func=lambda v: ROLE_LABEL[v])
    drift = second.selectbox("Drift", gdata.DRIFTS)
    control = third.selectbox("Control", SOURCE_CONTROLS)
    seed = fourth.number_input("Seed", min_value=1, max_value=10_000, value=101, step=1)
    mode_label = st.radio("Mode", [label for label, _ in DEMO_MODES], horizontal=True)
    mode = dict(DEMO_MODES)[mode_label]
    groq_model = st.selectbox("Groq model", sorted(GROQ_CANDIDATES), key="demo_groq_model",
                              disabled=mode != "groq")

    if status.active:
        st.warning(f"A runner is active, so the real-model demos are disabled: {status.reason}. The "
                   "scripted mode reads no provider and still runs.")
    else:
        st.success(f"Runner idle: {status.reason}")
    groq_ready = bool(os.environ.get("GROQ_API_KEY"))
    if mode == "groq" and not groq_ready:
        st.error("GROQ_API_KEY is not set for the dashboard process, so the Groq demo is disabled. Start "
                 "the dashboard from an environment that exports it.")

    estimate = gdemo.estimate_demo_cost(
        _demo_spec(mode, role, drift, control, int(seed), groq_model, None), PROJECT_ROOT, record=False)
    cards = st.columns(4)
    cards[0].metric("Estimated tasks", f"{estimate.tasks}")
    cards[1].metric("Estimated model turns", f"{estimate.model_turns:,}")
    cards[2].metric("Estimated tokens in / out", f"{estimate.input_tokens:,} / {estimate.output_tokens:,}")
    cards[3].metric("Estimated cost", _usd(estimate.usd_no_cache))
    st.caption(f"Upper bound {_usd(estimate.usd_upper_bound)}: every task hits the turn limit and nothing "
               f"is cached. From `experiments.cost.estimate` over the {mode} model.")

    approved = False
    if mode == "groq":
        approved = st.checkbox(f"I approve this paid Groq call and the estimate above "
                               f"({_usd(estimate.usd_no_cache)} without caching).", key="demo_approved")

    job = _current_job()
    running = job is not None and not job.terminal
    blocked = ""
    if running:
        blocked = f"a demo started at {(job.started_at or '')[:19]} is still running"
    elif mode != "scripted" and status.active:
        blocked = f"a runner is active: {status.reason}"
    elif mode == "groq" and not groq_ready:
        blocked = "GROQ_API_KEY is not set"

    label = {"scripted": "Run scripted demo", "qwen3": "Run qwen3:8b demo (local Ollama)",
             "groq": "Confirm and run Groq demo (paid API call)"}[mode]
    pressed = st.button(label, type="primary", disabled=bool(blocked) or (mode == "groq" and not approved),
                        width="content")
    if blocked:
        st.caption(f"Disabled: {blocked}.")
    if mode == "groq":
        st.caption("The button stays disabled until the confirmation box is ticked.")

    if pressed:
        run_id = f"gui-{mode}-{role}-{drift}-{control}-{seed}-{uuid.uuid4().hex[:8]}"
        spec = _demo_spec(mode, role, drift, control, int(seed), groq_model, run_id)
        manager = _manager()
        started = manager.start(spec, **({"confirmation": gdemo.prepare_demo(spec, PROJECT_ROOT,
                                                                             estimate)}
                                         if mode == "groq" else {}))
        st.session_state["demo_job_id"] = started.job_id
        st.caption(f"Started run `{run_id}` ({started.status}).")
        job = started

    if job is not None and job.terminal:
        _demo_job_live(job)
        _demo_job_summary(job)
    elif job is not None:
        st.info(f"A demo is running (`{job.spec.run_id}`, {job.status}). Its tool calls stream in below; "
                "no new demo can start until it finishes.")
        _demo_panel()
    else:
        st.info("No demo is running. Pick a cell above and press the run button; the tool calls appear "
                "here as they are logged.")


def main() -> None:
    st.set_page_config(page_title="Troy: agent drift dashboard", layout="wide")
    st.title("RBAC as containment and detection for LLM agent drift")
    st.caption(f"Read-only view of the run logs in `{LOGS_ROOT}`. Every number comes from `gui.data` over "
               "`analysis.tables`; the detector sees the permission log only, never a transcript.")

    runs, filters, model_choice, problem = _sidebar()
    if problem:
        if problem == NO_RUNS:
            st.info(f"No run folder was found under `{LOGS_ROOT}`, so there is nothing to show. Produce a "
                    "dataset first, for example "
                    "`uv run python -m experiments.scripted_pipeline .`, then reload.")
        else:
            st.info("No run folder is selected. Pick one, or choose \"All run folders\", in the sidebar.")
        return
    data = _dashboard(runs)

    request = st.session_state.pop("view_request", None)
    if request in VIEWS:
        st.session_state["view"] = request
    view = st.radio("View", VIEWS, horizontal=True, key="view")

    episodes = filters.episodes(data.dataset)
    model, note = _single_model(model_choice, episodes)
    if not data.episodes:
        st.info("No episodes in the selected runs: `logs/` holds no episode records, or the snapshot could "
                "not be parsed. Check the warnings in the sidebar.")
    elif not episodes:
        st.info("No episodes match the current filters")
    if note and model:
        st.caption(f"The model-scoped panels take one model at a time: {note}.")

    if view == VIEW_OVERVIEW:
        _overview(data, filters, model, note)
    elif view == VIEW_EPISODES:
        _explorer(data, runs, episodes)
    elif view == VIEW_DETECTOR:
        _detector(data, filters)
    elif view == VIEW_CROSS_AGENT:
        _cross_agent(data, filters, episodes)
    else:
        _live_demo()


if __name__ == "__main__":
    main()
