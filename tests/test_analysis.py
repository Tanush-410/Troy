"""Analysis pipeline (milestone 8), end to end on scripted-agent logs."""

import csv
import json
from collections import Counter

import pytest

from analysis import stats
from analysis.run_all import run
from analysis.stats import bootstrap_ci, fisher, mann_whitney, ratio_stat
from analysis.tables import Dataset
from experiments.scripted_dataset import generate


def test_bootstrap_resamples_units_and_is_seeded():
    units = [(1, 10), (0, 10), (5, 10), (2, 10)]
    stat = ratio_stat(lambda u: u[0], lambda u: u[1])
    a, b = bootstrap_ci(units, stat, n_boot=300), bootstrap_ci(units, stat, n_boot=300)
    assert a == b and a[0] == pytest.approx(0.2) and a[1] <= a[0] <= a[2]
    assert bootstrap_ci([], stat) == (None, None, None)


def test_fisher_and_mann_whitney():
    assert fisher(10, 0, 0, 10) < 0.001 and fisher(5, 5, 5, 5) == pytest.approx(1.0)
    assert fisher(0, 0, 1, 1) is None
    assert mann_whitney([1, 2, 3], [10, 11, 12]) < 0.2 and mann_whitney([], [1]) is None


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    saved, stats.N_BOOT = stats.N_BOOT, 200  # keep the test fast; the real pipeline uses 2000
    try:
        root = tmp_path_factory.mktemp("pipeline")
        paths, records = generate(root, "ds", 6)
        out = root / "results"
        index = run([paths.logs], out)
        yield paths, records, out, index
    finally:
        stats.N_BOOT = saved


def read(out, name):
    with (out / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def test_every_output_is_written(results):
    _, _, out, index = results
    expected = {"rq1_drift_types.csv", "rq2_controls.csv", "rq3_detection.csv", "rq3_lead_time_tests.csv",
                "ablation.csv", "d0_harm.csv", "format_errors.csv", "context_overflow_escalation.csv",
                "cross_agent_taint.csv", "summary.md", "harm_rates_scripted.png"}
    assert expected <= set(index["files"])
    assert any(f.startswith("roc_") for f in index["files"]) and any(f.startswith("dt_") for f in index["files"])
    for f in index["files"]:
        assert (out / f).stat().st_size > 0


def test_rq1_counts_match_the_raw_log(results):
    paths, _, out, _ = results
    events = [json.loads(line) for line in paths.events.read_text().splitlines()]
    c1 = Counter(e["drift_type"] for e in events if e["control_condition"] == "C1" and not e["format_error"])
    row = next(r for r in read(out, "rq1_drift_types") if r["agent"] == "all" and r["control"] == "C1")
    for t in ("I", "II", "III"):
        assert int(row[f"type_{t}"]) == c1[t]
    assert int(row["calls"]) == sum(c1.values())


def test_rq2_rates_match_and_cis_bracket_the_point(results):
    paths, _, out, _ = results
    ds = Dataset.load([paths.logs])
    for r in read(out, "rq2_controls"):
        for k in ("harm_executed_rate_c1", "task_success_rate_c2"):
            if r[k]:
                assert float(r[f"{k}_lo"]) - 1e-9 <= float(r[k]) <= float(r[f"{k}_hi"]) + 1e-9
        p = r["fisher_p_episodes_with_executed_harm"]
        assert p == "" or 0 <= float(p) <= 1
    eps = ds.select(model="scripted", control_condition="C1")
    direct = sum(e.harm and e.executed for ep in eps for e in ep.wf) / sum(len(ep.wf) for ep in eps)
    row = next(r for r in read(out, "rq2_controls") if r["agent"] == "all")
    assert float(row["harm_executed_rate_c1"]) == pytest.approx(direct)


def test_rq3_and_ablation_present_for_both_controls_and_baselines(results):
    _, _, out, _ = results
    rows = read(out, "rq3_detection")
    assert {(r["control"], r["baseline"]) for r in rows} == {
        ("C3", "all_d0"), ("C3", "harm_free_d0"), ("C4", "all_d0"), ("C4", "harm_free_d0")}
    assert {r["condition"] for r in rows} == {"D1", "D2", "D3"}
    assert {r["feature_removed"] for r in read(out, "ablation")} == {
        "deny_rate", "novel_action_rate", "action_freq_z", "param_z", "seq_surprise", "expansion_rate"}


def test_summary_has_every_section_and_no_nan(results):
    _, _, out, _ = results
    text = (out / "summary.md").read_text()
    for heading in ("RQ1", "RQ2", "RQ3", "Other rates", "Cross-agent taint"):
        assert heading in text
    assert "nan" not in text.lower().replace("n/a", "")


def test_rerun_regenerates_identical_tables(results):
    paths, _, out, _ = results
    out2 = out.parent / "results2"
    run([paths.logs], out2)
    for name in ("rq1_drift_types", "rq2_controls", "rq3_detection", "d0_harm"):
        assert (out / f"{name}.csv").read_text() == (out2 / f"{name}.csv").read_text()
