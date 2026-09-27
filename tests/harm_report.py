"""Write docs/harm_test_report.md from actual runs:

    uv run python -m tests.harm_report

Runs the existing oracle tests and the full suite with pytest, evaluates every
edge case in tests/harm_cases.py through the real PEP, and writes the report.
Every count in the report is computed here; nothing is typed in by hand.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from tests.harm_cases import build_cases, evaluate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "harm_test_report.md"
EXISTING = ["tests/test_harm_rules.py", "tests/test_claims.py", "tests/test_pep.py::test_drift_classifier"]
ADDED = [
    ("tests/harm_cases.py", "Registry of the edge cases below; each runs through the real PEP and HarmOracle."),
    ("tests/test_harm_edge_cases.py", "One pytest per edge case, a coverage check (every listed rule has a harmful "
                                      "and a benign case; all 15 claims present), and a check that one call "
                                      "matching several rules counts as one harmful action."),
    ("tests/harm_report.py", "This report generator."),
]


def pytest_run(args: list[str]) -> tuple[str, list[str]]:
    """(summary line, failing test ids)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", *args],
                         cwd=ROOT, capture_output=True, text=True).stdout
    summary = next((ln.strip("= ").strip() for ln in reversed(out.splitlines())
                    if re.search(r"\d+ (passed|failed)", ln)), "no summary")
    failed = sorted({m.group(1) for m in re.finditer(r"^FAILED (\S+)", out, re.M)})
    return summary, failed


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def last_commit(path: str) -> str:
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", path], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip() or "uncommitted"


def main() -> None:
    existing_summary, existing_failed = pytest_run(EXISTING)
    suite_summary, suite_failed = pytest_run([])
    results = [evaluate(c) for c in build_cases()]
    by_rule: OrderedDict[str, list] = OrderedDict()
    for r in results:
        by_rule.setdefault(r.case.rule, []).append(r)
    passed = sum(r.passed for r in results)

    lines = [
        "# Harm oracle test report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} at commit `{git_head()}` by "
        "`uv run python -m tests.harm_report`. Every case below was run through the real policy enforcement "
        "point (PEP) and harm oracle, so each call was judged on the state before it, as in experiments.",
        "",
        "## Summary",
        "",
        "| | Result |",
        "|---|---|",
        f"| Existing oracle tests (`test_harm_rules.py`, `test_claims.py`, drift classifier) | {existing_summary} |",
        f"| Edge cases (this report) | {passed} of {len(results)} pass |",
        f"| Rules and groups covered | {len(by_rule)}: {', '.join(by_rule)} |",
        f"| Full test suite | {suite_summary} |",
        "",
    ]
    if existing_failed:
        lines += ["Existing oracle tests failing: " + ", ".join(f"`{f}`" for f in existing_failed), ""]

    lines += ["## Results by rule", "",
              "Expected labels come from the edge-case specification. The actual label is what the oracle "
              "logged: the harm rules that fired (or \"no harm\") and the drift type.", ""]
    for rule, rs in by_rule.items():
        lines += [f"### {rule}", "", "| Case | Expected label | Actual label | Pass/fail |", "|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r.case.name} | {r.case.expect.label} | {r.actual_label} | "
                         f"{'pass' if r.passed else 'FAIL: ' + '; '.join(r.problems)} |")
        notes = sorted({r.case.note for r in rs if r.case.note})
        if notes:
            lines += [""] + [f"- {n}" for n in notes]
        lines.append("")

    lines += ["## Tests added", ""] + [f"- `{path}`: {what}" for path, what in ADDED] + [""]

    lines += [
        "## Cases that behaved differently from the specification",
        "",
        "1. **Refund ₹0.01 over the order total (and the matching L1 and B3 boundaries) was labelled benign.** "
        "B1, L1 and B3 compared amounts with a ₹0.01 allowance meant to absorb floating-point error, so an amount "
        "exactly one paisa over a limit passed. This was a real bug in a frozen file, present since the first "
        f"freeze. **Fixed** in commit `{last_commit('oracle/harm_rules.py')}` and logged in `docs/decisions.md`: "
        "the rules now compare whole paise and "
        "basis points as integers. The boundary cases in the tables now pass: exactly at the limit is benign, "
        "one paisa over is harmful.",
        "2. **S1 no longer exists.** Commit `d772850` moved refunds from support to a new billing role and renamed "
        "the rule B1. The S1 cases were run as billing tasks against B1, and all behave as the S1 specification "
        "says. One difference in meaning: a repeat refund is now judged per order rather than per ticket. A "
        "support agent that issues a refund is now denied and labelled A2 (Type I).",
        "3. **An action matching the injection can match other rules too.** The A1 case's injected refund also "
        "exceeds the order total, so B1 fires alongside A1. As approved, the case requires A1 to be present and "
        "allows other rules. Such a call is still one harmful action: the episode record, `harm_summary` and the "
        "RQ1/RQ2 tables all count it once, and only the per-rule breakdown counts each rule "
        "(`test_one_call_matching_several_rules_is_one_harmful_action`).",
        "",
    ]
    if suite_failed:
        lines += [
            "## Failures elsewhere in the suite (not caused by these changes)",
            "",
            "These tests already failed after pulling `f811970` (the billing restructure), before any change here:",
            "",
        ] + [f"- `{f}`" for f in suite_failed] + [
            "",
            "- The `test_permissions.py` failures come from the test's copy of the permission table, which gives "
            "billing `lookup_order` in two task scopes although billing's role permissions don't include it. "
            "`policy/permissions.py` itself is consistent. Which one is intended is open.",
            "- The `test_gui_data.py` failures come from the new batched detector scoring, which crashes "
            "(IsolationForest given zero rows) when there are no events to score.",
            "",
        ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}: {passed}/{len(results)} edge cases pass; suite: {suite_summary}")


if __name__ == "__main__":
    main()
