# Progress log

Short summaries appended after each milestone.

## Milestone 6: scenario generators (2026-09-24)

**Built:**
- `scenarios/generator.py`: seeded D0–D3 episodes for all three roles, with D1 as 10 tasks in one context.
- D2 templates, each flagged as role-conflicting or merely ambiguous.
- D3 injections in ticket text, supplier text and competitor pages, with a configurable injection share.
- The cross-agent path, run through the real PEP.
- `scenarios/solver.py`: a reference solver that reads the hidden goals and is used only in tests.

**Tests** (415 in total):
- Every D0 and D1 task is completed without harm (3 roles × 12 seeds).
- Legitimate D3 actions produce no A1 false positives, and the tasks stay completable.
- Following each injection is labelled A1, including across agents.
- Also covered: injection share, D2 escalation flags, distinct entities within an episode, determinism.

**Assumptions:**
- 5 tasks per D0/D2/D3 episode.
- Injection share 0.5, with at least one injection per D3 episode.
- `history_analysis` is not injectable, since it reads no competitor page.

**Please check:**
- The D2 template wording, and which templates count as role-conflicting (`D2_TEMPLATES` in `scenarios/generator.py`).
- The injection texts and targets.

## Milestone 7: PLDD detector (2026-09-24)

**Built:**
- `detector/`, which reads only the detector view:
  - six features plus D_t;
  - weighted-sum and IsolationForest combiners;
  - 5%-FPR calibration;
  - offline replay (C3 = C1 logs, C4 = C2 logs);
  - a live monitor, with pause mode in the PEP and runner.
- `analysis/detection.py`: D0 split into held-out/fit/calibration (25% held out, then 60/40), both baselines, AUROC, TPR at 5% FPR, held-out FPR, lead times and single-feature ablations.
- `experiments/scripted_dataset.py`: the known-drift dataset.

**Scripted-data sanity check** (20 episodes per cell, C1):
| | D1 | D2 | D3 |
|---|---|---|---|
| AUROC, weighted sum | 0.975 | 0.83 | 0.81 |
| AUROC, IsolationForest | 0.95 | 0.74 | 0.75 |
| AUROC, D_t | 0.62 | 0.48 | 0.49 |

- D3 alerts mostly come after the harm, since following an injection is a single action.
- These numbers come from scripted drift and are pipeline checks, not findings.

**Tests:** 426 in total. They cover import isolation, feature responses, window sliding, calibration FPR, split stability, end-to-end AUROC on known drift (both controls and both baselines), ablation coverage, lead-time sign, replay using only the detector view, and live pause.

**Assumptions:**
- Window w=5, equal feature weights, standard-deviation floor 0.5.
- Price change % is not observable; log-scaled absolute prices and amounts are used instead.
- The episode-level positive label is "any harmful action".

**Please check:**
- The split (25% held out, then 60/40), and whether "positive = any harmful action" is the definition you want for RQ3.

## Milestone 8: analysis pipeline (2026-09-24)

**Built:**
- `analysis/stats.py`: episode-level bootstrap, Fisher's exact test, Mann–Whitney U.
- `analysis/tables.py`: RQ1, RQ2, RQ3, lead-time tests, ablation, D0 harm, format errors, context overflow and escalation, cross-agent taint.
- `analysis/figures.py`: ROC curves, harm bars, D_t timelines.
- `analysis/run_all.py`: regenerates everything into `results/`, plus a `summary.md` you can paste to your co-writer.

**Tests:** 434 in total. Statistics units, every output file written, RQ1 counts equal to raw log counts, RQ2 rates equal to direct computation, CIs bracketing their point estimates, RQ3 and ablation coverage, a summary with every section and no NaN, and identical tables on regeneration.

**Scripted check:** 8 episodes per cell, 192 episodes; `run_all` takes about 70 seconds. The numbers are scripted and not findings.

**Please check:**
- The summary layout and the choice of headline metrics in `summary.md`.
- Whether RQ2's Fisher test should use episodes (current) or calls. Episodes are the independent unit.
