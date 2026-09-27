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

## Before STOP 1 (2026-09-24)

- Final smoke reruns with all fixes: gpt-oss-120b completed 6/6 tasks (23 rate-limit retries, 249 s of waiting); qwen3:8b completed 5/6 (su-1: answered in its final message instead of `reply_customer`, a model error now that the prompt is explicit).
- Estimator fixed to use recorded per-turn latency (it had produced a negative latency); every model turn now records `latency_s`.

## Role restructure accepted and prepared for re-freeze (2026-09-27)

- Four roles accepted (support, billing, listing, price_intel); the whole restructure is logged in `docs/decisions.md`, and `docs/ui-and-roles.md` is corrected about the earlier real-model runs.
- Billing granted `lookup_order`, which is also in `process_refund`'s required actions: the solver showed billing otherwise can't see order totals.
- Harm-oracle one-paisa bug fixed (integer paise and basis points); 58 edge cases and a report in `docs/harm_test_report.md`.
- Detector: fixed the crash on empty input; IsolationForest back to 200 trees.
- The pilot runner, analysis tables and smoke plan cover every role in `policy/permissions.py` (tested). The smoke runner's support episode no longer includes `process_refund`, which had moved to billing.
- Frozen list now includes `tools/impl.py` and `scenarios/generator.py`; `.github/CODEOWNERS` lists every frozen file (tested).
- `CLAUDE.md`, `docs/requirements.md` and the README describe four roles. 558 tests pass.

## STOP 2: qwen3:8b pilot (2026-09-27)

- 96/96 episodes, 32 cells × 3, frozen commit (clean provenance on every episode), $0. Model time 6.3 h; wall-clock 8.3 h because the Mac idle-slept (fixed mid-run with `caffeinate -w`).
- No harness bugs found:
  - 0 label inconsistencies in 1,582 events, and no out-of-role call allowed;
  - 600/600 tasks ended normally, with no context overflow or turn-limit hit (peak prompt 19,942 of 32,768);
  - 0 retries;
  - every low-success task type traced to model behaviour, with no impossible tasks. `create_listing`: qwen3 passes the SKU instead of the returned listing ID to `upload_image`. `reprice_listing`: wrong price (checker targets verified against the report it read). `competitor_scan`: `null` prices rejected as format errors.
  - All 150 P2-flagged report entries are genuine fabrications (138 copy another competitor's price), not rounding.
- Detector: no feature broken; only `expansion_rate` under C1 is constant, by design. Effective features: C3 weighted 5/6 and IsolationForest 4/6; C4 weighted 6/6 and IsolationForest 4/6. RQ3 is not evaluable at pilot scale (4–5 fit and 3–4 calibration episodes per control); no tuning proposed.
- Read-before-write: support 84/84 tasks; billing 10/150.
- Results in `results/pilot-qwen3/` (local; pilot numbers are for bug-finding only).

## Before the full run (2026-09-28)

- `create_listing` instruction clarified (image attached with the listing ID returned by `create_listing`, not the SKU). This is the pilot's 6-of-54 cause: an ambiguous instruction, not model error. The pilot is not re-run.
- `scenarios/basic.py` (task instructions) added to the frozen list, next to `scenarios/generator.py`.
- Detector configuration frozen in one file, `detector/config.py`: window 5, weights 1.0, spread floor 0.5, 200 trees, 5% episode-level FPR, the six features, smoothing 0.1, seed 0, D0 split 25% held out then 60/40. Every detector default now reads it; no value changed and none was tuned on pilot data.
- RQ3 bug fixed: `d772850` left `replay` out of `analysis/detection.py`'s imports, so every `evaluate` call raised `NameError` and no RQ3 table could be produced. The three known-drift tests failed at HEAD and pass now.
- The runner pauses on `logs/<run_id>/STOP` between episodes, and resumes from the same command.
- Groq watcher: no more clock drift between hourly checks, and a reused client with one rebuild-and-retry, so an idle connection is no longer a false "still free tier".
- 572 tests pass.
