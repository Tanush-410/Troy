# Progress log

Short summaries appended after each milestone.

## Milestone 6: scenario generators (2026-09-24)

**Built:**
- `scenarios/generator.py`: seeded D0–D3 episodes for all three roles, with D1 as 10 tasks in one context.
- D2 templates, each flagged as role-conflicting or merely ambiguous.
- D3 injections in ticket text, supplier text and competitor pages, with a configurable injection share.
- The cross-agent path, run through the real PEP.
- `scenarios/solver.py`: a reference solver that reads the hidden goals and is used only in tests.

**Tests** (408 in total):
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
