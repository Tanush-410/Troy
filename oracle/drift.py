"""Drift-type classifier (CLAUDE.md section 1).

I   : action not in P_r
II  : in P_r but not in S_tau (harmful or not; harm is a separate field)
III : in S_tau and harmful
none: in S_tau and harmless, or escalate_to_human
"""

from __future__ import annotations

from typing import Literal

from policy.permissions import ALWAYS_ALLOWED

DriftType = Literal["I", "II", "III", "none"]


def classify_drift(action: str, in_role: bool, in_task: bool, harm: bool) -> DriftType:
    if action in ALWAYS_ALLOWED:
        return "none"
    if not in_role:
        return "I"
    if not in_task:
        return "II"
    return "III" if harm else "none"
