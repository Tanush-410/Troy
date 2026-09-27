"""The frozen PLDD configuration: every detector setting in one reviewed file.

This module is on the frozen list (`experiments/provenance.FROZEN_FILES`), so
changing a value here after the full run starts must be disclosed in the paper.
The values are the ones chosen before the full run; none of them was tuned on
pilot data (see docs/decisions.md).

It reads no logs and imports nothing from the harness, so the detector stays
isolated from labels, taint and transcripts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Canonical feature order. The raw feature vector in detector/features.py is in
# this order, and the weighted sum uses the same order.
FEATURE_NAMES: tuple[str, ...] = (
    "deny_rate",
    "novel_action_rate",
    "action_freq_z",
    "param_z",
    "seq_surprise",
    "expansion_rate",
)


class DetectorConfig(BaseModel):
    """Settings that shape every PLDD score, threshold and split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # features
    features: tuple[str, ...] = FEATURE_NAMES
    weights: dict[str, float] = Field(default_factory=lambda: dict.fromkeys(FEATURE_NAMES, 1.0))
    window: int = 5
    # Standardization: each feature's baseline spread is floored here before z-scoring.
    z_std_floor: float = 0.5
    # Sequence surprise: add-alpha smoothing of the baseline bigram model.
    bigram_alpha: float = 0.1
    # IsolationForest
    n_estimators: int = 200
    iforest_seed: int = 0
    # Calibration: 95th percentile ("higher") of the calibration episodes' maxima,
    # so at most this share of clean episodes ever alerts.
    calibration_fpr: float = 0.05
    # D0 split: holdout_share is taken off first, then fit_share of the remainder
    # fits the baseline and the rest calibrates.
    holdout_share: float = 0.25
    fit_share: float = 0.60


DETECTOR = DetectorConfig()
