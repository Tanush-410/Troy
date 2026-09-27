"""Detector feature health on real logs: is any PLDD feature constant, degenerate or broken?

For each (model, control) slice this fits the baseline exactly as the RQ3
evaluation does (fit split of D0, all-D0 baseline), computes every raw feature
over every scored window, and reports per feature:
  - distinct values, standard deviation, share of windows where it is non-zero
  - NaN or infinite values (broken)
  - AUROC of the feature alone, using each episode's maximum (harmful episode = positive)
  - whether it is effectively used: by the weighted sum (its positive z-score is non-zero in
    at least one scored window) and by IsolationForest (at least one tree splits on it).
    A feature constant in the baseline gets z = 0 while it stays constant, so it adds nothing
    to the weighted sum, and IsolationForest can never split on it.
A feature is flagged "constant" if it never varies, "near-constant" if it is
non-zero in under 1% of windows, and "broken" if any value is NaN or infinite.
"""

from __future__ import annotations

import math
from typing import Any

from sklearn.metrics import roc_auc_score

from analysis.detection import split_d0
from analysis.tables import Dataset
from detector.config import DETECTOR
from detector.features import FEATURES, raw_features, standardize, windows
from detector.pldd import PLDD
from detector.replay import episodes_from_log
from tools import TOOLS

NEAR_CONSTANT = 0.01


def feature_health(ds: Dataset, window: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window = DETECTOR.window if window is None else window
    eps_by_id = episodes_from_log(ds.event_paths)
    for model in ds.models:
        for control in ("C1", "C2"):
            recs = {ep.record.episode_id: ep.record for ep in ds.select(model=model, control_condition=control)}
            split = split_d0([i for i, r in recs.items() if r.drift_condition == "D0"])
            fit = [eps_by_id[i] for i in split["fit"] if eps_by_id.get(i)]
            if not fit:
                continue
            pldd = PLDD(window=window).fit(fit, sorted(TOOLS))  # same settings as the RQ3 evaluation
            baseline = pldd.baseline
            iforest_used = pldd.iforest_features_used()
            scored = [i for i in recs if i not in split["fit"] and i not in split["calib"] and eps_by_id.get(i)]
            values: list[list[float]] = [[] for _ in FEATURES]
            contributes: list[int] = [0 for _ in FEATURES]
            ep_max: dict[str, list[float]] = {}
            for eid in scored:
                per_window = [raw_features(w, baseline) for w in windows(eps_by_id[eid], window)]
                for w in per_window:
                    z = standardize(w, baseline)
                    for k, v in enumerate(w):
                        values[k].append(v)
                        contributes[k] += z[k] > 0
                ep_max[eid] = [max(w[k] for w in per_window) for k in range(len(FEATURES))]
            labels = [int(recs[i].total_harmful_attempted > 0) for i in ep_max]
            for k, name in enumerate(FEATURES):
                vs = values[k]
                finite = [v for v in vs if math.isfinite(v)]
                broken = len(finite) != len(vs)
                mean = sum(finite) / len(finite) if finite else 0.0
                std = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite)) if finite else 0.0
                nonzero = sum(v != 0 for v in finite) / len(finite) if finite else 0.0
                distinct = len({round(v, 9) for v in finite})
                scores = [ep_max[i][k] for i in ep_max]
                auroc = (float(roc_auc_score(labels, scores))
                         if 0 < sum(labels) < len(labels) and all(map(math.isfinite, scores)) else None)
                status = ("broken" if broken else "constant" if distinct <= 1
                          else "near-constant" if nonzero < NEAR_CONSTANT else "ok")
                rows.append({"model": model, "control": control, "feature": name, "windows": len(vs),
                             "distinct_values": distinct, "std": std, "nonzero_share": nonzero,
                             "non_finite": len(vs) - len(finite), "auroc_alone": auroc, "status": status,
                             "weighted_contributing_share": contributes[k] / len(vs) if vs else 0.0,
                             "used_by_weighted": contributes[k] > 0, "used_by_iforest": name in iforest_used})
    return rows
