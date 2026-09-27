"""The permission-log drift detector (PLDD): fit, calibrate, score.

Input is only lists of DetectorEvent episodes. Which episodes to fit and
calibrate on (e.g. the D0 split, or its harm-free subset) is the caller's
choice; this package never sees labels.

Combiners over standardized features z (only upward deviations count for the
weighted sum):
  weighted  sum_f w_f * max(0, z_f)
  iforest   -IsolationForest.score_samples(z), fitted on baseline windows
Thresholds are calibrated per combiner so that at most `fpr` of calibration
episodes ever alert (episode-level false-positive rate).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from detector.features import FEATURES, Baseline, drift_jsd, fit_baseline, raw_features, standardize, windows
from policy.log_schema import DetectorEvent

COMBINERS = ("weighted", "iforest")


@dataclass(frozen=True)
class StepScore:
    step: int
    task_id: str | None
    raw: tuple[float, ...]
    z: tuple[float, ...]
    weighted: float
    iforest: float
    drift_jsd: float


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    steps: tuple[StepScore, ...]
    alert_step: dict[str, int | None]  # combiner -> first step whose score exceeds its threshold

    def max_score(self, combiner: str) -> float:
        return max((getattr(s, combiner) for s in self.steps), default=float("-inf"))


class PLDD:
    def __init__(self, window: int = 5, features: Sequence[str] = FEATURES,
                 weights: dict[str, float] | None = None, seed: int = 0,
                 n_estimators: int = 200) -> None:
        unknown = set(features) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features: {unknown}")
        self.window = window
        self.features = tuple(features)
        self._idx = [FEATURES.index(f) for f in self.features]
        self.weights = np.array([(weights or {}).get(f, 1.0) for f in self.features])
        self.seed = seed
        self.n_estimators = n_estimators
        self.baseline: Baseline | None = None
        self._forest: IsolationForest | None = None
        self.thresholds: dict[str, float] = {}

    # ----------------------------------------------------------------- fit

    def fit(self, episodes: Sequence[Sequence[DetectorEvent]], all_actions: Sequence[str]) -> PLDD:
        self.baseline = fit_baseline(episodes, self.window, all_actions)
        z = np.array([self._z(win) for ep in episodes for win in windows(ep, self.window)])
        self._forest = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(z)
        return self

    def calibrate(self, episodes: Sequence[Sequence[DetectorEvent]], fpr: float = 0.05) -> PLDD:
        for c in COMBINERS:
            maxima = np.array([self._max_raw(ep, c) for ep in episodes])
            # smallest threshold with at most `fpr` of calibration episodes strictly above it
            self.thresholds[c] = float(np.quantile(maxima, 1 - fpr, method="higher"))
        return self

    def iforest_features_used(self) -> set[str]:
        """Features the IsolationForest actually splits on.

        A feature that is constant in the baseline windows can never be split on,
        so IsolationForest ignores it entirely, even if it varies when scored.
        """
        assert self._forest is not None, "fit first"
        used: set[str] = set()
        for tree, cols in zip(self._forest.estimators_, self._forest.estimators_features_, strict=True):
            used.update(self.features[cols[i]] for i in tree.tree_.feature if i >= 0)
        return used

    # --------------------------------------------------------------- score

    def _z(self, win: Sequence[DetectorEvent]) -> list[float]:
        assert self.baseline is not None
        z = standardize(raw_features(win, self.baseline), self.baseline)
        return [z[i] for i in self._idx]

    def _combine(self, z: list[float]) -> tuple[float, float]:
        weighted, iforest = self._combine_many(np.array([z]))
        return float(weighted[0]), float(iforest[0])

    def _combine_many(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scores for a batch of standardized feature rows (one IsolationForest call)."""
        if z.shape[0] == 0:  # nothing to score (e.g. no events yet): no IsolationForest call
            return np.zeros(0), np.zeros(0)
        assert self._forest is not None
        weighted = np.sum(self.weights * np.maximum(0.0, z), axis=1)
        return weighted, -self._forest.score_samples(z)

    def _max_raw(self, ep: Sequence[DetectorEvent], combiner: str) -> float:
        wins = windows(ep, self.window)
        if not wins:
            return float("-inf")
        scores = self._combine_many(np.array([self._z(w) for w in wins]))[COMBINERS.index(combiner)]
        return float(scores.max())

    def score(self, ep: Sequence[DetectorEvent], episode_id: str | None = None) -> EpisodeScore:
        return self.score_many([ep], [episode_id])[0]

    def score_many(self, episodes: Sequence[Sequence[DetectorEvent]],
                   episode_ids: Sequence[str | None] | None = None) -> list[EpisodeScore]:
        """Score many episodes with a single IsolationForest call.

        Identical per-episode results to calling `score` in a loop: the forest is
        deterministic and does not depend on batch composition. Batching matters
        because `score_samples` costs far more in per-call overhead than per row.
        """
        assert self.baseline is not None, "fit first"
        ids = list(episode_ids) if episode_ids is not None else [None] * len(episodes)
        per_ep: list[tuple[list[list[float]], list[tuple[float, ...]], list[tuple[float, ...]]]] = []
        flat: list[list[float]] = []
        for ep in episodes:
            wins = windows(ep, self.window)
            raws = [tuple(raw_features(w, self.baseline)) for w in wins]
            zs = [tuple(standardize(list(r), self.baseline)) for r in raws]
            per_ep.append((wins, raws, zs))  # type: ignore[arg-type]
            flat.extend(z[i] for z in zs for i in self._idx)
        weighted, iforest = self._combine_many(
            np.array(flat, dtype=float).reshape(-1, len(self.features)))

        out: list[EpisodeScore] = []
        cursor = 0
        for ep, (wins, raws, zs), eid in zip(episodes, per_ep, ids, strict=True):
            n = len(wins)
            steps = [
                StepScore(step=win[-1].step, task_id=win[-1].task_id, raw=raws[k], z=zs[k],
                          weighted=float(weighted[cursor + k]), iforest=float(iforest[cursor + k]),
                          drift_jsd=drift_jsd(win, self.baseline))
                for k, win in enumerate(wins)
            ]
            cursor += n
            alerts: dict[str, int | None] = {}
            for c in COMBINERS:
                t = self.thresholds.get(c)
                alerts[c] = next((s.step for s in steps if t is not None and getattr(s, c) > t), None)
            out.append(EpisodeScore(
                episode_id=eid if eid is not None else (ep[0].episode_id if ep else ""),
                steps=tuple(steps), alert_step=alerts))
        return out


class LiveMonitor:
    """Scores an episode as its events arrive; used by the PEP's live-pause mode."""

    def __init__(self, pldd: PLDD, combiner: str = "weighted") -> None:
        if combiner not in COMBINERS or combiner not in pldd.thresholds:
            raise ValueError("monitor needs a fitted, calibrated combiner")
        self.pldd = pldd
        self.combiner = combiner
        self.events: list[DetectorEvent] = []

    def __call__(self, event: DetectorEvent) -> bool:
        self.events.append(event)
        win = self.events[-self.pldd.window:]
        z = self.pldd._z(win)
        score = self.pldd._combine(z)[COMBINERS.index(self.combiner)]
        return score > self.pldd.thresholds[self.combiner]
