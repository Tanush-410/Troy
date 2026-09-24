"""Offline replay: score logged episodes as if the PLDD had been running (C3 = C1 + PLDD, C4 = C2 + PLDD)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from detector.pldd import PLDD, EpisodeScore
from policy.log_schema import DetectorEvent, read_detector_events


def episodes_from_log(paths: Path | Sequence[Path]) -> dict[str, list[DetectorEvent]]:
    """Detector-view episodes from one or more PEP logs, in step order."""
    out: dict[str, list[DetectorEvent]] = defaultdict(list)
    for path in [paths] if isinstance(paths, Path) else paths:
        for e in read_detector_events(path):
            out[e.episode_id].append(e)
    return {k: sorted(v, key=lambda e: e.step) for k, v in out.items()}


def replay(pldd: PLDD, episodes: dict[str, list[DetectorEvent]], ids: Iterable[str]) -> dict[str, EpisodeScore]:
    return {eid: pldd.score(episodes[eid], eid) for eid in ids if eid in episodes}
