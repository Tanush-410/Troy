"""PLDD features, computed only from the PEP's detector view (DetectorEvent).

Per step t, over the sliding window of the last `w` events of the episode:
  deny_rate          share of calls refused
  novel_action_rate  share of actions never seen in the baseline for their task type
  action_freq_z      largest count z-score of any action in the window vs baseline windows
  param_z            largest |z| of a numeric parameter (log-scaled) vs baseline values
  seq_surprise       S_t: mean -log P_base(a_i | a_{i-1}) under the baseline bigram model
  expansion_rate     share of calls that needed a scope-expansion request
and the reference signal
  drift_jsd          D_t: JSD(window action distribution || pi*_tau of the window's task types)

This module must never see labels, taint, tokens, prompts or transcripts: it
imports only DetectorEvent from the log schema.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from policy.log_schema import DetectorEvent

FEATURES = ("deny_rate", "novel_action_rate", "action_freq_z", "param_z", "seq_surprise", "expansion_rate")
START = "<start>"
Z_STD_FLOOR = 0.5  # keeps a z-score finite when the baseline never varies
BIGRAM_ALPHA = 0.1  # add-alpha smoothing


def _task_key(e: DetectorEvent) -> str:
    return e.task_type or "<no_task>"


def numeric_params(e: DetectorEvent) -> dict[str, float]:
    """Numeric parameters as `action.path` -> log1p(|value|); prices and amounts span orders of magnitude."""
    out: dict[str, float] = {}

    def walk(prefix: str, v: object) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out[prefix] = math.log1p(abs(float(v)))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}", x)
        elif isinstance(v, list):
            for x in v:
                walk(f"{prefix}[]", x)

    walk(e.action, e.params)
    return out


def windows(episode: Sequence[DetectorEvent], w: int) -> list[list[DetectorEvent]]:
    return [list(episode[max(0, i - w + 1): i + 1]) for i in range(len(episode))]


@dataclass
class Baseline:
    """Everything fitted on clean episodes. Built by `fit_baseline`."""

    w: int
    actions: set[str] = field(default_factory=set)
    seen: dict[str, set[str]] = field(default_factory=dict)  # task type -> actions seen
    policy: dict[str, dict[str, float]] = field(default_factory=dict)  # pi*_tau
    count_stats: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    param_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    bigram: dict[str, dict[str, float]] = field(default_factory=dict)  # role -> "prev|next" -> log p
    vocab: set[str] = field(default_factory=set)
    feat_mean: list[float] = field(default_factory=list)
    feat_std: list[float] = field(default_factory=list)


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, Z_STD_FLOOR
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, max(math.sqrt(var), Z_STD_FLOOR)


def _pairs(episode: Sequence[DetectorEvent]) -> list[tuple[str, str]]:
    out, prev, prev_task = [], START, None
    for e in episode:
        if e.task_id != prev_task:
            prev, prev_task = START, e.task_id
        out.append((prev, e.action))
        prev = e.action
    return out


def fit_baseline(episodes: Sequence[Sequence[DetectorEvent]], w: int, all_actions: Sequence[str]) -> Baseline:
    b = Baseline(w=w, vocab=set(all_actions) | {START})
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    params: dict[str, list[float]] = defaultdict(list)
    bigrams: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for ep in episodes:
        for e in ep:
            per_task[_task_key(e)][e.action] += 1
            for k, v in numeric_params(e).items():
                params[k].append(v)
        if ep:
            bigrams[ep[0].agent_role].update(_pairs(ep))
    b.seen = {t: set(c) for t, c in per_task.items()}
    b.policy = {t: {a: n / sum(c.values()) for a, n in c.items()} for t, c in per_task.items()}
    b.param_stats = {k: _mean_std(v) for k, v in params.items()}
    for role, counts in bigrams.items():
        prev_totals: Counter[str] = Counter()
        for (p, _), n in counts.items():
            prev_totals[p] += n
        v = len(b.vocab)
        b.bigram[role] = {
            f"{p}|{a}": math.log((counts.get((p, a), 0) + BIGRAM_ALPHA) / (prev_totals[p] + BIGRAM_ALPHA * v))
            for p in b.vocab for a in b.vocab
        }
    # per-window action counts, per (task type, action)
    counts_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ep in episodes:
        for win in windows(ep, w):
            key_task = _task_key(win[-1])
            c = Counter(e.action for e in win)
            for a in b.seen.get(key_task, ()):
                counts_by_key[(key_task, a)].append(float(c.get(a, 0)))
    b.count_stats = {k: _mean_std(v) for k, v in counts_by_key.items()}
    # standardization of the raw feature vectors
    raw = [raw_features(win, b) for ep in episodes for win in windows(ep, w)]
    cols = list(zip(*raw)) if raw else [()] * len(FEATURES)
    stats = [_mean_std(list(col)) for col in cols]
    b.feat_mean = [m for m, _ in stats]
    b.feat_std = [s for _, s in stats]
    return b


def _unseen_logp(b: Baseline) -> float:
    return math.log(BIGRAM_ALPHA / (BIGRAM_ALPHA * len(b.vocab)))


def raw_features(win: Sequence[DetectorEvent], b: Baseline) -> list[float]:
    n = len(win)
    deny = sum(e.decision == "deny" for e in win) / n
    novel = sum(e.action not in b.seen.get(_task_key(e), set()) for e in win) / n
    c = Counter(e.action for e in win)
    task = _task_key(win[-1])
    zs = []
    for a, k in c.items():
        m, s = b.count_stats.get((task, a), (0.0, Z_STD_FLOOR))
        zs.append((k - m) / s)
    freq_z = max(zs) if zs else 0.0
    pz = [abs(v - b.param_stats[k][0]) / b.param_stats[k][1] if k in b.param_stats else abs(v) / Z_STD_FLOOR
          for e in win for k, v in numeric_params(e).items()]
    param_z = max(pz) if pz else 0.0
    role_bigram = b.bigram.get(win[0].agent_role, {})
    pairs = _pairs(win)
    surprise = -sum(role_bigram.get(f"{p}|{a}", _unseen_logp(b)) for p, a in pairs) / len(pairs)
    expansion = sum(e.expansion_requested for e in win) / n
    return [deny, novel, freq_z, param_z, surprise, expansion]


def standardize(raw: Sequence[float], b: Baseline) -> list[float]:
    return [(x - m) / s for x, m, s in zip(raw, b.feat_mean, b.feat_std, strict=True)]


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(x: dict[str, float]) -> float:
        return sum(v * math.log2(v / m[k]) for k, v in x.items() if v > 0)

    return 0.5 * kl(p) + 0.5 * kl(q)


def drift_jsd(win: Sequence[DetectorEvent], b: Baseline) -> float:
    """D_t: JSD between the window's action distribution and pi*_tau (the window's
    task types mixed in proportion to their share of the window)."""
    n = len(win)
    p = {a: k / n for a, k in Counter(e.action for e in win).items()}
    q: dict[str, float] = defaultdict(float)
    tasks = Counter(_task_key(e) for e in win)
    for t, k in tasks.items():
        ref = b.policy.get(t)
        if ref is None:
            q["<unseen_task>"] += k / n
            continue
        for a, pa in ref.items():
            q[a] += pa * k / n
    return _jsd(p, dict(q))
