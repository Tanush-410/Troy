"""Statistics: episode-level bootstrap CIs, Fisher's exact test, Mann-Whitney U.

Events within an episode are not independent, so every CI resamples whole
episodes with replacement, never individual events.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

T = TypeVar("T")
N_BOOT = 2000
SEED = 12345


def bootstrap_ci(units: Sequence[T], stat: Callable[[Sequence[T]], float | None], n_boot: int | None = None,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float | None, float | None, float | None]:
    """(point estimate, lower, upper) with a percentile bootstrap over `units` (episodes)."""
    n_boot = N_BOOT if n_boot is None else n_boot
    point = stat(units) if units else None
    if point is None or len(units) < 2:
        return point, None, None
    rng = np.random.default_rng(seed)
    idx = np.arange(len(units))
    draws = []
    for _ in range(n_boot):
        sample = [units[i] for i in rng.choice(idx, size=len(units), replace=True)]
        v = stat(sample)
        if v is not None and not math.isnan(v):
            draws.append(v)
    if not draws:
        return point, None, None
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def ratio_stat(num: Callable[[T], float], den: Callable[[T], float]) -> Callable[[Sequence[T]], float | None]:
    """Sum of numerators over sum of denominators across the sampled episodes."""
    def stat(units: Sequence[T]) -> float | None:
        d = sum(den(u) for u in units)
        return sum(num(u) for u in units) / d if d else None
    return stat


def fisher(a_yes: int, a_no: int, b_yes: int, b_no: int) -> float | None:
    """Two-sided Fisher's exact test p-value for a 2x2 table [[a_yes, a_no], [b_yes, b_no]]."""
    if (a_yes + a_no) == 0 or (b_yes + b_no) == 0:
        return None
    return float(fisher_exact([[a_yes, a_no], [b_yes, b_no]], alternative="two-sided")[1])


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Two-sided Mann-Whitney U p-value; None if either sample is empty."""
    if not x or not y:
        return None
    return float(mannwhitneyu(list(x), list(y), alternative="two-sided").pvalue)


def fmt_ci(point: float | None, lo: float | None, hi: float | None, pct: bool = True) -> str:
    if point is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{f(point)} [{f(lo)}, {f(hi)}]" if lo is not None and hi is not None else f(point)
