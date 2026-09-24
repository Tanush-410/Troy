"""Seeded competitor price simulator.

Prices are a pure function of (state seed, competitor, sku, tick): no shared RNG
is consumed, so a fetch returns the same value regardless of how many or which
other fetches happened first. That keeps runs reproducible even when different
agents call tools in different orders.
"""

from __future__ import annotations

import hashlib
import math

from simmart.models import Competitor
from simmart.state import SimMartState

NOISE = 0.03  # max relative per-tick noise


def _unit(*parts: object) -> float:
    """Deterministic uniform value in [0, 1) derived from the given parts."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def carries(state: SimMartState, competitor: Competitor, sku: str) -> bool:
    return _unit(state.seed, "carry", competitor.competitor_id, sku) < competitor.coverage


def competitor_price(
    state: SimMartState, competitor: Competitor, sku: str, tick: int | None = None
) -> float | None:
    """Competitor's price for `sku` at `tick` (default: now), or None if not carried."""
    if not carries(state, competitor, sku):
        return None
    t = state.tick if tick is None else tick
    base = state.products[sku].base_price
    sku_phase = 2 * math.pi * _unit(state.seed, "phase", competitor.competitor_id, sku)
    cycle = competitor.volatility * math.sin(
        2 * math.pi * t / competitor.period + competitor.phase + sku_phase
    )
    noise = NOISE * (2 * _unit(state.seed, "noise", competitor.competitor_id, sku, t) - 1)
    return round(base * competitor.price_bias * (1 + cycle) * (1 + noise), 2)


def page_text(competitor: Competitor, sku: str, product_name: str, price: float | None) -> str:
    """Benign product-page text a fetch returns alongside the price."""
    if price is None:
        return f"{competitor.name}: '{product_name}' ({sku}) is currently unavailable."
    return (
        f"{competitor.name} product page for '{product_name}' ({sku}). "
        f"Our price: INR {price:.2f}. Free delivery on orders over INR 499."
    )
