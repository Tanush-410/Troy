"""In-memory marketplace state.

Plain pydantic state rather than SQLite: episodes are short and single-threaded,
a deep copy gives a cheap exact snapshot for replay, and every mutation goes
through Python code that appends a StateChange, so harm and task success can be
measured from the change ledger without diffing databases.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from simmart.models import (
    Competitor,
    Customer,
    DataExport,
    Email,
    Escalation,
    Injection,
    Listing,
    Order,
    PriceReport,
    Product,
    Refund,
    StateChange,
    Ticket,
)


class SimMartState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    tick: int = 0
    customers: dict[str, Customer] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict)
    listings: dict[str, Listing] = Field(default_factory=dict)
    orders: dict[str, Order] = Field(default_factory=dict)
    tickets: dict[str, Ticket] = Field(default_factory=dict)
    refunds: dict[str, Refund] = Field(default_factory=dict)
    competitors: dict[str, Competitor] = Field(default_factory=dict)
    price_reports: dict[str, PriceReport] = Field(default_factory=dict)
    emails: dict[str, Email] = Field(default_factory=dict)
    escalations: dict[str, Escalation] = Field(default_factory=dict)
    exports: dict[str, DataExport] = Field(default_factory=dict)
    # Keyed by "<competitor_id>|<sku>"; set by scenario generators (D3).
    competitor_page_injections: dict[str, Injection] = Field(default_factory=dict)
    changes: list[StateChange] = Field(default_factory=list)
    id_counters: dict[str, int] = Field(default_factory=dict)

    def next_id(self, prefix: str, width: int = 4) -> str:
        n = self.id_counters.get(prefix, 0) + 1
        self.id_counters[prefix] = n
        return f"{prefix}{n:0{width}d}"

    def record_change(
        self,
        tool: str,
        entity_type: str,
        entity_id: str,
        field: str,
        before: Any,
        after: Any,
        customer_id: str | None = None,
    ) -> None:
        self.changes.append(
            StateChange(
                seq=len(self.changes),
                tick=self.tick,
                tool=tool,
                entity_type=entity_type,
                entity_id=entity_id,
                field=field,
                before=before,
                after=after,
                customer_id=customer_id,
            )
        )

    def advance_clock(self, n: int = 1) -> None:
        """Advance simulated time. Competitor prices are a function of tick.

        Tools never advance the clock; the experiment runner does (once per task).
        """
        if n < 0:
            raise ValueError("clock cannot go backwards")
        self.tick += n

    def snapshot(self) -> SimMartState:
        return self.model_copy(deep=True)

    def find_competitor(self, key: str) -> Competitor | None:
        """Look up a competitor by id or (case-insensitive) name."""
        if key in self.competitors:
            return self.competitors[key]
        for c in self.competitors.values():
            if c.name.lower() == key.lower():
                return c
        return None
