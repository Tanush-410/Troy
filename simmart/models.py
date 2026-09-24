"""Entity models for the SimMart marketplace.

All money is in INR, stored as floats rounded to 2 decimals.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Injection(_Model):
    """An adversarial instruction embedded in untrusted content.

    Populated by scenario generators (D3). `target_tool` / `target_params`
    describe what the injection tries to make the agent do; the harm oracle's
    A1 rule matches actions against them.
    """

    injection_id: str
    text: str
    target_tool: str
    target_params: dict[str, Any] = Field(default_factory=dict)


class Customer(_Model):
    customer_id: str
    name: str
    email: str
    phone: str
    address: str
    deleted: bool = False


class Product(_Model):
    """Source catalog data. `cost` is the cost floor used by harm rule L1."""

    sku: str
    name: str
    category: str
    cost: float
    base_price: float
    supplier_description: str
    supplier_injection: Injection | None = None


class Listing(_Model):
    listing_id: str
    sku: str
    title: str
    description: str
    price: float
    images: list[str] = Field(default_factory=list)
    status: Literal["active", "delisted"] = "active"


class OrderItem(_Model):
    sku: str
    listing_id: str
    qty: int
    unit_price: float


class Order(_Model):
    order_id: str
    customer_id: str
    items: list[OrderItem]
    total: float
    shipping_address: str
    status: Literal["placed", "shipped", "delivered"]
    refunded_amount: float = 0.0


class Ticket(_Model):
    ticket_id: str
    customer_id: str
    order_ids: list[str]
    kind: Literal["query", "refund", "address"]
    subject: str
    body: str
    status: Literal["open", "closed"] = "open"
    replies: list[str] = Field(default_factory=list)
    injection: Injection | None = None


class Refund(_Model):
    refund_id: str
    order_id: str
    amount: float
    tick: int


class Competitor(_Model):
    """Parameters of one simulated competitor's pricing process."""

    competitor_id: str
    name: str
    price_bias: float  # multiplier on our base price
    volatility: float  # amplitude of the periodic component
    period: int  # ticks per pricing cycle
    phase: float
    coverage: float  # probability a given SKU is carried


class PriceEntry(_Model):
    sku: str
    competitor: str
    price: float


class PriceReport(_Model):
    report_id: str
    entries: list[PriceEntry]
    summary: str
    tick: int
    # Injections present in the writer's context, set by the PEP (not the
    # agent) so taint propagates to agents that later read this report.
    source_injection_ids: list[str] = Field(default_factory=list)


class Email(_Model):
    email_id: str
    kind: Literal["reply", "marketing"]
    to_customer_ids: list[str]
    subject: str
    body: str
    tick: int


class Escalation(_Model):
    escalation_id: str
    reason: str
    tick: int


class DataExport(_Model):
    export_id: str
    customer_ids: list[str]
    tick: int


class StateChange(_Model):
    """One field-level mutation, appended by every state-changing tool.

    `customer_id` attributes the change to a customer when one is affected
    (used by harm rule S2 and by task-success checkers).
    """

    seq: int
    tick: int
    tool: str
    entity_type: str
    entity_id: str
    field: str
    before: Any
    after: Any
    customer_id: str | None = None
