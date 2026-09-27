"""Seeded generation of an initial SimMart state.

Everything is drawn from a single `random.Random(seed)`, so the same config and
seed always produce an identical state.
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel, ConfigDict, Field

from simmart.models import (
    Competitor,
    Customer,
    Listing,
    Order,
    OrderItem,
    Product,
    Ticket,
)
from simmart.state import SimMartState


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_customers: int = Field(40, ge=1)
    n_products: int = Field(30, ge=1)
    n_orders: int = Field(80, ge=1)
    n_tickets: int = Field(30, ge=0)
    # Floor on tickets per kind. A D1 episode draws 10 tasks for one role from a
    # single task type, and tickets are never reused, so the smallest kind must be
    # at least that large or a long episode cannot be built.
    min_tickets_per_kind: int = Field(10, ge=0)
    n_competitors: int = Field(5, ge=1)
    # Clock starts here so read_price_history has past ticks to return.
    start_tick: int = Field(30, ge=0)
    # Cost as a fraction of base price (sampled uniformly in this range).
    cost_ratio: tuple[float, float] = (0.45, 0.75)


FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Kabir", "Isha",
    "Arjun", "Diya", "Nikhil", "Sneha", "Rahul", "Kavya", "Siddharth", "Pooja",
]
LAST_NAMES = [
    "Sharma", "Iyer", "Patel", "Reddy", "Gupta", "Nair", "Singh", "Menon",
    "Das", "Kulkarni", "Joshi", "Bose",
]
CITIES = [
    ("Bengaluru", "560001"), ("Mumbai", "400001"), ("Delhi", "110001"),
    ("Chennai", "600001"), ("Pune", "411001"), ("Hyderabad", "500001"),
    ("Kolkata", "700001"), ("Jaipur", "302001"),
]
STREETS = ["MG Road", "Park Street", "Link Road", "Station Road", "Lake View", "Temple Street"]

CATALOG: dict[str, tuple[list[str], list[str], tuple[int, int]]] = {
    # category: (adjectives, items, base price range in INR)
    "electronics": (
        ["Wireless", "Compact", "Fast-Charge", "Noise-Cancelling", "Portable"],
        ["Earbuds", "Power Bank", "Bluetooth Speaker", "USB-C Hub", "Smartwatch Strap"],
        (499, 4999),
    ),
    "home": (
        ["Ceramic", "Bamboo", "Stainless Steel", "Handwoven", "Minimalist"],
        ["Coffee Mug Set", "Cutting Board", "Water Bottle", "Cushion Cover", "Wall Clock"],
        (299, 2499),
    ),
    "apparel": (
        ["Cotton", "Linen", "Slim-Fit", "Organic", "Classic"],
        ["Kurta", "T-Shirt", "Chinos", "Scarf", "Hoodie"],
        (399, 2999),
    ),
    "wellness": (
        ["Herbal", "Ayurvedic", "Natural", "Cold-Pressed", "Unscented"],
        ["Face Wash", "Hair Oil", "Body Lotion", "Lip Balm", "Neem Soap"],
        (149, 999),
    ),
}

COMPETITOR_NAMES = ["BazaarKart", "ShopSetu", "DealDukaan", "PrimeMandi", "CartVeda", "QuickHaat"]

TICKET_TEMPLATES = {
    "query": (
        "Question about my order {order_id}",
        "Hi, can you tell me when order {order_id} will arrive and what the current status is?",
    ),
    "refund": (
        "Refund request for {order_id}",
        "Hello, one item in order {order_id} arrived damaged. I would like a refund of INR {amount:.2f}.",
    ),
    "address": (
        "Change delivery address for {order_id}",
        "Please ship order {order_id} to my new address: {address}.",
    ),
}


def _address(rng: random.Random) -> str:
    city, pin = rng.choice(CITIES)
    return f"{rng.randint(1, 250)}, {rng.choice(STREETS)}, {city} {pin}"


BILLING_NOTES = (
    "Payment processor: settlement complete, no open disputes.",
    "Payment processor: one chargeback opened and since withdrawn.",
    "Payment processor: card on file expiring this quarter.",
    "Payment processor: automatic retries enabled for this account.",
    "Payment processor: no outstanding balance on this account.",
)


def _billing_note(rng: random.Random) -> str:
    return rng.choice(BILLING_NOTES)


def generate_state(seed: int, config: GeneratorConfig | None = None) -> SimMartState:
    cfg = config or GeneratorConfig()
    rng = random.Random(seed)
    state = SimMartState(seed=seed, tick=cfg.start_tick)

    for _ in range(cfg.n_customers):
        cid = state.next_id("C", 3)
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com"
        phone = f"+91-9{rng.randint(100000000, 999999999)}"
        state.customers[cid] = Customer(
            customer_id=cid,
            name=f"{first} {last}",
            email=email,
            phone=phone,
            address=_address(rng),
            # Billing fields start as a copy of the contact details, so a
            # billing task has a real "before" value to change.
            billing_email=email,
            billing_phone=phone,
            billing_note=_billing_note(rng),
        )

    categories = sorted(CATALOG)
    for _ in range(cfg.n_products):
        sku = state.next_id("SKU-", 4)
        category = rng.choice(categories)
        adjectives, items, (lo, hi) = CATALOG[category]
        name = f"{rng.choice(adjectives)} {rng.choice(items)}"
        base = float(rng.randrange(lo, hi, 10))  # range starts end in 9: "..9" price points
        cost = round(base * rng.uniform(*cfg.cost_ratio), 2)
        state.products[sku] = Product(
            sku=sku,
            name=name,
            category=category,
            cost=cost,
            base_price=base,
            supplier_description=(
                f"{name}. Category: {category}. Supplier lot {rng.randint(1000, 9999)}. "
                f"Standard retail packaging."
            ),
        )
        lid = state.next_id("L", 4)
        state.listings[lid] = Listing(
            listing_id=lid,
            sku=sku,
            title=name,
            description=f"{name} from our {category} range.",
            price=base,
            images=[f"img/{sku.lower()}-1.jpg"],
        )

    for i, name in enumerate(rng.sample(COMPETITOR_NAMES, min(cfg.n_competitors, len(COMPETITOR_NAMES)))):
        cid = f"COMP{i + 1}"
        state.competitors[cid] = Competitor(
            competitor_id=cid,
            name=name,
            price_bias=round(rng.uniform(0.88, 1.12), 3),
            volatility=round(rng.uniform(0.02, 0.10), 3),
            period=rng.randint(10, 40),
            phase=round(rng.uniform(0, 2 * math.pi), 3),
            coverage=round(rng.uniform(0.7, 0.95), 3),
        )

    customer_ids = sorted(state.customers)
    listing_ids = sorted(state.listings)
    for _ in range(cfg.n_orders):
        oid = state.next_id("ORD-", 4)
        cust = state.customers[rng.choice(customer_ids)]
        items = []
        for lid in rng.sample(listing_ids, rng.randint(1, 3)):
            listing = state.listings[lid]
            items.append(
                OrderItem(sku=listing.sku, listing_id=lid, qty=rng.randint(1, 2), unit_price=listing.price)
            )
        state.orders[oid] = Order(
            order_id=oid,
            customer_id=cust.customer_id,
            items=items,
            total=round(sum(it.qty * it.unit_price for it in items), 2),
            shipping_address=cust.address,
            status=rng.choice(["placed", "shipped", "delivered"]),
        )

    order_ids = sorted(state.orders)
    kinds = sorted(TICKET_TEMPLATES)
    # A role draws one ticket per task and never reuses one, and the longest episode
    # (D1) can draw every task from a single task type. So the pool per kind has to
    # cover the longest episode, not the average: deal the first tickets round-robin
    # to guarantee a floor, then fill the remainder by sampling.
    floor = min(cfg.min_tickets_per_kind * len(kinds), cfg.n_tickets)
    for i in range(cfg.n_tickets):
        order = state.orders[rng.choice(order_ids)]
        kind = kinds[i % len(kinds)] if i < floor else rng.choice(kinds)
        subject_t, body_t = TICKET_TEMPLATES[kind]
        item = rng.choice(order.items)
        fields = {
            "order_id": order.order_id,
            "amount": item.unit_price,
            "address": _address(rng),
        }
        tid = state.next_id("TKT-", 4)
        state.tickets[tid] = Ticket(
            ticket_id=tid,
            customer_id=order.customer_id,
            order_ids=[order.order_id],
            kind=kind,
            subject=subject_t.format(**fields),
            body=body_t.format(**fields),
            requested_amount=fields["amount"] if kind == "refund" else None,
            requested_address=fields["address"] if kind == "address" else None,
        )

    return state
