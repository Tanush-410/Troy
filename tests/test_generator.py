import pytest

from simmart import GeneratorConfig, generate_state
from tools import execute


def test_same_seed_same_state():
    assert generate_state(3).model_dump() == generate_state(3).model_dump()


def test_different_seed_different_state():
    assert generate_state(3).model_dump() != generate_state(4).model_dump()


def test_counts_follow_config():
    cfg = GeneratorConfig(n_customers=5, n_products=4, n_orders=6, n_tickets=3, n_competitors=2, start_tick=10)
    s = generate_state(1, cfg)
    assert (len(s.customers), len(s.products), len(s.listings)) == (5, 4, 4)
    assert (len(s.orders), len(s.tickets), len(s.competitors)) == (6, 3, 2)
    assert s.tick == 10
    assert s.changes == []


def test_referential_integrity(state):
    for p in state.products.values():
        assert 0 < p.cost < p.base_price
    for o in state.orders.values():
        assert o.customer_id in state.customers
        assert all(it.listing_id in state.listings for it in o.items)
        assert o.total == round(sum(it.qty * it.unit_price for it in o.items), 2)
    for t in state.tickets.values():
        assert all(state.orders[oid].customer_id == t.customer_id for oid in t.order_ids)
        assert t.injection is None


def test_snapshot_is_independent(state):
    snap = state.snapshot()
    next(iter(state.listings.values())).price = 1.0
    assert snap.model_dump() != state.model_dump()


@pytest.mark.parametrize("seed", range(20))
def test_ticket_text_states_hidden_goal(seed):
    """What the agent can read must match the hidden goal, so no task is impossible."""
    s = generate_state(seed)
    for t in s.tickets.values():
        visible = execute(s, "read_ticket", {"ticket_id": t.ticket_id}).data
        text = f"{visible['subject']}\n{visible['body']}"
        assert all(oid in text for oid in t.order_ids), t.ticket_id
        if t.kind == "refund":
            assert t.requested_amount is not None and t.requested_address is None
            assert f"INR {t.requested_amount:.2f}" in text, t.ticket_id
        elif t.kind == "address":
            assert t.requested_address is not None and t.requested_amount is None
            assert t.requested_address in text, t.ticket_id
        else:
            assert t.requested_amount is None and t.requested_address is None
