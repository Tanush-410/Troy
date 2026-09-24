from simmart.competitors import carries, competitor_price


def _carried(state):
    for comp in state.competitors.values():
        for sku in state.products:
            if carries(state, comp, sku):
                return comp, sku
    raise AssertionError("no carried sku")


def test_price_is_pure_function_of_seed_and_tick(state):
    comp, sku = _carried(state)
    p1 = competitor_price(state, comp, sku)
    competitor_price(state, comp, sku, tick=state.tick + 5)  # unrelated call
    assert competitor_price(state, comp, sku) == p1


def test_prices_change_over_time(state):
    comp, sku = _carried(state)
    prices = {competitor_price(state, comp, sku, t) for t in range(50)}
    assert len(prices) > 10


def test_prices_stay_near_base(state):
    for comp in state.competitors.values():
        for sku, p in state.products.items():
            price = competitor_price(state, comp, sku)
            if price is not None:
                assert 0.7 * p.base_price < price < 1.3 * p.base_price


def test_some_skus_not_carried(state):
    pairs = [(c, s) for c in state.competitors.values() for s in state.products]
    n_carried = sum(carries(state, c, s) for c, s in pairs)
    assert 0 < n_carried < len(pairs)
