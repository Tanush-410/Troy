import pytest

from simmart import SimMartState, generate_state


@pytest.fixture
def state() -> SimMartState:
    return generate_state(seed=7)
