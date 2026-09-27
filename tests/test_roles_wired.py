"""Every role in the permission tables must be run by the pilot and reported by the analysis."""

from analysis import run_all, tables
from experiments import pilot
from policy.permissions import ROLES, TASK_ROLE


def test_pilot_runs_every_role():
    assert set(pilot.ROLES) == set(ROLES)
    planned = {role for role, *_ in pilot.plan("m", 1)}
    assert planned == set(ROLES)


def test_analysis_reports_every_role():
    assert set(tables.ROLES) == set(ROLES)
    assert run_all.ROLES is tables.ROLES


def test_every_role_owns_a_task_type():
    assert set(TASK_ROLE.values()) == set(ROLES)


def test_pilot_size_is_24_cells_per_role():
    n = 3
    assert len(pilot.plan("m", n)) == len(ROLES) * 4 * 2 * n


def test_smoke_plan_covers_every_role_with_matching_tasks():
    from experiments.smoke import PLAN, SEED
    from simmart.generator import generate_state

    assert {role for role, _ in PLAN} == set(ROLES)
    for role, build in PLAN:
        tasks = build(generate_state(SEED))
        assert tasks and all(t.role == role for t in tasks), role
