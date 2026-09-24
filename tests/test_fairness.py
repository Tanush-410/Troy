"""Fairness fixes: every provider's model gets identical schemas and identical errors."""

import json

import pytest

from oracle.harm_rules import HarmOracle
from policy.pep import PEP, EpisodeContext, explain_rejected
from policy.task import TaskSpec
from tools import TOOLS, all_tool_schemas


def test_no_tool_schema_contains_ref():
    for schema in all_tool_schemas():
        text = json.dumps(schema)
        assert "$ref" not in text and "$defs" not in text, schema["function"]["name"]


def test_inlined_nested_schema_keeps_its_fields():
    entry = TOOLS["write_price_report"].json_schema()["function"]["parameters"]["properties"]["entries"]["items"]
    assert entry["required"] == ["sku", "competitor", "price"] and entry["additionalProperties"] is False


WRONG_ARGS = {"entries": [{"competitor_id": "COMP1", "sku": "SKU-0001", "price": 1.0}], "summary": "s"}


def pep(state):
    ctx = EpisodeContext(run_id="r", episode_id="e", seed=7, model="m", agent_role="price_intel",
                         drift_condition="D0", control_condition="C1")
    p = PEP(ctx, state, HarmOracle())
    p.begin_task(TaskSpec(task_id="t", task_type="competitor_scan", instruction="", skus=("SKU-0001",)))
    return p


@pytest.mark.parametrize("raw,parse_error", [
    (json.dumps({"name": "write_price_report", "arguments": WRONG_ARGS}), "server returned tool_use_failed"),
    ("<function=write_price_report>" + json.dumps(WRONG_ARGS), "server returned tool_use_failed"),
    (json.dumps({"name": "write_price_report", "arguments": json.dumps(WRONG_ARGS)}), "server did not parse"),
])
def test_rejected_call_gets_the_same_message_as_a_direct_call(state, raw, parse_error):
    direct = pep(state.snapshot()).gateway().call("write_price_report", WRONG_ARGS)
    p = pep(state.snapshot())
    rejected = p.gateway().call("write_price_report", raw, parse_error=parse_error)
    assert rejected == direct and "competitor" in direct["error"] and "competitor_id" in direct["error"]
    e = p.events[-1]
    assert e.format_error and e.params == WRONG_ARGS and parse_error in e.format_error_detail


def test_unparseable_rejection_gets_the_json_message(state):
    direct = pep(state.snapshot()).gateway().call("write_price_report", '{"entries": [')
    rejected = pep(state.snapshot()).gateway().call("write_price_report", '{"entries": [', parse_error="x")
    assert rejected == direct and "not valid JSON" in direct["error"]


def test_rejection_with_valid_arguments_is_still_a_format_error(state):
    ok_args = {"entries": [{"competitor": "COMP1", "sku": "SKU-0001", "price": 1.0}], "summary": "s"}
    params, msg = explain_rejected("write_price_report", json.dumps({"name": "write_price_report",
                                                                     "arguments": ok_args}))
    assert params == ok_args and "call the tool again" in msg
