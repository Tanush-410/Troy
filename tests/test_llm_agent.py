"""LLM agent loop and providers, with fake model clients (no network, no cost)."""

import json
from types import SimpleNamespace as NS

import pytest

from agents.llm_agent import LLMAgent
from agents.providers.anthropic_provider import AnthropicProvider
from agents.providers.base import ModelTurn, ProviderConfig, ToolCall, ToolOutcome, Usage
from agents.providers.openai_compat import OpenAICompatProvider
from experiments.config import EpisodeConfig
from experiments.cost import actual_cost, estimate
from experiments.runner import RunPaths, run_episode
from oracle.harm_rules import HarmOracle
from policy.pep import PEP, EpisodeContext
from policy.task import AgentTask, TaskSpec
from scenarios import basic
from tools import all_tool_schemas

# ------------------------------------------------------------------ fakes


class FakeSession:
    def __init__(self, turns):
        self.turns = list(turns)
        self.users, self.results = [], []

    def add_user(self, text):
        self.users.append(text)

    def step(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.results.append(results)


class FakeProvider:
    def __init__(self, *turn_lists):
        self.config = ProviderConfig(kind="openai_compat", context_window=200_000, model="fake")
        self.pending = list(turn_lists)
        self.sessions = []

    def new_session(self, system, tools):
        s = FakeSession(self.pending.pop(0) if self.pending else [])
        s.system, s.tools = system, tools
        self.sessions.append(s)
        return s


def turn(*calls, text="", usage=Usage(input_tokens=100, output_tokens=20)):
    tc = [ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)]
    return ModelTurn(text=text, tool_calls=tc, stop_reason="tool_use" if tc else "end_turn", usage=usage)


def make_pep(state, role="support"):
    ctx = EpisodeContext(run_id="r", episode_id="e", seed=7, model="fake", agent_role=role,
                         drift_condition="D0", control_condition="C1")
    pep = PEP(ctx, state, HarmOracle())
    pep.begin_task(TaskSpec(task_id="t1", task_type="answer_query", instruction="Answer TKT-0001",
                            ticket_id="TKT-0001"))
    return pep


# -------------------------------------------------------------- the loop


def test_loop_routes_calls_through_pep_and_feeds_results_back(state):
    provider = FakeProvider([
        turn(("read_ticket", {"ticket_id": "TKT-0001"}), ("lookup_order", '{"order_id": "ORD-0041"}')),
        turn(text="Done."),
    ])
    agent = LLMAgent(provider, "support")
    pep = make_pep(state)
    run = agent.run_task(AgentTask(task_id="t1", instruction="Answer TKT-0001"), pep.gateway())
    assert [e.action for e in pep.events] == ["read_ticket", "lookup_order"]
    assert [(e.tokens_in, e.tokens_out) for e in pep.events] == [(100, 20), (0, 0)]  # first call carries the turn
    (results,) = provider.sessions[0].results
    assert [r.call_id for r in results] == ["c0", "c1"] and not any(r.is_error for r in results)
    assert json.loads(results[0].content)["data"]["ticket_id"] == "TKT-0001"
    assert (run.model_turns, run.input_tokens, run.output_tokens, run.end) == (2, 200, 40, "done")
    assert [e["role"] for e in run.transcript][:2] == ["system", "user"]


def test_every_tool_schema_is_offered_whatever_the_role():
    provider = FakeProvider([turn(text="ok")])
    agent = LLMAgent(provider, "price_intel")
    agent.reset()
    assert provider.sessions[0].tools == all_tool_schemas()
    assert "Price Intelligence" in provider.sessions[0].system


def test_malformed_arguments_become_format_errors(state):
    provider = FakeProvider([turn(("read_ticket", '{"ticket_id": TKT-0001}')), turn(text="sorry")])
    pep = make_pep(state)
    LLMAgent(provider, "support").run_task(AgentTask(task_id="t1", instruction="x"), pep.gateway())
    e = pep.events[0]
    assert e.format_error and e.drift_type is None
    assert provider.sessions[0].results[0][0].is_error


def test_turn_limit(state):
    loop = [turn(("read_ticket", {"ticket_id": "TKT-0001"})) for _ in range(5)]
    run = LLMAgent(FakeProvider(loop), "support", max_turns_per_task=3).run_task(
        AgentTask(task_id="t1", instruction="x"), make_pep(state).gateway())
    assert (run.end, run.model_turns) == ("step_limit", 3)


def test_refusal_and_max_tokens_end_the_task(state):
    for reason, end in [("refusal", "refusal"), ("max_tokens", "max_tokens")]:
        t = ModelTurn(text="", tool_calls=[], stop_reason=reason, usage=Usage())
        run = LLMAgent(FakeProvider([t]), "support").run_task(AgentTask(task_id="t1", instruction="x"),
                                                              make_pep(state).gateway())
        assert run.end == end


def test_context_persists_until_reset():
    provider = FakeProvider([turn(text="a"), turn(text="b")], [turn(text="c")])
    agent = LLMAgent(provider, "support")
    gw = None
    agent.run_task(AgentTask(task_id="t1", instruction="one"), gw)
    agent.run_task(AgentTask(task_id="t2", instruction="two"), gw)  # same session (D1)
    agent.reset()
    agent.run_task(AgentTask(task_id="t3", instruction="three"), gw)
    assert [s.users for s in provider.sessions] == [["one", "two"], ["three"]]


# ------------------------------------------------------------ providers


class FakeAnthropicClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def anthropic_response(blocks, stop="tool_use"):
    return NS(content=blocks, stop_reason=stop,
              usage=NS(input_tokens=50, output_tokens=10, cache_read_input_tokens=30,
                       cache_creation_input_tokens=5))


def test_anthropic_session_message_shapes():
    tool_use = NS(type="tool_use", id="tu1", name="read_ticket", input={"ticket_id": "TKT-0001"})
    client = FakeAnthropicClient([anthropic_response([NS(type="text", text="Looking."), tool_use])])
    cfg = ProviderConfig(kind="anthropic", context_window=200_000, model="claude-haiku-4-5-20251001", temperature=0.5)
    s = AnthropicProvider(cfg, client=client).new_session("SYS", all_tool_schemas())
    s.add_user("hi")
    t = s.step()
    assert t.tool_calls == [ToolCall(id="tu1", name="read_ticket", arguments={"ticket_id": "TKT-0001"})]
    assert (t.text, t.stop_reason) == ("Looking.", "tool_use")
    assert (t.usage.total_input, t.usage.cache_read_tokens) == (85, 30)
    call = client.calls[0]
    assert call["system"] == "SYS" and call["extra_body"] == {"temperature": 0.5}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert {x["name"] for x in call["tools"]} == {x["function"]["name"] for x in all_tool_schemas()}
    assert set(call["tools"][0]) == {"name", "description", "input_schema"}
    s.add_tool_results([ToolOutcome("tu1", "{}", False), ToolOutcome("tu2", "{}", True)])
    last = s.messages[-1]
    assert last["role"] == "user" and [b["tool_use_id"] for b in last["content"]] == ["tu1", "tu2"]
    assert s.messages[1] == {"role": "assistant", "content": [NS(type="text", text="Looking."), tool_use]}


def test_anthropic_omits_temperature_by_default():
    client = FakeAnthropicClient([anthropic_response([NS(type="text", text="ok")], stop="end_turn")])
    s = AnthropicProvider(ProviderConfig(kind="anthropic", context_window=200_000, model="m"), client=client).new_session("S", [])
    s.add_user("hi")
    assert s.step().stop_reason == "end_turn"
    assert "extra_body" not in client.calls[0]


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def test_openai_compat_session_keeps_raw_arguments():
    call = NS(id="x1", function=NS(name="read_ticket", arguments='{"ticket_id": "TKT-0001"'))  # truncated JSON
    resp = NS(choices=[NS(finish_reason="tool_calls",
                          message=NS(content="<think>hmm</think>On it.", tool_calls=[call]))],
              usage=NS(prompt_tokens=70, completion_tokens=9))
    client = FakeOpenAIClient([resp])
    cfg = ProviderConfig(kind="openai_compat", context_window=200_000, model="qwen3:8b", base_url="http://x/v1", seed=3)
    s = OpenAICompatProvider(cfg, client=client).new_session("SYS", all_tool_schemas())
    s.add_user("hi")
    t = s.step()
    assert t.tool_calls[0].arguments == '{"ticket_id": "TKT-0001"'
    assert (t.text, t.usage.total_input, t.usage.output_tokens) == ("On it.", 70, 9)
    assert client.calls[0]["seed"] == 3 and "temperature" not in client.calls[0]
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "SYS"}
    s.add_tool_results([ToolOutcome("x1", "{}", True)])
    assert s.messages[-1] == {"role": "tool", "tool_call_id": "x1", "content": "{}"}
    assert s.messages[-2]["tool_calls"][0]["id"] == "x1"


# --------------------------------------------------- runner + cost + builders


def test_runner_records_agent_usage(state, tmp_path):
    provider = FakeProvider([turn(("read_ticket", {"ticket_id": "TKT-0001"})), turn(text="Done.")])
    cfg = EpisodeConfig(run_id="r", episode_id="e1", seed=7, model="fake", agent_role="support",
                        drift_condition="D0", control_condition="C1",
                        provider=ProviderConfig(kind="openai_compat", context_window=200_000, model="fake"))
    task = TaskSpec(task_id="t1", task_type="answer_query", instruction="x", ticket_id="TKT-0001",
                    expected_fact="placed")
    rec, _ = run_episode(cfg, [task], LLMAgent(provider, "support"), RunPaths.for_run(tmp_path, "r"), {},
                         state=state)
    assert (rec.model_turns, rec.input_tokens, rec.output_tokens) == (2, 200, 40)
    assert rec.task_ends == {"t1": "done"} and rec.n_steps == 1
    transcript = (tmp_path / "transcripts" / "r" / "e1.jsonl").read_text().splitlines()
    assert json.loads(transcript[0])["role"] == "system"


def test_config_hash_covers_provider_settings():
    base = dict(run_id="r", episode_id="e", seed=1, model="m", agent_role="support",
                drift_condition="D0", control_condition="C1")
    a = EpisodeConfig(**base, provider=ProviderConfig(kind="anthropic", context_window=200_000, model="m"))
    b = EpisodeConfig(**base, provider=ProviderConfig(kind="anthropic", context_window=200_000, model="m", temperature=0.0))
    assert a.config_hash() != b.config_hash()


def test_cost_estimate_and_actual():
    e = estimate("claude-haiku-4-5-20251001", [("support", 2, True)], max_turns=12)
    assert 0 < e.usd_no_cache < e.usd_upper_bound
    assert estimate("qwen3:8b", [("support", 2, True)], 12).usd_no_cache is None
    fresh = estimate("claude-haiku-4-5-20251001", [("support", 10, True)], 12)
    shared = estimate("claude-haiku-4-5-20251001", [("support", 10, False)], 12)
    assert shared.input_tokens > fresh.input_tokens  # D1-style growing context costs more
    rec = NS(input_tokens=1_000_000, cache_read_tokens=0, cache_write_tokens=0, output_tokens=0)
    assert actual_cost("claude-haiku-4-5-20251001", [rec]) == pytest.approx(1.0)
    rec = NS(input_tokens=1_000_000, cache_read_tokens=1_000_000, cache_write_tokens=0, output_tokens=0)
    assert actual_cost("claude-haiku-4-5-20251001", [rec]) == pytest.approx(0.1)


def test_basic_builders_state_what_the_agent_needs(state):
    lid, rid = basic.reprice_candidate(state)
    t = basic.reprice_listing("t", state, lid, rid)
    cost = state.products[state.listings[lid].sku].cost
    assert f"{cost:.2f}" in t.instruction and rid in t.instruction
    prices = [e.price for e in state.price_reports[rid].entries]
    assert basic.reprice_target_reachable(state, lid, prices)
    scan = basic.competitor_scan("t", state, "SKU-0001")
    assert all(cid in scan.instruction and c.name in scan.instruction for cid, c in state.competitors.items())
    q = basic.answer_query("t", state, basic.tickets_of_kind(state, "query")[0])
    assert q.expected_fact in {"placed", "shipped", "delivered"}
