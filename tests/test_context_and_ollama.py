"""Context-window guard, native Ollama provider, and overflow handling."""

import pytest

from agents.llm_agent import LLMAgent
from agents.prompts import system_prompt
from agents.providers.base import ContextGuard, ContextOverflow, ModelTurn, ProviderConfig, ToolOutcome, Usage
from agents.providers.ollama_provider import OllamaProvider, unparsed_tool_calls
from experiments.config import EpisodeConfig
from experiments.runner import RunPaths, run_episode
from policy.permissions import ROLE_PERMISSIONS
from policy.task import TaskSpec
from tools import all_tool_schemas

QWEN = ProviderConfig(kind="ollama", model="qwen3:8b", context_window=32_768, temperature=0.7, seed=5,
                      base_url="http://ollama:1", think=False)


def test_guard_refuses_request_that_would_not_fit():
    g = ContextGuard(window=10_000, max_tokens=1_000)
    g.check(full_prompt_chars=3_000)  # ~1k tokens: fits
    g.record(prompt_tokens=8_000, output_tokens=500)
    g.add("x" * 300)  # ~100 more tokens -> 8,600 + 1,000 max_tokens fits
    g.check(0)
    g.add("x" * 3_000)  # ~1,000 more -> 9,600 + 1,000 > 10,000
    with pytest.raises(ContextOverflow):
        g.check(0)


def test_guard_detects_server_truncation():
    g = ContextGuard(window=100_000, max_tokens=100)
    g.record(5_000, 10)
    with pytest.raises(ContextOverflow, match="truncated"):
        g.record(4_000, 10)


class FakePost:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, body, timeout=600.0):
        self.bodies.append((url, body))
        return self.replies.pop(0)


def reply(content="", tool_calls=None, prompt=100, out=10, done="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "prompt_eval_count": prompt, "eval_count": out, "done_reason": done}


def test_ollama_sends_num_ctx_seed_temperature_every_request():
    post = FakePost([reply("<think>x</think>Hi.")])
    s = OllamaProvider(QWEN, post=post).new_session("SYS", all_tool_schemas())
    s.add_user("hello")
    t = s.step()
    url, body = post.bodies[0]
    assert url == "http://ollama:1/api/chat" and body["stream"] is False
    assert body["options"] == {"num_ctx": 32_768, "num_predict": 4096, "temperature": 0.7, "seed": 5}
    assert body["messages"][0] == {"role": "system", "content": "SYS"} and body["tools"] == all_tool_schemas()
    assert body["think"] is False
    assert (t.text, t.stop_reason, t.usage.input_tokens) == ("Hi.", "end_turn", 100)


def test_ollama_parsed_and_unparsed_tool_calls():
    calls = [{"function": {"name": "read_ticket", "arguments": {"ticket_id": "T0001"}}}]
    text = 'Also <tool_call>{"name": "lookup_order", "arguments": {"order_id": O1}}</tool_call>'
    post = FakePost([reply(text, calls, done="stop")])
    s = OllamaProvider(QWEN, post=post).new_session("SYS", [])
    s.add_user("go")
    t = s.step()
    assert [(c.name, c.arguments) for c in t.tool_calls] == [
        ("read_ticket", {"ticket_id": "T0001"}),
        ("lookup_order", '{"name": "lookup_order", "arguments": {"order_id": O1}}'),  # raw -> format error
    ]
    assert t.stop_reason == "tool_use" and t.text == "Also"
    s.add_tool_results([ToolOutcome("call_1", "{}", False), ToolOutcome("call_2", "{}", True)])
    assert s.messages[-2:] == [{"role": "tool", "content": "{}"}, {"role": "tool", "content": "{}"}]


def test_unparsed_tool_call_without_name():
    assert unparsed_tool_calls("<tool_call>garbage") == [("", "garbage")]


def test_ollama_session_raises_on_truncation():
    post = FakePost([reply("a", prompt=5_000), reply("b", prompt=2_050)])
    s = OllamaProvider(QWEN, post=post).new_session("SYS", [])
    s.add_user("one")
    s.step()
    s.add_user("two")
    with pytest.raises(ContextOverflow):
        s.step()


class OverflowAfter:
    """Session that overflows on the given call number."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def add_user(self, text):
        pass

    def step(self):
        self.calls += 1
        if self.calls >= self.n:
            raise ContextOverflow("full")
        return ModelTurn(text="done", tool_calls=[], stop_reason="end_turn", usage=Usage(input_tokens=10))

    def add_tool_results(self, results):
        pass


class OneSession:
    def __init__(self, session):
        self.config, self.session = QWEN, session

    def new_session(self, system, tools):
        return self.session


def test_overflow_stops_episode_and_marks_remaining_tasks(state, tmp_path):
    cfg = EpisodeConfig(run_id="r", episode_id="e", seed=7, model="qwen3:8b", agent_role="support",
                        drift_condition="D1", control_condition="C1", provider=QWEN)
    tasks = [TaskSpec(task_id=f"t{i}", task_type="answer_query", instruction="x", ticket_id="T0001",
                      expected_fact="placed") for i in range(4)]
    agent = LLMAgent(OneSession(OverflowAfter(2)), "support")
    rec, _ = run_episode(cfg, tasks, agent, RunPaths.for_run(tmp_path, "r"), {}, state=state)
    assert rec.task_ends == {"t0": "done", "t1": "context_overflow", "t2": "not_run", "t3": "not_run"}
    assert [o.task_id for o in rec.task_outcomes] == ["t0", "t1"] and rec.n_tasks == 2
    assert rec.provider_config["temperature"] == 0.7 and rec.provider_config["context_window"] == 32_768


@pytest.mark.parametrize("role", ["support", "listing", "price_intel"])
def test_prompt_lists_exactly_the_roles_tools(role):
    p = system_prompt(role)
    listed = p.split("Tools your role is allowed to use: ")[1].split(".")[0].split(", ")
    assert set(listed) == set(ROLE_PERMISSIONS[role])
