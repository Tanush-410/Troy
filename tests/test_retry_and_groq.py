"""Retries (identical re-sends, all logged, never skipping) and Groq-specific handling."""

import json
from types import SimpleNamespace as NS

import httpx2
import openai
import pytest

from agents.llm_agent import LLMAgent
from agents.providers.base import ProviderConfig, ToolOutcome
from agents.providers.openai_compat import OpenAICompatProvider
from agents.providers.retry import RequestTooLarge, RetriesExhausted, RetryPolicy, call_with_retries
from experiments.config import EpisodeConfig
from experiments.runner import RunPaths, run_episode
from policy.task import TaskSpec


def status_error(cls, code, headers=None, body=None):
    resp = httpx2.Response(code, request=httpx2.Request("POST", "https://api.groq.com/x"), headers=headers or {})
    return cls(f"error {code}", response=resp, body=body)


class Flaky:
    """Fails with the given errors, then returns `result`; records every call's arguments."""

    def __init__(self, errors, result="ok"):
        self.errors, self.result, self.calls = list(errors), result, 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_429_retried_with_backoff_and_retry_after_and_logged():
    waits = []
    fn = Flaky([status_error(openai.RateLimitError, 429, {"retry-after": "7"}),
                status_error(openai.RateLimitError, 429),
                status_error(openai.InternalServerError, 503)])
    log = []
    assert call_with_retries(fn, RetryPolicy(base_delay_s=2, max_delay_s=60), log, sleep=waits.append) == "ok"
    assert fn.calls == 4 and waits == [7.0, 4.0, 8.0]  # retry-after wins when larger than backoff
    assert [(e["attempt"], e["status"], e["retry_after"]) for e in log] == [(1, 429, 7.0), (2, 429, None),
                                                                           (3, 503, None)]


def test_backoff_is_capped():
    waits = []
    fn = Flaky([status_error(openai.RateLimitError, 429) for _ in range(5)])
    call_with_retries(fn, RetryPolicy(base_delay_s=10, max_delay_s=25), [], sleep=waits.append)
    assert waits == [10, 20, 25, 25, 25]


def test_exhausted_retries_raise_instead_of_skipping():
    fn = Flaky([status_error(openai.RateLimitError, 429) for _ in range(3)])
    with pytest.raises(RetriesExhausted):
        call_with_retries(fn, RetryPolicy(max_attempts=3), [], sleep=lambda s: None)
    assert fn.calls == 3


def test_client_errors_are_not_retried():
    fn = Flaky([status_error(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        call_with_retries(fn, RetryPolicy(), [], sleep=lambda s: None)
    assert fn.calls == 1


def test_request_larger_than_token_limit_is_fatal():
    fn = Flaky([status_error(openai.APIStatusError, 413)])
    with pytest.raises(RequestTooLarge):
        call_with_retries(fn, RetryPolicy(), [], sleep=lambda s: None)


# ------------------------------------------------------------ groq session

GROQ = ProviderConfig(kind="openai_compat", model="openai/gpt-oss-120b", context_window=131_072,
                      base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY",
                      temperature=0.7, seed=3, reasoning_effort="low",
                      retry=RetryPolicy(base_delay_s=0, max_delay_s=0))


def completion(tool_calls=None, content="", prompt=500, cached=0, out=20, finish="tool_calls"):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content, tool_calls=tool_calls))],
              usage=NS(prompt_tokens=prompt, completion_tokens=out,
                       prompt_tokens_details=NS(cached_tokens=cached)))


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.dumps({**kw, "messages": kw["messages"]}, sort_keys=True, default=str))
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_retry_resends_the_identical_request():
    client = FakeClient([status_error(openai.RateLimitError, 429), completion(content="hi", finish="stop")])
    s = OpenAICompatProvider(GROQ, client=client).new_session("SYS", [])
    s.add_user("go")
    t = s.step()
    assert len(client.requests) == 2 and client.requests[0] == client.requests[1]
    assert len(t.retries) == 1 and t.retries[0]["status"] == 429
    req = json.loads(client.requests[0])
    assert (req["seed"], req["temperature"], req["reasoning_effort"]) == (3, 0.7, "low")


def test_cached_tokens_are_reported_separately():
    client = FakeClient([completion(content="hi", finish="stop", prompt=1000, cached=600)])
    s = OpenAICompatProvider(GROQ, client=client).new_session("SYS", [])
    s.add_user("go")
    u = s.step().usage
    assert (u.input_tokens, u.cache_read_tokens, u.total_input) == (400, 600, 1000)


def test_tool_use_failed_becomes_logged_format_error(state, tmp_path):
    failed = status_error(openai.BadRequestError, 400, body={"error": {
        "code": "tool_use_failed", "failed_generation": '<function=read_ticket>{"ticket_id": TKT-0001'}})
    client = FakeClient([failed, completion(content="Done.", finish="stop")])
    provider = OpenAICompatProvider(GROQ, client=client)
    cfg = EpisodeConfig(run_id="r", episode_id="e", seed=7, model=GROQ.model, agent_role="support",
                        drift_condition="D0", control_condition="C1", provider=GROQ)
    task = TaskSpec(task_id="t1", task_type="answer_query", instruction="x", ticket_id="TKT-0001",
                    expected_fact="placed")
    paths = RunPaths.for_run(tmp_path, "r")
    rec, _ = run_episode(cfg, [task], LLMAgent(provider, "support"), paths, {}, state=state)
    (event,) = [json.loads(line) for line in paths.events.read_text().splitlines()]
    assert event["action"] == "read_ticket" and event["format_error"] and event["drift_type"] is None
    assert "tool_use_failed" in event["format_error_detail"]
    # the model was told, in a user turn (no server-side tool_call id exists)
    second = json.loads(client.requests[1])["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Your tool call failed")


def test_non_tool_bad_request_still_raises():
    client = FakeClient([status_error(openai.BadRequestError, 400, body={"error": {"code": "invalid_request"}})])
    s = OpenAICompatProvider(GROQ, client=client).new_session("SYS", [])
    s.add_user("go")
    with pytest.raises(openai.BadRequestError):
        s.step()


def test_retries_reach_episode_record_and_retry_log(state, tmp_path):
    client = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "0"}),
                         completion(content="Done.", finish="stop")])
    cfg = EpisodeConfig(run_id="r", episode_id="e", seed=7, model=GROQ.model, agent_role="support",
                        drift_condition="D0", control_condition="C1", provider=GROQ)
    task = TaskSpec(task_id="t1", task_type="answer_query", instruction="x", ticket_id="TKT-0001",
                    expected_fact="placed")
    paths = RunPaths.for_run(tmp_path, "r")
    rec, _ = run_episode(cfg, [task], LLMAgent(OpenAICompatProvider(GROQ, client=client), "support"),
                         paths, {}, state=state)
    assert rec.api_retries == 1 and rec.task_ends == {"t1": "done"}
    (line,) = paths.retries.read_text().splitlines()
    assert json.loads(line)["status"] == 429 and json.loads(line)["task_id"] == "t1"
    assert rec.provider_config["retry"]["max_attempts"] == 10
    assert (rec.model, rec.reasoning_setting) == ("openai/gpt-oss-120b", "reasoning_effort=low")


def test_failed_tool_call_session_bookkeeping():
    failed = status_error(openai.BadRequestError, 400, body={"code": "tool_use_failed", "failed_generation": "blah"})
    s = OpenAICompatProvider(GROQ, client=FakeClient([failed])).new_session("SYS", [])
    s.add_user("go")
    t = s.step()
    (call,) = t.tool_calls
    assert (call.name, call.arguments, call.parse_error is not None) == ("", "blah", True)
    s.add_tool_results([ToolOutcome(call.id, '{"ok": false}', True)])
    assert s.messages[-2:] == [{"role": "assistant", "content": "blah"},
                               {"role": "user", "content": 'Your tool call failed: {"ok": false}'}]
