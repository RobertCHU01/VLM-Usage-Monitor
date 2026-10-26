"""Offline tests: real openai/anthropic SDK clients talking to an httpx mock server.

Run: VLM_USAGE_AUTO_UPDATE=0 python -m pytest tests -q
Prices come from the bundled snapshot (auto-update and the download cache are
disabled below), so expected costs are exact.
"""

import asyncio
import json
import os

import pytest

os.environ["VLM_USAGE_AUTO_UPDATE"] = "0"
os.environ["VLM_USAGE_CACHE_DIR"] = "/nonexistent-vlm-usage-cache"

try:  # anthropic>=1 / openai>=3 use httpx2; older SDKs use httpx
    import httpx2 as httpx
except ImportError:
    import httpx  # noqa: E402

import vlm_usage_monitor as vum  # noqa: E402
from vlm_usage_monitor.cli import main as cli_main  # noqa: E402

ANTHROPIC_MSG = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {
        "input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
    },
}
# Opus 5.5: $4 in, $20 out, $0.20 cache read, $5 5m write, $8 1h write (per 1M)
ANTHROPIC_COST = (1000 * 4 + 2000 * 0.2 + 1000 * 5 + 2000 * 8 + 500 * 20) / 1e6

CHAT = {
    "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5.5",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 400},
        "completion_tokens_details": {"reasoning_tokens": 50},
    },
}
# gpt-5.5: $5 in, $0.50 cached, $30 out
CHAT_COST = (600 * 5 + 400 * 0.5 + 200 * 30) / 1e6

RESPONSE = {
    "id": "r1", "object": "response", "created_at": 0, "model": "gpt-5.5", "status": "completed",
    "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "usage": {
        "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}, "output_tokens": 200,
        "output_tokens_details": {"reasoning_tokens": 50}, "total_tokens": 1200,
    },
}


def sse(events, named=False):
    out = []
    for ev in events:
        if named:
            out.append(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n")
        else:
            out.append(f"data: {json.dumps(ev)}\n\n")
    if not named:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


ANTHROPIC_STREAM = sse([
    {"type": "message_start", "message": {**ANTHROPIC_MSG, "content": [], "stop_reason": None,
                                           "usage": {**ANTHROPIC_MSG["usage"], "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 500}},
    {"type": "message_stop"},
], named=True)

CHAT_STREAM = sse([
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5.5",
     "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5.5",
     "choices": [], "usage": CHAT["usage"]},
])


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    path = request.url.path
    if path.endswith("/messages"):
        if body.get("stream"):
            return httpx.Response(200, content=ANTHROPIC_STREAM, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=ANTHROPIC_MSG)
    if path.endswith("/chat/completions"):
        if body.get("stream"):
            return httpx.Response(200, content=CHAT_STREAM, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={**CHAT, "model": body["model"]})
    if path.endswith("/responses"):
        return httpx.Response(200, json=RESPONSE)
    return httpx.Response(404)


@pytest.fixture
def tracker(tmp_path):
    return vum.UsageTracker(run="test", log_path=str(tmp_path / "log.jsonl"), summary_at_exit=False)


def anthropic_client(tracker, async_=False):
    import anthropic
    if async_:
        c = anthropic.AsyncAnthropic(api_key="x", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    else:
        c = anthropic.Anthropic(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return vum.track(c, tracker=tracker, tag="planner")


def openai_client(tracker, base_url=None, async_=False, **kw):
    import openai
    if async_:
        c = openai.AsyncOpenAI(api_key="x", base_url=base_url,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    else:
        c = openai.OpenAI(api_key="x", base_url=base_url,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return vum.track(c, tracker=tracker, **kw)


MSG_KW = dict(model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
CHAT_KW = dict(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}])


def test_anthropic_create(tracker):
    resp = anthropic_client(tracker).messages.create(**MSG_KW)
    assert resp.content[0].text == "hi"
    rec = tracker.records[-1]
    assert (rec.input_tokens, rec.cache_read_tokens, rec.cache_write_tokens, rec.cache_write_1h_tokens,
            rec.output_tokens) == (1000, 2000, 1000, 2000, 500)
    assert rec.cost_usd == pytest.approx(ANTHROPIC_COST)
    assert rec.tag == "planner" and rec.provider == "anthropic"


def test_anthropic_stream_true_and_helper(tracker):
    client = anthropic_client(tracker)
    for _ in client.messages.create(stream=True, **MSG_KW):
        pass
    assert tracker.records[-1].output_tokens == 500
    assert tracker.records[-1].cost_usd == pytest.approx(ANTHROPIC_COST)
    with client.messages.stream(**MSG_KW) as stream:
        assert stream.get_final_message().content[0].text == "hi"
    assert tracker.n_calls == 2
    assert tracker.records[-1].cost_usd == pytest.approx(ANTHROPIC_COST)


def test_anthropic_async(tracker):
    client = anthropic_client(tracker, async_=True)

    async def go():
        await client.messages.create(**MSG_KW)
        async with client.messages.stream(**MSG_KW) as stream:
            await stream.get_final_message()

    asyncio.run(go())
    assert tracker.n_calls == 2
    assert tracker.total_cost == pytest.approx(2 * ANTHROPIC_COST)


def test_openai_chat_and_responses(tracker):
    client = openai_client(tracker)
    client.chat.completions.create(**CHAT_KW)
    rec = tracker.records[-1]
    assert (rec.input_tokens, rec.cache_read_tokens, rec.output_tokens, rec.reasoning_tokens) == (600, 400, 200, 50)
    assert rec.cost_usd == pytest.approx(CHAT_COST)
    client.responses.create(model="gpt-5.5", input="hi")
    assert tracker.records[-1].cost_usd == pytest.approx(CHAT_COST)


def test_openai_stream_and_inject(tracker):
    client = openai_client(tracker, inject_stream_usage=True)
    chunks = list(client.chat.completions.create(stream=True, **CHAT_KW))
    assert len(chunks) == 2
    assert tracker.records[-1].cost_usd == pytest.approx(CHAT_COST)


def test_openai_async(tracker):
    client = openai_client(tracker, async_=True)

    async def go():
        await client.chat.completions.create(**CHAT_KW)
        stream = await client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **CHAT_KW)
        async for _ in stream:
            pass

    asyncio.run(go())
    assert tracker.n_calls == 2


def test_vllm_local_unpriced_then_registered(tracker):
    client = openai_client(tracker, base_url="http://localhost:8000/v1")
    client.chat.completions.create(model="Qwen/Qwen2.5-VL-7B-Instruct", messages=[{"role": "user", "content": "hi"}])
    rec = tracker.records[-1]
    assert rec.provider == "local" and rec.cost_usd is None and rec.output_tokens == 200
    vum.register_price("Qwen/Qwen2.5-VL-7B-Instruct", input=0.2, output=0.6)
    client.chat.completions.create(model="Qwen/Qwen2.5-VL-7B-Instruct", messages=[{"role": "user", "content": "hi"}])
    # no cache-read price registered: cached tokens charged at the input rate
    assert tracker.records[-1].cost_usd == pytest.approx((1000 * 0.2 + 200 * 0.6) / 1e6)


def test_raw_dicts_and_tokens(tracker):
    tracker.record(CHAT)  # e.g. requests.post(...).json()
    tracker.record(ANTHROPIC_MSG)
    tracker.record({"model_version": "gemini-2.5-pro", "usage_metadata": {
        "prompt_token_count": 1000, "candidates_token_count": 100, "thoughts_token_count": 50}})
    assert tracker.records[-1].cost_usd == pytest.approx((1000 * 1.25 + 150 * 10) / 1e6)
    tracker.record_tokens("claude-opus-5-5", input_tokens=1_000_000, output_tokens=0)
    assert tracker.records[-1].cost_usd == pytest.approx(4.0)
    assert tracker.records[0].cost_usd == pytest.approx(CHAT_COST)
    assert tracker.records[1].cost_usd == pytest.approx(ANTHROPIC_COST)


def test_model_name_variants():
    for name in ["claude-opus-5-5", "anthropic/claude-opus-5-5", "anthropic.claude-opus-5-5",
                 "claude-sonnet-4-5-20250929", "gpt-5.5-2026-04-23", "openai/gpt-5.5"]:
        assert vum.lookup_price(name) is not None, name
    assert vum.lookup_price("/scratch/models/my-finetuned-qwen") is None


def test_long_context_and_batch_tiers():
    u = vum.Usage(provider="anthropic", model="claude-haiku-5-5", input_tokens=150_000, output_tokens=1000)
    assert vum.compute_cost(u).total_usd == pytest.approx((150_000 * 0.5 + 1000 * 2.5) / 1e6)
    u = vum.Usage(provider="anthropic", model="claude-opus-5-5", input_tokens=1000, output_tokens=1000,
                  service_tier="batch")
    assert vum.compute_cost(u).total_usd == pytest.approx((1000 * 2 + 1000 * 10) / 1e6)


def test_budget(tmp_path):
    t = vum.UsageTracker(budget_usd=0.05, hard_budget=True, summary_at_exit=False, verbose=False)
    t.record(ANTHROPIC_MSG)
    with pytest.raises(vum.BudgetExceededError):
        t.record(ANTHROPIC_MSG)


def test_tag_context_and_cli_summary(tracker, capsys):
    with tracker.tag("judge"):
        tracker.record(CHAT)
    tracker.record(ANTHROPIC_MSG, tag="planner")
    assert [r.tag for r in tracker.records] == ["judge", "planner"]
    capsys.readouterr()
    assert cli_main(["summary", tracker.log_path, "--by", "model,tag", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert sum(r["cost_usd"] for r in rows) == pytest.approx(CHAT_COST + ANTHROPIC_COST)
    assert cli_main(["summary", tracker.log_path]) == 0
    assert "TOTAL" in capsys.readouterr().out


def test_vllm_offline_request_output(tracker):
    class Out:
        token_ids = [1, 2, 3]

    class RequestOutput:
        prompt_token_ids = list(range(10))
        outputs = [Out()]

    rec = tracker.record([RequestOutput(), RequestOutput()], model="Qwen/Qwen2.5-VL-7B-Instruct")
    assert (rec.input_tokens, rec.output_tokens) == (20, 6)


def test_price_correction_applied():
    u = vum.Usage(provider="anthropic", model="claude-sonnet-5-5", cache_read_tokens=1_000_000)
    assert vum.compute_cost(u).total_usd == pytest.approx(0.20)


def test_with_options_copy_is_tracked(tracker):
    client = openai_client(tracker).with_options(timeout=5)
    client.chat.completions.create(**CHAT_KW)
    assert tracker.n_calls == 1
