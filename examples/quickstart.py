"""Minimal end-to-end example. Needs the SDK(s) and API key(s) you want to try.

    VLM_USAGE_LOG=logs/vlm_usage.jsonl python examples/quickstart.py
    vlm-usage summary logs/vlm_usage.jsonl --by model,tag
"""

import os

from vlm_usage_monitor import configure, get_tracker, tag, track

configure(run="quickstart", log_path=os.environ.get("VLM_USAGE_LOG", "logs/vlm_usage.jsonl"), budget_usd=1.0)

if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic

    claude = track(anthropic.Anthropic(), tag="claude")
    claude.messages.create(
        model="claude-haiku-5-5", max_tokens=64,
        messages=[{"role": "user", "content": "Say hi in three words."}],
    )

if os.environ.get("OPENAI_API_KEY"):
    import openai

    gpt = track(openai.OpenAI(), tag="openai")
    gpt.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": "Say hi."}])

if os.environ.get("VLLM_BASE_URL"):  # e.g. http://localhost:8000/v1
    import openai

    local = track(openai.OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key="EMPTY"))
    model = local.models.list().data[0].id
    with tag("local"):
        local.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hi."}])

print(f"Tracked ${get_tracker().total_cost:.6f} over {get_tracker().n_calls} calls")
