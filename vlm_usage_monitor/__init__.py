"""Track the token usage and USD cost of every VLM/LLM call in an experiment.

Quick start:

    from vlm_usage_monitor import track, configure
    configure(run="sweep_v2", log_path="logs/vlm_usage.jsonl", budget_usd=20)
    client = track(anthropic.Anthropic())      # or openai.OpenAI(...), incl. vLLM
    client.messages.create(...)                # prints tokens + $ for this call

    # anything else (raw HTTP JSON, Gemini, vLLM LLM.generate outputs):
    from vlm_usage_monitor import record, record_tokens
    record(resp.json())
    record_tokens("Qwen/Qwen2.5-VL-7B-Instruct", input_tokens=812, output_tokens=64)

Summarize logs: ``vlm-usage summary logs/vlm_usage.jsonl --by model,tag``.
"""

from .pricing import compute_cost, describe as describe_price, lookup as lookup_price, register_price, update_prices
from .tracker import (
    BudgetExceededError, CallRecord, UsageTracker, configure, get_tracker, load_records,
    record, record_tokens, summarize, tag,
)
from .usage import Usage, extract_usage
from .vllm_metrics import vllm_server_tokens
from .wrap import track

__version__ = "0.1.0"

__all__ = [
    "BudgetExceededError", "CallRecord", "Usage", "UsageTracker", "compute_cost", "configure",
    "describe_price", "extract_usage", "get_tracker", "load_records", "lookup_price", "record",
    "record_tokens", "register_price", "summarize", "tag", "track", "update_prices",
    "vllm_server_tokens",
]
