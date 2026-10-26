"""Normalize token usage from any provider's response into one ``Usage`` record.

Handles, as SDK objects or plain dicts (e.g. ``requests.post(...).json()``):

- Anthropic Messages (``usage.input_tokens`` excludes cached tokens; cache reads
  and writes are reported separately, writes split into 5-minute and 1-hour TTL).
- OpenAI Chat Completions and OpenAI-compatible servers such as vLLM, SGLang,
  Ollama and LM Studio (``prompt_tokens`` includes cached tokens).
- OpenAI Responses API (``input_tokens`` includes cached tokens).
- Google Gemini (``usage_metadata``).
- vLLM offline ``LLM.generate()`` outputs (``RequestOutput``; tokens counted from ids).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class Usage:
    provider: str
    model: Optional[str] = None
    input_tokens: int = 0  # uncached input tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0  # 5-minute TTL (Anthropic) or the only write tier
    cache_write_1h_tokens: int = 0
    output_tokens: int = 0  # includes reasoning/thinking tokens
    reasoning_tokens: int = 0  # informational subset of output_tokens
    service_tier: Optional[str] = None
    speed: Optional[str] = None
    web_search_requests: int = 0

    @property
    def prompt_tokens(self) -> int:
        return (
            self.input_tokens + self.cache_read_tokens
            + self.cache_write_tokens + self.cache_write_1h_tokens
        )

    def to_dict(self) -> dict:
        return asdict(self)


def to_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _i(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _vllm_request_output(obj: Any, model: Optional[str]) -> Optional[Usage]:
    prompt_ids = getattr(obj, "prompt_token_ids", None)
    outputs = getattr(obj, "outputs", None)
    if prompt_ids is None or outputs is None:
        return None
    out = sum(len(getattr(o, "token_ids", None) or []) for o in outputs)
    cached = _i(getattr(obj, "num_cached_tokens", 0))
    return Usage(
        provider="local", model=model, input_tokens=len(prompt_ids) - cached,
        cache_read_tokens=cached, output_tokens=out,
    )


def extract_usage(
    response: Any, model: Optional[str] = None, provider: Optional[str] = None
) -> Optional[Usage]:
    """Return a ``Usage`` for a response object or dict, or None if it has no usage."""
    if isinstance(response, list):  # vLLM LLM.generate() returns a list
        parts = [extract_usage(r, model, provider) for r in response]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        total = parts[0]
        for p in parts[1:]:
            total.input_tokens += p.input_tokens
            total.cache_read_tokens += p.cache_read_tokens
            total.output_tokens += p.output_tokens
        return total

    vllm = _vllm_request_output(response, model)
    if vllm is not None:
        return vllm

    d = to_dict(response)
    if not isinstance(d, dict):
        return None
    model = model or d.get("model") or d.get("model_version")

    if d.get("usage_metadata") is not None:  # Gemini
        u = to_dict(d["usage_metadata"]) or {}
        cached = _i(u.get("cached_content_token_count"))
        thoughts = _i(u.get("thoughts_token_count"))
        return Usage(
            provider=provider or "gemini", model=model,
            input_tokens=_i(u.get("prompt_token_count")) - cached, cache_read_tokens=cached,
            output_tokens=_i(u.get("candidates_token_count")) + thoughts,
            reasoning_tokens=thoughts,
        )

    u = d.get("usage")
    if u is None and ("prompt_tokens" in d or "input_tokens" in d):
        u = d  # a bare usage object was passed
    u = to_dict(u)
    if not isinstance(u, dict):
        return None

    anthropic_like = (
        provider == "anthropic"
        or "cache_read_input_tokens" in u
        or "cache_creation_input_tokens" in u
        or d.get("type") == "message"
    )
    if anthropic_like:
        write_total = _i(u.get("cache_creation_input_tokens"))
        detail = to_dict(u.get("cache_creation")) or {}
        write_1h = _i(detail.get("ephemeral_1h_input_tokens"))
        server_tools = to_dict(u.get("server_tool_use")) or {}
        return Usage(
            provider=provider or "anthropic", model=model,
            input_tokens=_i(u.get("input_tokens")),
            cache_read_tokens=_i(u.get("cache_read_input_tokens")),
            cache_write_tokens=max(write_total - write_1h, 0),
            cache_write_1h_tokens=write_1h,
            output_tokens=_i(u.get("output_tokens")),
            service_tier=u.get("service_tier"), speed=u.get("speed"),
            web_search_requests=_i(server_tools.get("web_search_requests")),
        )

    if "prompt_tokens" in u:  # OpenAI Chat Completions / OpenAI-compatible servers
        prompt = _i(u.get("prompt_tokens"))
        in_detail = to_dict(u.get("prompt_tokens_details")) or {}
        out_detail = to_dict(u.get("completion_tokens_details")) or {}
        cached = _i(in_detail.get("cached_tokens"))
        return Usage(
            provider=provider or "openai", model=model,
            input_tokens=prompt - cached, cache_read_tokens=cached,
            output_tokens=_i(u.get("completion_tokens")),
            reasoning_tokens=_i(out_detail.get("reasoning_tokens")),
            service_tier=d.get("service_tier"),
        )

    if "input_tokens" in u:  # OpenAI Responses API
        in_detail = to_dict(u.get("input_tokens_details")) or {}
        out_detail = to_dict(u.get("output_tokens_details")) or {}
        cached = _i(in_detail.get("cached_tokens"))
        return Usage(
            provider=provider or "openai", model=model,
            input_tokens=_i(u.get("input_tokens")) - cached, cache_read_tokens=cached,
            output_tokens=_i(u.get("output_tokens")),
            reasoning_tokens=_i(out_detail.get("reasoning_tokens")),
            service_tier=d.get("service_tier"),
        )
    return None
