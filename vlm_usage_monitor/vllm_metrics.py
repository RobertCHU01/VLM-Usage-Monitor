"""Server-side token totals from a vLLM server's Prometheus ``/metrics`` endpoint.

Counts every request the server has handled since it started, from any
client, so it also covers code that was not wrapped with ``track()``.
"""

from __future__ import annotations

import re
import urllib.request
from collections import defaultdict
from typing import Dict

_COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:request_success_total": "requests",
    "vllm:prefix_cache_hits_total": "prefix_cache_hit_tokens",
    "vllm:prefix_cache_queries_total": "prefix_cache_query_tokens",
}
_LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$')
_MODEL = re.compile(r'model_name="([^"]*)"')


def vllm_server_tokens(base_url: str, timeout_s: float = 5.0) -> Dict[str, Dict[str, float]]:
    """Return ``{model_name: {prompt_tokens, generation_tokens, requests, ...}}``.

    ``base_url`` may include the ``/v1`` suffix used by OpenAI clients.
    """
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    with urllib.request.urlopen(f"{root}/metrics", timeout=timeout_s) as resp:
        text = resp.read().decode("utf-8")
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m or m.group(1) not in _COUNTERS:
            continue
        model = _MODEL.search(m.group(2) or "")
        out[model.group(1) if model else "?"][_COUNTERS[m.group(1)]] += float(m.group(3))
    return {k: dict(v) for k, v in out.items()}
