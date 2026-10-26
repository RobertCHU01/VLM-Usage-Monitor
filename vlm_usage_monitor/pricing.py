"""Model price lookup.

Prices come from LiteLLM's community-maintained table
(model_prices_and_context_window.json), which covers Anthropic, OpenAI, Gemini
and many other providers, including cache, batch, flex/priority and
long-context tiers. Sources, highest priority first:

1. Prices you register with ``register_price()`` or put in the JSON file named
   by ``$VLM_USAGE_PRICES`` (use this for local models or negotiated rates).
2. A downloaded copy of the LiteLLM table in ``~/.cache/vlm_usage_monitor``,
   refreshed automatically when older than 7 days (disable with
   ``VLM_USAGE_AUTO_UPDATE=0``, e.g. on nodes without internet).
3. The snapshot bundled in ``data/prices_snapshot.json``.

``data/price_corrections.json`` is applied on top of 2 and 3 for the few
entries where LiteLLM disagrees with the provider's published price.

All rates are USD per token internally; the public helpers that take or show
prices use USD per 1M tokens, the unit provider pricing pages use.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
SNAPSHOT_PATH = Path(__file__).parent / "data" / "prices_snapshot.json"
CORRECTIONS_PATH = Path(__file__).parent / "data" / "price_corrections.json"
MAX_CACHE_AGE_S = 7 * 24 * 3600

_TIER_SUFFIX = {"batch": "_batches", "flex": "_flex", "priority": "_priority"}
_STRIP_PREFIXES = (
    "anthropic/", "openai/", "gemini/", "models/", "us.anthropic.", "eu.anthropic.",
    "global.anthropic.", "anthropic.",
)
_DATE_SUFFIX = re.compile(r"(-\d{8}|-\d{4}-\d{2}-\d{2}|@\d{8})$")

_lock = threading.Lock()
_table: Optional[Dict[str, dict]] = None
_overrides: Dict[str, dict] = {}
_download_tried = False


def cache_path() -> Path:
    root = os.environ.get("VLM_USAGE_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vlm_usage_monitor"
    )
    return Path(root) / "litellm_prices.json"


def update_prices(timeout_s: float = 10.0) -> Path:
    """Download the latest LiteLLM price table into the cache. Raises on failure."""
    global _table
    with urllib.request.urlopen(LITELLM_URL, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    with _lock:
        _table = None
    return path


def _maybe_auto_update() -> None:
    global _download_tried
    if _download_tried or os.environ.get("VLM_USAGE_AUTO_UPDATE", "1") == "0":
        return
    _download_tried = True
    path = cache_path()
    if path.exists() and time.time() - path.stat().st_mtime < MAX_CACHE_AGE_S:
        return
    try:
        update_prices(timeout_s=5.0)
    except Exception:  # offline node, proxy, etc.: fall back to cache/snapshot
        pass


def _load_table() -> Dict[str, dict]:
    global _table
    _maybe_auto_update()
    with _lock:
        if _table is not None:
            return _table
        table = json.loads(SNAPSHOT_PATH.read_text())
        table.pop("_meta", None)
        path = cache_path()
        if path.exists():
            try:
                table.update(json.loads(path.read_text()))
            except Exception as exc:
                warnings.warn(f"vlm_usage_monitor: ignoring unreadable price cache {path}: {exc}")
        for name, fix in json.loads(CORRECTIONS_PATH.read_text()).items():
            if name in table and isinstance(fix, dict):
                table[name] = {**table[name], **{k: v for k, v in fix.items() if not k.startswith("_")}}
        env_file = os.environ.get("VLM_USAGE_PRICES")
        if env_file:
            for name, spec in json.loads(Path(env_file).read_text()).items():
                table[name] = _entry_from_per_mtok(**spec)
        _table = table
        return table


def _entry_from_per_mtok(
    input: float = 0.0,
    output: float = 0.0,
    cache_read: Optional[float] = None,
    cache_write: Optional[float] = None,
    cache_write_1h: Optional[float] = None,
    provider: str = "custom",
) -> dict:
    entry = {
        "input_cost_per_token": input / 1e6,
        "output_cost_per_token": output / 1e6,
        "litellm_provider": provider,
    }
    if cache_read is not None:
        entry["cache_read_input_token_cost"] = cache_read / 1e6
    if cache_write is not None:
        entry["cache_creation_input_token_cost"] = cache_write / 1e6
    if cache_write_1h is not None:
        entry["cache_creation_input_token_cost_above_1hr"] = cache_write_1h / 1e6
    return entry


def register_price(
    model: str,
    input: float,
    output: float,
    cache_read: Optional[float] = None,
    cache_write: Optional[float] = None,
    cache_write_1h: Optional[float] = None,
) -> None:
    """Set a model's price in USD per 1M tokens. Overrides every other source.

    Useful for local models (e.g. price a vLLM Qwen at what a hosted provider
    would charge, or at your GPU cost per token) and for custom deployments.
    """
    _overrides[model] = _entry_from_per_mtok(
        input, output, cache_read, cache_write, cache_write_1h
    )


def lookup(model: Optional[str]) -> Optional[tuple]:
    """Return ``(matched_key, entry)`` for a model name, or None if unpriced."""
    if not model:
        return None
    if model in _overrides:
        return model, _overrides[model]
    table = _load_table()
    candidates = [model, model.lower()]
    name = model.lower()
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"(-v\d+)?:\d+$", "", name)  # bedrock "...-v1:0"
    candidates += [name, _DATE_SUFFIX.sub("", name)]
    for cand in candidates:
        if cand in _overrides:
            return cand, _overrides[cand]
        entry = table.get(cand)
        if entry and "input_cost_per_token" in entry:
            return cand, entry
    for provider in ("anthropic", "openai", "gemini", "xai", "deepseek", "mistral"):
        entry = table.get(f"{provider}/{candidates[-1]}")
        if entry and "input_cost_per_token" in entry:
            return f"{provider}/{candidates[-1]}", entry
    return None


def _rate(entry: dict, base: str, prompt_tokens: int, tier: str) -> Optional[float]:
    """Per-token rate for ``base`` at this prompt length and service tier."""
    sfx = _TIER_SUFFIX.get(tier or "", "")
    above = None
    pat = re.compile(re.escape(base) + r"_above_(\d+)k_tokens$")
    for key in entry:
        m = pat.match(key)
        if m and prompt_tokens > int(m.group(1)) * 1000:
            if above is None or int(m.group(1)) > int(above.group(1)):
                above = m
    if above is not None:
        key = above.group(0)
        if sfx and key + sfx in entry:
            return entry[key + sfx]
        rate = entry[key]
        if sfx and base + sfx in entry and entry.get(base):
            rate *= entry[base + sfx] / entry[base]
        return rate
    if sfx and base + sfx in entry:
        return entry[base + sfx]
    return entry.get(base)


@dataclass
class Cost:
    total_usd: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    price_key: Optional[str] = None
    notes: list = field(default_factory=list)


def compute_cost(usage, model: Optional[str] = None) -> Optional[Cost]:
    """Cost of one call from a normalized ``Usage``. None if the model is unpriced."""
    found = lookup(model or usage.model)
    if found is None:
        return None
    key, entry = found
    prompt = usage.prompt_tokens
    tier = (usage.service_tier or "").lower()
    if tier in ("default", "standard", "auto", "scale"):
        tier = ""
    notes = []
    inp = _rate(entry, "input_cost_per_token", prompt, tier) or 0.0
    out = _rate(entry, "output_cost_per_token", prompt, tier) or 0.0
    is_anthropic = entry.get("litellm_provider", "").startswith(("anthropic", "bedrock"))

    read = _rate(entry, "cache_read_input_token_cost", prompt, tier)
    if read is None and usage.cache_read_tokens:
        read = inp
        notes.append("no cache-read price; charged at input rate")
    write = _rate(entry, "cache_creation_input_token_cost", prompt, tier)
    if write is None and usage.cache_write_tokens:
        write = inp * (1.25 if is_anthropic else 1.0)
        notes.append("no cache-write price; estimated")
    write_1h = _rate(entry, "cache_creation_input_token_cost_above_1hr", prompt, tier)
    if write_1h is None and usage.cache_write_1h_tokens:
        write_1h = inp * 2.0
        notes.append("no 1h cache-write price; estimated as 2x input")

    breakdown = {
        "input": usage.input_tokens * inp,
        "cache_read": usage.cache_read_tokens * (read or 0.0),
        "cache_write": usage.cache_write_tokens * (write or 0.0),
        "cache_write_1h": usage.cache_write_1h_tokens * (write_1h or 0.0),
        "output": usage.output_tokens * out,
    }
    total = sum(breakdown.values())
    if (usage.speed or "") == "fast":
        total *= 2.0
        notes.append("fast mode billed as 2x standard")
    if usage.web_search_requests:
        # Anthropic web search: $10 per 1,000 searches.
        breakdown["web_search"] = usage.web_search_requests * 0.01
        total += breakdown["web_search"]
    return Cost(total_usd=total, breakdown=breakdown, price_key=key, notes=notes)


def describe(model: str) -> Optional[dict]:
    """Human-readable rates (USD per 1M tokens) for a model, for the CLI."""
    found = lookup(model)
    if found is None:
        return None
    key, entry = found
    out = {"matched": key, "provider": entry.get("litellm_provider")}
    for k, v in sorted(entry.items()):
        if isinstance(v, (int, float)) and "cost" in k and "token" in k:
            out[k.replace("_cost_per_token", "").replace("_token_cost", "")] = round(v * 1e6, 6)
    return out
