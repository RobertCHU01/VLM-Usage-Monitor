"""Per-call cost/token records, running totals, JSONL logging and budgets."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import datetime as _dt
import json
import math
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from . import pricing
from .usage import Usage, extract_usage

_current_tag: contextvars.ContextVar = contextvars.ContextVar("vlm_usage_tag", default=None)

TOKEN_FIELDS = (
    "input_tokens", "cache_read_tokens", "cache_write_tokens",
    "cache_write_1h_tokens", "output_tokens", "reasoning_tokens",
)


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class CallRecord:
    ts: str
    run: str
    tag: Optional[str]
    provider: str
    model: Optional[str]
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: Optional[float]  # None when the model has no known price (e.g. local)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    price_key: Optional[str] = None
    service_tier: Optional[str] = None
    latency_s: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    host: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "")


def fmt_usd(x: float) -> str:
    """4 decimals for normal amounts, 3 significant digits for sub-cent ones."""
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.{2 - math.floor(math.log10(abs(x)))}f}"


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


class UsageTracker:
    """Collects one record per VLM/LLM call.

    Args:
        run: run name stored with every record (default ``$VLM_USAGE_RUN`` or
            the script name).
        log_path: JSONL file every record is appended to (default
            ``$VLM_USAGE_LOG``; None keeps records in memory only). Safe to
            share between processes; summarize with ``vlm-usage summary``.
        budget_usd: warn once when the tracked cost passes this (default
            ``$VLM_USAGE_BUDGET``). With ``hard_budget=True`` raise
            ``BudgetExceededError`` instead, stopping the experiment.
        verbose: print one line per call to stderr (default ``$VLM_USAGE_VERBOSE`` or True).
        summary_at_exit: print a per-model summary when the process exits.
    """

    def __init__(
        self,
        run: Optional[str] = None,
        log_path: Optional[str] = None,
        budget_usd: Optional[float] = None,
        hard_budget: bool = False,
        verbose: Optional[bool] = None,
        summary_at_exit: Optional[bool] = None,
    ):
        self.run = run or os.environ.get("VLM_USAGE_RUN") or os.path.basename(sys.argv[0] or "python")
        self.log_path = log_path if log_path is not None else os.environ.get("VLM_USAGE_LOG")
        if budget_usd is None and os.environ.get("VLM_USAGE_BUDGET"):
            budget_usd = float(os.environ["VLM_USAGE_BUDGET"])
        self.budget_usd = budget_usd
        self.hard_budget = hard_budget
        self.verbose = _env_flag("VLM_USAGE_VERBOSE", True) if verbose is None else verbose
        self.records: List[CallRecord] = []
        self._lock = threading.Lock()
        self._budget_warned = False
        self._warned_unpriced: set = set()
        if summary_at_exit is None:
            summary_at_exit = _env_flag("VLM_USAGE_SUMMARY_AT_EXIT", True)
        if summary_at_exit:
            atexit.register(self._exit_summary)

    # ------------------------------------------------------------------ record
    def record(
        self,
        response: Any,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tag: Optional[str] = None,
        latency_s: Optional[float] = None,
        service_tier: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Optional[CallRecord]:
        """Record one response (SDK object or dict). Returns None if it has no usage.

        ``model`` overrides the model name in the response (useful when a local
        server reports a filesystem path). ``service_tier="batch"`` prices a
        Message Batches / Batch API result at batch rates.
        """
        usage = extract_usage(response, model=model, provider=provider)
        if usage is None:
            return None
        if model:
            usage.model = model
        if service_tier:
            usage.service_tier = service_tier
        return self.record_usage(usage, tag=tag, latency_s=latency_s, extra=extra)

    def record_tokens(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        reasoning_tokens: int = 0,
        provider: str = "custom",
        service_tier: Optional[str] = None,
        tag: Optional[str] = None,
        latency_s: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> CallRecord:
        """Record a call from raw token counts (HF ``generate``, custom servers, ...)."""
        usage = Usage(
            provider=provider, model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens, cache_write_1h_tokens=cache_write_1h_tokens,
            reasoning_tokens=reasoning_tokens, service_tier=service_tier,
        )
        return self.record_usage(usage, tag=tag, latency_s=latency_s, extra=extra)

    def record_usage(
        self,
        usage: Usage,
        *,
        tag: Optional[str] = None,
        latency_s: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> CallRecord:
        cost = pricing.compute_cost(usage)
        provider = usage.provider
        if cost is None and provider in ("openai", "openai_compatible"):
            provider = "local"  # an unpriced model behind an OpenAI-style API: vLLM etc.
        rec = CallRecord(
            ts=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            run=self.run,
            tag=tag if tag is not None else _current_tag.get(),
            provider=provider,
            model=usage.model,
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_write_1h_tokens=usage.cache_write_1h_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost_usd=None if cost is None else round(cost.total_usd, 8),
            cost_breakdown={} if cost is None else {k: round(v, 8) for k, v in cost.breakdown.items() if v},
            price_key=None if cost is None else cost.price_key,
            service_tier=usage.service_tier,
            latency_s=None if latency_s is None else round(latency_s, 3),
            notes=[] if cost is None else cost.notes,
            extra=extra or {},
        )
        with self._lock:
            self.records.append(rec)
            if self.log_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            total = self.total_cost
        if self.verbose:
            self._print_call(rec, total)
        self._check_budget(total)
        return rec

    # ------------------------------------------------------------------ tags
    @contextlib.contextmanager
    def tag(self, name: str):
        """Attribute every call inside the block to ``name`` (e.g. "planner", "judge")."""
        token = _current_tag.set(name)
        try:
            yield
        finally:
            _current_tag.reset(token)

    # ------------------------------------------------------------------ totals
    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.records)

    @property
    def n_calls(self) -> int:
        return len(self.records)

    def summary(self, by: Iterable[str] = ("model",)) -> List[dict]:
        return summarize([r.to_dict() for r in self.records], by=by)

    def summary_str(self, by: Iterable[str] = ("model",)) -> str:
        return format_summary(self.summary(by=by), by=by)

    def print_summary(self, by: Iterable[str] = ("model",), file=None) -> None:
        print(self.summary_str(by=by), file=file or sys.stderr)

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self._budget_warned = False

    # ------------------------------------------------------------------ internals
    def _print_call(self, rec: CallRecord, total: float) -> None:
        cache = ""
        if rec.cache_read_tokens or rec.cache_write_tokens or rec.cache_write_1h_tokens:
            cache = f" cache_read={_fmt_tokens(rec.cache_read_tokens)} cache_write={_fmt_tokens(rec.cache_write_tokens + rec.cache_write_1h_tokens)}"
        reason = f" (reasoning {_fmt_tokens(rec.reasoning_tokens)})" if rec.reasoning_tokens else ""
        speed = ""
        if rec.latency_s:
            speed = f" {rec.latency_s:.2f}s"
            if rec.output_tokens:
                speed += f" {rec.output_tokens / rec.latency_s:.1f} tok/s"
        cost = "unpriced" if rec.cost_usd is None else f"${fmt_usd(rec.cost_usd)}"
        tag = f"[{rec.tag}] " if rec.tag else ""
        print(
            f"[vlm-usage] {tag}{rec.model} in={_fmt_tokens(rec.input_tokens)}{cache} "
            f"out={_fmt_tokens(rec.output_tokens)}{reason} {cost}{speed} | "
            f"run total ${fmt_usd(total)} over {self.n_calls} calls",
            file=sys.stderr,
        )
        if rec.cost_usd is None and rec.model not in self._warned_unpriced:
            self._warned_unpriced.add(rec.model)
            print(
                f"[vlm-usage] no price for {rec.model!r}: tracking tokens only "
                "(set one with vlm_usage_monitor.register_price if you want a $ estimate)",
                file=sys.stderr,
            )

    def _check_budget(self, total: float) -> None:
        if self.budget_usd is None or total <= self.budget_usd:
            return
        msg = f"VLM spend ${fmt_usd(total)} exceeded budget ${self.budget_usd:.2f} (run {self.run})"
        if self.hard_budget:
            raise BudgetExceededError(msg)
        if not self._budget_warned:
            self._budget_warned = True
            print(f"[vlm-usage] WARNING: {msg}", file=sys.stderr)

    def _exit_summary(self) -> None:
        if self.records:
            try:
                self.print_summary()
            except Exception:
                pass


# ---------------------------------------------------------------------- summaries
def summarize(records: Iterable[dict], by: Iterable[str] = ("model",)) -> List[dict]:
    by = tuple(by)
    groups: Dict[tuple, dict] = defaultdict(
        lambda: {"calls": 0, "unpriced_calls": 0, "cost_usd": 0.0, **{f: 0 for f in TOKEN_FIELDS}}
    )
    for r in records:
        g = groups[tuple(r.get(k) for k in by)]
        g["calls"] += 1
        for f in TOKEN_FIELDS:
            g[f] += int(r.get(f) or 0)
        if r.get("cost_usd") is None:
            g["unpriced_calls"] += 1
        else:
            g["cost_usd"] += r["cost_usd"]
    rows = [{**dict(zip(by, key)), **vals} for key, vals in groups.items()]
    rows.sort(key=lambda row: -row["cost_usd"])
    return rows


def format_summary(rows: List[dict], by: Iterable[str] = ("model",)) -> str:
    by = tuple(by)
    head = list(by) + ["calls", "input", "cache_read", "cache_write", "output", "cost_usd"]
    table = []
    for r in rows:
        cost = fmt_usd(r["cost_usd"])
        if r["unpriced_calls"]:
            cost += f" (+{r['unpriced_calls']} unpriced)"
        table.append([str(r.get(k)) for k in by] + [
            str(r["calls"]), _fmt_tokens(r["input_tokens"]), _fmt_tokens(r["cache_read_tokens"]),
            _fmt_tokens(r["cache_write_tokens"] + r["cache_write_1h_tokens"]),
            _fmt_tokens(r["output_tokens"]), cost,
        ])
    total_cost = sum(r["cost_usd"] for r in rows)
    total_calls = sum(r["calls"] for r in rows)
    table.append(["TOTAL"] + [""] * (len(by) - 1) + [str(total_calls), "", "", "", "", fmt_usd(total_cost)])
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(head)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in table]
    return "\n".join(lines)


def load_records(paths: Iterable[str]) -> List[dict]:
    out = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------- global tracker
_global: Optional[UsageTracker] = None
_global_lock = threading.Lock()


def get_tracker() -> UsageTracker:
    """The process-wide default tracker (created on first use from env vars)."""
    global _global
    with _global_lock:
        if _global is None:
            _global = UsageTracker()
        return _global


def configure(**kwargs) -> UsageTracker:
    """Replace the default tracker, e.g. ``configure(run="sweep_v2", log_path="logs/vlm.jsonl", budget_usd=20)``."""
    global _global
    with _global_lock:
        _global = UsageTracker(**kwargs)
        return _global


def record(response: Any, **kwargs) -> Optional[CallRecord]:
    return get_tracker().record(response, **kwargs)


def record_tokens(model: str, input_tokens: int = 0, output_tokens: int = 0, **kwargs) -> CallRecord:
    return get_tracker().record_tokens(model, input_tokens, output_tokens, **kwargs)


def tag(name: str):
    return get_tracker().tag(name)
