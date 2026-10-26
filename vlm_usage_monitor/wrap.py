"""``track(client)``: patch an OpenAI or Anthropic client so every call is recorded.

The client is patched in place and returned, so existing code keeps working:

    client = track(anthropic.Anthropic())
    client = track(openai.OpenAI())                                  # OpenAI
    client = track(openai.OpenAI(base_url="http://node:8000/v1", api_key="x"))  # vLLM

Covered: sync and async clients; ``create``/``parse`` on Chat Completions,
Responses, Completions, Embeddings and Messages (incl. ``beta``); streaming via
``stream=True`` and Anthropic's ``messages.stream()`` helper. For OpenAI Chat
Completions streams, usage is only sent when you pass
``stream_options={"include_usage": True}`` (``inject_stream_usage=True`` adds it).
"""

from __future__ import annotations

import functools
import inspect
import sys
import time
from typing import Any, Callable, Optional

from .tracker import BudgetExceededError, UsageTracker, get_tracker
from .usage import to_dict

_OPENAI_METHODS = (
    "chat.completions.create", "chat.completions.parse", "beta.chat.completions.parse",
    "responses.create", "responses.parse", "completions.create", "embeddings.create",
)
_ANTHROPIC_METHODS = (
    "messages.create", "messages.parse", "beta.messages.create", "beta.messages.parse",
)
_ANTHROPIC_STREAM_HELPERS = ("messages.stream", "beta.messages.stream")
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[vlm-usage] {msg}", file=sys.stderr)


def _resolve(root: Any, path: str):
    *parents, name = path.split(".")
    obj = root
    for p in parents:
        try:
            obj = getattr(obj, p)
        except Exception:
            return None, name
    return obj, name


class _UsageCollector:
    """Accumulates usage from streamed events of any supported API."""

    def __init__(self):
        self.final = None
        self.anthropic_msg = None
        self.anthropic_usage: dict = {}

    def feed(self, ev: Any) -> None:
        t = getattr(ev, "type", None)
        if t == "message_start":
            self.anthropic_msg = to_dict(ev.message)
            self.anthropic_usage = dict(self.anthropic_msg.get("usage") or {})
        elif t == "message_delta":
            u = to_dict(getattr(ev, "usage", None)) or {}
            self.anthropic_usage.update({k: v for k, v in u.items() if v is not None})
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            self.final = ev.response
        elif t is None and getattr(ev, "usage", None) is not None:
            self.final = ev  # Chat Completions chunk carrying usage (last chunk)

    def result(self):
        if self.anthropic_msg is not None:
            return {**self.anthropic_msg, "type": "message", "usage": self.anthropic_usage}
        return self.final


class _StreamProxy:
    """Wraps an SDK Stream/AsyncStream; records usage when the stream ends."""

    def __init__(self, stream: Any, finish: Callable[[Any], None]):
        self._stream = stream
        self._finish = finish
        self._collector = _UsageCollector()
        self._it = None
        self._done = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _close(self) -> None:
        if not self._done:
            self._done = True
            result = self._collector.result()
            if result is not None:
                self._finish(result)

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._stream)
        try:
            ev = next(self._it)
        except StopIteration:
            self._close()
            raise
        self._collector.feed(ev)
        return ev

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._stream.__aiter__()
        try:
            ev = await self._it.__anext__()
        except StopAsyncIteration:
            self._close()
            raise
        self._collector.feed(ev)
        return ev

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._stream.__exit__(*exc)
        finally:
            self._close()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            return await self._stream.__aexit__(*exc)
        finally:
            self._close()

    def close(self):
        try:
            return self._stream.close()
        finally:
            self._close()


class _MessageStreamManagerProxy:
    """Wraps Anthropic's ``messages.stream()`` manager (sync and async)."""

    def __init__(self, manager: Any, finish: Callable[[Any], None]):
        self._manager = manager
        self._finish = finish
        self._stream = None

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def _record(self) -> None:
        try:
            snapshot = self._stream.current_message_snapshot
        except Exception:
            return
        self._finish(snapshot)

    def __enter__(self):
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, *exc):
        try:
            return self._manager.__exit__(*exc)
        finally:
            self._record()

    async def __aenter__(self):
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, *exc):
        try:
            return await self._manager.__aexit__(*exc)
        finally:
            self._record()


def track(
    client: Any,
    tracker: Optional[UsageTracker] = None,
    tag: Optional[str] = None,
    model: Optional[str] = None,
    inject_stream_usage: bool = False,
) -> Any:
    """Patch ``client`` in place so each call is recorded; returns the client.

    Args:
        tracker: where records go (default: the process-wide tracker).
        tag: label stored with every call from this client (e.g. "planner").
        model: override the reported model name (e.g. when vLLM serves a local
            path and you want a readable name or a registered price).
        inject_stream_usage: add ``stream_options={"include_usage": True}`` to
            streamed Chat Completions calls so their usage can be recorded.
    """
    if getattr(client, "_vlm_usage_tracked", False):
        return client
    module = type(client).__module__.split(".")[0]
    if module == "anthropic":
        provider, methods = "anthropic", _ANTHROPIC_METHODS
    elif module == "openai":
        base_url = str(getattr(client, "base_url", "") or "")
        provider = "openai" if ("api.openai.com" in base_url or "azure" in base_url) else "openai_compatible"
        methods = _OPENAI_METHODS
    else:
        raise TypeError(f"track() supports openai and anthropic clients, got {type(client)!r}")

    def finish(result: Any, latency_s: float) -> None:
        trk = tracker or get_tracker()
        try:
            trk.record(result, model=model, provider=provider, tag=tag, latency_s=latency_s)
        except BudgetExceededError:
            raise
        except Exception as exc:  # never break the caller's experiment
            _warn_once(f"record:{type(exc).__name__}", f"could not record usage: {exc!r}")

    def make_wrapper(orig: Callable, path: str) -> Callable:
        chat_stream = path.startswith(("chat.completions", "completions"))

        def before(kwargs: dict) -> None:
            if kwargs.get("stream") and chat_stream and provider != "anthropic":
                if inject_stream_usage and "stream_options" not in kwargs:
                    kwargs["stream_options"] = {"include_usage": True}
                elif not (kwargs.get("stream_options") or {}).get("include_usage"):
                    _warn_once("stream_usage", "streamed chat completion without "
                               "stream_options={'include_usage': True}: usage will not be recorded")

        def after(resp: Any, t0: float, kwargs: dict) -> Any:
            if kwargs.get("stream"):
                return _StreamProxy(resp, lambda r: finish(r, time.perf_counter() - t0))
            finish(resp, time.perf_counter() - t0)
            return resp

        # Async SDK methods are not always detectable with iscoroutinefunction
        # (they can be wrapped by decorators), so check the returned value.
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            before(kwargs)
            t0 = time.perf_counter()
            resp = orig(*args, **kwargs)
            if inspect.isawaitable(resp):
                async def finish_async():
                    return after(await resp, t0, kwargs)
                return finish_async()
            return after(resp, t0, kwargs)
        return wrapper

    def make_stream_helper(orig: Callable) -> Callable:
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            return _MessageStreamManagerProxy(
                orig(*args, **kwargs), lambda r: finish(r, time.perf_counter() - t0)
            )
        return wrapper

    patched = 0
    for path in methods:
        owner, name = _resolve(client, path)
        orig = getattr(owner, name, None) if owner is not None else None
        if callable(orig):
            setattr(owner, name, make_wrapper(orig, path))
            patched += 1
    if provider == "anthropic":
        for path in _ANTHROPIC_STREAM_HELPERS:
            owner, name = _resolve(client, path)
            orig = getattr(owner, name, None) if owner is not None else None
            if callable(orig):
                setattr(owner, name, make_stream_helper(orig))
                patched += 1
    if patched == 0:
        raise RuntimeError(f"track(): found no methods to patch on {type(client)!r}")
    # client.with_options(timeout=...) / client.copy() build a new client; track it too.
    for name in ("copy", "with_options"):
        orig = getattr(client, name, None)
        if callable(orig):
            setattr(client, name, functools.wraps(orig)(
                lambda *a, _orig=orig, **kw: track(
                    _orig(*a, **kw), tracker=tracker, tag=tag, model=model,
                    inject_stream_usage=inject_stream_usage,
                )
            ))
    client._vlm_usage_tracked = True
    return client
