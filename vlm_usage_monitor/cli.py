"""``vlm-usage`` command line: summarize logs, look up prices, read vLLM metrics."""

from __future__ import annotations

import argparse
import glob
import json
import sys

from . import pricing
from .tracker import format_summary, load_records, summarize
from .usage import Usage
from .vllm_metrics import vllm_server_tokens


def _summary(args) -> int:
    paths = [p for pattern in args.logs for p in (glob.glob(pattern) or [pattern])]
    records = load_records(paths)
    if args.run:
        records = [r for r in records if r.get("run") == args.run]
    if args.since:
        records = [r for r in records if r.get("ts", "") >= args.since]
    by = [b.strip() for b in args.by.split(",") if b.strip()]
    rows = summarize(records, by=by)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_summary(rows, by=by))
    return 0


def _price(args) -> int:
    info = pricing.describe(args.model)
    if info is None:
        print(f"No price found for {args.model!r} (local/unknown model; use register_price or $VLM_USAGE_PRICES).")
        return 1
    print(f"{args.model} -> {info.pop('matched')} ({info.pop('provider')}), USD per 1M tokens:")
    for k, v in info.items():
        print(f"  {k:55s} {v}")
    if args.input or args.output:
        usage = Usage(provider="?", model=args.model, input_tokens=args.input,
                      output_tokens=args.output, service_tier=args.tier)
        print(f"Cost of {args.input:,} input + {args.output:,} output tokens: "
              f"${pricing.compute_cost(usage).total_usd:.6f}")
    return 0


def _update(args) -> int:
    path = pricing.update_prices()
    print(f"Saved latest LiteLLM price table to {path}")
    return 0


def _vllm(args) -> int:
    print(json.dumps(vllm_server_tokens(args.url), indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vlm-usage", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="aggregate JSONL usage logs")
    s.add_argument("logs", nargs="+", help="JSONL log files or globs")
    s.add_argument("--by", default="model", help="comma list of: model,tag,run,provider,host")
    s.add_argument("--run", help="only this run name")
    s.add_argument("--since", help="only records at/after this ISO date, e.g. 2026-10-01")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_summary)

    p = sub.add_parser("price", help="show a model's prices")
    p.add_argument("model")
    p.add_argument("--input", type=int, default=0, help="example input tokens")
    p.add_argument("--output", type=int, default=0, help="example output tokens")
    p.add_argument("--tier", default=None, help="batch | flex | priority")
    p.set_defaults(fn=_price)

    u = sub.add_parser("update-prices", help="download the latest price table")
    u.set_defaults(fn=_update)

    v = sub.add_parser("vllm-metrics", help="token totals from a vLLM server's /metrics")
    v.add_argument("url", help="e.g. http://localhost:8000 or http://localhost:8000/v1")
    v.set_defaults(fn=_vllm)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
