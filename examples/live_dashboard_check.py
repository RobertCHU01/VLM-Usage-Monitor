"""Spend a small, known amount (default ~$0.02) on Claude so you can compare the
monitor's numbers with the Anthropic Console (Usage and Cost pages).

    python examples/live_dashboard_check.py                     # Haiku 5.5, ~$0.02
    python examples/live_dashboard_check.py --target-usd 0.05 --log logs/check.jsonl

Two phases, both with ~20K-token prompts (below Haiku 5.5's 100K long-prompt tier):
1. Uncached calls, each with a unique prompt: checks the input/output prices.
2. Calls sharing one cached system prompt: 1 cache write, then cache reads.

Needs ANTHROPIC_API_KEY. Run it on a day you don't otherwise use this model
with this key, so the Console totals for that model are this script alone.
"""

import argparse
import sys
import uuid

import anthropic

import vlm_usage_monitor as vum

# Anthropic's published Claude Haiku 5.5 prices, USD per 1M tokens (prompts <= 100K),
# typed in independently of the package's price table for the hand check.
HAIKU_55 = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write_5m": 0.125}


def filler(n_lines: int, seed: str) -> str:
    colors = ["red", "blue", "green", "yellow", "purple"]
    places = ["shelf 1", "shelf 2", "the drawer", "the table", "the bin"]
    return "\n".join(
        f"Log {seed}-{i}: the robot moved the {colors[i % 5]} cube from "
        f"{places[(i * 3) % 5]} to {places[(i * 7 + 1) % 5]} in {(i * 37) % 90 + 10} steps."
        for i in range(n_lines)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--target-usd", type=float, default=0.02)
    ap.add_argument("--prompt-tokens", type=int, default=20_000)
    ap.add_argument("--log", default="logs/live_dashboard_check.jsonl")
    args = ap.parse_args()

    tracker = vum.configure(run="live_dashboard_check", log_path=args.log,
                            budget_usd=2.5 * args.target_usd, hard_budget=True)
    raw = anthropic.Anthropic()
    client = vum.track(anthropic.Anthropic())

    # Size the filler text to ~prompt_tokens with the free token-counting endpoint.
    probe = filler(200, "probe")
    n = raw.messages.count_tokens(model=args.model, messages=[{"role": "user", "content": probe}]).input_tokens
    n_lines = int(200 * args.prompt_tokens / n)
    in_rate = vum.lookup_price(args.model)[1]["input_cost_per_token"]
    n_uncached = max(1, min(20, round(0.8 * args.target_usd / (args.prompt_tokens * in_rate))))
    print(f"{args.model}: {n_lines} lines per prompt (~{args.prompt_tokens:,} tokens), "
          f"{n_uncached} uncached calls + 4 cached calls\n")

    question = "In one short sentence: which color appears most often in the logs above?"
    kw = dict(model=args.model, max_tokens=300, output_config={"effort": "low"})
    run_id = uuid.uuid4().hex[:8]

    with vum.tag("uncached"):
        for i in range(n_uncached):
            text = filler(n_lines, f"{run_id}u{i}")  # unique per call
            client.messages.create(**kw, messages=[{"role": "user", "content": f"{text}\n\n{question}"}])

    shared = [{"type": "text", "text": filler(n_lines, f"{run_id}c"), "cache_control": {"type": "ephemeral"}}]
    with vum.tag("cached"):
        for q in ["Which color appears most often?", "Which place appears most often?",
                  "What is the largest step count?", "How many distinct colors are there?"]:
            client.messages.create(**kw, system=shared,
                                   messages=[{"role": "user", "content": q + " One short sentence."}])

    print()
    hand_total = 0.0
    for r in tracker.records:
        line = (f"{r.tag:9s} in={r.input_tokens:6,} cache_write={r.cache_write_tokens + r.cache_write_1h_tokens:6,} "
                f"cache_read={r.cache_read_tokens:6,} out={r.output_tokens:4,}  monitor ${r.cost_usd:.7f}")
        if args.model == "claude-haiku-5-5":
            hand = (r.input_tokens * HAIKU_55["input"] + r.output_tokens * HAIKU_55["output"]
                    + r.cache_read_tokens * HAIKU_55["cache_read"]
                    + r.cache_write_tokens * HAIKU_55["cache_write_5m"]) / 1e6
            hand_total += hand
            line += f"  hand ${hand:.7f}  {'OK' if abs(hand - r.cost_usd) < 1e-8 else 'MISMATCH'}"
        print(line)

    tot = {f: sum(getattr(r, f) for r in tracker.records)
           for f in ("input_tokens", "cache_write_tokens", "cache_write_1h_tokens", "cache_read_tokens", "output_tokens")}
    print(f"\nTotals to compare with Console -> Usage ({args.model}):")
    print(f"  uncached input  {tot['input_tokens']:,}")
    print(f"  cache write     {tot['cache_write_tokens'] + tot['cache_write_1h_tokens']:,}")
    print(f"  cache read      {tot['cache_read_tokens']:,}")
    print(f"  output          {tot['output_tokens']:,}")
    print(f"Monitor cost ${tracker.total_cost:.6f}" + (f" | hand ${hand_total:.6f}" if hand_total else ""))
    print("Console dates are in UTC; allow a few minutes for usage to show up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
