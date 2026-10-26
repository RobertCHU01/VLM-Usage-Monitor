# VLM Usage Monitor

See the token usage and USD cost of every VLM/LLM call in your experiments:
Claude (Anthropic), OpenAI, Gemini, and local models served through vLLM or any
OpenAI-compatible server. It is a Python package and an agent skill in one folder,
so your coding agent can add cost tracking to new experiments by itself.

```
[vlm-usage] [planner] claude-sonnet-5-5 in=3,200 cache_read=12,000 cache_write=0 out=410 $0.01170 | run total $0.0117 over 1 calls
[vlm-usage] [local] Qwen/Qwen2.5-VL-7B-Instruct in=900 out=40 unpriced 0.80s 50.0 tok/s | run total $0.0117 over 2 calls
```

```
$ vlm-usage summary logs/vlm_usage.jsonl --by model,tag
model                        tag      calls  input  cache_read  cache_write  output  cost_usd
---------------------------  -------  -----  -----  ----------  -----------  ------  --------------------
claude-sonnet-5-5            planner  1      3,200  12,000      0            410     0.0117
Qwen/Qwen2.5-VL-7B-Instruct  local    1      900    0           0            40      0.0000 (+1 unpriced)
TOTAL                                 2                                              0.0117
```

## Install

```bash
git clone https://github.com/RobertCHU01/VLM-Usage-Monitor.git && cd VLM-Usage-Monitor
bash install.sh      # pip install -e . into the active env + register the skill
```

`install.sh` symlinks this folder into `~/.claude/skills/vlm-usage-monitor`
(Claude Code), and into `~/.codex/skills` / `~/.agents/skills` if those exist.
For any other agent, point it at `SKILL.md`. No dependencies beyond the standard
library; it uses whichever `openai`/`anthropic` SDK version you already have
(tested with openai 1.109 and 3.38, anthropic 0.125 and 2.3).

## Use

```python
from vlm_usage_monitor import track, record, record_tokens, configure, tag

configure(run="shelf_sweep_v3", log_path="outputs/vlm_usage.jsonl", budget_usd=20)

client = track(anthropic.Anthropic())                     # every call now logged
client = track(openai.OpenAI())
client = track(openai.OpenAI(base_url="http://node:8000/v1", api_key="EMPTY"), tag="local")

record(requests.post(url, json=payload).json())           # raw HTTP responses
record_tokens("my-model", input_tokens=812, output_tokens=64)   # just counts
with tag("judge"): ...                                    # attribute calls to a stage
```

Or configure with environment variables in your launch/sbatch script, with no
code change beyond `track(...)`:

```bash
export VLM_USAGE_LOG=$OUT_DIR/vlm_usage.jsonl VLM_USAGE_RUN=$SLURM_JOB_NAME VLM_USAGE_BUDGET=20
```

`SKILL.md` has the full list of patterns, env vars and CLI commands.

## How it works

- **Usage** is read from each response's `usage` block and normalized: Anthropic
  reports uncached input, cache reads and cache writes (5-minute / 1-hour)
  separately, while OpenAI includes cached tokens in its input count. Reasoning
  tokens count as output. Streams are handled by watching the events.
- **Prices** come from [LiteLLM's price table](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
  (MIT), maintained by the community and covering most providers, including
  batch, flex/priority and long-context tiers. It is downloaded to
  `~/.cache/vlm_usage_monitor` and refreshed weekly; a snapshot is bundled for
  offline nodes. `data/price_corrections.json` fixes entries that disagree with
  the provider's own price page. Your own prices (`register_price`,
  `$VLM_USAGE_PRICES`) override everything.
- **Local models** have no price, so they are logged as tokens, latency and
  tokens/s. `vlm-usage vllm-metrics http://node:8000` reads a vLLM server's
  Prometheus counters for totals across all its clients.

## Why not an online tool?

Hosted tools exist and are worth knowing about:

- **Provider dashboards**: Anthropic Console (Usage, Cost), the Anthropic Admin
  API usage and cost reports, and the OpenAI usage dashboard / Costs API. These
  are the billing ground truth, but only per API key or workspace per day, not
  per experiment or per call.
- **LiteLLM** (`litellm.completion_cost`, or its proxy with spend tracking): the
  source of our price table; the proxy needs a server you run, and all calls must
  go through it.
- **Langfuse, Helicone** and similar LLM-observability platforms: dashboards and
  traces, but they need an account or a self-hosted server plus an SDK
  integration or proxy, and they don't track local vLLM runs on the cluster as easily.

This package is the lightweight local version: a one-line change per client,
logs that live next to each experiment's outputs, and no account or network
service required. Use a provider dashboard to check its totals now and then.

## Tests

```bash
pip install -e '.[test]' && VLM_USAGE_AUTO_UPDATE=0 python -m pytest tests -q
```

The tests run real `openai`/`anthropic` clients against a mock HTTP server (no
API calls, no cost) and check costs against hand-computed values.

To check the numbers against your own Anthropic Console, `examples/live_dashboard_check.py`
spends about $0.02 on Claude Haiku 5.5 (cached and uncached calls) and prints the
token totals the Console's Usage page should show.

## License

MIT, see `LICENSE`. The bundled price snapshot comes from LiteLLM (MIT); see
`THIRD_PARTY_NOTICES.md`.
