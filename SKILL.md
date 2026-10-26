---
name: vlm-usage-monitor
description: Track the token usage and USD cost of every VLM/LLM API call in an experiment (Claude/Anthropic, OpenAI, Gemini, and local models served by vLLM or other OpenAI-compatible servers), with per-call printouts, JSONL logs, per-run summaries and spend budgets. Use when writing, modifying or running code that calls a VLM/LLM API, when adding a VLM planner, judge or labeller to an experiment, or when the user asks how much an experiment, run or model call costs, about token usage, API spend or credits, or wants a budget cap.
---

# VLM Usage Monitor

A small stdlib-only Python package (`vlm_usage_monitor`, in this skill's folder)
that records tokens and cost for each model call. Prices come from LiteLLM's
public price table (auto-refreshed weekly, bundled snapshot when offline), with
cache-read/write, batch, flex/priority and long-context tiers handled.

## 1. Make sure it is installed

```bash
python -c "import vlm_usage_monitor" 2>/dev/null || pip install -e <this skill's folder>
```

`<this skill's folder>` is the directory that contains this SKILL.md (resolve
symlinks, e.g. `readlink -f ~/.claude/skills/vlm-usage-monitor`). Install it into
the same environment the experiment runs in (conda env, venv, container).

## 2. Instrument the code

Pick the pattern that matches how the code calls the model. Change as little of
the user's code as possible; `track()` patches the client in place and returns it.

**SDK clients (preferred).** Wrap the client right where it is constructed:

```python
from vlm_usage_monitor import track
client = track(anthropic.Anthropic())                         # Claude
client = track(openai.OpenAI())                               # OpenAI
client = track(openai.OpenAI(base_url="http://node:8000/v1", api_key="EMPTY"))  # vLLM / SGLang / Ollama
client = track(openai.OpenAI(...), tag="planner")             # label calls by role
```

This covers sync and async clients, `chat.completions`, `responses`,
`completions`, `embeddings`, `messages` (and `beta.messages`), `parse`, and
streaming. For streamed OpenAI Chat Completions, usage only arrives if the call
passes `stream_options={"include_usage": True}`; pass `track(..., inject_stream_usage=True)`
to add it automatically (check the code tolerates a final chunk with empty `choices`).

**Raw HTTP / JSON responses** (`requests.post(...).json()`), Gemini responses,
Anthropic batch results, vLLM offline `LLM.generate()` outputs:

```python
from vlm_usage_monitor import record
data = requests.post(f"{url}/chat/completions", json=payload).json()
record(data, tag="planner")
record(batch_result.result.message, service_tier="batch")    # batch pricing
record(llm.generate(prompts, params), model="Qwen/Qwen2.5-VL-7B-Instruct")
```

**Only token counts available** (HF `generate`, custom servers):

```python
from vlm_usage_monitor import record_tokens
record_tokens("Qwen/Qwen2.5-VL-7B-Instruct", input_tokens=n_in, output_tokens=n_out, latency_s=dt)
```

**Attribute calls to a stage** with a context manager:

```python
from vlm_usage_monitor import tag
with tag("subgoal_labelling"):
    ...
```

## 3. Configure the run

Set these in the launch script or sbatch file rather than in code:

| Env var | Meaning |
|---|---|
| `VLM_USAGE_LOG` | JSONL file every call is appended to. Put it in the experiment's output folder. Safe across processes/jobs. |
| `VLM_USAGE_RUN` | Run name stored on each record (default: script name). |
| `VLM_USAGE_BUDGET` | Warn once when tracked spend for this process passes this many USD. |
| `VLM_USAGE_VERBOSE=0` | Silence the one-line-per-call printout (summary at exit still prints). |
| `VLM_USAGE_SUMMARY_AT_EXIT=0` | Skip the per-model summary at process exit. |
| `VLM_USAGE_AUTO_UPDATE=0` | Never download prices (use cache/snapshot); for nodes without internet. |
| `VLM_USAGE_PRICES` | JSON file of custom prices, `{"model": {"input": 0.2, "output": 0.6, "cache_read": 0.02}}` in USD per 1M tokens. |

Or in code: `configure(run=..., log_path=..., budget_usd=20, hard_budget=True)`;
`hard_budget=True` raises `BudgetExceededError` to stop a runaway job.

Local models have no price, so they are logged as tokens only (`cost_usd: null`).
If the user wants a dollar figure for them, `register_price(model, input=..., output=...)`
in USD per 1M tokens (e.g. a hosted provider's price for the same model, or
GPU $/hour divided by measured tokens/hour).

## 4. Report spend

```bash
vlm-usage summary path/to/vlm_usage.jsonl --by model          # also: tag, run, provider, host
vlm-usage summary 'experiments/*/vlm_usage.jsonl' --by run,model --since 2026-10-01
vlm-usage price claude-opus-5-5 --input 20000 --output 1000    # rates + example cost
vlm-usage vllm-metrics http://node:8000                        # server-side token totals (all clients)
vlm-usage update-prices                                        # force a price refresh
```

In Python: `get_tracker().summary(by=("model", "tag"))`, `get_tracker().total_cost`.
When reporting to the user, give total USD, calls, and the input/cache/output
token split per model, and say which calls were unpriced.

## Accuracy notes

- Costs are estimates from public list prices. The provider's billing console
  (Anthropic Console usage/cost, OpenAI usage dashboard) is the ground truth;
  negotiated, credit-based, or cloud-marketplace (Bedrock/Vertex) prices differ.
- Anthropic `input_tokens` excludes cached tokens; OpenAI `prompt_tokens`/`input_tokens`
  include them. The package normalizes both to `input_tokens` = uncached input.
- Reasoning/thinking tokens are billed as output and included in `output_tokens`.
- Anthropic fast mode is priced as 2x standard; web search as $10 per 1,000 searches.
  Other server-side tool fees, image-generation and audio are not priced.
- `data/price_corrections.json` holds fixes where LiteLLM disagrees with the
  provider's own price page; add an entry there if you find another.
- Calls made through `with_raw_response` / `with_streaming_response` are not
  patched; record those with `record(response.parse())`.
