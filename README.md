# LLM-token-optimizer

A zero-dependency Python toolkit that helps you stop burning money on LLM tokens. It counts tokens, trims conversations to fit a budget, estimates and tracks cost, caches repeat responses, rotates API keys, and routes requests to the cheapest/fastest provider — all in one small library you can drop into any app.

> The importable Python package is `smarttokenoptimizer` (`pip install -e .`, `from smarttokenoptimizer import ...`) — same project, that's just its module name.

If you've ever watched a chat app blow through its context window, or paid for the same prompt twice, this is the thing that fixes that.

## Why this exists

Talking to LLMs gets expensive and messy fast once your app is more than a toy:

- Conversations grow past the model's context window and you have to trim them — badly, usually by just chopping off the oldest messages and hoping nothing important was in there.
- You keep re-sending near-identical prompts and paying for them every single time.
- You're juggling multiple API keys across providers and have no clean way to rotate them or fail over when one gets rate-limited.
- You have no idea what you're actually spending until the bill shows up.

LLM-token-optimizer wraps all of that into small, composable pieces you can use individually or wire together into one pipeline.

## What's inside

- **Tokenization** — count tokens for plain text or full chat conversations. Ships with a fast, dependency-free heuristic counter, and an exact `tiktoken`-backed counter if you install the extra.
- **Budgeting** — fit a conversation into a max-token budget using pluggable strategies (drop oldest turns, compress, dedupe — or compose your own pipeline). The system prompt is always protected.
- **Compression** — safely strip redundant whitespace from messages without mangling code indentation.
- **Context tools** — deduplicate repeated messages, slide a window over long histories.
- **Caching** — an in-memory or SQLite-backed cache so identical requests don't hit the API twice.
- **Cost estimation & analytics** — know the dollar cost of a request before *and* after you send it, plus running usage stats (spend, cache hit rate, success rate).
- **Credential management** — pool multiple API keys per provider, rotate them (round-robin, weighted, priority), and track rate-limit/health state per key.
- **Routing** — route a request across multiple providers by policy (cheapest, lowest latency), with automatic failover when one is rate-limited.
- **Middleware pipeline** — `OptimizingPipeline` ties all of the above around your own `call_fn`, so a single `.complete(messages)` call gets optimized, cached, routed, costed, and tracked.
- **CLI (`sto`)** — count, optimize, and cost conversations from the terminal, plus a live analytics dashboard. No dependencies required.
- **stochat** (in `apps/stochat`) — a full interactive terminal chat app built on top of the library, talking to Claude, ChatGPT, Groq, OpenRouter, Together, or a local Ollama model, with every prompt optimized before it's sent.

The core library has **zero required runtime dependencies**. Everything above works out of the box; `tiktoken` is only needed if you want exact (rather than estimated) token counts.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/NSGsanji/LLM-token-optimizer.git
cd LLM-token-optimizer
pip install -e .
```

Want exact token counts instead of the fast heuristic estimate?

```bash
pip install -e ".[tiktoken]"
```

## Quick start

### Count tokens

```python
from smarttokenoptimizer import Message, get_counter

counter = get_counter("gpt-4o")

text = "Hello, world!"
print(counter.count_text(text))

conversation: list[Message] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
print(counter.count_messages(conversation))
```

### Fit a conversation into a token budget

```python
from smarttokenoptimizer import SmartTokenOptimizer

optimizer = SmartTokenOptimizer(max_tokens=4000, model="gpt-4o")
result = optimizer.optimize_detailed(long_conversation)

print(f"{result.original_tokens} -> {result.optimized_tokens} tokens")
print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%} smaller)")
```

### The full pipeline: optimize + cache + route + cost, in one call

```python
from smarttokenoptimizer import SmartTokenOptimizer, Message
from smarttokenoptimizer.cache import MemoryCache
from smarttokenoptimizer.cost import CostEstimator, UsageTracker
from smarttokenoptimizer.credentials import CredentialPool
from smarttokenoptimizer.middleware import CallRequest, OptimizingPipeline
from smarttokenoptimizer.routing import Provider, Router

provider = Provider("openai", pool=CredentialPool(), models=["gpt-4o"])
provider.pool.add_key("sk-openai-...", provider="openai")

def call_fn(request: CallRequest) -> dict:
    # Call your real provider SDK here using request.api_key / request.messages.
    return {"text": "...", "usage": {"input": 100, "output": 50}}

pipeline = OptimizingPipeline(
    call_fn,
    model="gpt-4o",
    optimizer=SmartTokenOptimizer(max_tokens=4000, model="gpt-4o"),
    cache=MemoryCache(default_ttl=3600),
    router=Router([provider]),
    tracker=UsageTracker(),
    estimator=CostEstimator(),
    usage_extractor=lambda r: (r["usage"]["input"], r["usage"]["output"]),
)

messages: list[Message] = [{"role": "user", "content": "Explain vector databases."}]
response = pipeline.complete(messages)
print(response.cached, response.provider, response.cost)
```

More runnable, offline examples (no real API calls) live in [`examples/`](examples/):

| Script | Shows |
| ------ | ----- |
| [`01_token_counting.py`](examples/01_token_counting.py) | Counting tokens for strings and conversations |
| [`02_optimize_conversation.py`](examples/02_optimize_conversation.py) | Fitting a long conversation into a budget with a composed strategy pipeline |
| [`03_cost_and_cache.py`](examples/03_cost_and_cache.py) | Estimating cost, caching responses, and usage analytics |
| [`04_credentials_and_routing.py`](examples/04_credentials_and_routing.py) | Key rotation, rate-limit failover, and cheapest-provider routing |
| [`05_full_pipeline.py`](examples/05_full_pipeline.py) | The end-to-end `OptimizingPipeline` combining every module |

```bash
python examples/01_token_counting.py
```

## Using the CLI

The `sto` command wraps the library for quick terminal use — no code required. It reads chat messages as JSON (a list of `{"role", "content"}` objects, or an object with a `"messages"` key) from a file or stdin.

```bash
# Count tokens
sto count conversation.json

# Fit a conversation into a budget, keeping the last 2 turns intact
sto optimize conversation.json --max 2000 --keep-last 2 -o trimmed.json

# See how much a request would cost
sto cost conversation.json --model gpt-4o --output-tokens 300

# View the analytics dashboard (or a live, auto-refreshing one)
sto dashboard
sto dashboard --watch

# Interactive menu / example browser
sto ui
sto examples
```

Run `sto --help` or `sto <command> --help` for the full set of flags.

## stochat — chat from your terminal, optimized automatically

`apps/stochat` is a small interactive chat client built on top of this library. It optimizes every prompt (trims redundant history, dedupes, compresses whitespace without touching code indentation) before sending it, and works with Claude, ChatGPT, Groq, OpenRouter, Together, or a free local model via Ollama.

```bash
pip install -e .
pip install openai
cd apps
python -m stochat --provider ollama --model llama3.1:8b   # local, free, no API key
```

See [`apps/stochat/README.md`](apps/stochat/README.md) for setting up other providers, chat commands, and flags.

## Project layout

```
src/smarttokenoptimizer/
├── tokenization/   # token counting (heuristic + tiktoken)
├── budgeting/       # fit conversations into a token budget
├── compression/      # whitespace/text compression strategies
├── context/          # dedupe, sliding window
├── cache/             # in-memory / SQLite response caching
├── cost/               # cost estimation, pricing, usage analytics
├── credentials/        # API key pooling and rotation
├── routing/             # multi-provider routing and failover
├── middleware/           # OptimizingPipeline tying it all together
└── cli/                   # the `sto` command-line interface
apps/stochat/               # interactive terminal chat built on the library
examples/                    # runnable, offline usage examples
tests/                        # test suite
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Status

This is an actively developed **0.x alpha** — the public API may still shift between minor versions. Check [`CHANGELOG.md`](CHANGELOG.md) for what's changed.

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to get set up.

## License

MIT — see [`LICENSE`](LICENSE).
