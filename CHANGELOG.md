# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-09

Adds the **stochat** app and richer cost tooling.

### Added

- **stochat** (`apps/stochat`) — an interactive terminal chat that optimizes
  every prompt before sending, working with any OpenAI-compatible coding agent
  (Claude, ChatGPT, Groq, OpenRouter, Together, or a local model via Ollama):
  - Provider presets selectable with `--provider` / `STOCHAT_PROVIDER`;
    `stochat providers` lists them and shows which keys are set.
  - Streams replies; mention a file/dir to attach it; attached files are
    **pinned** so trimming never drops the content you're asking about;
    whitespace compression preserves code indentation; up-arrow history and
    Tab path-completion; `/budget`, `/model`, `/tokens`, `/clear` commands.
- **Cost**: current-generation model pricing (Claude Opus 4.x, Sonnet 5/4.x,
  Haiku 4.5, Fable 5, plus existing OpenAI/Google/Claude 3 entries) and a
  `sto optimize --show-savings` flag that prints the estimated per-request cost
  saved for the chosen model.

## [0.2.0] - 2026-07-09

Adds a clean, zero-dependency terminal UI over the library: the `sto` command,
a reusable dashboard renderer, an interactive home menu, a live dashboard, and
a separate examples browser.

### Added

- **Command-line interface** (`smarttokenoptimizer.cli`, the `sto` command):
  - `sto count` / `sto optimize` / `sto cost` — operate on chat messages read
    as JSON from a file or stdin; `optimize` can write the result back with `-o`.
  - `sto dashboard` — print a boxed analytics view.
  - `render_dashboard()` — a reusable, pure-string renderer (tokens, cost,
    savings, cache hit-rate, success, and optional provider/credential health)
    that apps can embed anywhere.
  - `sto ui` — an interactive home menu; `sto examples` — a browser that lists
    the bundled example scripts (title + description) and runs the chosen one.
  - `sto dashboard --watch` — a live, auto-refreshing dashboard on the alternate
    screen (spinner + help footer), with `--interval` and `--iterations`.
  - Interactive foundation: ANSI style helpers honouring `NO_COLOR` /
    `FORCE_COLOR` / TTY auto-detection, and a numbered, stream-driven menu — all
    stdlib-only and fully testable without a terminal.

## [0.1.0] - 2026-07-09

First tagged release. The full roadmap is implemented across nine modules with a
strongly-typed, zero-required-dependency core, ~99% test coverage and runnable
examples.

### Added

- **Examples** — a runnable `examples/` directory with five offline scripts
  (token counting, conversation optimization, cost & cache, credentials &
  routing, and the full `OptimizingPipeline`), each exercised by the test
  suite and type-checked in CI.

- **Framework middleware** (`smarttokenoptimizer.middleware`):
  - `OptimizingPipeline` — a framework-agnostic pipeline that wires
    optimization, caching, provider routing, cost estimation and analytics
    around a single injected `call_fn` (no provider SDK dependency).
    `complete()` returns a `CompletionResult` with the response plus
    cached/provider/token/cost metadata.
  - `CallRequest` carries model, messages, api_key, provider and params to the
    caller, so wrapping the OpenAI/Anthropic SDKs, FastAPI, LiteLLM or LangChain
    is a thin adapter.

- **Provider routing** (`smarttokenoptimizer.routing`):
  - `Provider` — bundles a credential pool with routing metadata (served
    models, priority, weight, price hint) and a latency EWMA.
  - Routing policies: `PriorityPolicy`, `RoundRobinPolicy`, `CheapestPolicy`,
    `LowestLatencyPolicy`.
  - `Router` — ranks providers serving the requested model and hands out a
    credential from the first available one, failing over across providers;
    `dispatch()` records success/failure and latency automatically.

- **Credential management** (`smarttokenoptimizer.credentials`):
  - `Credential` — an API key plus selection metadata with strict secret
    hygiene (the key never appears in `repr` / `str` / logs; `masked_key`
    exposes a safe preview and a stable id is derived without revealing it).
  - Selection strategies (deterministic): `RoundRobinStrategy`,
    `PriorityStrategy`, `LeastUsedStrategy`, `WeightedRoundRobinStrategy`.
  - `CredentialPool` — thread-safe pool with rotation, rate-limit cooldown, a
    circuit breaker for failover, a `borrow()` context manager, and
    per-credential `health()` snapshots.

- **Prompt cache** (`smarttokenoptimizer.cache`):
  - `make_key` — stable, order-insensitive SHA-256 key from model, messages and
    request params.
  - `PromptCache` interface with request-keyed helpers and thread-safe
    `CacheStats` (hit/miss/hit-rate).
  - `MemoryCache` — in-process cache with optional LRU eviction and TTL.
  - `SQLiteCache` — durable, cross-process cache via stdlib `sqlite3` (zero
    dependencies), with JSON values, TTL and `purge_expired()`.
- **Cost & analytics** (`smarttokenoptimizer.cost`):
  - `ModelPricing` and a built-in per-model pricing table (USD per million
    tokens) with longest-prefix matching and runtime overrides
    (`register_pricing` / `clear_custom_pricing`).
  - `CostEstimator` — estimate request cost from token counts or directly from
    messages, returning a `CostEstimate` (input/output/total breakdown).
  - `UsageTracker` — thread-safe accumulator for tokens, cost, tokens/cost
    saved, cache hits and successes; `AnalyticsSnapshot` exposes totals plus
    cache-hit rate, success rate, average cost per request and savings ratio.

- **Prompt compression** (`smarttokenoptimizer.compression`):
  - `TextCompressor` interface and `WhitespaceCompressor` — normalise line
    endings, collapse space/tab runs, strip per-line and overall whitespace,
    and collapse long blank-line runs (all configurable, idempotent).
  - `CompositeCompressor` — chain compressors in sequence.
  - `CompressionStrategy` — adapt a compressor to the `BudgetStrategy`
    interface so compression composes with dedup/window/drop-oldest. Compresses
    message content in place (never drops), reporting the changed count.
- **Context optimization** (`smarttokenoptimizer.context`):
  - `DeduplicateStrategy` — remove exact-duplicate messages (by role+content or
    content only; keep first or last occurrence; optional protected roles).
  - `SlidingWindowStrategy` — retain protected messages plus the newest N
    non-protected turns (count-based, independent of the token budget).
  - `CompositeStrategy` (in `budgeting`) — chain strategies in sequence,
    threading messages through each and aggregating accounting; e.g.
    deduplicate → sliding window → drop-oldest.
  - All implement the `BudgetStrategy` interface, so they compose in
    `SmartTokenOptimizer`.
- **Token budgeting** (`smarttokenoptimizer.budgeting`):
  - `SmartTokenOptimizer(max_tokens=...)` with `optimize(messages)` to fit a
    conversation into a token budget, and `optimize_detailed()` returning an
    `OptimizationResult` (tokens saved, compression ratio, messages dropped,
    `within_budget`).
  - Pluggable `BudgetStrategy` interface and default `DropOldestStrategy`
    (preserves protected roles such as `system`, supports `keep_last`, drops
    oldest turns first). Input is never mutated.
  - Re-exported from the top-level package.
- Project scaffolding: packaging (`pyproject.toml`), tooling configuration
  (Ruff, Black, mypy, pytest), MIT license, contribution guide and CI.
- **Token counting** (`smarttokenoptimizer.tokenization`):
  - `TokenCounter` base class with chat-message accounting (per-message and
    per-name overhead, reply priming) matching OpenAI-compatible chat formats.
  - `HeuristicTokenCounter` — fast, deterministic, **zero-dependency** estimator.
  - `TiktokenCounter` — exact counts via the optional `[tiktoken]` extra, with a
    clear `BackendUnavailableError` when the dependency is missing.
  - `get_counter()` factory and model→encoding registry that prefer an exact
    tokenizer and fall back to the heuristic automatically.
  - Re-exported from the top-level package for convenience.

[Unreleased]: https://github.com/NSGsanji/ai-cred-management/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NSGsanji/ai-cred-management/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NSGsanji/ai-cred-management/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NSGsanji/ai-cred-management/releases/tag/v0.1.0
