# stochat

An interactive terminal chat that **optimizes every prompt with
[SmartTokenOptimizer](../../README.md) before sending it** — trimming redundant
history, duplicates and whitespace so you spend fewer tokens. It talks to **any
OpenAI-compatible coding agent**: Claude, ChatGPT, Groq, OpenRouter, Together,
or a **local model via Ollama** (free, offline).

```
you› review ~/proj/app.py and find bugs
     (attached: ~/proj/app.py — pinned)
[sto] 1712→1712 tok (saved 0; budget 8000)
ai› 1. divide() doesn't guard against zero …    (streams live)
```

## What it does

- **Optimizes each turn** — compress whitespace (safely; never mangles code
  indentation), drop duplicate messages, and drop the oldest turns once the
  conversation exceeds your token budget.
- **Attach files by mention** — write a path in your message and it's read in
  automatically and **pinned** (never dropped by trimming). Directories attach
  as a file listing.
- **Streams replies**, with `↑` history and `Tab` path-completion.
- **Any provider** through one OpenAI-compatible interface.

> Reality check: stochat reduces *redundancy and old history*. It will **not**
> delete the file you're actively asking about — that content is the point. If
> your essential content is larger than the budget, it keeps it and warns you.

## Install

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # the smarttokenoptimizer core
pip install openai          # the client stochat uses to talk to providers
```

Run it:

```bash
python -m stochat providers          # list providers + which keys are set
python -m stochat --provider ollama  # start chatting
```

(Run from the `apps/` directory, or `cd apps && python -m stochat ...`.)

## Set up a coding agent (pick one)

stochat reads each provider's API key from an environment variable. Set the key,
then pass `--provider`.

### 🖥️ Ollama — local, free, offline (recommended to start)

```bash
# install + pull a model (once)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b          # or: qwen2.5-coder for coding, faster

python -m stochat --provider ollama --model llama3.1:8b
```
No API key or account needed. First reply is slow (model loads into VRAM), then fast.

### 🤖 OpenAI (ChatGPT models)

```bash
export OPENAI_API_KEY="sk-..."           # platform.openai.com/api-keys
python -m stochat --provider openai --model gpt-4o
```

### 🧠 Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # console.anthropic.com
python -m stochat --provider anthropic --model claude-opus-4-8
```
> ⚠️ This uses Anthropic's official OpenAI-compatible endpoint. **Proxies that
> restrict access to the official Claude Code client (e.g. freemodel) will
> reject stochat** — that's their Terms of Service, not a bug. Use a direct
> Anthropic key, or route Claude through OpenRouter instead.

### ⚡ Groq — fast, generous free tier

```bash
export GROQ_API_KEY="gsk_..."            # console.groq.com/keys
python -m stochat --provider groq --model llama-3.3-70b-versatile
```

### 🌐 OpenRouter — Claude, GPT, and more via one key

```bash
export OPENROUTER_API_KEY="sk-or-..."    # openrouter.ai/keys
python -m stochat --provider openrouter --model anthropic/claude-3.5-sonnet
# or:  --model openai/gpt-4o
```

### 🔗 Together AI

```bash
export TOGETHER_API_KEY="..."            # api.together.ai
python -m stochat --provider together
```

Make a provider the default without `--provider` every time:
```bash
export STOCHAT_PROVIDER=groq
```

## Chat commands

| Command | Does |
| --- | --- |
| `mention a path` | attach that file (pinned) or directory listing |
| `/budget N` | set the token budget — **lower = trims more** |
| `/model M` | switch model mid-chat |
| `/tokens` | show current conversation size |
| `/clear` | forget the conversation |
| `/exit` (or Ctrl-D) | quit |

## Flags

```
--provider NAME     ollama | openai | groq | openrouter | anthropic | together
--model NAME        override the provider's default model
--budget N          prompt token budget (default 8000)
--keep-last N       recent turns never dropped (default 2; lower = trims more)
--reply-tokens N    max reply length (default 1000)
--count-model NAME  real model id used only for token counting
```

## How to actually reduce tokens

- **Long conversations** — as history grows past `--budget`, old turns are
  dropped. Lower the budget (`/budget 2000`) or `--keep-last 1` to trim harder.
- **Only attach what's needed** — reference a single function, not a whole file,
  when you can. A 2000-token file you need read in full is a floor stochat won't
  (and shouldn't) cut.
- On a **local/free** provider the win is speed + never overflowing context;
  on a **paid** provider it's also money.
