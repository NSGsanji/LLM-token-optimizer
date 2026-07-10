"""Provider presets for stochat.

Each preset maps a friendly name to the base URL, the environment variable that
holds the API key, and a sensible default model. stochat talks to any provider
through the OpenAI-compatible chat-completions API, so adding a new one is just
another row here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provider:
    """A chat provider stochat can talk to.

    Attributes:
        name: Friendly identifier used with ``--provider``.
        base_url: OpenAI-compatible base URL (``None`` = the SDK default,
            i.e. api.openai.com).
        key_env: Environment variable that holds the API key.
        default_model: Model used when ``--model`` is not given.
        count_model: A real model id used only for token counting/pricing
            (proxy aliases like ``opus[1m]`` aren't in the tokenizer tables).
        note: One-line hint shown in ``stochat providers``.
    """

    name: str
    base_url: str | None
    key_env: str
    default_model: str
    count_model: str
    note: str


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        "ollama",
        "http://localhost:11434/v1",
        "OLLAMA_API_KEY",
        "llama3.1:8b",
        "gpt-4o",
        "Local, free, offline. Run `ollama serve`; any key value works.",
    ),
    "openai": Provider(
        "openai",
        None,
        "OPENAI_API_KEY",
        "gpt-4o",
        "gpt-4o",
        "api.openai.com — set OPENAI_API_KEY (sk-...).",
    ),
    "groq": Provider(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
        "gpt-4o",
        "Fast, generous free tier — set GROQ_API_KEY (gsk_...).",
    ),
    "openrouter": Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "anthropic/claude-3.5-sonnet",
        "gpt-4o",
        "Many models incl. Claude/GPT — set OPENROUTER_API_KEY (sk-or-...).",
    ),
    "anthropic": Provider(
        "anthropic",
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "claude-opus-4-8",
        "claude-opus-4-8",
        "Anthropic's OpenAI-compatible endpoint — set ANTHROPIC_API_KEY. "
        "Note: proxies that restrict access to the official client won't work.",
    ),
    "together": Provider(
        "together",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "gpt-4o",
        "Together AI — set TOGETHER_API_KEY.",
    ),
}

DEFAULT_PROVIDER = os.environ.get("STOCHAT_PROVIDER", "ollama")


def resolve(name: str | None) -> Provider:
    """Return the provider preset for ``name`` (or the default)."""
    key = (name or DEFAULT_PROVIDER).lower()
    if key not in PROVIDERS:
        raise SystemExit(f"unknown provider {key!r}. Known: {', '.join(PROVIDERS)}")
    return PROVIDERS[key]
