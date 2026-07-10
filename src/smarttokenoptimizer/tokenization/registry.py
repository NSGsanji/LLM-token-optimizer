"""Model-to-tokenizer mapping and counter factory.

This module knows how to turn a *model name* (e.g. ``"gpt-4o"`` or
``"claude-3-5-sonnet"``) into an appropriate :class:`TokenCounter`. It prefers
an exact tokenizer when one is available and always offers a dependency-free
fallback so callers never hard-fail purely because an optional backend is
missing.
"""

from __future__ import annotations

from .base import TokenCounter
from .heuristic import HeuristicTokenCounter
from .tiktoken_backend import TiktokenCounter, is_available
from .types import DEFAULT_OVERHEAD, MessageOverhead

# Maps a tiktoken encoding name to the model-name prefixes that use it. Ordered
# from most specific to least specific; the first matching prefix wins.
_ENCODING_BY_PREFIX: tuple[tuple[str, str], ...] = (
    # o-series and GPT-4o family use o200k_base.
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4", "o200k_base"),
    ("chatgpt-4o", "o200k_base"),
    # GPT-4 / GPT-3.5 family and text-embedding-3 use cl100k_base.
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("text-embedding-3", "cl100k_base"),
    ("text-embedding-ada-002", "cl100k_base"),
)

# The encoding assumed for unknown OpenAI-compatible models. cl100k_base is the
# most widely-compatible choice for chat models.
_DEFAULT_ENCODING = "cl100k_base"


def encoding_name_for_model(model: str) -> str:
    """Return the tiktoken encoding name best matching ``model``.

    Matching is done by case-insensitive prefix. Unknown models fall back to
    :data:`_DEFAULT_ENCODING` (``cl100k_base``), a safe default for
    OpenAI-compatible chat models.

    Args:
        model: A model identifier such as ``"gpt-4o-mini"``.

    Returns:
        A tiktoken encoding name.
    """
    normalized = model.strip().lower()
    for prefix, encoding in _ENCODING_BY_PREFIX:
        if normalized.startswith(prefix):
            return encoding
    return _DEFAULT_ENCODING


def get_counter(
    model: str | None = None,
    *,
    prefer_exact: bool = True,
    overhead: MessageOverhead = DEFAULT_OVERHEAD,
) -> TokenCounter:
    """Return the best available token counter for ``model``.

    The factory prefers an exact tokenizer when possible and falls back to the
    dependency-free :class:`HeuristicTokenCounter` otherwise, so it never raises
    merely because an optional backend is absent.

    Args:
        model: A model identifier. If ``None``, a general-purpose counter is
            returned (exact counter uses the default encoding).
        prefer_exact: When ``True`` (default) and the tiktoken backend is
            available, an exact counter is returned. When ``False``, the fast
            heuristic counter is always returned.
        overhead: Per-message overhead accounting for chat formats.

    Returns:
        A ready-to-use :class:`TokenCounter`.
    """
    if prefer_exact and is_available():
        encoding = (
            encoding_name_for_model(model) if model is not None else _DEFAULT_ENCODING
        )
        return TiktokenCounter(encoding, overhead=overhead)
    return HeuristicTokenCounter(overhead=overhead)
