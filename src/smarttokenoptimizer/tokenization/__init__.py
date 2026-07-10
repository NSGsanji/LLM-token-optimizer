"""Token counting for LLM prompts and conversations.

This subpackage provides a small, well-typed interface for counting tokens:

- :class:`HeuristicTokenCounter` — fast, approximate, zero dependencies.
- :class:`TiktokenCounter` — exact, backed by the optional ``tiktoken`` extra.
- :func:`get_counter` — factory that picks the best available counter for a
  model, with automatic fallback to the heuristic.

Example:
    >>> from smarttokenoptimizer.tokenization import get_counter
    >>> counter = get_counter("gpt-4o", prefer_exact=False)
    >>> counter.count_text("Hello, world!")
    4
"""

from __future__ import annotations

from .base import TokenCounter
from .errors import (
    BackendUnavailableError,
    TokenizationError,
    UnknownModelError,
)
from .heuristic import HeuristicTokenCounter
from .registry import encoding_name_for_model, get_counter
from .tiktoken_backend import TiktokenCounter, is_available
from .types import DEFAULT_OVERHEAD, Message, MessageOverhead

__all__ = [
    "DEFAULT_OVERHEAD",
    "BackendUnavailableError",
    "HeuristicTokenCounter",
    "Message",
    "MessageOverhead",
    "TiktokenCounter",
    "TokenCounter",
    "TokenizationError",
    "UnknownModelError",
    "encoding_name_for_model",
    "get_counter",
    "is_available",
]
