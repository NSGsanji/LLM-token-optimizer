"""Prompt caching: reuse responses for identical requests.

Caching identical prompts avoids repeat API calls, cutting both latency and
cost. This subpackage provides a common :class:`PromptCache` interface with a
stable key derived from the model, messages and request parameters, plus two
zero-dependency backends:

- :class:`MemoryCache` — fast in-process cache with LRU eviction and TTL.
- :class:`SQLiteCache` — durable, cross-process cache backed by SQLite.

Both track hit/miss statistics (:class:`CacheStats`) that feed naturally into
:class:`~smarttokenoptimizer.cost.UsageTracker`.

Example:
    >>> from smarttokenoptimizer.cache import MemoryCache
    >>> cache = MemoryCache(max_size=1000, default_ttl=3600)
    >>> key = cache.key_for("gpt-4o", [{"role": "user", "content": "hi"}])
    >>> cache.set(key, {"reply": "hello"})
    >>> cache.get(key)
    {'reply': 'hello'}
"""

from __future__ import annotations

from .base import CacheStats, PromptCache
from .key import make_key
from .memory import MemoryCache
from .sqlite import SQLiteCache

__all__ = [
    "CacheStats",
    "MemoryCache",
    "PromptCache",
    "SQLiteCache",
    "make_key",
]
