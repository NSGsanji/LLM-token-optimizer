"""In-memory prompt cache with LRU eviction and TTL expiry."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from .base import PromptCache


class MemoryCache(PromptCache):
    """Fast, thread-safe in-process cache with optional LRU and TTL.

    Entries live in a bounded, least-recently-used ordering: reads and writes
    move an entry to the most-recently-used position, and once ``max_size`` is
    exceeded the least-recently-used entry is evicted. Entries may also carry a
    time-to-live after which they are treated as absent.

    Elapsed time is measured with a monotonic clock, so TTL behaviour is
    unaffected by system clock changes.

    Args:
        max_size: Maximum number of entries to retain. ``None`` (default) means
            unbounded — no LRU eviction. Must be positive when provided.
        default_ttl: Default time-to-live in seconds applied to entries stored
            without an explicit ``ttl``. ``None`` means no default expiry.

    Raises:
        ValueError: If ``max_size`` or ``default_ttl`` is non-positive.

    Example:
        >>> cache = MemoryCache(max_size=1000, default_ttl=3600)
        >>> cache.set("k", {"answer": 42})
        >>> cache.get("k")
        {'answer': 42}
    """

    def __init__(
        self,
        *,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ) -> None:
        super().__init__()
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when provided")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be positive when provided")
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        # key -> (value, expiry_monotonic_or_None)
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """See :meth:`PromptCache.get`."""
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._record_miss()
                return None
            value, expiry = entry
            if expiry is not None and expiry <= now:
                # Lazily evict the expired entry.
                del self._store[key]
                self._record_miss()
                return None
            self._store.move_to_end(key)
            self._record_hit()
            return value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """See :meth:`PromptCache.set`."""
        self._validate_set(value, ttl)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.monotonic() + effective_ttl if effective_ttl is not None else None
        with self._lock:
            self._store[key] = (value, expiry)
            self._store.move_to_end(key)
            if self._max_size is not None:
                while len(self._store) > self._max_size:
                    self._store.popitem(last=False)

    def delete(self, key: str) -> bool:
        """See :meth:`PromptCache.delete`."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries and reset hit/miss statistics."""
        with self._lock:
            self._store.clear()
        self._reset_stats()

    def __len__(self) -> int:
        """Return the number of live (non-lazily-expired) entries.

        Expired-but-not-yet-evicted entries are purged as part of this call.
        """
        now = time.monotonic()
        with self._lock:
            expired = [
                k
                for k, (_, expiry) in self._store.items()
                if expiry is not None and expiry <= now
            ]
            for k in expired:
                del self._store[k]
            return len(self._store)
