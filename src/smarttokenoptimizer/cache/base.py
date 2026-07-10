"""Prompt cache interface and shared statistics."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..tokenization.types import Message
from .key import make_key


@dataclass(frozen=True, slots=True)
class CacheStats:
    """A point-in-time view of cache hit/miss statistics.

    Attributes:
        hits: Number of lookups that returned a stored value.
        misses: Number of lookups that found nothing (or an expired entry).
    """

    hits: int = 0
    misses: int = 0

    @property
    def lookups(self) -> int:
        """Total number of lookups (``hits + misses``)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that hit, in ``[0, 1]`` (``0.0`` when none)."""
        return self.hits / self.lookups if self.lookups else 0.0


class _StatsMixin:
    """Thread-safe hit/miss counters shared by cache implementations."""

    def __init__(self) -> None:
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _record_hit(self) -> None:
        with self._stats_lock:
            self._hits += 1

    def _record_miss(self) -> None:
        with self._stats_lock:
            self._misses += 1

    def _reset_stats(self) -> None:
        with self._stats_lock:
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> CacheStats:
        """Return a snapshot of the cache's hit/miss statistics."""
        with self._stats_lock:
            return CacheStats(hits=self._hits, misses=self._misses)


class PromptCache(_StatsMixin, ABC):
    """Abstract base class for prompt/response caches.

    A cache maps a stable key (see :func:`make_key`) to an arbitrary stored
    value — typically an LLM response — with optional per-entry time-to-live.
    Concrete backends differ only in where the data lives (memory, SQLite,
    disk, …); they share this interface and hit/miss accounting.

    Implementations override :meth:`get`, :meth:`set`, :meth:`delete` and
    :meth:`clear`. Convenience helpers key an entry directly from a request.

    Note:
        ``None`` is not a storable value; storing ``None`` is treated the same
        as a missing entry. Persistent backends additionally require values to
        be JSON-serialisable.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the value stored under ``key``, or ``None`` if absent/expired.

        Implementations must record a hit or miss via :meth:`_record_hit` /
        :meth:`_record_miss`.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store ``value`` under ``key``.

        Args:
            key: The cache key.
            value: The value to store. Must not be ``None``.
            ttl: Optional time-to-live in seconds. ``None`` means no expiry.

        Raises:
            ValueError: If ``value`` is ``None`` or ``ttl`` is non-positive.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete ``key``. Returns ``True`` if an entry was removed."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""
        raise NotImplementedError

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` maps to a live entry (counts as a lookup)."""
        return isinstance(key, str) and self.get(key) is not None

    # -- Request-keyed convenience helpers ---------------------------------

    @staticmethod
    def key_for(model: str, messages: Sequence[Message], **params: Any) -> str:
        """Return the stable cache key for a request (see :func:`make_key`)."""
        return make_key(model, messages, **params)

    def get_response(
        self, model: str, messages: Sequence[Message], **params: Any
    ) -> Any | None:
        """Look up a cached response for a request."""
        return self.get(self.key_for(model, messages, **params))

    def set_response(
        self,
        model: str,
        messages: Sequence[Message],
        value: Any,
        *,
        ttl: float | None = None,
        **params: Any,
    ) -> None:
        """Cache a response for a request under its stable key."""
        self.set(self.key_for(model, messages, **params), value, ttl=ttl)

    @staticmethod
    def _validate_set(value: Any, ttl: float | None) -> None:
        """Shared validation for :meth:`set` implementations."""
        if value is None:
            raise ValueError("cannot cache None; it is indistinguishable from a miss")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive when provided")
