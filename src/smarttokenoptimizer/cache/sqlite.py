"""Persistent prompt cache backed by SQLite (standard library only)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from types import TracebackType
from typing import Any

from .base import PromptCache


class SQLiteCache(PromptCache):
    """Durable prompt cache stored in a SQLite database.

    Values persist across process restarts and are shared by any process
    opening the same database file. Uses only the Python standard library, so
    it adds no dependencies. Values must be JSON-serialisable.

    Because expiry must survive restarts, TTL is measured against the wall
    clock (Unix time) rather than a monotonic clock.

    Args:
        path: Filesystem path to the database. Use ``":memory:"`` for an
            ephemeral in-memory database (useful in tests).
        default_ttl: Default time-to-live in seconds for entries stored without
            an explicit ``ttl``. ``None`` means no default expiry.
        table: Name of the backing table. Defaults to ``"prompt_cache"``.

    Raises:
        ValueError: If ``default_ttl`` is non-positive.

    Example:
        >>> cache = SQLiteCache(":memory:")
        >>> cache.set("k", {"answer": 42})
        >>> cache.get("k")
        {'answer': 42}
        >>> cache.close()
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        default_ttl: float | None = None,
        table: str = "prompt_cache",
    ) -> None:
        super().__init__()
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be positive when provided")
        if not table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        self._default_ttl = default_ttl
        self._table = table
        self._lock = threading.Lock()
        # check_same_thread=False plus our own lock makes the connection safe to
        # share across threads.
        self._conn = sqlite3.connect(os.fspath(path), check_same_thread=False)
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  expiry REAL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        """See :meth:`PromptCache.get`."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                f"SELECT value, expiry FROM {self._table} WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                self._record_miss()
                return None
            value_json, expiry = row
            if expiry is not None and expiry <= now:
                self._conn.execute(f"DELETE FROM {self._table} WHERE key = ?", (key,))
                self._conn.commit()
                self._record_miss()
                return None
            self._record_hit()
            return json.loads(value_json)

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """See :meth:`PromptCache.set`.

        Raises:
            ValueError: If ``value`` is ``None`` or ``ttl`` is non-positive.
            TypeError: If ``value`` is not JSON-serialisable.
        """
        self._validate_set(value, ttl)
        value_json = json.dumps(value)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + effective_ttl if effective_ttl is not None else None
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self._table} (key, value, expiry) "
                "VALUES (?, ?, ?)",
                (key, value_json, expiry),
            )
            self._conn.commit()

    def delete(self, key: str) -> bool:
        """See :meth:`PromptCache.delete`."""
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM {self._table} WHERE key = ?", (key,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove all entries and reset hit/miss statistics."""
        with self._lock:
            self._conn.execute(f"DELETE FROM {self._table}")
            self._conn.commit()
        self._reset_stats()

    def purge_expired(self) -> int:
        """Delete all expired entries eagerly. Returns the number removed."""
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM {self._table} WHERE expiry IS NOT NULL AND expiry <= ?",
                (now,),
            )
            self._conn.commit()
            return cursor.rowcount

    def __len__(self) -> int:
        """Return the number of live (non-expired) entries."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._table} "
                "WHERE expiry IS NULL OR expiry > ?",
                (now,),
            ).fetchone()
            return int(row[0])

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
