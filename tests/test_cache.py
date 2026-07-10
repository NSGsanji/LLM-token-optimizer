"""Tests for the prompt cache module (key, memory and SQLite backends)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from smarttokenoptimizer.cache import (
    CacheStats,
    MemoryCache,
    PromptCache,
    SQLiteCache,
    make_key,
)


class TestMakeKey:
    def test_deterministic(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        assert make_key("gpt-4o", messages) == make_key("gpt-4o", messages)

    def test_insensitive_to_key_order(self) -> None:
        a = make_key("gpt-4o", [{"role": "user", "content": "hi"}], b=1, a=2)
        b = make_key("gpt-4o", [{"content": "hi", "role": "user"}], a=2, b=1)
        assert a == b

    def test_none_params_ignored(self) -> None:
        with_none = make_key("gpt-4o", [], temperature=None)
        without = make_key("gpt-4o", [])
        assert with_none == without

    def test_different_model_differs(self) -> None:
        assert make_key("gpt-4o", []) != make_key("gpt-4o-mini", [])

    def test_different_params_differ(self) -> None:
        a = make_key("gpt-4o", [], temperature=0.7)
        b = make_key("gpt-4o", [], temperature=0.9)
        assert a != b

    def test_is_sha256_hex(self) -> None:
        key = make_key("gpt-4o", [])
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_handles_non_json_param(self) -> None:
        # A non-JSON value (e.g. a set) must not raise and must be stable.
        a = make_key("gpt-4o", [], tags={"x", "y"})
        b = make_key("gpt-4o", [], tags={"y", "x"})
        assert a == b


class TestCacheStats:
    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=3, misses=1)
        assert stats.lookups == 4
        assert stats.hit_rate == pytest.approx(0.75)

    def test_empty_hit_rate(self) -> None:
        assert CacheStats().hit_rate == 0.0


# A backend factory list lets us run the shared contract against every backend.
def _memory() -> MemoryCache:
    return MemoryCache()


def _sqlite() -> SQLiteCache:
    return SQLiteCache(":memory:")


@pytest.fixture(params=[_memory, _sqlite], ids=["memory", "sqlite"])
def cache(request: pytest.FixtureRequest) -> Iterator[PromptCache]:
    instance: PromptCache = request.param()
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            close()


class TestSharedContract:
    """Behaviour every PromptCache backend must satisfy."""

    def test_set_and_get(self, cache: PromptCache) -> None:
        cache.set("k", {"answer": 42})
        assert cache.get("k") == {"answer": 42}

    def test_missing_key_returns_none(self, cache: PromptCache) -> None:
        assert cache.get("absent") is None

    def test_overwrite(self, cache: PromptCache) -> None:
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2

    def test_delete(self, cache: PromptCache) -> None:
        cache.set("k", 1)
        assert cache.delete("k") is True
        assert cache.get("k") is None
        assert cache.delete("k") is False

    def test_clear(self, cache: PromptCache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_contains(self, cache: PromptCache) -> None:
        cache.set("k", "v")
        assert "k" in cache
        assert "nope" not in cache

    def test_contains_non_string(self, cache: PromptCache) -> None:
        assert 123 not in cache

    def test_cannot_store_none(self, cache: PromptCache) -> None:
        with pytest.raises(ValueError, match="None"):
            cache.set("k", None)

    def test_invalid_ttl(self, cache: PromptCache) -> None:
        with pytest.raises(ValueError, match="ttl"):
            cache.set("k", 1, ttl=0)
        with pytest.raises(ValueError, match="ttl"):
            cache.set("k", 1, ttl=-5)

    def test_ttl_expiry(self, cache: PromptCache) -> None:
        cache.set("k", "v", ttl=0.05)
        assert cache.get("k") == "v"
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_stats_tracking(self, cache: PromptCache) -> None:
        cache.set("k", 1)
        cache.get("k")  # hit
        cache.get("missing")  # miss
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_request_keyed_helpers(self, cache: PromptCache) -> None:
        messages = [{"role": "user", "content": "hi"}]
        cache.set_response("gpt-4o", messages, {"reply": "yo"}, temperature=0.5)
        assert cache.get_response("gpt-4o", messages, temperature=0.5) == {
            "reply": "yo"
        }
        # Different params -> different key -> miss.
        assert cache.get_response("gpt-4o", messages, temperature=0.9) is None

    def test_length(self, cache: PromptCache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2  # type: ignore[arg-type]


class TestMemoryCache:
    def test_lru_eviction(self) -> None:
        cache = MemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # 'a' now most-recently used
        cache.set("c", 3)  # evicts least-recently used 'b'
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_unbounded_by_default(self) -> None:
        cache = MemoryCache()
        for i in range(1000):
            cache.set(str(i), i)
        assert len(cache) == 1000

    def test_default_ttl(self) -> None:
        cache = MemoryCache(default_ttl=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_explicit_ttl_overrides_default(self) -> None:
        cache = MemoryCache(default_ttl=0.05)
        cache.set("k", "v", ttl=5)
        time.sleep(0.08)
        assert cache.get("k") == "v"

    def test_len_purges_expired(self) -> None:
        cache = MemoryCache()
        cache.set("k", "v", ttl=0.05)
        time.sleep(0.08)
        assert len(cache) == 0

    def test_invalid_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            MemoryCache(max_size=0)

    def test_invalid_default_ttl(self) -> None:
        with pytest.raises(ValueError, match="default_ttl"):
            MemoryCache(default_ttl=0)

    def test_thread_safe(self) -> None:
        cache = MemoryCache()

        def worker(offset: int) -> None:
            for i in range(500):
                cache.set(f"{offset}-{i}", i)
                cache.get(f"{offset}-{i}")

        threads = [threading.Thread(target=worker, args=(o,)) for o in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(cache) == 8 * 500


class TestSQLiteCache:
    def test_persists_across_instances(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/cache.db"  # type: ignore[operator]
        first = SQLiteCache(db)
        first.set("k", {"v": 1})
        first.close()

        second = SQLiteCache(db)
        assert second.get("k") == {"v": 1}
        second.close()

    def test_context_manager_closes(self) -> None:
        with SQLiteCache(":memory:") as cache:
            cache.set("k", 1)
            assert cache.get("k") == 1

    def test_purge_expired(self) -> None:
        cache = SQLiteCache(":memory:")
        cache.set("live", 1)
        cache.set("dead", 2, ttl=0.05)
        time.sleep(0.08)
        removed = cache.purge_expired()
        assert removed == 1
        assert cache.get("live") == 1
        cache.close()

    def test_non_json_value_raises(self) -> None:
        cache = SQLiteCache(":memory:")
        with pytest.raises(TypeError):
            cache.set("k", {1, 2, 3})  # sets are not JSON-serialisable
        cache.close()

    def test_invalid_table_name(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            SQLiteCache(":memory:", table="bad name!")

    def test_invalid_default_ttl(self) -> None:
        with pytest.raises(ValueError, match="default_ttl"):
            SQLiteCache(":memory:", default_ttl=-1)
