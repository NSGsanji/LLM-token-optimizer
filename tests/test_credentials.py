"""Tests for the credential management module."""

from __future__ import annotations

import threading

import pytest

from smarttokenoptimizer.credentials import (
    Credential,
    CredentialPool,
    DuplicateCredentialError,
    LeastUsedStrategy,
    NoAvailableCredentialError,
    PriorityStrategy,
    RoundRobinStrategy,
    UnknownCredentialError,
    WeightedRoundRobinStrategy,
)
from smarttokenoptimizer.credentials.strategies import CredentialView


class FakeClock:
    """A controllable monotonic clock for deterministic time-based tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestCredential:
    def test_requires_non_empty_key(self) -> None:
        with pytest.raises(ValueError, match="key"):
            Credential("")

    def test_requires_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Credential("sk-abc", weight=0)

    def test_key_hidden_in_repr(self) -> None:
        cred = Credential("sk-supersecretvalue", provider="openai")
        assert "supersecret" not in repr(cred)
        assert "supersecret" not in str(cred)

    def test_masked_key_short(self) -> None:
        assert Credential("shortkey").masked_key == "****"

    def test_masked_key_long(self) -> None:
        masked = Credential("sk-1234567890abcd").masked_key
        assert masked == "sk-…abcd"
        assert "1234567890" not in masked

    def test_derived_id_is_stable_and_hides_key(self) -> None:
        a = Credential("sk-identical")
        b = Credential("sk-identical")
        assert a.id == b.id
        assert "identical" not in a.id

    def test_explicit_id(self) -> None:
        assert Credential("sk-abc", id="my-key").id == "my-key"

    def test_key_is_accessible(self) -> None:
        # The raw key must remain retrievable for actually making requests.
        assert Credential("sk-abc").key == "sk-abc"


class TestStrategies:
    def _views(self, *specs: tuple[str, int, int, float]) -> list[CredentialView]:
        # (id, priority, uses, weight)
        views = []
        for cid, priority, uses, weight in specs:
            cred = Credential(
                f"sk-{cid}-value", id=cid, priority=priority, weight=weight
            )
            views.append(CredentialView(cred, uses))
        return views

    def test_round_robin_cycles(self) -> None:
        strategy = RoundRobinStrategy()
        views = self._views(("a", 0, 0, 1.0), ("b", 0, 0, 1.0))
        picks = [strategy.select(views).id for _ in range(4)]
        assert picks == ["a", "b", "a", "b"]

    def test_priority_prefers_highest(self) -> None:
        strategy = PriorityStrategy()
        views = self._views(("a", 1, 0, 1.0), ("b", 5, 0, 1.0), ("c", 3, 0, 1.0))
        assert strategy.select(views).id == "b"

    def test_priority_tie_breaks_on_uses(self) -> None:
        strategy = PriorityStrategy()
        views = self._views(("a", 5, 10, 1.0), ("b", 5, 2, 1.0))
        assert strategy.select(views).id == "b"

    def test_least_used(self) -> None:
        strategy = LeastUsedStrategy()
        views = self._views(("a", 0, 7, 1.0), ("b", 0, 3, 1.0), ("c", 0, 9, 1.0))
        assert strategy.select(views).id == "b"

    def test_weighted_distribution(self) -> None:
        strategy = WeightedRoundRobinStrategy()
        views = self._views(("a", 0, 0, 3.0), ("b", 0, 0, 1.0))
        picks = [strategy.select(views).id for _ in range(8)]
        assert picks.count("a") == 6
        assert picks.count("b") == 2

    def test_weighted_prunes_stale_ids(self) -> None:
        strategy = WeightedRoundRobinStrategy()
        first = self._views(("a", 0, 0, 1.0), ("b", 0, 0, 1.0))
        strategy.select(first)
        # 'b' disappears; strategy must not error and must pick from live set.
        second = self._views(("a", 0, 0, 1.0))
        assert strategy.select(second).id == "a"


class TestPoolMembership:
    def test_add_and_len_and_contains(self) -> None:
        pool = CredentialPool()
        cred = pool.add_key("sk-aaaaaaaaaa", provider="openai")
        assert len(pool) == 1
        assert cred.id in pool
        assert "missing" not in pool

    def test_duplicate_rejected(self) -> None:
        pool = CredentialPool()
        pool.add_key("sk-aaaaaaaaaa", id="dup")
        with pytest.raises(DuplicateCredentialError):
            pool.add_key("sk-bbbbbbbbbb", id="dup")

    def test_remove(self) -> None:
        pool = CredentialPool()
        cred = pool.add_key("sk-aaaaaaaaaa")
        pool.remove(cred.id)
        assert cred.id not in pool

    def test_remove_unknown_raises(self) -> None:
        with pytest.raises(UnknownCredentialError):
            CredentialPool().remove("nope")

    def test_ids_in_insertion_order(self) -> None:
        pool = CredentialPool()
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        b = pool.add_key("sk-bbbbbbbbbb", id="b")
        assert pool.ids() == [a.id, b.id]

    def test_construct_with_initial_credentials(self) -> None:
        creds = [Credential("sk-aaaaaaaaaa"), Credential("sk-bbbbbbbbbb")]
        pool = CredentialPool(creds)
        assert len(pool) == 2

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CredentialPool(failure_threshold=0)
        with pytest.raises(ValueError, match="cooldown"):
            CredentialPool(cooldown=0)


class TestPoolAcquisition:
    def test_acquire_empty_raises(self) -> None:
        with pytest.raises(NoAvailableCredentialError):
            CredentialPool().acquire()

    def test_acquire_skips_disabled(self) -> None:
        pool = CredentialPool()
        pool.add(Credential("sk-aaaaaaaaaa", id="a", enabled=False))
        b = pool.add_key("sk-bbbbbbbbbb", id="b")
        assert pool.acquire().id == b.id

    def test_acquire_increments_uses(self) -> None:
        pool = CredentialPool(strategy=LeastUsedStrategy())
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        b = pool.add_key("sk-bbbbbbbbbb", id="b")
        first = pool.acquire()
        second = pool.acquire()
        # Least-used must alternate because uses are tracked.
        assert {first.id, second.id} == {a.id, b.id}


class TestFailoverAndHealth:
    def test_circuit_breaker_trips_after_threshold(self) -> None:
        clock = FakeClock()
        pool = CredentialPool(failure_threshold=2, cooldown=10, time_fn=clock)
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        b = pool.add_key("sk-bbbbbbbbbb", id="b")
        pool.record_failure(a)
        assert a.id in pool.available_ids()  # one failure, still ok
        pool.record_failure(a)
        assert pool.available_ids() == [b.id]  # tripped

    def test_circuit_recovers_after_cooldown(self) -> None:
        clock = FakeClock()
        pool = CredentialPool(failure_threshold=1, cooldown=10, time_fn=clock)
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        pool.record_failure(a)
        assert pool.available_ids() == []
        clock.now = 11
        assert pool.available_ids() == [a.id]

    def test_success_resets_failures(self) -> None:
        clock = FakeClock()
        pool = CredentialPool(failure_threshold=2, cooldown=10, time_fn=clock)
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        pool.record_failure(a)
        pool.record_success(a)
        pool.record_failure(a)  # consecutive resets, so not yet tripped
        assert a.id in pool.available_ids()

    def test_rate_limit_cooldown(self) -> None:
        clock = FakeClock()
        pool = CredentialPool(time_fn=clock)
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        pool.record_rate_limited(a, retry_after=5)
        assert pool.available_ids() == []
        clock.now = 6
        assert pool.available_ids() == [a.id]

    def test_rate_limit_negative_raises(self) -> None:
        pool = CredentialPool()
        a = pool.add_key("sk-aaaaaaaaaa")
        with pytest.raises(ValueError, match="retry_after"):
            pool.record_rate_limited(a, retry_after=-1)

    def test_record_by_id(self) -> None:
        pool = CredentialPool(failure_threshold=1, cooldown=10)
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        pool.record_failure("a")  # by id string
        assert a.id not in pool.available_ids()

    def test_record_unknown_raises(self) -> None:
        with pytest.raises(UnknownCredentialError):
            CredentialPool().record_success("nope")

    def test_health_snapshot(self) -> None:
        clock = FakeClock()
        pool = CredentialPool(failure_threshold=1, cooldown=10, time_fn=clock)
        a = pool.add_key("sk-aaaaaaaaaa", id="a", provider="openai")
        pool.acquire()
        pool.record_failure(a, error="boom")
        health = {h.id: h for h in pool.health()}
        assert health["a"].provider == "openai"
        assert health["a"].uses == 1
        assert health["a"].failures == 1
        assert health["a"].circuit_open is True
        assert health["a"].available is False
        assert health["a"].last_error == "boom"


class TestBorrow:
    def test_borrow_records_success(self) -> None:
        pool = CredentialPool()
        a = pool.add_key("sk-aaaaaaaaaa", id="a")
        with pool.borrow() as cred:
            assert cred.id == a.id
        assert pool.health()[0].successes == 1

    def test_borrow_records_failure_and_reraises(self) -> None:
        pool = CredentialPool(failure_threshold=1, cooldown=10)
        pool.add_key("sk-aaaaaaaaaa", id="a")
        with pytest.raises(RuntimeError, match="boom"), pool.borrow():
            raise RuntimeError("boom")
        assert pool.available_ids() == []
        assert pool.health()[0].last_error == "boom"


class TestThreadSafety:
    def test_concurrent_acquire_and_record(self) -> None:
        pool = CredentialPool(strategy=RoundRobinStrategy())
        for i in range(4):
            pool.add_key(f"sk-{i}aaaaaaaaaa", id=f"c{i}")

        def worker() -> None:
            for _ in range(500):
                cred = pool.acquire()
                pool.record_success(cred)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_uses = sum(h.uses for h in pool.health())
        assert total_uses == 8 * 500
