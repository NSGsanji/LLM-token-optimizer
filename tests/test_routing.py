"""Tests for the provider routing module."""

from __future__ import annotations

import threading

import pytest

from smarttokenoptimizer.credentials import CredentialPool
from smarttokenoptimizer.routing import (
    CheapestPolicy,
    DuplicateProviderError,
    LowestLatencyPolicy,
    NoAvailableProviderError,
    PriorityPolicy,
    Provider,
    RoundRobinPolicy,
    Router,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_provider(
    name: str,
    *,
    keys: int = 1,
    clock: FakeClock | None = None,
    **kwargs: object,
) -> Provider:
    pool = CredentialPool(time_fn=clock or (lambda: 0.0))
    for i in range(keys):
        pool.add_key(f"sk-{name}-{i}-secretvalue", id=f"{name}-{i}")
    return Provider(name, pool=pool, **kwargs)  # type: ignore[arg-type]


class TestProvider:
    def test_requires_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Provider("", pool=CredentialPool())

    def test_requires_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Provider("p", pool=CredentialPool(), weight=0)

    def test_invalid_latency_alpha(self) -> None:
        with pytest.raises(ValueError, match="latency_alpha"):
            Provider("p", pool=CredentialPool(), latency_alpha=0)
        with pytest.raises(ValueError, match="latency_alpha"):
            Provider("p", pool=CredentialPool(), latency_alpha=1.5)

    def test_serves_any_when_no_models(self) -> None:
        p = make_provider("p")
        assert p.serves("anything") is True
        assert p.serves(None) is True

    def test_serves_specific_models(self) -> None:
        p = make_provider("p", models=["gpt-4o"])
        assert p.serves("gpt-4o") is True
        assert p.serves("claude-3-opus") is False
        assert p.serves(None) is True

    def test_availability(self) -> None:
        p = make_provider("p")
        assert p.available is True
        p.enabled = False
        assert p.available is False

    def test_latency_ewma(self) -> None:
        p = make_provider("p", latency_alpha=0.5)
        assert p.avg_latency is None
        p.record_latency(1.0)
        assert p.avg_latency == 1.0
        p.record_latency(3.0)
        # 0.5*3 + 0.5*1 = 2.0
        assert p.avg_latency == pytest.approx(2.0)
        assert p.latency_samples == 2

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency"):
            make_provider("p").record_latency(-1)

    def test_effective_price_hint(self) -> None:
        p = make_provider("p", price_hint=7.5)
        assert p.effective_price("anything") == 7.5

    def test_effective_price_from_table(self) -> None:
        p = make_provider("p")
        # gpt-4o = 2.5 + 10.0 = 12.5 combined per million.
        assert p.effective_price("gpt-4o") == pytest.approx(12.5)

    def test_effective_price_unknown(self) -> None:
        p = make_provider("p")
        assert p.effective_price("made-up-model") is None
        assert p.effective_price(None) is None

    def test_repr_includes_name_and_models(self) -> None:
        p = make_provider("openai", models=["gpt-4o"])
        text = repr(p)
        assert "openai" in text
        assert "gpt-4o" in text


class TestPolicies:
    def test_priority(self) -> None:
        a = make_provider("a", priority=1)
        b = make_provider("b", priority=5)
        c = make_provider("c", priority=3)
        ranked = PriorityPolicy().rank([a, b, c], model=None)
        assert [p.name for p in ranked] == ["b", "c", "a"]

    def test_round_robin_rotates(self) -> None:
        a = make_provider("a")
        b = make_provider("b")
        policy = RoundRobinPolicy()
        first = policy.rank([a, b], model=None)
        second = policy.rank([a, b], model=None)
        assert first[0].name == "a"
        assert second[0].name == "b"

    def test_cheapest_prefers_lower_price(self) -> None:
        cheap = make_provider("cheap", price_hint=1.0)
        pricey = make_provider("pricey", price_hint=9.0)
        ranked = CheapestPolicy().rank([pricey, cheap], model="gpt-4o")
        assert [p.name for p in ranked] == ["cheap", "pricey"]

    def test_cheapest_unknown_price_last(self) -> None:
        known = make_provider("known", price_hint=5.0)
        unknown = make_provider("unknown")
        ranked = CheapestPolicy().rank([unknown, known], model="made-up")
        assert ranked[0].name == "known"

    def test_lowest_latency(self) -> None:
        slow = make_provider("slow")
        fast = make_provider("fast")
        slow.record_latency(2.0)
        fast.record_latency(0.1)
        ranked = LowestLatencyPolicy().rank([slow, fast], model=None)
        assert ranked[0].name == "fast"

    def test_lowest_latency_unmeasured_first(self) -> None:
        measured = make_provider("measured")
        measured.record_latency(1.0)
        fresh = make_provider("fresh")
        ranked = LowestLatencyPolicy().rank([measured, fresh], model=None)
        # Unmeasured provider (latency 0) is explored first.
        assert ranked[0].name == "fresh"


class TestRouterMembership:
    def test_add_len_contains(self) -> None:
        router = Router()
        router.add(make_provider("a"))
        assert len(router) == 1
        assert "a" in router
        assert "b" not in router

    def test_duplicate_rejected(self) -> None:
        router = Router()
        router.add(make_provider("a"))
        with pytest.raises(DuplicateProviderError):
            router.add(make_provider("a"))

    def test_remove(self) -> None:
        router = Router([make_provider("a")])
        router.remove("a")
        assert "a" not in router

    def test_provider_names_order(self) -> None:
        router = Router([make_provider("a"), make_provider("b")])
        assert router.provider_names() == ["a", "b"]


class TestRouting:
    def test_routes_by_priority(self) -> None:
        router = Router(
            [make_provider("a", priority=1), make_provider("b", priority=9)],
            policy=PriorityPolicy(),
        )
        assert router.route().provider.name == "b"

    def test_model_filtering(self) -> None:
        gpt = make_provider("gpt", models=["gpt-4o"], priority=1)
        claude = make_provider("claude", models=["claude-3-opus"], priority=9)
        router = Router([gpt, claude], policy=PriorityPolicy())
        # claude has higher priority but does not serve gpt-4o.
        assert router.route(model="gpt-4o").provider.name == "gpt"

    def test_no_provider_serves_model(self) -> None:
        router = Router([make_provider("gpt", models=["gpt-4o"])])
        with pytest.raises(NoAvailableProviderError):
            router.route(model="claude-3-opus")

    def test_empty_router_raises(self) -> None:
        with pytest.raises(NoAvailableProviderError):
            Router().route()

    def test_route_returns_credential(self) -> None:
        router = Router([make_provider("a")])
        route = router.route()
        assert route.key.startswith("sk-a-")
        assert route.credential is not None

    def test_failover_to_next_provider(self) -> None:
        clock = FakeClock()
        primary = make_provider("primary", priority=9, clock=clock)
        backup = make_provider("backup", priority=1, clock=clock)
        router = Router([primary, backup], policy=PriorityPolicy(), time_fn=clock)
        # Rate-limit the only key on the higher-priority provider.
        primary.pool.record_rate_limited(primary.pool.ids()[0], retry_after=100)
        assert router.route().provider.name == "backup"

    def test_available_providers(self) -> None:
        clock = FakeClock()
        a = make_provider("a", clock=clock)
        b = make_provider("b", clock=clock)
        router = Router([a, b], time_fn=clock)
        assert set(router.available_providers()) == {"a", "b"}
        b.pool.record_rate_limited(b.pool.ids()[0], retry_after=100)
        assert router.available_providers() == ["a"]


class TestDispatch:
    def test_dispatch_records_success_and_latency(self) -> None:
        clock = FakeClock()
        provider = make_provider("a", clock=clock)
        router = Router([provider], time_fn=clock)
        with router.dispatch() as route:
            clock.now = 0.25
            assert route.provider.name == "a"
        assert provider.avg_latency == pytest.approx(0.25)
        assert provider.pool.health()[0].successes == 1

    def test_dispatch_records_failure_and_reraises(self) -> None:
        clock = FakeClock()
        provider = make_provider("a", clock=clock)
        router = Router([provider], time_fn=clock)
        with pytest.raises(RuntimeError, match="boom"), router.dispatch():
            raise RuntimeError("boom")
        assert provider.pool.health()[0].failures == 1
        assert provider.pool.health()[0].last_error == "boom"


class TestThreadSafety:
    def test_concurrent_routing(self) -> None:
        clock = FakeClock()
        providers = [make_provider(f"p{i}", keys=2, clock=clock) for i in range(3)]
        router = Router(providers, policy=RoundRobinPolicy(), time_fn=clock)

        def worker() -> None:
            for _ in range(300):
                route = router.route()
                route.provider.pool.record_success(route.credential)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = sum(h.uses for p in providers for h in p.pool.health())
        assert total == 6 * 300
