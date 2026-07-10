"""Tests for the cost estimation and analytics module."""

from __future__ import annotations

import threading

import pytest

from smarttokenoptimizer.cost import (
    AnalyticsSnapshot,
    CostEstimator,
    ModelPricing,
    UnknownPricingError,
    UsageTracker,
    clear_custom_pricing,
    get_pricing,
    register_pricing,
)


class TestModelPricing:
    def test_input_and_output_cost(self) -> None:
        pricing = ModelPricing(input_per_million=2.50, output_per_million=10.0)
        assert pricing.input_cost(1_000_000) == pytest.approx(2.50)
        assert pricing.output_cost(500_000) == pytest.approx(5.0)

    def test_zero_tokens(self) -> None:
        pricing = ModelPricing(1.0, 2.0)
        assert pricing.input_cost(0) == 0.0
        assert pricing.output_cost(0) == 0.0


class TestGetPricing:
    def test_known_model(self) -> None:
        pricing = get_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_per_million == pytest.approx(2.50)

    def test_longest_prefix_wins(self) -> None:
        # gpt-4o-mini must not resolve to the shorter gpt-4o entry.
        mini = get_pricing("gpt-4o-mini-2024-07-18")
        assert mini is not None
        assert mini.input_per_million == pytest.approx(0.15)

    def test_current_claude_models(self) -> None:
        # Current-generation Claude models resolve to their published rates.
        opus = get_pricing("claude-opus-4-8")
        assert opus is not None
        assert (opus.input_per_million, opus.output_per_million) == (5.0, 25.0)
        sonnet = get_pricing("claude-sonnet-5")
        assert sonnet is not None
        assert (sonnet.input_per_million, sonnet.output_per_million) == (3.0, 15.0)
        haiku = get_pricing("claude-haiku-4-5")
        assert haiku is not None
        assert (haiku.input_per_million, haiku.output_per_million) == (1.0, 5.0)

    def test_unknown_model_returns_none(self) -> None:
        assert get_pricing("totally-made-up-model") is None

    def test_case_insensitive(self) -> None:
        assert get_pricing("GPT-4O") is not None

    def test_explicit_table(self) -> None:
        table = {"foo": ModelPricing(1.0, 2.0)}
        assert get_pricing("foo-bar", pricing_table=table) is not None
        assert get_pricing("gpt-4o", pricing_table=table) is None


class TestRegisterPricing:
    def teardown_method(self) -> None:
        clear_custom_pricing()

    def test_override_takes_precedence(self) -> None:
        register_pricing("gpt-4o", ModelPricing(99.0, 99.0))
        pricing = get_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_per_million == 99.0

    def test_register_new_model(self) -> None:
        register_pricing("my-local-model", ModelPricing(0.0, 0.0))
        assert get_pricing("my-local-model-v2") is not None

    def test_clear_custom_pricing(self) -> None:
        register_pricing("temp-model", ModelPricing(1.0, 1.0))
        clear_custom_pricing()
        assert get_pricing("temp-model") is None


class TestCostEstimator:
    def test_estimate_input_and_output(self) -> None:
        est = CostEstimator().estimate("gpt-4o", input_tokens=1000, output_tokens=500)
        assert est.input_cost == pytest.approx(0.0025)
        assert est.output_cost == pytest.approx(0.005)
        assert est.total_cost == pytest.approx(0.0075)
        assert est.total_tokens == 1500

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownPricingError):
            CostEstimator().estimate("nope-model", input_tokens=10)

    def test_negative_tokens_raise(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostEstimator().estimate("gpt-4o", input_tokens=-1)

    def test_estimate_messages_counts_input(self) -> None:
        est = CostEstimator().estimate_messages(
            [{"role": "user", "content": "hello world"}],
            model="gpt-4o",
            expected_output_tokens=100,
        )
        assert est.input_tokens > 0
        assert est.output_tokens == 100
        assert est.total_cost > 0

    def test_custom_pricing_table(self) -> None:
        table = {"x": ModelPricing(1_000_000.0, 0.0)}  # $1 per token input
        est = CostEstimator(pricing_table=table).estimate("x", input_tokens=3)
        assert est.input_cost == pytest.approx(3.0)

    def test_pricing_for_raises_on_unknown(self) -> None:
        with pytest.raises(UnknownPricingError):
            CostEstimator().pricing_for("unknown")


class TestUsageTracker:
    def test_records_and_aggregates(self) -> None:
        tracker = UsageTracker()
        tracker.record(
            model="gpt-4o", input_tokens=1000, output_tokens=200, cost=0.0045
        )
        tracker.record(model="gpt-4o", input_tokens=500, cost=0.00125)
        snap = tracker.snapshot()
        assert isinstance(snap, AnalyticsSnapshot)
        assert snap.requests == 2
        assert snap.input_tokens == 1500
        assert snap.output_tokens == 200
        assert snap.total_tokens == 1700
        assert snap.cost == pytest.approx(0.00575)
        assert snap.average_cost_per_request == pytest.approx(0.002875)

    def test_savings_tracking(self) -> None:
        tracker = UsageTracker()
        tracker.record(input_tokens=700, tokens_saved=300, cost_saved=0.01)
        snap = tracker.snapshot()
        assert snap.tokens_saved == 300
        assert snap.cost_saved == pytest.approx(0.01)
        # baseline = 700 + 300 = 1000; saved 300 -> 0.3
        assert snap.savings_ratio == pytest.approx(0.3)

    def test_cache_hit_rate(self) -> None:
        tracker = UsageTracker()
        tracker.record(cache_hit=True)
        tracker.record(cache_hit=True)
        tracker.record(cache_hit=False)
        tracker.record(cache_hit=None)  # not counted in cache stats
        snap = tracker.snapshot()
        assert snap.cache_hits == 2
        assert snap.cache_misses == 1
        assert snap.cache_hit_rate == pytest.approx(2 / 3)

    def test_success_rate(self) -> None:
        tracker = UsageTracker()
        tracker.record(success=True)
        tracker.record(success=False)
        snap = tracker.snapshot()
        assert snap.errors == 1
        assert snap.success_rate == pytest.approx(0.5)

    def test_empty_snapshot_defaults(self) -> None:
        snap = UsageTracker().snapshot()
        assert snap.requests == 0
        assert snap.cache_hit_rate == 0.0
        assert snap.success_rate == 1.0
        assert snap.average_cost_per_request == 0.0
        assert snap.savings_ratio == 0.0

    def test_usage_by_model(self) -> None:
        tracker = UsageTracker()
        tracker.record(model="gpt-4o")
        tracker.record(model="gpt-4o")
        tracker.record(model="claude-3-opus")
        assert tracker.usage_by_model() == {"gpt-4o": 2, "claude-3-opus": 1}

    def test_reset(self) -> None:
        tracker = UsageTracker()
        tracker.record(input_tokens=100)
        tracker.reset()
        assert tracker.snapshot().requests == 0

    def test_negative_amounts_raise(self) -> None:
        tracker = UsageTracker()
        with pytest.raises(ValueError):
            tracker.record(input_tokens=-1)
        with pytest.raises(ValueError):
            tracker.record(tokens_saved=-5)

    def test_currency_propagates(self) -> None:
        tracker = UsageTracker(currency="EUR")
        assert tracker.snapshot().currency == "EUR"

    def test_thread_safe_accumulation(self) -> None:
        tracker = UsageTracker()

        def worker() -> None:
            for _ in range(1000):
                tracker.record(input_tokens=1, cost=0.001)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snap = tracker.snapshot()
        assert snap.requests == 8000
        assert snap.input_tokens == 8000
        assert snap.cost == pytest.approx(8.0)
