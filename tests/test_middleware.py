"""Tests for the framework-agnostic OptimizingPipeline middleware."""

from __future__ import annotations

import pytest

from smarttokenoptimizer.budgeting import DropOldestStrategy, SmartTokenOptimizer
from smarttokenoptimizer.cache import MemoryCache
from smarttokenoptimizer.cost import CostEstimator, UsageTracker
from smarttokenoptimizer.credentials import CredentialPool
from smarttokenoptimizer.middleware import (
    CallRequest,
    CompletionResult,
    OptimizingPipeline,
)
from smarttokenoptimizer.routing import NoAvailableProviderError, Provider, Router
from smarttokenoptimizer.tokenization import Message, MessageOverhead, TokenCounter


class WordCounter(TokenCounter):
    def __init__(self) -> None:
        super().__init__(
            overhead=MessageOverhead(
                tokens_per_message=0, tokens_per_name=0, reply_priming=0
            )
        )

    def count_text(self, text: str) -> int:
        return len(text.split())


def echo_call(request: CallRequest) -> dict[str, object]:
    return {"provider": request.provider, "n": len(request.messages)}


MESSAGES: list[Message] = [{"role": "user", "content": "hello world"}]


class TestBasics:
    def test_requires_model(self) -> None:
        pipeline = OptimizingPipeline(echo_call)
        with pytest.raises(ValueError, match="model"):
            pipeline.complete(MESSAGES)

    def test_minimal_pipeline_calls_through(self) -> None:
        pipeline = OptimizingPipeline(echo_call, model="gpt-4o")
        result = pipeline.complete(MESSAGES)
        assert isinstance(result, CompletionResult)
        assert result.cached is False
        assert result.response["n"] == 1
        assert result.model == "gpt-4o"

    def test_model_override(self) -> None:
        seen = {}

        def call(request: CallRequest) -> str:
            seen["model"] = request.model
            return "ok"

        pipeline = OptimizingPipeline(call, model="gpt-4o")
        pipeline.complete(MESSAGES, model="gpt-4o-mini")
        assert seen["model"] == "gpt-4o-mini"

    def test_params_forwarded(self) -> None:
        seen = {}

        def call(request: CallRequest) -> str:
            seen.update(request.params)
            return "ok"

        OptimizingPipeline(call, model="gpt-4o").complete(MESSAGES, temperature=0.5)
        assert seen["temperature"] == 0.5


class TestCaching:
    def test_second_identical_call_is_cached(self) -> None:
        calls = []

        def call(request: CallRequest) -> str:
            calls.append(1)
            return "response"

        pipeline = OptimizingPipeline(call, model="gpt-4o", cache=MemoryCache())
        first = pipeline.complete(MESSAGES)
        second = pipeline.complete(MESSAGES)
        assert first.cached is False
        assert second.cached is True
        assert second.response == "response"
        assert len(calls) == 1  # provider only hit once

    def test_different_params_bypass_cache(self) -> None:
        calls = []

        def call(request: CallRequest) -> str:
            calls.append(1)
            return "r"

        pipeline = OptimizingPipeline(call, model="gpt-4o", cache=MemoryCache())
        pipeline.complete(MESSAGES, temperature=0.1)
        pipeline.complete(MESSAGES, temperature=0.9)
        assert len(calls) == 2


class TestOptimization:
    def test_optimizer_shrinks_messages(self) -> None:
        counter = WordCounter()
        seen = {}

        def call(request: CallRequest) -> str:
            seen["count"] = len(request.messages)
            return "ok"

        long_convo: list[Message] = [
            {"role": "system", "content": "sys"},
            *[{"role": "user", "content": f"m{i}"} for i in range(20)],
        ]
        optimizer = SmartTokenOptimizer(
            5, counter=counter, strategy=DropOldestStrategy()
        )
        pipeline = OptimizingPipeline(call, model="gpt-4o", optimizer=optimizer)
        result = pipeline.complete(long_convo)
        assert seen["count"] < len(long_convo)
        assert result.tokens_saved > 0


class TestRouting:
    def _router(self) -> tuple[Router, Provider]:
        provider = Provider("openai", pool=CredentialPool(), models=["gpt-4o"])
        provider.pool.add_key("sk-openai-secret", id="k1")
        return Router([provider]), provider

    def test_routes_and_records_success(self) -> None:
        router, provider = self._router()

        def call(request: CallRequest) -> str:
            assert request.api_key == "sk-openai-secret"
            assert request.provider == "openai"
            return "ok"

        pipeline = OptimizingPipeline(call, model="gpt-4o", router=router)
        result = pipeline.complete(MESSAGES)
        assert result.provider == "openai"
        assert provider.pool.health()[0].successes == 1

    def test_failure_records_and_reraises(self) -> None:
        router, provider = self._router()

        def call(request: CallRequest) -> str:
            raise RuntimeError("boom")

        pipeline = OptimizingPipeline(call, model="gpt-4o", router=router)
        with pytest.raises(RuntimeError, match="boom"):
            pipeline.complete(MESSAGES)
        assert provider.pool.health()[0].failures == 1

    def test_no_provider_raises(self) -> None:
        router = Router([Provider("x", pool=CredentialPool(), models=["other"])])
        pipeline = OptimizingPipeline(echo_call, model="gpt-4o", router=router)
        with pytest.raises(NoAvailableProviderError):
            pipeline.complete(MESSAGES)


class TestAnalyticsAndCost:
    def test_tracker_records_usage(self) -> None:
        tracker = UsageTracker()

        def call(request: CallRequest) -> dict[str, int]:
            return {"in": 100, "out": 20}

        pipeline = OptimizingPipeline(
            call,
            model="gpt-4o",
            tracker=tracker,
            estimator=CostEstimator(),
            usage_extractor=lambda r: (r["in"], r["out"]),
        )
        result = pipeline.complete(MESSAGES)
        snap = tracker.snapshot()
        assert snap.requests == 1
        assert snap.input_tokens == 100
        assert snap.output_tokens == 20
        assert result.cost > 0
        assert snap.cost == pytest.approx(result.cost)

    def test_cache_hit_recorded_in_analytics(self) -> None:
        tracker = UsageTracker()
        pipeline = OptimizingPipeline(
            echo_call, model="gpt-4o", cache=MemoryCache(), tracker=tracker
        )
        pipeline.complete(MESSAGES)
        pipeline.complete(MESSAGES)
        snap = tracker.snapshot()
        assert snap.cache_hits == 1
        assert snap.cache_misses == 1
        assert snap.cache_hit_rate == pytest.approx(0.5)

    def test_failure_recorded_as_error(self) -> None:
        tracker = UsageTracker()

        def call(request: CallRequest) -> str:
            raise RuntimeError("nope")

        pipeline = OptimizingPipeline(call, model="gpt-4o", tracker=tracker)
        with pytest.raises(RuntimeError):
            pipeline.complete(MESSAGES)
        snap = tracker.snapshot()
        assert snap.errors == 1
        assert snap.success_rate == 0.0

    def test_unknown_model_cost_is_zero(self) -> None:
        def call(request: CallRequest) -> str:
            return "ok"

        pipeline = OptimizingPipeline(
            call, model="made-up-model", estimator=CostEstimator()
        )
        result = pipeline.complete(MESSAGES)
        assert result.cost == 0.0

    def test_savings_cost_recorded(self) -> None:
        counter = WordCounter()
        tracker = UsageTracker()
        long_convo: list[Message] = [
            {"role": "user", "content": f"m{i} filler words here"} for i in range(20)
        ]
        pipeline = OptimizingPipeline(
            lambda r: "ok",
            model="gpt-4o",
            optimizer=SmartTokenOptimizer(5, counter=counter),
            tracker=tracker,
            estimator=CostEstimator(),
        )
        result = pipeline.complete(long_convo)
        assert result.tokens_saved > 0
        # cost_saved is derived from the saved tokens at the model's input price.
        assert tracker.snapshot().cost_saved > 0

    def test_savings_cost_zero_for_unknown_model(self) -> None:
        counter = WordCounter()
        tracker = UsageTracker()
        long_convo: list[Message] = [
            {"role": "user", "content": f"m{i} filler words here"} for i in range(20)
        ]
        pipeline = OptimizingPipeline(
            lambda r: "ok",
            model="made-up-model",
            optimizer=SmartTokenOptimizer(5, counter=counter),
            tracker=tracker,
            estimator=CostEstimator(),
        )
        result = pipeline.complete(long_convo)
        assert result.tokens_saved > 0
        # No pricing for the model -> cost_saved stays zero.
        assert tracker.snapshot().cost_saved == 0.0

    def test_input_tokens_counted_without_extractor(self) -> None:
        pipeline = OptimizingPipeline(echo_call, model="gpt-4o", counter=WordCounter())
        result = pipeline.complete(MESSAGES, expected_output_tokens=7)
        # "user" + "hello world" = 3 words with WordCounter.
        assert result.input_tokens == 3
        assert result.output_tokens == 7


class TestFullStack:
    def test_all_components_together(self) -> None:
        counter = WordCounter()
        tracker = UsageTracker()
        provider = Provider("openai", pool=CredentialPool(), models=["gpt-4o"])
        provider.pool.add_key("sk-openai-secret")

        def call(request: CallRequest) -> dict[str, int]:
            return {"in": 50, "out": 10}

        pipeline = OptimizingPipeline(
            call,
            model="gpt-4o",
            optimizer=SmartTokenOptimizer(1000, counter=counter),
            cache=MemoryCache(),
            router=Router([provider]),
            tracker=tracker,
            estimator=CostEstimator(),
            usage_extractor=lambda r: (r["in"], r["out"]),
        )
        first = pipeline.complete(MESSAGES)
        second = pipeline.complete(MESSAGES)
        assert first.cached is False
        assert second.cached is True
        assert tracker.snapshot().requests == 2
        assert provider.pool.health()[0].successes == 1  # only one real call
