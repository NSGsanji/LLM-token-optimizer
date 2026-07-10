"""A framework-agnostic optimizing pipeline that ties the library together.

:class:`OptimizingPipeline` orchestrates the whole request lifecycle around a
single user-supplied ``call_fn``:

1. **Optimize** the conversation to fit a token budget (optional).
2. **Cache** lookup — return a stored response on a hit (optional).
3. **Route** to a provider and acquire a credential (optional).
4. **Call** the provider via the injected ``call_fn``.
5. **Cache** the response and **record** usage, cost and savings (optional).

Every collaborator is injected, so the pipeline depends on no provider SDK and
is fully testable with a fake ``call_fn``. Wrapping the OpenAI SDK, Anthropic
SDK, FastAPI handlers, LiteLLM, etc. is a thin adapter around ``call_fn``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# Optional collaborators are imported lazily/for typing only.
from ..budgeting.optimizer import SmartTokenOptimizer
from ..cache.base import PromptCache
from ..cost.analytics import UsageTracker
from ..cost.errors import UnknownPricingError
from ..cost.estimator import CostEstimator
from ..routing.router import Router
from ..tokenization.base import TokenCounter
from ..tokenization.registry import get_counter
from ..tokenization.types import Message


@dataclass(frozen=True, slots=True)
class CallRequest:
    """The request handed to a pipeline's ``call_fn``.

    Attributes:
        model: The resolved model id.
        messages: The (already optimized) messages to send.
        api_key: The credential key to authenticate with, if routing is used.
        provider: The selected provider name, if routing is used.
        params: Any additional request parameters (temperature, tools, …).
    """

    model: str
    messages: list[Message]
    api_key: str | None
    provider: str | None
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """The outcome of :meth:`OptimizingPipeline.complete`.

    Attributes:
        response: The value returned by ``call_fn`` (or the cached value).
        model: The model used.
        cached: Whether the response came from the cache.
        provider: The provider that served the request, if routed.
        input_tokens: Prompt tokens sent (after optimization).
        output_tokens: Completion tokens, if known (else ``0``).
        tokens_saved: Tokens saved by optimization.
        cost: Estimated request cost (``0.0`` on a cache hit or if unknown).
    """

    response: Any
    model: str
    cached: bool
    provider: str | None
    input_tokens: int
    output_tokens: int
    tokens_saved: int
    cost: float


#: Extracts ``(input_tokens, output_tokens)`` from a provider response. Return
#: ``None`` for a field that cannot be determined.
UsageExtractor = Callable[[Any], "tuple[int | None, int | None]"]


class OptimizingPipeline:
    """Wire optimization, caching, routing, cost and analytics around a call.

    Args:
        call_fn: The function that actually performs the request. It receives a
            :class:`CallRequest` and returns the provider response.
        model: Default model id used when ``complete`` is called without one.
        optimizer: Optional :class:`SmartTokenOptimizer` applied to messages.
        cache: Optional :class:`PromptCache` for response reuse.
        router: Optional :class:`Router` selecting provider + credential.
        tracker: Optional :class:`UsageTracker` for analytics.
        estimator: Optional :class:`CostEstimator` for cost/savings figures.
        counter: Token counter for measuring input tokens. Defaults to a
            model-appropriate counter.
        cache_ttl: TTL (seconds) for cached responses. ``None`` means no expiry.
        usage_extractor: Optional callable returning ``(input, output)`` token
            counts from a response, used in preference to estimation.
    """

    def __init__(
        self,
        call_fn: Callable[[CallRequest], Any],
        *,
        model: str | None = None,
        optimizer: SmartTokenOptimizer | None = None,
        cache: PromptCache | None = None,
        router: Router | None = None,
        tracker: UsageTracker | None = None,
        estimator: CostEstimator | None = None,
        counter: TokenCounter | None = None,
        cache_ttl: float | None = None,
        usage_extractor: UsageExtractor | None = None,
    ) -> None:
        self._call_fn = call_fn
        self._model = model
        self._optimizer = optimizer
        self._cache = cache
        self._router = router
        self._tracker = tracker
        self._estimator = estimator
        self._counter = counter
        self._cache_ttl = cache_ttl
        self._usage_extractor = usage_extractor

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        expected_output_tokens: int = 0,
        **params: Any,
    ) -> CompletionResult:
        """Run the full pipeline for one completion request.

        Args:
            messages: The conversation to send.
            model: Overrides the pipeline's default model.
            expected_output_tokens: Assumed completion length for cost
                estimation when the response carries no usage information.
            **params: Extra request parameters forwarded to ``call_fn`` and
                included in the cache key.

        Returns:
            A :class:`CompletionResult`.

        Raises:
            ValueError: If no model is available.
        """
        resolved = model or self._model
        if not resolved:
            raise ValueError("a model must be provided to complete()")

        optimized, tokens_saved = self._optimize(messages)

        cached = self._cache_lookup(resolved, optimized, params)
        if cached is not None:
            self._record(
                model=resolved,
                input_tokens=0,
                output_tokens=0,
                tokens_saved=tokens_saved,
                cost=0.0,
                cache_hit=True,
                success=True,
            )
            return CompletionResult(
                response=cached,
                model=resolved,
                cached=True,
                provider=None,
                input_tokens=0,
                output_tokens=0,
                tokens_saved=tokens_saved,
                cost=0.0,
            )

        response, provider, input_tokens, output_tokens = self._invoke(
            resolved, optimized, params, expected_output_tokens
        )

        if self._cache is not None:
            self._cache.set_response(
                resolved, optimized, response, ttl=self._cache_ttl, **params
            )

        cost = self._estimate_cost(resolved, input_tokens, output_tokens)
        self._record(
            model=resolved,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            cost=cost,
            cache_hit=False if self._cache is not None else None,
            success=True,
            cost_saved=self._estimate_saved_cost(resolved, tokens_saved),
        )
        return CompletionResult(
            response=response,
            model=resolved,
            cached=False,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            cost=cost,
        )

    # -- Steps -------------------------------------------------------------

    def _optimize(self, messages: Sequence[Message]) -> tuple[list[Message], int]:
        if self._optimizer is None:
            return list(messages), 0
        result = self._optimizer.optimize_detailed(messages)
        return result.messages, result.tokens_saved

    def _cache_lookup(
        self, model: str, messages: list[Message], params: dict[str, Any]
    ) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get_response(model, messages, **params)

    def _invoke(
        self,
        model: str,
        messages: list[Message],
        params: dict[str, Any],
        expected_output_tokens: int,
    ) -> tuple[Any, str | None, int, int]:
        route = None
        provider_name: str | None = None
        api_key: str | None = None
        if self._router is not None:
            route = self._router.route(model=model)
            provider_name = route.provider.name
            api_key = route.key

        request = CallRequest(
            model=model,
            messages=messages,
            api_key=api_key,
            provider=provider_name,
            params=params,
        )
        try:
            response = self._call_fn(request)
        except Exception as exc:
            if route is not None:
                route.provider.pool.record_failure(route.credential, error=str(exc))
            self._record(
                model=model,
                input_tokens=0,
                output_tokens=0,
                tokens_saved=0,
                cost=0.0,
                cache_hit=None,
                success=False,
            )
            raise

        if route is not None:
            route.provider.pool.record_success(route.credential)

        input_tokens, output_tokens = self._resolve_usage(
            model, messages, response, expected_output_tokens
        )
        return response, provider_name, input_tokens, output_tokens

    def _resolve_usage(
        self,
        model: str,
        messages: list[Message],
        response: Any,
        expected_output_tokens: int,
    ) -> tuple[int, int]:
        extracted_in: int | None = None
        extracted_out: int | None = None
        if self._usage_extractor is not None:
            extracted_in, extracted_out = self._usage_extractor(response)
        input_tokens = (
            extracted_in
            if extracted_in is not None
            else self._count_input(model, messages)
        )
        output_tokens = (
            extracted_out if extracted_out is not None else expected_output_tokens
        )
        return input_tokens, output_tokens

    def _count_input(self, model: str, messages: list[Message]) -> int:
        counter = self._counter
        if counter is None:
            counter = (
                self._optimizer.counter
                if self._optimizer is not None
                else get_counter(model)
            )
        return counter.count_messages(messages)

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        if self._estimator is None:
            return 0.0
        try:
            estimate = self._estimator.estimate(
                model, input_tokens=input_tokens, output_tokens=output_tokens
            )
        except UnknownPricingError:
            return 0.0
        return estimate.total_cost

    def _estimate_saved_cost(self, model: str, tokens_saved: int) -> float:
        if self._estimator is None or tokens_saved <= 0:
            return 0.0
        try:
            pricing = self._estimator.pricing_for(model)
        except UnknownPricingError:
            return 0.0
        return pricing.input_cost(tokens_saved)

    def _record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        cost: float,
        cache_hit: bool | None,
        success: bool,
        cost_saved: float = 0.0,
    ) -> None:
        if self._tracker is None:
            return
        self._tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            tokens_saved=tokens_saved,
            cost_saved=cost_saved,
            cache_hit=cache_hit,
            success=success,
        )
