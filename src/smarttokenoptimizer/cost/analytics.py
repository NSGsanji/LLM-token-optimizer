"""Aggregate usage analytics: tokens, cost, savings, cache and success rates."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshot:
    """An immutable point-in-time view of aggregated usage statistics.

    Attributes:
        requests: Total number of recorded requests.
        input_tokens: Total prompt tokens across all requests.
        output_tokens: Total completion tokens across all requests.
        cost: Total estimated cost across all requests.
        tokens_saved: Total tokens saved by optimization.
        cost_saved: Total cost saved by optimization.
        cache_hits: Number of requests served from cache.
        cache_misses: Number of requests that missed the cache.
        errors: Number of failed requests.
        currency: ISO currency code for monetary amounts.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    currency: str = "USD"

    @property
    def total_tokens(self) -> int:
        """Combined input and output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cache-tracked requests that hit, in ``[0, 1]``.

        Returns ``0.0`` when no cache lookups were recorded.
        """
        lookups = self.cache_hits + self.cache_misses
        return self.cache_hits / lookups if lookups else 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of requests that succeeded, in ``[0, 1]``.

        Returns ``1.0`` when no requests were recorded.
        """
        if self.requests == 0:
            return 1.0
        return (self.requests - self.errors) / self.requests

    @property
    def average_cost_per_request(self) -> float:
        """Mean cost per recorded request (``0.0`` when there are none)."""
        return self.cost / self.requests if self.requests else 0.0

    @property
    def savings_ratio(self) -> float:
        """Saved tokens as a fraction of what usage would have been, ``[0, 1]``.

        The pre-optimization total is approximated as ``total_tokens +
        tokens_saved``. Returns ``0.0`` when nothing was processed.
        """
        baseline = self.total_tokens + self.tokens_saved
        return self.tokens_saved / baseline if baseline else 0.0


@dataclass
class _MutableTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    per_model: dict[str, int] = field(default_factory=dict)


class UsageTracker:
    """Thread-safe accumulator for usage and savings analytics.

    Record one entry per request with :meth:`record`; read a consistent
    aggregate at any time with :meth:`snapshot`. All mutating operations are
    guarded by a lock, so a single tracker can be shared across threads.

    Args:
        currency: ISO currency code reported in snapshots. Defaults to ``USD``.

    Example:
        >>> tracker = UsageTracker()
        >>> tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=200,
        ...                 cost=0.0045)
        >>> tracker.snapshot().requests
        1
    """

    def __init__(self, *, currency: str = "USD") -> None:
        self._lock = threading.Lock()
        self._totals = _MutableTotals()
        self._currency = currency

    def record(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        tokens_saved: int = 0,
        cost_saved: float = 0.0,
        cache_hit: bool | None = None,
        success: bool = True,
    ) -> None:
        """Record a single request's usage.

        Args:
            model: Optional model identifier; per-model request counts are kept.
            input_tokens: Prompt tokens used. Must be ``>= 0``.
            output_tokens: Completion tokens used. Must be ``>= 0``.
            cost: Estimated cost of the request.
            tokens_saved: Tokens saved by optimization for this request.
            cost_saved: Cost saved by optimization for this request.
            cache_hit: ``True`` if served from cache, ``False`` if it missed,
                ``None`` if caching does not apply to this request.
            success: Whether the request succeeded. Failures increment the
                error count used by :attr:`AnalyticsSnapshot.success_rate`.

        Raises:
            ValueError: If any token/cost amount is negative.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if tokens_saved < 0:
            raise ValueError("tokens_saved must be non-negative")

        with self._lock:
            t = self._totals
            t.requests += 1
            t.input_tokens += input_tokens
            t.output_tokens += output_tokens
            t.cost += cost
            t.tokens_saved += tokens_saved
            t.cost_saved += cost_saved
            if cache_hit is True:
                t.cache_hits += 1
            elif cache_hit is False:
                t.cache_misses += 1
            if not success:
                t.errors += 1
            if model is not None:
                t.per_model[model] = t.per_model.get(model, 0) + 1

    def snapshot(self) -> AnalyticsSnapshot:
        """Return an immutable snapshot of the current totals."""
        with self._lock:
            t = self._totals
            return AnalyticsSnapshot(
                requests=t.requests,
                input_tokens=t.input_tokens,
                output_tokens=t.output_tokens,
                cost=t.cost,
                tokens_saved=t.tokens_saved,
                cost_saved=t.cost_saved,
                cache_hits=t.cache_hits,
                cache_misses=t.cache_misses,
                errors=t.errors,
                currency=self._currency,
            )

    def usage_by_model(self) -> dict[str, int]:
        """Return a copy of the per-model request counts."""
        with self._lock:
            return dict(self._totals.per_model)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self._totals = _MutableTotals()
