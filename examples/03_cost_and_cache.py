"""Example: estimating cost and caching responses with analytics.

Run with:  python examples/03_cost_and_cache.py
"""

from __future__ import annotations

from smarttokenoptimizer import Message
from smarttokenoptimizer.cache import MemoryCache
from smarttokenoptimizer.cost import CostEstimator, UsageTracker


def main() -> None:
    estimator = CostEstimator()
    tracker = UsageTracker()
    cache = MemoryCache(max_size=1000, default_ttl=3600)

    messages: list[Message] = [
        {"role": "user", "content": "Summarise the theory of relativity."},
    ]

    # First request: a cache miss, so we "call the model" and cache the answer.
    cached = cache.get_response("gpt-4o", messages)
    if cached is None:
        response = {"text": "Relativity relates space, time, mass and energy."}
        estimate = estimator.estimate("gpt-4o", input_tokens=1200, output_tokens=250)
        cache.set_response("gpt-4o", messages, response)
        tracker.record(
            model="gpt-4o",
            input_tokens=1200,
            output_tokens=250,
            cost=estimate.total_cost,
            cache_hit=False,
        )
        print(f"MISS -> called model, cost ${estimate.total_cost:.4f}")

    # Second identical request: a cache hit, so no spend.
    cached = cache.get_response("gpt-4o", messages)
    if cached is not None:
        tracker.record(model="gpt-4o", cache_hit=True)
        print("HIT  -> served from cache, cost $0.0000")

    snap = tracker.snapshot()
    print(f"Requests: {snap.requests}  cache hit rate: {snap.cache_hit_rate:.0%}")
    print(
        f"Total spend: ${snap.cost:.4f}  avg/request: "
        f"${snap.average_cost_per_request:.4f}"
    )


if __name__ == "__main__":
    main()
