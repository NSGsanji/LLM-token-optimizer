"""Example: the end-to-end OptimizingPipeline tying everything together.

Run with:  python examples/05_full_pipeline.py

Wires optimization + caching + routing + cost + analytics around a single
``call_fn``. The ``call_fn`` here is a stand-in for a real provider SDK, so the
example runs offline.
"""

from __future__ import annotations

from typing import Any

from smarttokenoptimizer import Message, SmartTokenOptimizer
from smarttokenoptimizer.cache import MemoryCache
from smarttokenoptimizer.cost import CostEstimator, UsageTracker
from smarttokenoptimizer.credentials import CredentialPool
from smarttokenoptimizer.middleware import CallRequest, OptimizingPipeline
from smarttokenoptimizer.routing import Provider, Router


def make_call_fn() -> Any:
    """Return a fake provider call that reports token usage like a real SDK."""

    def call_fn(request: CallRequest) -> dict[str, Any]:
        # A real adapter would call e.g. the OpenAI SDK with request.api_key.
        return {
            "text": f"Answer from {request.provider}.",
            "usage": {"input": 30 * len(request.messages), "output": 120},
        }

    return call_fn


def main() -> None:
    provider = Provider("openai", pool=CredentialPool(), models=["gpt-4o"])
    provider.pool.add_key("sk-openai-example", provider="openai")

    tracker = UsageTracker()
    pipeline = OptimizingPipeline(
        make_call_fn(),
        model="gpt-4o",
        optimizer=SmartTokenOptimizer(max_tokens=4000, model="gpt-4o"),
        cache=MemoryCache(default_ttl=3600),
        router=Router([provider]),
        tracker=tracker,
        estimator=CostEstimator(),
        usage_extractor=lambda r: (r["usage"]["input"], r["usage"]["output"]),
    )

    messages: list[Message] = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain vector databases in one sentence."},
    ]

    first = pipeline.complete(messages)
    print(
        f"1st call  cached={first.cached}  provider={first.provider}  "
        f"in={first.input_tokens} out={first.output_tokens}  "
        f"cost=${first.cost:.4f}"
    )

    second = pipeline.complete(messages)
    print(
        f"2nd call  cached={second.cached}  cost=${second.cost:.4f}  "
        f"(served from cache)"
    )

    snap = tracker.snapshot()
    print(
        f"\nTotals: requests={snap.requests}  "
        f"cache_hit_rate={snap.cache_hit_rate:.0%}  "
        f"spend=${snap.cost:.4f}  success_rate={snap.success_rate:.0%}"
    )


if __name__ == "__main__":
    main()
