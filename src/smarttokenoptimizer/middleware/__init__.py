"""Framework-agnostic middleware tying the whole pipeline together.

:class:`OptimizingPipeline` orchestrates optimization, caching, provider
routing, cost estimation and analytics around a single injected ``call_fn``,
depending on no provider SDK. Framework integrations (OpenAI SDK, Anthropic
SDK, FastAPI, LiteLLM, LangChain) are thin adapters that supply ``call_fn``.

Example:
    >>> from smarttokenoptimizer.middleware import OptimizingPipeline
    >>> def call(request):
    ...     return {"text": "hi"}  # wrap your provider SDK here
    >>> pipeline = OptimizingPipeline(call, model="gpt-4o")
    >>> result = pipeline.complete([{"role": "user", "content": "hello"}])
    >>> result.response
    {'text': 'hi'}
"""

from __future__ import annotations

from .pipeline import (
    CallRequest,
    CompletionResult,
    OptimizingPipeline,
    UsageExtractor,
)

__all__ = [
    "CallRequest",
    "CompletionResult",
    "OptimizingPipeline",
    "UsageExtractor",
]
