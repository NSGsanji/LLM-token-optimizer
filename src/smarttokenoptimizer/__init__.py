"""SmartTokenOptimizer.

Token optimization and AI credential management framework for LLM applications.

The public API is re-exported from this top-level package. During the ``0.x``
alpha the surface is still stabilising; see the project CHANGELOG for details.
"""

from __future__ import annotations

from .budgeting import (
    BudgetStrategy,
    CompositeStrategy,
    DropOldestStrategy,
    OptimizationResult,
    SmartTokenOptimizer,
)
from .cache import (
    CacheStats,
    MemoryCache,
    PromptCache,
    SQLiteCache,
)
from .compression import (
    CompositeCompressor,
    CompressionStrategy,
    TextCompressor,
    WhitespaceCompressor,
)
from .context import (
    DeduplicateStrategy,
    SlidingWindowStrategy,
)
from .cost import (
    AnalyticsSnapshot,
    CostEstimate,
    CostEstimator,
    ModelPricing,
    UsageTracker,
)
from .credentials import (
    Credential,
    CredentialHealth,
    CredentialPool,
    NoAvailableCredentialError,
    PriorityStrategy,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
)
from .middleware import (
    CallRequest,
    CompletionResult,
    OptimizingPipeline,
)
from .routing import (
    CheapestPolicy,
    LowestLatencyPolicy,
    NoAvailableProviderError,
    Provider,
    Route,
    Router,
)
from .tokenization import (
    HeuristicTokenCounter,
    Message,
    MessageOverhead,
    TiktokenCounter,
    TokenCounter,
    get_counter,
)

__all__ = [
    "AnalyticsSnapshot",
    "BudgetStrategy",
    "CacheStats",
    "CallRequest",
    "CheapestPolicy",
    "CompletionResult",
    "CompositeCompressor",
    "CompositeStrategy",
    "CompressionStrategy",
    "CostEstimate",
    "CostEstimator",
    "Credential",
    "CredentialHealth",
    "CredentialPool",
    "DeduplicateStrategy",
    "DropOldestStrategy",
    "HeuristicTokenCounter",
    "LowestLatencyPolicy",
    "MemoryCache",
    "Message",
    "MessageOverhead",
    "ModelPricing",
    "NoAvailableCredentialError",
    "NoAvailableProviderError",
    "OptimizationResult",
    "OptimizingPipeline",
    "PriorityStrategy",
    "PromptCache",
    "Provider",
    "RoundRobinStrategy",
    "Route",
    "Router",
    "SQLiteCache",
    "SlidingWindowStrategy",
    "SmartTokenOptimizer",
    "TextCompressor",
    "TiktokenCounter",
    "TokenCounter",
    "UsageTracker",
    "WeightedRoundRobinStrategy",
    "WhitespaceCompressor",
    "__version__",
    "get_counter",
]

__version__ = "0.3.0"
