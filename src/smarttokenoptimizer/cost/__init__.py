"""Cost estimation and usage analytics.

This subpackage turns token counts into money and tracks usage over time:

- :class:`ModelPricing` and :func:`get_pricing` — per-model token prices.
- :class:`CostEstimator` — compute request cost from token counts or messages.
- :class:`UsageTracker` — thread-safe accumulator for tokens, cost, savings,
  cache-hit rate and success rate, read via :class:`AnalyticsSnapshot`.

Example:
    >>> from smarttokenoptimizer.cost import CostEstimator
    >>> estimate = CostEstimator().estimate("gpt-4o", input_tokens=1000)
    >>> round(estimate.total_cost, 4)
    0.0025
"""

from __future__ import annotations

from .analytics import AnalyticsSnapshot, UsageTracker
from .errors import CostError, UnknownPricingError
from .estimator import CostEstimate, CostEstimator
from .pricing import (
    ModelPricing,
    clear_custom_pricing,
    get_pricing,
    register_pricing,
)

__all__ = [
    "AnalyticsSnapshot",
    "CostError",
    "CostEstimate",
    "CostEstimator",
    "ModelPricing",
    "UnknownPricingError",
    "UsageTracker",
    "clear_custom_pricing",
    "get_pricing",
    "register_pricing",
]
