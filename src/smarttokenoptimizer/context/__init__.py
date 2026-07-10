"""Context optimization: reduce conversation context before budgeting.

These strategies trim conversations independently of a token budget — removing
duplicated content and bounding history with a sliding window. They implement
the :class:`~smarttokenoptimizer.budgeting.BudgetStrategy` interface, so they can
be used directly or layered with budget-driven strategies via
:class:`~smarttokenoptimizer.budgeting.CompositeStrategy`.

Example:
    >>> from smarttokenoptimizer.budgeting import (
    ...     CompositeStrategy,
    ...     DropOldestStrategy,
    ...     SmartTokenOptimizer,
    ... )
    >>> from smarttokenoptimizer.context import (
    ...     DeduplicateStrategy,
    ...     SlidingWindowStrategy,
    ... )
    >>> optimizer = SmartTokenOptimizer(
    ...     max_tokens=8000,
    ...     strategy=CompositeStrategy(
    ...         DeduplicateStrategy(),
    ...         SlidingWindowStrategy(max_messages=20),
    ...         DropOldestStrategy(),
    ...     ),
    ... )
"""

from __future__ import annotations

from .dedup import DeduplicateStrategy
from .window import SlidingWindowStrategy

__all__ = [
    "DeduplicateStrategy",
    "SlidingWindowStrategy",
]
