"""Token budgeting: fit conversations inside a token budget.

This subpackage provides the headline :class:`SmartTokenOptimizer` entry point
plus the pluggable strategies it uses to reduce conversations.

Example:
    >>> from smarttokenoptimizer.budgeting import SmartTokenOptimizer
    >>> optimizer = SmartTokenOptimizer(max_tokens=16000)
    >>> fitted = optimizer.optimize(messages)  # doctest: +SKIP
"""

from __future__ import annotations

from .optimizer import SmartTokenOptimizer
from .result import OptimizationResult
from .strategies import (
    BudgetStrategy,
    CompositeStrategy,
    DropOldestStrategy,
    StrategyOutcome,
)

__all__ = [
    "BudgetStrategy",
    "CompositeStrategy",
    "DropOldestStrategy",
    "OptimizationResult",
    "SmartTokenOptimizer",
    "StrategyOutcome",
]
