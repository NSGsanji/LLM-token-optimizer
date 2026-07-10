"""Sliding-window context reduction: keep only the most recent turns."""

from __future__ import annotations

from collections.abc import Sequence

from ..budgeting.strategies import BudgetStrategy, StrategyOutcome
from ..tokenization.base import TokenCounter
from ..tokenization.types import Message


class SlidingWindowStrategy(BudgetStrategy):
    """Keep only the most recent ``max_messages`` non-protected messages.

    A sliding window is a simple, predictable way to bound conversation growth:
    protected messages (system prompts by default) are always retained, and of
    the remaining messages only the newest ``max_messages`` survive, in order.

    Unlike :class:`~smarttokenoptimizer.budgeting.DropOldestStrategy`, the window
    is defined by a *message count* rather than a token budget, so it applies
    unconditionally regardless of ``max_tokens``. It composes naturally before a
    token-driven strategy (see
    :class:`~smarttokenoptimizer.budgeting.CompositeStrategy`).

    Args:
        max_messages: Maximum number of non-protected messages to retain. Must
            be non-negative. ``0`` keeps only protected messages.
        protected_roles: Roles that are always retained. Defaults to
            ``{"system"}``.

    Raises:
        ValueError: If ``max_messages`` is negative.

    Example:
        >>> strategy = SlidingWindowStrategy(max_messages=10)
    """

    def __init__(
        self,
        max_messages: int,
        *,
        protected_roles: frozenset[str] | set[str] | None = None,
    ) -> None:
        if max_messages < 0:
            raise ValueError("max_messages must be >= 0")
        self._max_messages = max_messages
        self._protected = frozenset(
            protected_roles if protected_roles is not None else {"system"}
        )

    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """See :meth:`BudgetStrategy.apply`. Applies the sliding window."""
        original = list(messages)
        protected_flags = [m.get("role", "") in self._protected for m in original]
        non_protected = [i for i, p in enumerate(protected_flags) if not p]

        if len(non_protected) <= self._max_messages:
            return StrategyOutcome(messages=original, note="within window")

        # Keep the newest `max_messages` non-protected indices.
        keep_recent = (
            set(non_protected[len(non_protected) - self._max_messages :])
            if self._max_messages
            else set()
        )
        keep = [protected_flags[i] or i in keep_recent for i in range(len(original))]
        result = [m for i, m in enumerate(original) if keep[i]]
        dropped = len(original) - len(result)
        return StrategyOutcome(
            messages=result,
            dropped=dropped,
            note=f"kept newest {self._max_messages} message(s), dropped {dropped}",
        )
