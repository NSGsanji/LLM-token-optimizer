"""Budget-fitting strategies.

A *strategy* takes a conversation and a token budget and returns a (possibly
reduced) conversation that fits, along with accounting about what it changed.
Strategies are composable building blocks; the :class:`SmartTokenOptimizer`
applies one (or, in future, a pipeline) to bring a conversation within budget.

The default :class:`DropOldestStrategy` preserves protected messages (system
prompts by default) and drops the oldest non-protected turns first — the most
common, least-surprising behaviour for chat applications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..tokenization.base import TokenCounter
from ..tokenization.types import Message


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    """Internal result of applying a single strategy.

    Attributes:
        messages: The reduced messages.
        dropped: Number of messages removed entirely.
        truncated: Number of messages whose content was shortened.
        note: Human-readable description of what the strategy did.
    """

    messages: list[Message]
    dropped: int = 0
    truncated: int = 0
    note: str = ""


class BudgetStrategy(ABC):
    """Interface for strategies that fit a conversation into a token budget."""

    @abstractmethod
    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """Reduce ``messages`` to fit within ``max_tokens``.

        Args:
            messages: The conversation to fit, oldest first.
            max_tokens: The token budget to target.
            counter: The token counter used to measure messages.

        Returns:
            A :class:`StrategyOutcome` with the reduced messages and accounting.
            Implementations should get as close to the budget as possible but
            must never drop protected messages.
        """
        raise NotImplementedError


def _is_protected(message: Message, protected_roles: frozenset[str]) -> bool:
    return message.get("role", "") in protected_roles


class DropOldestStrategy(BudgetStrategy):
    """Drop the oldest non-protected messages until the budget is met.

    Protected messages (system prompts by default) are always retained, in
    order. Remaining messages are kept newest-first, so the most recent context
    survives — the standard behaviour for chat assistants.

    Args:
        protected_roles: Roles that must never be dropped. Defaults to
            ``{"system"}``.
        keep_last: Always retain at least this many of the most recent
            non-protected messages when possible, even under pressure. Defaults
            to ``0``.
    """

    def __init__(
        self,
        *,
        protected_roles: frozenset[str] | set[str] | None = None,
        keep_last: int = 0,
    ) -> None:
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        self._protected = frozenset(
            protected_roles if protected_roles is not None else {"system"}
        )
        self._keep_last = keep_last

    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """See :meth:`BudgetStrategy.apply`."""
        original = list(messages)
        if counter.count_messages(original) <= max_tokens:
            return StrategyOutcome(messages=original, note="already within budget")

        protected_flags = [_is_protected(m, self._protected) for m in original]
        n = len(original)

        # Indices of non-protected messages, oldest first — these are the
        # candidates for removal.
        droppable = [i for i in range(n) if not protected_flags[i]]
        # Protect the most recent `keep_last` non-protected messages from
        # removal when possible.
        if self._keep_last:
            droppable = droppable[: max(0, len(droppable) - self._keep_last)]

        keep = [True] * n
        dropped = 0
        for idx in droppable:
            current = [original[i] for i in range(n) if keep[i]]
            if counter.count_messages(current) <= max_tokens:
                break
            keep[idx] = False
            dropped += 1

        result = [original[i] for i in range(n) if keep[i]]
        note = (
            f"dropped {dropped} oldest message(s)" if dropped else "no messages dropped"
        )
        return StrategyOutcome(messages=result, dropped=dropped, note=note)


class CompositeStrategy(BudgetStrategy):
    """Apply several strategies in sequence, threading messages through each.

    Each strategy receives the output of the previous one, so transforms can be
    layered — for example: deduplicate, then apply a sliding window, then drop
    the oldest remaining turns to meet the budget. Accounting is aggregated
    across all steps.

    Args:
        strategies: The strategies to apply, in order. Must be non-empty.

    Raises:
        ValueError: If ``strategies`` is empty.

    Example:
        >>> from smarttokenoptimizer.budgeting import (
        ...     CompositeStrategy,
        ...     DropOldestStrategy,
        ... )
        >>> from smarttokenoptimizer.context import DeduplicateStrategy
        >>> strategy = CompositeStrategy(
        ...     DeduplicateStrategy(),
        ...     DropOldestStrategy(),
        ... )
    """

    def __init__(self, *strategies: BudgetStrategy) -> None:
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one strategy")
        self._strategies = strategies

    @property
    def strategies(self) -> tuple[BudgetStrategy, ...]:
        """The ordered strategies this composite applies."""
        return self._strategies

    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """See :meth:`BudgetStrategy.apply`. Applies each strategy in turn."""
        current: list[Message] = list(messages)
        total_dropped = 0
        total_truncated = 0
        notes: list[str] = []
        for strategy in self._strategies:
            outcome = strategy.apply(current, max_tokens=max_tokens, counter=counter)
            current = outcome.messages
            total_dropped += outcome.dropped
            total_truncated += outcome.truncated
            if outcome.note:
                notes.append(f"{type(strategy).__name__}: {outcome.note}")
        return StrategyOutcome(
            messages=current,
            dropped=total_dropped,
            truncated=total_truncated,
            note="; ".join(notes),
        )
