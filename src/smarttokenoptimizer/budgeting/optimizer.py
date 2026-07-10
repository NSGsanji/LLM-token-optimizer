"""The high-level :class:`SmartTokenOptimizer` budgeting entry point."""

from __future__ import annotations

from collections.abc import Sequence

from ..tokenization.base import TokenCounter
from ..tokenization.registry import get_counter
from ..tokenization.types import Message
from .result import OptimizationResult
from .strategies import BudgetStrategy, DropOldestStrategy


class SmartTokenOptimizer:
    """Fit conversations inside a token budget.

    This is the headline entry point of the library. Construct it with a token
    budget and call :meth:`optimize` to reduce a conversation until it fits.

    Args:
        max_tokens: The maximum number of prompt tokens the optimized
            conversation may use. Must be positive.
        model: Optional model identifier used to select an appropriate token
            counter (e.g. ``"gpt-4o"``). Ignored when ``counter`` is given.
        counter: An explicit :class:`TokenCounter` to measure messages. When
            omitted, one is chosen via
            :func:`~smarttokenoptimizer.tokenization.registry.get_counter`.
        strategy: The :class:`BudgetStrategy` used to reduce conversations.
            Defaults to :class:`DropOldestStrategy` (preserve system prompts,
            drop oldest turns first).

    Example:
        >>> optimizer = SmartTokenOptimizer(max_tokens=16000)
        >>> messages = optimizer.optimize(messages)  # doctest: +SKIP
    """

    def __init__(
        self,
        max_tokens: int,
        *,
        model: str | None = None,
        counter: TokenCounter | None = None,
        strategy: BudgetStrategy | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._max_tokens = max_tokens
        self._counter = counter if counter is not None else get_counter(model)
        self._strategy = strategy if strategy is not None else DropOldestStrategy()

    @property
    def max_tokens(self) -> int:
        """The configured token budget."""
        return self._max_tokens

    @property
    def counter(self) -> TokenCounter:
        """The token counter used to measure conversations."""
        return self._counter

    def count(self, messages: Sequence[Message]) -> int:
        """Return the prompt-token count of ``messages`` (including overhead)."""
        return self._counter.count_messages(messages)

    def fits(self, messages: Sequence[Message]) -> bool:
        """Return whether ``messages`` already fit within the budget."""
        return self.count(messages) <= self._max_tokens

    def optimize(self, messages: Sequence[Message]) -> list[Message]:
        """Reduce ``messages`` to fit within the budget.

        Args:
            messages: The conversation to optimize, oldest message first.

        Returns:
            A new list of messages that fits within :attr:`max_tokens` when
            possible. The input is never mutated. If the conversation already
            fits, an equivalent copy is returned unchanged.

        Note:
            For detailed accounting (tokens saved, messages dropped, whether the
            budget was met), use :meth:`optimize_detailed` instead.
        """
        return self.optimize_detailed(messages).messages

    def optimize_detailed(self, messages: Sequence[Message]) -> OptimizationResult:
        """Like :meth:`optimize`, but return a full :class:`OptimizationResult`.

        Args:
            messages: The conversation to optimize, oldest message first.

        Returns:
            An :class:`OptimizationResult` describing the optimized messages and
            what was changed to reach the budget.

        Note:
            The configured strategy is always applied, even when the
            conversation already fits the budget. This lets budget-independent
            strategies (e.g. deduplication or a sliding window) still take
            effect. Budget-driven strategies such as
            :class:`~smarttokenoptimizer.budgeting.DropOldestStrategy` are
            idempotent when already within budget and return the conversation
            unchanged.
        """
        original = list(messages)
        original_tokens = self._counter.count_messages(original)

        outcome = self._strategy.apply(
            original, max_tokens=self._max_tokens, counter=self._counter
        )
        optimized_tokens = self._counter.count_messages(outcome.messages)

        return OptimizationResult(
            messages=outcome.messages,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            max_tokens=self._max_tokens,
            dropped_messages=outcome.dropped,
            truncated_messages=outcome.truncated,
            within_budget=optimized_tokens <= self._max_tokens,
            notes=[outcome.note] if outcome.note else [],
        )
