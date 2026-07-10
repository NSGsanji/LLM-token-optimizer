"""Result types returned by budgeting operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tokenization.types import Message


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The outcome of fitting a conversation into a token budget.

    Instances are returned by
    :meth:`~smarttokenoptimizer.budgeting.optimizer.SmartTokenOptimizer.optimize`
    when detailed reporting is requested. The optimized messages plus rich
    accounting make it easy to log savings and audit what was changed.

    Attributes:
        messages: The resulting messages that fit within the budget.
        original_tokens: Token count of the input conversation.
        optimized_tokens: Token count of :attr:`messages`.
        max_tokens: The budget the optimizer targeted.
        dropped_messages: Number of messages removed entirely.
        truncated_messages: Number of messages whose content was shortened.
        within_budget: Whether :attr:`optimized_tokens` fits within
            :attr:`max_tokens`. This can be ``False`` only when the budget is
            too small to hold even the protected (e.g. system) messages.
    """

    messages: list[Message]
    original_tokens: int
    optimized_tokens: int
    max_tokens: int
    dropped_messages: int = 0
    truncated_messages: int = 0
    within_budget: bool = True
    #: Free-form notes describing the strategies applied, for logging/debugging.
    notes: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """Tokens removed relative to the original conversation (``>= 0``)."""
        return max(0, self.original_tokens - self.optimized_tokens)

    @property
    def compression_ratio(self) -> float:
        """Fraction of the original tokens that were removed, in ``[0, 1]``.

        Returns ``0.0`` when the original conversation was empty.
        """
        if self.original_tokens <= 0:
            return 0.0
        return self.tokens_saved / self.original_tokens

    @property
    def changed(self) -> bool:
        """Whether the optimizer altered the conversation at all."""
        return self.dropped_messages > 0 or self.truncated_messages > 0
