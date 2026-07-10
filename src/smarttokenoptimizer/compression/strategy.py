"""Bridge text compression into the budgeting strategy pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from ..budgeting.strategies import BudgetStrategy, StrategyOutcome
from ..tokenization.base import TokenCounter
from ..tokenization.types import Message
from .base import TextCompressor
from .whitespace import WhitespaceCompressor


class CompressionStrategy(BudgetStrategy):
    """Compress each message's textual content in place.

    This adapts a :class:`TextCompressor` to the
    :class:`~smarttokenoptimizer.budgeting.BudgetStrategy` interface so prompt
    compression can be layered with deduplication, windowing and budget-driven
    trimming via
    :class:`~smarttokenoptimizer.budgeting.CompositeStrategy`.

    Compression is applied unconditionally (independent of the token budget). No
    messages are dropped; only their ``content`` is shortened. The number of
    messages whose content changed is reported as ``truncated``.

    Args:
        compressor: The compressor to apply to message content. Defaults to a
            :class:`WhitespaceCompressor`.
        roles: If given, only compress messages whose ``role`` is in this set.
            When ``None`` (default), all messages are compressed.

    Example:
        >>> strategy = CompressionStrategy()
    """

    def __init__(
        self,
        compressor: TextCompressor | None = None,
        *,
        roles: frozenset[str] | set[str] | None = None,
    ) -> None:
        self._compressor = (
            compressor if compressor is not None else WhitespaceCompressor()
        )
        self._roles = frozenset(roles) if roles is not None else None

    @property
    def compressor(self) -> TextCompressor:
        """The compressor applied to message content."""
        return self._compressor

    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """See :meth:`BudgetStrategy.apply`. Compresses message content."""
        result: list[Message] = []
        changed = 0
        for message in messages:
            content = message.get("content")
            role = message.get("role", "")
            should = self._roles is None or role in self._roles
            if should and isinstance(content, str) and content:
                compressed = self._compressor.compress(content)
                if compressed != content:
                    new_message: Message = dict(message)  # type: ignore[assignment]
                    new_message["content"] = compressed
                    result.append(new_message)
                    changed += 1
                    continue
            result.append(message)

        note = (
            f"compressed {changed} message(s)" if changed else "no content compressed"
        )
        return StrategyOutcome(messages=result, truncated=changed, note=note)
