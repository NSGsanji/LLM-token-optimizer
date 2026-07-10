"""Duplicate-message removal for conversation context."""

from __future__ import annotations

from collections.abc import Sequence

from ..budgeting.strategies import BudgetStrategy, StrategyOutcome
from ..tokenization.base import TokenCounter
from ..tokenization.types import Message


def _dedup_key(message: Message, *, by_role: bool) -> tuple[str, str]:
    """Build the identity key used to detect duplicate messages."""
    content = message.get("content", "")
    role = message.get("role", "") if by_role else ""
    return (role, content)


class DeduplicateStrategy(BudgetStrategy):
    """Remove repeated messages with identical content.

    Duplicated context — repeated system reminders, re-pasted documents, or
    echoed tool output — wastes tokens without adding information. This strategy
    removes exact duplicates while preserving conversation order.

    The token budget is ignored: deduplication is unconditional and safe, so it
    composes well as a first step ahead of budget-driven strategies (see
    :class:`~smarttokenoptimizer.budgeting.CompositeStrategy`).

    Args:
        by_role: When ``True`` (default), two messages are duplicates only if
            both their role and content match. When ``False``, identical content
            is treated as a duplicate regardless of role.
        keep: Which occurrence to retain — ``"first"`` (default) removes later
            duplicates, ``"last"`` removes earlier ones.
        protected_roles: Roles that are never removed even when duplicated.
            Defaults to an empty set, since removing exact duplicates is safe.

    Raises:
        ValueError: If ``keep`` is not ``"first"`` or ``"last"``.

    Example:
        >>> strategy = DeduplicateStrategy()
        >>> messages = [
        ...     {"role": "user", "content": "hi"},
        ...     {"role": "user", "content": "hi"},
        ... ]
        >>> strategy.apply(messages, max_tokens=100, counter=counter)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        by_role: bool = True,
        keep: str = "first",
        protected_roles: frozenset[str] | set[str] | None = None,
    ) -> None:
        if keep not in ("first", "last"):
            raise ValueError("keep must be 'first' or 'last'")
        self._by_role = by_role
        self._keep = keep
        self._protected = frozenset(protected_roles or ())

    def apply(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        counter: TokenCounter,
    ) -> StrategyOutcome:
        """See :meth:`BudgetStrategy.apply`. Removes duplicate messages."""
        original = list(messages)
        # Walking newest-first and keeping the first-seen occurrence retains the
        # *last* copy; walking oldest-first retains the *first* copy.
        indexed = list(enumerate(original))
        order = indexed if self._keep == "first" else list(reversed(indexed))

        seen: set[tuple[str, str]] = set()
        removed: set[int] = set()
        for idx, message in order:
            if message.get("role", "") in self._protected:
                continue
            key = _dedup_key(message, by_role=self._by_role)
            if key in seen:
                removed.add(idx)
            else:
                seen.add(key)

        result = [m for i, m in indexed if i not in removed]
        dropped = len(removed)
        note = (
            f"removed {dropped} duplicate message(s)"
            if dropped
            else "no duplicates found"
        )
        return StrategyOutcome(messages=result, dropped=dropped, note=note)
