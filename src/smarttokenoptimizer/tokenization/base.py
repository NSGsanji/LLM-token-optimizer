"""Abstract base class for token counters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .types import DEFAULT_OVERHEAD, Message, MessageOverhead


class TokenCounter(ABC):
    """Common interface for all token counters.

    A token counter turns text (or chat messages) into an integer token count.
    Concrete implementations may be *exact* (backed by the model's real
    tokenizer) or *approximate* (fast heuristics). Both honour the same
    interface so callers can swap between accuracy and speed transparently.

    Subclasses only need to implement :meth:`count_text`. Message counting is
    provided here in terms of ``count_text`` plus a configurable
    :class:`~smarttokenoptimizer.tokenization.types.MessageOverhead`.
    """

    #: Whether this counter reproduces the model's exact tokenization.
    exact: bool = False

    def __init__(self, *, overhead: MessageOverhead = DEFAULT_OVERHEAD) -> None:
        self._overhead = overhead

    @property
    def overhead(self) -> MessageOverhead:
        """The per-message overhead accounting used by this counter."""
        return self._overhead

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Return the number of tokens in ``text``.

        Args:
            text: The raw string to tokenize.

        Returns:
            The token count, always ``>= 0``.
        """
        raise NotImplementedError

    def count_message(self, message: Message) -> int:
        """Return the token count of a single chat message, including overhead.

        Args:
            message: An OpenAI-compatible chat message. Every string value in
                the mapping contributes tokens; the structural per-message
                overhead is added on top.

        Returns:
            The token count for the message.
        """
        total = self._overhead.tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                # Non-string values (rare, e.g. structured content) are
                # stringified so their textual payload is still accounted for.
                value = str(value)
            total += self.count_text(value)
            if key == "name":
                total += self._overhead.tokens_per_name
        return total

    def count_messages(self, messages: Iterable[Message]) -> int:
        """Return the total token count for a sequence of chat messages.

        This includes each message's overhead plus a single reply-priming
        allowance, matching how chat completion endpoints bill a request.

        Args:
            messages: An iterable of OpenAI-compatible chat messages.

        Returns:
            The total prompt token count for the conversation.
        """
        total = sum(self.count_message(message) for message in messages)
        return total + self._overhead.reply_priming

    def __call__(self, text: str) -> int:
        """Alias for :meth:`count_text` so counters are usable as callables."""
        return self.count_text(text)
