"""Abstract base class for text compressors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextCompressor(ABC):
    """Common interface for rule-based text compressors.

    A compressor takes a string and returns a shorter, semantically-equivalent
    (or near-equivalent) string, reducing token usage. Implementations are
    expected to be deterministic and dependency-free.

    Subclasses implement :meth:`compress`. Instances are also callable.
    """

    @abstractmethod
    def compress(self, text: str) -> str:
        """Return a compressed version of ``text``.

        Args:
            text: The string to compress.

        Returns:
            The compressed string. Implementations must return ``""`` unchanged
            for empty input and must never raise on ordinary text.
        """
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        """Alias for :meth:`compress` so compressors are usable as callables."""
        return self.compress(text)
