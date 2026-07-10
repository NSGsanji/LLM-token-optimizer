"""Compose multiple text compressors into one."""

from __future__ import annotations

from .base import TextCompressor


class CompositeCompressor(TextCompressor):
    """Apply several compressors in sequence.

    The output of each compressor is fed into the next, so transformations
    layer cleanly (e.g. normalise whitespace, then strip boilerplate).

    Args:
        compressors: The compressors to apply, in order. Must be non-empty.

    Raises:
        ValueError: If ``compressors`` is empty.

    Example:
        >>> from smarttokenoptimizer.compression import WhitespaceCompressor
        >>> compressor = CompositeCompressor(WhitespaceCompressor())
        >>> compressor.compress("a    b")
        'a b'
    """

    def __init__(self, *compressors: TextCompressor) -> None:
        if not compressors:
            raise ValueError("CompositeCompressor requires at least one compressor")
        self._compressors = compressors

    @property
    def compressors(self) -> tuple[TextCompressor, ...]:
        """The ordered compressors this composite applies."""
        return self._compressors

    def compress(self, text: str) -> str:
        """See :meth:`TextCompressor.compress`. Applies each compressor in turn."""
        result = text
        for compressor in self._compressors:
            result = compressor.compress(result)
        return result
