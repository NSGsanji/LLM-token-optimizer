"""Prompt compression: shrink message text without changing its meaning.

Rule-based, dependency-free compressors reduce the tokens spent on formatting
and redundant whitespace. They can be used directly on strings, composed with
:class:`CompositeCompressor`, or plugged into the budgeting pipeline through
:class:`CompressionStrategy`.

Example:
    >>> from smarttokenoptimizer.compression import WhitespaceCompressor
    >>> WhitespaceCompressor().compress("too    many     spaces")
    'too many spaces'
"""

from __future__ import annotations

from .base import TextCompressor
from .pipeline import CompositeCompressor
from .strategy import CompressionStrategy
from .whitespace import WhitespaceCompressor

__all__ = [
    "CompositeCompressor",
    "CompressionStrategy",
    "TextCompressor",
    "WhitespaceCompressor",
]
