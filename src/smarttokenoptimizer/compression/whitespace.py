"""Rule-based whitespace compression.

Collapsing redundant whitespace is the safest, highest-value form of prompt
compression: it removes tokens spent on formatting without altering wording.
This is especially effective on pasted documents, logs and code output, which
often carry runs of spaces, trailing whitespace and large blank gaps.
"""

from __future__ import annotations

import re

from .base import TextCompressor

_HORIZONTAL_WS = re.compile(r"[ \t\f\v]+")
_TRAILING_WS = re.compile(r"[ \t\f\v]+(?=\n)")
_LEADING_WS = re.compile(r"(?<=\n)[ \t\f\v]+")


class WhitespaceCompressor(TextCompressor):
    """Collapse redundant whitespace while preserving wording.

    The transformations are applied in a fixed, deterministic order:

    1. Normalise line endings (``\\r\\n`` and ``\\r`` become ``\\n``).
    2. Optionally collapse runs of spaces/tabs into a single space.
    3. Optionally strip whitespace at the start/end of each line.
    4. Optionally collapse long runs of blank lines.
    5. Optionally strip leading/trailing whitespace of the whole string.

    Args:
        collapse_spaces: Collapse runs of horizontal whitespace (spaces, tabs)
            into a single space. Note this flattens indentation, so disable it
            for whitespace-significant content you must preserve verbatim.
            Defaults to ``True``.
        strip_lines: Remove leading and trailing horizontal whitespace on each
            line. Defaults to ``True``.
        max_consecutive_newlines: Maximum number of consecutive newlines to
            keep (``2`` allows a single blank line between paragraphs). ``0`` or
            negative disables blank-line collapsing. Defaults to ``2``.
        strip: Strip leading/trailing whitespace of the entire result. Defaults
            to ``True``.

    Example:
        >>> WhitespaceCompressor().compress("hello    world")
        'hello world'
    """

    def __init__(
        self,
        *,
        collapse_spaces: bool = True,
        strip_lines: bool = True,
        max_consecutive_newlines: int = 2,
        strip: bool = True,
    ) -> None:
        self._collapse_spaces = collapse_spaces
        self._strip_lines = strip_lines
        self._max_consecutive_newlines = max_consecutive_newlines
        self._strip = strip
        self._newline_re = (
            re.compile(rf"\n{{{max_consecutive_newlines + 1},}}")
            if max_consecutive_newlines > 0
            else None
        )

    def compress(self, text: str) -> str:
        """See :meth:`TextCompressor.compress`."""
        if not text:
            return ""

        # 1. Normalise line endings.
        result = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Collapse horizontal whitespace runs.
        if self._collapse_spaces:
            result = _HORIZONTAL_WS.sub(" ", result)

        # 3. Strip per-line leading/trailing horizontal whitespace.
        if self._strip_lines:
            result = _TRAILING_WS.sub("", result)
            result = _LEADING_WS.sub("", result)

        # 4. Collapse long runs of blank lines.
        if self._newline_re is not None:
            replacement = "\n" * self._max_consecutive_newlines
            result = self._newline_re.sub(replacement, result)

        # 5. Strip the whole string.
        if self._strip:
            result = result.strip()

        return result
