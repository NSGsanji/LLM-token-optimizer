"""Fast, dependency-free heuristic token estimation.

This module provides :class:`HeuristicTokenCounter`, an approximate token
counter that needs no external tokenizer. It is designed to be:

- **Fast** — a single regex pass plus cheap arithmetic, suitable for hot paths
  and large batches.
- **Reasonably accurate** — typically within ~10-20% of real BPE tokenizers on
  natural-language English, which is good enough for budgeting decisions.
- **Deterministic** — the same input always yields the same estimate.

For exact counts, use the optional tiktoken-backed counter instead. The
heuristic is a safe default: it never requires network access or model
downloads and degrades predictably on unusual input.
"""

from __future__ import annotations

import re

from .base import TokenCounter

# Approximates the pre-tokenization step of GPT-style BPE tokenizers: contiguous
# runs of letters, of digits, or of individual punctuation/symbol characters.
# Leading whitespace is intentionally excluded from the captured group; GPT
# tokenizers fold a single leading space into the following token, which our
# per-chunk length model already accounts for on average.
_CHUNK_RE = re.compile(
    r"""
    (?:
        [^\W\d_]+      # a run of unicode letters
      | \d+            # a run of digits
      | [^\w\s]        # a single punctuation / symbol character
    )
    """,
    re.VERBOSE,
)

# Average characters per BPE token within a run of letters. Calibrated so that
# common English words (<= ~6 letters) map to a single token while longer words
# split into proportionally more sub-word tokens.
_CHARS_PER_TOKEN = 4.0

# Digit runs tokenize more densely (GPT BPE emits ~1 token per 2 digits).
_CHARS_PER_DIGIT_TOKEN = 2.0


class HeuristicTokenCounter(TokenCounter):
    """Approximate token counter requiring no external dependencies.

    The estimate is produced by pre-tokenizing ``text`` into word, number and
    symbol chunks (mirroring GPT-style tokenizers) and then estimating how many
    sub-word tokens each chunk contributes based on its length.

    Example:
        >>> counter = HeuristicTokenCounter()
        >>> counter.count_text("Hello, world!")
        4
    """

    exact = False

    def count_text(self, text: str) -> int:
        """Estimate the number of tokens in ``text``.

        Args:
            text: The raw string to estimate.

        Returns:
            A non-negative token estimate. Empty or whitespace-only input
            returns ``0``.
        """
        if not text:
            return 0

        total = 0
        for match in _CHUNK_RE.finditer(text):
            chunk = match.group()
            first = chunk[0]
            if first.isdigit():
                total += _round_half_up(len(chunk) / _CHARS_PER_DIGIT_TOKEN)
            elif first.isalpha():
                total += max(1, _round_half_up(len(chunk) / _CHARS_PER_TOKEN))
            else:
                # Single punctuation / symbol character: almost always one token.
                total += 1
        return total


def _round_half_up(value: float) -> int:
    """Round ``value`` to the nearest integer, rounding halves upward.

    Unlike :func:`round`, this avoids banker's rounding so estimates are
    predictable and monotonic in the input length.
    """
    return int(value + 0.5)
