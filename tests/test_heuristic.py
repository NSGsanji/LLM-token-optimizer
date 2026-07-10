"""Tests for the dependency-free heuristic token counter."""

from __future__ import annotations

import pytest

from smarttokenoptimizer.tokenization import HeuristicTokenCounter


@pytest.fixture
def counter() -> HeuristicTokenCounter:
    return HeuristicTokenCounter()


class TestCountText:
    def test_empty_string_is_zero(self, counter: HeuristicTokenCounter) -> None:
        assert counter.count_text("") == 0

    def test_whitespace_only_is_zero(self, counter: HeuristicTokenCounter) -> None:
        assert counter.count_text("   \n\t  ") == 0

    def test_single_word(self, counter: HeuristicTokenCounter) -> None:
        # A short common word is a single token.
        assert counter.count_text("hello") == 1

    def test_hello_world_matches_reference(
        self, counter: HeuristicTokenCounter
    ) -> None:
        # Reference BPE tokenizers count "Hello, world!" as 4 tokens.
        assert counter.count_text("Hello, world!") == 4

    def test_punctuation_counts_individually(
        self, counter: HeuristicTokenCounter
    ) -> None:
        assert counter.count_text("!?.") == 3

    def test_long_word_splits_into_subwords(
        self, counter: HeuristicTokenCounter
    ) -> None:
        # A long word must produce more than one token.
        assert counter.count_text("antidisestablishmentarianism") > 1

    def test_digits_are_counted(self, counter: HeuristicTokenCounter) -> None:
        assert counter.count_text("12345") >= 1

    def test_result_is_non_negative(self, counter: HeuristicTokenCounter) -> None:
        assert counter.count_text("anything at all") >= 0

    def test_deterministic(self, counter: HeuristicTokenCounter) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert counter.count_text(text) == counter.count_text(text)

    def test_monotonic_in_length(self, counter: HeuristicTokenCounter) -> None:
        short = counter.count_text("word")
        longer = counter.count_text("word word word")
        assert longer > short

    def test_unicode_letters(self, counter: HeuristicTokenCounter) -> None:
        # Non-ASCII letters are treated as a word run, not punctuation.
        assert counter.count_text("naïve café") >= 2

    def test_callable_alias(self, counter: HeuristicTokenCounter) -> None:
        assert counter("hello") == counter.count_text("hello")

    def test_is_not_exact(self, counter: HeuristicTokenCounter) -> None:
        assert counter.exact is False


class TestAccuracyEnvelope:
    """The heuristic should stay within a sane band of a real tokenizer.

    We do not require exactness, only that estimates are in a useful range so
    budgeting decisions are safe.
    """

    @pytest.mark.parametrize(
        ("text", "low", "high"),
        [
            ("The quick brown fox jumps over the lazy dog.", 8, 14),
            ("SmartTokenOptimizer reduces token usage.", 6, 14),
            ("a " * 50, 40, 60),
        ],
    )
    def test_within_envelope(
        self,
        counter: HeuristicTokenCounter,
        text: str,
        low: int,
        high: int,
    ) -> None:
        assert low <= counter.count_text(text) <= high
