"""Tests for the prompt compression module."""

from __future__ import annotations

import pytest

from smarttokenoptimizer.compression import (
    CompositeCompressor,
    CompressionStrategy,
    TextCompressor,
    WhitespaceCompressor,
)
from smarttokenoptimizer.tokenization import Message, MessageOverhead, TokenCounter


class WordCounter(TokenCounter):
    """One token per whitespace word, zero overhead."""

    def __init__(self) -> None:
        super().__init__(
            overhead=MessageOverhead(
                tokens_per_message=0, tokens_per_name=0, reply_priming=0
            )
        )

    def count_text(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


class TestWhitespaceCompressor:
    def test_empty_string(self) -> None:
        assert WhitespaceCompressor().compress("") == ""

    def test_collapses_spaces(self) -> None:
        assert WhitespaceCompressor().compress("a    b\tc") == "a b c"

    def test_strips_result(self) -> None:
        assert WhitespaceCompressor().compress("   hi   ") == "hi"

    def test_normalises_line_endings(self) -> None:
        assert WhitespaceCompressor().compress("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_blank_lines(self) -> None:
        assert (
            WhitespaceCompressor(max_consecutive_newlines=2).compress("a\n\n\n\n\nb")
            == "a\n\nb"
        )

    def test_strips_trailing_line_whitespace(self) -> None:
        result = WhitespaceCompressor(collapse_spaces=False).compress(
            "line1   \nline2\t\n"
        )
        assert result == "line1\nline2"

    def test_strips_leading_line_whitespace(self) -> None:
        result = WhitespaceCompressor(collapse_spaces=False).compress(
            "text\n    indented"
        )
        assert result == "text\nindented"

    def test_collapse_spaces_can_be_disabled(self) -> None:
        # With collapse disabled, interior runs are preserved (minus line strip).
        result = WhitespaceCompressor(
            collapse_spaces=False, strip_lines=False, strip=False
        ).compress("a    b")
        assert result == "a    b"

    def test_disabling_newline_collapse(self) -> None:
        result = WhitespaceCompressor(max_consecutive_newlines=0).compress("a\n\n\n\nb")
        assert result == "a\n\n\n\nb"

    def test_idempotent(self) -> None:
        compressor = WhitespaceCompressor()
        once = compressor.compress("a  b\n\n\n\nc   ")
        twice = compressor.compress(once)
        assert once == twice

    def test_never_grows(self) -> None:
        compressor = WhitespaceCompressor()
        for text in ["hello", "a  b  c", "x\n\n\n\ny", "   spaced   "]:
            assert len(compressor.compress(text)) <= len(text)

    def test_callable_alias(self) -> None:
        compressor = WhitespaceCompressor()
        assert compressor("a  b") == compressor.compress("a  b")


class TestCompositeCompressor:
    def test_requires_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeCompressor()

    def test_applies_in_sequence(self) -> None:
        class Suffix(TextCompressor):
            def __init__(self, s: str) -> None:
                self.s = s

            def compress(self, text: str) -> str:
                return text + self.s

        composite = CompositeCompressor(Suffix("1"), Suffix("2"))
        assert composite.compress("x") == "x12"

    def test_exposes_compressors(self) -> None:
        a = WhitespaceCompressor()
        composite = CompositeCompressor(a)
        assert composite.compressors == (a,)

    def test_composes_whitespace(self) -> None:
        composite = CompositeCompressor(WhitespaceCompressor())
        assert composite.compress("a    b") == "a b"


class TestCompressionStrategy:
    def test_compresses_message_content(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "user", "content": "too    many     spaces"},
        ]
        outcome = CompressionStrategy().apply(messages, max_tokens=999, counter=counter)
        assert outcome.messages[0]["content"] == "too many spaces"
        assert outcome.truncated == 1
        assert outcome.dropped == 0

    def test_does_not_mutate_input(self, counter: WordCounter) -> None:
        messages: list[Message] = [{"role": "user", "content": "a    b"}]
        snapshot = [dict(m) for m in messages]
        CompressionStrategy().apply(messages, max_tokens=999, counter=counter)
        assert messages == snapshot

    def test_unchanged_content_is_not_counted(self, counter: WordCounter) -> None:
        messages: list[Message] = [{"role": "user", "content": "already clean"}]
        outcome = CompressionStrategy().apply(messages, max_tokens=999, counter=counter)
        assert outcome.truncated == 0
        assert "no content compressed" in outcome.note
        # The original message object is passed through unchanged.
        assert outcome.messages[0] is messages[0]

    def test_roles_filter(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "sys    prompt"},
            {"role": "user", "content": "user    text"},
        ]
        outcome = CompressionStrategy(roles={"user"}).apply(
            messages, max_tokens=999, counter=counter
        )
        # Only the user message is compressed.
        assert outcome.messages[0]["content"] == "sys    prompt"
        assert outcome.messages[1]["content"] == "user text"
        assert outcome.truncated == 1

    def test_non_string_content_is_skipped(self, counter: WordCounter) -> None:
        messages = [{"role": "user", "content": None}]
        outcome = CompressionStrategy().apply(
            messages,  # type: ignore[arg-type]
            max_tokens=999,
            counter=counter,
        )
        assert outcome.truncated == 0

    def test_custom_compressor(self, counter: WordCounter) -> None:
        class Upper(TextCompressor):
            def compress(self, text: str) -> str:
                return text.upper()

        strategy = CompressionStrategy(Upper())
        assert strategy.compressor.__class__ is Upper
        messages: list[Message] = [{"role": "user", "content": "hello"}]
        outcome = strategy.apply(messages, max_tokens=999, counter=counter)
        assert outcome.messages[0]["content"] == "HELLO"


class TestPipelineIntegration:
    def test_composes_with_budget_pipeline(self, counter: WordCounter) -> None:
        from smarttokenoptimizer import SmartTokenOptimizer
        from smarttokenoptimizer.budgeting import CompositeStrategy, DropOldestStrategy

        messages: list[Message] = [
            {"role": "user", "content": f"msg {i}    with    spaces"} for i in range(5)
        ]
        strategy = CompositeStrategy(
            CompressionStrategy(),
            DropOldestStrategy(),
        )
        optimizer = SmartTokenOptimizer(
            max_tokens=1000, counter=counter, strategy=strategy
        )
        result = optimizer.optimize_detailed(messages)
        # Every message's content was compressed (runs of spaces collapsed).
        assert all("  " not in m["content"] for m in result.messages)
        assert result.truncated_messages == 5
