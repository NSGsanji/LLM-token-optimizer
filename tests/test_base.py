"""Tests for the TokenCounter base class message-counting logic.

A tiny deterministic counter (one token per whitespace-separated word) is used
so the overhead arithmetic can be asserted exactly, independent of any real
tokenizer heuristics.
"""

from __future__ import annotations

import pytest

from smarttokenoptimizer.tokenization import (
    DEFAULT_OVERHEAD,
    Message,
    MessageOverhead,
    TokenCounter,
)


class WordCounter(TokenCounter):
    """Counts one token per whitespace-delimited word; exact for testing."""

    def count_text(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


class TestCountMessage:
    def test_includes_per_message_overhead(self, counter: WordCounter) -> None:
        message: Message = {"role": "user", "content": "hello world"}
        # role=1 word, content=2 words, +3 per-message overhead = 6
        assert counter.count_message(message) == 6

    def test_name_adds_extra_overhead(self, counter: WordCounter) -> None:
        without: Message = {"role": "user", "content": "hi"}
        with_name: Message = {"role": "user", "content": "hi", "name": "bob"}
        # name adds its own word (+1) plus tokens_per_name (+1) = +2
        assert counter.count_message(with_name) == counter.count_message(without) + 2

    def test_non_string_value_is_stringified(self, counter: WordCounter) -> None:
        message = {"role": "user", "content": "hi", "extra": 123}
        # Should not raise; the stringified value contributes at least one word.
        assert counter.count_message(message) > 0  # type: ignore[arg-type]


class TestCountMessages:
    def test_adds_reply_priming_once(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello there"},
        ]
        per_message = sum(counter.count_message(m) for m in messages)
        assert (
            counter.count_messages(messages)
            == per_message + DEFAULT_OVERHEAD.reply_priming
        )

    def test_empty_conversation_is_just_priming(self, counter: WordCounter) -> None:
        assert counter.count_messages([]) == DEFAULT_OVERHEAD.reply_priming

    def test_accepts_any_iterable(self, counter: WordCounter) -> None:
        messages: list[Message] = [{"role": "user", "content": "a b c"}]
        assert counter.count_messages(iter(messages)) == counter.count_messages(
            messages
        )


class TestCustomOverhead:
    def test_custom_overhead_is_respected(self) -> None:
        overhead = MessageOverhead(
            tokens_per_message=5, tokens_per_name=2, reply_priming=7
        )
        counter = WordCounter(overhead=overhead)
        assert counter.overhead is overhead
        message: Message = {"role": "user", "content": "hi"}
        # role=1, content=1, +5 overhead = 7
        assert counter.count_message(message) == 7
        assert counter.count_messages([message]) == 7 + 7
