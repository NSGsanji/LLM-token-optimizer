"""Tests for the token budgeting module (SmartTokenOptimizer + strategies)."""

from __future__ import annotations

import pytest

from smarttokenoptimizer import (
    Message,
    OptimizationResult,
    SmartTokenOptimizer,
)
from smarttokenoptimizer.budgeting import DropOldestStrategy, StrategyOutcome
from smarttokenoptimizer.tokenization import MessageOverhead, TokenCounter


class WordCounter(TokenCounter):
    """Deterministic counter: one token per whitespace-delimited word.

    Uses zero structural overhead so token math in tests is exact and obvious.
    """

    def __init__(self) -> None:
        super().__init__(
            overhead=MessageOverhead(
                tokens_per_message=0, tokens_per_name=0, reply_priming=0
            )
        )

    def count_text(self, text: str) -> int:
        return len(text.split())


def make_messages(n: int, words: int = 2) -> list[Message]:
    """Build ``n`` user messages, each with ``words`` content words.

    Note: with :class:`WordCounter`, each message also contributes one token for
    its ``role`` value (``"user"``), so a message costs ``words + 1`` tokens.
    """
    return [{"role": "user", "content": " ".join(["w"] * words)} for _ in range(n)]


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


class TestConstruction:
    def test_rejects_non_positive_budget(self, counter: WordCounter) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            SmartTokenOptimizer(0, counter=counter)
        with pytest.raises(ValueError, match="max_tokens"):
            SmartTokenOptimizer(-5, counter=counter)

    def test_exposes_budget_and_counter(self, counter: WordCounter) -> None:
        opt = SmartTokenOptimizer(100, counter=counter)
        assert opt.max_tokens == 100
        assert opt.counter is counter

    def test_default_counter_is_created_when_omitted(self) -> None:
        opt = SmartTokenOptimizer(100, model="gpt-4o")
        assert isinstance(opt.counter, TokenCounter)


class TestPassThrough:
    def test_conversation_within_budget_is_unchanged(
        self, counter: WordCounter
    ) -> None:
        messages = make_messages(3, words=2)  # 3 * (2+1) = 9 tokens
        opt = SmartTokenOptimizer(100, counter=counter)
        result = opt.optimize(messages)
        assert result == messages

    def test_fits_predicate(self, counter: WordCounter) -> None:
        messages = make_messages(3, words=2)  # 9 tokens
        assert SmartTokenOptimizer(9, counter=counter).fits(messages) is True
        assert SmartTokenOptimizer(8, counter=counter).fits(messages) is False

    def test_count_matches_counter(self, counter: WordCounter) -> None:
        messages = make_messages(4, words=3)  # 4 * (3+1) = 16 tokens
        opt = SmartTokenOptimizer(100, counter=counter)
        assert opt.count(messages) == 16

    def test_input_is_not_mutated(self, counter: WordCounter) -> None:
        messages = make_messages(10, words=2)
        snapshot = [dict(m) for m in messages]
        SmartTokenOptimizer(4, counter=counter).optimize(messages)
        assert messages == snapshot

    def test_empty_conversation(self, counter: WordCounter) -> None:
        opt = SmartTokenOptimizer(100, counter=counter)
        assert opt.optimize([]) == []

    def test_empty_conversation_ratio_is_zero(self, counter: WordCounter) -> None:
        result = SmartTokenOptimizer(100, counter=counter).optimize_detailed([])
        assert result.original_tokens == 0
        assert result.compression_ratio == 0.0


class TestDropOldest:
    def test_drops_oldest_until_within_budget(self, counter: WordCounter) -> None:
        messages = make_messages(10, words=2)  # 10 * 3 = 30 tokens total
        opt = SmartTokenOptimizer(6, counter=counter)  # room for 2 messages
        result = opt.optimize_detailed(messages)
        assert result.within_budget
        assert result.optimized_tokens <= 6
        assert result.dropped_messages == 8
        # The most recent messages are the ones retained.
        assert result.messages == messages[-2:]

    def test_preserves_system_message(self, counter: WordCounter) -> None:
        messages: list[Message] = [{"role": "system", "content": "sys prompt here"}]
        messages += make_messages(20, words=2)
        opt = SmartTokenOptimizer(8, counter=counter)
        result = opt.optimize_detailed(messages)
        assert result.messages[0]["role"] == "system"
        assert result.within_budget

    def test_reports_accounting(self, counter: WordCounter) -> None:
        messages = make_messages(10, words=2)  # 10 * 3 = 30 tokens
        result = SmartTokenOptimizer(6, counter=counter).optimize_detailed(messages)
        assert isinstance(result, OptimizationResult)
        assert result.original_tokens == 30
        assert result.tokens_saved == result.original_tokens - result.optimized_tokens
        assert 0 <= result.compression_ratio <= 1
        assert result.changed is True

    def test_within_budget_result_when_no_change_needed(
        self, counter: WordCounter
    ) -> None:
        messages = make_messages(2, words=2)  # 4 tokens
        result = SmartTokenOptimizer(100, counter=counter).optimize_detailed(messages)
        assert result.changed is False
        assert result.tokens_saved == 0
        assert result.compression_ratio == 0.0
        assert "already within budget" in result.notes


class TestUnsatisfiableBudget:
    def test_protected_messages_exceed_budget(self, counter: WordCounter) -> None:
        # Two system messages worth 6 tokens, but a budget of only 2.
        messages: list[Message] = [
            {"role": "system", "content": "one two three"},
            {"role": "system", "content": "four five six"},
        ]
        result = SmartTokenOptimizer(2, counter=counter).optimize_detailed(messages)
        # System messages cannot be dropped, so the budget is not met but the
        # protected content is preserved.
        assert result.within_budget is False
        assert all(m["role"] == "system" for m in result.messages)


class TestKeepLast:
    def test_keep_last_retains_recent_messages(self, counter: WordCounter) -> None:
        messages = make_messages(10, words=2)  # 20 tokens
        strategy = DropOldestStrategy(keep_last=2)
        opt = SmartTokenOptimizer(2, counter=counter, strategy=strategy)
        result = opt.optimize_detailed(messages)
        # Even under a tight budget, the last 2 messages are retained.
        assert result.messages[-2:] == messages[-2:]
        assert len(result.messages) >= 2

    def test_keep_last_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="keep_last"):
            DropOldestStrategy(keep_last=-1)


class TestStrategyDirect:
    """Exercise DropOldestStrategy directly, independent of the optimizer."""

    def test_within_budget_returns_input_unchanged(self, counter: WordCounter) -> None:
        messages = make_messages(2, words=1)  # 2 * 2 = 4 tokens
        outcome = DropOldestStrategy().apply(messages, max_tokens=100, counter=counter)
        assert outcome.messages == messages
        assert outcome.dropped == 0
        assert "within budget" in outcome.note

    def test_note_when_nothing_droppable(self, counter: WordCounter) -> None:
        # Over budget, but every message is protected -> nothing to drop.
        messages: list[Message] = [
            {"role": "system", "content": "a b c"},
            {"role": "system", "content": "d e f"},
        ]
        outcome = DropOldestStrategy().apply(messages, max_tokens=1, counter=counter)
        assert outcome.dropped == 0
        assert outcome.note == "no messages dropped"


class TestCustomStrategy:
    def test_optimizer_uses_injected_strategy(self, counter: WordCounter) -> None:
        class NoopStrategy(DropOldestStrategy):
            def apply(self, messages, *, max_tokens, counter):  # type: ignore[no-untyped-def]
                return StrategyOutcome(messages=list(messages), note="noop")

        messages = make_messages(10, words=5)  # 50 tokens, over budget
        opt = SmartTokenOptimizer(5, counter=counter, strategy=NoopStrategy())
        result = opt.optimize_detailed(messages)
        # The noop strategy leaves everything, so the budget is not met.
        assert result.within_budget is False
        assert len(result.messages) == 10


class TestProtectedRoles:
    def test_custom_protected_roles(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "developer", "content": "keep me around please"},
            *make_messages(20, words=2),
        ]
        strategy = DropOldestStrategy(protected_roles={"developer"})
        opt = SmartTokenOptimizer(8, counter=counter, strategy=strategy)
        result = opt.optimize_detailed(messages)
        assert result.messages[0]["role"] == "developer"
