"""Tests for context optimization strategies (dedup, sliding window, composite)."""

from __future__ import annotations

import pytest

from smarttokenoptimizer.budgeting import (
    CompositeStrategy,
    DropOldestStrategy,
    StrategyOutcome,
)
from smarttokenoptimizer.context import DeduplicateStrategy, SlidingWindowStrategy
from smarttokenoptimizer.tokenization import Message, MessageOverhead, TokenCounter


class WordCounter(TokenCounter):
    """One token per whitespace word, zero overhead — for exact assertions."""

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


def user(content: str) -> Message:
    return {"role": "user", "content": content}


class TestDeduplicate:
    def test_removes_exact_duplicates_keep_first(self, counter: WordCounter) -> None:
        messages = [user("a"), user("b"), user("a"), user("b"), user("c")]
        outcome = DeduplicateStrategy(keep="first").apply(
            messages, max_tokens=999, counter=counter
        )
        assert [m["content"] for m in outcome.messages] == ["a", "b", "c"]
        assert outcome.dropped == 2

    def test_keep_last_retains_final_occurrence(self, counter: WordCounter) -> None:
        messages = [user("a"), user("b"), user("a")]
        outcome = DeduplicateStrategy(keep="last").apply(
            messages, max_tokens=999, counter=counter
        )
        # The earlier "a" is removed; order of survivors is preserved.
        assert [m["content"] for m in outcome.messages] == ["b", "a"]
        assert outcome.dropped == 1

    def test_no_duplicates_is_noop(self, counter: WordCounter) -> None:
        messages = [user("a"), user("b"), user("c")]
        outcome = DeduplicateStrategy().apply(messages, max_tokens=999, counter=counter)
        assert outcome.dropped == 0
        assert outcome.messages == messages
        assert "no duplicates" in outcome.note

    def test_by_role_distinguishes_roles(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        ]
        # Same content, different roles -> not duplicates when by_role=True.
        outcome = DeduplicateStrategy(by_role=True).apply(
            messages, max_tokens=999, counter=counter
        )
        assert outcome.dropped == 0

    def test_by_role_false_ignores_role(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        ]
        outcome = DeduplicateStrategy(by_role=False).apply(
            messages, max_tokens=999, counter=counter
        )
        assert outcome.dropped == 1

    def test_protected_roles_are_never_removed(self, counter: WordCounter) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "s"},
            {"role": "system", "content": "s"},
        ]
        outcome = DeduplicateStrategy(protected_roles={"system"}).apply(
            messages, max_tokens=999, counter=counter
        )
        assert outcome.dropped == 0

    def test_invalid_keep_raises(self) -> None:
        with pytest.raises(ValueError, match="keep"):
            DeduplicateStrategy(keep="middle")

    def test_empty_conversation(self, counter: WordCounter) -> None:
        outcome = DeduplicateStrategy().apply([], max_tokens=999, counter=counter)
        assert outcome.messages == []
        assert outcome.dropped == 0


class TestSlidingWindow:
    def test_keeps_newest_messages(self, counter: WordCounter) -> None:
        messages = [user(str(i)) for i in range(5)]
        outcome = SlidingWindowStrategy(max_messages=2).apply(
            messages, max_tokens=999, counter=counter
        )
        assert [m["content"] for m in outcome.messages] == ["3", "4"]
        assert outcome.dropped == 3

    def test_preserves_protected_and_windows_the_rest(
        self, counter: WordCounter
    ) -> None:
        messages: list[Message] = [{"role": "system", "content": "sys"}]
        messages += [user(str(i)) for i in range(5)]
        outcome = SlidingWindowStrategy(max_messages=2).apply(
            messages, max_tokens=999, counter=counter
        )
        assert [m["content"] for m in outcome.messages] == ["sys", "3", "4"]

    def test_within_window_is_noop(self, counter: WordCounter) -> None:
        messages = [user("a"), user("b")]
        outcome = SlidingWindowStrategy(max_messages=5).apply(
            messages, max_tokens=999, counter=counter
        )
        assert outcome.messages == messages
        assert outcome.dropped == 0
        assert outcome.note == "within window"

    def test_zero_window_keeps_only_protected(self, counter: WordCounter) -> None:
        messages: list[Message] = [{"role": "system", "content": "sys"}]
        messages += [user("a"), user("b")]
        outcome = SlidingWindowStrategy(max_messages=0).apply(
            messages, max_tokens=999, counter=counter
        )
        assert [m["content"] for m in outcome.messages] == ["sys"]

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError, match="max_messages"):
            SlidingWindowStrategy(-1)


class TestComposite:
    def test_requires_at_least_one_strategy(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeStrategy()

    def test_applies_in_order(self, counter: WordCounter) -> None:
        # Duplicates then a window: dedup first shrinks, then window bounds.
        messages = [user("a"), user("a"), user("b"), user("c"), user("d")]
        composite = CompositeStrategy(
            DeduplicateStrategy(),  # -> a, b, c, d
            SlidingWindowStrategy(max_messages=2),  # -> c, d
        )
        outcome = composite.apply(messages, max_tokens=999, counter=counter)
        assert [m["content"] for m in outcome.messages] == ["c", "d"]

    def test_aggregates_accounting(self, counter: WordCounter) -> None:
        messages = [user("a"), user("a"), user("b"), user("c"), user("d")]
        composite = CompositeStrategy(
            DeduplicateStrategy(),  # drops 1
            SlidingWindowStrategy(max_messages=2),  # drops 2
        )
        outcome = composite.apply(messages, max_tokens=999, counter=counter)
        assert outcome.dropped == 3
        assert "DeduplicateStrategy" in outcome.note
        assert "SlidingWindowStrategy" in outcome.note

    def test_exposes_strategies(self) -> None:
        d = DeduplicateStrategy()
        w = SlidingWindowStrategy(3)
        composite = CompositeStrategy(d, w)
        assert composite.strategies == (d, w)

    def test_meets_budget_when_final_strategy_is_drop_oldest(
        self, counter: WordCounter
    ) -> None:
        messages = [user("dup dup"), user("dup dup")]
        messages += [user(f"m{i} filler") for i in range(10)]
        composite = CompositeStrategy(
            DeduplicateStrategy(),
            DropOldestStrategy(),
        )
        outcome = composite.apply(messages, max_tokens=6, counter=counter)
        assert counter.count_messages(outcome.messages) <= 6

    def test_note_skips_empty_notes(self, counter: WordCounter) -> None:
        class SilentStrategy(DeduplicateStrategy):
            def apply(self, messages, *, max_tokens, counter):  # type: ignore[no-untyped-def]
                return StrategyOutcome(messages=list(messages), note="")

        composite = CompositeStrategy(SilentStrategy())
        outcome = composite.apply([user("a")], max_tokens=999, counter=counter)
        assert outcome.note == ""


class TestOptimizerRunsStrategyWithinBudget:
    """Budget-independent strategies must run even when already within budget."""

    def test_dedup_runs_when_within_budget(self, counter: WordCounter) -> None:
        from smarttokenoptimizer import SmartTokenOptimizer

        messages = [user("dup"), user("dup"), user("tail")]
        # Budget is generous; without running the strategy nothing would change.
        opt = SmartTokenOptimizer(
            max_tokens=10_000, counter=counter, strategy=DeduplicateStrategy()
        )
        result = opt.optimize_detailed(messages)
        assert result.dropped_messages == 1
        assert [m["content"] for m in result.messages] == ["dup", "tail"]
        assert result.within_budget is True
