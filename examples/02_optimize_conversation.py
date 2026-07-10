"""Example: fitting a long conversation into a token budget.

Run with:  python examples/02_optimize_conversation.py

Shows the headline ``SmartTokenOptimizer`` plus a composed strategy pipeline
(compress whitespace -> remove duplicates -> drop oldest turns).
"""

from __future__ import annotations

from smarttokenoptimizer import Message, SmartTokenOptimizer
from smarttokenoptimizer.budgeting import CompositeStrategy, DropOldestStrategy
from smarttokenoptimizer.compression import CompressionStrategy
from smarttokenoptimizer.context import DeduplicateStrategy


def build_conversation() -> list[Message]:
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    # A repeated (duplicate) message and lots of chatter to trim.
    for i in range(30):
        messages.append(
            {"role": "user", "content": f"Question {i} with    extra    spaces."}
        )
        messages.append({"role": "assistant", "content": "Here is a detailed answer."})
    messages.append({"role": "assistant", "content": "Here is a detailed answer."})
    return messages


def main() -> None:
    messages = build_conversation()

    optimizer = SmartTokenOptimizer(
        max_tokens=200,
        model="gpt-4o",
        strategy=CompositeStrategy(
            CompressionStrategy(),
            DeduplicateStrategy(),
            DropOldestStrategy(),
        ),
    )

    result = optimizer.optimize_detailed(messages)
    print(f"Original messages: {len(messages)}  tokens: {result.original_tokens}")
    print(
        f"Optimized messages: {len(result.messages)}  "
        f"tokens: {result.optimized_tokens}"
    )
    print(
        f"Dropped: {result.dropped_messages}  "
        f"compressed: {result.truncated_messages}"
    )
    print(
        f"Tokens saved: {result.tokens_saved} "
        f"({result.compression_ratio:.0%} smaller)"
    )
    print(f"Within budget: {result.within_budget}")
    # The system prompt is always preserved.
    assert result.messages[0]["role"] == "system"


if __name__ == "__main__":
    main()
