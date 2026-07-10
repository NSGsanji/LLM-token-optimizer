"""Example: counting tokens for strings and chat conversations.

Run with:  python examples/01_token_counting.py

Uses the dependency-free heuristic counter by default; install the ``tiktoken``
extra for exact counts (``pip install "smarttokenoptimizer[tiktoken]"``).
"""

from __future__ import annotations

from smarttokenoptimizer import Message, get_counter


def main() -> None:
    # ``prefer_exact=False`` forces the fast, zero-dependency estimator so this
    # example runs identically with or without the tiktoken extra installed.
    counter = get_counter("gpt-4o", prefer_exact=False)

    text = "Hello, world! SmartTokenOptimizer counts tokens quickly."
    print(f"Text tokens: {counter.count_text(text)}")

    conversation: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    print(
        f"Conversation tokens (incl. chat overhead): "
        f"{counter.count_messages(conversation)}"
    )


if __name__ == "__main__":
    main()
