"""Tests for the optional tiktoken-backed exact counter.

Tests that require the real backend are skipped automatically when ``tiktoken``
is not installed, so the suite passes in a zero-dependency environment while
still validating exactness when the extra is present.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import pytest

from smarttokenoptimizer.tokenization import (
    BackendUnavailableError,
    TiktokenCounter,
    is_available,
    tiktoken_backend,
)

requires_tiktoken = pytest.mark.skipif(
    not is_available(), reason="tiktoken backend not installed"
)


class TestAvailability:
    def test_is_available_returns_bool(self) -> None:
        assert isinstance(is_available(), bool)


class TestBackendMissing:
    def test_raises_backend_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear the encoding cache and force `import tiktoken` to fail.
        tiktoken_backend._load_encoding.cache_clear()
        real_import: Callable[..., Any] = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tiktoken":
                raise ImportError("simulated missing tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(BackendUnavailableError):
            TiktokenCounter("cl100k_base")
        # Avoid leaking a poisoned cache entry to other tests.
        tiktoken_backend._load_encoding.cache_clear()


@requires_tiktoken
class TestExactCounting:
    @pytest.fixture
    def counter(self) -> TiktokenCounter:
        return TiktokenCounter("cl100k_base")

    def test_is_exact(self, counter: TiktokenCounter) -> None:
        assert counter.exact is True

    def test_empty_string_is_zero(self, counter: TiktokenCounter) -> None:
        assert counter.count_text("") == 0

    def test_hello_world(self, counter: TiktokenCounter) -> None:
        assert counter.count_text("Hello, world!") == 4

    def test_matches_reference_encoding(self, counter: TiktokenCounter) -> None:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        assert counter.count_text(text) == len(enc.encode(text))

    def test_encoding_name_exposed(self, counter: TiktokenCounter) -> None:
        assert counter.encoding_name == "cl100k_base"

    def test_special_tokens_in_user_text_do_not_raise(
        self, counter: TiktokenCounter
    ) -> None:
        # The literal "<|endoftext|>" must be treated as ordinary text.
        assert counter.count_text("<|endoftext|>") >= 1

    def test_message_counting(self, counter: TiktokenCounter) -> None:
        tokens = counter.count_messages([{"role": "user", "content": "Hello"}])
        assert tokens > 0
