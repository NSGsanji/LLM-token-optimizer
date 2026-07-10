"""Tests for the model registry and counter factory."""

from __future__ import annotations

import pytest

from smarttokenoptimizer.tokenization import (
    HeuristicTokenCounter,
    encoding_name_for_model,
    get_counter,
    registry,
)


class TestEncodingNameForModel:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "o200k_base"),
            ("gpt-4o-mini", "o200k_base"),
            ("GPT-4O", "o200k_base"),
            ("o1-preview", "o200k_base"),
            ("o3-mini", "o200k_base"),
            ("gpt-4", "cl100k_base"),
            ("gpt-4-turbo", "cl100k_base"),
            ("gpt-3.5-turbo", "cl100k_base"),
            ("text-embedding-3-small", "cl100k_base"),
        ],
    )
    def test_known_models(self, model: str, expected: str) -> None:
        assert encoding_name_for_model(model) == expected

    def test_unknown_model_falls_back_to_default(self) -> None:
        assert encoding_name_for_model("some-unknown-model") == "cl100k_base"

    def test_whitespace_is_stripped(self) -> None:
        assert encoding_name_for_model("  gpt-4o  ") == "o200k_base"


class TestGetCounter:
    def test_prefer_exact_false_returns_heuristic(self) -> None:
        counter = get_counter("gpt-4o", prefer_exact=False)
        assert isinstance(counter, HeuristicTokenCounter)

    def test_no_model_returns_a_counter(self) -> None:
        counter = get_counter(prefer_exact=False)
        assert isinstance(counter, HeuristicTokenCounter)

    def test_falls_back_when_backend_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate tiktoken being unavailable.
        monkeypatch.setattr(registry, "is_available", lambda: False)
        counter = get_counter("gpt-4o", prefer_exact=True)
        assert isinstance(counter, HeuristicTokenCounter)

    def test_returned_counter_counts(self) -> None:
        counter = get_counter("gpt-4o", prefer_exact=False)
        assert counter.count_text("hello world") >= 1

    def test_prefer_exact_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the "backend available" branch regardless of the environment by
        # substituting a stub exact counter, so the exact-selection path in the
        # factory is exercised deterministically.
        monkeypatch.setattr(registry, "is_available", lambda: True)
        captured: dict[str, str] = {}

        class StubExactCounter(HeuristicTokenCounter):
            def __init__(self, encoding: str, **kwargs: object) -> None:
                super().__init__()
                captured["encoding"] = encoding

        monkeypatch.setattr(registry, "TiktokenCounter", StubExactCounter)
        counter = get_counter("gpt-4o", prefer_exact=True)
        assert isinstance(counter, StubExactCounter)
        # gpt-4o maps to the o200k_base encoding.
        assert captured["encoding"] == "o200k_base"

    def test_prefer_exact_without_model_uses_default_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "is_available", lambda: True)
        captured: dict[str, str] = {}

        class StubExactCounter(HeuristicTokenCounter):
            def __init__(self, encoding: str, **kwargs: object) -> None:
                super().__init__()
                captured["encoding"] = encoding

        monkeypatch.setattr(registry, "TiktokenCounter", StubExactCounter)
        get_counter(None, prefer_exact=True)
        assert captured["encoding"] == "cl100k_base"
