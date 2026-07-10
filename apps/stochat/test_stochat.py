"""Tests for the stochat app's pure logic (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.stochat.__main__ import _attach_paths, _optimize
from apps.stochat.providers import resolve

from smarttokenoptimizer.tokenization import get_counter


class TestAttachPaths:
    def test_attaches_a_file(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def hi():\n    return 1\n", encoding="utf-8")
        content, attached = _attach_paths(f"review {f} please")
        assert str(f) in attached
        assert "def hi():" in content
        assert "    return 1" in content  # indentation preserved

    def test_attaches_a_directory_listing(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x=1", encoding="utf-8")
        (tmp_path / "b.py").write_text("y=2", encoding="utf-8")
        content, attached = _attach_paths(f"look at {tmp_path}")
        assert f"{tmp_path}/" in attached
        assert "a.py" in content and "b.py" in content

    def test_ignores_nonexistent_paths(self) -> None:
        content, attached = _attach_paths("just a normal question here")
        assert attached == []
        assert content == "just a normal question here"


class TestOptimizePinning:
    def _counter(self):
        return get_counter("gpt-4o", prefer_exact=False)

    def test_pinned_file_survives_trimming(self) -> None:
        big_file = "UNIQUE_MARKER\n" + ("some detailed code line here\n" * 400)
        history = [
            {"role": "user", "content": "read this\n" + big_file, "pin": True},
            {"role": "assistant", "content": "ok", "pin": False},
        ]
        for i in range(6):
            history.append(
                {"role": "user", "content": f"chatter {i} " * 10, "pin": False}
            )
            history.append(
                {"role": "assistant", "content": f"reply {i} " * 10, "pin": False}
            )
        history.append(
            {"role": "user", "content": "now list the problems", "pin": False}
        )

        _result, outgoing = _optimize(
            history, self._counter(), budget=1500, keep_last=2
        )
        joined = " ".join(m["content"] for m in outgoing)
        assert "UNIQUE_MARKER" in joined  # the pinned file is never dropped
        assert "now list the problems" in joined  # the question survives
        assert len(outgoing) < len(history)  # chatter WAS trimmed
        assert all(m["role"] in ("user", "assistant") for m in outgoing)  # no 'pinned'

    def test_short_conversation_is_untouched(self) -> None:
        history = [{"role": "user", "content": "hello there", "pin": False}]
        result, outgoing = _optimize(history, self._counter(), budget=8000, keep_last=2)
        assert result.tokens_saved == 0
        assert len(outgoing) == 1


class TestProviders:
    def test_resolve_known(self) -> None:
        assert resolve("groq").name == "groq"

    def test_resolve_default(self) -> None:
        assert resolve(None).name  # returns some provider, no crash

    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(SystemExit):
            resolve("nope-provider")
