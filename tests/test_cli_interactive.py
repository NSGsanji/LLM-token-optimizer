"""Tests for the interactive CLI foundation: ANSI styling and the menu."""

from __future__ import annotations

import io

import pytest

from smarttokenoptimizer.cli.interactive import (
    Menu,
    MenuItem,
    MenuResult,
    render_menu,
)
from smarttokenoptimizer.cli.style import Ansi, supports_color


class TestSupportsColor:
    def test_no_color_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert supports_color(io.StringIO()) is False

    def test_force_color_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert supports_color(io.StringIO()) is True

    def test_no_color_beats_force_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert supports_color(io.StringIO()) is False

    def test_non_tty_stream_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        # A StringIO is not a TTY.
        assert supports_color(io.StringIO()) is False

    def test_tty_stream_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)

        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        assert supports_color(FakeTTY()) is True

    def test_default_stream_uses_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)

        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        # With no stream argument, detection falls back to sys.stdout.
        monkeypatch.setattr("sys.stdout", FakeTTY())
        assert supports_color() is True

    def test_isatty_error_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)

        class Broken(io.StringIO):
            def isatty(self) -> bool:
                raise ValueError("stream is closed")

        # A stream whose isatty() raises must be treated as non-colour.
        assert supports_color(Broken()) is False


class TestAnsi:
    def test_enabled_wraps(self) -> None:
        ansi = Ansi(enabled=True)
        assert ansi.bold("x") == "\033[1mx\033[0m"
        assert ansi.cyan("x") == "\033[36mx\033[0m"

    def test_disabled_passthrough(self) -> None:
        ansi = Ansi(enabled=False)
        for method in ("bold", "dim", "red", "green", "yellow", "cyan", "grey"):
            assert getattr(ansi, method)("x") == "x"

    def test_multiple_styles(self) -> None:
        ansi = Ansi(enabled=True)
        assert ansi.apply("x", "bold", "red") == "\033[1;31mx\033[0m"

    def test_unknown_style_ignored(self) -> None:
        ansi = Ansi(enabled=True)
        assert ansi.apply("x", "not-a-style") == "x"

    def test_auto_detect_from_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # With colour env overrides cleared, a non-TTY stream disables colour.
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert Ansi(stream=io.StringIO()).enabled is False


class TestMenuItem:
    def test_payload_defaults_to_label(self) -> None:
        assert MenuItem("Go").payload == "Go"

    def test_payload_uses_value(self) -> None:
        assert MenuItem("Go", value=42).payload == 42


class TestRenderMenu:
    def _items(self) -> list[MenuItem]:
        return [MenuItem("First", "does A"), MenuItem("Second")]

    def test_contains_title_and_items(self) -> None:
        out = render_menu("Title", self._items())
        assert "Title" in out
        assert "First" in out
        assert "Second" in out
        assert "does A" in out

    def test_numbers_items(self) -> None:
        out = render_menu("Title", self._items())
        assert "1)" in out
        assert "2)" in out

    def test_footer_shown(self) -> None:
        out = render_menu("Title", self._items(), footer="Custom footer.")
        assert "Custom footer." in out

    def test_plain_when_ansi_disabled(self) -> None:
        out = render_menu("Title", self._items(), ansi=Ansi(enabled=False))
        assert "\033[" not in out


class TestMenuRun:
    def _items(self) -> list[MenuItem]:
        return [MenuItem("Alpha"), MenuItem("Beta"), MenuItem("Gamma")]

    def _menu(self, keys: str) -> tuple[Menu, io.StringIO]:
        out = io.StringIO()
        menu = Menu(
            "Pick",
            self._items(),
            ansi=Ansi(enabled=False),
            input=io.StringIO(keys),
            output=out,
        )
        return menu, out

    def test_requires_items(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Menu("Empty", [])

    def test_valid_selection(self) -> None:
        menu, _ = self._menu("2\n")
        result = menu.run()
        assert isinstance(result, MenuResult)
        assert result.index == 1
        assert result.item is not None
        assert result.item.label == "Beta"
        assert result.cancelled is False

    def test_quit(self) -> None:
        menu, _ = self._menu("q\n")
        result = menu.run()
        assert result.cancelled is True
        assert result.index == -1

    def test_eof_cancels(self) -> None:
        menu, _ = self._menu("")  # immediate EOF
        assert menu.run().cancelled is True

    def test_invalid_then_valid(self) -> None:
        menu, out = self._menu("9\nabc\n1\n")
        result = menu.run()
        assert result.item is not None
        assert result.item.label == "Alpha"
        assert "Invalid choice" in out.getvalue()

    def test_exit_alias(self) -> None:
        menu, _ = self._menu("exit\n")
        assert menu.run().cancelled is True
