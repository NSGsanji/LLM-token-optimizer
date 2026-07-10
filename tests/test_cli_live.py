"""Tests for the screen helpers and the live auto-refreshing dashboard."""

from __future__ import annotations

import io

import pytest

from smarttokenoptimizer.cli.live import build_frame, run_live
from smarttokenoptimizer.cli.screen import Screen
from smarttokenoptimizer.cli.style import Ansi
from smarttokenoptimizer.cost import UsageTracker


def _snapshot() -> object:
    tracker = UsageTracker()
    tracker.record(
        model="gpt-4o", input_tokens=1200, output_tokens=300, cost=0.01, cache_hit=True
    )
    return tracker.snapshot()


class TestScreen:
    def test_enabled_emits_sequences(self) -> None:
        buf = io.StringIO()
        screen = Screen(stream=buf, enabled=True)
        screen.enter_alt()
        screen.clear()
        screen.home()
        screen.write("hello")
        screen.exit_alt()
        out = buf.getvalue()
        assert "\033[?1049h" in out  # alt enter
        assert "\033[2J" in out  # clear
        assert "\033[H" in out  # home
        assert "hello" in out
        assert "\033[?1049l" in out  # alt exit
        assert "\033[?25h" in out  # show cursor

    def test_disabled_is_noop(self) -> None:
        buf = io.StringIO()
        screen = Screen(stream=buf, enabled=False)
        screen.enter_alt()
        screen.clear()
        screen.write("hello")
        screen.exit_alt()
        assert buf.getvalue() == ""

    def test_context_manager_wraps_alt_screen(self) -> None:
        buf = io.StringIO()
        with Screen(stream=buf, enabled=True) as screen:
            screen.write("body")
        out = buf.getvalue()
        assert out.index("\033[?1049h") < out.index("body")
        assert out.index("body") < out.index("\033[?1049l")

    def test_context_manager_restores_on_error(self) -> None:
        buf = io.StringIO()
        with pytest.raises(RuntimeError), Screen(stream=buf, enabled=True):
            raise RuntimeError("boom")
        # Alt screen must still be exited despite the error.
        assert "\033[?1049l" in buf.getvalue()


class TestBuildFrame:
    def test_contains_dashboard_and_footer(self) -> None:
        frame = build_frame(_snapshot(), None, tick=0)
        assert "SmartTokenOptimizer" in frame
        assert "live" in frame
        assert "Ctrl-C" in frame

    def test_spinner_advances_with_tick(self) -> None:
        chars = {
            build_frame(_snapshot(), None, tick=t).splitlines()[-1][0] for t in range(4)
        }
        # The four spinner frames should differ.
        assert len(chars) == 4


class TestRunLive:
    def test_draws_bounded_iterations(self) -> None:
        buf = io.StringIO()
        screen = Screen(stream=buf, enabled=True)
        slept: list[float] = []
        rc = run_live(
            lambda: (_snapshot(), None),
            screen=screen,
            ansi=Ansi(enabled=False),
            iterations=3,
            sleep=slept.append,
        )
        out = buf.getvalue()
        assert rc == 0
        assert out.count("SmartTokenOptimizer") == 3
        # No trailing sleep after the final frame.
        assert slept == [1.0, 1.0]

    def test_zero_iterations_draws_nothing(self) -> None:
        buf = io.StringIO()
        rc = run_live(
            lambda: (_snapshot(), None),
            screen=Screen(stream=buf, enabled=True),
            iterations=0,
            sleep=lambda _s: None,
        )
        assert rc == 0
        assert "SmartTokenOptimizer" not in buf.getvalue()
        assert "\033[?1049l" in buf.getvalue()

    def test_alt_screen_entered_and_exited(self) -> None:
        buf = io.StringIO()
        run_live(
            lambda: (_snapshot(), None),
            screen=Screen(stream=buf, enabled=True),
            iterations=1,
            sleep=lambda _s: None,
        )
        out = buf.getvalue()
        assert "\033[?1049h" in out
        assert "\033[?1049l" in out

    def test_keyboard_interrupt_stops_cleanly(self) -> None:
        buf = io.StringIO()

        def boom() -> tuple[object, None]:
            raise KeyboardInterrupt

        rc = run_live(
            boom,
            screen=Screen(stream=buf, enabled=True),
            iterations=5,
            sleep=lambda _s: None,
        )
        assert rc == 0
        # Even interrupted mid-loop, the alt screen is restored.
        assert "\033[?1049l" in buf.getvalue()

    def test_interrupt_during_sleep(self) -> None:
        buf = io.StringIO()

        def interrupt(_seconds: float) -> None:
            raise KeyboardInterrupt

        rc = run_live(
            lambda: (_snapshot(), None),
            screen=Screen(stream=buf, enabled=True),
            iterations=None,
            sleep=interrupt,
        )
        assert rc == 0
        assert buf.getvalue().count("SmartTokenOptimizer") == 1


class TestDashboardWatchCommand:
    def test_watch_flag_runs_live(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from smarttokenoptimizer.cli.main import main

        # Avoid real sleeping and bound the loop via --iterations.
        monkeypatch.setattr("time.sleep", lambda _s: None)
        rc = main(["dashboard", "--watch", "--iterations", "2", "--interval", "0"])
        assert rc == 0
        assert "SmartTokenOptimizer" in capsys.readouterr().out
