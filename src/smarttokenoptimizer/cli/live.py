"""A live, auto-refreshing dashboard view for the terminal.

The frame is a pure function (:func:`build_frame`) so it is trivial to test;
:func:`run_live` drives it on the alternate screen with an injectable clock and
sleep function, redrawing a bounded or unbounded number of times. A
``KeyboardInterrupt`` (Ctrl-C) ends the loop cleanly and restores the screen.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..cost.analytics import AnalyticsSnapshot
from ..credentials.pool import CredentialHealth
from .dashboard import render_dashboard
from .screen import Screen
from .style import Ansi

#: A source of the current analytics snapshot (and optional provider health).
SnapshotProvider = Callable[
    [], "tuple[AnalyticsSnapshot, Sequence[CredentialHealth] | None]"
]


def build_frame(
    snapshot: AnalyticsSnapshot,
    providers: Sequence[CredentialHealth] | None,
    *,
    tick: int,
    ansi: Ansi | None = None,
    width: int = 42,
) -> str:
    """Return a full live-view frame: the dashboard plus a help footer.

    Args:
        snapshot: The analytics snapshot to render.
        providers: Optional provider/credential health rows.
        tick: The current refresh count, shown as a heartbeat in the footer.
        ansi: Styler for the footer; a disabled styler is used when ``None``.
        width: Dashboard box width.

    Returns:
        The frame as a multi-line string (no screen-control codes).
    """
    styler = ansi if ansi is not None else Ansi(enabled=False)
    body = render_dashboard(snapshot, providers, width=width)
    spinner = "|/-\\"[tick % 4]
    footer = styler.dim(f"{spinner} live · press Ctrl-C to exit")
    return f"{body}\n{footer}"


def run_live(
    source: SnapshotProvider,
    *,
    screen: Screen,
    ansi: Ansi | None = None,
    width: int = 42,
    interval: float = 1.0,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Render an auto-refreshing dashboard until stopped.

    Args:
        source: Called each tick to obtain the current snapshot and providers.
        screen: The :class:`Screen` to draw on (used as a context manager).
        ansi: Styler for the footer.
        width: Dashboard box width.
        interval: Seconds between refreshes.
        iterations: Number of frames to draw, or ``None`` to run until a
            ``KeyboardInterrupt``. Bounded values make the loop testable.
        sleep: Sleep function (injectable for tests).

    Returns:
        A process exit code (``0``).
    """
    styler = ansi if ansi is not None else Ansi(enabled=screen.enabled)
    tick = 0
    with screen:
        try:
            while iterations is None or tick < iterations:
                snapshot, providers = source()
                screen.home()
                screen.clear()
                frame = build_frame(
                    snapshot, providers, tick=tick, ansi=styler, width=width
                )
                screen.write(frame + "\n")
                tick += 1
                if iterations is not None and tick >= iterations:
                    break
                sleep(interval)
        except KeyboardInterrupt:
            pass
    return 0
