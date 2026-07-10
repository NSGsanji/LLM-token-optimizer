"""Minimal, dependency-free terminal screen control.

Thin wrappers over the ANSI escape sequences needed for a live view: clearing
the screen, moving the cursor home, switching to the alternate screen buffer,
and hiding/showing the cursor. Every operation is gated by an ``enabled`` flag
and writes to an injectable stream, so a disabled :class:`Screen` is a complete
no-op — which keeps output clean in pipes, files and tests.
"""

from __future__ import annotations

import sys
from types import TracebackType
from typing import IO

_CLEAR = "\033[2J"
_HOME = "\033[H"
_ALT_ENTER = "\033[?1049h"
_ALT_EXIT = "\033[?1049l"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


class Screen:
    """Control a terminal screen via ANSI sequences.

    Args:
        stream: Where control sequences are written. Defaults to ``sys.stdout``.
        enabled: When ``True`` (default) sequences are emitted; when ``False``
            every method does nothing, so callers need no conditionals.

    The class is also a context manager: entering switches to the alternate
    screen buffer and hides the cursor; exiting restores both — even on error.
    """

    def __init__(self, stream: IO[str] | None = None, *, enabled: bool = True) -> None:
        self._out = stream if stream is not None else sys.stdout
        self.enabled = enabled

    def _write(self, text: str) -> None:
        if self.enabled:
            self._out.write(text)
            self._out.flush()

    def write(self, text: str) -> None:
        """Write ``text`` to the screen when enabled (flushes immediately).

        Unlike the control helpers this is for ordinary content (e.g. a rendered
        frame); it still respects ``enabled`` so a disabled screen stays silent.
        """
        self._write(text)

    def clear(self) -> None:
        """Clear the screen and move the cursor to the top-left."""
        self._write(_CLEAR + _HOME)

    def home(self) -> None:
        """Move the cursor to the top-left without clearing."""
        self._write(_HOME)

    def enter_alt(self) -> None:
        """Switch to the alternate screen buffer and hide the cursor."""
        self._write(_ALT_ENTER + _HIDE_CURSOR)

    def exit_alt(self) -> None:
        """Restore the primary screen buffer and show the cursor."""
        self._write(_SHOW_CURSOR + _ALT_EXIT)

    def __enter__(self) -> Screen:
        self.enter_alt()
        self.clear()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.exit_alt()
