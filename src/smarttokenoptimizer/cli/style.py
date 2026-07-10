"""Minimal ANSI styling helpers with automatic, well-behaved disabling.

Colour is applied only when it makes sense: never when ``NO_COLOR`` is set,
never when the target stream is not a TTY, and always when ``FORCE_COLOR`` is
set (handy for tests and CI capture). An :class:`Ansi` instance carries a single
``enabled`` flag so call sites stay clean and output is deterministic in tests.
"""

from __future__ import annotations

import os
from typing import IO

_RESET = "\033[0m"
_CODES = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "grey": "90",
}


def supports_color(stream: IO[str] | None = None) -> bool:
    """Return whether ANSI colour should be emitted to ``stream``.

    Precedence: ``NO_COLOR`` disables unconditionally; ``FORCE_COLOR`` enables
    unconditionally; otherwise colour is enabled only when ``stream`` is a TTY.

    Args:
        stream: The output stream to check. Defaults to ``sys.stdout``.

    Returns:
        ``True`` if colour output is appropriate.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if stream is None:
        import sys

        stream = sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class Ansi:
    """Apply ANSI styles, honouring an ``enabled`` flag.

    When disabled, every method returns its input unchanged, so the same code
    path produces plain text for pipes, files and tests.

    Args:
        enabled: Force styling on/off. When ``None`` (default), it is decided by
            :func:`supports_color`.
        stream: Stream used for the auto-detection when ``enabled`` is ``None``.
    """

    def __init__(
        self, enabled: bool | None = None, *, stream: IO[str] | None = None
    ) -> None:
        self.enabled = supports_color(stream) if enabled is None else enabled

    def apply(self, text: str, *styles: str) -> str:
        """Wrap ``text`` in the named ``styles`` (e.g. ``"bold"``, ``"cyan"``).

        Unknown style names are ignored. Returns ``text`` unchanged when styling
        is disabled or no valid styles are given.
        """
        if not self.enabled:
            return text
        codes = [_CODES[s] for s in styles if s in _CODES]
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}{_RESET}"

    def bold(self, text: str) -> str:
        """Return ``text`` in bold."""
        return self.apply(text, "bold")

    def dim(self, text: str) -> str:
        """Return ``text`` dimmed."""
        return self.apply(text, "dim")

    def red(self, text: str) -> str:
        """Return ``text`` in red."""
        return self.apply(text, "red")

    def green(self, text: str) -> str:
        """Return ``text`` in green."""
        return self.apply(text, "green")

    def yellow(self, text: str) -> str:
        """Return ``text`` in yellow."""
        return self.apply(text, "yellow")

    def cyan(self, text: str) -> str:
        """Return ``text`` in cyan."""
        return self.apply(text, "cyan")

    def grey(self, text: str) -> str:
        """Return ``text`` in grey."""
        return self.apply(text, "grey")
