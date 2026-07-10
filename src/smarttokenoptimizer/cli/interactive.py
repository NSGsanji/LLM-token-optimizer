"""A small, testable numbered-menu primitive for interactive terminal UIs.

Rather than depend on ``curses`` (which needs a real TTY and is awkward to
test), the menu uses numbered selection: it renders a titled list and reads a
line of input. This works in every terminal, over pipes, and in tests — where
input and output streams are injected. Rendering is a pure function; the
:class:`Menu` loop drives it.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import IO

from .style import Ansi


@dataclass(frozen=True, slots=True)
class MenuItem:
    """One selectable menu entry.

    Attributes:
        label: The text shown for the option.
        description: Optional secondary text shown dimmed after the label.
        value: An arbitrary payload returned when the item is chosen. Defaults
            to the label.
    """

    label: str
    description: str = ""
    value: object = None

    @property
    def payload(self) -> object:
        """The value to return on selection (``label`` if none was given)."""
        return self.label if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class MenuResult:
    """The outcome of running a :class:`Menu`.

    Attributes:
        index: The zero-based index of the chosen item, or ``-1`` if cancelled.
        item: The chosen :class:`MenuItem`, or ``None`` if cancelled.
    """

    index: int
    item: MenuItem | None = field(default=None)

    @property
    def cancelled(self) -> bool:
        """Whether the menu was cancelled (quit / EOF) rather than chosen."""
        return self.item is None


def render_menu(
    title: str,
    items: Sequence[MenuItem],
    *,
    ansi: Ansi | None = None,
    footer: str = "Enter a number, or 'q' to quit.",
) -> str:
    """Return the menu as a styled multi-line string (pure, no I/O).

    Args:
        title: The heading shown above the options.
        items: The selectable items, rendered as a numbered list.
        ansi: Styler to use; a disabled styler is used when ``None``.
        footer: Hint line shown below the options.

    Returns:
        The rendered menu.
    """
    styler = ansi if ansi is not None else Ansi(enabled=False)
    lines = [styler.bold(title), ""]
    width = len(str(len(items)))
    for number, item in enumerate(items, start=1):
        marker = styler.cyan(f"{number:>{width}})")
        line = f"  {marker} {item.label}"
        if item.description:
            line += "  " + styler.grey(f"— {item.description}")
        lines.append(line)
    lines.append("")
    lines.append(styler.dim(footer))
    return "\n".join(lines)


class Menu:
    """Render a numbered menu and read the user's choice.

    Args:
        title: The menu heading.
        items: The selectable items (must be non-empty).
        ansi: Styler; auto-detected against ``output`` when ``None``.
        input: Input stream to read choices from. Defaults to ``sys.stdin``.
        output: Output stream to render to. Defaults to ``sys.stdout``.
        prompt: The input prompt string.

    Raises:
        ValueError: If ``items`` is empty.
    """

    def __init__(
        self,
        title: str,
        items: Sequence[MenuItem],
        *,
        ansi: Ansi | None = None,
        input: IO[str] | None = None,
        output: IO[str] | None = None,
        prompt: str = "> ",
    ) -> None:
        if not items:
            raise ValueError("a menu needs at least one item")
        self._title = title
        self._items = list(items)
        self._in = input if input is not None else sys.stdin
        self._out = output if output is not None else sys.stdout
        self._ansi = ansi if ansi is not None else Ansi(stream=self._out)
        self._prompt = prompt

    def run(self) -> MenuResult:
        """Display the menu and loop until a valid choice or a quit/EOF.

        Returns:
            A :class:`MenuResult`; ``cancelled`` is ``True`` on ``q``/EOF.
        """
        self._out.write(render_menu(self._title, self._items, ansi=self._ansi) + "\n")
        while True:
            self._out.write(self._prompt)
            self._out.flush()
            raw = self._in.readline()
            if raw == "":  # EOF
                return MenuResult(index=-1)
            choice = raw.strip().lower()
            if choice in ("q", "quit", "exit"):
                return MenuResult(index=-1)
            if choice.isdigit():
                number = int(choice)
                if 1 <= number <= len(self._items):
                    index = number - 1
                    return MenuResult(index=index, item=self._items[index])
            self._out.write(self._ansi.red(f"Invalid choice: {raw.strip()!r}") + "\n")
