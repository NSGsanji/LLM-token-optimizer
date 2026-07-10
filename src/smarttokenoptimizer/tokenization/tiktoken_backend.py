"""Exact token counting backed by OpenAI's :mod:`tiktoken`.

This backend is optional. It is only importable when ``tiktoken`` is installed
(``pip install "smarttokenoptimizer[tiktoken]"``). When the dependency is
missing, constructing :class:`TiktokenCounter` raises a clear
:class:`~smarttokenoptimizer.tokenization.errors.BackendUnavailableError` so
callers can fall back to the heuristic counter.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from .base import TokenCounter
from .errors import BackendUnavailableError
from .types import DEFAULT_OVERHEAD, MessageOverhead


@cache
def _load_encoding(name: str) -> Any:
    """Load and cache a tiktoken encoding by its canonical name.

    Args:
        name: A tiktoken encoding name (e.g. ``"cl100k_base"``).

    Returns:
        The loaded ``tiktoken.Encoding`` instance.

    Raises:
        BackendUnavailableError: If ``tiktoken`` is not installed.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise BackendUnavailableError(
            "tiktoken is not installed. Install it with "
            '`pip install "smarttokenoptimizer[tiktoken]"` to use exact '
            "token counting, or use HeuristicTokenCounter instead."
        ) from exc
    return tiktoken.get_encoding(name)


class TiktokenCounter(TokenCounter):
    """Exact token counter using a tiktoken BPE encoding.

    Args:
        encoding_name: The tiktoken encoding to use. Defaults to
            ``"cl100k_base"``, which powers GPT-3.5/GPT-4 family models.
        overhead: Per-message overhead accounting for chat formats.

    Raises:
        BackendUnavailableError: If ``tiktoken`` is not installed.

    Example:
        >>> counter = TiktokenCounter()          # doctest: +SKIP
        >>> counter.count_text("Hello, world!")  # doctest: +SKIP
        4
    """

    exact = True

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        *,
        overhead: MessageOverhead = DEFAULT_OVERHEAD,
    ) -> None:
        super().__init__(overhead=overhead)
        self._encoding_name = encoding_name
        # Eagerly load so construction fails fast if the backend is missing.
        self._encoding = _load_encoding(encoding_name)

    @property
    def encoding_name(self) -> str:
        """The name of the tiktoken encoding backing this counter."""
        return self._encoding_name

    def count_text(self, text: str) -> int:
        """Return the exact number of tokens in ``text``.

        Args:
            text: The raw string to tokenize.

        Returns:
            The exact token count.
        """
        if not text:
            return 0
        # ``disallowed_special=()`` ensures special-token sequences appearing in
        # user text are encoded as ordinary text rather than raising.
        return len(self._encoding.encode(text, disallowed_special=()))


def is_available() -> bool:
    """Return whether the tiktoken backend can be used.

    Returns:
        ``True`` if ``tiktoken`` is importable, ``False`` otherwise.
    """
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True
