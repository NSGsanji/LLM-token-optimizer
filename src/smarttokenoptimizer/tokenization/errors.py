"""Exceptions raised by the tokenization subpackage."""

from __future__ import annotations


class TokenizationError(Exception):
    """Base class for all tokenization-related errors."""


class BackendUnavailableError(TokenizationError):
    """Raised when an optional tokenizer backend is requested but unavailable.

    Typically this means an optional dependency (such as ``tiktoken``) is not
    installed. Callers can catch this to fall back to the dependency-free
    heuristic counter.
    """


class UnknownModelError(TokenizationError):
    """Raised when a model name cannot be mapped to a tokenizer encoding."""
