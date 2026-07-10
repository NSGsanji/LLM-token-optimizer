"""Exceptions raised by the routing subpackage."""

from __future__ import annotations

from ..credentials.errors import CredentialError


class RoutingError(Exception):
    """Base class for all routing-related errors."""


class NoAvailableProviderError(RoutingError):
    """Raised when no provider can serve a request.

    This means every candidate provider is disabled, does not serve the
    requested model, or has no usable credential (all rate-limited or with an
    open circuit breaker). Like :class:`CredentialError`, the condition is
    usually transient.
    """


class DuplicateProviderError(RoutingError):
    """Raised when registering a provider whose name already exists."""


__all__ = [
    "CredentialError",
    "DuplicateProviderError",
    "NoAvailableProviderError",
    "RoutingError",
]
