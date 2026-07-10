"""Exceptions raised by the credentials subpackage."""

from __future__ import annotations


class CredentialError(Exception):
    """Base class for all credential-related errors."""


class NoAvailableCredentialError(CredentialError):
    """Raised when a pool has no usable credential to hand out.

    This typically means every credential is disabled, rate-limited, or has
    tripped its circuit breaker. The condition is usually transient — callers
    can back off and retry once a cooldown elapses.
    """


class DuplicateCredentialError(CredentialError):
    """Raised when adding a credential whose id already exists in a pool."""


class UnknownCredentialError(CredentialError):
    """Raised when referring to a credential that is not in the pool."""
