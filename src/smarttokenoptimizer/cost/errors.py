"""Exceptions raised by the cost subpackage."""

from __future__ import annotations


class CostError(Exception):
    """Base class for all cost-related errors."""


class UnknownPricingError(CostError):
    """Raised when no pricing is known for a requested model."""
