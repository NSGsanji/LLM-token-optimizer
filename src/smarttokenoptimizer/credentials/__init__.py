"""AI credential management: pools, rotation, failover and health tracking.

This subpackage manages multiple API keys across providers:

- :class:`Credential` — a secret key plus selection metadata, with strict
  secret hygiene (the key never appears in ``repr`` / ``str`` / logs).
- Selection strategies — :class:`RoundRobinStrategy`, :class:`PriorityStrategy`,
  :class:`LeastUsedStrategy`, :class:`WeightedRoundRobinStrategy`.
- :class:`CredentialPool` — hand out healthy credentials, with rate-limit
  cooldown, a circuit breaker for failover, and per-credential health snapshots.

Example:
    >>> from smarttokenoptimizer.credentials import CredentialPool
    >>> pool = CredentialPool()
    >>> _ = pool.add_key("sk-aaa", provider="openai")
    >>> _ = pool.add_key("sk-bbb", provider="openai")
    >>> with pool.borrow() as credential:
    ...     ...  # use credential.key to make a request
"""

from __future__ import annotations

from .credential import Credential
from .errors import (
    CredentialError,
    DuplicateCredentialError,
    NoAvailableCredentialError,
    UnknownCredentialError,
)
from .pool import CredentialHealth, CredentialPool
from .strategies import (
    CredentialView,
    LeastUsedStrategy,
    PriorityStrategy,
    RoundRobinStrategy,
    SelectionStrategy,
    WeightedRoundRobinStrategy,
)

__all__ = [
    "Credential",
    "CredentialError",
    "CredentialHealth",
    "CredentialPool",
    "CredentialView",
    "DuplicateCredentialError",
    "LeastUsedStrategy",
    "NoAvailableCredentialError",
    "PriorityStrategy",
    "RoundRobinStrategy",
    "SelectionStrategy",
    "UnknownCredentialError",
    "WeightedRoundRobinStrategy",
]
