"""Provider routing: pick a provider/credential per request.

This subpackage composes credential pools and cost data into a router that
chooses which provider to send a request to, based on a pluggable policy
(priority, round-robin, cheapest, or lowest latency) with automatic failover
across providers.

Example:
    >>> from smarttokenoptimizer.credentials import CredentialPool
    >>> from smarttokenoptimizer.routing import Provider, Router, CheapestPolicy
    >>> openai = Provider("openai", pool=CredentialPool(), models=["gpt-4o"])
    >>> _ = openai.pool.add_key("sk-openai")
    >>> router = Router([openai], policy=CheapestPolicy())
    >>> with router.dispatch(model="gpt-4o") as route:
    ...     ...  # call route.provider using route.key
"""

from __future__ import annotations

from .errors import (
    DuplicateProviderError,
    NoAvailableProviderError,
    RoutingError,
)
from .policies import (
    CheapestPolicy,
    LowestLatencyPolicy,
    PriorityPolicy,
    RoundRobinPolicy,
    RoutingPolicy,
)
from .provider import Provider
from .router import Route, Router

__all__ = [
    "CheapestPolicy",
    "DuplicateProviderError",
    "LowestLatencyPolicy",
    "NoAvailableProviderError",
    "PriorityPolicy",
    "Provider",
    "RoundRobinPolicy",
    "Route",
    "Router",
    "RoutingError",
    "RoutingPolicy",
]
