"""Provider selection policies.

A policy *ranks* candidate providers best-first for a given request. The router
then tries them in order, so a policy expresses preference while the router
still provides failover to the next-best provider when the top choice has no
usable credential.

All policies are deterministic (no randomness) so routing is reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .provider import Provider


class RoutingPolicy(ABC):
    """Interface for ranking candidate providers for a request."""

    @abstractmethod
    def rank(
        self, candidates: Sequence[Provider], *, model: str | None
    ) -> list[Provider]:
        """Return ``candidates`` ordered most- to least-preferred.

        Args:
            candidates: Providers that are enabled and serve the model, in a
                stable registration order.
            model: The requested model, or ``None`` if unspecified.

        Returns:
            A new list containing the same providers, ordered by preference.
        """
        raise NotImplementedError


class PriorityPolicy(RoutingPolicy):
    """Prefer the highest-priority provider (ties broken by registration order)."""

    def rank(
        self, candidates: Sequence[Provider], *, model: str | None
    ) -> list[Provider]:
        ranked = sorted(
            enumerate(candidates), key=lambda item: (-item[1].priority, item[0])
        )
        return [provider for _, provider in ranked]


class RoundRobinPolicy(RoutingPolicy):
    """Rotate through candidates so load spreads evenly across providers."""

    def __init__(self) -> None:
        self._cursor = 0

    def rank(
        self, candidates: Sequence[Provider], *, model: str | None
    ) -> list[Provider]:
        n = len(candidates)
        offset = self._cursor % n
        self._cursor += 1
        ordered = list(candidates)
        return ordered[offset:] + ordered[:offset]


class CheapestPolicy(RoutingPolicy):
    """Prefer the provider with the lowest price for the requested model.

    Prices come from each provider's ``price_hint`` or, failing that, the
    built-in pricing table. Providers with no known price are ranked last;
    ties are broken by higher priority, then registration order.
    """

    def rank(
        self, candidates: Sequence[Provider], *, model: str | None
    ) -> list[Provider]:
        def key(item: tuple[int, Provider]) -> tuple[float, int, int]:
            index, provider = item
            price = provider.effective_price(model)
            # Unknown price sorts last via infinity.
            price_key = price if price is not None else float("inf")
            return (price_key, -provider.priority, index)

        return [p for _, p in sorted(enumerate(candidates), key=key)]


class LowestLatencyPolicy(RoutingPolicy):
    """Prefer the provider with the lowest observed average latency.

    Providers with no latency samples yet are treated as latency ``0`` so they
    are tried first — this ensures every provider gets measured (exploration)
    before the policy settles on the fastest. Ties are broken by higher
    priority, then registration order.
    """

    def rank(
        self, candidates: Sequence[Provider], *, model: str | None
    ) -> list[Provider]:
        def key(item: tuple[int, Provider]) -> tuple[float, int, int]:
            index, provider = item
            latency = provider.avg_latency
            latency_key = latency if latency is not None else 0.0
            return (latency_key, -provider.priority, index)

        return [p for _, p in sorted(enumerate(candidates), key=key)]
