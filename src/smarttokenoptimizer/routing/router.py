"""The :class:`Router` that selects a provider and credential per request."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..credentials.credential import Credential
from ..credentials.errors import NoAvailableCredentialError
from .errors import DuplicateProviderError, NoAvailableProviderError
from .policies import PriorityPolicy, RoutingPolicy
from .provider import Provider


@dataclass(frozen=True, slots=True)
class Route:
    """The result of routing a request: a provider and a chosen credential.

    Attributes:
        provider: The selected provider.
        credential: A credential acquired from the provider's pool.
    """

    provider: Provider
    credential: Credential

    @property
    def key(self) -> str:
        """Shortcut for the acquired credential's secret key."""
        return self.credential.key


class Router:
    """Select a provider and credential for each request, with failover.

    The router holds a set of :class:`Provider` objects and a
    :class:`RoutingPolicy`. For each request it ranks the providers that serve
    the requested model and hands out a credential from the first one with
    availability — automatically failing over to the next-ranked provider when a
    provider's credentials are all rate-limited or circuit-broken.

    Args:
        providers: Optional initial providers.
        policy: The ranking policy. Defaults to :class:`PriorityPolicy`.
        time_fn: Monotonic clock source (injectable for testing). Defaults to
            :func:`time.monotonic`.
    """

    def __init__(
        self,
        providers: list[Provider] | None = None,
        *,
        policy: RoutingPolicy | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy if policy is not None else PriorityPolicy()
        self._time = time_fn
        self._lock = threading.Lock()
        self._providers: dict[str, Provider] = {}
        self._order: list[str] = []
        for provider in providers or []:
            self.add(provider)

    # -- Membership --------------------------------------------------------

    def add(self, provider: Provider) -> None:
        """Register ``provider``.

        Raises:
            DuplicateProviderError: If a provider with the same name exists.
        """
        with self._lock:
            if provider.name in self._providers:
                raise DuplicateProviderError(
                    f"provider {provider.name!r} is already registered"
                )
            self._providers[provider.name] = provider
            self._order.append(provider.name)

    def remove(self, name: str) -> None:
        """Unregister a provider by name.

        Raises:
            KeyError: If no provider with that name is registered.
        """
        with self._lock:
            del self._providers[name]
            self._order.remove(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._providers)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._providers

    def provider_names(self) -> list[str]:
        """Return registered provider names in registration order."""
        with self._lock:
            return list(self._order)

    # -- Routing -----------------------------------------------------------

    def route(self, *, model: str | None = None) -> Route:
        """Select a provider and acquire a credential for ``model``.

        Args:
            model: The model to route for, or ``None`` to consider all
                providers.

        Returns:
            A :class:`Route` with the chosen provider and credential.

        Raises:
            NoAvailableProviderError: If no enabled provider serves the model
                with an available credential.
        """
        with self._lock:
            candidates = [
                self._providers[name]
                for name in self._order
                if self._providers[name].enabled and self._providers[name].serves(model)
            ]
            ranked = self._policy.rank(candidates, model=model)

        for provider in ranked:
            try:
                credential = provider.pool.acquire()
            except NoAvailableCredentialError:
                continue
            return Route(provider=provider, credential=credential)

        raise NoAvailableProviderError(
            f"no provider is available to serve model {model!r}"
            if model is not None
            else "no provider is available"
        )

    @contextmanager
    def dispatch(self, *, model: str | None = None) -> Iterator[Route]:
        """Route a request, recording success/failure and latency automatically.

        On clean exit the credential's success is recorded and the elapsed time
        is fed into the provider's latency estimate. If the body raises, a
        failure is recorded on the credential and the exception re-raised.

        Args:
            model: The model to route for.

        Yields:
            The :class:`Route` to use for the request.
        """
        route = self.route(model=model)
        start = self._time()
        try:
            yield route
        except Exception as exc:
            route.provider.pool.record_failure(route.credential, error=str(exc))
            raise
        else:
            route.provider.pool.record_success(route.credential)
            route.provider.record_latency(self._time() - start)

    # -- Introspection -----------------------------------------------------

    def available_providers(self, *, model: str | None = None) -> list[str]:
        """Return names of providers that can currently serve ``model``."""
        with self._lock:
            return [
                name
                for name in self._order
                if self._providers[name].serves(model)
                and self._providers[name].available
            ]
