"""A pool of credentials with rotation, failover and health tracking."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .credential import Credential
from .errors import (
    DuplicateCredentialError,
    NoAvailableCredentialError,
    UnknownCredentialError,
)
from .strategies import CredentialView, RoundRobinStrategy, SelectionStrategy


@dataclass
class _State:
    """Mutable per-credential health/usage state (pool-internal)."""

    uses: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    rate_limited_until: float | None = None
    circuit_open_until: float | None = None
    last_error: str | None = None

    def available_at(self, now: float) -> bool:
        rate_limited = (
            self.rate_limited_until is not None and now < self.rate_limited_until
        )
        circuit_open = (
            self.circuit_open_until is not None and now < self.circuit_open_until
        )
        return not (rate_limited or circuit_open)


@dataclass(frozen=True, slots=True)
class CredentialHealth:
    """An immutable snapshot of a credential's health within a pool.

    Attributes:
        id: The credential id.
        provider: The credential's provider, if any.
        enabled: Whether the credential is enabled.
        available: Whether it can currently be handed out.
        uses: Total times it has been handed out.
        successes: Recorded successes.
        failures: Recorded failures.
        consecutive_failures: Current consecutive-failure streak.
        rate_limited: Whether it is currently in rate-limit cooldown.
        circuit_open: Whether its circuit breaker is currently open.
        last_error: The most recent recorded error message, if any.
    """

    id: str
    provider: str | None
    enabled: bool
    available: bool
    uses: int
    successes: int
    failures: int
    consecutive_failures: int
    rate_limited: bool
    circuit_open: bool
    last_error: str | None


class CredentialPool:
    """A thread-safe pool of API credentials with rotation and failover.

    Credentials are handed out by :meth:`acquire` according to a pluggable
    :class:`SelectionStrategy`, skipping any that are disabled, in rate-limit
    cooldown, or whose circuit breaker is open. Callers report the outcome of
    each use so the pool can react:

    - :meth:`record_success` clears failure state.
    - :meth:`record_failure` counts failures and, past a threshold, opens the
      credential's circuit breaker for a cooldown period (failover).
    - :meth:`record_rate_limited` puts a credential in cooldown until a
      provider-supplied retry time.

    The :meth:`borrow` context manager wires success/failure reporting to
    ordinary control flow.

    Args:
        credentials: Optional initial credentials.
        strategy: Selection strategy. Defaults to :class:`RoundRobinStrategy`.
        failure_threshold: Consecutive failures that trip the circuit breaker.
            Must be positive. Defaults to ``5``.
        cooldown: Seconds a tripped circuit stays open. Must be positive.
            Defaults to ``30.0``.
        time_fn: Monotonic clock source (injectable for testing). Defaults to
            :func:`time.monotonic`.

    Raises:
        ValueError: If ``failure_threshold`` or ``cooldown`` is non-positive.
    """

    def __init__(
        self,
        credentials: list[Credential] | None = None,
        *,
        strategy: SelectionStrategy | None = None,
        failure_threshold: int = 5,
        cooldown: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if cooldown <= 0:
            raise ValueError("cooldown must be positive")
        self._strategy = strategy if strategy is not None else RoundRobinStrategy()
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._time = time_fn
        self._lock = threading.Lock()
        self._credentials: dict[str, Credential] = {}
        self._order: list[str] = []
        self._state: dict[str, _State] = {}
        for credential in credentials or []:
            self.add(credential)

    # -- Membership --------------------------------------------------------

    def add(self, credential: Credential) -> None:
        """Add ``credential`` to the pool.

        Raises:
            DuplicateCredentialError: If its id is already present.
        """
        with self._lock:
            if credential.id in self._credentials:
                raise DuplicateCredentialError(
                    f"credential {credential.id!r} is already in the pool"
                )
            self._credentials[credential.id] = credential
            self._order.append(credential.id)
            self._state[credential.id] = _State()

    def add_key(self, key: str, **kwargs: object) -> Credential:
        """Convenience: build a :class:`Credential` from ``key`` and add it.

        Returns:
            The created credential.
        """
        credential = Credential(key, **kwargs)  # type: ignore[arg-type]
        self.add(credential)
        return credential

    def remove(self, credential_id: str) -> None:
        """Remove a credential by id.

        Raises:
            UnknownCredentialError: If the id is not in the pool.
        """
        with self._lock:
            if credential_id not in self._credentials:
                raise UnknownCredentialError(credential_id)
            del self._credentials[credential_id]
            del self._state[credential_id]
            self._order.remove(credential_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._credentials)

    def __contains__(self, credential_id: object) -> bool:
        with self._lock:
            return credential_id in self._credentials

    def ids(self) -> list[str]:
        """Return the credential ids in insertion order."""
        with self._lock:
            return list(self._order)

    # -- Acquisition -------------------------------------------------------

    def acquire(self) -> Credential:
        """Return an available credential chosen by the strategy.

        Raises:
            NoAvailableCredentialError: If no credential is currently usable.
        """
        with self._lock:
            now = self._time()
            candidates = [
                CredentialView(self._credentials[cid], self._state[cid].uses)
                for cid in self._order
                if self._credentials[cid].enabled and self._state[cid].available_at(now)
            ]
            if not candidates:
                raise NoAvailableCredentialError(
                    "no credential is currently available (all disabled, "
                    "rate-limited, or circuit-open)"
                )
            chosen = self._strategy.select(candidates)
            self._state[chosen.id].uses += 1
            return chosen

    @contextmanager
    def borrow(self) -> Iterator[Credential]:
        """Acquire a credential, auto-recording success or failure.

        On clean exit the credential's success is recorded; if the body raises,
        a failure is recorded and the exception is re-raised.

        Yields:
            The acquired credential.
        """
        credential = self.acquire()
        try:
            yield credential
        except Exception as exc:
            self.record_failure(credential, error=str(exc))
            raise
        else:
            self.record_success(credential)

    # -- Outcome reporting -------------------------------------------------

    def record_success(self, credential: Credential | str) -> None:
        """Record a successful use, clearing failure and cooldown state."""
        with self._lock:
            state = self._state_for(credential)
            state.successes += 1
            state.consecutive_failures = 0
            state.circuit_open_until = None
            state.rate_limited_until = None
            state.last_error = None

    def record_failure(
        self, credential: Credential | str, *, error: str | None = None
    ) -> None:
        """Record a failed use; trips the circuit breaker past the threshold."""
        with self._lock:
            state = self._state_for(credential)
            state.failures += 1
            state.consecutive_failures += 1
            state.last_error = error
            if state.consecutive_failures >= self._failure_threshold:
                state.circuit_open_until = self._time() + self._cooldown

    def record_rate_limited(
        self, credential: Credential | str, *, retry_after: float
    ) -> None:
        """Put a credential in cooldown for ``retry_after`` seconds.

        Raises:
            ValueError: If ``retry_after`` is negative.
        """
        if retry_after < 0:
            raise ValueError("retry_after must be non-negative")
        with self._lock:
            state = self._state_for(credential)
            state.rate_limited_until = self._time() + retry_after

    # -- Introspection -----------------------------------------------------

    def available_ids(self) -> list[str]:
        """Return the ids of credentials currently available for use."""
        with self._lock:
            now = self._time()
            return [
                cid
                for cid in self._order
                if self._credentials[cid].enabled and self._state[cid].available_at(now)
            ]

    def health(self) -> list[CredentialHealth]:
        """Return a health snapshot for every credential, in pool order."""
        with self._lock:
            now = self._time()
            snapshots = []
            for cid in self._order:
                cred = self._credentials[cid]
                state = self._state[cid]
                rate_limited = (
                    state.rate_limited_until is not None
                    and now < state.rate_limited_until
                )
                circuit_open = (
                    state.circuit_open_until is not None
                    and now < state.circuit_open_until
                )
                snapshots.append(
                    CredentialHealth(
                        id=cid,
                        provider=cred.provider,
                        enabled=cred.enabled,
                        available=cred.enabled and state.available_at(now),
                        uses=state.uses,
                        successes=state.successes,
                        failures=state.failures,
                        consecutive_failures=state.consecutive_failures,
                        rate_limited=rate_limited,
                        circuit_open=circuit_open,
                        last_error=state.last_error,
                    )
                )
            return snapshots

    # -- Internal ----------------------------------------------------------

    def _state_for(self, credential: Credential | str) -> _State:
        cid = credential.id if isinstance(credential, Credential) else credential
        state = self._state.get(cid)
        if state is None:
            raise UnknownCredentialError(cid)
        return state
