"""Credential selection strategies.

A strategy chooses one credential from the set currently *available* (enabled,
not rate-limited, not circuit-broken). Strategies may keep internal state (e.g.
a rotation cursor) but must be deterministic so behaviour is reproducible and
testable — no randomness is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .credential import Credential


@dataclass(frozen=True, slots=True)
class CredentialView:
    """A read-only snapshot of a candidate credential for selection.

    Attributes:
        credential: The candidate credential.
        uses: How many times the pool has handed this credential out so far.
    """

    credential: Credential
    uses: int

    @property
    def id(self) -> str:
        """The candidate credential's id."""
        return self.credential.id


class SelectionStrategy(ABC):
    """Interface for choosing a credential from available candidates."""

    @abstractmethod
    def select(self, candidates: Sequence[CredentialView]) -> Credential:
        """Return the chosen credential.

        Args:
            candidates: A non-empty sequence of available candidates, in a
                stable pool-defined order.

        Returns:
            One credential from ``candidates``.
        """
        raise NotImplementedError


class RoundRobinStrategy(SelectionStrategy):
    """Rotate through the available candidates in order.

    A monotonic internal cursor advances on every selection, so successive
    calls spread load evenly across whatever credentials are available at each
    call (unavailable credentials are simply skipped by the pool).
    """

    def __init__(self) -> None:
        self._cursor = 0

    def select(self, candidates: Sequence[CredentialView]) -> Credential:
        chosen = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return chosen.credential


class PriorityStrategy(SelectionStrategy):
    """Prefer the highest-priority credential.

    Ties are broken by fewest uses (to balance load among equal-priority keys),
    then by the pool's stable ordering.
    """

    def select(self, candidates: Sequence[CredentialView]) -> Credential:
        best = min(
            enumerate(candidates),
            key=lambda item: (-item[1].credential.priority, item[1].uses, item[0]),
        )
        return best[1].credential


class LeastUsedStrategy(SelectionStrategy):
    """Choose the credential handed out the fewest times so far.

    Ties are broken by the pool's stable ordering. This naturally balances load
    and favours freshly-recovered credentials.
    """

    def select(self, candidates: Sequence[CredentialView]) -> Credential:
        best = min(
            enumerate(candidates),
            key=lambda item: (item[1].uses, item[0]),
        )
        return best[1].credential


class WeightedRoundRobinStrategy(SelectionStrategy):
    """Smooth weighted round-robin over credential weights.

    Implements the deterministic "smooth weighted round-robin" algorithm (as
    used by nginx): each candidate accumulates its weight per call and the one
    with the highest running total is chosen, then has the total weight
    subtracted. Over time selections are distributed in proportion to weights,
    without bursts and without randomness.
    """

    def __init__(self) -> None:
        self._current: dict[str, float] = {}

    def select(self, candidates: Sequence[CredentialView]) -> Credential:
        total = 0.0
        best_id: str | None = None
        best_value = float("-inf")
        live_ids = set()

        for view in candidates:
            weight = view.credential.weight
            total += weight
            live_ids.add(view.id)
            running = self._current.get(view.id, 0.0) + weight
            self._current[view.id] = running
            if running > best_value:
                best_value = running
                best_id = view.id

        # Drop bookkeeping for candidates no longer present to avoid unbounded
        # growth as the available set changes over time.
        for stale in [cid for cid in self._current if cid not in live_ids]:
            del self._current[stale]

        assert best_id is not None  # candidates is non-empty
        self._current[best_id] -= total
        # Return the chosen credential object.
        for view in candidates:
            if view.id == best_id:
                return view.credential
        raise AssertionError(  # pragma: no cover - best_id always matches
            "unreachable: best_id must match a candidate"
        )
