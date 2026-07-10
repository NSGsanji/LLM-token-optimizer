"""The :class:`Credential` value type, with secret-hygiene guarantees."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _mask(secret: str) -> str:
    """Return a non-reversible, display-safe masked form of ``secret``.

    Short secrets are fully masked; longer ones reveal only a few leading and
    trailing characters so they can be told apart in logs without exposing the
    material.
    """
    if len(secret) <= 8:
        return "****"
    return f"{secret[:3]}…{secret[-4:]}"


@dataclass(slots=True)
class Credential:
    """An API credential (secret key) plus its selection metadata.

    The secret ``key`` is treated as sensitive: it is excluded from ``repr``
    and is never rendered by :meth:`__str__`, so a credential can be safely
    logged or included in tracebacks without leaking the key. Access the raw
    value explicitly via :attr:`key` only when making a request.

    Args:
        key: The secret API key. Required and non-empty.
        id: A stable identifier used across a pool. When omitted, a
            deterministic id is derived from the key (without exposing it).
        provider: Optional provider name (e.g. ``"openai"``) this key is for.
        priority: Selection priority; **higher values are preferred** by
            priority-based strategies. Defaults to ``0``.
        weight: Relative weight for weighted strategies. Must be positive.
            Defaults to ``1.0``.
        enabled: Whether the credential may be handed out. Defaults to ``True``.
        metadata: Free-form, non-secret metadata. Do **not** store secrets here;
            it is included in ``repr``.

    Raises:
        ValueError: If ``key`` is empty or ``weight`` is not positive.
    """

    key: str = field(repr=False)
    id: str = ""
    provider: str | None = None
    priority: int = 0
    weight: float = 1.0
    enabled: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("credential key must be a non-empty string")
        if self.weight <= 0:
            raise ValueError("credential weight must be positive")
        if not self.id:
            self.id = self._derive_id(self.key)

    @staticmethod
    def _derive_id(key: str) -> str:
        """Derive a stable, non-revealing id from the secret key."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"cred-{digest[:12]}"

    @property
    def masked_key(self) -> str:
        """A display-safe, non-reversible masked form of the key."""
        return _mask(self.key)

    def __str__(self) -> str:
        provider = f" provider={self.provider}" if self.provider else ""
        return f"Credential(id={self.id}{provider} key={self.masked_key})"
