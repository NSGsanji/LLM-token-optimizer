"""The :class:`Provider` descriptor used by the router."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from ..cost.pricing import ModelPricing, get_pricing
from ..credentials.pool import CredentialPool


class Provider:
    """A configured LLM provider endpoint the router can dispatch to.

    A provider bundles the credentials used to reach it (a
    :class:`~smarttokenoptimizer.credentials.CredentialPool`) with routing
    metadata: which models it serves, its selection priority/weight, an optional
    price hint, and a running estimate of its observed latency.

    Args:
        name: A unique provider name (e.g. ``"openai"``, ``"openrouter"``).
        pool: The credential pool used to authenticate to this provider.
        models: The models this provider serves. An empty/omitted set means the
            provider is considered able to serve *any* model.
        priority: Selection priority; **higher is preferred**. Defaults to ``0``.
        weight: Relative weight for weighted policies. Must be positive.
        enabled: Whether the provider may be routed to. Defaults to ``True``.
        price_hint: Optional combined price (USD per million tokens, input +
            output) used by the cheapest policy. When omitted, the cheapest
            policy falls back to the built-in pricing table for the model.
        latency_alpha: Smoothing factor for the latency EWMA, in ``(0, 1]``.
            Higher reacts faster to recent samples. Defaults to ``0.3``.

    Raises:
        ValueError: If ``name`` is empty, ``weight`` is non-positive, or
            ``latency_alpha`` is outside ``(0, 1]``.
    """

    def __init__(
        self,
        name: str,
        *,
        pool: CredentialPool,
        models: Iterable[str] | None = None,
        priority: int = 0,
        weight: float = 1.0,
        enabled: bool = True,
        price_hint: float | None = None,
        latency_alpha: float = 0.3,
    ) -> None:
        if not name:
            raise ValueError("provider name must be a non-empty string")
        if weight <= 0:
            raise ValueError("provider weight must be positive")
        if not 0.0 < latency_alpha <= 1.0:
            raise ValueError("latency_alpha must be in the interval (0, 1]")
        self.name = name
        self.pool = pool
        self.models = frozenset(models) if models is not None else frozenset()
        self.priority = priority
        self.weight = weight
        self.enabled = enabled
        self.price_hint = price_hint
        self._alpha = latency_alpha
        self._lock = threading.Lock()
        self._avg_latency: float | None = None
        self._samples = 0

    def serves(self, model: str | None) -> bool:
        """Return whether this provider can serve ``model``.

        A provider with no declared models serves any model. When ``model`` is
        ``None`` the provider is always considered a match.
        """
        if model is None or not self.models:
            return True
        return model in self.models

    @property
    def available(self) -> bool:
        """Whether the provider is enabled and has a usable credential."""
        return self.enabled and bool(self.pool.available_ids())

    @property
    def avg_latency(self) -> float | None:
        """The exponentially-weighted average observed latency, or ``None``."""
        with self._lock:
            return self._avg_latency

    @property
    def latency_samples(self) -> int:
        """The number of latency samples recorded so far."""
        with self._lock:
            return self._samples

    def record_latency(self, seconds: float) -> None:
        """Record an observed request latency and update the EWMA.

        Args:
            seconds: The measured latency. Must be non-negative.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError("latency must be non-negative")
        with self._lock:
            if self._avg_latency is None:
                self._avg_latency = seconds
            else:
                self._avg_latency = (
                    self._alpha * seconds + (1 - self._alpha) * self._avg_latency
                )
            self._samples += 1

    def effective_price(self, model: str | None) -> float | None:
        """Return the comparable price for ``model`` (USD per million tokens).

        Uses :attr:`price_hint` when set, otherwise the built-in pricing table
        (input + output rate). Returns ``None`` when no price is known.
        """
        if self.price_hint is not None:
            return self.price_hint
        if model is None:
            return None
        pricing: ModelPricing | None = get_pricing(model)
        if pricing is None:
            return None
        return pricing.input_per_million + pricing.output_per_million

    def __repr__(self) -> str:
        return (
            f"Provider(name={self.name!r}, models={sorted(self.models)}, "
            f"priority={self.priority}, enabled={self.enabled})"
        )
