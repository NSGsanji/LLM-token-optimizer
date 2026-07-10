"""Per-model token pricing.

Prices are expressed in **US dollars per one million tokens**, the unit most
providers quote. The built-in table is a convenience snapshot and is inevitably
approximate — provider prices change and vary by region and tier. Treat these as
sensible defaults and override them for anything billing-critical via
:class:`CostEstimator`'s ``pricing_table`` argument or :func:`register_pricing`.

The prices below were compiled in early 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Divisor to convert a "per million tokens" rate into a per-token rate.
TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Input/output token pricing for a model, in USD per million tokens.

    Attributes:
        input_per_million: Cost of one million prompt (input) tokens, in USD.
        output_per_million: Cost of one million completion (output) tokens, USD.
        currency: ISO currency code for the amounts. Defaults to ``"USD"``.
    """

    input_per_million: float
    output_per_million: float
    currency: str = "USD"

    def input_cost(self, tokens: int) -> float:
        """Return the cost of ``tokens`` input tokens."""
        return tokens / TOKENS_PER_PRICE_UNIT * self.input_per_million

    def output_cost(self, tokens: int) -> float:
        """Return the cost of ``tokens`` output tokens."""
        return tokens / TOKENS_PER_PRICE_UNIT * self.output_per_million


# Built-in pricing table, keyed by model-name prefix (most specific first). USD
# per million tokens, compiled early 2025 — see module docstring.
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60),
    "gpt-4.1": ModelPricing(2.00, 8.00),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "o1-mini": ModelPricing(1.10, 4.40),
    "o1": ModelPricing(15.00, 60.00),
    "o3-mini": ModelPricing(1.10, 4.40),
    # Anthropic — current generation (USD per million tokens)
    "claude-fable-5": ModelPricing(10.00, 50.00),
    "claude-opus-4": ModelPricing(5.00, 25.00),  # 4.6 / 4.7 / 4.8 share this rate
    "claude-sonnet-5": ModelPricing(3.00, 15.00),
    "claude-sonnet-4": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
    # Anthropic — Claude 3 family
    "claude-3-5-haiku": ModelPricing(0.80, 4.00),
    "claude-3-5-sonnet": ModelPricing(3.00, 15.00),
    "claude-3-haiku": ModelPricing(0.25, 1.25),
    "claude-3-opus": ModelPricing(15.00, 75.00),
    "claude-3-sonnet": ModelPricing(3.00, 15.00),
    # Google
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),
}

# Overrides/additions registered at runtime take precedence over the defaults.
_CUSTOM_PRICING: dict[str, ModelPricing] = {}


def register_pricing(prefix: str, pricing: ModelPricing) -> None:
    """Register or override pricing for a model-name prefix.

    Args:
        prefix: A model-name prefix (matched case-insensitively) such as
            ``"gpt-4o"`` or a full model id.
        pricing: The pricing to associate with the prefix.
    """
    _CUSTOM_PRICING[prefix.strip().lower()] = pricing


def clear_custom_pricing() -> None:
    """Remove all runtime-registered pricing overrides."""
    _CUSTOM_PRICING.clear()


def get_pricing(
    model: str,
    *,
    pricing_table: dict[str, ModelPricing] | None = None,
) -> ModelPricing | None:
    """Return the pricing best matching ``model``, or ``None`` if unknown.

    Matching is by longest case-insensitive prefix, so ``"gpt-4o-mini-2024"``
    resolves to the ``"gpt-4o-mini"`` entry rather than ``"gpt-4o"``. Runtime
    overrides (see :func:`register_pricing`) take precedence over the built-in
    table unless an explicit ``pricing_table`` is supplied.

    Args:
        model: The model identifier to look up.
        pricing_table: An explicit table to search instead of the built-in one
            plus runtime overrides.

    Returns:
        The matching :class:`ModelPricing`, or ``None`` when no prefix matches.
    """
    normalized = model.strip().lower()
    if pricing_table is not None:
        table = pricing_table
    else:
        table = {**_DEFAULT_PRICING, **_CUSTOM_PRICING}

    best: ModelPricing | None = None
    best_len = -1
    for prefix, pricing in table.items():
        key = prefix.lower()
        if normalized.startswith(key) and len(key) > best_len:
            best = pricing
            best_len = len(key)
    return best
