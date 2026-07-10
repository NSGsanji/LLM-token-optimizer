"""Estimate the monetary cost of LLM requests from token counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..tokenization.base import TokenCounter
from ..tokenization.registry import get_counter
from ..tokenization.types import Message
from .errors import UnknownPricingError
from .pricing import ModelPricing, get_pricing


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The estimated cost of a request.

    Attributes:
        model: The model the estimate was computed for.
        input_tokens: Number of prompt (input) tokens.
        output_tokens: Number of completion (output) tokens.
        input_cost: Cost attributable to input tokens.
        output_cost: Cost attributable to output tokens.
        currency: ISO currency code for the amounts.
    """

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    currency: str = "USD"

    @property
    def total_cost(self) -> float:
        """The combined input and output cost."""
        return self.input_cost + self.output_cost

    @property
    def total_tokens(self) -> int:
        """The combined input and output token count."""
        return self.input_tokens + self.output_tokens


class CostEstimator:
    """Compute request costs for a model using a token pricing table.

    Args:
        pricing_table: An explicit pricing table (prefix -> ``ModelPricing``).
            When omitted, the built-in table plus any runtime overrides is used.
        counter: A token counter used by :meth:`estimate_messages`. When
            omitted, one is chosen per-model via the tokenization registry.

    Example:
        >>> estimator = CostEstimator()
        >>> estimate = estimator.estimate("gpt-4o", input_tokens=1000)
        >>> round(estimate.total_cost, 4)
        0.0025
    """

    def __init__(
        self,
        *,
        pricing_table: dict[str, ModelPricing] | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self._pricing_table = pricing_table
        self._counter = counter

    def pricing_for(self, model: str) -> ModelPricing:
        """Return the pricing for ``model`` or raise if it is unknown.

        Args:
            model: The model identifier.

        Returns:
            The matching :class:`ModelPricing`.

        Raises:
            UnknownPricingError: If no pricing is known for ``model``.
        """
        pricing = get_pricing(model, pricing_table=self._pricing_table)
        if pricing is None:
            raise UnknownPricingError(
                f"No pricing known for model {model!r}. Register it with "
                "smarttokenoptimizer.cost.register_pricing() or pass a "
                "pricing_table."
            )
        return pricing

    def estimate(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate the cost of a request with the given token counts.

        Args:
            model: The model identifier.
            input_tokens: Number of prompt tokens. Must be ``>= 0``.
            output_tokens: Number of completion tokens. Must be ``>= 0``.

        Returns:
            A :class:`CostEstimate` for the request.

        Raises:
            ValueError: If either token count is negative.
            UnknownPricingError: If no pricing is known for ``model``.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        pricing = self.pricing_for(model)
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=pricing.input_cost(input_tokens),
            output_cost=pricing.output_cost(output_tokens),
            currency=pricing.currency,
        )

    def estimate_messages(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        expected_output_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate the cost of sending ``messages`` as a prompt.

        Input tokens are measured with this estimator's counter (or a
        model-appropriate one). Output tokens are not known ahead of time, so
        supply ``expected_output_tokens`` to include a completion estimate.

        Args:
            messages: The prompt messages.
            model: The model identifier.
            expected_output_tokens: Assumed completion length, in tokens.

        Returns:
            A :class:`CostEstimate` for the request.

        Raises:
            UnknownPricingError: If no pricing is known for ``model``.
        """
        counter = self._counter if self._counter is not None else get_counter(model)
        input_tokens = counter.count_messages(messages)
        return self.estimate(
            model,
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
        )
