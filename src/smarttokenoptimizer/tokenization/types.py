"""Shared types for the tokenization subpackage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict, total=False):
    """A single chat message in the OpenAI-compatible format.

    Only ``role`` and ``content`` are commonly required by providers; ``name``
    is optional. Additional keys are permitted by most APIs and are counted
    generically by the token counters (their string values contribute tokens).
    """

    role: str
    content: str
    name: str


@dataclass(frozen=True, slots=True)
class MessageOverhead:
    """Per-message bookkeeping tokens added by chat completion formats.

    Chat APIs wrap each message in structural tokens (role delimiters, message
    separators) and prime the model for a reply. These constants mirror the
    accounting documented by OpenAI for chat models and are a close
    approximation for other OpenAI-compatible providers.

    Attributes:
        tokens_per_message: Fixed tokens added for every message.
        tokens_per_name: Extra tokens added when a message carries a ``name``.
        reply_priming: Tokens added once for the assistant's reply priming.
    """

    tokens_per_message: int = 3
    tokens_per_name: int = 1
    reply_priming: int = 3


#: Default overhead used when a model-specific value is not known. Matches the
#: accounting used by current OpenAI chat models (gpt-3.5-turbo / gpt-4 / gpt-4o).
DEFAULT_OVERHEAD = MessageOverhead()
