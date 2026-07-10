"""Stable cache-key generation for prompts.

A cache key must be *stable* (the same logical request always maps to the same
key, regardless of dict ordering) and *collision-resistant*. We build a
canonical representation of the request and hash it with SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..tokenization.types import Message


def make_key(
    model: str,
    messages: Sequence[Message],
    **params: Any,
) -> str:
    """Return a stable cache key for a prompt request.

    The key is deterministic across processes and insensitive to the ordering
    of ``params`` and of keys within message/param dictionaries, so logically
    identical requests always collide (a cache *hit*) while different requests
    do not.

    Args:
        model: The model identifier the request targets.
        messages: The chat messages that make up the prompt.
        **params: Any additional request parameters that affect the response
            (e.g. ``temperature``, ``max_tokens``, ``tools``). Parameters whose
            value is ``None`` are ignored so that omitting a parameter and
            passing it as ``None`` produce the same key.

    Returns:
        A 64-character hexadecimal SHA-256 digest.
    """
    payload = {
        "model": model,
        "messages": [_canonical(dict(m)) for m in messages],
        "params": _canonical({k: v for k, v in params.items() if v is not None}),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-canonicalisable structure.

    Mappings and sequences are normalised so that ``json.dumps(sort_keys=True)``
    yields a stable string. Sets are sorted; other non-JSON types fall back to
    their ``repr`` to remain deterministic.
    """
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, str | bytes):
        return value.decode() if isinstance(value, bytes) else value
    if isinstance(value, set | frozenset):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, Sequence):
        return [_canonical(v) for v in value]
    if isinstance(value, bool | int | float) or value is None:
        return value
    return repr(value)
