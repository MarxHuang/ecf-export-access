"""Minimal schema-aware mapping codec; unknown fields fail closed."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from r4r.errors import SerializationError
from .canonical_json import canonical_value


def encode(value: Any) -> Any:
    return canonical_value(value)


def decode_mapping(payload: Mapping[str, Any], *, required: Iterable[str] = (), allowed: Iterable[str] | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SerializationError("object payload must be a mapping")
    required_set = set(required)
    allowed_set = set(allowed) if allowed is not None else None
    missing = sorted(required_set - set(payload))
    if missing:
        raise SerializationError("missing required fields: " + ",".join(missing))
    if allowed_set is not None:
        unknown = sorted(set(payload) - allowed_set)
        if unknown:
            raise SerializationError("unknown fields: " + ",".join(unknown))
    return {str(key): canonical_value(payload[key]) for key in sorted(payload)}
