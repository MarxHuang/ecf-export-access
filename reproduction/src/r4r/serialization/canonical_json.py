"""canonical-json-v1: deterministic and fail-closed JSON representation."""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from r4r.errors import SerializationError


def canonical_value(value: Any) -> Any:
    """Convert a registered value to JSON-compatible canonical data."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("non-finite numbers are forbidden")
        return 0.0 if value == 0 else value
    if isinstance(value, Enum):
        return canonical_value(value.value)
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return canonical_value(to_json())
    if dataclasses.is_dataclass(value):
        return canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SerializationError("canonical JSON object keys must be strings")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    raise SerializationError(f"unregistered object type: {type(value).__name__}")


def canonical_dumps(value: Any) -> str:
    return json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")
