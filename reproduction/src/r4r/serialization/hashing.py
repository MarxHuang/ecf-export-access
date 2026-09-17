"""Stable SHA-256 over canonical-json-v1 bytes."""
from __future__ import annotations

import hashlib
from typing import Any

from .canonical_json import canonical_bytes


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()
