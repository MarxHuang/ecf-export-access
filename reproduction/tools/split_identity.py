"""Canonical identity helpers for development/evaluation split manifests.

The pre-freeze v1 split used ``split_hash`` (hash of the object before that
field was added).  The post-freeze v2 overlay follows the repository-wide
``result_hash`` convention and hashes an object whose ``result_hash`` is
explicitly ``null`` during canonicalization.  Downstream tools must bind to
the declared identity rather than silently comparing missing values.
"""
from __future__ import annotations

from typing import Any, Mapping

from r4r.serialization import canonical_hash


V2_SERIALIZATION_ID = "ieee141_m1_v2_development_evaluation_split.v2"


def declared_split_hash(payload: Mapping[str, Any]) -> str | None:
    """Return the canonical identity for either supported split format."""

    if payload.get("serialization_id") == V2_SERIALIZATION_ID:
        value = payload.get("result_hash")
    else:
        value = payload.get("split_hash")
    return value if isinstance(value, str) else None


def split_hash_is_self_consistent(payload: Mapping[str, Any]) -> bool:
    """Check the format-specific canonical hash without mutating ``payload``."""

    declared = declared_split_hash(payload)
    if declared is None:
        return False
    body = dict(payload)
    if payload.get("serialization_id") == V2_SERIALIZATION_ID:
        body["result_hash"] = None
    else:
        body.pop("split_hash", None)
    return declared == canonical_hash(body)


__all__ = ["declared_split_hash", "split_hash_is_self_consistent"]
