"""Read-only access to the active scientific registries."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "specifications" / "active"
REGISTRY_FILES = (
    "DATA_TYPE_REGISTRY.yaml", "ENTITY_AND_UNIT_REGISTRY.yaml", "VARIABLE_REGISTRY.yaml",
    "PARAMETER_REGISTRY.yaml", "EQUATION_REGISTRY.yaml", "STATUS_AND_GATE_DEFINITIONS.yaml",
    "CLAIM_TO_EVIDENCE_MAP.yaml", "MODULE_READINESS_MATRIX.yaml",
)


def load_active_registries() -> dict[str, Any]:
    return {name: yaml.safe_load((ACTIVE / name).read_text(encoding="utf-8")) for name in REGISTRY_FILES}


@lru_cache(maxsize=1)
def registered_failure_reason_ids() -> frozenset[str]:
    """Return the closed set of failure-reason tokens in the active contract."""
    document = yaml.safe_load(
        (ACTIVE / "STATUS_AND_GATE_DEFINITIONS.yaml").read_text(encoding="utf-8")
    )
    entries = document.get("failure_reason_registry", [])
    return frozenset(entry["failure_reason_id"] for entry in entries)


@lru_cache(maxsize=1)
def gate_failure_reason_ids() -> dict[str, frozenset[str]]:
    """Return the closed failure-reason set allowed by each registered gate."""
    document = yaml.safe_load(
        (ACTIVE / "STATUS_AND_GATE_DEFINITIONS.yaml").read_text(encoding="utf-8")
    )
    registered = registered_failure_reason_ids()
    mapping: dict[str, frozenset[str]] = {}
    for gate in document.get("gates", []):
        gate_id = gate["gate_id"]
        reasons = frozenset(gate.get("failure_reasons", []) or [])
        unknown = reasons - registered
        if unknown:
            raise ValueError(f"gate {gate_id} references unregistered failure reasons: {sorted(unknown)}")
        mapping[gate_id] = reasons
    return mapping


def require_registered_failure_reasons(reasons: Any, *, gate_ids: tuple[str, ...], field: str) -> None:
    """Fail closed when a typed result carries an unknown or wrong-family reason.

    Result classes declare the gate family that owns their failure vocabulary;
    this prevents a generic identifier vector from silently accepting tokens
    from another evidence layer.
    """
    values = getattr(reasons, "values", None)
    if values is None:
        raise ValueError(f"{field} must expose identifier values")
    allowed_by_gate = gate_failure_reason_ids()
    unknown_gates = set(gate_ids) - set(allowed_by_gate)
    if unknown_gates:
        raise ValueError(f"{field} declares unregistered gate family: {sorted(unknown_gates)}")
    allowed = set().union(*(allowed_by_gate[gate_id] for gate_id in gate_ids))
    actual = {item.value for item in values}
    invalid = actual - allowed
    if invalid:
        raise ValueError(f"{field} contains unregistered or wrong-family failure reasons: {sorted(invalid)}")
