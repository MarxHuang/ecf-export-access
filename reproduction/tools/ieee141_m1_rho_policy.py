"""Registered rho-band policy helpers for the diagnostic EQ067 trust gate.

The legacy joint-trust artifacts use one shared ``0 <= rho <= 3.5``
partition.  New development policies may instead declare a Q-specific
``PAR068`` upper bound.  This module keeps the legacy behaviour available by
default while making a non-legacy policy an explicit, hash-bound input to
source construction and screening.

It intentionally does not read an active scientific registry by default.  A
caller must pass a policy file when it wants a non-legacy domain, so a new
candidate cannot silently inherit a changed registry value.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from r4r.serialization import canonical_hash


Q_MODES = ("Q0", "Q95")
LEGACY_RHO_BANDS: tuple[tuple[str, float, float], ...] = (
    ("B0", 0.0, 0.05),
    ("B1", 0.05, 0.25),
    ("B2", 0.25, 1.0),
    ("B3", 1.0, 2.5),
    ("B4", 2.5, 3.5),
)
_LOWER_EDGES = tuple(item[1] for item in LEGACY_RHO_BANDS)
_BAND_IDS = tuple(item[0] for item in LEGACY_RHO_BANDS)


def _finite_positive(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("RHO_POLICY_BOUND_INVALID")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("RHO_POLICY_BOUND_INVALID") from exc
    if not math.isfinite(converted) or converted <= _LOWER_EDGES[-1]:
        raise ValueError("RHO_POLICY_BOUND_INVALID")
    return converted


def _policy_core(*, serialization_id: str, policy_id: str, parameter_id: str, status: str, bounds: Mapping[str, float]) -> dict[str, Any]:
    return {
        "serialization_id": serialization_id,
        "policy_id": policy_id,
        "parameter_id": parameter_id,
        "status": status,
        "q_modes": list(Q_MODES),
        "rho_max_by_q": {q_mode: float(bounds[q_mode]) for q_mode in Q_MODES},
        "fixed_lower_band_edges": list(_LOWER_EDGES),
        "band_ids": list(_BAND_IDS),
    }


@dataclass(frozen=True)
class RhoBandPolicy:
    """A validated, hash-bound Q-specific upper rho policy."""

    serialization_id: str
    policy_id: str
    parameter_id: str
    status: str
    rho_max_by_q: Mapping[str, float]
    policy_hash: str

    def assert_integrity(self) -> None:
        """Verify that this in-memory object still matches its policy hash.

        ``RhoBandPolicy.create`` is the normal constructor, but the public
        dataclass can still be instantiated directly by a caller.  Consumers
        therefore recheck the self-hash instead of trusting a type annotation
        as evidence that a policy object was constructed correctly.
        """
        if not all(isinstance(value, str) and value for value in (self.policy_id, self.parameter_id, self.status, self.serialization_id, self.policy_hash)):
            raise ValueError("RHO_POLICY_IDENTITY_INVALID")
        if not isinstance(self.rho_max_by_q, Mapping) or set(self.rho_max_by_q) != set(Q_MODES):
            raise ValueError("RHO_POLICY_Q_SCOPE_INVALID")
        bounds = {q_mode: _finite_positive(self.rho_max_by_q[q_mode]) for q_mode in Q_MODES}
        expected_hash = canonical_hash(
            _policy_core(
                serialization_id=self.serialization_id,
                policy_id=self.policy_id,
                parameter_id=self.parameter_id,
                status=self.status,
                bounds=bounds,
            )
        )
        if self.policy_hash != expected_hash:
            raise ValueError("RHO_POLICY_HASH_MISMATCH")

    @classmethod
    def create(
        cls,
        *,
        rho_max_by_q: Mapping[str, Any],
        policy_id: str = "PAR068_V2",
        parameter_id: str = "PAR068",
        status: str = "CANDIDATE_VALUE",
        serialization_id: str = "ieee141_m1_rho_band_policy.v2",
        declared_hash: str | None = None,
    ) -> "RhoBandPolicy":
        if not isinstance(rho_max_by_q, Mapping) or set(rho_max_by_q) != set(Q_MODES):
            raise ValueError("RHO_POLICY_Q_SCOPE_INVALID")
        if not all(isinstance(value, str) and value for value in (policy_id, parameter_id, status, serialization_id)):
            raise ValueError("RHO_POLICY_IDENTITY_INVALID")
        bounds = {q_mode: _finite_positive(rho_max_by_q[q_mode]) for q_mode in Q_MODES}
        expected_hash = canonical_hash(
            _policy_core(
                serialization_id=serialization_id,
                policy_id=policy_id,
                parameter_id=parameter_id,
                status=status,
                bounds=bounds,
            )
        )
        if declared_hash is not None and declared_hash != expected_hash:
            raise ValueError("RHO_POLICY_HASH_MISMATCH")
        return cls(
            serialization_id=serialization_id,
            policy_id=policy_id,
            parameter_id=parameter_id,
            status=status,
            # A frozen dataclass does not freeze a normal nested dict.  Keep
            # the bound vector immutable so its hash cannot become stale in
            # memory between source-library construction and screening.
            rho_max_by_q=MappingProxyType(bounds),
            policy_hash=expected_hash,
        )

    @property
    def is_legacy(self) -> bool:
        self.assert_integrity()
        return self.policy_hash == LEGACY_RHO_POLICY.policy_hash

    def bands_for(self, q_mode: str) -> tuple[tuple[str, float, float], ...]:
        self.assert_integrity()
        if q_mode not in Q_MODES:
            raise ValueError("RHO_POLICY_Q_MODE_INVALID")
        return (
            ("B0", 0.0, 0.05),
            ("B1", 0.05, 0.25),
            ("B2", 0.25, 1.0),
            ("B3", 1.0, 2.5),
            ("B4", 2.5, float(self.rho_max_by_q[q_mode])),
        )

    def to_json(self) -> dict[str, Any]:
        self.assert_integrity()
        body = _policy_core(
            serialization_id=self.serialization_id,
            policy_id=self.policy_id,
            parameter_id=self.parameter_id,
            status=self.status,
            bounds=self.rho_max_by_q,
        )
        body["policy_hash"] = self.policy_hash
        return body


LEGACY_RHO_POLICY = RhoBandPolicy.create(
    rho_max_by_q={"Q0": 3.5, "Q95": 3.5},
    policy_id="LEGACY_RHO_BANDS_V1",
    parameter_id="PAR068_LEGACY",
    status="LEGACY_DIAGNOSTIC_COMPATIBILITY",
    serialization_id="ieee141_m1_rho_band_policy.v1",
)


def policy_from_payload(payload: Mapping[str, Any], *, parameter_id: str = "PAR068") -> RhoBandPolicy:
    """Load a direct policy object or a parameter-registry-like envelope.

    Accepted direct forms contain either ``rho_max_by_q``, ``bounds`` or the
    parameter-registry ``value`` field.  An envelope may contain a
    ``parameters`` list, from which the requested parameter is selected.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("RHO_POLICY_PAYLOAD_INVALID")
    item: Mapping[str, Any] = payload
    parameters = payload.get("parameters")
    if isinstance(parameters, list):
        matches = [value for value in parameters if isinstance(value, Mapping) and value.get("parameter_id") == parameter_id]
        if len(matches) != 1:
            raise ValueError("RHO_POLICY_PARAMETER_NOT_FOUND")
        item = matches[0]
    actual_parameter_id = str(item.get("parameter_id", parameter_id))
    if actual_parameter_id != parameter_id:
        raise ValueError("RHO_POLICY_PARAMETER_ID_MISMATCH")
    declared_q_modes = item.get("q_modes")
    if declared_q_modes is not None:
        if not isinstance(declared_q_modes, (list, tuple)) or tuple(declared_q_modes) != Q_MODES:
            raise ValueError("RHO_POLICY_Q_SCOPE_INVALID")
    declared_edges = item.get("fixed_lower_band_edges")
    if declared_edges is not None:
        if not isinstance(declared_edges, (list, tuple)):
            raise ValueError("RHO_POLICY_BAND_PARTITION_INVALID")
        try:
            parsed_edges = tuple(float(value) for value in declared_edges)
        except (TypeError, ValueError) as exc:
            raise ValueError("RHO_POLICY_BAND_PARTITION_INVALID") from exc
        if parsed_edges != _LOWER_EDGES:
            raise ValueError("RHO_POLICY_BAND_PARTITION_INVALID")
    declared_band_ids = item.get("band_ids")
    if declared_band_ids is not None:
        if not isinstance(declared_band_ids, (list, tuple)) or tuple(declared_band_ids) != _BAND_IDS:
            raise ValueError("RHO_POLICY_BAND_PARTITION_INVALID")
    bounds = item.get("rho_max_by_q")
    if bounds is None:
        bounds = item.get("bounds")
    if bounds is None:
        bounds = item.get("value")
    policy_id = str(item.get("policy_id") or f"{actual_parameter_id}_V1")
    status = str(item.get("status") or "CANDIDATE_VALUE")
    serialization_id = str(item.get("serialization_id") or "ieee141_m1_rho_band_policy.v2")
    declared = item.get("policy_hash")
    if declared is not None and not isinstance(declared, str):
        raise ValueError("RHO_POLICY_HASH_MISMATCH")
    return RhoBandPolicy.create(
        rho_max_by_q=bounds,
        policy_id=policy_id,
        parameter_id=actual_parameter_id,
        status=status,
        serialization_id=serialization_id,
        declared_hash=declared,
    )


def load_rho_policy(path: Path, *, parameter_id: str = "PAR068") -> RhoBandPolicy:
    """Load a JSON or YAML policy and validate its declared hash when present."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError("RHO_POLICY_FILE_TYPE_INVALID")
    return policy_from_payload(payload, parameter_id=parameter_id)


def band_json(policy: RhoBandPolicy, q_mode: str) -> list[dict[str, Any]]:
    bands = policy.bands_for(q_mode)
    return [
        {"id": name, "lower": lower, "upper": upper, "upper_inclusive": index == len(bands) - 1}
        for index, (name, lower, upper) in enumerate(bands)
    ]


def same_band_partition(left: RhoBandPolicy, right: RhoBandPolicy) -> bool:
    left.assert_integrity()
    right.assert_integrity()
    return all(left.bands_for(q_mode) == right.bands_for(q_mode) for q_mode in Q_MODES)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parameter-id", default="PAR068")
    args = parser.parse_args()
    print(json.dumps(load_rho_policy(args.policy, parameter_id=args.parameter_id).to_json(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEGACY_RHO_BANDS",
    "LEGACY_RHO_POLICY",
    "Q_MODES",
    "RhoBandPolicy",
    "band_json",
    "load_rho_policy",
    "policy_from_payload",
    "same_band_partition",
]
