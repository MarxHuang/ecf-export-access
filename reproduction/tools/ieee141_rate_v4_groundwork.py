"""Pure, fail-closed groundwork for a non-circular CASE141 synthetic RATE V4.

This module deliberately does *not* parse a network, solve AC power flow, fit a
proxy, select a trust region, calibrate margins, or run mitigation.  It only
builds and validates the deterministic artifacts that must exist before those
computations are authorised:

``root split -> rate-independent anchor definition -> finite candidate library
-> two-ended AC-flow anchor -> development-only rate candidates -> selection
-> immutable pre-source freeze``.

The functions accept already-produced fresh baseline/reference flow records.
Their input schema makes it impossible to silently incorporate evaluation,
mitigation, candidate/trust, guard, holdout, calibration, realized-audit, or
legacy V3 output records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from r4r.serialization import canonical_dumps, canonical_hash


PROTOCOL_SERIALIZATION_ID = "case141_rate_design_protocol.v1"
PROTOCOL_ID = "CASE141_RATE_DESIGN_PROTOCOL_V1"
ANCHOR_DOMAIN_SERIALIZATION_ID = "case141_rate_anchor_domain.v1"
ANCHOR_DOMAIN_ID = "RATE_ANCHOR_DOMAIN_V1"
ROOT_SPLIT_SERIALIZATION_ID = "case141_rate_design_root_split.v1"
CANDIDATE_LIBRARY_SERIALIZATION_ID = "case141_rate_v4_candidate_library.v1"
ANCHOR_OBSERVATION_SERIALIZATION_ID = "case141_rate_v4_two_ended_anchor_observation.v1"
CANDIDATE_POLICY_SERIALIZATION_ID = "case141_synthetic_rate_v4_candidate_policy.v1"
CANDIDATE_POLICY_BUNDLE_SERIALIZATION_ID = "case141_synthetic_rate_v4_candidate_policy_bundle.v1"
SELECTION_SERIALIZATION_ID = "case141_rate_v4_development_selection.v1"
FREEZE_ATTESTATION_SERIALIZATION_ID = "case141_rate_v4_freeze_attestation.v1"

Q_MODES = ("Q0", "Q95")
BRANCH_CLASSES = ("ROOT_FEEDER", "TRUNK", "LATERAL", "TRANSFORMER")
ROOT_ROLES = (
    "RATE_DESIGN",
    "SOURCE_SUPPORT",
    "GUARD",
    "PROFILE_LORO",
    "FAMILY_HOLDOUT",
    "CALIBRATION_TRAIN",
    "CALIBRATION_HOLDOUT",
    "REALIZED_AUDIT",
    "EVALUATION",
)
READ_ASSERTION_FIELDS = (
    "evaluation_results_read",
    "mitigation_results_read",
    "source_results_read",
    "candidate_results_read",
    "trust_candidate_results_read",
    "guard_results_read",
    "profile_loro_results_read",
    "family_holdout_results_read",
    "holdout_results_read",
    "calibration_results_read",
    "realized_audit_results_read",
    "legacy_v3_outputs_read",
)
_LEGACY_V3_TOKENS = ("m1_v3", "rate_v3", "synthetic_rate_policy_v3", "legacy_v3")


class RateV4GroundworkError(ValueError):
    """Raised when a RATE V4 groundwork artifact violates the pre-freeze boundary."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RateV4GroundworkError(f"{label}_MAPPING_REQUIRED")
    return dict(value)


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RateV4GroundworkError(f"{label}_NONEMPTY_STRING_REQUIRED")
    return value


def _identifier(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if any(character.isspace() for character in result):
        raise RateV4GroundworkError(f"{label}_WHITESPACE_FORBIDDEN")
    return result


def _finite(value: Any, *, label: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RateV4GroundworkError(f"{label}_FINITE_NUMBER_REQUIRED")
    result = float(value)
    if not math.isfinite(result):
        raise RateV4GroundworkError(f"{label}_FINITE_NUMBER_REQUIRED")
    if positive and result <= 0.0:
        raise RateV4GroundworkError(f"{label}_POSITIVE_REQUIRED")
    if nonnegative and result < 0.0:
        raise RateV4GroundworkError(f"{label}_NONNEGATIVE_REQUIRED")
    return result


def _sha256(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise RateV4GroundworkError(f"{label}_SHA256_REQUIRED")
    return text.upper()


def _self_hash(value: Mapping[str, Any], *, field: str, label: str) -> str:
    payload = dict(value)
    declared = payload.get(field)
    if not isinstance(declared, str):
        raise RateV4GroundworkError(f"{label}_{field.upper()}_MISSING")
    payload[field] = None
    expected = canonical_hash(payload)
    if declared != expected:
        raise RateV4GroundworkError(f"{label}_{field.upper()}_MISMATCH")
    return declared


def _finalize(value: Mapping[str, Any], *, field: str = "result_hash") -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload[field] = None
    payload[field] = canonical_hash(payload)
    return payload


def _policy_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["policy_hash"] = None
    return canonical_hash(payload)


def _assert_exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise RateV4GroundworkError(f"{label}_FIELDS_INVALID:missing={missing}:extra={extra}")


def _contains_legacy_v3(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_legacy_v3(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_legacy_v3(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in _LEGACY_V3_TOKENS)
    return False


def _validate_input_read_assertions(value: Any, *, label: str) -> dict[str, bool]:
    assertions = _mapping(value, label=f"{label}_INPUT_READ_ASSERTIONS")
    _assert_exact_keys(assertions, READ_ASSERTION_FIELDS, label=f"{label}_INPUT_READ_ASSERTIONS")
    if any(assertions[field] is not False for field in READ_ASSERTION_FIELDS):
        raise RateV4GroundworkError(f"{label}_FORBIDDEN_DOWNSTREAM_OR_LEGACY_READ")
    return {field: False for field in READ_ASSERTION_FIELDS}


def _fresh_read_assertions() -> dict[str, bool]:
    return {field: False for field in READ_ASSERTION_FIELDS}


def validate_rate_design_protocol(protocol: Mapping[str, Any]) -> str:
    """Validate the static RATE V4 protocol and return its canonical hash.

    This validator intentionally accepts no ``result_hash`` field: the YAML is
    a human-approved candidate specification, whereas all runtime artifacts
    built from it are self-hashed.
    """

    payload = _mapping(protocol, label="RATE_DESIGN_PROTOCOL")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "protocol_id", "version", "status", "scientific_role", "scope",
            "q_modes", "branch_classes", "root_split", "anchor_construction",
            "candidate_policy_library", "quality_selection", "freeze", "provenance_minimum",
        },
        label="RATE_DESIGN_PROTOCOL",
    )
    if payload["serialization_id"] != PROTOCOL_SERIALIZATION_ID or payload["protocol_id"] != PROTOCOL_ID:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_IDENTITY_INVALID")
    if payload["status"] != "CANDIDATE_DEVELOPMENT_ONLY_NOT_FROZEN":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_STATUS_INVALID")
    if payload["scientific_role"] != "SYNTHETIC_SCENARIO_RATE_DESIGN_ONLY":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_ROLE_INVALID")
    if payload["q_modes"] != list(Q_MODES) or payload["branch_classes"] != list(BRANCH_CLASSES):
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_Q_OR_CLASS_ROSTER_INVALID")

    scope = _mapping(payload["scope"], label="RATE_DESIGN_PROTOCOL_SCOPE")
    if scope.get("primary_evidence_eligible") is not False or scope.get("manuscript_evidence_eligible") is not False:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_PREMATURE_PROMOTION")
    if scope.get("native_matpower_rate_policy") != "preserve_zero_rate_a_b_c_as_unavailable":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_NATIVE_RATE_SEMANTICS_INVALID")
    if scope.get("rate_description") != "synthetic_scenario_defined_two_ended_mva_limit":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_RATE_DESCRIPTION_INVALID")

    split = _mapping(payload["root_split"], label="RATE_DESIGN_PROTOCOL_ROOT_SPLIT")
    if split.get("partition_unit") != "root_profile_id" or split.get("required_roles") != list(ROOT_ROLES):
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_ROOT_SPLIT_ROSTER_INVALID")
    if split.get("all_roles_pairwise_disjoint") is not True or split.get("rate_design_must_be_disjoint_from_all_later_roles") is not True:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_ROOT_SPLIT_DISJOINTNESS_INVALID")
    if split.get("mitigation_scope") != "EVALUATION_ONLY_AFTER_RATE_FREEZE":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_MITIGATION_SCOPE_INVALID")

    anchor = _mapping(payload["anchor_construction"], label="RATE_DESIGN_PROTOCOL_ANCHOR")
    if anchor.get("baseline_observable") != "TWO_ENDED_APPARENT_POWER_MVA":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_BASELINE_OBSERVABLE_INVALID")
    if anchor.get("anchor_domain_id") != ANCHOR_DOMAIN_ID or anchor.get("honest_reference_method") != "DEC015_TWO_STAGE_MAX_EXPORT":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_ANCHOR_BINDING_INVALID")
    if anchor.get("thermal_semantics") != "TWO_ENDED_MVA" or anchor.get("rooted_orientation_role") != "INTERPRETATION_ONLY":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_THERMAL_SEMANTICS_INVALID")
    if anchor.get("clipping") != "FORBIDDEN":
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_CLIPPING_INVALID")

    library = _mapping(payload["candidate_policy_library"], label="RATE_DESIGN_PROTOCOL_LIBRARY")
    if library.get("finite_predeclared_before_ac_execution") is not True:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_LIBRARY_NOT_PREDECLARED")
    if library.get("selection_parameter_order") != [
        "anchor_multiplier", "anchor_minimum_mva", "final_headroom_ratio", "final_minimum_mva",
    ]:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_PARAMETER_ORDER_INVALID")
    candidates = library.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_CANDIDATES_REQUIRED")
    identifiers: set[str] = set()
    for raw in candidates:
        candidate = _mapping(raw, label="RATE_DESIGN_PROTOCOL_CANDIDATE")
        _assert_exact_keys(candidate, {"candidate_id", "class_parameters"}, label="RATE_DESIGN_PROTOCOL_CANDIDATE")
        candidate_id = _identifier(candidate["candidate_id"], label="RATE_DESIGN_PROTOCOL_CANDIDATE_ID")
        if candidate_id in identifiers:
            raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_CANDIDATE_ID_DUPLICATE")
        identifiers.add(candidate_id)
        _validate_class_parameters(candidate["class_parameters"], label=f"RATE_DESIGN_PROTOCOL_{candidate_id}")

    quality = _mapping(payload["quality_selection"], label="RATE_DESIGN_PROTOCOL_QUALITY")
    if quality.get("baseline_ac_all_pass_required") is not True:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_BASELINE_GATE_INVALID")
    reference_minimum = _finite(quality.get("honest_reference_minimum_pass_rate"), label="RATE_DESIGN_PROTOCOL_REFERENCE_MINIMUM", nonnegative=True)
    interval = quality.get("strategic_unmitigated_pair_pass_rate_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_STRATEGIC_INTERVAL_INVALID")
    lower = _finite(interval[0], label="RATE_DESIGN_PROTOCOL_STRATEGIC_LOWER", nonnegative=True)
    upper = _finite(interval[1], label="RATE_DESIGN_PROTOCOL_STRATEGIC_UPPER", nonnegative=True)
    target = _finite(quality.get("strategic_unmitigated_pair_pass_rate_target"), label="RATE_DESIGN_PROTOCOL_STRATEGIC_TARGET", nonnegative=True)
    if not (0.0 <= reference_minimum <= 1.0 and 0.0 <= lower <= target <= upper <= 1.0):
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_QUALITY_RANGE_INVALID")
    forbidden = quality.get("prohibited_selection_inputs")
    required_forbidden = {
        "evaluation_ac_flow", "evaluation_pass_fail", "mitigation_result", "trust_candidate_result",
        "guard_result", "profile_loro_result", "family_holdout_result", "calibration_result",
        "realized_audit_result", "historical_v3_output",
    }
    if not isinstance(forbidden, list) or not required_forbidden.issubset(set(forbidden)):
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_PROHIBITED_INPUTS_INCOMPLETE")
    if quality.get("deterministic_order") != [
        "eligible_first", "absolute_distance_to_strategic_target_ascending", "rate_policy_hash_ascending",
    ]:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_TIEBREAK_INVALID")

    freeze = _mapping(payload["freeze"], label="RATE_DESIGN_PROTOCOL_FREEZE")
    if freeze.get("attestation_serialization_id") != FREEZE_ATTESTATION_SERIALIZATION_ID:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_FREEZE_SERIALIZATION_INVALID")
    if freeze.get("immutable_after_freeze") is not True:
        raise RateV4GroundworkError("RATE_DESIGN_PROTOCOL_FREEZE_IMMUTABILITY_INVALID")
    return canonical_hash(payload)


def validate_rate_anchor_domain(anchor_domain: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Validate the rate-independent, still-network-constrained anchor domain."""

    validate_rate_design_protocol(protocol)
    payload = _mapping(anchor_domain, label="RATE_ANCHOR_DOMAIN")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "domain_id", "version", "status", "scientific_role", "scope",
            "allocation_bounds", "network_constraints", "honest_reference", "input_exclusions",
            "required_execution_bindings", "output_role",
        },
        label="RATE_ANCHOR_DOMAIN",
    )
    if payload["serialization_id"] != ANCHOR_DOMAIN_SERIALIZATION_ID or payload["domain_id"] != ANCHOR_DOMAIN_ID:
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_IDENTITY_INVALID")
    if payload["status"] != "CANDIDATE_RATE_INDEPENDENT_DEVELOPMENT_DOMAIN":
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_STATUS_INVALID")
    if payload["scientific_role"] != "RATE_DESIGN_ONLY_NETWORK_CONSTRAINED_ANCHOR":
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_ROLE_INVALID")
    scope = _mapping(payload["scope"], label="RATE_ANCHOR_DOMAIN_SCOPE")
    if scope.get("q_modes") != list(Q_MODES) or scope.get("allocation_domain_is_raw_capacity_box") is not False:
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_SCOPE_INVALID")
    if scope.get("development_only") is not True or scope.get("primary_evidence_eligible") is not False:
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_PROMOTION_INVALID")
    constraints = _mapping(payload["network_constraints"], label="RATE_ANCHOR_DOMAIN_NETWORK_CONSTRAINTS")
    branch = _string(constraints.get("branch"), label="RATE_ANCHOR_DOMAIN_BRANCH")
    if "hypot(P_proxy_l^f(x), Q_proxy_l^f(x))" not in branch or "hypot(P_proxy_l^t(x), Q_proxy_l^t(x))" not in branch:
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_TWO_ENDED_MVA_REQUIRED")
    if constraints.get("topology_orientation") != "rooted orientation is explanation-only and never replaces the two-ended MVA predicate.":
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_ORIENTATION_INVALID")
    reference = _mapping(payload["honest_reference"], label="RATE_ANCHOR_DOMAIN_REFERENCE")
    if reference.get("method") != "DEC015_TWO_STAGE_MAX_EXPORT":
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_REFERENCE_METHOD_INVALID")
    prohibited = reference.get("prohibited")
    if not isinstance(prohibited, list) or not {
        "raw_capacity_box_reference", "epsilon_tie_break", "historical_reference_allocation",
    }.issubset(set(prohibited)):
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_REFERENCE_EXCLUSIONS_INVALID")
    exclusions = _mapping(payload["input_exclusions"], label="RATE_ANCHOR_DOMAIN_EXCLUSIONS")
    expected_exclusions = {
        "evaluation_results_read", "mitigation_results_read", "source_results_read", "trust_candidate_results_read",
        "guard_results_read", "profile_loro_results_read", "family_holdout_results_read", "holdout_results_read", "calibration_results_read",
        "realized_audit_results_read", "legacy_v3_outputs_read",
    }
    if set(exclusions) != expected_exclusions or any(value is not False for value in exclusions.values()):
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_EXCLUSIONS_INVALID")
    output_role = _mapping(payload["output_role"], label="RATE_ANCHOR_DOMAIN_OUTPUT_ROLE")
    forbidden = output_role.get("forbidden")
    if not isinstance(forbidden, list) or not {"T2", "T3", "T4", "T5", "G019"}.issubset(set(forbidden)):
        raise RateV4GroundworkError("RATE_ANCHOR_DOMAIN_OUTPUT_BOUNDARY_INVALID")
    return canonical_hash(payload)


def _validate_class_parameters(value: Any, *, label: str) -> dict[str, dict[str, float]]:
    parameters = _mapping(value, label=f"{label}_CLASS_PARAMETERS")
    if set(parameters) != set(BRANCH_CLASSES):
        raise RateV4GroundworkError(f"{label}_CLASS_ROSTER_INVALID")
    normalized: dict[str, dict[str, float]] = {}
    for branch_class in BRANCH_CLASSES:
        row = _mapping(parameters[branch_class], label=f"{label}_{branch_class}")
        _assert_exact_keys(
            row,
            {"anchor_multiplier", "anchor_minimum_mva", "final_headroom_ratio", "final_minimum_mva"},
            label=f"{label}_{branch_class}",
        )
        multiplier = _finite(row["anchor_multiplier"], label=f"{label}_{branch_class}_MULTIPLIER", positive=True)
        if multiplier <= 1.0:
            raise RateV4GroundworkError(f"{label}_{branch_class}_MULTIPLIER_MUST_EXCEED_ONE")
        normalized[branch_class] = {
            "anchor_multiplier": multiplier,
            "anchor_minimum_mva": _finite(row["anchor_minimum_mva"], label=f"{label}_{branch_class}_ANCHOR_MINIMUM", positive=True),
            "final_headroom_ratio": _finite(row["final_headroom_ratio"], label=f"{label}_{branch_class}_HEADROOM", nonnegative=True),
            "final_minimum_mva": _finite(row["final_minimum_mva"], label=f"{label}_{branch_class}_FINAL_MINIMUM", positive=True),
        }
    return normalized


def build_rate_design_root_split(
    *,
    protocol: Mapping[str, Any],
    root_role_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build a self-hashed, root-level split that prevents later leakage."""

    protocol_hash = validate_rate_design_protocol(protocol)
    roles = _mapping(root_role_ids, label="RATE_DESIGN_ROOT_ROLE_IDS")
    if set(roles) != set(ROOT_ROLES):
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_ROLE_ROSTER_INVALID")
    normalized: dict[str, list[str]] = {}
    claimed: dict[str, str] = {}
    for role in ROOT_ROLES:
        values = roles[role]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RateV4GroundworkError(f"RATE_DESIGN_ROOT_ROLE_{role}_VECTOR_REQUIRED")
        identifiers = [_identifier(value, label=f"RATE_DESIGN_ROOT_{role}") for value in values]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise RateV4GroundworkError(f"RATE_DESIGN_ROOT_ROLE_{role}_EMPTY_OR_DUPLICATE")
        for root_id in identifiers:
            previous = claimed.setdefault(root_id, role)
            if previous != role:
                raise RateV4GroundworkError(f"RATE_DESIGN_ROOT_ROLE_OVERLAP:{previous}:{role}:{root_id}")
        normalized[role] = sorted(identifiers)
    result = {
        "serialization_id": ROOT_SPLIT_SERIALIZATION_ID,
        "status": "RATE_DESIGN_ROOT_SPLIT_FROZEN_PRE_EXECUTION",
        "protocol_hash": protocol_hash,
        "partition_unit": "root_profile_id",
        "root_role_ids": normalized,
        "root_role_hashes": {role: canonical_hash(normalized[role]) for role in ROOT_ROLES},
        "all_roles_pairwise_disjoint": True,
        "rate_design_excluded_roles": list(ROOT_ROLES[1:]),
        "mitigation_scope": "EVALUATION_ONLY_AFTER_RATE_FREEZE",
        "input_read_assertions": _fresh_read_assertions(),
        "result_hash": None,
    }
    return _finalize(result)


def validate_rate_design_root_split(split: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Validate a root split, including pairwise root-level disjointness."""

    protocol_hash = validate_rate_design_protocol(protocol)
    payload = _mapping(split, label="RATE_DESIGN_ROOT_SPLIT")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "partition_unit", "root_role_ids",
            "root_role_hashes", "all_roles_pairwise_disjoint", "rate_design_excluded_roles",
            "mitigation_scope", "input_read_assertions", "result_hash",
        },
        label="RATE_DESIGN_ROOT_SPLIT",
    )
    if payload["serialization_id"] != ROOT_SPLIT_SERIALIZATION_ID or payload["status"] != "RATE_DESIGN_ROOT_SPLIT_FROZEN_PRE_EXECUTION":
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_SPLIT_IDENTITY_INVALID")
    if payload["protocol_hash"] != protocol_hash or payload["partition_unit"] != "root_profile_id":
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_SPLIT_PROTOCOL_BINDING_INVALID")
    if payload["all_roles_pairwise_disjoint"] is not True or payload["rate_design_excluded_roles"] != list(ROOT_ROLES[1:]):
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_SPLIT_DISJOINTNESS_INVALID")
    if payload["mitigation_scope"] != "EVALUATION_ONLY_AFTER_RATE_FREEZE":
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_SPLIT_MITIGATION_SCOPE_INVALID")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_DESIGN_ROOT_SPLIT")
    rebuilt = build_rate_design_root_split(protocol=protocol, root_role_ids=_mapping(payload["root_role_ids"], label="RATE_DESIGN_ROOT_SPLIT_ROLE_IDS"))
    if payload != rebuilt:
        raise RateV4GroundworkError("RATE_DESIGN_ROOT_SPLIT_REPLAY_MISMATCH")
    return _self_hash(payload, field="result_hash", label="RATE_DESIGN_ROOT_SPLIT")


def build_predeclared_rate_candidate_library(*, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the finite candidate parameter library before any AC result."""

    protocol_hash = validate_rate_design_protocol(protocol)
    candidates = _mapping(protocol, label="RATE_DESIGN_PROTOCOL")["candidate_policy_library"]["candidates"]
    records: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = _mapping(raw, label="RATE_CANDIDATE")
        candidate_id = _identifier(candidate["candidate_id"], label="RATE_CANDIDATE_ID")
        parameters = _validate_class_parameters(candidate["class_parameters"], label=f"RATE_CANDIDATE_{candidate_id}")
        record = {
            "candidate_id": candidate_id,
            "class_parameters": parameters,
            "candidate_parameter_hash": canonical_hash({"candidate_id": candidate_id, "class_parameters": parameters}),
        }
        records.append(record)
    result = {
        "serialization_id": CANDIDATE_LIBRARY_SERIALIZATION_ID,
        "status": "FINITE_PREDECLARED_RATE_V4_CANDIDATE_LIBRARY",
        "protocol_hash": protocol_hash,
        "q_modes": list(Q_MODES),
        "branch_classes": list(BRANCH_CLASSES),
        "candidate_records": records,
        "ac_or_evaluation_result_read": False,
        "input_read_assertions": _fresh_read_assertions(),
        "result_hash": None,
    }
    return _finalize(result)


def validate_predeclared_rate_candidate_library(library: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    protocol_hash = validate_rate_design_protocol(protocol)
    payload = _mapping(library, label="RATE_CANDIDATE_LIBRARY")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "q_modes", "branch_classes",
            "candidate_records", "ac_or_evaluation_result_read", "input_read_assertions", "result_hash",
        },
        label="RATE_CANDIDATE_LIBRARY",
    )
    if payload["serialization_id"] != CANDIDATE_LIBRARY_SERIALIZATION_ID or payload["status"] != "FINITE_PREDECLARED_RATE_V4_CANDIDATE_LIBRARY":
        raise RateV4GroundworkError("RATE_CANDIDATE_LIBRARY_IDENTITY_INVALID")
    if payload["protocol_hash"] != protocol_hash or payload["q_modes"] != list(Q_MODES) or payload["branch_classes"] != list(BRANCH_CLASSES):
        raise RateV4GroundworkError("RATE_CANDIDATE_LIBRARY_BINDING_INVALID")
    if payload["ac_or_evaluation_result_read"] is not False:
        raise RateV4GroundworkError("RATE_CANDIDATE_LIBRARY_PREMATURE_RESULT_READ")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_CANDIDATE_LIBRARY")
    rebuilt = build_predeclared_rate_candidate_library(protocol=protocol)
    if payload != rebuilt:
        raise RateV4GroundworkError("RATE_CANDIDATE_LIBRARY_REPLAY_MISMATCH")
    return _self_hash(payload, field="result_hash", label="RATE_CANDIDATE_LIBRARY")


def two_ended_apparent_power_mva(
    *,
    p_from_mw: float,
    q_from_mvar: float,
    p_to_mw: float,
    q_to_mvar: float,
) -> dict[str, float]:
    """Return signed-flow-preserving two-ended apparent-power magnitudes."""

    p_from = _finite(p_from_mw, label="P_FROM_MW")
    q_from = _finite(q_from_mvar, label="Q_FROM_MVAR")
    p_to = _finite(p_to_mw, label="P_TO_MW")
    q_to = _finite(q_to_mvar, label="Q_TO_MVAR")
    s_from = math.hypot(p_from, q_from)
    s_to = math.hypot(p_to, q_to)
    return {"s_from_mva": s_from, "s_to_mva": s_to, "s_max_mva": max(s_from, s_to)}


def _validate_flow_row(
    value: Any,
    *,
    role: str,
    rate_design_roots: set[str],
    candidate_ids: set[str],
    candidate_parameter_hashes: Mapping[str, str],
    anchor_domain_hash: str,
) -> dict[str, Any]:
    row = _mapping(value, label=f"RATE_V4_{role}_FLOW")
    expected = {
        "root_id", "q_mode", "branch_id", "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar",
        "flow_role", "source_kind", "historical_output_reused", "input_read_assertions",
    }
    if role == "REFERENCE":
        expected |= {"rate_candidate_id", "anchor_domain_hash", "anchor_parameter_hash"}
    _assert_exact_keys(row, expected, label=f"RATE_V4_{role}_FLOW")
    root_id = _identifier(row["root_id"], label=f"RATE_V4_{role}_ROOT")
    if root_id not in rate_design_roots:
        raise RateV4GroundworkError(f"RATE_V4_{role}_ROOT_NOT_IN_RATE_DESIGN_SPLIT")
    q_mode = _string(row["q_mode"], label=f"RATE_V4_{role}_Q_MODE")
    if q_mode not in Q_MODES:
        raise RateV4GroundworkError(f"RATE_V4_{role}_Q_MODE_INVALID")
    branch_id = _identifier(row["branch_id"], label=f"RATE_V4_{role}_BRANCH")
    expected_role = "BASELINE" if role == "BASELINE" else "HONEST_REFERENCE_ANCHOR"
    if row["flow_role"] != expected_role:
        raise RateV4GroundworkError(f"RATE_V4_{role}_FLOW_ROLE_INVALID")
    expected_kind = "FRESH_RATE_DESIGN_AC_BASELINE" if role == "BASELINE" else "FRESH_RATE_ANCHOR_AC_REFERENCE"
    if row["source_kind"] != expected_kind:
        raise RateV4GroundworkError(f"RATE_V4_{role}_SOURCE_KIND_INVALID")
    if row["historical_output_reused"] is not False:
        raise RateV4GroundworkError(f"RATE_V4_{role}_HISTORICAL_OUTPUT_FORBIDDEN")
    _validate_input_read_assertions(row["input_read_assertions"], label=f"RATE_V4_{role}_FLOW")
    if _contains_legacy_v3(row):
        raise RateV4GroundworkError(f"RATE_V4_{role}_LEGACY_V3_REFERENCE_FORBIDDEN")
    result = {
        "root_id": root_id,
        "q_mode": q_mode,
        "branch_id": branch_id,
        "p_from_mw": _finite(row["p_from_mw"], label=f"RATE_V4_{role}_P_FROM"),
        "q_from_mvar": _finite(row["q_from_mvar"], label=f"RATE_V4_{role}_Q_FROM"),
        "p_to_mw": _finite(row["p_to_mw"], label=f"RATE_V4_{role}_P_TO"),
        "q_to_mvar": _finite(row["q_to_mvar"], label=f"RATE_V4_{role}_Q_TO"),
    }
    if role == "REFERENCE":
        candidate_id = _identifier(row["rate_candidate_id"], label="RATE_V4_REFERENCE_CANDIDATE")
        if candidate_id not in candidate_ids:
            raise RateV4GroundworkError("RATE_V4_REFERENCE_CANDIDATE_UNKNOWN")
        if _sha256(row["anchor_domain_hash"], label="RATE_V4_REFERENCE_ANCHOR_DOMAIN_HASH") != anchor_domain_hash:
            raise RateV4GroundworkError("RATE_V4_REFERENCE_ANCHOR_DOMAIN_MISMATCH")
        if _sha256(row["anchor_parameter_hash"], label="RATE_V4_REFERENCE_ANCHOR_PARAMETER_HASH") != candidate_parameter_hashes[candidate_id]:
            raise RateV4GroundworkError("RATE_V4_REFERENCE_ANCHOR_PARAMETER_MISMATCH")
        result["rate_candidate_id"] = candidate_id
        result["anchor_parameter_hash"] = candidate_parameter_hashes[candidate_id]
    return result


def build_two_ended_rate_anchor_observation(
    *,
    protocol: Mapping[str, Any],
    anchor_domain: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_library: Mapping[str, Any],
    baseline_flow_rows: Sequence[Mapping[str, Any]],
    reference_flow_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fresh baseline/reference two-ended MVA anchor for every candidate.

    The reference flow for candidate ``theta`` is required to come from the
    rate-independent anchor domain parameterised by that candidate.  This is
    why reference rows carry ``rate_candidate_id`` instead of being shared
    indiscriminately across final RATE candidates.
    """

    protocol_hash = validate_rate_design_protocol(protocol)
    anchor_domain_hash = validate_rate_anchor_domain(anchor_domain, protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    library_hash = validate_predeclared_rate_candidate_library(candidate_library, protocol)
    rate_roots = set(_mapping(root_split, label="RATE_DESIGN_ROOT_SPLIT")["root_role_ids"]["RATE_DESIGN"])
    records = _mapping(candidate_library, label="RATE_CANDIDATE_LIBRARY")["candidate_records"]
    candidate_ids = {_identifier(item["candidate_id"], label="RATE_CANDIDATE_ID") for item in records}
    candidate_parameter_hashes = {str(item["candidate_id"]): str(item["candidate_parameter_hash"]) for item in records}
    if not isinstance(baseline_flow_rows, Sequence) or isinstance(baseline_flow_rows, (str, bytes)):
        raise RateV4GroundworkError("RATE_V4_BASELINE_FLOW_ROWS_VECTOR_REQUIRED")
    if not isinstance(reference_flow_rows, Sequence) or isinstance(reference_flow_rows, (str, bytes)):
        raise RateV4GroundworkError("RATE_V4_REFERENCE_FLOW_ROWS_VECTOR_REQUIRED")
    baseline = [
        _validate_flow_row(
            item, role="BASELINE", rate_design_roots=rate_roots, candidate_ids=candidate_ids,
            candidate_parameter_hashes=candidate_parameter_hashes,
            anchor_domain_hash=anchor_domain_hash,
        )
        for item in baseline_flow_rows
    ]
    reference = [
        _validate_flow_row(
            item, role="REFERENCE", rate_design_roots=rate_roots, candidate_ids=candidate_ids,
            candidate_parameter_hashes=candidate_parameter_hashes,
            anchor_domain_hash=anchor_domain_hash,
        )
        for item in reference_flow_rows
    ]
    if not baseline or not reference:
        raise RateV4GroundworkError("RATE_V4_BASELINE_AND_REFERENCE_FLOWS_REQUIRED")
    baseline_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in baseline:
        key = (row["root_id"], row["q_mode"], row["branch_id"])
        if key in baseline_index:
            raise RateV4GroundworkError("RATE_V4_BASELINE_FLOW_DUPLICATE")
        baseline_index[key] = row
    branch_ids = {row["branch_id"] for row in baseline}
    expected_baseline = {(root, q_mode, branch) for root in rate_roots for q_mode in Q_MODES for branch in branch_ids}
    if set(baseline_index) != expected_baseline:
        raise RateV4GroundworkError("RATE_V4_BASELINE_FLOW_COVERAGE_INCOMPLETE")
    reference_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in reference:
        key = (row["rate_candidate_id"], row["root_id"], row["q_mode"], row["branch_id"])
        if key in reference_index:
            raise RateV4GroundworkError("RATE_V4_REFERENCE_FLOW_DUPLICATE")
        reference_index[key] = row
    expected_reference = {
        (candidate, root, q_mode, branch)
        for candidate in candidate_ids for root in rate_roots for q_mode in Q_MODES for branch in branch_ids
    }
    if set(reference_index) != expected_reference:
        raise RateV4GroundworkError("RATE_V4_REFERENCE_FLOW_COVERAGE_INCOMPLETE")

    baseline_mva = {
        key: two_ended_apparent_power_mva(
            p_from_mw=row["p_from_mw"], q_from_mvar=row["q_from_mvar"],
            p_to_mw=row["p_to_mw"], q_to_mvar=row["q_to_mvar"],
        )
        for key, row in baseline_index.items()
    }
    reference_mva = {
        key: two_ended_apparent_power_mva(
            p_from_mw=row["p_from_mw"], q_from_mvar=row["q_from_mvar"],
            p_to_mw=row["p_to_mw"], q_to_mvar=row["q_to_mvar"],
        )
        for key, row in reference_index.items()
    }
    baseline_by_branch = {
        branch: max(baseline_mva[(root, q_mode, branch)]["s_max_mva"] for root in rate_roots for q_mode in Q_MODES)
        for branch in sorted(branch_ids)
    }
    candidate_anchor_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        branch_rows: list[dict[str, Any]] = []
        for branch_id in sorted(branch_ids):
            reference_anchor = max(
                reference_mva[(candidate_id, root, q_mode, branch_id)]["s_max_mva"]
                for root in rate_roots for q_mode in Q_MODES
            )
            baseline_anchor = baseline_by_branch[branch_id]
            branch_rows.append({
                "branch_id": branch_id,
                "baseline_anchor_mva": baseline_anchor,
                "honest_reference_anchor_mva": reference_anchor,
                "anchor_mva": max(baseline_anchor, reference_anchor),
                "observable": "MAX_TWO_ENDED_APPARENT_POWER_MVA",
            })
        candidate_anchor_rows.append({
            "rate_candidate_id": candidate_id,
            "branch_anchors": branch_rows,
            "branch_anchor_hash": canonical_hash(branch_rows),
        })
    result = {
        "serialization_id": ANCHOR_OBSERVATION_SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_TWO_ENDED_ANCHOR_READY",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "rate_design_split_hash": split_hash,
        "candidate_library_hash": library_hash,
        "q_modes": list(Q_MODES),
        "baseline_flow_input_hash": canonical_hash(baseline),
        "reference_flow_input_hash": canonical_hash(reference),
        "candidate_anchor_rows": candidate_anchor_rows,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(result)


def validate_two_ended_rate_anchor_observation(
    observation: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    anchor_domain: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_library: Mapping[str, Any],
) -> str:
    """Perform structural/hash validation on a generated anchor observation."""

    protocol_hash = validate_rate_design_protocol(protocol)
    anchor_domain_hash = validate_rate_anchor_domain(anchor_domain, protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    library_hash = validate_predeclared_rate_candidate_library(candidate_library, protocol)
    payload = _mapping(observation, label="RATE_V4_ANCHOR_OBSERVATION")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "anchor_domain_hash", "rate_design_split_hash",
            "candidate_library_hash", "q_modes", "baseline_flow_input_hash", "reference_flow_input_hash",
            "candidate_anchor_rows", "input_read_assertions", "historical_output_reused", "result_hash",
        },
        label="RATE_V4_ANCHOR_OBSERVATION",
    )
    if payload["serialization_id"] != ANCHOR_OBSERVATION_SERIALIZATION_ID or payload["status"] != "FRESH_RATE_DESIGN_TWO_ENDED_ANCHOR_READY":
        raise RateV4GroundworkError("RATE_V4_ANCHOR_OBSERVATION_IDENTITY_INVALID")
    if (
        payload["protocol_hash"] != protocol_hash or payload["anchor_domain_hash"] != anchor_domain_hash
        or payload["rate_design_split_hash"] != split_hash or payload["candidate_library_hash"] != library_hash
        or payload["q_modes"] != list(Q_MODES)
    ):
        raise RateV4GroundworkError("RATE_V4_ANCHOR_OBSERVATION_BINDING_INVALID")
    _sha256(payload["baseline_flow_input_hash"], label="RATE_V4_ANCHOR_BASELINE_HASH")
    _sha256(payload["reference_flow_input_hash"], label="RATE_V4_ANCHOR_REFERENCE_HASH")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_V4_ANCHOR_OBSERVATION")
    if payload["historical_output_reused"] is not False or _contains_legacy_v3(payload):
        raise RateV4GroundworkError("RATE_V4_ANCHOR_OBSERVATION_LEGACY_OR_PROMOTION_INVALID")
    expected_candidates = {
        item["candidate_id"] for item in _mapping(candidate_library, label="RATE_CANDIDATE_LIBRARY")["candidate_records"]
    }
    rows = payload["candidate_anchor_rows"]
    if not isinstance(rows, list) or {item.get("rate_candidate_id") for item in rows if isinstance(item, Mapping)} != expected_candidates:
        raise RateV4GroundworkError("RATE_V4_ANCHOR_OBSERVATION_CANDIDATE_COVERAGE_INVALID")
    observed_branch_ids: set[str] | None = None
    for raw in rows:
        row = _mapping(raw, label="RATE_V4_ANCHOR_CANDIDATE_ROW")
        _assert_exact_keys(row, {"rate_candidate_id", "branch_anchors", "branch_anchor_hash"}, label="RATE_V4_ANCHOR_CANDIDATE_ROW")
        anchors = row["branch_anchors"]
        if not isinstance(anchors, list) or not anchors:
            raise RateV4GroundworkError("RATE_V4_ANCHOR_BRANCH_ROWS_REQUIRED")
        if row["branch_anchor_hash"] != canonical_hash(anchors):
            raise RateV4GroundworkError("RATE_V4_ANCHOR_BRANCH_HASH_INVALID")
        branch_ids: set[str] = set()
        for raw_anchor in anchors:
            anchor = _mapping(raw_anchor, label="RATE_V4_BRANCH_ANCHOR")
            _assert_exact_keys(
                anchor,
                {"branch_id", "baseline_anchor_mva", "honest_reference_anchor_mva", "anchor_mva", "observable"},
                label="RATE_V4_BRANCH_ANCHOR",
            )
            branch_id = _identifier(anchor["branch_id"], label="RATE_V4_BRANCH_ANCHOR_ID")
            if branch_id in branch_ids:
                raise RateV4GroundworkError("RATE_V4_BRANCH_ANCHOR_DUPLICATE")
            branch_ids.add(branch_id)
            baseline = _finite(anchor["baseline_anchor_mva"], label="RATE_V4_BRANCH_BASELINE_ANCHOR", nonnegative=True)
            reference = _finite(anchor["honest_reference_anchor_mva"], label="RATE_V4_BRANCH_REFERENCE_ANCHOR", nonnegative=True)
            actual = _finite(anchor["anchor_mva"], label="RATE_V4_BRANCH_ANCHOR", nonnegative=True)
            if not math.isclose(actual, max(baseline, reference), rel_tol=0.0, abs_tol=1e-12):
                raise RateV4GroundworkError("RATE_V4_BRANCH_ANCHOR_FORMULA_INVALID")
            if anchor["observable"] != "MAX_TWO_ENDED_APPARENT_POWER_MVA":
                raise RateV4GroundworkError("RATE_V4_BRANCH_ANCHOR_OBSERVABLE_INVALID")
        if observed_branch_ids is None:
            observed_branch_ids = branch_ids
        elif observed_branch_ids != branch_ids:
            raise RateV4GroundworkError("RATE_V4_ANCHOR_BRANCH_COVERAGE_MISMATCH")
    return _self_hash(payload, field="result_hash", label="RATE_V4_ANCHOR_OBSERVATION")


def _validate_topology_rows(value: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RateV4GroundworkError("RATE_V4_TOPOLOGY_ROWS_VECTOR_REQUIRED")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in value:
        row = _mapping(raw, label="RATE_V4_TOPOLOGY_ROW")
        _assert_exact_keys(row, {"branch_id", "from_bus_id", "to_bus_id", "branch_class"}, label="RATE_V4_TOPOLOGY_ROW")
        branch_id = _identifier(row["branch_id"], label="RATE_V4_TOPOLOGY_BRANCH_ID")
        if branch_id in seen:
            raise RateV4GroundworkError("RATE_V4_TOPOLOGY_BRANCH_DUPLICATE")
        seen.add(branch_id)
        branch_class = _string(row["branch_class"], label="RATE_V4_TOPOLOGY_BRANCH_CLASS")
        if branch_class not in BRANCH_CLASSES:
            raise RateV4GroundworkError("RATE_V4_TOPOLOGY_BRANCH_CLASS_INVALID")
        from_bus = _identifier(row["from_bus_id"], label="RATE_V4_TOPOLOGY_FROM_BUS")
        to_bus = _identifier(row["to_bus_id"], label="RATE_V4_TOPOLOGY_TO_BUS")
        if from_bus == to_bus:
            raise RateV4GroundworkError("RATE_V4_TOPOLOGY_SELF_LOOP_FORBIDDEN")
        rows.append({"branch_id": branch_id, "from_bus_id": from_bus, "to_bus_id": to_bus, "branch_class": branch_class})
    if not rows:
        raise RateV4GroundworkError("RATE_V4_TOPOLOGY_ROWS_REQUIRED")
    return sorted(rows, key=lambda item: item["branch_id"])


def _validate_candidate_policy(policy: Mapping[str, Any]) -> str:
    payload = _mapping(policy, label="RATE_V4_CANDIDATE_POLICY")
    required = {
        "serialization_id", "policy_id", "version", "candidate_id", "policy_status", "scientific_role",
        "network_hash", "raw_case_hash", "protocol_hash", "anchor_domain_hash", "rate_design_split_hash",
        "candidate_library_hash", "anchor_observation_hash", "candidate_parameter_hash", "q_modes",
        "branch_ratings", "input_read_assertions", "historical_output_reused", "policy_hash",
    }
    _assert_exact_keys(payload, required, label="RATE_V4_CANDIDATE_POLICY")
    if payload["serialization_id"] != CANDIDATE_POLICY_SERIALIZATION_ID or payload["version"] != "V4":
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_IDENTITY_INVALID")
    if payload["policy_status"] not in {"DEVELOPMENT_ONLY_NOT_FROZEN", "FROZEN_PRE_SOURCE_SCENARIO_INPUT"}:
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_STATUS_INVALID")
    if payload["scientific_role"] != "SYNTHETIC_SCENARIO_DEFINED_FUTURE_INPUT":
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_ROLE_INVALID")
    _sha256(payload["network_hash"], label="RATE_V4_CANDIDATE_POLICY_NETWORK_HASH")
    _sha256(payload["raw_case_hash"], label="RATE_V4_CANDIDATE_POLICY_CASE_HASH")
    for field in (
        "protocol_hash", "anchor_domain_hash", "rate_design_split_hash", "candidate_library_hash",
        "anchor_observation_hash", "candidate_parameter_hash",
    ):
        _sha256(payload[field], label=f"RATE_V4_CANDIDATE_POLICY_{field.upper()}")
    if payload["q_modes"] != list(Q_MODES):
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_Q_MODES_INVALID")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_V4_CANDIDATE_POLICY")
    if payload["historical_output_reused"] is not False or _contains_legacy_v3(payload):
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_LEGACY_INPUT_FORBIDDEN")
    rows = payload["branch_ratings"]
    if not isinstance(rows, list) or not rows:
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_BRANCH_RATINGS_REQUIRED")
    branch_ids: set[str] = set()
    for raw in rows:
        rating = _mapping(raw, label="RATE_V4_BRANCH_RATING")
        _assert_exact_keys(
            rating,
            {
                "branch_id", "from_bus_id", "to_bus_id", "branch_class", "anchor_mva",
                "baseline_anchor_mva", "honest_reference_anchor_mva", "anchor_multiplier",
                "anchor_minimum_mva", "anchor_limit_mva", "final_headroom_ratio", "final_minimum_mva", "rate_a_mva",
            },
            label="RATE_V4_BRANCH_RATING",
        )
        branch_id = _identifier(rating["branch_id"], label="RATE_V4_BRANCH_RATING_ID")
        if branch_id in branch_ids:
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_DUPLICATE")
        branch_ids.add(branch_id)
        if _string(rating["branch_class"], label="RATE_V4_BRANCH_CLASS") not in BRANCH_CLASSES:
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_CLASS_INVALID")
        baseline_anchor = _finite(rating["baseline_anchor_mva"], label="RATE_V4_BRANCH_RATING_BASELINE_ANCHOR", nonnegative=True)
        reference_anchor = _finite(rating["honest_reference_anchor_mva"], label="RATE_V4_BRANCH_RATING_REFERENCE_ANCHOR", nonnegative=True)
        anchor = _finite(rating["anchor_mva"], label="RATE_V4_BRANCH_RATING_ANCHOR", nonnegative=True)
        if not math.isclose(anchor, max(baseline_anchor, reference_anchor), rel_tol=0.0, abs_tol=1e-12):
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_OBSERVATION_FORMULA_INVALID")
        multiplier = _finite(rating["anchor_multiplier"], label="RATE_V4_BRANCH_RATING_MULTIPLIER", positive=True)
        if multiplier <= 1.0:
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_MULTIPLIER_INVALID")
        anchor_minimum = _finite(rating["anchor_minimum_mva"], label="RATE_V4_BRANCH_RATING_ANCHOR_MINIMUM", positive=True)
        anchor_limit = _finite(rating["anchor_limit_mva"], label="RATE_V4_BRANCH_RATING_ANCHOR_LIMIT", positive=True)
        if not math.isclose(anchor_limit, max(anchor_minimum, multiplier * baseline_anchor), rel_tol=0.0, abs_tol=1e-12):
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_ANCHOR_LIMIT_FORMULA_INVALID")
        headroom = _finite(rating["final_headroom_ratio"], label="RATE_V4_BRANCH_RATING_HEADROOM", nonnegative=True)
        minimum = _finite(rating["final_minimum_mva"], label="RATE_V4_BRANCH_RATING_FINAL_MINIMUM", positive=True)
        actual = _finite(rating["rate_a_mva"], label="RATE_V4_BRANCH_RATING_RATE", positive=True)
        expected = max(minimum, (1.0 + headroom) * anchor)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RateV4GroundworkError("RATE_V4_BRANCH_RATING_FORMULA_INVALID")
    declared = payload.get("policy_hash")
    if declared != _policy_hash(payload):
        raise RateV4GroundworkError("RATE_V4_CANDIDATE_POLICY_HASH_MISMATCH")
    return str(declared)


def build_rate_v4_candidate_policy_bundle(
    *,
    protocol: Mapping[str, Any],
    anchor_domain: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_library: Mapping[str, Any],
    anchor_observation: Mapping[str, Any],
    topology_rows: Sequence[Mapping[str, Any]],
    network_hash: str,
    raw_case_hash: str,
) -> dict[str, Any]:
    """Apply the V4 rate formula to fresh per-candidate anchor observations."""

    protocol_hash = validate_rate_design_protocol(protocol)
    anchor_domain_hash = validate_rate_anchor_domain(anchor_domain, protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    library_hash = validate_predeclared_rate_candidate_library(candidate_library, protocol)
    observation_hash = validate_two_ended_rate_anchor_observation(
        anchor_observation,
        protocol=protocol,
        anchor_domain=anchor_domain,
        root_split=root_split,
        candidate_library=candidate_library,
    )
    network = _sha256(network_hash, label="RATE_V4_NETWORK_HASH")
    case = _sha256(raw_case_hash, label="RATE_V4_RAW_CASE_HASH")
    topology = _validate_topology_rows(topology_rows)
    topology_by_branch = {row["branch_id"]: row for row in topology}
    observation_rows = _mapping(anchor_observation, label="RATE_V4_ANCHOR_OBSERVATION")["candidate_anchor_rows"]
    observations_by_candidate = {row["rate_candidate_id"]: row for row in observation_rows}
    policies: list[dict[str, Any]] = []
    for candidate in _mapping(candidate_library, label="RATE_CANDIDATE_LIBRARY")["candidate_records"]:
        candidate_id = candidate["candidate_id"]
        parameters = candidate["class_parameters"]
        anchor_rows = observations_by_candidate[candidate_id]["branch_anchors"]
        anchor_by_branch = {row["branch_id"]: row for row in anchor_rows}
        if set(anchor_by_branch) != set(topology_by_branch):
            raise RateV4GroundworkError("RATE_V4_POLICY_TOPOLOGY_ANCHOR_COVERAGE_MISMATCH")
        ratings: list[dict[str, Any]] = []
        for branch_id in sorted(topology_by_branch):
            topology_row = topology_by_branch[branch_id]
            branch_class = topology_row["branch_class"]
            parameter = parameters[branch_class]
            anchor_row = anchor_by_branch[branch_id]
            baseline_anchor_mva = float(anchor_row["baseline_anchor_mva"])
            honest_reference_anchor_mva = float(anchor_row["honest_reference_anchor_mva"])
            anchor_mva = float(anchor_row["anchor_mva"])
            ratings.append({
                "branch_id": branch_id,
                "from_bus_id": topology_row["from_bus_id"],
                "to_bus_id": topology_row["to_bus_id"],
                "branch_class": branch_class,
                "anchor_mva": anchor_mva,
                "baseline_anchor_mva": baseline_anchor_mva,
                "honest_reference_anchor_mva": honest_reference_anchor_mva,
                "anchor_multiplier": parameter["anchor_multiplier"],
                "anchor_minimum_mva": parameter["anchor_minimum_mva"],
                "anchor_limit_mva": max(
                    parameter["anchor_minimum_mva"],
                    parameter["anchor_multiplier"] * baseline_anchor_mva,
                ),
                "final_headroom_ratio": parameter["final_headroom_ratio"],
                "final_minimum_mva": parameter["final_minimum_mva"],
                "rate_a_mva": max(
                    parameter["final_minimum_mva"],
                    (1.0 + parameter["final_headroom_ratio"]) * anchor_mva,
                ),
            })
        policy = {
            "serialization_id": CANDIDATE_POLICY_SERIALIZATION_ID,
            "policy_id": f"CASE141_SYNTHETIC_RATE_V4::{candidate_id}",
            "version": "V4",
            "candidate_id": candidate_id,
            "policy_status": "DEVELOPMENT_ONLY_NOT_FROZEN",
            "scientific_role": "SYNTHETIC_SCENARIO_DEFINED_FUTURE_INPUT",
            "network_hash": network,
            "raw_case_hash": case,
            "protocol_hash": protocol_hash,
            "anchor_domain_hash": anchor_domain_hash,
            "rate_design_split_hash": split_hash,
            "candidate_library_hash": library_hash,
            "anchor_observation_hash": observation_hash,
            "candidate_parameter_hash": candidate["candidate_parameter_hash"],
            "q_modes": list(Q_MODES),
            "branch_ratings": ratings,
            "input_read_assertions": _fresh_read_assertions(),
            "historical_output_reused": False,
            "policy_hash": None,
        }
        policy["policy_hash"] = _policy_hash(policy)
        _validate_candidate_policy(policy)
        policies.append(policy)
    result = {
        "serialization_id": CANDIDATE_POLICY_BUNDLE_SERIALIZATION_ID,
        "status": "RATE_V4_CANDIDATE_POLICIES_READY_FOR_DEVELOPMENT_QUALITY_AUDIT",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "rate_design_split_hash": split_hash,
        "candidate_library_hash": library_hash,
        "anchor_observation_hash": observation_hash,
        "network_hash": network,
        "raw_case_hash": case,
        "topology_hash": canonical_hash(topology),
        "candidate_policies": policies,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(result)


def validate_rate_v4_candidate_policy_bundle(bundle: Mapping[str, Any]) -> str:
    payload = _mapping(bundle, label="RATE_V4_POLICY_BUNDLE")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "anchor_domain_hash", "rate_design_split_hash",
            "candidate_library_hash", "anchor_observation_hash", "network_hash", "raw_case_hash",
            "topology_hash", "candidate_policies", "input_read_assertions", "historical_output_reused", "result_hash",
        },
        label="RATE_V4_POLICY_BUNDLE",
    )
    if payload["serialization_id"] != CANDIDATE_POLICY_BUNDLE_SERIALIZATION_ID or payload["status"] != "RATE_V4_CANDIDATE_POLICIES_READY_FOR_DEVELOPMENT_QUALITY_AUDIT":
        raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_IDENTITY_INVALID")
    for field in (
        "protocol_hash", "anchor_domain_hash", "rate_design_split_hash", "candidate_library_hash",
        "anchor_observation_hash", "network_hash", "raw_case_hash", "topology_hash",
    ):
        _sha256(payload[field], label=f"RATE_V4_POLICY_BUNDLE_{field.upper()}")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_V4_POLICY_BUNDLE")
    if payload["historical_output_reused"] is not False or _contains_legacy_v3(payload):
        raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_LEGACY_INPUT_FORBIDDEN")
    policies = payload["candidate_policies"]
    if not isinstance(policies, list) or not policies:
        raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_POLICIES_REQUIRED")
    identifiers: set[str] = set()
    for policy in policies:
        policy_hash = _validate_candidate_policy(policy)
        if policy["policy_status"] != "DEVELOPMENT_ONLY_NOT_FROZEN":
            raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_PREMATURE_FREEZE_FORBIDDEN")
        candidate_id = policy["candidate_id"]
        if candidate_id in identifiers:
            raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_CANDIDATE_DUPLICATE")
        identifiers.add(candidate_id)
        if policy["protocol_hash"] != payload["protocol_hash"] or policy["anchor_domain_hash"] != payload["anchor_domain_hash"]:
            raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_POLICY_BINDING_MISMATCH")
        if policy_hash != policy["policy_hash"]:
            raise RateV4GroundworkError("RATE_V4_POLICY_BUNDLE_POLICY_HASH_INVALID")
    return _self_hash(payload, field="result_hash", label="RATE_V4_POLICY_BUNDLE")


def _validate_quality_record(
    value: Any,
    *,
    candidate_ids: set[str],
    policy_hashes: Mapping[str, str],
    split_hash: str,
) -> dict[str, Any]:
    record = _mapping(value, label="RATE_V4_QUALITY_RECORD")
    _assert_exact_keys(
        record,
        {
            "candidate_id", "policy_hash", "rate_design_split_hash", "baseline_ac_all_pass",
            "honest_reference_pass_rate", "strategic_unmitigated_pair_pass_rate",
            "input_read_assertions", "historical_output_reused",
        },
        label="RATE_V4_QUALITY_RECORD",
    )
    candidate_id = _identifier(record["candidate_id"], label="RATE_V4_QUALITY_CANDIDATE")
    if candidate_id not in candidate_ids or record["policy_hash"] != policy_hashes[candidate_id]:
        raise RateV4GroundworkError("RATE_V4_QUALITY_POLICY_BINDING_INVALID")
    if record["rate_design_split_hash"] != split_hash:
        raise RateV4GroundworkError("RATE_V4_QUALITY_SPLIT_BINDING_INVALID")
    if not isinstance(record["baseline_ac_all_pass"], bool):
        raise RateV4GroundworkError("RATE_V4_QUALITY_BASELINE_BOOLEAN_REQUIRED")
    reference = _finite(record["honest_reference_pass_rate"], label="RATE_V4_QUALITY_REFERENCE_RATE", nonnegative=True)
    strategic = _finite(record["strategic_unmitigated_pair_pass_rate"], label="RATE_V4_QUALITY_STRATEGIC_RATE", nonnegative=True)
    if reference > 1.0 or strategic > 1.0:
        raise RateV4GroundworkError("RATE_V4_QUALITY_RATE_RANGE_INVALID")
    _validate_input_read_assertions(record["input_read_assertions"], label="RATE_V4_QUALITY_RECORD")
    if record["historical_output_reused"] is not False or _contains_legacy_v3(record):
        raise RateV4GroundworkError("RATE_V4_QUALITY_LEGACY_INPUT_FORBIDDEN")
    return {
        "candidate_id": candidate_id,
        "policy_hash": record["policy_hash"],
        "rate_design_split_hash": split_hash,
        "baseline_ac_all_pass": record["baseline_ac_all_pass"],
        "honest_reference_pass_rate": reference,
        "strategic_unmitigated_pair_pass_rate": strategic,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
    }


def select_rate_v4_candidate(
    *,
    protocol: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_policy_bundle: Mapping[str, Any],
    quality_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one rate candidate using only pre-registered development quality gates."""

    protocol_hash = validate_rate_design_protocol(protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    bundle_hash = validate_rate_v4_candidate_policy_bundle(candidate_policy_bundle)
    bundle = _mapping(candidate_policy_bundle, label="RATE_V4_POLICY_BUNDLE")
    if bundle["protocol_hash"] != protocol_hash or bundle["rate_design_split_hash"] != split_hash:
        raise RateV4GroundworkError("RATE_V4_SELECTION_BUNDLE_BINDING_INVALID")
    policies = bundle["candidate_policies"]
    candidate_ids = {policy["candidate_id"] for policy in policies}
    policy_by_id = {policy["candidate_id"]: policy for policy in policies}
    policy_hashes = {candidate_id: policy["policy_hash"] for candidate_id, policy in policy_by_id.items()}
    if not isinstance(quality_records, Sequence) or isinstance(quality_records, (str, bytes)):
        raise RateV4GroundworkError("RATE_V4_QUALITY_RECORDS_VECTOR_REQUIRED")
    normalized = [
        _validate_quality_record(
            row, candidate_ids=candidate_ids, policy_hashes=policy_hashes, split_hash=split_hash,
        )
        for row in quality_records
    ]
    normalized.sort(key=lambda item: item["candidate_id"])
    if len(normalized) != len(candidate_ids) or {row["candidate_id"] for row in normalized} != candidate_ids:
        raise RateV4GroundworkError("RATE_V4_QUALITY_CANDIDATE_COVERAGE_INCOMPLETE")
    quality = _mapping(_mapping(protocol, label="RATE_DESIGN_PROTOCOL")["quality_selection"], label="RATE_DESIGN_QUALITY")
    minimum_reference = float(quality["honest_reference_minimum_pass_rate"])
    lower, upper = [float(item) for item in quality["strategic_unmitigated_pair_pass_rate_interval"]]
    target = float(quality["strategic_unmitigated_pair_pass_rate_target"])
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[float, str, dict[str, Any]]] = []
    for row in sorted(normalized, key=lambda item: item["candidate_id"]):
        reasons: list[str] = []
        if row["baseline_ac_all_pass"] is not True:
            reasons.append("BASELINE_AC_NOT_ALL_PASS")
        if row["honest_reference_pass_rate"] < minimum_reference:
            reasons.append("HONEST_REFERENCE_PASS_RATE_BELOW_MINIMUM")
        if not lower <= row["strategic_unmitigated_pair_pass_rate"] <= upper:
            reasons.append("STRATEGIC_UNMITIGATED_PAIR_PASS_RATE_OUTSIDE_PREREGISTERED_INTERVAL")
        score = abs(row["strategic_unmitigated_pair_pass_rate"] - target)
        decision = {
            **row,
            "eligible": not reasons,
            "failure_reasons": reasons,
            "strategic_target_distance": score,
        }
        decisions.append(decision)
        if not reasons:
            eligible.append((score, row["policy_hash"], decision))
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected = None
    status = "NO_RATE_V4_CANDIDATE_PASSED_DEVELOPMENT_GATE"
    if eligible:
        winner = eligible[0][2]
        selected = {
            "candidate_id": winner["candidate_id"],
            "policy_hash": winner["policy_hash"],
            "strategic_target_distance": winner["strategic_target_distance"],
        }
        status = "RATE_V4_CANDIDATE_SELECTED_DEVELOPMENT_ONLY"
    result = {
        "serialization_id": SELECTION_SERIALIZATION_ID,
        "status": status,
        "protocol_hash": protocol_hash,
        "rate_design_split_hash": split_hash,
        "candidate_policy_bundle_hash": bundle_hash,
        "quality_records_hash": canonical_hash(normalized),
        "quality_decisions": decisions,
        "selected": selected,
        "selection_order": [
            "eligible_first", "absolute_distance_to_strategic_target_ascending", "rate_policy_hash_ascending",
        ],
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(result)


def validate_rate_v4_selection(
    selection: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_policy_bundle: Mapping[str, Any],
) -> str:
    """Validate a selection result structurally; replay requires the raw quality input."""

    protocol_hash = validate_rate_design_protocol(protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    bundle_hash = validate_rate_v4_candidate_policy_bundle(candidate_policy_bundle)
    payload = _mapping(selection, label="RATE_V4_SELECTION")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "rate_design_split_hash",
            "candidate_policy_bundle_hash", "quality_records_hash", "quality_decisions", "selected",
            "selection_order", "input_read_assertions", "historical_output_reused", "result_hash",
        },
        label="RATE_V4_SELECTION",
    )
    if payload["serialization_id"] != SELECTION_SERIALIZATION_ID or payload["status"] not in {
        "RATE_V4_CANDIDATE_SELECTED_DEVELOPMENT_ONLY", "NO_RATE_V4_CANDIDATE_PASSED_DEVELOPMENT_GATE",
    }:
        raise RateV4GroundworkError("RATE_V4_SELECTION_IDENTITY_INVALID")
    if payload["protocol_hash"] != protocol_hash or payload["rate_design_split_hash"] != split_hash or payload["candidate_policy_bundle_hash"] != bundle_hash:
        raise RateV4GroundworkError("RATE_V4_SELECTION_BINDING_INVALID")
    _sha256(payload["quality_records_hash"], label="RATE_V4_SELECTION_QUALITY_HASH")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_V4_SELECTION")
    if payload["historical_output_reused"] is not False or _contains_legacy_v3(payload):
        raise RateV4GroundworkError("RATE_V4_SELECTION_LEGACY_INPUT_FORBIDDEN")
    decisions = payload["quality_decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise RateV4GroundworkError("RATE_V4_SELECTION_DECISIONS_REQUIRED")
    raw_quality: list[dict[str, Any]] = []
    for item in decisions:
        decision = _mapping(item, label="RATE_V4_SELECTION_DECISION")
        try:
            raw_quality.append({
                key: decision[key]
                for key in (
                    "candidate_id", "policy_hash", "rate_design_split_hash", "baseline_ac_all_pass",
                    "honest_reference_pass_rate", "strategic_unmitigated_pair_pass_rate",
                    "input_read_assertions", "historical_output_reused",
                )
            })
        except KeyError as exc:
            raise RateV4GroundworkError("RATE_V4_SELECTION_DECISION_FIELDS_INVALID") from exc
    expected = select_rate_v4_candidate(
        protocol=protocol,
        root_split=root_split,
        candidate_policy_bundle=candidate_policy_bundle,
        quality_records=raw_quality,
    )
    if payload != expected:
        raise RateV4GroundworkError("RATE_V4_SELECTION_REPLAY_MISMATCH")
    return _self_hash(payload, field="result_hash", label="RATE_V4_SELECTION")


def build_immutable_rate_v4_freeze_attestation(
    *,
    protocol: Mapping[str, Any],
    anchor_domain: Mapping[str, Any],
    root_split: Mapping[str, Any],
    candidate_library: Mapping[str, Any],
    anchor_observation: Mapping[str, Any],
    candidate_policy_bundle: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one selected V4 policy before successor source execution.

    The returned policy is a new serialization value with a new policy hash;
    no candidate policy is mutated in place.  It remains a frozen scenario
    input, not primary evidence or a manuscript result.
    """

    protocol_hash = validate_rate_design_protocol(protocol)
    anchor_domain_hash = validate_rate_anchor_domain(anchor_domain, protocol)
    split_hash = validate_rate_design_root_split(root_split, protocol)
    library_hash = validate_predeclared_rate_candidate_library(candidate_library, protocol)
    anchor_hash = validate_two_ended_rate_anchor_observation(
        anchor_observation,
        protocol=protocol,
        anchor_domain=anchor_domain,
        root_split=root_split,
        candidate_library=candidate_library,
    )
    bundle_hash = validate_rate_v4_candidate_policy_bundle(candidate_policy_bundle)
    selection_hash = validate_rate_v4_selection(
        selection,
        protocol=protocol,
        root_split=root_split,
        candidate_policy_bundle=candidate_policy_bundle,
    )
    selected = _mapping(selection, label="RATE_V4_SELECTION").get("selected")
    if not isinstance(selected, Mapping):
        raise RateV4GroundworkError("RATE_V4_FREEZE_REQUIRES_SELECTED_CANDIDATE")
    selected_id = _identifier(selected.get("candidate_id"), label="RATE_V4_FREEZE_CANDIDATE_ID")
    source_policy = next(
        (item for item in _mapping(candidate_policy_bundle, label="RATE_V4_POLICY_BUNDLE")["candidate_policies"] if item["candidate_id"] == selected_id),
        None,
    )
    if source_policy is None or source_policy["policy_hash"] != selected.get("policy_hash"):
        raise RateV4GroundworkError("RATE_V4_FREEZE_SELECTED_POLICY_BINDING_INVALID")
    frozen_policy = deepcopy(source_policy)
    frozen_policy["policy_status"] = "FROZEN_PRE_SOURCE_SCENARIO_INPUT"
    frozen_policy["policy_hash"] = None
    frozen_policy["policy_hash"] = _policy_hash(frozen_policy)
    frozen_hash = _validate_candidate_policy(frozen_policy)
    result = {
        "serialization_id": FREEZE_ATTESTATION_SERIALIZATION_ID,
        "status": "FROZEN_RATE_V4_SCENARIO_INPUT_PRE_SOURCE",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "rate_design_split_hash": split_hash,
        "candidate_library_hash": library_hash,
        "baseline_anchor_hash": anchor_hash,
        "candidate_policy_bundle_hash": bundle_hash,
        "selection_hash": selection_hash,
        "pre_freeze_candidate_policy_hash": source_policy["policy_hash"],
        "frozen_rate_policy_hash": frozen_hash,
        "frozen_rate_policy": frozen_policy,
        "freeze_boundary": {
            "must_precede": [
                "V12_3_OR_SUCCESSOR_SOURCE_ENVELOPE",
                "V12_3_OR_SUCCESSOR_SOURCE_CERTIFICATE_EXECUTION",
                "CANDIDATE_TRUST_CONSTRUCTION",
                "CALIBRATION",
                "FINAL_EVALUATION",
                "MITIGATION",
            ],
            "immutable_after_freeze": True,
            "policy_mutation_requires_new_version_new_split_new_full_evidence_chain": True,
            "frozen_policy_is_primary_evidence": False,
        },
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(result)


def validate_immutable_rate_v4_freeze_attestation(attestation: Mapping[str, Any]) -> str:
    """Validate the immutable-freeze schema without reading any external output."""

    payload = _mapping(attestation, label="RATE_V4_FREEZE_ATTESTATION")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "protocol_hash", "anchor_domain_hash", "rate_design_split_hash",
            "candidate_library_hash", "baseline_anchor_hash", "candidate_policy_bundle_hash", "selection_hash",
            "pre_freeze_candidate_policy_hash", "frozen_rate_policy_hash", "frozen_rate_policy", "freeze_boundary",
            "input_read_assertions", "historical_output_reused", "result_hash",
        },
        label="RATE_V4_FREEZE_ATTESTATION",
    )
    if payload["serialization_id"] != FREEZE_ATTESTATION_SERIALIZATION_ID or payload["status"] != "FROZEN_RATE_V4_SCENARIO_INPUT_PRE_SOURCE":
        raise RateV4GroundworkError("RATE_V4_FREEZE_ATTESTATION_IDENTITY_INVALID")
    for field in (
        "protocol_hash", "anchor_domain_hash", "rate_design_split_hash", "candidate_library_hash",
        "baseline_anchor_hash", "candidate_policy_bundle_hash", "selection_hash",
        "pre_freeze_candidate_policy_hash", "frozen_rate_policy_hash",
    ):
        _sha256(payload[field], label=f"RATE_V4_FREEZE_{field.upper()}")
    _validate_input_read_assertions(payload["input_read_assertions"], label="RATE_V4_FREEZE_ATTESTATION")
    if payload["historical_output_reused"] is not False or _contains_legacy_v3(payload):
        raise RateV4GroundworkError("RATE_V4_FREEZE_LEGACY_INPUT_FORBIDDEN")
    frozen = _mapping(payload["frozen_rate_policy"], label="RATE_V4_FREEZE_POLICY")
    if _validate_candidate_policy(frozen) != payload["frozen_rate_policy_hash"]:
        raise RateV4GroundworkError("RATE_V4_FREEZE_POLICY_HASH_INVALID")
    if frozen["policy_status"] != "FROZEN_PRE_SOURCE_SCENARIO_INPUT":
        raise RateV4GroundworkError("RATE_V4_FREEZE_POLICY_STATUS_INVALID")
    boundary = _mapping(payload["freeze_boundary"], label="RATE_V4_FREEZE_BOUNDARY")
    if boundary.get("immutable_after_freeze") is not True or boundary.get("frozen_policy_is_primary_evidence") is not False:
        raise RateV4GroundworkError("RATE_V4_FREEZE_BOUNDARY_INVALID")
    return _self_hash(payload, field="result_hash", label="RATE_V4_FREEZE_ATTESTATION")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return _mapping(value, label=str(path))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, label=str(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI covered through pure builders
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-specifications")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--anchor-domain", type=Path, required=True)
    library = subparsers.add_parser("build-library")
    library.add_argument("--protocol", type=Path, required=True)
    library.add_argument("--output", type=Path, required=True)
    split = subparsers.add_parser("build-split")
    split.add_argument("--protocol", type=Path, required=True)
    split.add_argument("--root-roles", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    anchor = subparsers.add_parser("build-anchor")
    anchor.add_argument("--protocol", type=Path, required=True)
    anchor.add_argument("--anchor-domain", type=Path, required=True)
    anchor.add_argument("--root-split", type=Path, required=True)
    anchor.add_argument("--candidate-library", type=Path, required=True)
    anchor.add_argument("--baseline-flows", type=Path, required=True)
    anchor.add_argument("--reference-flows", type=Path, required=True)
    anchor.add_argument("--output", type=Path, required=True)
    policies = subparsers.add_parser("build-policies")
    policies.add_argument("--protocol", type=Path, required=True)
    policies.add_argument("--anchor-domain", type=Path, required=True)
    policies.add_argument("--root-split", type=Path, required=True)
    policies.add_argument("--candidate-library", type=Path, required=True)
    policies.add_argument("--anchor-observation", type=Path, required=True)
    policies.add_argument("--topology", type=Path, required=True)
    policies.add_argument("--network-hash", required=True)
    policies.add_argument("--raw-case-hash", required=True)
    policies.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select-freeze")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--anchor-domain", type=Path, required=True)
    select.add_argument("--root-split", type=Path, required=True)
    select.add_argument("--candidate-library", type=Path, required=True)
    select.add_argument("--anchor-observation", type=Path, required=True)
    select.add_argument("--policy-bundle", type=Path, required=True)
    select.add_argument("--quality-records", type=Path, required=True)
    select.add_argument("--selection-output", type=Path, required=True)
    select.add_argument("--freeze-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-specifications":
        protocol = _load_yaml(args.protocol)
        domain = _load_yaml(args.anchor_domain)
        print(canonical_dumps({
            "status": "PASS",
            "protocol_hash": validate_rate_design_protocol(protocol),
            "anchor_domain_hash": validate_rate_anchor_domain(domain, protocol),
        }))
        return 0
    protocol = _load_yaml(args.protocol)
    if args.command == "build-library":
        result = build_predeclared_rate_candidate_library(protocol=protocol)
        _write_json(args.output, result)
        print(canonical_dumps({"status": result["status"], "result_hash": result["result_hash"]}))
        return 0
    if args.command == "build-split":
        roles = _load_json(args.root_roles)
        role_ids = roles.get("root_role_ids", roles)
        result = build_rate_design_root_split(protocol=protocol, root_role_ids=_mapping(role_ids, label="ROOT_ROLE_IDS"))
        _write_json(args.output, result)
        print(canonical_dumps({"status": result["status"], "result_hash": result["result_hash"]}))
        return 0
    anchor_domain = _load_yaml(args.anchor_domain)
    root_split = _load_json(args.root_split)
    candidate_library = _load_json(args.candidate_library)
    if args.command == "build-anchor":
        baseline = _load_json(args.baseline_flows).get("rows")
        reference = _load_json(args.reference_flows).get("rows")
        result = build_two_ended_rate_anchor_observation(
            protocol=protocol, anchor_domain=anchor_domain, root_split=root_split,
            candidate_library=candidate_library, baseline_flow_rows=baseline, reference_flow_rows=reference,
        )
        _write_json(args.output, result)
        print(canonical_dumps({"status": result["status"], "result_hash": result["result_hash"]}))
        return 0
    anchor_observation = _load_json(args.anchor_observation)
    if args.command == "build-policies":
        topology = _load_json(args.topology).get("rows")
        result = build_rate_v4_candidate_policy_bundle(
            protocol=protocol, anchor_domain=anchor_domain, root_split=root_split,
            candidate_library=candidate_library, anchor_observation=anchor_observation,
            topology_rows=topology, network_hash=args.network_hash, raw_case_hash=args.raw_case_hash,
        )
        _write_json(args.output, result)
        print(canonical_dumps({"status": result["status"], "result_hash": result["result_hash"]}))
        return 0
    quality = _load_json(args.quality_records).get("rows")
    bundle = _load_json(args.policy_bundle)
    selection = select_rate_v4_candidate(
        protocol=protocol, root_split=root_split, candidate_policy_bundle=bundle, quality_records=quality,
    )
    _write_json(args.selection_output, selection)
    freeze = build_immutable_rate_v4_freeze_attestation(
        protocol=protocol, anchor_domain=anchor_domain, root_split=root_split,
        candidate_library=candidate_library, anchor_observation=anchor_observation,
        candidate_policy_bundle=bundle, selection=selection,
    )
    _write_json(args.freeze_output, freeze)
    print(canonical_dumps({
        "selection_status": selection["status"], "selection_hash": selection["result_hash"],
        "freeze_status": freeze["status"], "freeze_hash": freeze["result_hash"],
    }))
    return 0


__all__ = [
    "ANCHOR_DOMAIN_ID", "ANCHOR_DOMAIN_SERIALIZATION_ID", "ANCHOR_OBSERVATION_SERIALIZATION_ID",
    "BRANCH_CLASSES", "CANDIDATE_LIBRARY_SERIALIZATION_ID", "CANDIDATE_POLICY_BUNDLE_SERIALIZATION_ID",
    "CANDIDATE_POLICY_SERIALIZATION_ID", "FREEZE_ATTESTATION_SERIALIZATION_ID", "PROTOCOL_ID",
    "PROTOCOL_SERIALIZATION_ID", "Q_MODES", "ROOT_ROLES", "ROOT_SPLIT_SERIALIZATION_ID",
    "RateV4GroundworkError", "SELECTION_SERIALIZATION_ID", "build_immutable_rate_v4_freeze_attestation",
    "build_predeclared_rate_candidate_library", "build_rate_design_root_split",
    "build_rate_v4_candidate_policy_bundle", "build_two_ended_rate_anchor_observation", "main",
    "select_rate_v4_candidate", "two_ended_apparent_power_mva", "validate_immutable_rate_v4_freeze_attestation",
    "validate_predeclared_rate_candidate_library", "validate_rate_anchor_domain", "validate_rate_design_protocol",
    "validate_rate_design_root_split", "validate_rate_v4_candidate_policy_bundle", "validate_rate_v4_selection",
    "validate_two_ended_rate_anchor_observation",
]
