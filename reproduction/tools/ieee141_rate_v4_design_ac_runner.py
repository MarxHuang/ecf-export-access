"""Fresh, development-only AC execution for the IEEE-141 RATE V4 design.

This module is intentionally narrower than a study runner.  It performs the
one-way *scenario-design* portion of the V4 RATE path:

``fresh case -> deterministic RATE_DESIGN split -> fresh baseline AC ->
fresh finite-difference anchor proxy -> candidate-specific DEC015 reference
solve -> fresh reference AC -> development-only quality records``.

The module never reads a historical result directory and it has no inputs for
calibration, trust/candidate construction, guard, evaluation, mitigation, or
manuscript evidence.  Its output is useful only as the fresh input to
``tools.ieee141_rate_v4_groundwork``.  In particular, a successful execution
is a *synthetic-scenario design record*, not T2--T5 evidence.

The actual RATE V4 policy is constructed by the existing groundwork helpers
from the emitted two-ended rows.  The runner then evaluates only the
pre-registered, unmitigated RATE_DESIGN rows against those candidate policies;
it does not choose, freeze, or promote a policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from r4r.ac_solver import ACPowerFlowSolution, solve_ac_power_flow
from r4r.independent_ac_solver import (
    IndependentACPowerFlowSolution,
    crosscheck_ac_solutions,
    solve_independent_radial_ac,
)
from r4r.injection_adapter import build_net_injections_from_operating_point
from r4r.network_parser import ParsedMatpowerCase, parse_matpower_case
from r4r.operating_point import build_canonical_operating_point
from r4r.participant_registry import ParticipantSelection, build_participant_selection
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.types import FloatVector, QAssumption
from tools.diagnostic_ieee141_context import IEEE141_CASE_SHA256
from tools.ieee141_rate_v4_groundwork import (
    ANCHOR_DOMAIN_ID,
    BRANCH_CLASSES,
    Q_MODES,
    ROOT_ROLES,
    RateV4GroundworkError,
    build_predeclared_rate_candidate_library,
    build_rate_design_root_split,
    build_rate_v4_candidate_policy_bundle,
    build_two_ended_rate_anchor_observation,
    validate_rate_anchor_domain,
    validate_rate_design_protocol,
)


SERIALIZATION_ID = "case141_rate_v4_design_ac_execution.v1"
INPUT_SERIALIZATION_ID = "case141_rate_v4_design_execution_input.v1"
BASELINE_SERIALIZATION_ID = "case141_rate_v4_baseline_ac_rows.v1"
REFERENCE_SERIALIZATION_ID = "case141_rate_v4_reference_ac_rows.v1"
PROXY_SERIALIZATION_ID = "case141_rate_v4_anchor_proxy_response.v1"
QUALITY_SERIALIZATION_ID = "case141_rate_v4_development_quality_records.v1"
KERNEL_SERIALIZATION_ID = "case141_rate_v4_dec015_anchor_certificate.v1"
SMOKE_SERIALIZATION_ID = "case141_rate_v4_case_smoke.v1"

PARTICIPANT_IDS = tuple(f"p{index:03d}" for index in range(1, 31))
PARTICIPANT_BUS_IDS = (
    8, 9, 12, 13, 17, 20, 21, 23, 26, 27, 29, 32, 34, 35, 36,
    37, 39, 41, 44, 48, 49, 51, 52, 53, 56, 58, 59, 61, 62, 64,
)
_Q95_SIGN = "NEGATIVE_BUS_INJECTION"
_Q95_MAGNITUDE_RULE = "FIXED_POWER_FACTOR_PARTICIPANT_REACTIVE_INJECTION"
_TOPOLOGY_CLASS_POLICY = "ROOT_FEEDER_SLACK_INCIDENT_TRUNK_DOWNSTREAM_PARTICIPANTS_GE_3_LATERAL_OTHER_TAP_TRANSFORMER_V1"
_FORBIDDEN_INPUT_TOKENS = (
    "legacy", "historical", "rate_v3", "m1_v3", "synthetic_rate_v3",
    "mitigation_result", "calibration_result", "guard_result", "evaluation_result",
    "trust_candidate_result", "source_certificate_result",
)
_READ_ASSERTION_FIELDS = (
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
_CROSSCHECK_FIELDS = (
    "voltage_pu", "angle_deg", "branch_p_mw", "branch_q_mvar",
    "bus_p_mismatch_mw", "bus_q_mismatch_mvar", "slack_p_mw", "slack_q_mvar",
    "active_loss_mw", "reactive_loss_mvar",
)

# DEC015 is frozen by the active scientific contract.  These are semantic
# constants of the certified two-stage hierarchy, not solver epsilon knobs.
DEC015_A_H_MW = 2.0e-7
# Compatibility spelling for the equation's symbolic a_H.  Runtime code uses
# the unit-bearing name above; this alias prevents a test/demonstration kernel
# from silently substituting an unrelated tolerance.
DEC015_A_H = DEC015_A_H_MW
DEC015_R_H = 2.0e-8
DEC015_ETA_H = 0.25
DEC015_KAPPA = 1.0
DEC015_SREF_FLOOR_MW = 1.0
DEC015_NUMERICAL_BOUND_GUARD_MULTIPLIER = 10.0
DEC015_CAPACITY_BOUND_GUARD_MULTIPLIER = 100.0
DEC015_STAGE2_MIN_QCP_TOLERANCE = 1.0e-6
DEC015_STAGE1_MIN_QCP_TOLERANCE = 1.0e-8
DEC015_PRIMARY_FACE_GUARD_MW = 2.0e-7


class RateV4DesignExecutionError(ValueError):
    """Raised when a fresh V4 RATE-design input is malformed or unsafe."""


class CertifiedDec015KernelUnavailable(RuntimeError):
    """Raised only when a certified DEC015 QCP backend cannot be started."""


class CertifiedDec015KernelFailure(RuntimeError):
    """Raised when a started DEC015 backend cannot certify the two stages."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RateV4DesignExecutionError(f"{label}_MAPPING_REQUIRED")
    return dict(value)


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RateV4DesignExecutionError(f"{label}_NONEMPTY_STRING_REQUIRED")
    return value


def _identifier(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if any(character.isspace() for character in result):
        raise RateV4DesignExecutionError(f"{label}_WHITESPACE_FORBIDDEN")
    return result


def _finite(value: Any, *, label: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RateV4DesignExecutionError(f"{label}_FINITE_NUMBER_REQUIRED")
    result = float(value)
    if not math.isfinite(result):
        raise RateV4DesignExecutionError(f"{label}_FINITE_NUMBER_REQUIRED")
    if positive and result <= 0.0:
        raise RateV4DesignExecutionError(f"{label}_POSITIVE_REQUIRED")
    if nonnegative and result < 0.0:
        raise RateV4DesignExecutionError(f"{label}_NONNEGATIVE_REQUIRED")
    return result


def _sha256(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdefABCDEF" for character in result):
        raise RateV4DesignExecutionError(f"{label}_SHA256_REQUIRED")
    return result.upper()


def _finalize(value: Mapping[str, Any], *, field: str = "result_hash") -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[field] = None
    result[field] = canonical_hash(result)
    return result


def _assert_exact_keys(value: Mapping[str, Any], expected: Sequence[str] | set[str], *, label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise RateV4DesignExecutionError(f"{label}_FIELDS_INVALID:missing={missing}:extra={extra}")


def _fresh_read_assertions() -> dict[str, bool]:
    return {field: False for field in _READ_ASSERTION_FIELDS}


def _validate_fresh_read_assertions(value: Any, *, label: str) -> dict[str, bool]:
    assertions = _mapping(value, label=f"{label}_INPUT_READ_ASSERTIONS")
    _assert_exact_keys(assertions, _READ_ASSERTION_FIELDS, label=f"{label}_INPUT_READ_ASSERTIONS")
    if any(assertions[field] is not False for field in _READ_ASSERTION_FIELDS):
        raise RateV4DesignExecutionError(f"{label}_DOWNSTREAM_OR_HISTORICAL_READ_FORBIDDEN")
    return _fresh_read_assertions()


def _contains_forbidden_token(value: Any) -> bool:
    if isinstance(value, Mapping):
        # Schema field names such as ``source_results_read`` are deliberate
        # negative assertions, not an attempt to ingest a source-result
        # artifact.  Only user-supplied *values* can establish a forbidden
        # provenance link.  Scanning mapping keys made every valid input fail
        # closed merely because its required read-assertion schema contained
        # the word "source".
        return any(_contains_forbidden_token(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_token(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in _FORBIDDEN_INPUT_TOKENS)
    return False


def _vector(value: Any, *, label: str, length: int, positive: bool = False, nonnegative: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RateV4DesignExecutionError(f"{label}_VECTOR_LENGTH_{length}_REQUIRED")
    return [_finite(item, label=f"{label}_{index}", positive=positive, nonnegative=nonnegative) for index, item in enumerate(value)]


def _self_hash(value: Mapping[str, Any], *, label: str, field: str = "result_hash") -> str:
    payload = dict(value)
    declared = _sha256(payload.get(field), label=f"{label}_{field.upper()}")
    payload[field] = None
    if canonical_hash(payload) != declared:
        raise RateV4DesignExecutionError(f"{label}_{field.upper()}_MISMATCH")
    return declared


def _input_hash(value: Mapping[str, Any]) -> str:
    return canonical_hash(value)


def _validate_crosscheck_profile(value: Any) -> dict[str, float]:
    profile = _mapping(value, label="RATE_V4_CROSSCHECK_TOLERANCES")
    _assert_exact_keys(profile, _CROSSCHECK_FIELDS, label="RATE_V4_CROSSCHECK_TOLERANCES")
    return {field: _finite(profile[field], label=f"RATE_V4_CROSSCHECK_{field}", nonnegative=True) for field in _CROSSCHECK_FIELDS}


def _validate_execution_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate raw, pre-registered RATE_DESIGN inputs without reading results.

    The full root roster contains identifiers only.  Numerical vectors are
    supplied for the deterministic RATE_DESIGN subset *after* that subset is
    computed; no data for source, guard, calibration, or evaluation roles can
    enter this input format.
    """

    payload = _mapping(value, label="RATE_V4_EXECUTION_INPUT")
    _assert_exact_keys(
        payload,
        {
            "serialization_id", "status", "root_profile_ids", "rate_design_root_count",
            "rate_design_profiles", "participant_mapping", "q95_specification",
            "anchor_execution", "input_read_assertions", "historical_output_reused",
        },
        label="RATE_V4_EXECUTION_INPUT",
    )
    if payload["serialization_id"] != INPUT_SERIALIZATION_ID:
        raise RateV4DesignExecutionError("RATE_V4_EXECUTION_INPUT_SERIALIZATION_INVALID")
    if payload["status"] != "FRESH_RATE_DESIGN_EXECUTION_INPUT":
        raise RateV4DesignExecutionError("RATE_V4_EXECUTION_INPUT_STATUS_INVALID")
    if payload["historical_output_reused"] is not False:
        raise RateV4DesignExecutionError("RATE_V4_EXECUTION_HISTORICAL_OUTPUT_FORBIDDEN")
    _validate_fresh_read_assertions(payload["input_read_assertions"], label="RATE_V4_EXECUTION_INPUT")
    if _contains_forbidden_token({key: value for key, value in payload.items() if key not in {"serialization_id", "status"}}):
        raise RateV4DesignExecutionError("RATE_V4_EXECUTION_FORBIDDEN_RESULT_REFERENCE")

    roots_raw = payload["root_profile_ids"]
    if not isinstance(roots_raw, list):
        raise RateV4DesignExecutionError("RATE_V4_ROOT_PROFILE_IDS_VECTOR_REQUIRED")
    roots = [_identifier(item, label="RATE_V4_ROOT_PROFILE_ID") for item in roots_raw]
    if len(roots) < len(ROOT_ROLES) or len(set(roots)) != len(roots):
        raise RateV4DesignExecutionError("RATE_V4_ROOT_PROFILE_ROSTER_INSUFFICIENT_OR_DUPLICATE")
    count = payload["rate_design_root_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count > len(roots) - (len(ROOT_ROLES) - 1):
        raise RateV4DesignExecutionError("RATE_V4_RATE_DESIGN_ROOT_COUNT_INVALID")

    mapping = _mapping(payload["participant_mapping"], label="RATE_V4_PARTICIPANT_MAPPING")
    _assert_exact_keys(mapping, {"participant_ids", "participant_bus_ids", "mapping_policy"}, label="RATE_V4_PARTICIPANT_MAPPING")
    if mapping["participant_ids"] != list(PARTICIPANT_IDS) or mapping["participant_bus_ids"] != list(PARTICIPANT_BUS_IDS):
        raise RateV4DesignExecutionError("RATE_V4_IMMUTABLE_PARTICIPANT_MAPPING_MISMATCH")
    if mapping["mapping_policy"] != "CASE141_FROZEN_TYPED_PARTICIPANT_MAPPING_V10":
        raise RateV4DesignExecutionError("RATE_V4_PARTICIPANT_MAPPING_POLICY_INVALID")

    q95 = _mapping(payload["q95_specification"], label="RATE_V4_Q95_SPECIFICATION")
    _assert_exact_keys(q95, {"power_factor", "q_sign_convention", "magnitude_rule"}, label="RATE_V4_Q95_SPECIFICATION")
    pf = _finite(q95["power_factor"], label="RATE_V4_Q95_POWER_FACTOR", positive=True)
    if pf > 1.0:
        raise RateV4DesignExecutionError("RATE_V4_Q95_POWER_FACTOR_RANGE_INVALID")
    if q95["q_sign_convention"] != _Q95_SIGN or q95["magnitude_rule"] != _Q95_MAGNITUDE_RULE:
        raise RateV4DesignExecutionError("RATE_V4_Q95_SEMANTICS_INVALID")

    execution = _mapping(payload["anchor_execution"], label="RATE_V4_ANCHOR_EXECUTION")
    _assert_exact_keys(
        execution,
        {
            "finite_difference_delta_mw", "anchor_branch_margin_mva", "anchor_voltage_margin_pu",
            "primary_ac_tolerance", "primary_ac_max_iterations", "independent_ac_tolerance",
            "independent_ac_max_iterations", "crosscheck_tolerances", "quality_mva_tolerance",
            "quality_voltage_tolerance_pu", "dec015_solver_feasibility_tolerance",
            "dec015_solver_optimality_tolerance", "dec015_solver_threads",
        },
        label="RATE_V4_ANCHOR_EXECUTION",
    )
    normalized_execution = {
        "finite_difference_delta_mw": _finite(execution["finite_difference_delta_mw"], label="RATE_V4_FD_DELTA", positive=True),
        "anchor_branch_margin_mva": _finite(execution["anchor_branch_margin_mva"], label="RATE_V4_ANCHOR_BRANCH_MARGIN", nonnegative=True),
        "anchor_voltage_margin_pu": _finite(execution["anchor_voltage_margin_pu"], label="RATE_V4_ANCHOR_VOLTAGE_MARGIN", nonnegative=True),
        "primary_ac_tolerance": _finite(execution["primary_ac_tolerance"], label="RATE_V4_PRIMARY_AC_TOL", positive=True),
        "independent_ac_tolerance": _finite(execution["independent_ac_tolerance"], label="RATE_V4_INDEPENDENT_AC_TOL", positive=True),
        "quality_mva_tolerance": _finite(execution["quality_mva_tolerance"], label="RATE_V4_QUALITY_MVA_TOL", nonnegative=True),
        "quality_voltage_tolerance_pu": _finite(execution["quality_voltage_tolerance_pu"], label="RATE_V4_QUALITY_VOLTAGE_TOL", nonnegative=True),
        "dec015_solver_feasibility_tolerance": _finite(execution["dec015_solver_feasibility_tolerance"], label="RATE_V4_DEC015_FEASIBILITY_TOL", positive=True),
        "dec015_solver_optimality_tolerance": _finite(execution["dec015_solver_optimality_tolerance"], label="RATE_V4_DEC015_OPTIMALITY_TOL", positive=True),
        "crosscheck_tolerances": _validate_crosscheck_profile(execution["crosscheck_tolerances"]),
    }
    for name in ("primary_ac_max_iterations", "independent_ac_max_iterations", "dec015_solver_threads"):
        raw = execution[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RateV4DesignExecutionError(f"RATE_V4_{name.upper()}_POSITIVE_INTEGER_REQUIRED")
        normalized_execution[name] = int(raw)

    profiles_raw = payload["rate_design_profiles"]
    if not isinstance(profiles_raw, list):
        raise RateV4DesignExecutionError("RATE_V4_RATE_DESIGN_PROFILES_VECTOR_REQUIRED")
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in profiles_raw:
        profile = _mapping(raw, label="RATE_V4_RATE_DESIGN_PROFILE")
        _assert_exact_keys(
            profile,
            {
                "root_id", "capacity_mw", "honest_request_mw", "strategic_capacity_mw",
                "strategic_unmitigated_allocation_mw",
                "strategic_allocation_provenance",
            },
            label="RATE_V4_RATE_DESIGN_PROFILE",
        )
        root_id = _identifier(profile["root_id"], label="RATE_V4_PROFILE_ROOT")
        if root_id not in roots or root_id in seen:
            raise RateV4DesignExecutionError("RATE_V4_RATE_DESIGN_PROFILE_ROOT_INVALID_OR_DUPLICATE")
        seen.add(root_id)
        capacity = _vector(profile["capacity_mw"], label="RATE_V4_CAPACITY_MW", length=30, positive=True)
        request = _vector(profile["honest_request_mw"], label="RATE_V4_HONEST_REQUEST_MW", length=30, nonnegative=True)
        strategic_capacity = _vector(
            profile["strategic_capacity_mw"], label="RATE_V4_STRATEGIC_CAPACITY_MW", length=30, positive=True,
        )
        strategic = _vector(profile["strategic_unmitigated_allocation_mw"], label="RATE_V4_STRATEGIC_ALLOCATION_MW", length=30, nonnegative=True)
        if any(value > limit for value, limit in zip(strategic, strategic_capacity)):
            raise RateV4DesignExecutionError("RATE_V4_STRATEGIC_ALLOCATION_EXCEEDS_REPORTED_CAPACITY_NO_CLIPPING")
        if not any(stress > base for stress, base in zip(strategic_capacity, capacity)):
            raise RateV4DesignExecutionError("RATE_V4_STRATEGIC_CAPACITY_MUST_DEFINE_INDEPENDENT_STRESS_ENVELOPE")
        if profile["strategic_allocation_provenance"] != "PRE_REGISTERED_UNMITIGATED_STRATEGIC_DEVELOPMENT_ALLOCATION":
            raise RateV4DesignExecutionError("RATE_V4_STRATEGIC_ALLOCATION_PROVENANCE_INVALID")
        profiles.append({
            "root_id": root_id,
            "capacity_mw": capacity,
            "honest_request_mw": request,
            "strategic_capacity_mw": strategic_capacity,
            "strategic_unmitigated_allocation_mw": strategic,
            "strategic_allocation_provenance": profile["strategic_allocation_provenance"],
        })
    return {
        "root_profile_ids": roots,
        "rate_design_root_count": int(count),
        "profiles": profiles,
        "participant_mapping": {
            "participant_ids": list(PARTICIPANT_IDS),
            "participant_bus_ids": list(PARTICIPANT_BUS_IDS),
            "mapping_policy": "CASE141_FROZEN_TYPED_PARTICIPANT_MAPPING_V10",
        },
        "q95_specification": {
            "power_factor": pf,
            "q_sign_convention": _Q95_SIGN,
            "magnitude_rule": _Q95_MAGNITUDE_RULE,
        },
        "anchor_execution": normalized_execution,
        "input_hash": _input_hash(payload),
    }


def build_deterministic_rate_design_root_roles(
    *, protocol: Mapping[str, Any], root_profile_ids: Sequence[str], rate_design_root_count: int,
) -> dict[str, Any]:
    """Assign every root to exactly one role before reading numerical profiles.

    Hash ranking prevents source-order effects.  The first pre-registered
    number of rank-ordered roots become ``RATE_DESIGN``; the remaining roots
    are allocated cyclically across every later role, with one root guaranteed
    for each role.  This is a root-level split, not a scenario-result split.
    """

    protocol_hash = validate_rate_design_protocol(protocol)
    roots = [_identifier(item, label="RATE_V4_ROOT_PROFILE_ID") for item in root_profile_ids]
    if len(roots) < len(ROOT_ROLES) or len(set(roots)) != len(roots):
        raise RateV4DesignExecutionError("RATE_V4_ROOT_ROSTER_INSUFFICIENT_OR_DUPLICATE")
    if isinstance(rate_design_root_count, bool) or not isinstance(rate_design_root_count, int):
        raise RateV4DesignExecutionError("RATE_V4_RATE_DESIGN_ROOT_COUNT_INTEGER_REQUIRED")
    later = ROOT_ROLES[1:]
    if rate_design_root_count <= 0 or rate_design_root_count > len(roots) - len(later):
        raise RateV4DesignExecutionError("RATE_V4_RATE_DESIGN_ROOT_COUNT_INVALID")
    ranked = sorted(
        roots,
        key=lambda root_id: (canonical_hash({"protocol_hash": protocol_hash, "root_id": root_id}), root_id),
    )
    role_ids: dict[str, list[str]] = {"RATE_DESIGN": sorted(ranked[:rate_design_root_count])}
    remaining = ranked[rate_design_root_count:]
    for offset, role in enumerate(later):
        role_ids[role] = [remaining[offset]]
    for offset, root_id in enumerate(remaining[len(later):]):
        role_ids[later[offset % len(later)]].append(root_id)
    for role in ROOT_ROLES:
        role_ids[role].sort()
    try:
        return build_rate_design_root_split(protocol=protocol, root_role_ids=role_ids)
    except RateV4GroundworkError as exc:  # translate module-local type
        raise RateV4DesignExecutionError(f"RATE_V4_ROOT_SPLIT_INVALID:{exc}") from exc


def _selection_for_profile(parsed: ParsedMatpowerCase, *, capacity_mw: Sequence[float]) -> ParticipantSelection:
    bus_ids = tuple(int(bus.bus_id) for bus in parsed.network.buses)
    slack = next(int(bus.bus_id) for bus, bus_type in zip(parsed.network.buses, parsed.bus_types) if bus_type == 3)
    load_by_bus = dict(zip(bus_ids, parsed.p_load_mw.values))
    selected = tuple(PARTICIPANT_BUS_IDS)
    capacity = [float(item) for item in capacity_mw]
    return build_participant_selection(
        selected,
        network_bus_ids=bus_ids,
        slack_bus_id=slack,
        load_mw_by_bus=load_by_bus,
        rating_mw_by_bus={bus_id: value for bus_id, value in zip(selected, capacity)},
        availability_mw_by_bus={bus_id: value for bus_id, value in zip(selected, capacity)},
        participant_count=30,
        scientific_status="AUTHOR_FROZEN",
        declared_complete="COMPLETE",
        ordering_rule="CASE141_FROZEN_TYPED_PARTICIPANT_MAPPING_V10",
        co_location_policy="ONE_PARTICIPANT_PER_BUS",
        slack_bus_policy="EXCLUDE_SLACK",
    )


def _participant_q(*, allocation_mw: Sequence[float], q_mode: str, q95_specification: Mapping[str, Any]) -> list[float]:
    if q_mode == "Q0":
        return [0.0] * len(allocation_mw)
    if q_mode != "Q95":
        raise RateV4DesignExecutionError("RATE_V4_Q_MODE_INVALID")
    pf = float(q95_specification["power_factor"])
    magnitude = math.tan(math.acos(pf))
    if q95_specification["q_sign_convention"] != _Q95_SIGN:
        raise RateV4DesignExecutionError("RATE_V4_Q95_SIGN_INVALID")
    return [-magnitude * float(value) for value in allocation_mw]


def _q_assumption(q_mode: str) -> QAssumption:
    if q_mode == "Q0":
        return QAssumption.Q0
    if q_mode == "Q95":
        return QAssumption.Q95
    raise RateV4DesignExecutionError("RATE_V4_Q_MODE_INVALID")


def _solution_summary(solution: ACPowerFlowSolution | IndependentACPowerFlowSolution) -> dict[str, Any]:
    return {
        "solution_hash": solution.solution_hash.to_json(),
        "network_input_hash": solution.network_input_hash.to_json(),
        "solve_success": bool(solution.solver_result.solve_success),
        "solver_status": solution.solver_result.normalized_status.value,
        "failure_reasons": solution.solver_result.failure_reasons.to_json(),
        "p_injection_hash": solution.p_injection_hash.to_json(),
        "q_injection_hash": solution.q_injection_hash.to_json(),
        "bus_voltage_pu": list(solution.bus_voltage_pu.values),
        "branch_p_from_mw": list(solution.branch_p_from_mw.values),
        "branch_q_from_mvar": list(solution.branch_q_from_mvar.values),
        "branch_p_to_mw": list(solution.branch_p_to_mw.values),
        "branch_q_to_mvar": list(solution.branch_q_to_mvar.values),
        "max_abs_bus_p_mismatch_mw": max((abs(value) for value in solution.bus_p_mismatch_mw.values), default=0.0),
        "max_abs_bus_q_mismatch_mvar": max((abs(value) for value in solution.bus_q_mismatch_mvar.values), default=0.0),
    }


def _run_ac_pair(
    *,
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    q_mode: str,
    q95_specification: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    q_assumption = _q_assumption(q_mode)
    operating_point = build_canonical_operating_point(parsed, q_assumption=q_assumption)
    participant_q = _participant_q(allocation_mw=allocation_mw, q_mode=q_mode, q95_specification=q95_specification)
    injection = build_net_injections_from_operating_point(
        parsed,
        selection,
        allocation_mw,
        operating_point,
        q_assumption=q_assumption,
        participant_q_mvar=participant_q,
    )
    primary = solve_ac_power_flow(
        parsed,
        p_injection_mw=injection.p_mw,
        q_injection_mvar=injection.q_mvar,
        tolerance=float(execution["primary_ac_tolerance"]),
        max_iterations=int(execution["primary_ac_max_iterations"]),
        solver_version="rate-v4-design-primary-v1",
    )
    independent = solve_independent_radial_ac(
        parsed,
        p_injection_mw=injection.p_mw,
        q_injection_mvar=injection.q_mvar,
        tolerance=float(execution["independent_ac_tolerance"]),
        max_iterations=int(execution["independent_ac_max_iterations"]),
        solver_version="rate-v4-design-independent-v1",
    )
    crosscheck, metrics = crosscheck_ac_solutions(
        primary,
        independent,
        tolerance_profile=execution["crosscheck_tolerances"],
    )
    flow_hash = canonical_hash({
        "p_from_mw": list(primary.branch_p_from_mw.values),
        "q_from_mvar": list(primary.branch_q_from_mvar.values),
        "p_to_mw": list(primary.branch_p_to_mw.values),
        "q_to_mvar": list(primary.branch_q_to_mvar.values),
    })
    q_spec_hash = canonical_hash({
        "q_mode": q_mode,
        "q95_specification": dict(q95_specification) if q_mode == "Q95" else None,
        "participant_q_mvar": participant_q,
    })
    result = {
        "q_mode": q_mode,
        "allocation_mw": [float(item) for item in allocation_mw],
        "allocation_hash": canonical_hash({"participant_ids": list(PARTICIPANT_IDS), "values_mw": [float(item) for item in allocation_mw]}),
        "participant_q_mvar": participant_q,
        "participant_q_hash": canonical_hash({"values_mvar": participant_q}),
        "q_specification_hash": q_spec_hash,
        "operating_point_hash": canonical_hash(operating_point.to_json()),
        "participant_registry_hash": selection.alignment_hash.upper(),
        "primary": _solution_summary(primary),
        "independent": _solution_summary(independent),
        "crosscheck_status": crosscheck.status.value,
        "crosscheck_metrics": metrics,
        "crosscheck_hash": canonical_hash({"result": crosscheck.to_json(), "metrics": metrics}),
        "two_ended_flow_hash": flow_hash,
        "primary_ac_solution_hash": primary.solution_hash.to_json(),
        "independent_ac_solution_hash": independent.solution_hash.to_json(),
        "ac_converged_both": bool(primary.solver_result.solve_success and independent.solver_result.solve_success),
    }
    result["result_hash"] = None
    result["result_hash"] = canonical_hash(result)
    return result


def _rooted_branch_topology(parsed: ParsedMatpowerCase) -> list[dict[str, str]]:
    """Classify a radial tree without assigning a physical conductor type.

    ``ROOT_FEEDER`` means incident to the slack.  A non-root edge is ``TRUNK``
    when its rooted downstream subtree contains at least three members of the
    frozen 30-participant roster; otherwise it is ``LATERAL``.  No transformer
    is asserted for this literal case unless a MATPOWER tap differs from 0/1.
    """

    bus_index = {int(bus.bus_id): index for index, bus in enumerate(parsed.network.buses)}
    slack_index = next(index for index, kind in enumerate(parsed.bus_types) if kind == 3)
    adjacency: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(parsed.network.buses))}
    for branch_index, branch in enumerate(parsed.network.branches):
        left = bus_index[int(branch.from_bus.object_id.value)]
        right = bus_index[int(branch.to_bus.object_id.value)]
        adjacency[left].append((right, branch_index))
        adjacency[right].append((left, branch_index))
    parent = {slack_index: -1}
    branch_for_child: dict[int, int] = {}
    order = [slack_index]
    for node in order:
        for neighbour, branch_index in sorted(adjacency[node], key=lambda item: item[0]):
            if neighbour in parent:
                continue
            parent[neighbour] = node
            branch_for_child[neighbour] = branch_index
            order.append(neighbour)
    if len(parent) != len(parsed.network.buses):
        raise RateV4DesignExecutionError("RATE_V4_ROOTED_TOPOLOGY_INCOMPLETE")
    children: dict[int, list[int]] = {node: [] for node in parent}
    for child, parent_node in parent.items():
        if parent_node >= 0:
            children[parent_node].append(child)
    tap_ratios = _matpower_branch_tap_ratios(parsed)
    participant_bus_set = set(PARTICIPANT_BUS_IDS)

    def downstream_participant_count(node: int) -> int:
        return int(int(parsed.network.buses[node].bus_id) in participant_bus_set) + sum(
            downstream_participant_count(child) for child in children[node]
        )

    classes: dict[int, str] = {}
    for child, branch_index in branch_for_child.items():
        if tap_ratios[branch_index] not in {0.0, 1.0}:
            classes[branch_index] = "TRANSFORMER"
        elif parent[child] == slack_index:
            classes[branch_index] = "ROOT_FEEDER"
        elif downstream_participant_count(child) >= 3:
            classes[branch_index] = "TRUNK"
        else:
            classes[branch_index] = "LATERAL"
    rows: list[dict[str, str]] = []
    for branch_index, branch in enumerate(parsed.network.branches):
        branch_class = classes.get(branch_index)
        if branch_class not in BRANCH_CLASSES:
            raise RateV4DesignExecutionError("RATE_V4_BRANCH_CLASSIFICATION_INVALID")
        rows.append({
            "branch_id": f"branch_{branch_index + 1:03d}",
            "from_bus_id": str(branch.from_bus.object_id.value),
            "to_bus_id": str(branch.to_bus.object_id.value),
            "branch_class": branch_class,
        })
    return rows


def _matpower_branch_tap_ratios(parsed: ParsedMatpowerCase) -> list[float]:
    """Read only literal MATPOWER TAP fields for topology-class provenance.

    The static parser intentionally does not use a tap in its present IEEE-141
    electrical model.  RATE scenario classification still records the raw TAP
    value so a future source with a non-0/non-1 transformer cannot be silently
    treated as a line.  This helper is a literal, read-only audit of column 9
    of the already-pinned branch matrix; it never executes the case source.
    """

    source = Path(parsed.source_path).read_text(encoding="utf-8")
    match = re.search(r"mpc\.branch\s*=\s*\[(?P<body>.*?)\];", source, re.S)
    if match is None:
        raise RateV4DesignExecutionError("RATE_V4_RAW_BRANCH_MATRIX_MISSING")
    taps: list[float] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("%", 1)[0].strip().rstrip(";").strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) != 13:
            raise RateV4DesignExecutionError("RATE_V4_RAW_BRANCH_ROW_SHAPE_INVALID")
        try:
            tap = float(tokens[8].replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise RateV4DesignExecutionError("RATE_V4_RAW_BRANCH_TAP_INVALID") from exc
        if not math.isfinite(tap):
            raise RateV4DesignExecutionError("RATE_V4_RAW_BRANCH_TAP_INVALID")
        taps.append(tap)
    if len(taps) != len(parsed.network.branches):
        raise RateV4DesignExecutionError("RATE_V4_RAW_BRANCH_TAP_COVERAGE_INVALID")
    return taps


def _topology_source_rows(parsed: ParsedMatpowerCase, topology_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    taps = _matpower_branch_tap_ratios(parsed)
    if len(taps) != len(topology_rows):
        raise RateV4DesignExecutionError("RATE_V4_TOPOLOGY_SOURCE_COVERAGE_INVALID")
    return [
        {
            "branch_id": row["branch_id"],
            "from_bus_id": row["from_bus_id"],
            "to_bus_id": row["to_bus_id"],
            "matpower_tap_ratio": tap,
            "branch_class": row["branch_class"],
        }
        for row, tap in zip(topology_rows, taps)
    ]


def _flow_rows(
    *,
    root_id: str,
    q_mode: str,
    ac: Mapping[str, Any],
    role: str,
    candidate_id: str | None = None,
    anchor_domain_hash: str | None = None,
    anchor_parameter_hash: str | None = None,
) -> list[dict[str, Any]]:
    primary = _mapping(ac["primary"], label="RATE_V4_AC_PRIMARY")
    if primary["solve_success"] is not True:
        raise RateV4DesignExecutionError("RATE_V4_PRIMARY_AC_NOT_CONVERGED_FOR_FLOW_ROWS")
    p_from = primary["branch_p_from_mw"]
    q_from = primary["branch_q_from_mvar"]
    p_to = primary["branch_p_to_mw"]
    q_to = primary["branch_q_to_mvar"]
    if not all(isinstance(values, list) and len(values) == len(p_from) for values in (q_from, p_to, q_to)):
        raise RateV4DesignExecutionError("RATE_V4_PRIMARY_FLOW_SHAPE_INVALID")
    result: list[dict[str, Any]] = []
    for index, values in enumerate(zip(p_from, q_from, p_to, q_to), 1):
        pf, qf, pt, qt = [float(item) for item in values]
        row: dict[str, Any] = {
            "root_id": root_id,
            "q_mode": q_mode,
            "branch_id": f"branch_{index:03d}",
            "p_from_mw": pf,
            "q_from_mvar": qf,
            "p_to_mw": pt,
            "q_to_mvar": qt,
            "flow_role": "BASELINE" if role == "BASELINE" else "HONEST_REFERENCE_ANCHOR",
            "source_kind": "FRESH_RATE_DESIGN_AC_BASELINE" if role == "BASELINE" else "FRESH_RATE_ANCHOR_AC_REFERENCE",
            "historical_output_reused": False,
            "input_read_assertions": _fresh_read_assertions(),
        }
        if role == "REFERENCE":
            row.update({
                "rate_candidate_id": candidate_id,
                "anchor_domain_hash": anchor_domain_hash,
                "anchor_parameter_hash": anchor_parameter_hash,
            })
        result.append(row)
    return result


def _run_fd_perturbation_worker(task: tuple[int, ParsedMatpowerCase, ParticipantSelection, str, Mapping[str, Any], Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Process-pool worker for one fresh finite-difference AC ray."""

    participant_index, parsed, selection, q_mode, q95_specification, execution = task
    allocation = [0.0] * 30
    allocation[participant_index] = float(execution["finite_difference_delta_mw"])
    perturbation = _run_ac_pair(
        parsed=parsed, selection=selection, allocation_mw=allocation, q_mode=q_mode,
        q95_specification=q95_specification, execution=execution,
    )
    if perturbation["ac_converged_both"] is not True:
        raise RateV4DesignExecutionError("RATE_V4_FD_AC_CROSS_SOLVER_CONVERGENCE_REQUIRED")
    return participant_index, perturbation


def _finite_difference_proxy(
    *,
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    q_mode: str,
    q95_specification: Mapping[str, Any],
    execution: Mapping[str, Any],
    baseline_ac: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fresh local affine design proxy around one zero allocation.

    This response is retained only to solve the rate-independent anchor
    domain.  It is not the future calibrated M1 proxy and carries an explicit
    design-only status so it cannot be substituted for it.
    """

    base_primary = _mapping(baseline_ac["primary"], label="RATE_V4_BASELINE_PRIMARY")
    if baseline_ac["ac_converged_both"] is not True:
        raise RateV4DesignExecutionError("RATE_V4_BASELINE_AC_CROSS_SOLVER_CONVERGENCE_REQUIRED")
    branch_count = len(base_primary["branch_p_from_mw"])
    bus_count = len(base_primary["bus_voltage_pu"])
    delta = float(execution["finite_difference_delta_mw"])
    fields = {
        "branch_p_from_mw": [[0.0] * 30 for _ in range(branch_count)],
        "branch_q_from_mvar": [[0.0] * 30 for _ in range(branch_count)],
        "branch_p_to_mw": [[0.0] * 30 for _ in range(branch_count)],
        "branch_q_to_mvar": [[0.0] * 30 for _ in range(branch_count)],
        "voltage_pu": [[0.0] * 30 for _ in range(bus_count)],
    }
    def run_perturbation(participant_index: int) -> tuple[int, dict[str, Any]]:
        allocation = [0.0] * 30
        allocation[participant_index] = delta
        perturbation = _run_ac_pair(
            parsed=parsed,
            selection=selection,
            allocation_mw=allocation,
            q_mode=q_mode,
            q95_specification=q95_specification,
            execution=execution,
        )
        if perturbation["ac_converged_both"] is not True:
            raise RateV4DesignExecutionError("RATE_V4_FD_AC_CROSS_SOLVER_CONVERGENCE_REQUIRED")
        return participant_index, perturbation

    # Each finite-difference ray is a fresh, read-only AC solve.  The CLI uses
    # processes because the Newton/BFS implementations are Python-heavy and
    # do not release the GIL.  Results are committed in participant order, so
    # canonical hashes remain deterministic.  A direct in-process caller may
    # set RATE_V4_FD_SERIAL=1 for environments that cannot spawn workers.
    import os
    if os.environ.get("RATE_V4_FD_SERIAL") == "1":
        perturbations = [run_perturbation(index) for index in range(30)]
    else:
        worker_count = min(8, 30)
        tasks = [
            (index, parsed, selection, q_mode, q95_specification, execution)
            for index in range(30)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            perturbations = list(executor.map(_run_fd_perturbation_worker, tasks))
    perturbation_hashes: list[str] = []
    for participant_index, perturbation in perturbations:
        primary = _mapping(perturbation["primary"], label="RATE_V4_FD_PRIMARY")
        for field in (
            "branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar", "voltage_pu",
        ):
            # ``voltage_pu`` is the proxy's canonical response coordinate;
            # AC summaries deliberately expose it as ``bus_voltage_pu`` to
            # distinguish it from a scalar constraint slack.  Translate only
            # at this boundary so both serialized contracts remain stable.
            ac_field = "bus_voltage_pu" if field == "voltage_pu" else field
            base_values = base_primary[ac_field]
            observed = primary[ac_field]
            if len(base_values) != len(observed) or len(fields[field]) != len(observed):
                raise RateV4DesignExecutionError("RATE_V4_FD_PROXY_SHAPE_INVALID")
            for row_index, (base, actual) in enumerate(zip(base_values, observed)):
                fields[field][row_index][participant_index] = (float(actual) - float(base)) / delta
        perturbation_hashes.append(perturbation["result_hash"])
    proxy = {
        "serialization_id": PROXY_SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_LOCAL_AC_FINITE_DIFFERENCE_PROXY_ONLY",
        "q_mode": q_mode,
        "finite_difference_delta_mw": delta,
        "baseline_ac_hash": baseline_ac["result_hash"],
        "primary_baseline_solution_hash": baseline_ac["primary_ac_solution_hash"],
        "independent_baseline_solution_hash": baseline_ac["independent_ac_solution_hash"],
        "perturbation_execution_hashes": perturbation_hashes,
        "proxy_base": {
            "branch_p_from_mw": [float(item) for item in base_primary["branch_p_from_mw"]],
            "branch_q_from_mvar": [float(item) for item in base_primary["branch_q_from_mvar"]],
            "branch_p_to_mw": [float(item) for item in base_primary["branch_p_to_mw"]],
            "branch_q_to_mvar": [float(item) for item in base_primary["branch_q_to_mvar"]],
            "voltage_pu": [float(item) for item in base_primary["bus_voltage_pu"]],
        },
        "derivative_matrices": fields,
        "physical_evidence_eligible": False,
        "calibration_proxy": False,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(proxy)


def _anchor_limits(
    *,
    baseline_anchor_mva: Sequence[float],
    topology_rows: Sequence[Mapping[str, Any]],
    candidate_record: Mapping[str, Any],
) -> tuple[list[float], str]:
    parameters = _mapping(candidate_record["class_parameters"], label="RATE_V4_CANDIDATE_PARAMETERS")
    if len(baseline_anchor_mva) != len(topology_rows):
        raise RateV4DesignExecutionError("RATE_V4_BASELINE_ANCHOR_TOPOLOGY_SHAPE_INVALID")
    limits: list[float] = []
    rows: list[dict[str, Any]] = []
    for baseline, topology in zip(baseline_anchor_mva, topology_rows):
        branch_class = topology["branch_class"]
        class_parameters = _mapping(parameters[branch_class], label="RATE_V4_CLASS_PARAMETERS")
        multiplier = float(class_parameters["anchor_multiplier"])
        minimum = float(class_parameters["anchor_minimum_mva"])
        limit = max(minimum, multiplier * float(baseline))
        limits.append(limit)
        rows.append({
            "branch_id": topology["branch_id"], "branch_class": branch_class,
            "baseline_anchor_mva": float(baseline), "anchor_multiplier": multiplier,
            "anchor_minimum_mva": minimum, "anchor_limit_mva": limit,
        })
    return limits, canonical_hash(rows)


def _proxy_residuals(
    *,
    proxy: Mapping[str, Any],
    allocation_mw: Sequence[float],
    capacity_mw: Sequence[float],
    anchor_limits_mva: Sequence[float],
    parsed: ParsedMatpowerCase,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every anchor-domain predicate, never repairing a solution."""

    base = _mapping(proxy["proxy_base"], label="RATE_V4_PROXY_BASE")
    matrices = _mapping(proxy["derivative_matrices"], label="RATE_V4_PROXY_MATRICES")
    x = [float(item) for item in allocation_mw]
    capacity = [float(item) for item in capacity_mw]
    if len(x) != 30 or len(capacity) != 30:
        raise RateV4DesignExecutionError("RATE_V4_PROXY_ALLOCATION_SHAPE_INVALID")
    rho = math.sqrt(math.fsum((value / limit) ** 2 for value, limit in zip(x, capacity)))
    branch_margin = float(execution["anchor_branch_margin_mva"])
    voltage_margin = float(execution["anchor_voltage_margin_pu"])
    branch_excess: list[float] = []
    for branch_index, limit in enumerate(anchor_limits_mva):
        p_from = float(base["branch_p_from_mw"][branch_index]) + math.fsum(float(coef) * value for coef, value in zip(matrices["branch_p_from_mw"][branch_index], x))
        q_from = float(base["branch_q_from_mvar"][branch_index]) + math.fsum(float(coef) * value for coef, value in zip(matrices["branch_q_from_mvar"][branch_index], x))
        p_to = float(base["branch_p_to_mw"][branch_index]) + math.fsum(float(coef) * value for coef, value in zip(matrices["branch_p_to_mw"][branch_index], x))
        q_to = float(base["branch_q_to_mvar"][branch_index]) + math.fsum(float(coef) * value for coef, value in zip(matrices["branch_q_to_mvar"][branch_index], x))
        branch_excess.extend([
            math.hypot(p_from, q_from) - (float(limit) - branch_margin),
            math.hypot(p_to, q_to) - (float(limit) - branch_margin),
        ])
    voltage_excess: list[float] = []
    for index, base_voltage in enumerate(base["voltage_pu"]):
        voltage = float(base_voltage) + math.fsum(float(coef) * value for coef, value in zip(matrices["voltage_pu"][index], x))
        lower, upper = _anchor_voltage_bounds(
            vmin=float(parsed.bus_vmin_pu.values[index]), vmax=float(parsed.bus_vmax_pu.values[index]),
            margin=voltage_margin,
        )
        voltage_excess.extend([lower - voltage, voltage - upper])
    box_excess = [
        -min(x, default=0.0),
        max((value - limit for value, limit in zip(x, capacity)), default=0.0),
    ]
    return {
        "rho": rho,
        "max_branch_excess_mva": max(branch_excess, default=-math.inf),
        "max_voltage_excess_pu": max(voltage_excess, default=-math.inf),
        "max_capacity_excess_mw": max(box_excess, default=-math.inf),
        "branch_excess_mva": branch_excess,
        "voltage_excess_pu": voltage_excess,
    }


def _anchor_voltage_bounds(*, vmin: float, vmax: float, margin: float) -> tuple[float, float]:
    """Return the registered anchor bounds without invalidating a fixed slack.

    MATPOWER represents the IEEE-141 slack setpoint as ``Vmin == Vmax ==
    1.0``.  That is an equality constraint, not an interval available for an
    inward voltage uncertainty margin.  Interval buses receive the declared
    inward margin; a non-fixed interval too narrow for that margin is a hard
    configuration error rather than a silently widened bound.
    """

    if not all(math.isfinite(value) for value in (vmin, vmax, margin)) or margin < 0.0 or vmin > vmax:
        raise CertifiedDec015KernelFailure("DEC015_ANCHOR_VOLTAGE_BOUND_INVALID")
    span = vmax - vmin
    if span <= 1.0e-12:
        return vmin, vmax
    if span < 2.0 * margin:
        raise CertifiedDec015KernelFailure("DEC015_ANCHOR_VOLTAGE_MARGIN_INVALID")
    return vmin + margin, vmax - margin


def _dec015_model_upper_bounds(*, raw_upper_mw: Sequence[float], feasibility_tolerance: float) -> tuple[list[float], float]:
    """Make the solver's numerical feasible set strictly internal to x<=min(C,rH).

    A continuous QCP solver is permitted to return a value a few feasibility
    tolerances above a variable's literal upper bound.  The injection adapter
    (correctly) rejects even that microscopic capacity exceedance.  Rather
    than silently clipping a returned allocation, declare an inward numerical
    guard before solving and attest both the raw and model upper vectors.
    """

    tolerance = _finite(feasibility_tolerance, label="DEC015_BOUND_GUARD_FEASIBILITY_TOLERANCE", positive=True)
    guard = DEC015_CAPACITY_BOUND_GUARD_MULTIPLIER * tolerance
    model_upper = [float(value) - guard for value in raw_upper_mw]
    if any(value <= 0.0 for value in model_upper):
        raise CertifiedDec015KernelFailure("DEC015_NUMERICAL_BOUND_GUARD_ELIMINATES_DOMAIN")
    return model_upper, guard


def _continuous_qcp_operational_upper(*, primal_total_mw: float, bar_qcp_convergence_tolerance: float) -> tuple[float, float]:
    """Return a conservative finite upper envelope for a continuous convex QCP.

    Gurobi's ``ObjBound`` is a MIP-oriented attribute and can be ``+inf`` for
    a continuous QCP even after ``GRB_OPTIMAL``.  For that model class Gurobi
    instead certifies optimality subject to ``BarQCPConvTol`` (relative primal
    and dual objective, feasibility, and complementarity).  This deterministic
    two-sided relative envelope is therefore the attested operational upper;
    it is deliberately conservative and never sourced from a MIP bound.
    """

    primal = _finite(primal_total_mw, label="DEC015_STAGE1_PRIMAL_TOTAL", nonnegative=True)
    tolerance = _finite(
        bar_qcp_convergence_tolerance, label="DEC015_BAR_QCP_CONVERGENCE_TOLERANCE", positive=True,
    )
    envelope = tolerance * max(1.0, abs(primal))
    return primal + envelope, envelope


def _dec015_branch_numerical_guard(*, physical_margin_mva: float, feasibility_tolerance: float) -> float:
    """Return a pre-solve inward cone guard while retaining the physical screen."""

    margin = _finite(physical_margin_mva, label="DEC015_BRANCH_PHYSICAL_MARGIN", nonnegative=True)
    tolerance = _finite(feasibility_tolerance, label="DEC015_BRANCH_FEASIBILITY_TOLERANCE", positive=True)
    return max(margin, DEC015_NUMERICAL_BOUND_GUARD_MULTIPLIER * tolerance)


def _solve_dec015_two_stage_gurobi(
    *,
    proxy: Mapping[str, Any],
    capacity_mw: Sequence[float],
    honest_request_mw: Sequence[float],
    anchor_limits_mva: Sequence[float],
    anchor_limit_hash: str,
    parsed: ParsedMatpowerCase,
    root_id: str,
    q_mode: str,
    candidate_id: str,
    candidate_parameter_hash: str,
    anchor_domain_hash: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Run DEC015 on one candidate-specific, rate-independent anchor domain.

    Gurobi solves the convex QCP directly.  The result is accepted only for a
    globally optimal continuous QCP status, a certified stage-one bound gap,
    and an explicit replay of every proxy predicate.  No tiny numerical
    violation is clipped or repaired.
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - environment dependent
        raise CertifiedDec015KernelUnavailable(f"GUROBI_QCP_IMPORT_UNAVAILABLE:{type(exc).__name__}") from exc
    try:
        capacity = [float(value) for value in capacity_mw]
        raw = [float(value) for value in honest_request_mw]
        if len(capacity) != 30 or len(raw) != 30 or any(value <= 0.0 for value in capacity) or any(value < 0.0 for value in raw):
            raise CertifiedDec015KernelFailure("DEC015_ANCHOR_VECTOR_DOMAIN_INVALID")
        raw_upper = [min(capacity_value, raw_value) for capacity_value, raw_value in zip(capacity, raw)]
        if any(value <= 0.0 for value in raw_upper):
            raise CertifiedDec015KernelFailure("DEC015_ANCHOR_RAW_UPPER_DOMAIN_INVALID")
        feasibility_tolerance = float(execution["dec015_solver_feasibility_tolerance"])
        upper, numerical_bound_guard = _dec015_model_upper_bounds(
            raw_upper_mw=raw_upper, feasibility_tolerance=feasibility_tolerance,
        )
        numerical_branch_guard = _dec015_branch_numerical_guard(
            physical_margin_mva=float(execution["anchor_branch_margin_mva"]),
            feasibility_tolerance=feasibility_tolerance,
        )
        primary_face_guard = max(DEC015_PRIMARY_FACE_GUARD_MW, DEC015_NUMERICAL_BOUND_GUARD_MULTIPLIER * feasibility_tolerance)
        stage1_qcp_tolerance = max(
            float(execution["dec015_solver_optimality_tolerance"]), DEC015_STAGE1_MIN_QCP_TOLERANCE,
        )
        total_scale = math.fsum(raw_upper)
        tau_h = DEC015_A_H_MW + DEC015_R_H * total_scale
        tau_stage1 = DEC015_ETA_H * tau_h
        s_ref = max(total_scale, DEC015_SREF_FLOOR_MW)
        proxy_base = _mapping(proxy["proxy_base"], label="RATE_V4_DEC015_PROXY_BASE")
        matrices = _mapping(proxy["derivative_matrices"], label="RATE_V4_DEC015_PROXY_MATRICES")
        if len(anchor_limits_mva) != len(proxy_base["branch_p_from_mw"]):
            raise CertifiedDec015KernelFailure("DEC015_ANCHOR_LIMIT_SHAPE_INVALID")

        def build_model(*, stage: str, primary_floor: float | None = None) -> tuple[Any, list[Any], Any]:
            model = gp.Model("rate_v4_dec015_anchor")
            model.Params.OutputFlag = 0
            model.Params.Threads = int(execution["dec015_solver_threads"])
            model.Params.FeasibilityTol = float(execution["dec015_solver_feasibility_tolerance"])
            stage_qcp_tolerance = stage1_qcp_tolerance
            if stage == "stage2":
                stage_qcp_tolerance = max(stage_qcp_tolerance, DEC015_STAGE2_MIN_QCP_TOLERANCE)
            model.Params.OptimalityTol = stage_qcp_tolerance
            model.Params.BarConvTol = stage_qcp_tolerance
            # Gurobi uses a distinct barrier convergence criterion for convex
            # QCPs.  Leaving it at its much looser default invalidates the
            # DEC015 certified stage-one interval even when OptimalityTol is
            # tight, so bind it to the frozen QCP optimality tolerance too.
            model.Params.BarQCPConvTol = stage_qcp_tolerance
            # The branch-response coefficients and MW-scaled tie objective
            # span several orders of magnitude.  These are numerical
            # conditioning controls only; the scientific tolerances and all
            # physical predicates remain unchanged.
            model.Params.NumericFocus = 3
            model.Params.ScaleFlag = 2
            model.Params.BarHomogeneous = 1
            # Stage 1 needs QCP duals for the explicit max-export certificate.
            # Stage 2's strictly convex tie objective can make Gurobi's
            # optional QCP-dual backsolve numerically singular even when the
            # primal convex problem is solved to OPTIMAL.  The stage-2 gate
            # therefore relies on convex OPTIMAL status plus independent
            # objective/constraint replay and records that dual backsolve was
            # not requested.
            model.Params.QCPDual = 1 if stage == "stage1" else 0
            x = [model.addVar(lb=0.0, ub=limit, name=f"x_{index + 1:03d}") for index, limit in enumerate(upper)]
            rho = model.addVar(lb=0.0, name="rho")
            model.addQConstr(gp.quicksum((x[index] / capacity[index]) * (x[index] / capacity[index]) for index in range(30)) <= rho * rho, name="rho_soc")
            branch_margin = float(execution["anchor_branch_margin_mva"])
            for branch_index, rate in enumerate(anchor_limits_mva):
                from_p = float(proxy_base["branch_p_from_mw"][branch_index]) + gp.quicksum(float(coef) * variable for coef, variable in zip(matrices["branch_p_from_mw"][branch_index], x))
                from_q = float(proxy_base["branch_q_from_mvar"][branch_index]) + gp.quicksum(float(coef) * variable for coef, variable in zip(matrices["branch_q_from_mvar"][branch_index], x))
                to_p = float(proxy_base["branch_p_to_mw"][branch_index]) + gp.quicksum(float(coef) * variable for coef, variable in zip(matrices["branch_p_to_mw"][branch_index], x))
                to_q = float(proxy_base["branch_q_to_mvar"][branch_index]) + gp.quicksum(float(coef) * variable for coef, variable in zip(matrices["branch_q_to_mvar"][branch_index], x))
                allowable = float(rate) - branch_margin - numerical_branch_guard
                if allowable <= 0.0:
                    raise CertifiedDec015KernelFailure("DEC015_ANCHOR_LIMIT_NONPOSITIVE_AFTER_MARGIN")
                model.addQConstr(from_p * from_p + from_q * from_q <= allowable * allowable, name=f"branch_from_{branch_index + 1:03d}")
                model.addQConstr(to_p * to_p + to_q * to_q <= allowable * allowable, name=f"branch_to_{branch_index + 1:03d}")
            voltage_margin = float(execution["anchor_voltage_margin_pu"])
            for bus_index, base_voltage in enumerate(proxy_base["voltage_pu"]):
                voltage = float(base_voltage) + gp.quicksum(float(coef) * variable for coef, variable in zip(matrices["voltage_pu"][bus_index], x))
                lower, upper_voltage = _anchor_voltage_bounds(
                    vmin=float(parsed.bus_vmin_pu.values[bus_index]), vmax=float(parsed.bus_vmax_pu.values[bus_index]),
                    margin=voltage_margin,
                )
                model.addConstr(voltage >= lower, name=f"voltage_lower_{bus_index + 1:03d}")
                model.addConstr(voltage <= upper_voltage, name=f"voltage_upper_{bus_index + 1:03d}")
            if primary_floor is not None:
                model.addConstr(
                    gp.quicksum(x) >= float(primary_floor) + primary_face_guard,
                    name="dec015_primary_face_with_numerical_guard",
                )
            if stage == "stage1":
                model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
            elif stage == "stage2":
                # Exact DEC015 canonical tie objective.  This is not a
                # semantic epsilon: the primary face was imposed separately.
                objective = gp.quicksum((index + 1) * x[index] for index in range(30))
                objective += 0.5 * DEC015_KAPPA * s_ref * (
                    gp.quicksum((x[index] / s_ref) * (x[index] / s_ref) for index in range(30)) + rho * rho
                )
                model.setObjective(objective, GRB.MINIMIZE)
            else:  # pragma: no cover - internal misuse
                raise CertifiedDec015KernelFailure("DEC015_STAGE_UNKNOWN")
            return model, x, rho

        stage1, stage1_x, stage1_rho = build_model(stage="stage1")
        stage1.optimize()
        if stage1.Status != GRB.OPTIMAL:
            raise CertifiedDec015KernelFailure(f"DEC015_STAGE1_NOT_OPTIMAL:{stage1.Status}")
        p1 = float(stage1.ObjVal)
        u1, solver_operational_envelope = _continuous_qcp_operational_upper(
            primal_total_mw=p1,
            bar_qcp_convergence_tolerance=stage1_qcp_tolerance,
        )
        if u1 < p1:
            # Reported bounds may differ at a last bit; the semantic ordering
            # cannot be silently corrected, so this is a hard failure.
            raise CertifiedDec015KernelFailure("DEC015_STAGE1_BOUND_ORDER_INVALID")
        gap = u1 - p1
        if gap > tau_stage1:
            raise CertifiedDec015KernelFailure(
                "DEC015_STAGE1_CERTIFIED_GAP_EXCEEDS_TAU_H_QUARTER:"
                f"primal={p1:.17g}:upper={u1:.17g}:gap={gap:.17g}:limit={tau_stage1:.17g}"
            )
        stage1_values = [float(variable.X) for variable in stage1_x]
        stage1_rho_value = float(stage1_rho.X)
        stage1_replay = _proxy_residuals(
            proxy=proxy, allocation_mw=stage1_values, capacity_mw=capacity,
            anchor_limits_mva=anchor_limits_mva, parsed=parsed, execution=execution,
        )
        tolerance = feasibility_tolerance
        if max(stage1_replay["max_branch_excess_mva"], stage1_replay["max_voltage_excess_pu"], stage1_replay["max_capacity_excess_mw"]) > tolerance:
            raise CertifiedDec015KernelFailure(
                "DEC015_STAGE1_PROXY_REPLAY_INFEASIBLE:"
                f"branch={stage1_replay['max_branch_excess_mva']:.17g}:"
                f"voltage={stage1_replay['max_voltage_excess_pu']:.17g}:"
                f"capacity={stage1_replay['max_capacity_excess_mw']:.17g}:tol={tolerance:.17g}"
            )
        primary_floor = u1 - tau_h
        stage2, stage2_x, stage2_rho = build_model(stage="stage2", primary_floor=primary_floor)
        stage2.optimize()
        if stage2.Status != GRB.OPTIMAL:
            raise CertifiedDec015KernelFailure(f"DEC015_STAGE2_NOT_OPTIMAL:{stage2.Status}")
        allocation = [float(variable.X) for variable in stage2_x]
        rho_value = float(stage2_rho.X)
        replay = _proxy_residuals(
            proxy=proxy, allocation_mw=allocation, capacity_mw=capacity,
            anchor_limits_mva=anchor_limits_mva, parsed=parsed, execution=execution,
        )
        if max(replay["max_branch_excess_mva"], replay["max_voltage_excess_pu"], replay["max_capacity_excess_mw"]) > tolerance:
            raise CertifiedDec015KernelFailure(
                "DEC015_STAGE2_PROXY_REPLAY_INFEASIBLE:"
                f"branch={replay['max_branch_excess_mva']:.17g}:"
                f"voltage={replay['max_voltage_excess_pu']:.17g}:"
                f"capacity={replay['max_capacity_excess_mw']:.17g}:tol={tolerance:.17g}"
            )
        total = math.fsum(allocation)
        if total + tolerance < primary_floor:
            raise CertifiedDec015KernelFailure(
                "DEC015_STAGE2_PRIMARY_FACE_VIOLATION:"
                f"total={total:.17g}:floor={primary_floor:.17g}:"
                f"deficit={primary_floor-total:.17g}:tol={tolerance:.17g}"
            )
        allocation_hash = canonical_hash({"participant_ids": list(PARTICIPANT_IDS), "values_mw": allocation})
        certificate = {
            "serialization_id": KERNEL_SERIALIZATION_ID,
            "status": "CERTIFIED_DEC015_TWO_STAGE_ANCHOR_QCP_DEVELOPMENT_ONLY",
            "kernel_id": "GUROBI_CONVEX_QCP_DEC015_V1",
            "root_id": root_id,
            "q_mode": q_mode,
            "rate_candidate_id": candidate_id,
            "candidate_parameter_hash": candidate_parameter_hash,
            "anchor_domain_hash": anchor_domain_hash,
            "anchor_proxy_hash": proxy["result_hash"],
            "anchor_limit_hash": anchor_limit_hash,
            "capacity_hash": canonical_hash({"values_mw": capacity}),
            "honest_request_hash": canonical_hash({"values_mw": raw}),
            "allocation_upper_hash": canonical_hash({"values_mw": upper}),
            "allocation_upper_raw_hash": canonical_hash({"values_mw": raw_upper}),
            "numerical_bound_guard_mw": numerical_bound_guard,
            "numerical_branch_guard_mva": numerical_branch_guard,
            "primary_face_guard_mw": primary_face_guard,
            "tau_h_mw": tau_h,
            "stage1": {
                "p1_primal_total_mw": p1,
                "u1_certified_upper_mw": u1,
                "u1_minus_p1_mw": gap,
                "tau1_stage1_gap_max_mw": tau_stage1,
                "upper_bound_method": "CONTINUOUS_CONVEX_QCP_OPTIMAL_STATUS_BARQCP_CONV_TOLERANCE_ENVELOPE",
                "solver_operational_envelope_mw": solver_operational_envelope,
                "allocation_mw": stage1_values,
                "rho": stage1_rho_value,
                "solver_status": int(stage1.Status),
                "proxy_replay": stage1_replay,
            },
            "stage2": {
                "primary_floor_u1_minus_tau_h_mw": primary_floor,
                "allocation_mw": allocation,
                "allocation_hash": allocation_hash,
                "total_export_mw": total,
                "rho": rho_value,
                "canonical_objective_value": float(stage2.ObjVal),
                "solver_status": int(stage2.Status),
                "proxy_replay": replay,
            },
            "tie_objective": {
                "formula": "sum_i_i_x_i_plus_kappa_Sref_over_2_times_sum_square_x_over_Sref_plus_rho_square",
                "kappa": DEC015_KAPPA,
                "s_ref_mw": s_ref,
                "s_ref_floor_mw": DEC015_SREF_FLOOR_MW,
                "epsilon_used": False,
            },
            "stage_solver_controls": {
                "stage1_qcp_dual_requested": True,
                "stage2_qcp_dual_requested": False,
                "stage1_bar_qcp_convergence_tolerance": stage1_qcp_tolerance,
                "stage2_bar_qcp_convergence_tolerance": DEC015_STAGE2_MIN_QCP_TOLERANCE,
                "numeric_focus": 3,
                "scale_flag": 2,
                "bar_homogeneous": 1,
            },
            "input_read_assertions": _fresh_read_assertions(),
            "historical_output_reused": False,
            "development_only": True,
            "result_hash": None,
        }
        return _finalize(certificate)
    except CertifiedDec015KernelFailure:
        raise
    except Exception as exc:  # license and backend failures are safely blocked by caller
        # Preserve the backend diagnostic in the blocked record.  Collapsing
        # every GurobiError to its class name made a genuine model-contract
        # defect indistinguishable from a missing licence and prevented a
        # reproducible repair.  This does not change the fail-closed status.
        detail = str(exc).replace("\n", " ").strip() or "NO_BACKEND_MESSAGE"
        raise CertifiedDec015KernelUnavailable(
            f"GUROBI_QCP_UNAVAILABLE_OR_UNLICENSED:{type(exc).__name__}:{detail}"
        ) from exc


def _build_baseline_anchor_mva(baseline_rows: Sequence[Mapping[str, Any]], *, branch_count: int) -> list[float]:
    values = [0.0] * branch_count
    for row in baseline_rows:
        index = int(str(row["branch_id"]).split("_")[-1]) - 1
        values[index] = max(
            values[index],
            math.hypot(float(row["p_from_mw"]), float(row["q_from_mvar"])),
            math.hypot(float(row["p_to_mw"]), float(row["q_to_mvar"])),
        )
    return values


def _screen_ac_against_policy(
    *, ac: Mapping[str, Any], policy: Mapping[str, Any], parsed: ParsedMatpowerCase, execution: Mapping[str, Any],
) -> dict[str, Any]:
    primary = _mapping(ac["primary"], label="RATE_V4_SCREEN_PRIMARY")
    ratings = _mapping(policy, label="RATE_V4_POLICY")["branch_ratings"]
    if len(ratings) != len(primary["branch_p_from_mw"]):
        raise RateV4DesignExecutionError("RATE_V4_SCREEN_POLICY_BRANCH_SHAPE_INVALID")
    branch_excess: list[float] = []
    for index, rating in enumerate(ratings):
        rate = float(rating["rate_a_mva"])
        s_from = math.hypot(float(primary["branch_p_from_mw"][index]), float(primary["branch_q_from_mvar"][index]))
        s_to = math.hypot(float(primary["branch_p_to_mw"][index]), float(primary["branch_q_to_mvar"][index]))
        branch_excess.append(max(s_from, s_to) - rate)
    voltage_excess: list[float] = []
    for index, voltage in enumerate(primary["bus_voltage_pu"]):
        voltage_excess.extend([
            float(parsed.bus_vmin_pu.values[index]) - float(voltage),
            float(voltage) - float(parsed.bus_vmax_pu.values[index]),
        ])
    mva_tolerance = float(execution["quality_mva_tolerance"])
    voltage_tolerance = float(execution["quality_voltage_tolerance_pu"])
    pass_flag = bool(
        ac["ac_converged_both"] is True
        and ac["crosscheck_status"] == "PASS"
        and max(branch_excess, default=-math.inf) <= mva_tolerance
        and max(voltage_excess, default=-math.inf) <= voltage_tolerance
    )
    return {
        "pass": pass_flag,
        "max_two_ended_mva_excess": max(branch_excess, default=0.0),
        "max_voltage_excess_pu": max(voltage_excess, default=0.0),
        "ac_converged_both": ac["ac_converged_both"],
        "crosscheck_status": ac["crosscheck_status"],
        "primary_ac_solution_hash": ac["primary_ac_solution_hash"],
        "independent_ac_solution_hash": ac["independent_ac_solution_hash"],
        "two_ended_flow_hash": ac["two_ended_flow_hash"],
    }


def _blocked_execution(
    *,
    status: str,
    protocol_hash: str,
    anchor_domain_hash: str,
    raw_case_hash: str,
    network_hash: str | None,
    root_split: Mapping[str, Any],
    candidate_library: Mapping[str, Any],
    input_hash: str,
    baseline_flows: Mapping[str, Any] | None,
    detail: str,
) -> dict[str, Any]:
    payload = {
        "serialization_id": SERIALIZATION_ID,
        "status": status,
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "network_hash": network_hash,
        "execution_input_hash": input_hash,
        "root_split": dict(root_split),
        "candidate_library": dict(candidate_library),
        "baseline_flows": dict(baseline_flows) if baseline_flows is not None else None,
        "reference_flows": None,
        "anchor_proxy_responses": None,
        "anchor_certificates": None,
        "anchor_observation": None,
        "candidate_policy_bundle": None,
        "development_quality_records": None,
        "detail": detail,
        "development_only": True,
        "primary_evidence_eligible": False,
        "manuscript_evidence_eligible": False,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(payload)


def run_rate_v4_design_ac_execution(
    *,
    case_path: str | Path,
    protocol: Mapping[str, Any],
    anchor_domain: Mapping[str, Any],
    execution_input: Mapping[str, Any],
    expected_case_sha256: str = IEEE141_CASE_SHA256,
) -> dict[str, Any]:
    """Run a fresh, fully bounded V4 RATE-design execution.

    A missing or unlicensed certified QCP kernel produces a self-hashed blocked
    result after fresh baseline work.  It never falls back to a heuristic
    allocation or a raw capacity box.
    """

    protocol_hash = validate_rate_design_protocol(protocol)
    anchor_domain_hash = validate_rate_anchor_domain(anchor_domain, protocol)
    inputs = _validate_execution_input(execution_input)
    parsed = parse_matpower_case(case_path, expected_sha256=expected_case_sha256)
    raw_case_hash = parsed.source_hash.to_json()
    root_split = build_deterministic_rate_design_root_roles(
        protocol=protocol,
        root_profile_ids=inputs["root_profile_ids"],
        rate_design_root_count=inputs["rate_design_root_count"],
    )
    candidate_library = build_predeclared_rate_candidate_library(protocol=protocol)
    rate_roots = set(root_split["root_role_ids"]["RATE_DESIGN"])
    profile_by_root = {profile["root_id"]: profile for profile in inputs["profiles"]}
    if set(profile_by_root) != rate_roots:
        raise RateV4DesignExecutionError("RATE_V4_PROFILE_COVERAGE_MUST_EQUAL_DETERMINISTIC_RATE_DESIGN_SUBSET")
    topology_rows = _rooted_branch_topology(parsed)
    topology_source_rows = _topology_source_rows(parsed, topology_rows)
    execution = inputs["anchor_execution"]
    baseline_rows: list[dict[str, Any]] = []
    baseline_records: dict[tuple[str, str], dict[str, Any]] = {}
    proxies: list[dict[str, Any]] = []
    network_hash: str | None = None
    for root_id in sorted(rate_roots):
        profile = profile_by_root[root_id]
        selection = _selection_for_profile(parsed, capacity_mw=profile["capacity_mw"])
        for q_mode in Q_MODES:
            baseline = _run_ac_pair(
                parsed=parsed,
                selection=selection,
                allocation_mw=[0.0] * 30,
                q_mode=q_mode,
                q95_specification=inputs["q95_specification"],
                execution=execution,
            )
            baseline_records[(root_id, q_mode)] = baseline
            # The network input hash, rather than a solution hash, binds the
            # candidate RATE policy.  It must agree across every fresh solve.
            current_network_hash = _sha256(
                baseline["primary"]["network_input_hash"], label="RATE_V4_AC_NETWORK_HASH",
            )
            if network_hash is None:
                network_hash = current_network_hash
            elif network_hash != current_network_hash:
                raise RateV4DesignExecutionError("RATE_V4_AC_NETWORK_HASH_INCONSISTENT")
            baseline_rows.extend(_flow_rows(root_id=root_id, q_mode=q_mode, ac=baseline, role="BASELINE"))
            proxy = _finite_difference_proxy(
                parsed=parsed,
                selection=selection,
                q_mode=q_mode,
                q95_specification=inputs["q95_specification"],
                execution=execution,
                baseline_ac=baseline,
            )
            proxy["root_id"] = root_id
            proxy["participant_registry_hash"] = selection.alignment_hash.upper()
            proxy["raw_case_hash"] = raw_case_hash
            proxy["network_hash"] = current_network_hash
            # Adding execution identity changes the self-hash; re-finalize it.
            proxy["result_hash"] = None
            proxy["result_hash"] = canonical_hash(proxy)
            proxies.append(proxy)
    if network_hash is None:  # defensive: a valid split always has roots
        raise RateV4DesignExecutionError("RATE_V4_BASELINE_NETWORK_HASH_MISSING")
    baseline_payload = _finalize({
        "serialization_id": BASELINE_SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_BASELINE_TWO_ENDED_AC_ROWS",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "root_split_hash": root_split["result_hash"],
        "rows": baseline_rows,
        "execution_records": [
            {"root_id": root, "q_mode": q_mode, "ac_execution_hash": record["result_hash"],
             "primary_ac_solution_hash": record["primary_ac_solution_hash"],
             "independent_ac_solution_hash": record["independent_ac_solution_hash"],
             "two_ended_flow_hash": record["two_ended_flow_hash"]}
            for (root, q_mode), record in sorted(baseline_records.items())
        ],
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    })
    baseline_anchor = _build_baseline_anchor_mva(baseline_rows, branch_count=len(parsed.network.branches))

    reference_rows: list[dict[str, Any]] = []
    reference_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    certificates: list[dict[str, Any]] = []
    proxy_by_root_q = {(item["root_id"], item["q_mode"]): item for item in proxies}
    try:
        for candidate in candidate_library["candidate_records"]:
            candidate_id = candidate["candidate_id"]
            limits, limit_hash = _anchor_limits(
                baseline_anchor_mva=baseline_anchor,
                topology_rows=topology_rows,
                candidate_record=candidate,
            )
            for root_id in sorted(rate_roots):
                profile = profile_by_root[root_id]
                selection = _selection_for_profile(parsed, capacity_mw=profile["capacity_mw"])
                for q_mode in Q_MODES:
                    proxy = proxy_by_root_q[(root_id, q_mode)]
                    try:
                        certificate = _solve_dec015_two_stage_gurobi(
                            proxy=proxy,
                            capacity_mw=profile["capacity_mw"],
                            honest_request_mw=profile["honest_request_mw"],
                            anchor_limits_mva=limits,
                            anchor_limit_hash=limit_hash,
                            parsed=parsed,
                            root_id=root_id,
                            q_mode=q_mode,
                            candidate_id=candidate_id,
                            candidate_parameter_hash=candidate["candidate_parameter_hash"],
                            anchor_domain_hash=anchor_domain_hash,
                            execution=execution,
                        )
                    except CertifiedDec015KernelUnavailable as exc:
                        raise CertifiedDec015KernelUnavailable(
                            f"candidate={candidate_id}:root={root_id}:q={q_mode}:{exc}"
                        ) from exc
                    except CertifiedDec015KernelFailure as exc:
                        raise CertifiedDec015KernelFailure(
                            f"candidate={candidate_id}:root={root_id}:q={q_mode}:{exc}"
                        ) from exc
                    allocation = certificate["stage2"]["allocation_mw"]
                    reference = _run_ac_pair(
                        parsed=parsed,
                        selection=selection,
                        allocation_mw=allocation,
                        q_mode=q_mode,
                        q95_specification=inputs["q95_specification"],
                        execution=execution,
                    )
                    if reference["ac_converged_both"] is not True:
                        raise CertifiedDec015KernelFailure("RATE_V4_REFERENCE_AC_CROSS_SOLVER_CONVERGENCE_REQUIRED")
                    certificates.append(certificate)
                    reference_records[(candidate_id, root_id, q_mode)] = reference
                    reference_rows.extend(_flow_rows(
                        root_id=root_id, q_mode=q_mode, ac=reference, role="REFERENCE",
                        candidate_id=candidate_id, anchor_domain_hash=anchor_domain_hash,
                        anchor_parameter_hash=candidate["candidate_parameter_hash"],
                    ))
    except CertifiedDec015KernelUnavailable as exc:
        return _blocked_execution(
            status="BLOCKED_CERTIFIED_DEC015_KERNEL_UNAVAILABLE",
            protocol_hash=protocol_hash, anchor_domain_hash=anchor_domain_hash,
            raw_case_hash=raw_case_hash, network_hash=network_hash,
            root_split=root_split, candidate_library=candidate_library,
            input_hash=inputs["input_hash"], baseline_flows=baseline_payload, detail=str(exc),
        )
    except CertifiedDec015KernelFailure as exc:
        return _blocked_execution(
            status="BLOCKED_CERTIFIED_DEC015_ANCHOR_SOLVE_FAILED",
            protocol_hash=protocol_hash, anchor_domain_hash=anchor_domain_hash,
            raw_case_hash=raw_case_hash, network_hash=network_hash,
            root_split=root_split, candidate_library=candidate_library,
            input_hash=inputs["input_hash"], baseline_flows=baseline_payload, detail=str(exc),
        )

    reference_payload = _finalize({
        "serialization_id": REFERENCE_SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_DEC015_ANCHOR_REFERENCE_TWO_ENDED_AC_ROWS",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "root_split_hash": root_split["result_hash"],
        "rows": reference_rows,
        "anchor_certificate_hashes": [item["result_hash"] for item in certificates],
        "execution_records": [
            {"rate_candidate_id": candidate, "root_id": root, "q_mode": q_mode,
             "ac_execution_hash": record["result_hash"], "primary_ac_solution_hash": record["primary_ac_solution_hash"],
             "independent_ac_solution_hash": record["independent_ac_solution_hash"], "two_ended_flow_hash": record["two_ended_flow_hash"]}
            for (candidate, root, q_mode), record in sorted(reference_records.items())
        ],
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    })
    anchor_observation = build_two_ended_rate_anchor_observation(
        protocol=protocol, anchor_domain=anchor_domain, root_split=root_split,
        candidate_library=candidate_library, baseline_flow_rows=baseline_rows,
        reference_flow_rows=reference_rows,
    )
    policy_bundle = build_rate_v4_candidate_policy_bundle(
        protocol=protocol, anchor_domain=anchor_domain, root_split=root_split,
        candidate_library=candidate_library, anchor_observation=anchor_observation,
        topology_rows=topology_rows, network_hash=network_hash, raw_case_hash=raw_case_hash,
    )
    strategic_records: dict[tuple[str, str], dict[str, Any]] = {}
    for root_id in sorted(rate_roots):
        profile = profile_by_root[root_id]
        # The stress vector is an explicitly reported-capacity diagnostic.  Its
        # larger envelope is not fed into DEC015 or the anchor; it is used only
        # to evaluate whether the frozen candidate creates a non-trivial AC
        # stress case before any downstream evidence role is assigned.
        selection = _selection_for_profile(parsed, capacity_mw=profile["strategic_capacity_mw"])
        for q_mode in Q_MODES:
            strategic_records[(root_id, q_mode)] = _run_ac_pair(
                parsed=parsed, selection=selection,
                allocation_mw=profile["strategic_unmitigated_allocation_mw"], q_mode=q_mode,
                q95_specification=inputs["q95_specification"], execution=execution,
            )
    quality_rows: list[dict[str, Any]] = []
    quality_detail: list[dict[str, Any]] = []
    for policy in policy_bundle["candidate_policies"]:
        candidate_id = policy["candidate_id"]
        baseline_screens = [
            _screen_ac_against_policy(ac=baseline_records[(root_id, q_mode)], policy=policy, parsed=parsed, execution=execution)
            for root_id in sorted(rate_roots) for q_mode in Q_MODES
        ]
        reference_screens = {
            (root_id, q_mode): _screen_ac_against_policy(
                ac=reference_records[(candidate_id, root_id, q_mode)], policy=policy, parsed=parsed, execution=execution,
            )
            for root_id in sorted(rate_roots) for q_mode in Q_MODES
        }
        strategic_screens = {
            (root_id, q_mode): _screen_ac_against_policy(
                ac=strategic_records[(root_id, q_mode)], policy=policy, parsed=parsed, execution=execution,
            )
            for root_id in sorted(rate_roots) for q_mode in Q_MODES
        }
        reference_pass_rate = sum(screen["pass"] for screen in reference_screens.values()) / len(reference_screens)
        strategic_pair_pass_rate = sum(
            reference_screens[key]["pass"] and strategic_screens[key]["pass"]
            for key in reference_screens
        ) / len(reference_screens)
        quality_rows.append({
            "candidate_id": candidate_id,
            "policy_hash": policy["policy_hash"],
            "rate_design_split_hash": root_split["result_hash"],
            "baseline_ac_all_pass": all(screen["pass"] for screen in baseline_screens),
            "honest_reference_pass_rate": reference_pass_rate,
            "strategic_unmitigated_pair_pass_rate": strategic_pair_pass_rate,
            "input_read_assertions": _fresh_read_assertions(),
            "historical_output_reused": False,
        })
        quality_detail.append({
            "candidate_id": candidate_id,
            "baseline_screens": baseline_screens,
            "honest_reference_screens": [
                {"root_id": root, "q_mode": q_mode, **screen}
                for (root, q_mode), screen in sorted(reference_screens.items())
            ],
            "strategic_unmitigated_screens": [
                {"root_id": root, "q_mode": q_mode, **screen}
                for (root, q_mode), screen in sorted(strategic_screens.items())
            ],
        })
    quality_payload = _finalize({
        "serialization_id": QUALITY_SERIALIZATION_ID,
        "status": "FRESH_RATE_V4_DEVELOPMENT_QUALITY_ONLY_NOT_SELECTION_OR_FREEZE",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "root_split_hash": root_split["result_hash"],
        "candidate_policy_bundle_hash": policy_bundle["result_hash"],
        "rows": quality_rows,
        "quality_detail": quality_detail,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    })
    proxies_payload = _finalize({
        "serialization_id": PROXY_SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_LOCAL_AC_FINITE_DIFFERENCE_PROXY_COLLECTION_ONLY",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "root_split_hash": root_split["result_hash"],
        "topology_class_policy": _TOPOLOGY_CLASS_POLICY,
        "responses": proxies,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    })
    result = {
        "serialization_id": SERIALIZATION_ID,
        "status": "FRESH_RATE_V4_DESIGN_AC_EXECUTION_COMPLETE_DEVELOPMENT_ONLY",
        "protocol_hash": protocol_hash,
        "anchor_domain_hash": anchor_domain_hash,
        "raw_case_hash": raw_case_hash,
        "network_hash": network_hash,
        "execution_input_hash": inputs["input_hash"],
        "participant_mapping_hash": canonical_hash(inputs["participant_mapping"]),
        "root_split": root_split,
        "candidate_library": candidate_library,
        "topology_rows": topology_rows,
        "topology_source_rows": topology_source_rows,
        "topology_source_rows_hash": canonical_hash(topology_source_rows),
        "topology_class_policy": _TOPOLOGY_CLASS_POLICY,
        "baseline_flows": baseline_payload,
        "reference_flows": reference_payload,
        "anchor_proxy_responses": proxies_payload,
        "anchor_certificates": _finalize({
            "serialization_id": KERNEL_SERIALIZATION_ID,
            "status": "FRESH_DEC015_ANCHOR_CERTIFICATE_COLLECTION_DEVELOPMENT_ONLY",
            "certificate_hashes": [item["result_hash"] for item in certificates],
            "certificates": certificates,
            "input_read_assertions": _fresh_read_assertions(),
            "historical_output_reused": False,
            "result_hash": None,
        }),
        "anchor_observation": anchor_observation,
        "candidate_policy_bundle": policy_bundle,
        "development_quality_records": quality_payload,
        "selection_or_freeze_run": False,
        "development_only": True,
        "primary_evidence_eligible": False,
        "manuscript_evidence_eligible": False,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    }
    return _finalize(result)


def run_case141_rate_v4_smoke(
    *, case_path: str | Path, expected_case_sha256: str = IEEE141_CASE_SHA256,
) -> dict[str, Any]:
    """Exercise fresh Q0/Q95 zero-allocation AC without a DEC015 allocation.

    This is deliberately a source/AC smoke test, not a RATE build.  It is the
    safe diagnostic path when the certified anchor-kernel input is absent.
    """

    parsed = parse_matpower_case(case_path, expected_sha256=expected_case_sha256)
    # A zero-allocation smoke does not need a participant selection.  Direct
    # canonical operating-point injections make it impossible to accidentally
    # inherit an old allocation or participant-Q result.
    rows: list[dict[str, Any]] = []
    for q_mode in Q_MODES:
        operating_point = build_canonical_operating_point(parsed, q_assumption=_q_assumption(q_mode))
        p_injection = FloatVector(
            generation - load for generation, load in zip(parsed.p_generation_mw.values, operating_point.p_load.values)
        )
        q_injection = FloatVector(
            generation - load for generation, load in zip(parsed.q_generation_mvar.values, operating_point.q_load.values)
        )
        primary = solve_ac_power_flow(
            parsed, p_injection_mw=p_injection, q_injection_mvar=q_injection,
            solver_version="rate-v4-design-smoke-primary-v1",
        )
        independent = solve_independent_radial_ac(
            parsed, p_injection_mw=p_injection, q_injection_mvar=q_injection,
            solver_version="rate-v4-design-smoke-independent-v1",
        )
        rows.append({
            "q_mode": q_mode,
            "primary_ac_solution_hash": primary.solution_hash.to_json(),
            "independent_ac_solution_hash": independent.solution_hash.to_json(),
            "primary_converged": bool(primary.solver_result.solve_success),
            "independent_converged": bool(independent.solver_result.solve_success),
            "signed_two_ended_flow_hash": canonical_hash({
                "p_from_mw": list(primary.branch_p_from_mw.values), "q_from_mvar": list(primary.branch_q_from_mvar.values),
                "p_to_mw": list(primary.branch_p_to_mw.values), "q_to_mvar": list(primary.branch_q_to_mvar.values),
            }),
        })
    return _finalize({
        "serialization_id": SMOKE_SERIALIZATION_ID,
        "status": "CASE141_FRESH_Q0_Q95_AC_SMOKE_ONLY_DEC015_NOT_INVOKED",
        "raw_case_hash": parsed.source_hash.to_json(),
        "network_bus_count": len(parsed.network.buses),
        "branch_count": len(parsed.network.branches),
        "rows": rows,
        "certified_dec015_reference_generated": False,
        "development_quality_generated": False,
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "result_hash": None,
    })


def _load_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8-sig")), label=str(path))


def _load_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label=str(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI is a thin adapter
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute fresh RATE_DESIGN AC and DEC015 anchors")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--anchor-domain", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--expected-case-sha256", default=IEEE141_CASE_SHA256)
    smoke = subparsers.add_parser("smoke", help="run fresh Q0/Q95 zero-allocation case smoke only")
    smoke.add_argument("--case", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--expected-case-sha256", default=IEEE141_CASE_SHA256)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = run_case141_rate_v4_smoke(case_path=args.case, expected_case_sha256=args.expected_case_sha256)
    else:
        result = run_rate_v4_design_ac_execution(
            case_path=args.case, protocol=_load_yaml(args.protocol), anchor_domain=_load_yaml(args.anchor_domain),
            execution_input=_load_json(args.input), expected_case_sha256=args.expected_case_sha256,
        )
    _write_json(args.output, result)
    print(canonical_dumps({"status": result["status"], "result_hash": result["result_hash"]}))
    return 0 if not str(result["status"]).startswith("BLOCKED_") else 2


__all__ = [
    "BASELINE_SERIALIZATION_ID", "CertifiedDec015KernelFailure", "CertifiedDec015KernelUnavailable",
    "INPUT_SERIALIZATION_ID", "KERNEL_SERIALIZATION_ID", "PARTICIPANT_BUS_IDS", "PARTICIPANT_IDS",
    "PROXY_SERIALIZATION_ID", "QUALITY_SERIALIZATION_ID", "REFERENCE_SERIALIZATION_ID",
    "RateV4DesignExecutionError", "SERIALIZATION_ID", "SMOKE_SERIALIZATION_ID",
    "build_deterministic_rate_design_root_roles", "main", "run_case141_rate_v4_smoke",
    "run_rate_v4_design_ac_execution",
]
