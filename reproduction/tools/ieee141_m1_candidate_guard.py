"""Fail-closed admission guards for the EQ067 allocation boundary.

The legacy common-M1 artifact is a local finite-difference/Q99 diagnostic.  It
is retained only as provenance and must never enter an active allocation
domain.  This module is deliberately independent of solver/project imports so
that every active entry point can perform the same validation before loading a
permissive compatibility reader.

Two public guards are provided:

``validate_diagnostic_unified_eq067_candidate``
    Admits only the current unfrozen, diagnostic candidate.

``validate_frozen_unified_eq067_candidate``
    Reserved for the evidence path.  It additionally requires the frozen
    candidate/trust statuses and therefore cannot be used accidentally by the
    diagnostic runner.

``validate_unified_eq067_candidate`` remains an alias for the diagnostic guard
for backwards compatibility with the current Round-0/diagnostic runners.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from r4r.serialization import canonical_hash
from tools.ieee141_m1_joint_trust import (
    FEATURE_NAMES,
    JOINT_DIRECTION_DISTANCE_CAP,
    JOINT_JACCARD_DISTANCE_CAP,
    RHO_BANDS,
    SIGNATURE_FAMILIES,
)


UNIFIED_EQ067_SERIALIZATION_ID = "ieee141_m1_v2_state_dependent_margin_candidate.v5_unified_eq067"
LEGACY_MARGIN_USAGE_POLICY = (
    "PROVENANCE_ONLY; REJECTED_AS_ACTIVE_EQ067_INPUT; "
    "NO_MAX_FLOOR; NO_GLOBAL_Q99_EXTRAPOLATION"
)

# Only the both-side, full-development audit is sufficient for a new EQ067
# candidate.  Earlier V4/V5 artifacts are kept replayable as history, but
# cannot be rebound to an active candidate.
ALLOWED_LORO_SERIALIZATION_IDS = frozenset(
    {"ieee141_m1_v2_balanced_panel_loro_audit.v6_full_development_constraint_specific_both_sides"}
)

ALLOWED_TRUST_SOURCE_POLICIES = frozenset(
    {
        "PREREGISTERED_GREEDY_FIRST_FIT_PROXY_ONLY_NO_CURRENT_EXECUTION",
        "PREREGISTERED_IMMUTABLE_CAPACITY_REQUEST_TEMPLATE_CROSS_PRODUCT_NO_REPORTER_IDENTITY",
        "PREREGISTERED_DEV_REPORTERS_MEANINGFUL_ELECTRICAL_RAY_FAMILIES",
        "PREREGISTERED_DEV_REPORTER_PROXY_ELECTRICAL_EFFECT_FAMILY_BASIS_V1",
        "PREREGISTERED_DEV_REPORTER_TOPOLOGY_REQUEST_PROXY_ELECTRICAL_EFFECT_FAMILY_BASIS_V1",
    }
)

EXPECTED_DEVELOPMENT_SCENARIO_IDS = frozenset(
    {
        "ieee141-reporter-scan-01",
        "ieee141-reporter-scan-08",
        "ieee141-reporter-scan-15",
    }
)

LEGACY_TRUST_FEATURE_NAMES = (
    "total_export_normalized",
    "branch_loading_ratio",
    "voltage_lower_slack_pu",
    "voltage_upper_slack_pu",
)

LEGACY_MARGIN_ALIASES = frozenset(
    {
        "branch_from_mva",
        "branch_to_mva",
        "voltage_lower_pu",
        "voltage_upper_pu",
        "base_margin_candidate_hash",
        "old_margin",
        "common_m1",
        "legacy_unbound_diagnostic",
    }
)

EXPECTED_Q_MODES = frozenset({"Q0", "Q95"})
EXPECTED_VECTOR_LENGTHS = {
    "branch_from_mva": 140,
    "branch_to_mva": 140,
    "voltage_lower_pu": 141,
    "voltage_upper_pu": 141,
}
EXPECTED_FORMULA = {
    "norm": "rho(x;C)=sqrt(sum_i((x_i/C_i)^2))",
    "center_rule": "zero_allocation_origin; no center subtraction",
    # The serialized candidate stores post-safety effective coefficients.  The
    # fit coefficients are retained separately and are checked against the
    # registered multiplier; the runtime must never apply the multiplier a
    # second time.
    "branch_mva": "m_branch(x)=effective_a_branch+effective_L_branch*rho(x;C)",
    "voltage_lower_pu": "m_voltage_lower(x)=effective_a_lower+effective_L_lower*rho(x;C)",
    "voltage_upper_pu": "m_voltage_upper(x)=effective_a_upper+effective_L_upper*rho(x;C); inactive families use explicit applicability guard",
    "coefficient_layers": "fit_* are pre-safety; effective_* = safety_multiplier * fit_*; runtime consumes effective_* only",
    "intercept_rho_cutoff": 1.0e-3,
    "loro_min_rho_separation": 0.20,
    "numerical_intercept_guard": 1.0e-6,
    "safety_multiplier": 1.25,
    "upper_voltage_activation_threshold_pu": 1.0e-6,
    "no_epsilon": True,
    "no_clipping": True,
}
EXPECTED_MARGIN_KEYS = {
    "branch_from_mva_fit_intercept",
    "branch_from_mva_fit_slope",
    "branch_from_mva_effective_intercept",
    "branch_from_mva_effective_slope",
    "branch_to_mva_fit_intercept",
    "branch_to_mva_fit_slope",
    "branch_to_mva_effective_intercept",
    "branch_to_mva_effective_slope",
    "voltage_lower_pu_fit_intercept",
    "voltage_lower_pu_fit_slope",
    "voltage_lower_pu_effective_intercept",
    "voltage_lower_pu_effective_slope",
    "voltage_upper_pu_fit_intercept",
    "voltage_upper_pu_fit_slope",
    "voltage_upper_pu_effective_intercept",
    "voltage_upper_pu_effective_slope",
    "branch_from_coverage_count",
    "branch_to_coverage_count",
    "voltage_lower_coverage_count",
    "voltage_upper_coverage_count",
    "upper_voltage_status",
    "upper_voltage_guard_pu",
    "upper_voltage_activation_threshold_pu",
}
TRUST_ACTIVE_FAMILIES = frozenset(
    {"branch_from_mva", "branch_to_mva", "voltage_lower_pu", "voltage_upper_pu"}
)

FEEDBACK_GUARD_FLAGS = frozenset(
    {
        "evaluation_results_read",
        "holdout_results_read",
        "mitigation_results_read",
        "failure_ids_read",
        "ac_truth_read",
        "ac_pass_fail_read",
        "rate_policy_mutated",
        "realized_audit_results_read",
        "calibration_read",
        "evaluation_feedback_used",
        "holdout_feedback_used",
        "realized_audit_feedback_used",
        "margin_refit_performed",
        "source_feedback_used",
    }
)


def _fail(code: str, *, legacy: bool = False) -> None:
    suffix = "; LEGACY_MARGIN_NOT_ACCEPTED_BY_EQ067_RUNNER" if legacy else ""
    raise ValueError(f"{code}{suffix}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_feedback_flags(value: Any, *, path: str = "candidate") -> None:
    """Reject downstream feedback flags anywhere in the candidate envelope."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FEEDBACK_GUARD_FLAGS and child is not False:
                _fail(f"FEEDBACK_FLAG_NOT_FALSE:{child_path}")
            _validate_feedback_flags(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_feedback_flags(child, path=f"{path}[{index}]")


def _validate_affine_vectors(q_payload: Mapping[str, Any]) -> None:
    """Validate typed 140/141 ``(intercept, slope)`` vectors."""
    if any(key in LEGACY_MARGIN_ALIASES for key in q_payload):
        _fail("LEGACY_MARGIN_ALIAS_IN_Q_PAYLOAD", legacy=True)
    missing = EXPECTED_MARGIN_KEYS.difference(q_payload)
    if missing:
        _fail("INVALID_EQ067_VECTOR_SHAPE", legacy=True)
    for family, length in EXPECTED_VECTOR_LENGTHS.items():
        vectors: dict[str, list[Any]] = {}
        for layer in ("fit", "effective"):
            intercept_key = f"{family}_{layer}_intercept"
            slope_key = f"{family}_{layer}_slope"
            intercept = q_payload.get(intercept_key)
            slope = q_payload.get(slope_key)
            if not isinstance(intercept, list) or not isinstance(slope, list):
                _fail("INVALID_EQ067_VECTOR_SHAPE", legacy=True)
            if len(intercept) != length or len(slope) != length:
                _fail("INVALID_EQ067_VECTOR_SHAPE", legacy=True)
            vectors[f"{layer}_intercept"] = intercept
            vectors[f"{layer}_slope"] = slope
        allow_inactive = family == "voltage_upper_pu"
        for index in range(length):
            fit_a = vectors["fit_intercept"][index]
            fit_l = vectors["fit_slope"][index]
            eff_a = vectors["effective_intercept"][index]
            eff_l = vectors["effective_slope"][index]
            values = (fit_a, fit_l, eff_a, eff_l)
            if any(value is None for value in values):
                if not allow_inactive or any(value is not None for value in values):
                    _fail("INVALID_EQ067_VECTOR_SHAPE", legacy=True)
                continue
            if any(not _finite(value) or float(value) < 0.0 for value in values):
                _fail("INVALID_EQ067_VECTOR_VALUE", legacy=True)
            if abs(float(eff_a) - 1.25 * float(fit_a)) > 1.0e-12 or abs(float(eff_l) - 1.25 * float(fit_l)) > 1.0e-12:
                _fail("SAFETY_COEFFICIENT_SERIALIZATION_MISMATCH")
    if q_payload.get("upper_voltage_status") not in {
        "INACTIVE_WITHIN_DEVELOPMENT_METHOD_DOMAIN",
        "ESTIMATED_FROM_DEVELOPMENT_METHOD_DOMAIN",
    }:
        _fail("UPPER_VOLTAGE_STATUS_INVALID")
    upper_fit_a = q_payload["voltage_upper_pu_fit_intercept"]
    upper_fit_l = q_payload["voltage_upper_pu_fit_slope"]
    upper_eff_a = q_payload["voltage_upper_pu_effective_intercept"]
    upper_eff_l = q_payload["voltage_upper_pu_effective_slope"]
    inactive = all(a is None and l is None for a, l in zip(upper_fit_a, upper_fit_l)) and all(a is None and l is None for a, l in zip(upper_eff_a, upper_eff_l))
    if inactive and q_payload.get("upper_voltage_status") != "INACTIVE_WITHIN_DEVELOPMENT_METHOD_DOMAIN":
        _fail("UPPER_VOLTAGE_STATUS_INVALID")
    if not inactive and q_payload.get("upper_voltage_status") != "ESTIMATED_FROM_DEVELOPMENT_METHOD_DOMAIN":
        _fail("UPPER_VOLTAGE_STATUS_INVALID")
    try:
        upper_guard = float(q_payload.get("upper_voltage_guard_pu"))
        upper_threshold = float(q_payload.get("upper_voltage_activation_threshold_pu"))
    except (TypeError, ValueError):
        _fail("UPPER_VOLTAGE_THRESHOLD_MISMATCH")
    if abs(upper_guard - 0.005) > 1.0e-12:
        _fail("UPPER_VOLTAGE_GUARD_MISMATCH")
    if abs(upper_threshold - 1.0e-6) > 1.0e-15:
        _fail("UPPER_VOLTAGE_THRESHOLD_MISMATCH")


def _validate_formula(margin: Mapping[str, Any]) -> None:
    formula = margin.get("formula")
    if not isinstance(formula, Mapping):
        _fail("FORMULA_CONTRACT_MISSING")
    for key, expected in EXPECTED_FORMULA.items():
        value = formula.get(key)
        if isinstance(expected, float):
            if not _finite(value) or abs(float(value) - expected) > 1.0e-12:
                _fail("FORMULA_CONTRACT_MISMATCH")
        elif value != expected:
            _fail("FORMULA_CONTRACT_MISMATCH")


def _validate_trust_regions(margin: Mapping[str, Any], *, frozen: bool) -> dict[str, Any]:
    trust_regions = margin.get("trust_regions")
    if not isinstance(trust_regions, Mapping) or set(trust_regions) != EXPECTED_Q_MODES:
        _fail("Q_SPECIFIC_TRUST_MISSING")
    for q_mode in ("Q0", "Q95"):
        trust = trust_regions.get(q_mode)
        if not isinstance(trust, Mapping):
            _fail("Q_SPECIFIC_TRUST_MISSING")
        if trust.get("coordinate") != EXPECTED_FORMULA["norm"]:
            _fail("TRUST_COORDINATE_MISMATCH")
        if trust.get("applicability_policy") != "FAIL_CLOSED_OUTSIDE_TRUST_REGION":
            _fail("TRUST_POLICY_NOT_FAIL_CLOSED")
        expected_trust_status = "TRUST_REGION_FROZEN" if frozen else "DECLARED_DIAGNOSTIC_NOT_FROZEN"
        if trust.get("status") != expected_trust_status:
            _fail("TRUST_STATUS_INVALID")
        try:
            rho_min, rho_max = float(trust["rho_min"]), float(trust["rho_max"])
        except (KeyError, TypeError, ValueError):
            _fail("TRUST_RHO_DECLARATION_INVALID")
        if not (math.isfinite(rho_min) and math.isfinite(rho_max) and rho_min == 0.0 and rho_max > 0.0):
            _fail("TRUST_RHO_DECLARATION_INVALID")
        if trust.get("upper_voltage_activation_threshold_pu") != 1.0e-6:
            _fail("UPPER_VOLTAGE_THRESHOLD_MISMATCH")
        direction_mode = str(trust.get("direction_embedding", "ALLOCATION_CAPACITY"))
        if direction_mode not in {"ALLOCATION_CAPACITY", "PROXY_ELECTRICAL_EFFECT"}:
            _fail("TRUST_DIRECTION_EMBEDDING_INVALID")
        direction = trust.get("normalized_allocation_direction")
        if not isinstance(direction, Mapping) or not isinstance(direction.get("reference_vectors"), list) or not direction["reference_vectors"]:
            _fail("TRUST_DIRECTION_LIBRARY_MISSING")
        expected_metric = "l2_on_x_over_C_over_rho" if direction_mode == "ALLOCATION_CAPACITY" else "l2_on_proxy_electrical_effect_delta_normalized_by_rate_and_0p05_pu"
        if direction.get("metric") != expected_metric:
            _fail("TRUST_DIRECTION_METRIC_INVALID")
        radius = direction.get("coverage_radius")
        if not _finite(radius) or not 0.0 <= float(radius) <= 2.0:
            _fail("TRUST_DIRECTION_RADIUS_INVALID")
        nearest_range = direction.get("nearest_neighbor_distance_range")
        if not isinstance(nearest_range, list) or len(nearest_range) != 2 or any(not _finite(v) for v in nearest_range) or not 0.0 <= float(nearest_range[0]) <= float(nearest_range[1]) <= 2.0:
            _fail("TRUST_DIRECTION_RADIUS_INVALID")
        reference_ids: set[str] = set()
        expected_direction_dimension = 30 if direction_mode == "ALLOCATION_CAPACITY" else 4 * 140 + 141
        for ref in direction["reference_vectors"]:
            if not isinstance(ref, Mapping) or not isinstance(ref.get("row_id"), str) or ref["row_id"] in reference_ids or not isinstance(ref.get("direction"), list) or len(ref["direction"]) != expected_direction_dimension or any(not _finite(v) for v in ref["direction"]):
                _fail("TRUST_DIRECTION_LIBRARY_INVALID")
            reference_ids.add(ref["row_id"])
            norm = math.sqrt(math.fsum(float(v) * float(v) for v in ref["direction"]))
            if not _finite(norm) or abs(norm - 1.0) > 1.0e-9:
                _fail("TRUST_DIRECTION_LIBRARY_INVALID")
        for name in ("total_export_normalized", "branch_loading_ratio", "voltage_lower_slack_pu", "voltage_upper_slack_pu"):
            raw = trust.get(f"{name}_range")
            if not isinstance(raw, list) or len(raw) != 2 or any(not _finite(v) for v in raw) or float(raw[0]) > float(raw[1]):
                _fail("TRUST_FEATURE_RANGE_INVALID")
        coverage = trust.get("active_constraint_family_coverage")
        if not isinstance(coverage, Mapping) or set(coverage) != TRUST_ACTIVE_FAMILIES or any(not isinstance(v, int) or v < 0 for v in coverage.values()):
            _fail("TRUST_ACTIVE_FAMILY_COVERAGE_INVALID")
        # Counts alone cannot distinguish a new branch/bus active set at the
        # same rho.  The executable domain therefore carries an identity
        # library and the proxy-near-active IDs that generated it.
        branch_ids = trust.get("proxy_near_active_branch_ids_and_ends")
        lower_ids = trust.get("proxy_near_active_lower_bus_ids")
        upper_ids = trust.get("proxy_near_active_upper_bus_ids")
        signature = trust.get("optimizer_active_constraint_signature")
        if not isinstance(branch_ids, Mapping) or set(branch_ids) != {"branch_from_mva", "branch_to_mva"} or any(not isinstance(branch_ids[k], list) for k in branch_ids):
            _fail("TRUST_ACTIVE_SET_COVERAGE_MISSING")
        if not isinstance(lower_ids, list) or not isinstance(upper_ids, list) or not isinstance(signature, Mapping) or not isinstance(signature.get("library"), list) or not signature["library"]:
            _fail("TRUST_ACTIVE_SET_COVERAGE_MISSING")
        if signature.get("serialization_id") != "proxy_active_set_signature.v2":
            _fail("TRUST_ACTIVE_SET_SERIALIZATION_INVALID")
        if signature.get("branch_id_semantics") != "MATPOWER_BRANCH_ROW_1_BASED" or signature.get("bus_id_semantics") != "MATPOWER_BUS_ROW_1_BASED":
            _fail("TRUST_ACTIVE_SET_ID_SEMANTICS_INVALID")
        thresholds = signature.get("thresholds")
        if not isinstance(thresholds, Mapping) or thresholds.get("branch_loading_ratio") != 0.95 or thresholds.get("voltage_slack_pu") != 0.005:
            _fail("TRUST_ACTIVE_SET_THRESHOLD_INVALID")

        def _ids(value: Any, *, upper: int) -> list[int]:
            if not isinstance(value, list) or any(not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= upper for v in value) or value != sorted(set(value)):
                _fail("TRUST_ACTIVE_SET_ID_INVALID")
            return list(value)

        branch_from_union: set[int] = set()
        branch_to_union: set[int] = set()
        lower_union: set[int] = set()
        upper_union: set[int] = set()
        for item in signature["library"]:
            if not isinstance(item, Mapping) or set(item) != TRUST_ACTIVE_FAMILIES:
                _fail("TRUST_ACTIVE_SET_SIGNATURE_INVALID")
            branch_from_union.update(_ids(item["branch_from_mva"], upper=140))
            branch_to_union.update(_ids(item["branch_to_mva"], upper=140))
            lower_union.update(_ids(item["voltage_lower_pu"], upper=141))
            upper_union.update(_ids(item["voltage_upper_pu"], upper=141))
        if _ids(branch_ids["branch_from_mva"], upper=140) != sorted(branch_from_union) or _ids(branch_ids["branch_to_mva"], upper=140) != sorted(branch_to_union) or _ids(lower_ids, upper=141) != sorted(lower_union) or _ids(upper_ids, upper=141) != sorted(upper_union):
            _fail("TRUST_ACTIVE_SET_UNION_INVALID")
        # A joint cluster is the executable replacement for the old
        # independent direction/signature cross-product. It is keyed by the
        # method and side available at the execution boundary and bounded by
        # the preregistered rho bands and geometry caps.
        cluster_policy = trust.get("joint_cluster_policy")
        clusters = trust.get("joint_active_set_clusters")
        declared_feature_names = cluster_policy.get("feature_names") if isinstance(cluster_policy, Mapping) else None
        if not isinstance(cluster_policy, Mapping) or cluster_policy.get("serialization_id") != "proxy_joint_trust_cluster_library.v1" or cluster_policy.get("direction_distance_cap") != JOINT_DIRECTION_DISTANCE_CAP or cluster_policy.get("nominal_jaccard_distance_cap") != JOINT_JACCARD_DISTANCE_CAP or cluster_policy.get("signature_families") != list(SIGNATURE_FAMILIES) or declared_feature_names not in (list(FEATURE_NAMES), list(LEGACY_TRUST_FEATURE_NAMES)):
            _fail("JOINT_TRUST_POLICY_INVALID")
        if not isinstance(clusters, list) or not clusters:
            _fail("JOINT_TRUST_CLUSTER_LIBRARY_MISSING")
        cluster_ids: set[str] = set()
        valid_bands = {name for name, _, _ in RHO_BANDS}
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                _fail("JOINT_TRUST_CLUSTER_INVALID")
            cluster_id = cluster.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id or cluster_id in cluster_ids:
                _fail("JOINT_TRUST_CLUSTER_INVALID")
            cluster_ids.add(cluster_id)
            if cluster.get("q_mode") != q_mode or cluster.get("method_id") not in {"weighted_proportional", "equal_kw_reduction", "flat_level", "max_export_lp", "fairness_qp"} or cluster.get("side") not in {"reference", "reported"} or cluster.get("rho_band") not in valid_bands:
                _fail("JOINT_TRUST_CLUSTER_INVALID")
            if cluster.get("direction_distance_cap") != JOINT_DIRECTION_DISTANCE_CAP or cluster.get("nominal_jaccard_distance_cap") != JOINT_JACCARD_DISTANCE_CAP:
                _fail("JOINT_TRUST_CLUSTER_CAP_INVALID")
            direction = cluster.get("nominal_direction")
            expected_direction_dimension = 30 if str(trust.get("direction_embedding", "ALLOCATION_CAPACITY")) == "ALLOCATION_CAPACITY" else 4 * 140 + 141
            if not isinstance(direction, list) or len(direction) != expected_direction_dimension or any(not _finite(v) for v in direction):
                _fail("JOINT_TRUST_CLUSTER_DIRECTION_INVALID")
            direction_norm = math.sqrt(math.fsum(float(v) * float(v) for v in direction))
            if cluster.get("rho_band") == "B0":
                if not (direction_norm <= 1.0e-9 or abs(direction_norm - 1.0) <= 1.0e-9):
                    _fail("JOINT_TRUST_CLUSTER_DIRECTION_INVALID")
            elif abs(direction_norm - 1.0) > 1.0e-9:
                _fail("JOINT_TRUST_CLUSTER_DIRECTION_INVALID")
            for sig_name in ("nominal_signature", "core_signature", "halo_signature"):
                signature_value = cluster.get(sig_name)
                if not isinstance(signature_value, Mapping) or set(signature_value) != set(SIGNATURE_FAMILIES):
                    _fail("JOINT_TRUST_CLUSTER_SIGNATURE_INVALID")
                for family in SIGNATURE_FAMILIES:
                    _ids(signature_value[family], upper=141 if family.startswith("voltage_") else 140)
            for family in SIGNATURE_FAMILIES:
                core_set, nominal_set, halo_set = set(cluster["core_signature"][family]), set(cluster["nominal_signature"][family]), set(cluster["halo_signature"][family])
                if not core_set.issubset(halo_set) or not nominal_set.issubset(halo_set):
                    _fail("JOINT_TRUST_CLUSTER_SIGNATURE_INVALID")
            feature_ranges = cluster.get("feature_ranges")
            if not isinstance(feature_ranges, Mapping) or set(feature_ranges) != set(declared_feature_names):
                _fail("JOINT_TRUST_CLUSTER_FEATURE_RANGE_INVALID")
            for name in declared_feature_names:
                raw_range = feature_ranges[name]
                if not isinstance(raw_range, list) or len(raw_range) != 2 or any(not _finite(v) for v in raw_range) or float(raw_range[0]) > float(raw_range[1]):
                    _fail("JOINT_TRUST_CLUSTER_FEATURE_RANGE_INVALID")
            source_ids = cluster.get("source_row_ids")
            if not isinstance(source_ids, list) or not source_ids or any(not isinstance(v, str) for v in source_ids) or len(set(source_ids)) != len(source_ids) or not isinstance(cluster.get("row_count"), int) or cluster.get("row_count") != len(source_ids):
                _fail("JOINT_TRUST_CLUSTER_ROSTER_INVALID")
            declared_cluster_hash = cluster.get("cluster_hash")
            cluster_body = dict(cluster); cluster_body.pop("cluster_hash", None)
            if not isinstance(declared_cluster_hash, str) or declared_cluster_hash != canonical_hash(cluster_body):
                _fail("JOINT_TRUST_CLUSTER_HASH_INVALID")
    return {str(q): dict(v) for q, v in trust_regions.items()}


def _validate_candidate(margin: Mapping[str, Any], *, frozen: bool) -> dict[str, Any]:
    if not isinstance(margin, Mapping):
        _fail("INVALID_EQ067_CANDIDATE")
    _validate_feedback_flags(margin)
    if margin.get("serialization_id") != UNIFIED_EQ067_SERIALIZATION_ID:
        _fail("INVALID_EQ067_SERIALIZATION")
    expected_status = "FROZEN" if frozen else "CANDIDATE_NOT_FROZEN"
    if margin.get("status") != expected_status:
        _fail("EQ067_STATUS_INVALID")
    if not frozen and margin.get("freeze_status") == "FROZEN":
        _fail("EQ067_STATUS_INVALID")
    if frozen and margin.get("freeze_status") != "FROZEN":
        _fail("EQ067_STATUS_INVALID")
    if margin.get("loro_coverage_status") != "PASS":
        _fail("LORO_COVERAGE_NOT_PASS")
    if margin.get("loro_audit_serialization_id") not in ALLOWED_LORO_SERIALIZATION_IDS:
        _fail("LORO_BINDING_MISMATCH")
    capability = margin.get("loro_audit_capability")
    if not isinstance(capability, Mapping):
        _fail("LORO_CAPABILITY_MISSING")
    fold_rosters = capability.get("fold_rosters")
    expected_fold_ids = {f"{q}::{reporter}" for q in ("Q0", "Q95") for reporter in ("ieee141-reporter-scan-01", "ieee141-reporter-scan-08", "ieee141-reporter-scan-15")}
    fold_ids: set[str] = set()
    fold_rosters_ok = isinstance(fold_rosters, list) and len(fold_rosters) == 6
    if fold_rosters_ok:
        for fold in fold_rosters:
            if not isinstance(fold, Mapping):
                fold_rosters_ok = False
                break
            fold_id = fold.get("fold_id")
            q_mode = fold.get("q_mode")
            left_out = fold.get("left_out_reporter_id")
            train_reporters = fold.get("train_reporter_ids")
            fold_rosters_ok &= (
                isinstance(fold_id, str)
                and fold_id not in fold_ids
                and isinstance(q_mode, str)
                and q_mode in {"Q0", "Q95"}
                and isinstance(left_out, str)
                and fold_id == f"{q_mode}::{left_out}"
                and isinstance(train_reporters, list)
                and len(train_reporters) == 2
                and len(set(str(value) for value in train_reporters)) == 2
                and left_out not in set(str(value) for value in train_reporters)
                and fold.get("train_row_count") == 340
                and fold.get("test_row_count") == 170
                and isinstance(fold.get("fold_payload_hash"), str)
                and len(fold.get("fold_payload_hash")) == 64
            )
            if isinstance(fold_id, str):
                fold_ids.add(fold_id)
    fold_rosters_ok &= fold_ids == expected_fold_ids
    if (
        capability.get("serialization_id") not in ALLOWED_LORO_SERIALIZATION_IDS
        or capability.get("status") != "DIAGNOSTIC_ONLY"
        or capability.get("coverage_status") != "PASS"
        or capability.get("fold_count") != 6
        or capability.get("development_source_policy") != "BALANCED_PANEL_PLUS_LOW_RHO_PLUS_METHOD_TRAIN_BOTH_SIDES"
        or capability.get("structured_development_probe_row_count") != 960
        or capability.get("method_train_row_count") != 60
        or capability.get("unified_fit_population_row_count") != 1020
        or capability.get("q_mode_row_counts") != {"Q0": 510, "Q95": 510}
        or not fold_rosters_ok
        or set(capability.get("q_modes") or []) != EXPECTED_Q_MODES
        or set(capability.get("constraint_families") or []) != TRUST_ACTIVE_FAMILIES
        or capability.get("constraint_epsilon_mva") != 1.0e-4
        or capability.get("constraint_epsilon_pu") != 1.0e-6
    ):
        _fail("LORO_CAPABILITY_MISMATCH")
    if margin.get("legacy_margin_usage_policy") != LEGACY_MARGIN_USAGE_POLICY:
        _fail("LEGACY_USAGE_POLICY_MISSING", legacy=True)
    if "base_margin_candidate_hash" in margin or any(key in LEGACY_MARGIN_ALIASES for key in margin):
        _fail("LEGACY_MARGIN_ALIAS_AT_TOP_LEVEL", legacy=True)
    provenance_hash = margin.get("legacy_margin_provenance_hash")
    if not isinstance(provenance_hash, str) or not provenance_hash:
        _fail("LEGACY_PROVENANCE_MISSING", legacy=True)
    _validate_formula(margin)
    q_payload = margin.get("q_state_dependent_margin_candidate")
    if not isinstance(q_payload, Mapping) or set(q_payload) != EXPECTED_Q_MODES:
        _fail("Q_SPECIFIC_MARGIN_MISSING")
    for q_mode in ("Q0", "Q95"):
        _validate_affine_vectors(q_payload[q_mode])
    trust_regions = _validate_trust_regions(margin, frozen=frozen)
    # The active trust library must be constructed from detached proxy
    # sources.  A prior diagnostic version admitted CURRENT_METHOD_EXECUTION
    # rows and then replayed those same rows, which is training-internal
    # closure rather than an applicability check.  Reject that construction at
    # the entry point; historical V7 remains provenance-only.
    expansion = margin.get("trust_region_expansion")
    if not isinstance(expansion, Mapping):
        _fail("JOINT_TRUST_EXPANSION_METADATA_MISSING")
    if (
        expansion.get("source_policy") not in ALLOWED_TRUST_SOURCE_POLICIES
        or expansion.get("proxy_only") is not True
        or expansion.get("ac_truth_read") is not False
        or expansion.get("ac_pass_fail_read") is not False
        or expansion.get("current_execution_rows_in_library") is not False
        or not isinstance(expansion.get("proxy_source_manifest_hash"), str)
        or not isinstance(expansion.get("row_count"), int)
        or expansion.get("row_count") < (600 if expansion.get("proxy_source_manifest_serialization_id") == "ieee141_m1_proxy_source_manifest.v6_dev_electrical_ray_families" else 1020)
        or expansion.get("evidence_minimum_source_rows") != 2
    ):
        _fail("JOINT_TRUST_SOURCE_POLICY_INVALID")
    manifest_serialization = expansion.get("proxy_source_manifest_serialization_id")
    if manifest_serialization in {
        "ieee141_m1_proxy_source_manifest.v8_topology_request_electrical_effect_basis",
        "ieee141_m1_proxy_source_manifest.v10_typed_source_binding",
    }:
        required_expansion_flags = {
            "evaluation_results_read",
            "holdout_results_read",
            "mitigation_results_read",
            "failure_ids_read",
            "ac_truth_read",
            "ac_pass_fail_read",
            "rate_policy_mutated",
        }
        if any(expansion.get(flag) is not False for flag in required_expansion_flags):
            _fail("JOINT_TRUST_V10_FEEDBACK_GUARD_INVALID" if manifest_serialization.endswith("v10_typed_source_binding") else "JOINT_TRUST_V8_FEEDBACK_GUARD_INVALID")
    if manifest_serialization == "ieee141_m1_proxy_source_manifest.v4_dev_reporters_fixed_solver_rays":
        declared_scenarios = expansion.get("source_scenario_ids")
        if set(declared_scenarios or []) != set(EXPECTED_DEVELOPMENT_SCENARIO_IDS) or expansion.get("source_reporter_count") != len(EXPECTED_DEVELOPMENT_SCENARIO_IDS):
            _fail("JOINT_TRUST_DEVELOPMENT_ROSTER_INVALID")
    elif manifest_serialization == "ieee141_m1_proxy_source_manifest.v5_immutable_capacity_request_templates":
        if expansion.get("source_policy") != "PREREGISTERED_IMMUTABLE_CAPACITY_REQUEST_TEMPLATE_CROSS_PRODUCT_NO_REPORTER_IDENTITY":
            _fail("JOINT_TRUST_TEMPLATE_SOURCE_POLICY_INVALID")
        if expansion.get("source_reporter_count") not in (0, None):
            _fail("JOINT_TRUST_TEMPLATE_REPORTER_IDENTITY_PRESENT")
        fold_ids = expansion.get("source_fold_ids")
        if fold_ids != [f"CAPACITY_PROFILE_{index:03d}" for index in range(1, 31)]:
            _fail("JOINT_TRUST_TEMPLATE_FOLD_ROSTER_INVALID")
    elif manifest_serialization == "ieee141_m1_proxy_source_manifest.v6_dev_electrical_ray_families":
        if expansion.get("source_policy") != "PREREGISTERED_DEV_REPORTERS_MEANINGFUL_ELECTRICAL_RAY_FAMILIES":
            _fail("JOINT_TRUST_ELECTRICAL_RAY_SOURCE_POLICY_INVALID")
        if not set(EXPECTED_DEVELOPMENT_SCENARIO_IDS).issubset(set(expansion.get("source_scenario_ids") or [])):
            _fail("JOINT_TRUST_DEVELOPMENT_ROSTER_INVALID")
        if not set(("REQUEST_CLIPPED", "CAPACITY_UNIFORM", "FLAT_LEVEL", "TOP12_REQUEST")).issubset(set(expansion.get("source_family_roster") or [])):
            _fail("JOINT_TRUST_ELECTRICAL_RAY_FAMILY_ROSTER_INVALID")
        if expansion.get("source_diversity_required") is not True or any(region.get("source_diversity_required") is not True for region in trust_regions.values()):
            _fail("JOINT_TRUST_SOURCE_DIVERSITY_GATE_MISSING")
    elif manifest_serialization == "ieee141_m1_proxy_source_manifest.v7_electrical_effect_preregistered_family_basis":
        if expansion.get("source_policy") != "PREREGISTERED_DEV_REPORTER_PROXY_ELECTRICAL_EFFECT_FAMILY_BASIS_V1":
            _fail("JOINT_TRUST_ELECTRICAL_EFFECT_SOURCE_POLICY_INVALID")
        if set(expansion.get("source_scenario_ids") or []) != set(EXPECTED_DEVELOPMENT_SCENARIO_IDS):
            _fail("JOINT_TRUST_DEVELOPMENT_ROSTER_INVALID")
        expected_families = {"COMMON_REQUEST_SCALE", "EQUAL_REDUCTION_BREAKPOINT", "FLAT_LEVEL_BREAKPOINT", "CAPACITY_BALANCED", "NEAR_FEEDER_CONCENTRATION", "MID_FEEDER_CONCENTRATION", "FAR_FEEDER_CONCENTRATION", "TOP_K_REQUEST", "PROXY_MAX_EXPORT", "PROXY_FAIRNESS_REFERENCE"}
        if set(expansion.get("source_family_roster") or []) != expected_families:
            _fail("JOINT_TRUST_ELECTRICAL_EFFECT_FAMILY_ROSTER_INVALID")
        if expansion.get("source_diversity_required") is not True or any(region.get("source_diversity_required") is not True for region in trust_regions.values()):
            _fail("JOINT_TRUST_SOURCE_DIVERSITY_GATE_MISSING")
    elif manifest_serialization in {
        "ieee141_m1_proxy_source_manifest.v8_topology_request_electrical_effect_basis",
        "ieee141_m1_proxy_source_manifest.v10_typed_source_binding",
    }:
        if expansion.get("source_policy") != "PREREGISTERED_DEV_REPORTER_TOPOLOGY_REQUEST_PROXY_ELECTRICAL_EFFECT_FAMILY_BASIS_V1":
            _fail("JOINT_TRUST_V10_SOURCE_POLICY_INVALID" if manifest_serialization.endswith("v10_typed_source_binding") else "JOINT_TRUST_V8_SOURCE_POLICY_INVALID")
        if not set(EXPECTED_DEVELOPMENT_SCENARIO_IDS).issubset(set(expansion.get("source_scenario_ids") or [])):
            _fail("JOINT_TRUST_DEVELOPMENT_ROSTER_INVALID")
        if expansion.get("authentic_family_generation") is not True or expansion.get("fallback_source_count") != 0 or expansion.get("rho_target_mismatch_count") != 0 or not isinstance(expansion.get("topology_distance_policy"), str):
            _fail("JOINT_TRUST_V10_GENERATION_BINDING_INVALID" if manifest_serialization.endswith("v10_typed_source_binding") else "JOINT_TRUST_V8_GENERATION_BINDING_INVALID")
        if expansion.get("source_diversity_required") is not True or any(region.get("source_diversity_required") is not True for region in trust_regions.values()):
            _fail("JOINT_TRUST_SOURCE_DIVERSITY_GATE_MISSING")
    elif frozen:
        _fail("JOINT_TRUST_DEVELOPMENT_ROSTER_NOT_FROZEN")
    source_kind_counts = expansion.get("source_kind_counts")
    if source_kind_counts is not None:
        if not isinstance(source_kind_counts, Mapping):
            _fail("JOINT_TRUST_SOURCE_ROSTER_INVALID")
        elif manifest_serialization == "ieee141_m1_proxy_source_manifest.v6_dev_electrical_ray_families":
            if int(source_kind_counts.get("STRUCTURED", 0)) <= 0 or int(source_kind_counts.get("METHOD_TRAIN", 0)) <= 0:
                _fail("JOINT_TRUST_SOURCE_ROSTER_INVALID")
        elif manifest_serialization == "ieee141_m1_proxy_source_manifest.v7_electrical_effect_preregistered_family_basis":
            if int(source_kind_counts.get("GENERATED_PROXY_EFFECT_SOURCE", 0)) != int(expansion.get("row_count")):
                _fail("JOINT_TRUST_SOURCE_ROSTER_INVALID")
        elif manifest_serialization in {
            "ieee141_m1_proxy_source_manifest.v8_topology_request_electrical_effect_basis",
            "ieee141_m1_proxy_source_manifest.v10_typed_source_binding",
        }:
            generated_count = int(source_kind_counts.get("GENERATED_PROXY_EFFECT_TOPOLOGY_SOURCE", 0)) + int(source_kind_counts.get("GENERATED_PROXY_EFFECT_METHOD_SOLVER_SOURCE", 0))
            if generated_count != int(expansion.get("row_count")):
                _fail("JOINT_TRUST_SOURCE_ROSTER_INVALID")
        elif int(source_kind_counts.get("STRUCTURED", -1)) != 960 or int(source_kind_counts.get("METHOD_TRAIN", -1)) != 60:
            _fail("JOINT_TRUST_SOURCE_ROSTER_INVALID")
        if int(expansion.get("row_count")) > 1020 and int(
            source_kind_counts.get("FIXED_PROXY_RAY", 0)
            + source_kind_counts.get("FIXED_PROXY_SOLVER_RAY", 0)
             + source_kind_counts.get("IMMUTABLE_TEMPLATE_PROXY_SOLVER_RAY", 0)
             + source_kind_counts.get("GENERATED_PROXY_EFFECT_SOURCE", 0)
             + source_kind_counts.get("GENERATED_PROXY_EFFECT_TOPOLOGY_SOURCE", 0)
         ) <= 0:
            _fail("JOINT_TRUST_FIXED_RAY_SOURCE_MISSING")
    for q_mode in ("Q0", "Q95"):
        for cluster in trust_regions[q_mode].get("joint_active_set_clusters", []):
            if any(str(row_id).startswith("CURRENT_METHOD_EXECUTION_PROXY_ONLY::") for row_id in cluster.get("source_row_ids", [])):
                _fail("CURRENT_METHOD_EXECUTION_SOURCE_IN_TRUST_LIBRARY")
    fit_count = margin.get("unified_fit_population_row_count")
    structured_count = margin.get("structured_development_probe_row_count")
    method_count = margin.get("method_train_row_count")
    if (structured_count, method_count, fit_count) != (960, 60, 1020):
        _fail("DEVELOPMENT_FIT_POPULATION_BINDING_MISMATCH")
    if frozen:
        for field in ("freeze_decision_id", "frozen_candidate_hash", "frozen_trust_q0_hash", "frozen_trust_q95_hash"):
            if not isinstance(margin.get(field), str) or not margin[field]:
                _fail("FROZEN_ATTESTATION_BINDING_MISSING")
    return trust_regions if frozen else dict(margin.get("trust_region", {})) | {"trust_regions": trust_regions}


def validate_diagnostic_unified_eq067_candidate(margin: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only the current unfrozen diagnostic candidate."""
    return _validate_candidate(margin, frozen=False)


def validate_frozen_unified_eq067_candidate(margin: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only a frozen candidate on the future evidence path."""
    return _validate_candidate(margin, frozen=True)


def validate_unified_eq067_candidate(margin: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the diagnostic guard."""
    return validate_diagnostic_unified_eq067_candidate(margin)


__all__ = [
    "ALLOWED_LORO_SERIALIZATION_IDS",
    "ALLOWED_TRUST_SOURCE_POLICIES",
    "EXPECTED_FORMULA",
    "LEGACY_MARGIN_ALIASES",
    "LEGACY_MARGIN_USAGE_POLICY",
    "UNIFIED_EQ067_SERIALIZATION_ID",
    "validate_diagnostic_unified_eq067_candidate",
    "validate_frozen_unified_eq067_candidate",
    "validate_unified_eq067_candidate",
]
