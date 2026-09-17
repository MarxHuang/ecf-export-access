"""Run the five M1 methods against the Q-specific state-dependent domain.

The legacy pilot used a single finite-difference Q99 vector as if it were a
global allocation boundary.  This diagnostic runner removes that invalid
extrapolation at the allocation boundary itself.  Every method is solved (or
constructed) under the same affine-in-``rho`` envelope, but with its own
registered objective/rule:

* weighted proportional: maximise a common request scale;
* equal-kW reduction: choose the smallest common reduction that is feasible;
* flat level: choose the largest common level that is feasible;
* max-export: two-stage total export plus deterministic tie-break;
* fairness QP: the frozen fairness objective with the max-export reference.

The old grids are immutable request-source diagnostics only.  No AC result,
mitigation result, or evaluation pass/fail is read by this tool, and no RATE or
margin artifact is modified.  The output is diagnostic until the candidate
margin is independently frozen and the three calibration layers are
re-attested.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from r4r.serialization import canonical_dumps, canonical_hash
from r4r.synthetic_rate import load_synthetic_rate_policy
from tools.audit_ieee141_m1_state_dependent_realized_allocation import _screen_state_margin
from tools.ieee141_m1_candidate_guard import (
    LEGACY_MARGIN_ALIASES as _LEGACY_MARGIN_ALIASES,
    UNIFIED_EQ067_SERIALIZATION_ID,
    validate_diagnostic_unified_eq067_candidate,
)
from tools.ieee141_m1_joint_trust import DEFAULT_DIRECTION_MODE, proxy_features, screen_joint_clusters
from tools.run_ieee141_m1_v2_directed_probe_ac_truth import _screen, _solve_one
from tools.run_ieee141_m1_v2_positive_export_feasibility import (
    _constraint_slacks,
    _margin_components,
    _rho,
)
from tools.max_export_canonical import (
    MAX_EXPORT_CANONICAL_ETA,
    MAX_EXPORT_OBJECTIVE_FORM,
    MAX_EXPORT_OBJECTIVE_ID,
    MAX_EXPORT_POLICY_HASH,
    MAX_EXPORT_WEIGHT_POLICY,
    centered_registry_weights,
    solve_certified_max_export,
)


METHOD_IDS = (
    "weighted_proportional",
    "equal_kw_reduction",
    "flat_level",
    "max_export_lp",
    "fairness_qp",
)
SIDES = ("reference", "reported")
# The policy screen tolerances are 1e-6 in the frozen RATE artifact.  The
# allocation domain therefore uses a small, explicit safety buffer rather than
# accepting a solver point that is numerically on the wrong side of the gate.
# Keep a stricter residual threshold than the policy's 1e-6 screen tolerance,
# while allowing the sub-1e-7 numerical residue of the deterministic SLSQP
# tie-break to be rechecked as a valid boundary point.
TOL = 5.0e-7
SAFE_SLACK = 2.0e-6
BRANCH_BUFFER_MVA = 2.0e-6
VOLTAGE_BUFFER_PU = 2.0e-6
BOUND_TOL = 2.0e-6
# An inactive upper-voltage residual family is not represented by a zero
# margin.  It has an explicit applicability guard so that the proxy is not
# allowed to run arbitrarily close to Vmax without calibration support.
UPPER_VOLTAGE_GUARD_PU = 5.0e-3
# Feature-domain comparisons are policy metadata, not physical tolerances.
# Keep the comparison epsilon separate from the AC screen tolerances so an
# allocation just outside the development domain cannot be admitted merely
# because the screen itself has a looser numerical tolerance.
TRUST_FEATURE_EPS = 1.0e-9
# These are applicability-domain identity thresholds, not physical screen
# tolerances.  They must match the thresholds serialized by the EQ067 trust
# builder and are deliberately kept separate from AC pass/fail tolerances.
ACTIVE_BRANCH_RATIO_THRESHOLD = 0.95
ACTIVE_VOLTAGE_SLACK_THRESHOLD_PU = 0.005
# The secondary objective is only a deterministic ordering after stage-one
# export has been fixed.  A positive scale preserves that ordering while
# avoiding the ill-conditioned dual system caused by weights 1..30 next to
# the physical SOC constraints.
# V12.3 active canonical max-export policy.  The historical linear scale and
# 1e-9 MW face tolerance are retained only in old artifacts, never imported
# into the active solver path.
TIE_BREAK_WEIGHT_SCALE = 1.0
TIE_BREAK_ETA = MAX_EXPORT_CANONICAL_ETA
FAIRNESS_ALPHA_MW = 0.20
FAIRNESS_EPSILON_MW = 1.0e-6

def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _validate_unified_margin_candidate(margin: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility export for tests and callers of the active runner.

    The implementation is shared with the alternate allocation entrypoints
    so a legacy common-M1 payload cannot bypass the guard through a different
    import path.
    """
    return validate_diagnostic_unified_eq067_candidate(margin)


def _finite_vector(values: Sequence[Any], *, name: str, length: int) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _trust_feature_values(
    *,
    observed: Mapping[str, Any],
    allocation: Sequence[float],
    capacity: Sequence[float],
    rates: Sequence[float],
    lower: float,
    upper: float,
    direction_mode: str = DEFAULT_DIRECTION_MODE,
    proxy_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute Q-specific trust features from proxy observables.

    These features deliberately use the predicted proxy arrays rather than AC
    truth residuals.  This keeps the applicability domain independent of the
    physical target that the margin is meant to screen.
    """
    alloc = [float(value) for value in allocation]
    cap = [float(value) for value in capacity]
    if len(alloc) != len(cap) or any(not math.isfinite(v) for v in alloc + cap):
        raise ValueError("trust feature allocation/capacity is invalid")
    return proxy_features(
        predicted=observed,
        allocation=alloc,
        capacity=cap,
        rates=rates,
        lower=lower,
        upper=upper,
        direction_mode=direction_mode,
        proxy_baseline=proxy_baseline,
    )


def _screen_q_specific_trust_region(
    *,
    trust: Mapping[str, Any] | None,
    features: Mapping[str, Any],
    q_mode: str,
    method_id: str | None = None,
    side: str | None = None,
) -> dict[str, Any]:
    """Apply the candidate's executable Q-specific feature-domain gate.

    A declared range is not enough: direction is compared with the stored
    development reference vectors and every scalar feature is checked against
    its registered interval.  Any malformed or missing declaration fails
    closed.  This remains diagnostic while the candidate is unfrozen.
    """
    result: dict[str, Any] = {
        "q_mode": q_mode,
        "status": "TRUST_REGION_NOT_EVALUATED",
        "pass": False,
        "failure_reasons": [],
        "features": dict(features),
    }
    if not isinstance(trust, Mapping):
        result["status"] = "TRUST_REGION_DECLARATION_MISSING"
        result["failure_reasons"] = ["TRUST_REGION_DECLARATION_MISSING"]
        return result
    # The active V19 candidate declares only the four scalar joint-cluster
    # features.  Keep the serialized replay shape aligned with that contract;
    # electrical-effect metadata remains available in the runner's internal
    # feature map but is not silently promoted into this historical
    # allocation-direction diagnostic payload.
    cluster_policy = trust.get("joint_cluster_policy")
    if isinstance(cluster_policy, Mapping) and isinstance(cluster_policy.get("feature_names"), list):
        declared_names = {str(name) for name in cluster_policy["feature_names"]}
        if declared_names == {"total_export_normalized", "branch_loading_ratio", "voltage_lower_slack_pu", "voltage_upper_slack_pu"}:
            result["features"] = {
                key: value
                for key, value in features.items()
                if key not in {"direction_mode", "direction_norm", "proxy_effect_norm", "proxy_effect_gain_per_rho"}
            }
    declared_direction_mode = str(trust.get("direction_embedding", DEFAULT_DIRECTION_MODE))
    observed_direction_mode = str(features.get("direction_mode", DEFAULT_DIRECTION_MODE))
    if declared_direction_mode != observed_direction_mode:
        result["status"] = "TRUST_DIRECTION_EMBEDDING_MISMATCH"
        result["failure_reasons"] = ["TRUST_DIRECTION_EMBEDDING_MISMATCH"]
        result["direction"] = {"pass": False, "declared": declared_direction_mode, "observed": observed_direction_mode}
        return result

    def interval(name: str, value: float) -> bool:
        raw = cluster_ranges.get(name) if isinstance(cluster_ranges, Mapping) else trust.get(f"{name}_range")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            result["failure_reasons"].append(f"{name.upper()}_RANGE_MISSING")
            return False
        try:
            lo, hi = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            result["failure_reasons"].append(f"{name.upper()}_RANGE_INVALID")
            return False
        ok = math.isfinite(value) and math.isfinite(lo) and math.isfinite(hi) and lo - TRUST_FEATURE_EPS <= value <= hi + TRUST_FEATURE_EPS
        if not ok:
            result["failure_reasons"].append(f"{name.upper()}_OUTSIDE_TRUST_REGION")
        result.setdefault("intervals", {})[name] = {"lower": lo, "upper": hi, "value": value, "pass": ok}
        return ok

    try:
        rho = float(features["rho"])
        rho_min, rho_max = float(trust.get("rho_min")), float(trust.get("rho_max"))
    except (KeyError, TypeError, ValueError):
        result["failure_reasons"] = ["RHO_DECLARATION_INVALID"]
        result["status"] = "TRUST_REGION_DECLARATION_INVALID"
        return result
    rho_ok = math.isfinite(rho) and math.isfinite(rho_min) and math.isfinite(rho_max) and rho_min - TRUST_FEATURE_EPS <= rho <= rho_max + TRUST_FEATURE_EPS
    result["rho"] = {"lower": rho_min, "upper": rho_max, "value": rho, "pass": rho_ok}
    if not rho_ok:
        result["failure_reasons"].append("RHO_OUTSIDE_TRUST_REGION")

    direction_meta = trust.get("normalized_allocation_direction")
    direction_ok = False
    nearest_distance = None
    if isinstance(direction_meta, Mapping) and isinstance(direction_meta.get("reference_vectors"), list):
        refs = direction_meta.get("reference_vectors")
        direction = [float(value) for value in features.get("direction", [])]
        distances: list[float] = []
        for ref in refs:
            if not isinstance(ref, Mapping) or not isinstance(ref.get("direction"), list) or len(ref["direction"]) != len(direction):
                result["failure_reasons"].append("DIRECTION_REFERENCE_INVALID")
                continue
            try:
                distances.append(math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(direction, ref["direction"]))))
            except (TypeError, ValueError):
                result["failure_reasons"].append("DIRECTION_REFERENCE_INVALID")
        if not bool(features.get("direction_defined")):
            direction_ok = True
            nearest_distance = None
        elif distances:
            nearest_distance = min(distances)
            try:
                radius = float(direction_meta.get("coverage_radius"))
            except (TypeError, ValueError):
                radius = float("nan")
            direction_ok = math.isfinite(radius) and nearest_distance <= radius + TRUST_FEATURE_EPS
            if not direction_ok and not isinstance(trust.get("joint_active_set_clusters"), list):
                result["failure_reasons"].append("ALLOCATION_DIRECTION_OUTSIDE_TRUST_REGION")
        else:
            result["failure_reasons"].append("DIRECTION_REFERENCE_EMPTY")
    else:
        result["failure_reasons"].append("DIRECTION_DOMAIN_NOT_EXECUTABLE")
    result["direction"] = {"nearest_distance": nearest_distance, "pass": direction_ok}

    joint_clusters = trust.get("joint_active_set_clusters")
    if isinstance(joint_clusters, list):
        joint_result = screen_joint_clusters(
            {"clusters": joint_clusters},
            q_mode=q_mode,
            method_id=str(method_id) if method_id is not None else "",
            side=str(side) if side is not None else "",
            features=features,
            min_source_rows=2,
            require_source_diversity=bool(trust.get("source_diversity_required", False)),
        )
        signature_ok = bool(joint_result.get("pass"))
        # The joint cluster is the single executable geometry gate.  Do not
        # retain the legacy independent direction predicate as a second gate;
        # that would recreate the forbidden direction/signature cross-product.
        direction_ok = signature_ok
        result["joint_cluster"] = joint_result
        if not signature_ok:
            result["failure_reasons"].append(str((joint_result.get("failure_reasons") or ["JOINT_CLUSTER_OUTSIDE_TRUST_REGION"])[0]))
    else:
        signature_meta = trust.get("optimizer_active_constraint_signature")
        signature_ok = False
        if isinstance(signature_meta, Mapping) and isinstance(signature_meta.get("library"), list):
            observed_signature = features.get("active_constraint_signature")
            if isinstance(observed_signature, Mapping):
                expected_keys = {"branch_from_mva", "branch_to_mva", "voltage_lower_pu", "voltage_upper_pu"}
                if set(observed_signature) == expected_keys:
                    signature_ok = any(
                        isinstance(item, Mapping)
                        and all(list(item.get(key, [])) == list(observed_signature.get(key, [])) for key in expected_keys)
                        for item in signature_meta["library"]
                    )
        if not signature_ok:
            result["failure_reasons"].append("ACTIVE_SET_SIGNATURE_OUTSIDE_TRUST_REGION")
    result["direction"]["pass"] = direction_ok
    result["active_set_signature"] = {"value": features.get("active_constraint_signature"), "pass": signature_ok}

    # Once the joint cluster has passed, use that cluster's local scalar
    # ranges.  The cluster is the atomic (Q, method, side, rho-band,
    # direction, active-set, feature) applicability object.  Applying the
    # legacy global ranges as an additional predicate can reject a point at a
    # valid cluster boundary and makes the reported gate internally
    # inconsistent.  If no joint cluster library is present, retain the
    # diagnostic legacy ranges for historical replay only.
    cluster_ranges = None
    joint_payload = result.get("joint_cluster")
    if isinstance(joint_payload, Mapping) and joint_payload.get("pass") is True and isinstance(joint_payload.get("feature_ranges"), Mapping):
        cluster_ranges = joint_payload["feature_ranges"]

    scalar_ok = all(
        interval(name, float(features[name]))
        for name in (
            "total_export_normalized",
            "branch_loading_ratio",
            "voltage_lower_slack_pu",
            "voltage_upper_slack_pu",
        )
    )
    result["pass"] = bool(rho_ok and direction_ok and scalar_ok and signature_ok and not result["failure_reasons"])
    result["status"] = "TRUST_REGION_PASS" if result["pass"] else "TRUST_REGION_FAIL_CLOSED"
    return result


def _constraint_builder(
    cp: Any,
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    requested: Sequence[float],
    rho_max: float,
) -> tuple[Any, Any, list[Any]]:
    """Create the common state-dependent SOCP constraints."""
    import numpy as np

    n = len(capacity)
    cap = np.asarray(capacity, dtype=float)
    raw = np.asarray(requested, dtype=float)
    x = cp.Variable(n, nonneg=True, name="allocation_mw")
    rho = cp.Variable(nonneg=True, name="rho")
    constraints: list[Any] = [
        x <= raw,
        x <= cap,
        rho <= float(rho_max),
        cp.norm(cp.multiply(1.0 / cap, x), 2) <= rho,
    ]
    base = model_q["proxy_base"]
    matrices = model_q["derivative_matrices"]
    branch_from_a, branch_from_l = _margin_components(margin_q, "branch_from_mva", len(rates))
    branch_to_a, branch_to_l = _margin_components(margin_q, "branch_to_mva", len(rates))
    lower_a, lower_l = _margin_components(margin_q, "voltage_lower_pu", len(base["voltage_pu"]))
    upper_a, upper_l = _margin_components(
        margin_q, "voltage_upper_pu", len(base["voltage_pu"]), allow_none=True
    )

    def affine(key: str, index: int) -> Any:
        return float(base[key][index]) + np.asarray(matrices[key][index], dtype=float) @ x

    for index, rate in enumerate(rates):
        constraints.append(
            cp.norm(cp.hstack([affine("branch_p_from_mw", index), affine("branch_q_from_mvar", index)]), 2)
            + float(branch_from_a[index]) + float(branch_from_l[index]) * rho <= float(rate) - BRANCH_BUFFER_MVA
        )
        constraints.append(
            cp.norm(cp.hstack([affine("branch_p_to_mw", index), affine("branch_q_to_mvar", index)]), 2)
            + float(branch_to_a[index]) + float(branch_to_l[index]) * rho <= float(rate) - BRANCH_BUFFER_MVA
        )
    for index in range(len(base["voltage_pu"])):
        constraints.append(
            affine("voltage_pu", index) - float(lower_a[index]) - float(lower_l[index]) * rho >= float(lower) + VOLTAGE_BUFFER_PU
        )
        if upper_a[index] is None:
            constraints.append(
                affine("voltage_pu", index)
                <= float(upper) - UPPER_VOLTAGE_GUARD_PU - VOLTAGE_BUFFER_PU
            )
        else:
            constraints.append(
                affine("voltage_pu", index) + float(upper_a[index]) + float(upper_l[index]) * rho <= float(upper) - VOLTAGE_BUFFER_PU
            )
    return x, rho, constraints


def _result(
    *,
    method_id: str,
    status: str,
    solver_status: str,
    allocation: Sequence[float],
    capacity: Sequence[float],
    requested: Sequence[float],
    rho_max: float,
    objective: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
    solver_success: bool = True,
) -> dict[str, Any]:
    # Bounds are repaired explicitly before this function is called.  Do not
    # hide a negative/non-box allocation here: a silent max(0, x) would make
    # the serialized result differ from the solver result without an audit
    # trail.
    values = [float(value) for value in allocation]
    slacks = None
    rho = _rho(values, list(capacity))
    # The conic/scalar result is a valid diagnostic solve, but no method is
    # promoted to an exact optimizer certificate by this runner.  Each method
    # may later attach a more specific diagnostic status; the default keeps
    # failures and future methods equally explicit.
    objective_payload = dict(objective or {})
    objective_payload.setdefault("method_optimality_status", "SOLVER_REPLAY_NOT_PROVEN")
    return {
        "method_id": method_id,
        "status": status,
        "solver_id": "clarabel" if method_id in {"weighted_proportional", "max_export_lp", "fairness_qp"} else "deterministic_scalar_search",
        "solver_status": solver_status,
        "solver_success": bool(solver_success),
        "failure_reason": failure_reason,
        "requested_allocation_mw": [float(v) for v in requested],
        "admitted_allocation_mw": values,
        "admitted_allocation_hash": canonical_hash({"values_mw": values}),
        "rho": float(rho),
        "rho_max": float(rho_max),
        "objective": objective_payload,
        "minimum_constraint_slack": slacks,
    }


def _repair_box_residual(
    allocation: Sequence[float], requested: Sequence[float], capacity: Sequence[float], *, tolerance: float = BOUND_TOL,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Make a solver's tiny box residual explicit, never silently.

    A conic solver can return a point a few 1e-7 MW outside an exact bound.
    Such a point is not accepted as-is.  It is projected to the declared
    request/capacity box only when the residual is within the pre-registered
    ``BOUND_TOL``; the count and maximum correction are serialized.  Larger
    residuals fail the method solve.
    """
    values = [float(value) for value in allocation]
    upper = [min(float(req), float(cap)) for req, cap in zip(requested, capacity)]
    repairs = 0
    max_abs = 0.0
    for index, (value, limit) in enumerate(zip(values, upper)):
        if not math.isfinite(value) or value < -float(tolerance) or value > limit + float(tolerance):
            return None, {"applied": False, "status": "BOUND_RESIDUAL_FAIL", "index": index, "value": value, "limit": limit, "tolerance": float(tolerance)}
        corrected = min(max(value, 0.0), limit)
        delta = abs(corrected - value)
        if delta > 0.0:
            repairs += 1
            max_abs = max(max_abs, delta)
        values[index] = corrected
    return values, {"applied": repairs > 0, "status": "BOUND_REPAIRED" if repairs else "BOUND_EXACT", "count": repairs, "max_abs_correction_mw": max_abs, "tolerance_mw": float(tolerance)}


def _scalar_rule(
    *,
    method_id: str,
    raw: list[float],
    capacity: list[float],
    model_q: Mapping[str, Any],
    rates: list[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    rho_max: float,
) -> dict[str, Any]:
    """Deterministically search the one-dimensional algebraic rule."""
    if method_id == "equal_kw_reduction":
        high = max(raw, default=0.0)

        def candidate(parameter: float) -> list[float]:
            # The typed request is an upper bound, not a capacity guarantee.
            # Apply the immutable feasible-box cap after the method rule so a
            # raw request above capacity is never treated as an invalid input
            # or allowed to enter the proxy outside its declared domain.
            return [min(max(0.0, value - parameter), float(limit)) for value, limit in zip(raw, capacity)]

        parameter_name = "reduction_mw"
    elif method_id == "flat_level":
        high = max(raw, default=0.0)

        def candidate(parameter: float) -> list[float]:
            return [min(value, parameter, float(limit)) for value, limit in zip(raw, capacity)]

        parameter_name = "flat_level_mw"
    else:  # pragma: no cover - guarded by caller
        raise ValueError(method_id)

    def feasible(values: list[float]) -> bool:
        slack = _constraint_slacks(model_q, rates, lower, upper, margin_q, capacity, rho_max, values, upper_guard_pu=UPPER_VOLTAGE_GUARD_PU)
        return bool(slack and min(float(v) for v in slack) >= SAFE_SLACK)

    # Check the correct zero-export endpoint for each algebraic rule.  For a
    # flat-level rule, ``level=0`` is zero export; for equal reduction,
    # ``d=max(request)`` is zero export.  The old implementation accidentally
    # tested the raw request for flat-level, turning ordinary overload into a
    # false domain-infeasible status.
    zero_parameter = high if method_id == "equal_kw_reduction" else 0.0
    if not feasible(candidate(zero_parameter)):
        return _result(
            method_id=method_id,
            status="DOMAIN_INFEASIBLE",
            solver_status="SCALAR_ENDPOINT_INFEASIBLE",
            allocation=[0.0] * len(raw),
            capacity=capacity,
            requested=raw,
            rho_max=rho_max,
            objective={parameter_name: zero_parameter, "feasibility_evaluations": 1, "safety_slack": SAFE_SLACK},
            failure_reason="M1_STATE_DEPENDENT_DOMAIN_INFEASIBLE_AT_ZERO_EXPORT",
            solver_success=False,
        )

    # Do not assume that the two-ended MVA/voltage intersection is globally
    # downward closed in a scalar rule.  Reversal can make a branch endpoint
    # or voltage residual non-monotone.  Search every piecewise box interval
    # induced by request/capacity breakpoints, with a deterministic dense
    # grid.  This is a reproducible diagnostic search; its result is not an
    # optimality certificate until the independent breakpoint audit is passed.
    if method_id == "equal_kw_reduction":
        breakpoints = {0.0, high}
        breakpoints.update(float(value) for value in raw if 0.0 < float(value) < high)
        breakpoints.update(float(value) - float(limit) for value, limit in zip(raw, capacity) if 0.0 < float(value) - float(limit) < high)
    else:
        breakpoints = {0.0, high}
        breakpoints.update(min(float(value), float(limit)) for value, limit in zip(raw, capacity) if 0.0 < min(float(value), float(limit)) < high)
    ordered = sorted(breakpoints)
    parameters: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        span = right - left
        # 64 interior subintervals per piece is the registered diagnostic
        # density.  Breakpoints themselves are always included exactly; the
        # later LORO/optimality gate may request a denser replay without
        # changing the method semantics.
        parameters.extend(left + span * index / 64.0 for index in range(65))
    parameters.extend(ordered)
    unique_parameters = sorted({round(float(value), 14) for value in parameters})
    feasible_points: list[tuple[float, list[float]]] = []
    evaluations = 0
    for value in unique_parameters:
        point = candidate(float(value)); evaluations += 1
        if feasible(point):
            feasible_points.append((float(value), point))
    if not feasible_points:
        return _result(
            method_id=method_id,
            status="DOMAIN_INFEASIBLE",
            solver_status="PIECEWISE_SCAN_NO_FEASIBLE_POINT",
            allocation=[0.0] * len(raw),
            capacity=capacity,
            requested=raw,
            rho_max=rho_max,
            objective={parameter_name: zero_parameter, "feasibility_evaluations": evaluations, "breakpoints": ordered, "interval_subdivisions": 64, "method_optimality_status": "PIECEWISE_SCAN_DIAGNOSTIC_NOT_PROVEN", "safety_slack": SAFE_SLACK},
            failure_reason="M1_STATE_DEPENDENT_DOMAIN_INFEASIBLE_PIECEWISE_SCAN",
            solver_success=False,
        )
    # Equal reduction chooses the smallest feasible reduction; flat level the
    # largest feasible level.  Export total is used as a deterministic
    # secondary key, so a non-monotone feasible island cannot be hidden.
    parameter, best = (min(feasible_points, key=lambda item: (item[0], -math.fsum(item[1]))) if method_id == "equal_kw_reduction" else max(feasible_points, key=lambda item: (item[0], math.fsum(item[1]))) )

    # Refine the selected feasible boundary instead of returning the nearest
    # point of the 64-cell diagnostic grid.  Within each request/capacity
    # breakpoint interval the rule is affine in the scalar parameter and the
    # proxy constraints are convex, so a feasible boundary can be isolated by
    # bisection once the surrounding diagnostic cells have been identified.
    # The dense scan remains the coverage search; this refinement removes its
    # MW-scale quantisation from the admitted allocation.
    refinement_iterations = 0
    ordered_parameters = unique_parameters
    if method_id == "equal_kw_reduction":
        lower_candidates = [value for value in ordered_parameters if value < parameter and not feasible(candidate(value))]
        if lower_candidates:
            left = max(lower_candidates)
            right = parameter
            for _ in range(80):
                midpoint = 0.5 * (left + right)
                refinement_iterations += 1
                if feasible(candidate(midpoint)):
                    right = midpoint
                else:
                    left = midpoint
            parameter = right
            best = candidate(parameter)
    else:
        upper_candidates = [value for value in ordered_parameters if value > parameter and not feasible(candidate(value))]
        if upper_candidates:
            left = parameter
            right = min(upper_candidates)
            for _ in range(80):
                midpoint = 0.5 * (left + right)
                refinement_iterations += 1
                if feasible(candidate(midpoint)):
                    left = midpoint
                else:
                    right = midpoint
            parameter = left
            best = candidate(parameter)
    final_feasible = feasible(best)
    out = _result(
        method_id=method_id,
        status="ADMITTED_STATE_DEPENDENT" if final_feasible else "SCALAR_RESIDUAL_FAIL",
        solver_status="DETERMINISTIC_PIECEWISE_SCAN",
        allocation=best,
        capacity=capacity,
        requested=raw,
        rho_max=rho_max,
        objective={parameter_name: parameter, "feasibility_evaluations": evaluations + refinement_iterations, "breakpoints": ordered, "interval_subdivisions": 64, "boundary_refinement_iterations": refinement_iterations, "method_optimality_status": "PIECEWISE_BREAKPOINT_ROOT_DIAGNOSTIC_NOT_PROVEN", "safety_slack": SAFE_SLACK},
        failure_reason=None if final_feasible else "M1_STATE_DEPENDENT_SCALAR_RESIDUAL_FAIL",
        solver_success=final_feasible,
    )
    out["minimum_constraint_slack"] = float(min(_constraint_slacks(model_q, rates, lower, upper, margin_q, capacity, rho_max, best, upper_guard_pu=UPPER_VOLTAGE_GUARD_PU), default=float("nan")))
    return out


def _slacks_and_jacobian(
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    rho_max: float,
    allocation: Sequence[float],
) -> tuple[Any, Any]:
    """Return exact proxy slacks and analytic Jacobian for SLSQP fallback."""
    import numpy as np

    x = np.asarray(allocation, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    base = model_q["proxy_base"]
    matrices = model_q["derivative_matrices"]
    rho = float(np.sqrt(np.sum((x / cap) ** 2)))
    drho = x / (cap * cap * max(rho, 1.0e-12))
    slacks: list[float] = [float(rho_max) - rho]
    jac: list[Any] = [-drho]
    af, lf = _margin_components(margin_q, "branch_from_mva", len(rates))
    at, lt = _margin_components(margin_q, "branch_to_mva", len(rates))
    al, ll = _margin_components(margin_q, "voltage_lower_pu", len(base["voltage_pu"]))
    au, lu = _margin_components(margin_q, "voltage_upper_pu", len(base["voltage_pu"]), allow_none=True)

    def field(key: str, i: int) -> tuple[float, Any]:
        row = np.asarray(matrices[key][i], dtype=float)
        return float(base[key][i]) + float(row @ x), row

    for i, rate in enumerate(rates):
        p, dp = field("branch_p_from_mw", i); q, dq = field("branch_q_from_mvar", i)
        norm = max(math.hypot(p, q), 1.0e-12)
        slacks.append(float(rate) - BRANCH_BUFFER_MVA - math.hypot(p, q) - float(af[i]) - float(lf[i]) * rho)
        jac.append(-(p * dp + q * dq) / norm - float(lf[i]) * drho)
        p, dp = field("branch_p_to_mw", i); q, dq = field("branch_q_to_mvar", i)
        norm = max(math.hypot(p, q), 1.0e-12)
        slacks.append(float(rate) - BRANCH_BUFFER_MVA - math.hypot(p, q) - float(at[i]) - float(lt[i]) * rho)
        jac.append(-(p * dp + q * dq) / norm - float(lt[i]) * drho)
    for i in range(len(base["voltage_pu"])):
        v, dv = field("voltage_pu", i)
        slacks.append(v - float(lower) - VOLTAGE_BUFFER_PU - float(al[i]) - float(ll[i]) * rho)
        jac.append(dv - float(ll[i]) * drho)
        if au[i] is None:
            slacks.append(float(upper) - VOLTAGE_BUFFER_PU - v - UPPER_VOLTAGE_GUARD_PU)
            jac.append(-dv)
        else:
            slacks.append(float(upper) - VOLTAGE_BUFFER_PU - v - float(au[i]) - float(lu[i]) * rho)
            jac.append(-dv - float(lu[i]) * drho)
    return np.asarray(slacks, dtype=float), np.asarray(jac, dtype=float)


def _slsqp_tie_break(
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    requested: Sequence[float],
    rho_max: float,
    stage1_values: Sequence[float],
    stage1_total: float,
    primary_floor_mw: float | None = None,
    s_t_mw: float | None = None,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Numerically robust deterministic stage-two tie-break.

    Clarabel can fail on the nearly-equality-preserving secondary SOC problem.
    SLSQP starts from the certified stage-one point and uses the exact same
    MVA/voltage/rho slacks with analytic gradients.  A result is accepted only
    when the primary total and every slack are independently rechecked.
    """
    import numpy as np
    from scipy.optimize import minimize

    start = np.asarray(stage1_values, dtype=float)
    weights = np.asarray(centered_registry_weights(len(start)), dtype=float)
    scale = float(s_t_mw if s_t_mw is not None else math.fsum(min(float(a), float(b)) for a, b in zip(requested, capacity)))
    if scale <= 0.0:
        return None, {"status": "SLSQP_ZERO_S_T_UNDEFINED_OBJECTIVE"}
    floor = float(primary_floor_mw if primary_floor_mw is not None else stage1_total)
    bound = [(0.0, min(float(a), float(b))) for a, b in zip(requested, capacity)]

    def values_and_jac(z: Any) -> tuple[Any, Any]:
        return _slacks_and_jacobian(
            model_q=model_q, rates=rates, lower=lower, upper=upper,
            margin_q=margin_q, capacity=capacity, rho_max=rho_max,
            allocation=z,
        )

    result = minimize(
        lambda z: float(weights @ z + TIE_BREAK_ETA / (2.0 * scale) * (z @ z)),
        start,
        jac=lambda z: weights + TIE_BREAK_ETA / scale * z,
        method="SLSQP",
        bounds=bound,
        constraints=(
            {"type": "ineq", "fun": lambda z: values_and_jac(z)[0], "jac": lambda z: values_and_jac(z)[1]},
            {"type": "ineq", "fun": lambda z: float(np.sum(z) - floor), "jac": lambda z: np.ones_like(z)},
        ),
        options={"maxiter": 1200, "ftol": 1.0e-10, "disp": False},
    )
    candidate = np.asarray(result.x, dtype=float) if result.x is not None and np.all(np.isfinite(result.x)) else None
    if candidate is None:
        return None, {"status": "SLSQP_NO_FINITE_SOLUTION", "message": str(result.message), "iterations": int(getattr(result, "nit", 0))}
    slacks, _ = values_and_jac(candidate)
    primary_gap = float(np.sum(candidate) - floor)
    valid = bool(
        result.success
        and np.min(slacks) >= -5.0e-7
        and primary_gap >= -1.0e-7
        and np.all(candidate >= -1.0e-8)
    )
    return (
        candidate.tolist() if valid else None,
        {
            "status": "SLSQP_ACCEPTED" if valid else "SLSQP_RESIDUAL_FAIL",
            "message": str(result.message), "iterations": int(getattr(result, "nit", 0)),
            "minimum_constraint_slack": float(np.min(slacks)), "primary_gap_mw": primary_gap,
            "optimizer_success_flag": bool(result.success),
        },
    )


def _recover_max_export_with_slsqp(
    canonical: Mapping[str, Any],
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    requested: Sequence[float],
    rho_max: float,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Recover the same strict-convex Stage 2 from a valid dual upper bound.

    The source-conic Stage 1 can fail its joint KKT flag because the auxiliary
    rho/SOC primal residual is a few ulps beyond its registered threshold even
    though the projected cone dual remains a valid upper bound.  Likewise, an
    ``optimal_inaccurate`` Stage-2 point can violate the participant box.  This
    fallback does not relax either condition: it uses the projected Stage-1
    upper bound to define the unchanged ``U1-tau_H`` face and independently
    solves/replays the same deterministic Stage-2 objective with SLSQP.
    """

    stage1 = canonical.get("stage1") if isinstance(canonical.get("stage1"), Mapping) else {}
    u1 = stage1.get("u1_mw")
    tau = canonical.get("tau_h_mw")
    s_t = canonical.get("s_t_mw")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (u1, tau, s_t)):
        return None, {"status": "SLSQP_RECOVERY_STAGE1_BRACKET_UNAVAILABLE"}
    if float(s_t) <= 0.0:
        return None, {"status": "SLSQP_RECOVERY_ZERO_S_T"}
    kkt = stage1.get("kkt") if isinstance(stage1.get("kkt"), Mapping) else {}
    dual = kkt.get("dual_objective") if isinstance(kkt.get("dual_objective"), Mapping) else {}
    if stage1.get("pass") is not True and dual.get("available") is not True:
        return None, {"status": "SLSQP_RECOVERY_DUAL_UPPER_UNAVAILABLE"}
    raw_start = canonical.get("allocation")
    if not isinstance(raw_start, list) or len(raw_start) != len(capacity):
        raw_start = kkt.get("primal_allocation_mw")
    if not isinstance(raw_start, list) or len(raw_start) != len(capacity):
        return None, {"status": "SLSQP_RECOVERY_START_UNAVAILABLE"}
    box_upper = [min(float(c), float(r)) for c, r in zip(capacity, requested)]
    start = [min(max(float(value), 0.0), limit) for value, limit in zip(raw_start, box_upper)]
    seed_repair = max(abs(float(a) - float(b)) for a, b in zip(raw_start, start))
    primary_floor = float(u1) - float(tau)
    recovered, details = _slsqp_tie_break(
        model_q=model_q,
        rates=rates,
        lower=lower,
        upper=upper,
        margin_q=margin_q,
        capacity=capacity,
        requested=requested,
        rho_max=rho_max,
        stage1_values=start,
        stage1_total=math.fsum(start),
        primary_floor_mw=primary_floor,
        s_t_mw=float(s_t),
    )
    metadata = {
        **details,
        "recovery_policy": "IDENTICAL_STAGE2_SLSQP_WITH_PROJECTED_STAGE1_DUAL_UPPER",
        "canonical_failure_status": canonical.get("status"),
        "stage1_solver_status": canonical.get("solver_status"),
        "stage1_dual_upper_u1_mw": float(u1),
        "tau_h_mw": float(tau),
        "primary_floor_mw": primary_floor,
        "s_t_mw": float(s_t),
        "initial_seed_box_repair_max_mw": seed_repair,
        "parameter_or_tolerance_change": False,
    }
    return recovered, metadata


def _solve_method(task: tuple[Any, ...]) -> dict[str, Any]:
    if len(task) == 11:
        (
            method_id, model_q, rates, lower, upper, margin_q, capacity, requested,
            rho_max, q_mode, fairness_alpha_mw,
        ) = task
    elif len(task) == 10:
        (
            method_id, model_q, rates, lower, upper, margin_q, capacity, requested, rho_max, q_mode,
        ) = task
        fairness_alpha_mw = FAIRNESS_ALPHA_MW
    else:
        (
            method_id, model_q, rates, lower, upper, margin_q, capacity, requested, rho_max,
        ) = task
        q_mode = "Q0"
        fairness_alpha_mw = FAIRNESS_ALPHA_MW
    fairness_alpha_mw = float(fairness_alpha_mw)
    if not math.isfinite(fairness_alpha_mw) or fairness_alpha_mw <= 0.0:
        return _result(
            method_id=method_id,
            status="INVALID_FAIRNESS_PARAMETER",
            solver_status="INPUT_VALIDATION_FAIL",
            allocation=[0.0] * len(capacity),
            capacity=[float(value) for value in capacity],
            requested=[float(value) for value in requested],
            rho_max=float(rho_max),
            failure_reason="FAIRNESS_ALPHA_MW_MUST_BE_FINITE_AND_POSITIVE",
            solver_success=False,
        )
    raw = [float(value) for value in requested]
    cap = [float(value) for value in capacity]
    if len(raw) != len(cap) or any(not math.isfinite(value) or value < -1e-10 for value in raw) or any(not math.isfinite(limit) or limit <= 0.0 for limit in cap):
        return _result(
            method_id=method_id,
            status="INVALID_REQUEST",
            solver_status="INPUT_VALIDATION_FAIL",
            allocation=[0.0] * len(cap),
            capacity=cap,
            requested=raw,
            rho_max=rho_max,
            failure_reason="REQUEST_OR_CAPACITY_DOMAIN_INVALID",
            solver_success=False,
        )
    if method_id in {"equal_kw_reduction", "flat_level"}:
        return _scalar_rule(
            method_id=method_id, raw=raw, capacity=cap, model_q=model_q, rates=rates,
            lower=lower, upper=upper, margin_q=margin_q, rho_max=rho_max,
        )
    try:
        import cvxpy as cp
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return _result(
            method_id=method_id, status="SOLVER_UNAVAILABLE", solver_status=type(exc).__name__,
            allocation=[0.0] * len(cap), capacity=cap, requested=raw, rho_max=rho_max,
            failure_reason="M1_STATE_DEPENDENT_SOLVER_UNAVAILABLE", solver_success=False,
        )

    x, rho, constraints = _constraint_builder(
        cp, model_q=model_q, rates=rates, lower=lower, upper=upper, margin_q=margin_q,
        capacity=cap, requested=raw, rho_max=rho_max,
    )
    def solve_problem(problem: Any) -> tuple[str, str | None]:
        """Solve with Clarabel, then use a recorded SCS fallback on failure."""
        try:
            problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
                max_iter=1000,
                tol_gap_abs=1.0e-11,
                tol_gap_rel=1.0e-11,
                tol_feas=1.0e-11,
            )
            return str(problem.status), None
        except Exception as first:
            try:
                problem.solve(solver=cp.SCS, verbose=False, eps=1.0e-6, max_iters=200000)
                return str(problem.status), type(first).__name__
            except Exception as second:
                return "SolverError", f"{type(first).__name__}:{type(second).__name__}"

    objective_meta: dict[str, Any] = {}
    try:
        if method_id == "weighted_proportional":
            lam = cp.Variable(nonneg=True, name="request_scale")
            constraints = [*constraints, lam <= 1.0, x == lam * np.asarray(raw, dtype=float)]
            problem = cp.Problem(cp.Maximize(lam), constraints)
            solver_status, fallback = solve_problem(problem)
            allocation_raw = [float(value) for value in x.value] if x.value is not None else [0.0] * len(cap)
            allocation, bound_repair = _repair_box_residual(allocation_raw, raw, cap)
            if allocation is None:
                return _result(method_id=method_id, status="SOLVER_RESIDUAL_FAIL", solver_status=solver_status, allocation=allocation_raw, capacity=cap, requested=raw, rho_max=rho_max, objective={"request_scale": float(lam.value) if lam.value is not None else None, "solver_fallback": fallback, "bound_repair": bound_repair}, failure_reason="REQUEST_BOX_RESIDUAL_FAIL", solver_success=False)
            objective_meta = {"request_scale": float(lam.value) if lam.value is not None else None, "solver_fallback": fallback, "bound_repair": bound_repair, "method_optimality_status": "CONIC_FORMULA_REPLAY_DIAGNOSTIC_NOT_PROVEN"}
        elif method_id == "max_export_lp":
            canonical = solve_certified_max_export(
                model_q=model_q, rates=rates, lower=lower, upper=upper,
                margin_q=margin_q, capacity=cap, raw_request=raw, rho_max=rho_max,
                q_mode=str(q_mode),
            )
            stage1 = canonical.get("stage1") if isinstance(canonical.get("stage1"), Mapping) else {}
            stage2 = canonical.get("stage2") if isinstance(canonical.get("stage2"), Mapping) else {}
            recovery: dict[str, Any] | None = None
            if canonical.get("status") == "CERTIFIED_MAX_EXPORT_STAGE2":
                allocation = list(canonical.get("allocation", [0.0] * len(cap)))
                final_solver_status = "optimal"
                final_objective_value = float(canonical.get("objective_mw")) if canonical.get("objective_mw") is not None else None
            else:
                recovered, recovery = _recover_max_export_with_slsqp(
                    canonical,
                    model_q=model_q,
                    rates=rates,
                    lower=lower,
                    upper=upper,
                    margin_q=margin_q,
                    capacity=cap,
                    requested=raw,
                    rho_max=rho_max,
                )
                if recovered is None:
                    failure_objective = {
                        "stage1_status": canonical.get("solver_status"),
                        "stage1_total_mw": stage1.get("p1_mw"),
                        "stage1_certified_upper_u1_mw": stage1.get("u1_mw"),
                        "stage1_certificate_pass": stage1.get("pass") is True,
                        "method_optimality_status": "MAX_EXPORT_CANONICAL_FAIL_CLOSED",
                        "canonical_status": canonical.get("status"),
                        "slsqp_recovery": recovery,
                        "failure_reason": "MAX_EXPORT_STAGE2_NOT_CERTIFIED",
                    }
                    return _result(
                        method_id=method_id,
                        status="SOLVER_RESIDUAL_FAIL",
                        solver_status=str(canonical.get("solver_status", "CANONICAL_STAGE2_FAILED")),
                        allocation=list(canonical.get("allocation", [0.0] * len(cap))) if isinstance(canonical.get("allocation"), list) else [0.0] * len(cap),
                        capacity=cap,
                        requested=raw,
                        rho_max=rho_max,
                        objective=failure_objective,
                        failure_reason="MAX_EXPORT_STAGE2_NOT_CERTIFIED",
                        solver_success=False,
                    )
                allocation = recovered
                final_solver_status = "SLSQP_ACCEPTED"
                weights = centered_registry_weights(len(allocation))
                s_t = float(recovery["s_t_mw"])
                final_objective_value = float(
                    math.fsum(weight * value for weight, value in zip(weights, allocation))
                    + MAX_EXPORT_CANONICAL_ETA / (2.0 * s_t) * math.fsum(value * value for value in allocation)
                )
            objective_meta = {
                "stage1_status": canonical.get("solver_status"),
                "stage1_total_mw": stage1.get("p1_mw"),
                "stage1_certified_upper_u1_mw": stage1.get("u1_mw"),
                "stage1_raw_upper_u1_mw": stage1.get("u1_raw_mw"),
                "stage1_upper_rounding_guard_mw": stage1.get("upper_rounding_guard_mw"),
                "stage1_dual_gap_mw": stage1.get("gap_mw"),
                "tau_h_mw": canonical.get("tau_h_mw"),
                "s_t_mw": canonical.get("s_t_mw"),
                "stage2_status": stage2.get("kkt", {}).get("solver_status", canonical.get("solver_status")) if isinstance(stage2.get("kkt"), Mapping) else canonical.get("solver_status"),
                "stage2_total_mw": stage2.get("total_mw"),
                "primary_floor_mw": stage2.get("primary_floor_mw"),
                "tie_break_sense": "MINIMIZE",
                "tie_break_weights": canonical.get("weights"),
                "tie_break_weight_policy": MAX_EXPORT_WEIGHT_POLICY,
                "canonical_tie_objective_id": MAX_EXPORT_OBJECTIVE_ID,
                "canonical_tie_objective_form": MAX_EXPORT_OBJECTIVE_FORM,
                "canonical_tie_policy_hash": MAX_EXPORT_POLICY_HASH,
                "canonical_tie_eta": canonical.get("eta_dimensionless"),
                "canonical_tie_s_t_mw": canonical.get("s_t_mw"),
                "rho_quadratic": "FORBIDDEN",
                "strong_convexity_modulus_mw_inverse": canonical.get("strong_convexity_modulus_mw_inverse"),
                "objective_gradient": canonical.get("objective_gradient"),
                "stage2_objective_mw": stage2.get("objective_mw"),
                "stage2_objective_replay_gap_mw": stage2.get("objective_replay_gap_mw"),
                "stage1_certificate_pass": stage1.get("pass") is True,
                "stage2_kkt_replay": stage2.get("kkt"),
                "stage2_bound_repair": stage2.get("bound_repair"),
                "method_optimality_status": (
                    "CERTIFIED_TAU_H_PRIMARY_FACE_STRICT_CONVEX_STAGE2"
                    if canonical.get("status") == "CERTIFIED_MAX_EXPORT_STAGE2"
                    else "PROJECTED_DUAL_UPPER_AND_SLSQP_REPLAYED_STRICT_CONVEX_STAGE2"
                ),
                "solver_attempts": stage2.get("solver_attempts"),
                "slsqp_recovery": recovery,
            }
            problem = None
        elif method_id == "fairness_qp":
            # Fairness must consume the same certified Stage-2 max-export atom;
            # it may not regenerate an old linear reference independently.
            canonical = solve_certified_max_export(
                model_q=model_q, rates=rates, lower=lower, upper=upper,
                margin_q=margin_q, capacity=cap, raw_request=raw, rho_max=rho_max,
                q_mode=str(q_mode),
            )
            stage1_info = canonical.get("stage1") if isinstance(canonical.get("stage1"), Mapping) else {}
            stage2_info = canonical.get("stage2") if isinstance(canonical.get("stage2"), Mapping) else {}
            stage1_total = float(stage1_info.get("p1_mw", 0.0))
            stage1_status = str(canonical.get("solver_status"))
            stage1_fallback = stage1_info.get("solver_attempts")
            tie_status = str(stage2_info.get("kkt", {}).get("solver_status", canonical.get("solver_status"))) if isinstance(stage2_info.get("kkt"), Mapping) else str(canonical.get("solver_status"))
            tie_fallback = stage2_info.get("solver_attempts")
            reference_recovery: dict[str, Any] | None = None
            if canonical.get("status") == "CERTIFIED_MAX_EXPORT_STAGE2":
                reference = canonical.get("allocation")
                reference_bound_repair = {"applied": False, "status": "CERTIFIED_CANONICAL_ATOM"}
            else:
                reference, reference_recovery = _recover_max_export_with_slsqp(
                    canonical,
                    model_q=model_q,
                    rates=rates,
                    lower=lower,
                    upper=upper,
                    margin_q=margin_q,
                    capacity=cap,
                    requested=raw,
                    rho_max=rho_max,
                )
                reference_bound_repair = {
                    "applied": False,
                    "status": "PROJECTED_DUAL_UPPER_AND_SLSQP_REPLAYED_CANONICAL_ATOM",
                }
                if reference is not None:
                    tie_status = "SLSQP_ACCEPTED"
                    tie_fallback = reference_recovery
            if not isinstance(reference, list) or not reference:
                return _result(method_id=method_id, status="REFERENCE_SOLVE_FAILED", solver_status=tie_status, allocation=[0.0]*len(cap), capacity=cap, requested=raw, rho_max=rho_max, objective={"reference_stage1_fallback": stage1_fallback, "reference_tie_break_fallback": tie_fallback}, failure_reason="M1_STATE_DEPENDENT_REFERENCE_TIE_BREAK_FAILED", solver_success=False)
            reference_slacks = _constraint_slacks(model_q, rates, lower, upper, margin_q, cap, rho_max, reference, upper_guard_pu=UPPER_VOLTAGE_GUARD_PU)
            if min(reference_slacks, default=-math.inf) < -5.0e-7:
                return _result(method_id=method_id, status="REFERENCE_SOLVE_FAILED", solver_status=tie_status, allocation=[0.0]*len(cap), capacity=cap, requested=raw, rho_max=rho_max, objective={"reference_stage1_fallback": stage1_fallback, "reference_tie_break_fallback": tie_fallback}, failure_reason="M1_STATE_DEPENDENT_REFERENCE_TIE_BREAK_FAILED", solver_success=False)
            reference = np.asarray(reference, dtype=float)
            d = np.asarray(raw, dtype=float) + FAIRNESS_EPSILON_MW
            phi = float(np.mean(reference / d))
            alpha = fairness_alpha_mw
            linear = -(1.0 + 2.0 * alpha * phi / d)
            qdiag = alpha / (d * d)
            xq, rhoq, cq = _constraint_builder(
                cp, model_q=model_q, rates=rates, lower=lower, upper=upper, margin_q=margin_q,
                capacity=cap, requested=raw, rho_max=rho_max,
            )
            qp = cp.Problem(cp.Minimize(cp.sum(cp.multiply(qdiag, cp.square(xq))) + linear @ xq), cq)
            qp_status, qp_fallback = solve_problem(qp)
            allocation_raw = [float(value) for value in xq.value] if xq.value is not None else [0.0] * len(cap)
            allocation, qp_bound_repair = _repair_box_residual(allocation_raw, raw, cap)
            if allocation is None:
                return _result(method_id=method_id, status="SOLVER_RESIDUAL_FAIL", solver_status=qp_status, allocation=allocation_raw, capacity=cap, requested=raw, rho_max=rho_max, objective={"solver_fallback": qp_fallback, "bound_repair": qp_bound_repair}, failure_reason="REQUEST_BOX_RESIDUAL_FAIL", solver_success=False)
            objective_meta = {
                "alpha": alpha, "epsilon_mw": FAIRNESS_EPSILON_MW, "reference_mode": "ENDOGENOUS_MAX_EXPORT",
                "method_optimality_status": "FAIRNESS_KKT_DIAGNOSTIC_NOT_PROVEN",
                "reference_allocation_mw": list(reference),
                "reference_allocation_hash": canonical_hash({"values_mw": list(reference)}),
                "reference_total_mw": float(math.fsum(reference)), "reference_stage1_total_mw": stage1_total,
                "reference_tie_break_status": tie_status, "reference_tie_break_sense": "MINIMIZE",
                "reference_tie_break_weight_policy": MAX_EXPORT_WEIGHT_POLICY,
                "reference_tie_break_objective_id": MAX_EXPORT_OBJECTIVE_ID,
                "reference_tie_break_objective_form": MAX_EXPORT_OBJECTIVE_FORM,
                "reference_tie_break_policy_hash": MAX_EXPORT_POLICY_HASH,
                "reference_tie_break_eta": canonical.get("eta_dimensionless"),
                "reference_tie_break_s_t_mw": canonical.get("s_t_mw"),
                "reference_tie_break_primary_floor_mw": stage2_info.get("primary_floor_mw"),
                "reference_tie_break_rho_quadratic": "FORBIDDEN",
                "reference_fraction": phi,
                "reference_stage1_status": stage1_status, "reference_stage1_fallback": stage1_fallback,
                "reference_tie_break_fallback": tie_fallback, "solver_fallback": qp_fallback, "bound_repair": qp_bound_repair,
                "reference_stage1_certificate": stage1_info,
                "reference_stage2_certificate": stage2_info,
                "reference_stage1_raw_upper_u1_mw": stage1_info.get("u1_raw_mw"),
                "reference_stage1_upper_rounding_guard_mw": stage1_info.get("upper_rounding_guard_mw"),
                "reference_stage2_bound_repair": stage2_info.get("bound_repair"),
                "reference_bound_repair": reference_bound_repair,
                "reference_slsqp_recovery": reference_recovery,
            }
            problem = qp
            final_solver_status = qp_status
            final_objective_value = float(problem.value) if problem.value is not None else None
        else:
            raise ValueError(f"unknown method {method_id}")
    except Exception as exc:  # pragma: no cover - recorded diagnostic failure
        return _result(method_id=method_id, status="SOLVER_ERROR", solver_status=type(exc).__name__, allocation=[0.0]*len(cap), capacity=cap, requested=raw, rho_max=rho_max, failure_reason="M1_STATE_DEPENDENT_SOLVER_ERROR", solver_success=False)

    solver_status = str(locals().get("final_solver_status", problem.status if problem is not None else "SOLVER_NOT_RUN"))
    slacks = _constraint_slacks(model_q, rates, lower, upper, margin_q, cap, rho_max, allocation, upper_guard_pu=UPPER_VOLTAGE_GUARD_PU)
    rho_value = _rho(allocation, cap)
    success = solver_status in {"optimal", "optimal_inaccurate", "SLSQP_ACCEPTED"} and min(slacks, default=-math.inf) >= -TOL and rho_value <= rho_max + 2.0e-5
    out = _result(
        method_id=method_id,
        status="ADMITTED_STATE_DEPENDENT" if success else "SOLVER_RESIDUAL_FAIL",
        solver_status=solver_status,
        allocation=allocation,
        capacity=cap,
        requested=raw,
        rho_max=rho_max,
        objective=objective_meta,
        failure_reason=None if success else "M1_STATE_DEPENDENT_SOLVER_RESIDUAL_FAIL",
        solver_success=success,
    )
    out["minimum_constraint_slack"] = float(min(slacks, default=float("nan")))
    out["objective_value"] = locals().get("final_objective_value", float(problem.value) if problem is not None and problem.value is not None else None)
    out["constraint_count"] = len(problem.constraints) if problem is not None else 0
    return out


def _grid_rows(*, payload: Mapping[str, Any], q_mode: str, all_ids: Sequence[str], policy_hash: str, allow_extra_ids: bool = False) -> tuple[list[dict[str, Any]], str]:
    # The pilot grid contract uses the explicit ``synthetic_rate_policy_hash``
    # field.  Do not accept a silently renamed alias here: the grid must bind
    # to exactly the same frozen synthetic RATE policy that the runner loaded.
    if payload.get("q_mode") != q_mode or payload.get("synthetic_rate_policy_hash") != policy_hash:
        raise ValueError(f"{q_mode} grid is not bound to the frozen RATE policy")
    grid_body = payload.get("grid")
    declared_grid_hash = payload.get("grid_hash")
    if not isinstance(grid_body, Mapping) or not isinstance(declared_grid_hash, str) or canonical_hash(grid_body) != declared_grid_hash:
        raise ValueError(f"{q_mode} source grid hash does not recompute from the loaded grid body")
    batches = (payload.get("grid") or {}).get("batches")
    if not isinstance(batches, list):
        raise ValueError(f"{q_mode} grid has no batches")
    by_id: dict[str, dict[str, Any]] = {}
    for batch in batches:
        ref = batch.get("reference_batch") if isinstance(batch, dict) else None
        scenario_id = str(ref.get("scenario_id", "")).removesuffix("__reference") if isinstance(ref, dict) else ""
        if not scenario_id or scenario_id in by_id:
            raise ValueError(f"{q_mode} grid has duplicate/empty scenario ID")
        by_id[scenario_id] = batch
    if (set(all_ids) - set(by_id)) or (not allow_extra_ids and set(by_id) != set(all_ids)):
        raise ValueError(f"{q_mode} grid does not contain the complete frozen split")
    rows: list[dict[str, Any]] = []
    for scenario_id in all_ids:
        batch = by_id[scenario_id]
        pair_inputs = batch.get("pair_inputs") if isinstance(batch.get("pair_inputs"), dict) else None
        if pair_inputs is None:
            raise ValueError(f"{q_mode}/{scenario_id} lacks typed pair_inputs")
        capacity_obj = pair_inputs.get("capacity_vector")
        if not isinstance(capacity_obj, dict):
            raise ValueError(f"{q_mode}/{scenario_id} lacks typed capacity vector payload")
        capacity_payload = dict(capacity_obj)
        if capacity_payload.get("q_mode") not in {None, q_mode}:
            raise ValueError(f"{q_mode}/{scenario_id} capacity Q mode is not bound")
        if capacity_payload.get("source_grid_row_id") not in {None, scenario_id}:
            raise ValueError(f"{q_mode}/{scenario_id} capacity source row is not bound")
        declared_capacity_hash = capacity_payload.pop("capacity_vector_hash", None)
        expected_capacity_hash = canonical_hash(capacity_payload)
        if not isinstance(declared_capacity_hash, str) or declared_capacity_hash != expected_capacity_hash:
            raise ValueError(f"{q_mode}/{scenario_id} capacity vector hash does not recompute")
        participant_ids = capacity_payload.get("participant_ids")
        if not isinstance(participant_ids, list) or len(participant_ids) != 30 or len(set(participant_ids)) != 30:
            raise ValueError(f"{q_mode}/{scenario_id} capacity participant order is invalid")
        participant_order_hash = canonical_hash({"participant_ids": [str(value) for value in participant_ids]})
        if capacity_payload.get("participant_order_hash") != participant_order_hash:
            raise ValueError(f"{q_mode}/{scenario_id} capacity participant order binding is invalid")
        pair_capacity = {
            str(item.get("method_id")): item.get("capacity_vector")
            for item in batch.get("method_pairs", []) if isinstance(item, dict)
        }
        for side in SIDES:
            side_payload = batch.get(f"{side}_batch")
            runs = side_payload.get("method_runs") if isinstance(side_payload, dict) else None
            if not isinstance(runs, list):
                raise ValueError("grid side method runs are missing")
            side_input = pair_inputs.get(f"{side}_side")
            raw_request = side_input.get("raw_request") if isinstance(side_input, dict) else None
            if not isinstance(raw_request, dict):
                raise ValueError(f"{q_mode}/{scenario_id}/{side} lacks typed raw request payload")
            raw_hash = canonical_hash(raw_request)
            raw_ids = raw_request.get("participant_ids")
            raw_values = raw_request.get("values_mw")
            if raw_ids != participant_ids or not isinstance(raw_values, list) or len(raw_values) != 30:
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request does not join capacity participant order")
            if raw_request.get("source_side") != side:
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request side is not bound")
            if raw_request.get("q_mode") != q_mode or raw_request.get("scenario_id") != scenario_id:
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request Q mode is not bound")
            if raw_request.get("participant_order_hash") != participant_order_hash:
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request participant order hash is not bound")
            if "reporter_id" not in raw_request or "request_spec_id" not in raw_request or not isinstance(raw_request.get("request_spec_id"), str) or not raw_request.get("request_spec_id"):
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request typed identity fields are missing")
            if raw_request.get("participant_registry_hash") != capacity_payload.get("participant_registry_hash"):
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request participant registry differs from capacity")
            declared_source_hash = side_input.get("request_source_hash") if isinstance(side_input, dict) else None
            if not isinstance(declared_source_hash, str) or len(declared_source_hash) != 64:
                raise ValueError(f"{q_mode}/{scenario_id}/{side} raw request source hash is missing")
            seen: set[str] = set()
            for run in runs:
                if not isinstance(run, dict):
                    continue
                method_id = str(run.get("method_id"))
                if method_id not in METHOD_IDS or method_id in seen:
                    raise ValueError("grid method roster is not canonical")
                seen.add(method_id)
                profile = run.get("profile") if isinstance(run.get("profile"), dict) else {}
                allocation = profile.get("allocation") if isinstance(profile.get("allocation"), dict) else None
                audit = run.get("proxy_audit") if isinstance(run.get("proxy_audit"), dict) else None
                values = allocation.get("values_mw") if allocation else None
                if values is None and audit:
                    values = audit.get("allocation_mw")
                    allocation = audit
                source_capacity = pair_capacity.get(method_id)
                if not isinstance(values, list) or not isinstance(source_capacity, dict):
                    raise ValueError("grid run lacks requested allocation or typed capacity")
                if source_capacity != capacity_obj:
                    raise ValueError(f"{q_mode}/{scenario_id}/{side}/{method_id} capacity payload differs from pair input")
                if source_capacity.get("q_mode") not in {None, q_mode} or source_capacity.get("source_grid_row_id") not in {None, scenario_id}:
                    raise ValueError(f"{q_mode}/{scenario_id}/{side}/{method_id} capacity provenance is not bound to scenario/Q")
                capacities = capacity_obj.get("values_mw")
                if not isinstance(capacities, list):
                    raise ValueError("grid capacity vector lacks values")
                rows.append({
                    "q_mode": q_mode, "scenario_id": scenario_id, "side": side, "method_id": method_id,
                    "requested_allocation_mw": _finite_vector(raw_values, name="raw strategic request", length=30),
                    "raw_request_mw": _finite_vector(raw_values, name="raw strategic request", length=30),
                    "raw_request": raw_request,
                    "raw_request_hash": raw_hash,
                    "raw_request_source_hash": declared_source_hash,
                    "capacity": _finite_vector(capacities, name="capacity", length=30),
                    "capacity_vector": dict(source_capacity),
                    "capacity_vector_hash": source_capacity.get("capacity_vector_hash"),
                    "capacity_spec_hash": source_capacity.get("capacity_spec_hash"),
                    "capacity_participant_ids": list(source_capacity.get("participant_ids", [])),
                    "capacity_participant_order_hash": source_capacity.get("participant_order_hash"),
                    "capacity_participant_registry_hash": source_capacity.get("participant_registry_hash"),
                    "capacity_profile_hash": source_capacity.get("profile_hash"),
                    "capacity_reporter_id": source_capacity.get("reporter_id"),
                    "capacity_q_mode": source_capacity.get("q_mode"),
                    "capacity_source_grid_row_id": source_capacity.get("source_grid_row_id"),
                    "capacity_source_grid_row_hash": source_capacity.get("source_grid_row_hash"),
                    "source_allocation_hash": allocation.get("allocation_hash") if isinstance(allocation, dict) else None,
                    "old_allocation_hash": allocation.get("allocation_hash") if isinstance(allocation, dict) else None,
                    "source_method_run_hash": canonical_hash(run), "source_grid_hash": declared_grid_hash,
                    "source_binding": {
                        "q_mode": q_mode, "scenario_id": scenario_id, "side": side,
                        "method_id": method_id, "source_method_run_hash": canonical_hash(run),
                        "raw_request_hash": raw_hash, "raw_request_source_hash": declared_source_hash, "source_request_hash": declared_source_hash, "capacity_vector_hash": source_capacity.get("capacity_vector_hash"),
                        "source_grid_row_id": source_capacity.get("source_grid_row_id"),
                        "request_spec_id": raw_request.get("request_spec_id"), "reporter_id": raw_request.get("reporter_id"),
                        "participant_ids": list(participant_ids), "participant_order_hash": participant_order_hash,
                        "participant_registry_hash": source_capacity.get("participant_registry_hash"),
                    },
                    "request_binding_semantics": "typed_raw_request_upper_bound; old_allocation_provenance_only",
                })
            if seen != set(METHOD_IDS):
                raise ValueError("grid side does not contain all five methods")
    return rows, str(payload.get("grid_hash"))


def _method_uniformity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["q_mode"]), str(row["scenario_id"]), str(row["side"])), []).append(row)
    checks = []
    for key, group in sorted(groups.items()):
        ref = list(group[0]["raw_request_mw"])
        diffs = [max(abs(float(a) - float(b)) for a, b in zip(ref, row["raw_request_mw"])) for row in group[1:]]
        # The immutable grid is generated through JSON decimal round-trips;
        # use a declared serialization tolerance rather than allowing a
        # one-ulp representation difference to turn a uniform request set into
        # a false non-uniformity finding.
        checks.append({"q_mode": key[0], "scenario_id": key[1], "side": key[2], "method_count": len(group), "max_abs_difference_mw": max(diffs or [0.0]), "uniform_within_tolerance": max(diffs or [0.0]) <= 2.0e-8})
    return {"group_count": len(checks), "all_uniform": all(bool(item["uniform_within_tolerance"]) for item in checks), "checks": checks}


def run_state_dependent_method_execution(*, case_path: Path, policy_path: Path, proxy_model_path: Path, margin_path: Path, split_path: Path, q0_grid_path: Path, q95_grid_path: Path, output_path: Path, rho_max: float = 3.5, workers: int = 12, development_only: bool = False) -> dict[str, Any]:
    if not math.isfinite(float(rho_max)) or float(rho_max) <= 0.0:
        raise ValueError("rho_max must be positive and finite")
    policy_payload = _load(policy_path)
    policy = load_synthetic_rate_policy(policy_payload)
    model = _load(proxy_model_path); margin = _load(margin_path); split = _load(split_path)
    policy_hash = policy.policy_hash.to_json()
    if model.get("policy_hash") != policy_hash or margin.get("policy_hash") != policy_hash:
        raise ValueError("state-dependent M1 inputs are not bound to the same RATE policy")
    # Preserve the explicit rho mismatch diagnostic even for historical
    # artifacts, while the actual solve remains fail-closed below.
    legacy_trust = margin.get("trust_region")
    if isinstance(legacy_trust, Mapping):
        try:
            legacy_rho_max = float(legacy_trust.get("rho_max"))
        except (TypeError, ValueError):
            legacy_rho_max = float("nan")
        if math.isfinite(legacy_rho_max) and abs(float(rho_max) - legacy_rho_max) > 1.0e-9:
            raise ValueError("execution rho_max must equal the candidate's registered trust-region bound")
    registered_trust = _validate_unified_margin_candidate(margin)
    # Keep the scalar/top-level domain and the Q-specific declarations as
    # separate typed fields.  The guard returns a merged convenience object,
    # but serializing that merge as ``margin_trust_region`` obscures the
    # contract boundary and makes independent replay ambiguous.
    top_level_trust = margin.get("trust_region")
    if not isinstance(top_level_trust, Mapping):
        raise ValueError("candidate top-level trust region is missing")
    registered_rho_max = float(registered_trust["rho_max"])
    if abs(float(rho_max) - registered_rho_max) > 1.0e-9:
        raise ValueError("execution rho_max must equal the candidate's registered trust-region bound")
    # A Q-specific declaration is optional for old diagnostic artifacts.  If
    # present, it is authoritative and is checked at this execution boundary.
    trust_by_q = margin.get("trust_regions")
    if trust_by_q is not None:
        if not isinstance(trust_by_q, Mapping) or set(trust_by_q) != {"Q0", "Q95"}:
            raise ValueError("candidate Q-specific trust regions are malformed")
        for q_mode in ("Q0", "Q95"):
            q_trust = trust_by_q[q_mode]
            if not isinstance(q_trust, Mapping) or q_trust.get("applicability_policy") != "FAIL_CLOSED_OUTSIDE_TRUST_REGION":
                raise ValueError("candidate Q-specific trust regions are not fail-closed")
            if q_trust.get("status") != "DECLARED_DIAGNOSTIC_NOT_FROZEN":
                raise ValueError("candidate Q-specific trust regions must remain diagnostic")
            try:
                q_rho_max = float(q_trust.get("rho_max"))
            except (TypeError, ValueError):
                raise ValueError("candidate Q-specific trust region has invalid rho_max") from None
            if not math.isfinite(q_rho_max) or q_rho_max <= 0.0 or abs(q_rho_max - float(rho_max)) > 1.0e-9:
                raise ValueError("execution rho_max must equal each Q-specific trust-region bound")
    margin_payload = margin.get("q_state_dependent_margin_candidate")
    if not isinstance(margin_payload, dict) or not all(isinstance(margin_payload.get(q), dict) for q in ("Q0", "Q95")):
        raise ValueError("margin candidate lacks Q-specific affine components")
    dev = [str(v) for v in split.get("development_scenario_ids", [])]; hold = [str(v) for v in split.get("holdout_scenario_ids", [])]; ev = [str(v) for v in split.get("evaluation_scenario_ids", [])]
    all_ids = dev + hold + ev
    if not all_ids or len(set(all_ids)) != len(all_ids):
        raise ValueError("split must contain one ordered disjoint full reporter set")
    # The structured 960-row corpus and the 60-row method-train corpus are
    # separate calibration inputs.  A development-only execution may consume
    # only the preregistered development reporters even when the surrounding
    # split manifest also names holdout/evaluation reporters.  It is explicit
    # at the call site and in the output; a partial grid can never silently be
    # treated as a full split.
    executed_ids = dev if development_only else all_ids
    if not executed_ids:
        raise ValueError("development-only execution requires a nonempty development stratum")
    source_rows: list[dict[str, Any]] = []; grid_hashes: dict[str, str] = {}
    for q_mode, path in (("Q0", q0_grid_path), ("Q95", q95_grid_path)):
        rows, grid_hash = _grid_rows(payload=_load(path), q_mode=q_mode, all_ids=executed_ids, policy_hash=policy_hash, allow_extra_ids=development_only)
        source_rows.extend(rows); grid_hashes[q_mode] = grid_hash
    rates = [float(item["rate_a_mva"]) for item in policy_payload["branch_ratings"]]
    lower = float(policy_payload["voltage_lower_limit_pu"]); upper = float(policy_payload["voltage_upper_limit_pu"])
    uniformity = _method_uniformity(source_rows)
    tasks = [(row["method_id"], model["q_models"][row["q_mode"]], rates, lower, upper, margin_payload[row["q_mode"]], row["capacity"], row["raw_request_mw"], float(rho_max)) for row in source_rows]
    if workers <= 1:
        method_results = [_solve_method(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            method_results = list(executor.map(_solve_method, tasks))
    for row, solved in zip(source_rows, method_results):
        row["method_execution"] = solved
    ac_tasks = []
    for row in source_rows:
        execution = row["method_execution"]
        allocation = execution.get("admitted_allocation_mw", [0.0] * len(row["capacity"]))
        probe = {"target_id": f"STATE_DEPENDENT_METHOD::{row['q_mode']}::{row['scenario_id']}::{row['side']}::{row['method_id']}", "target_kind": "STATE_DEPENDENT_METHOD_ALLOCATION", "status": execution.get("status"), "q_mode": row["q_mode"], "reporter_id": row["scenario_id"], "allocation_side": row["side"], "method_id": row["method_id"], "allocation_mw": allocation, "allocation_hash": execution.get("admitted_allocation_hash")}
        ac_tasks.append((str(case_path), policy_payload, model, probe))
    if workers <= 1:
        ac_results = [_solve_one(task) if bool(row["method_execution"].get("solver_success")) else {"solver_success": False, "solver_status": "SKIPPED_ALLOCATION_SOLVE_FAILED", "actual_screen": None, "predicted_values": None, "actual_values": None} for row, task in zip(source_rows, ac_tasks)]
    else:
        # Avoid feeding an invalid zero allocation to AC as a success result;
        # failed allocation rows remain explicit diagnostics.
        runnable = [(index, task) for index, (row, task) in enumerate(zip(source_rows, ac_tasks)) if bool(row["method_execution"].get("solver_success"))]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            solved = list(executor.map(_solve_one, [task for _, task in runnable]))
        ac_results = [{"solver_success": False, "solver_status": "SKIPPED_ALLOCATION_SOLVE_FAILED", "actual_screen": None, "predicted_values": None, "actual_values": None} for _ in source_rows]
        for (index, _), value in zip(runnable, solved):
            ac_results[index] = value
    rows: list[dict[str, Any]] = []
    for source, ac in zip(source_rows, ac_results):
        execution = source["method_execution"]
        if ac.get("solver_success"):
            # AC truth serializes the observable proxy fields separately from
            # the allocation binding.  The state-margin screen deliberately
            # requires both, so attach the already-solved vector explicitly;
            # this is a read-only join, never a post-solve adjustment.
            predicted = dict(ac["predicted_values"])
            predicted["allocation_mw"] = list(execution.get("admitted_allocation_mw", []))
            state = _screen_state_margin(predicted=predicted, actual=ac["actual_values"], rates=rates, lower=lower, upper=upper, margin_q=margin_payload[source["q_mode"]], capacity=source["capacity"], tolerance_mva=float(policy_payload["branch_tolerance_mva"]), tolerance_pu=float(policy_payload["voltage_tolerance_pu"]), upper_guard_pu=UPPER_VOLTAGE_GUARD_PU)
            predicted_screen = _screen(predicted, rates, lower, upper, float(policy_payload["branch_tolerance_mva"]), float(policy_payload["voltage_tolerance_pu"]))
            actual_screen = ac.get("actual_screen")
            branch_false_safe = bool(predicted_screen["branch_pass"] and isinstance(actual_screen, Mapping) and not actual_screen.get("branch_pass"))
            voltage_false_safe = bool(predicted_screen["voltage_pass"] and isinstance(actual_screen, Mapping) and not actual_screen.get("voltage_pass"))
            joint_false_safe = bool(predicted_screen["joint_pass"] and isinstance(actual_screen, Mapping) and not actual_screen.get("joint_pass"))
            m1_envelope_branch_false_safe = bool(state.get("proxy_branch_pass") and not state.get("ac_branch_pass"))
            m1_envelope_voltage_false_safe = bool((state.get("proxy_voltage_lower_pass") and state.get("proxy_voltage_upper_pass")) and not (state.get("ac_voltage_lower_pass") and state.get("ac_voltage_upper_pass")))
            m1_envelope_joint_false_safe = bool(state.get("proxy_joint_pass") and not state.get("ac_joint_pass"))
            trust_features = _trust_feature_values(
                observed=ac["predicted_values"],
                allocation=execution.get("admitted_allocation_mw", []),
                capacity=source["capacity"],
                rates=rates,
                lower=lower,
                upper=upper,
                direction_mode=str((trust_by_q.get(source["q_mode"]) or {}).get("direction_embedding", DEFAULT_DIRECTION_MODE)) if isinstance(trust_by_q, Mapping) else DEFAULT_DIRECTION_MODE,
                proxy_baseline=((model.get("q_models", {}).get(source["q_mode"], {}) or {}).get("proxy_base") if isinstance(model.get("q_models", {}).get(source["q_mode"], {}), Mapping) else None),
            )
            trust_screen = _screen_q_specific_trust_region(
                trust=trust_by_q.get(source["q_mode"]) if isinstance(trust_by_q, Mapping) else None,
                features=trust_features,
                q_mode=source["q_mode"],
                method_id=source["method_id"],
                side=source["side"],
            )
            state["trust_region_pass"] = bool(trust_screen.get("pass"))
            if not state["trust_region_pass"]:
                state.setdefault("failure_reasons", []).extend(str(v) for v in trust_screen.get("failure_reasons", []))
        else:
            state = {
                "proxy_branch_pass": False, "proxy_voltage_lower_pass": False,
                "proxy_voltage_upper_pass": False, "proxy_joint_pass": False,
                "envelope_pass": False, "ac_branch_pass": False,
                "ac_voltage_lower_pass": False, "ac_voltage_upper_pass": False,
                "ac_joint_pass": False, "upper_voltage_applicability_pass": False,
                "upper_voltage_applicability_guard_pu": UPPER_VOLTAGE_GUARD_PU,
                "upper_voltage_applicability_status": "NOT_EVALUATED_AC_FAILED",
                "trust_region_pass": False,
                "rho": execution.get("rho"),
                "failure_category": "AC_SKIPPED_OR_FAILED",
                "failure_reasons": ["AC_SKIPPED_OR_FAILED"],
            }
            predicted_screen = None
            branch_false_safe = False
            voltage_false_safe = False
            joint_false_safe = False
            m1_envelope_branch_false_safe = False
            m1_envelope_voltage_false_safe = False
            m1_envelope_joint_false_safe = False
            trust_features = None
            trust_screen = {
                "q_mode": source["q_mode"],
                "status": "TRUST_REGION_NOT_EVALUATED_AC_FAILED",
                "pass": False,
                "failure_reasons": ["AC_SKIPPED_OR_FAILED"],
            }
        rows.append({
            **source,
            "solution_hash": ac.get("solution_hash"),
            "solver_success": ac.get("solver_success"),
            "solver_status": ac.get("solver_status"),
            "ac_input": ac.get("ac_input"),
            "ac_input_hash": ac.get("ac_input_hash"),
            "predicted_values": ac.get("predicted_values"),
            "predicted_values_hash": ac.get("predicted_values_hash"),
            "actual_values": ac.get("actual_values"),
            "actual_values_hash": ac.get("actual_values_hash"),
            "solver_evidence": ac.get("solver_evidence"),
            "actual_screen": ac.get("actual_screen"),
            "predicted_screen": predicted_screen,
            "branch_false_safe": branch_false_safe,
            "voltage_false_safe": voltage_false_safe,
            "joint_false_safe": joint_false_safe,
            "raw_proxy_branch_false_safe": branch_false_safe,
            "raw_proxy_voltage_false_safe": voltage_false_safe,
            "raw_proxy_joint_false_safe": joint_false_safe,
            "m1_envelope_branch_false_safe": m1_envelope_branch_false_safe,
            "m1_envelope_voltage_false_safe": m1_envelope_voltage_false_safe,
            "m1_envelope_joint_false_safe": m1_envelope_joint_false_safe,
            "trust_region_features": trust_features,
            "trust_region_screen": trust_screen,
            "trust_region_observation_source": "PROXY_PREDICTED_VALUES_ONLY",
            "state_dependent_screen": state,
        })
    split_sets = {"development": set(dev), "holdout": set(hold), "evaluation": set(ev)}
    def rate(items: Sequence[Mapping[str, Any]], key: str) -> float:
        return sum(bool(item.get("state_dependent_screen", {}).get(key)) for item in items) / max(1, len(items))
    summary: dict[str, Any] = {"row_count": len(rows), "rho_max": float(rho_max), "method_solver_pass_rate": sum(bool(row["method_execution"].get("solver_success")) for row in rows) / max(1, len(rows)), "proxy_joint_pass_rate": rate(rows, "proxy_joint_pass"), "envelope_pass_rate": rate(rows, "envelope_pass"), "ac_joint_pass_rate": rate(rows, "ac_joint_pass"), "raw_proxy_false_safe_rate": sum(bool(row.get("raw_proxy_joint_false_safe")) for row in rows) / max(1, len(rows)), "m1_envelope_false_safe_rate": sum(bool(row.get("m1_envelope_joint_false_safe")) for row in rows) / max(1, len(rows)), "trust_region_pass_rate": sum(bool(row.get("trust_region_screen", {}).get("pass")) for row in rows) / max(1, len(rows)), "state_dependent_joint_pass_rate": sum(bool(row["method_execution"].get("solver_success") and row["state_dependent_screen"].get("proxy_joint_pass") and row["state_dependent_screen"].get("envelope_pass") and row["state_dependent_screen"].get("ac_joint_pass") and row.get("trust_region_screen", {}).get("pass")) for row in rows) / max(1, len(rows)), "by_method": {}, "by_q_mode": {}, "by_stratum": {}}
    for method_id in METHOD_IDS:
        selected = [row for row in rows if row["method_id"] == method_id]
        summary["by_method"][method_id] = {"row_count": len(selected), "method_solver_pass_rate": sum(bool(row["method_execution"].get("solver_success")) for row in selected) / max(1, len(selected)), "proxy_joint_pass_rate": rate(selected, "proxy_joint_pass"), "envelope_pass_rate": rate(selected, "envelope_pass"), "ac_joint_pass_rate": rate(selected, "ac_joint_pass"), "trust_region_pass_rate": sum(bool(row.get("trust_region_screen", {}).get("pass")) for row in selected) / max(1, len(selected)), "state_dependent_joint_pass_rate": sum(bool(row["method_execution"].get("solver_success") and row["state_dependent_screen"].get("proxy_joint_pass") and row["state_dependent_screen"].get("envelope_pass") and row["state_dependent_screen"].get("ac_joint_pass") and row.get("trust_region_screen", {}).get("pass")) for row in selected) / max(1, len(selected)), "total_export_mean_mw": math.fsum(math.fsum(row["method_execution"].get("admitted_allocation_mw", [])) for row in selected) / max(1, len(selected))}
    for q_mode in ("Q0", "Q95"):
        selected = [row for row in rows if row["q_mode"] == q_mode]
        summary["by_q_mode"][q_mode] = {"row_count": len(selected), "method_solver_pass_rate": sum(bool(row["method_execution"].get("solver_success")) for row in selected) / max(1, len(selected)), "proxy_joint_pass_rate": rate(selected, "proxy_joint_pass"), "envelope_pass_rate": rate(selected, "envelope_pass"), "ac_joint_pass_rate": rate(selected, "ac_joint_pass"), "trust_region_pass_rate": sum(bool(row.get("trust_region_screen", {}).get("pass")) for row in selected) / max(1, len(selected))}
    for name, ids in split_sets.items():
        selected = [row for row in rows if row["scenario_id"] in ids]
        summary["by_stratum"][name] = {"row_count": len(selected), "method_solver_pass_rate": sum(bool(row["method_execution"].get("solver_success")) for row in selected) / max(1, len(selected)), "proxy_joint_pass_rate": rate(selected, "proxy_joint_pass"), "envelope_pass_rate": rate(selected, "envelope_pass"), "ac_joint_pass_rate": rate(selected, "ac_joint_pass"), "trust_region_pass_rate": sum(bool(row.get("trust_region_screen", {}).get("pass")) for row in selected) / max(1, len(selected)), "state_dependent_joint_pass_rate": sum(bool(row["method_execution"].get("solver_success") and row["state_dependent_screen"].get("proxy_joint_pass") and row["state_dependent_screen"].get("envelope_pass") and row["state_dependent_screen"].get("ac_joint_pass") and row.get("trust_region_screen", {}).get("pass")) for row in selected) / max(1, len(selected))}
    rows.sort(key=lambda row: (row["q_mode"], executed_ids.index(row["scenario_id"]), row["side"], row["method_id"]))
    result: dict[str, Any] = {"serialization_id": "ieee141_m1_state_dependent_method_execution.v1", "status": "DIAGNOSTIC_STATE_DEPENDENT_METHOD_EXECUTION_COMPLETE", "policy_hash": policy_hash, "network_hash": policy.network_hash.to_json(), "proxy_model_hash": model.get("result_hash"), "margin_candidate_hash": margin.get("result_hash"), "margin_loro_audit_hash": margin.get("loro_audit_hash"), "margin_loro_tolerance_policy_hash": margin.get("loro_tolerance_policy_hash"), "legacy_margin_usage_policy": margin.get("legacy_margin_usage_policy"), "margin_trust_region": dict(top_level_trust), "trust_regions": margin.get("trust_regions"), "split_hash": canonical_hash({key: value for key, value in split.items() if key != "result_hash"}), "q_grid_hashes": grid_hashes, "scenario_ids": executed_ids, "execution_scope": "DEVELOPMENT_TRAIN_ONLY" if development_only else "FULL_SPLIT_DIAGNOSTIC", "development_scenario_ids": dev, "holdout_scenario_ids": hold, "evaluation_scenario_ids": ev, "rho_max": float(rho_max), "method_roster": list(METHOD_IDS), "method_uniformity_check": uniformity, "allocation_semantics": "method_specific_state_dependent_feasible_domain_execution", "state_dependent_domain": {"coordinate": "rho(x;C)=sqrt(sum_i((x_i/C_i)^2))", "center_rule": "zero_allocation_origin; no center subtraction", "immutable_bounds": "0 <= x_i <= min(request_i, capacity_i)", "branch_buffer_mva": BRANCH_BUFFER_MVA, "voltage_buffer_pu": VOLTAGE_BUFFER_PU, "upper_voltage_guard_pu": UPPER_VOLTAGE_GUARD_PU, "upper_voltage_guard_policy": "inactive upper margin requires Vmax - Vproxy >= guard within development method domain", "solver_residual_tolerance": TOL, "bound_residual_tolerance_mw": BOUND_TOL, "bound_residual_policy": "explicit repair only within tolerance; otherwise method failure", "trust_region_policy": "candidate_registered_fail_closed", "trust_region_feature_gate": "proxy_predicted_direction_export_loading_and_voltage_ranges_plus_active_set_fail_closed", "trust_region_observation_source": "PROXY_PREDICTED_VALUES_ONLY", "trust_region_feature_epsilon": TRUST_FEATURE_EPS}, "rows": rows, "summary": summary, "evaluation_executed": not development_only, "evaluation_execution_consumed": not development_only, "holdout_execution_consumed": not development_only, "evaluation_feedback_used": False, "holdout_feedback_used": False, "evaluation_results_read": False, "holdout_results_read": False, "realized_audit_results_read": False, "calibration_read": False, "mitigation_results_read": False, "rate_policy_mutated": False, "margin_refit_performed": False, "diagnostic_only": True, "primary_evidence_eligible": False, "t3_eligible": False, "t4_eligible": False, "t5_eligible": False, "result_hash": None}
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join(["# State-dependent M1 method execution (diagnostic)", "", f"- Execution scope: `{result['execution_scope']}`", f"- Result hash: `{result['result_hash']}`", f"- Method solver pass rate: `{summary['method_solver_pass_rate']}`", f"- State-dependent joint pass rate: `{summary['state_dependent_joint_pass_rate']}`", f"- Legacy request uniformity: `{uniformity['all_uniform']}`", "- This output is a development-only method-train replay when marked DEVELOPMENT_TRAIN_ONLY; holdout/evaluation rows are not consumed in that mode.", "- Evaluation feedback was not used to refit or select the margin.", "- Each method is solved with its own registered rule/objective under the same Q-specific state-dependent envelope.", "- The old common-margin grid supplies only immutable raw-request and provenance bindings; its admitted allocations do not define the new feasible domain."]) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "policy", "proxy-model", "margin", "split", "q0-grid", "q95-grid", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rho-max", type=float, default=3.5); parser.add_argument("--workers", type=int, default=12); parser.add_argument("--development-only", action="store_true", help="execute only the preregistered development reporter rows; never interpret a partial grid as a full split")
    args = parser.parse_args()
    result = run_state_dependent_method_execution(case_path=args.case, policy_path=args.policy, proxy_model_path=args.proxy_model, margin_path=args.margin, split_path=args.split, q0_grid_path=args.q0_grid, q95_grid_path=args.q95_grid, output_path=args.output, rho_max=args.rho_max, workers=args.workers, development_only=args.development_only)
    print(canonical_dumps({"status": result["status"], "summary": result["summary"], "result_hash": result["result_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_state_dependent_method_execution", "_validate_unified_margin_candidate", "_screen_q_specific_trust_region", "_trust_feature_values"]
