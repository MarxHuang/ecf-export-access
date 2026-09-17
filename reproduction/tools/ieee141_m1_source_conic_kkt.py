"""Source-only original-unit conic KKT certificate scaffolding for V12.2.

This module is intentionally upstream of trust construction, calibration, AC,
mitigation, and every evidence gate.  It rebuilds the allocation-domain SOCP
from a pinned method-manifold runtime in MW/MVAr/MVA/pu units, exposes every
linear and SOC dual, and evaluates primal feasibility, dual-cone feasibility,
stationarity, and complementarity.

The module does *not* silently choose mutable tie-break or fairness semantics.
Those must arrive in an externally supplied policy mapping.  In particular,
stage-two max-export and fairness remain fail-closed until their exact
in-memory parents and external hash-bound hierarchy/reference proofs are
validated against that static policy.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from r4r.serialization import canonical_hash


SERIALIZATION_ID = "ieee141_m1_source_conic_kkt_scaffold.v2"
SOURCE_TASK_BINDING_SCHEMA_ID = "ieee141_m1_source_conic_kkt_task_binding.v1"
HIERARCHY_ATTESTATION_SCHEMA_ID = "ieee141_m1_source_conic_kkt_hierarchy_attestation.v2"
FAIRNESS_REFERENCE_ATTESTATION_SCHEMA_ID = "ieee141_m1_source_conic_kkt_fairness_reference_attestation.v2"
SOURCE_METHOD_ATOM_BINDING_SCHEMA_ID = "ieee141_m1_source_conic_kkt_method_atom_binding.v1"
OBJECTIVE_MAX_EXPORT_STAGE1 = "MAX_EXPORT_STAGE1"
OBJECTIVE_MAX_EXPORT_TIE_STAGE2 = "MAX_EXPORT_TIE_STAGE2"
OBJECTIVE_FAIRNESS_QP = "FAIRNESS_QP"
OBJECTIVE_KINDS = (
    OBJECTIVE_MAX_EXPORT_STAGE1,
    OBJECTIVE_MAX_EXPORT_TIE_STAGE2,
    OBJECTIVE_FAIRNESS_QP,
)
PARTICIPANT_COUNT = 30
MAX_EXPORT_METHOD_ID = "max_export_lp"
FAIRNESS_METHOD_ID = "fairness_qp"
METHOD_ID_BY_OBJECTIVE = {
    OBJECTIVE_MAX_EXPORT_STAGE1: MAX_EXPORT_METHOD_ID,
    OBJECTIVE_MAX_EXPORT_TIE_STAGE2: MAX_EXPORT_METHOD_ID,
    OBJECTIVE_FAIRNESS_QP: FAIRNESS_METHOD_ID,
}


class SourceConicKKTError(ValueError):
    """Raised when a source-only conic KKT request is malformed."""


def _finite_vector(value: Any, *, name: str, length: int, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SourceConicKKTError(f"{name} must be a {length}-vector")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or (item <= 0.0 if positive else item < 0.0) for item in result):
        raise SourceConicKKTError(f"{name} contains an invalid value")
    return result


def _finite_signed_vector(value: Any, *, name: str, length: int) -> list[float]:
    """Return a finite vector without imposing an allocation-domain sign.

    Objective linear terms are coefficients, not allocations or capacities.
    In particular the registered fairness expansion has negative linear
    coefficients, so it must not be validated through the non-negative
    allocation-vector helper above.
    """
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SourceConicKKTError(f"{name} must be a {length}-vector")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise SourceConicKKTError(f"{name} contains a non-finite value")
    return result


def _hash_bound_mapping(value: Mapping[str, Any], *, field: str = "attestation_hash") -> str:
    body = dict(value)
    declared = body.pop(field, None)
    expected = canonical_hash(body)
    if not isinstance(declared, str) or declared != expected:
        raise SourceConicKKTError(f"{field} does not bind its attestation payload")
    return declared


def _policy_hash(policy: Mapping[str, Any]) -> str:
    body = dict(policy)
    declared = body.pop("policy_hash", None)
    expected = canonical_hash(body)
    if declared is not None and (not isinstance(declared, str) or declared != expected):
        raise SourceConicKKTError("external source conic policy hash is invalid")
    return expected


def _nonnegative_number(value: Any, *, name: str, strictly_positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or (float(value) <= 0.0 if strictly_positive else float(value) < 0.0):
        raise SourceConicKKTError(f"{name} must be a finite {'positive' if strictly_positive else 'non-negative'} number")
    return float(value)


def _finite_number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SourceConicKKTError(f"{name} must be a finite number")
    return float(value)


def validate_external_policy(policy_mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate an externally supplied numerical/method-policy mapping.

    No default tolerances, tie weights, fairness alpha, or epsilon are chosen
    here.  Missing policy is a valid *blocked* state for the public certifier,
    rather than a reason to import a mutable runner constant.

    Required common fields are ``constraint_constants`` (MVA/pu buffers),
    ``original_unit_primal_tolerances`` (MW, MVA, pu, and dimensionless), and
    ``kkt_tolerances`` (dual cone, stationarity, complementarity, and claimed
    execution replay).  A tie-stage policy additionally supplies weights, a
    static hierarchy-tolerance formula, and primary-total replay tolerance.
    A fairness policy
    additionally supplies alpha, epsilon, and the deterministic reference
    tie-break weights.  Result-specific authorizations are deliberately not
    accepted here: every parent result is supplied in memory and validated.
    The caller retains ownership of that static policy; this module only
    hashes and binds the exact mapping it received.
    """
    if not isinstance(policy_mapping, Mapping):
        raise SourceConicKKTError("EXTERNAL_POLICY_MAPPING_REQUIRED")
    policy = dict(policy_mapping)
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise SourceConicKKTError("external source conic policy_id is required")
    tie_policy = policy.get("tie_break")
    fairness_policy = policy.get("fairness")
    if isinstance(tie_policy, Mapping) and "accepted_hierarchy_attestation_hashes" in tie_policy:
        raise SourceConicKKTError("result-dependent hierarchy attestation allowlists are forbidden")
    if isinstance(fairness_policy, Mapping) and "accepted_reference_attestation_hashes" in fairness_policy:
        raise SourceConicKKTError("result-dependent fairness reference allowlists are forbidden")
    constants = policy.get("constraint_constants")
    tolerances = policy.get("kkt_tolerances")
    native_primal = policy.get("original_unit_primal_tolerances")
    if not isinstance(constants, Mapping) or not isinstance(tolerances, Mapping) or not isinstance(native_primal, Mapping):
        raise SourceConicKKTError("external source conic policy constants/native-primal/KKT tolerances are required")
    parsed_constants = {
        "branch_buffer_mva": _nonnegative_number(constants.get("branch_buffer_mva"), name="branch_buffer_mva"),
        "voltage_buffer_pu": _nonnegative_number(constants.get("voltage_buffer_pu"), name="voltage_buffer_pu"),
        "upper_voltage_guard_pu": _nonnegative_number(constants.get("upper_voltage_guard_pu"), name="upper_voltage_guard_pu"),
    }
    parsed_tolerances = {
        "dual_cone_inf": _nonnegative_number(tolerances.get("dual_cone_inf"), name="dual_cone_inf"),
        "stationarity_inf": _nonnegative_number(tolerances.get("stationarity_inf"), name="stationarity_inf"),
        "complementarity_inf": _nonnegative_number(tolerances.get("complementarity_inf"), name="complementarity_inf"),
        "execution_replay_inf_mw": _nonnegative_number(tolerances.get("execution_replay_inf_mw"), name="execution_replay_inf_mw"),
    }
    parsed_native_primal = {
        "allocation_box_mw": _nonnegative_number(native_primal.get("allocation_box_mw"), name="allocation_box_mw"),
        "branch_soc_mva": _nonnegative_number(native_primal.get("branch_soc_mva"), name="branch_soc_mva"),
        "voltage_pu": _nonnegative_number(native_primal.get("voltage_pu"), name="voltage_pu"),
        "rho_dimensionless": _nonnegative_number(native_primal.get("rho_dimensionless"), name="rho_dimensionless"),
    }
    return {
        "policy": policy,
        "policy_hash": _policy_hash(policy),
        "constraint_constants": parsed_constants,
        "kkt_tolerances": parsed_tolerances,
        "original_unit_primal_tolerances": parsed_native_primal,
    }


def _components(margin_q: Mapping[str, Any], key: str, count: int, *, allow_none: bool = False) -> tuple[list[float | None], list[float]]:
    intercept = margin_q.get(f"{key}_effective_intercept")
    slope = margin_q.get(f"{key}_effective_slope")
    if not isinstance(intercept, list) or not isinstance(slope, list) or len(intercept) != count or len(slope) != count:
        raise SourceConicKKTError(f"{key} effective coefficient shape is invalid")
    a: list[float | None] = []
    l: list[float] = []
    for raw_a, raw_l in zip(intercept, slope):
        if raw_a is None and allow_none:
            if raw_l is not None:
                raise SourceConicKKTError(f"{key} null/null policy is invalid")
            a.append(None)
            l.append(0.0)
            continue
        if not isinstance(raw_a, (int, float)) or not math.isfinite(float(raw_a)):
            raise SourceConicKKTError(f"{key} intercept is invalid")
        if not isinstance(raw_l, (int, float)) or not math.isfinite(float(raw_l)) or float(raw_l) < 0.0:
            raise SourceConicKKTError(f"{key} slope is invalid")
        a.append(float(raw_a))
        l.append(float(raw_l))
    return a, l


def _runtime_data(runtime: Mapping[str, Any], capacity_mw: Sequence[float], raw_request_mw: Sequence[float]) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise SourceConicKKTError("runtime is invalid")
    q_mode = runtime.get("q_mode")
    if q_mode not in {"Q0", "Q95"}:
        raise SourceConicKKTError("runtime Q mode is invalid")
    capacity = _finite_vector(capacity_mw, name="capacity_mw", length=PARTICIPANT_COUNT, positive=True)
    raw = _finite_vector(raw_request_mw, name="raw_request_mw", length=PARTICIPANT_COUNT)
    rates_raw = runtime.get("rates")
    if not isinstance(rates_raw, (list, tuple)) or not rates_raw:
        raise SourceConicKKTError("runtime branch rates are invalid")
    rates = [float(value) for value in rates_raw]
    if any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise SourceConicKKTError("runtime branch rates are invalid")
    lower = _nonnegative_number(runtime.get("lower"), name="lower")
    upper = _nonnegative_number(runtime.get("upper"), name="upper")
    if not lower < upper:
        raise SourceConicKKTError("runtime voltage limits are invalid")
    rho_max = _nonnegative_number(runtime.get("rho_max"), name="rho_max", strictly_positive=True)
    model = runtime.get("q_model")
    margin = runtime.get("margin_q")
    if not isinstance(model, Mapping) or not isinstance(margin, Mapping):
        raise SourceConicKKTError("runtime proxy model or margin is invalid")
    base, matrices = model.get("proxy_base"), model.get("derivative_matrices")
    if not isinstance(base, Mapping) or not isinstance(matrices, Mapping):
        raise SourceConicKKTError("runtime proxy affine model is invalid")
    branch_fields = ("branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar")
    for field in branch_fields:
        if not isinstance(base.get(field), list) or not isinstance(matrices.get(field), list) or len(base[field]) != len(rates) or len(matrices[field]) != len(rates):
            raise SourceConicKKTError(f"runtime {field} dimensions are invalid")
        for offset, row in zip(base[field], matrices[field]):
            if not isinstance(row, list) or len(row) != PARTICIPANT_COUNT or not isinstance(offset, (int, float)):
                raise SourceConicKKTError(f"runtime {field} affine values are invalid")
            if not math.isfinite(float(offset)) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in row):
                raise SourceConicKKTError(f"runtime {field} affine values are invalid")
    if not isinstance(base.get("voltage_pu"), list) or not isinstance(matrices.get("voltage_pu"), list) or len(base["voltage_pu"]) != len(matrices["voltage_pu"]):
        raise SourceConicKKTError("runtime voltage dimensions are invalid")
    if not base["voltage_pu"]:
        raise SourceConicKKTError("runtime voltage roster is empty")
    for offset, row in zip(base["voltage_pu"], matrices["voltage_pu"]):
        if not isinstance(row, list) or len(row) != PARTICIPANT_COUNT or not isinstance(offset, (int, float)):
            raise SourceConicKKTError("runtime voltage affine values are invalid")
        if not math.isfinite(float(offset)) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in row):
            raise SourceConicKKTError("runtime voltage affine values are invalid")
    return {
        "q_mode": str(q_mode), "capacity": capacity, "raw": raw, "rates": rates,
        "lower": lower, "upper": upper, "rho_max": rho_max, "model": model, "margin": margin,
        "base": base, "matrices": matrices,
    }


def source_task_binding(
    *,
    runtime: Mapping[str, Any],
    capacity_mw: Sequence[float],
    raw_request_mw: Sequence[float],
) -> dict[str, Any]:
    """Produce the immutable source task identity consumed by proof objects.

    A stage-one hierarchy or fairness reference is only meaningful for the
    exact Q-specific affine proxy, effective-margin snapshot, rate vector,
    request vector, and capacity vector which it was proved for.  This helper
    makes that dependency explicit without reading an execution shard,
    candidate, calibration, AC result, or mitigation outcome.
    """
    data = _runtime_data(runtime, capacity_mw, raw_request_mw)
    constraint_runtime = {
        "q_mode": data["q_mode"],
        "q_model": data["model"],
        "margin_q": data["margin"],
        "rates_mva": data["rates"],
        "voltage_lower_pu": data["lower"],
        "voltage_upper_pu": data["upper"],
        "rho_max": data["rho_max"],
    }
    binding = {
        "schema_id": SOURCE_TASK_BINDING_SCHEMA_ID,
        "q_mode": data["q_mode"],
        "capacity_hash": canonical_hash({"values_mw": data["capacity"]}),
        "raw_request_hash": canonical_hash({"values_mw": data["raw"]}),
        "allocation_upper_hash": canonical_hash(
            {"values_mw": np.minimum(np.asarray(data["capacity"]), np.asarray(data["raw"])).tolist()}
        ),
        "constraint_runtime_hash": canonical_hash(constraint_runtime),
        "envelope_hash": runtime.get("envelope_hash"),
        "proxy_model_hash": runtime.get("proxy_model_hash"),
        "synthetic_rate_policy_hash": runtime.get("synthetic_rate_policy_hash"),
        "effective_eq067_snapshot_hash": runtime.get("effective_eq067_snapshot_hash"),
        "effective_eq067_hash": runtime.get("effective_eq067_hash"),
        "rate_values_hash": runtime.get("rate_values_hash"),
    }
    return {**binding, "source_task_binding_hash": canonical_hash(binding)}


def source_method_atom_binding(*, source_task: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    """Bind a method atom without conflating fairness with max-export.

    Stage one and stage two belong to the same ``max_export_lp`` atom.  The
    fairness QP consumes that atom as a reference but is a distinct
    ``fairness_qp`` atom, even when Q, capacity, raw request, and runtime are
    otherwise identical.
    """
    if method_id not in {MAX_EXPORT_METHOD_ID, FAIRNESS_METHOD_ID}:
        raise SourceConicKKTError("unknown source method atom id")
    required = (
        "source_task_binding_hash",
        "q_mode",
        "capacity_hash",
        "raw_request_hash",
        "constraint_runtime_hash",
    )
    if not isinstance(source_task, Mapping) or any(not isinstance(source_task.get(key), str) or not source_task.get(key) for key in required):
        raise SourceConicKKTError("source task binding is insufficient for method atom binding")
    payload = {
        "schema_id": SOURCE_METHOD_ATOM_BINDING_SCHEMA_ID,
        "method_id": method_id,
        "source_task_binding_hash": source_task["source_task_binding_hash"],
        "q_mode": source_task["q_mode"],
        "capacity_hash": source_task["capacity_hash"],
        "raw_request_hash": source_task["raw_request_hash"],
        "constraint_runtime_hash": source_task["constraint_runtime_hash"],
    }
    return {**payload, "method_atom_binding_hash": canonical_hash(payload)}


def evaluate_original_unit_source_primal(
    *,
    runtime: Mapping[str, Any],
    capacity_mw: Sequence[float],
    raw_request_mw: Sequence[float],
    policy_mapping: Mapping[str, Any],
    allocation_mw: Sequence[float],
) -> dict[str, Any]:
    """Replay source-domain primal residuals in their native physical units.

    The allocation-domain slopes multiplying ``rho`` are non-negative.  Thus
    the minimum admissible ``rho=||x/C||_2`` is the least restrictive value
    for every branch and voltage inequality.  Replaying it directly proves
    whether an externally attested allocation has *some* valid source-domain
    rho, without importing a solver's mutable internal variables.
    """
    policy = validate_external_policy(policy_mapping)
    data = _runtime_data(runtime, capacity_mw, raw_request_mw)
    x = np.asarray(_finite_vector(allocation_mw, name="allocation_mw", length=PARTICIPANT_COUNT), dtype=float)
    capacity = np.asarray(data["capacity"], dtype=float)
    raw = np.asarray(data["raw"], dtype=float)
    constants = policy["constraint_constants"]
    rho = float(np.linalg.norm(x / capacity))
    branch_from_a, branch_from_l = _components(data["margin"], "branch_from_mva", len(data["rates"]))
    branch_to_a, branch_to_l = _components(data["margin"], "branch_to_mva", len(data["rates"]))
    lower_a, lower_l = _components(data["margin"], "voltage_lower_pu", len(data["base"]["voltage_pu"]))
    upper_a, upper_l = _components(data["margin"], "voltage_upper_pu", len(data["base"]["voltage_pu"]), allow_none=True)

    allocation_box_mw = max(
        0.0,
        -float(np.min(x)),
        float(np.max(x - raw)),
        float(np.max(x - capacity)),
    )
    branch_from: list[float] = []
    branch_to: list[float] = []
    for index, rate in enumerate(data["rates"]):
        p_from = float(data["base"]["branch_p_from_mw"][index]) + float(np.asarray(data["matrices"]["branch_p_from_mw"][index], dtype=float) @ x)
        q_from = float(data["base"]["branch_q_from_mvar"][index]) + float(np.asarray(data["matrices"]["branch_q_from_mvar"][index], dtype=float) @ x)
        p_to = float(data["base"]["branch_p_to_mw"][index]) + float(np.asarray(data["matrices"]["branch_p_to_mw"][index], dtype=float) @ x)
        q_to = float(data["base"]["branch_q_to_mvar"][index]) + float(np.asarray(data["matrices"]["branch_q_to_mvar"][index], dtype=float) @ x)
        t_from = float(rate) - constants["branch_buffer_mva"] - float(branch_from_a[index] or 0.0) - float(branch_from_l[index]) * rho
        t_to = float(rate) - constants["branch_buffer_mva"] - float(branch_to_a[index] or 0.0) - float(branch_to_l[index]) * rho
        branch_from.append(max(0.0, math.hypot(p_from, q_from) - t_from))
        branch_to.append(max(0.0, math.hypot(p_to, q_to) - t_to))
    voltage_lower: list[float] = []
    voltage_upper: list[float] = []
    for index, value in enumerate(data["base"]["voltage_pu"]):
        voltage = float(value) + float(np.asarray(data["matrices"]["voltage_pu"][index], dtype=float) @ x)
        voltage_lower.append(
            max(0.0, float(data["lower"]) + constants["voltage_buffer_pu"] + float(lower_a[index] or 0.0) + float(lower_l[index]) * rho - voltage)
        )
        if upper_a[index] is None:
            voltage_upper.append(
                max(0.0, voltage - float(data["upper"]) + constants["upper_voltage_guard_pu"] + constants["voltage_buffer_pu"])
            )
        else:
            voltage_upper.append(
                max(0.0, voltage + float(upper_a[index] or 0.0) + float(upper_l[index]) * rho - float(data["upper"]) + constants["voltage_buffer_pu"])
            )
    residuals = {
        "allocation_box_mw": allocation_box_mw,
        "branch_soc_mva": max([0.0, *branch_from, *branch_to]),
        "voltage_pu": max([0.0, *voltage_lower, *voltage_upper]),
        "rho_dimensionless": max(0.0, rho - float(data["rho_max"])),
    }
    tolerances = policy["original_unit_primal_tolerances"]
    passed = bool(all(residuals[key] <= tolerances[key] for key in residuals))
    return {
        "pass": passed,
        "primal_allocation_mw": x.tolist(),
        "minimal_feasible_rho": rho,
        "residuals": residuals,
        "tolerances": dict(tolerances),
        "branch_from_violation_mva": branch_from,
        "branch_to_violation_mva": branch_to,
        "voltage_lower_violation_pu": voltage_lower,
        "voltage_upper_violation_pu": voltage_upper,
    }


def _result_self_hash(result: Mapping[str, Any]) -> str:
    """Validate a result produced by this module before using it as a parent."""
    body = dict(result)
    declared = body.get("result_hash")
    body["result_hash"] = None
    expected = canonical_hash(body)
    if not isinstance(declared, str) or declared != expected:
        raise SourceConicKKTError("PARENT_RESULT_SELF_HASH_INVALID")
    return declared


def _tie_configuration(policy: Mapping[str, Any], *, constraint_constants: Mapping[str, float]) -> dict[str, Any]:
    tie = policy.get("tie_break")
    if not isinstance(tie, Mapping):
        raise SourceConicKKTError("TIE_BREAK_POLICY_MAPPING_REQUIRED")
    weights = _finite_signed_vector(tie.get("weights"), name="tie_break.weights", length=PARTICIPANT_COUNT)
    objective_id = tie.get("canonical_tie_objective_id")
    objective_form = tie.get("canonical_tie_objective_form")
    centered_expected = [
        (float(index) - (PARTICIPANT_COUNT - 1.0) / 2.0) / (PARTICIPANT_COUNT - 1.0)
        for index in range(PARTICIPANT_COUNT)
    ]
    legacy_expected = [float(index) for index in range(1, PARTICIPANT_COUNT + 1)]
    new_policy = objective_id == "CENTERED_INDEX_WEIGHTED_STRICTLY_CONVEX_CANONICAL_TIE_BREAK_V2"
    legacy_policy = objective_id == "DETERMINISTIC_INDEX_WEIGHTED_STRICTLY_CONVEX_CANONICAL_TIE_BREAK"
    if new_policy:
        if any(abs(left - right) > 1.0e-15 for left, right in zip(weights, centered_expected)):
            raise SourceConicKKTError("TIE_BREAK_CENTERED_WEIGHTS_INVALID")
        if objective_form != "centered_registry_index_weighted_x_plus_eta_over_2_ST_times_l2_x_squared":
            raise SourceConicKKTError("TIE_BREAK_OBJECTIVE_FORM_INVALID")
    elif legacy_policy:
        if weights != legacy_expected:
            raise SourceConicKKTError("TIE_BREAK_CANONICAL_UNSCALED_WEIGHTS_INVALID")
        if objective_form != (
            "sum_i_participant_index_i_times_x_i_plus_kappa_times_S_ref_over_2_times_"
            "sum_i_x_i_over_S_ref_squared_plus_rho_squared"
        ):
            raise SourceConicKKTError("TIE_BREAK_OBJECTIVE_FORM_INVALID")
    else:
        raise SourceConicKKTError("TIE_BREAK_OBJECTIVE_ID_INVALID")
    linear_scale = _nonnegative_number(
        tie.get("canonical_linear_weight_scale"),
        name="tie_break.canonical_linear_weight_scale",
        strictly_positive=True,
    )
    if not math.isclose(linear_scale, 1.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise SourceConicKKTError("TIE_BREAK_LINEAR_WEIGHT_SCALE_INVALID")
    normalized_enclosure_cap = _nonnegative_number(
        tie.get("normalized_decision_l2_optimality_enclosure_max"),
        name="tie_break.normalized_decision_l2_optimality_enclosure_max",
        strictly_positive=True,
    )
    if not math.isclose(normalized_enclosure_cap, 2.0e-3, rel_tol=0.0, abs_tol=1.0e-15):
        raise SourceConicKKTError("TIE_BREAK_NORMALIZED_DECISION_L2_ENCLOSURE_CAP_INVALID")
    if "solver_weight_scale" in tie or "semantic_epsilon_lexicographic_mw" in tie:
        raise SourceConicKKTError("TIE_BREAK_HIDDEN_SCALE_OR_EPSILON_FORBIDDEN")
    quadratic = tie.get("strict_convex_quadratic")
    if not isinstance(quadratic, Mapping):
        raise SourceConicKKTError("TIE_BREAK_STRICT_CONVEX_QUADRATIC_REQUIRED")
    if new_policy:
        eta = _nonnegative_number(
            quadratic.get("eta_dimensionless"),
            name="tie_break.strict_convex_quadratic.eta_dimensionless",
            strictly_positive=True,
        )
        if (
            quadratic.get("normalizer_definition") != "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST"
            or quadratic.get("rho_quadratic") != "FORBIDDEN"
            or quadratic.get("strict_convexity_rule") != "unique_allocation_x_on_nonempty_closed_convex_certified_tau_H_face"
            or quadratic.get("solver_objective_rescaling") != "FORBIDDEN"
            or quadratic.get("numerical_epsilon_tie_break") != "FORBIDDEN"
            or not math.isclose(eta, 1.0e-2, rel_tol=0.0, abs_tol=1.0e-15)
        ):
            raise SourceConicKKTError("TIE_BREAK_STRICT_CONVEX_QUADRATIC_INVALID")
        kappa = None
        normalizer_floor_mw = None
    else:
        kappa = _nonnegative_number(
            quadratic.get("kappa_dimensionless"),
            name="tie_break.strict_convex_quadratic.kappa_dimensionless",
            strictly_positive=True,
        )
        normalizer_floor_mw = _nonnegative_number(
            quadratic.get("normalizer_floor_mw"),
            name="tie_break.strict_convex_quadratic.normalizer_floor_mw",
            strictly_positive=True,
        )
        if (
            quadratic.get("normalizer_definition") != "MAX_OF_SUM_ALLOCATION_BOX_UPPER_MW_AND_1_MW"
            or quadratic.get("strict_convexity_rule") != "unique_x_and_rho_primal_minimizer_on_nonempty_closed_convex_certified_tau_H_face"
            or quadratic.get("solver_objective_rescaling") != "FORBIDDEN"
            or quadratic.get("numerical_epsilon_tie_break") != "FORBIDDEN"
            or not math.isclose(kappa, 1.0, rel_tol=0.0, abs_tol=1.0e-15)
            or not math.isclose(normalizer_floor_mw, 1.0, rel_tol=0.0, abs_tol=1.0e-15)
        ):
            raise SourceConicKKTError("TIE_BREAK_STRICT_CONVEX_QUADRATIC_INVALID")
    hierarchy = tie.get("hierarchy_tolerance")
    if not isinstance(hierarchy, Mapping):
        raise SourceConicKKTError("tie_break.hierarchy_tolerance mapping is required")
    formula_id = hierarchy.get("formula_id", "ABSOLUTE_PLUS_RELATIVE_TOTAL_SCALE")
    if formula_id != "ABSOLUTE_PLUS_RELATIVE_TOTAL_SCALE":
        raise SourceConicKKTError("hierarchy_tolerance formula_id is invalid")
    scale_definition = hierarchy.get("total_scale_definition", "SUM_ALLOCATION_BOX_UPPER_MW")
    if scale_definition != "SUM_ALLOCATION_BOX_UPPER_MW":
        raise SourceConicKKTError("hierarchy_tolerance total scale definition is invalid")
    # Keep the serialized output canonical while accepting the explicit
    # a_H/r_H spellings used by the scientific policy ledger during its
    # transition to this source-only verifier.
    absolute_raw = hierarchy.get("absolute_mw", hierarchy.get("a_H_mw", hierarchy.get("absolute_a_h_mw")))
    relative_raw = hierarchy.get("relative_total_scale", hierarchy.get("r_H", hierarchy.get("relative_r_h")))
    stage1_gap_raw = hierarchy.get(
        "stage1_gap_max",
        hierarchy.get(
            "stage1_gap_max_mw",
            hierarchy.get("tau1_stage1_gap_max_mw", hierarchy.get("stage1_primal_dual_gap_max_mw", tie.get("stage1_dual_gap_tolerance_mw"))),
        ),
    )
    absolute_mw = _nonnegative_number(absolute_raw, name="tie_break.hierarchy_tolerance.absolute_mw")
    relative_total_scale = _nonnegative_number(relative_raw, name="tie_break.hierarchy_tolerance.relative_total_scale")
    stage1_gap_max = _nonnegative_number(stage1_gap_raw, name="tie_break.hierarchy_tolerance.stage1_gap_max")
    primary_total_tolerance = _nonnegative_number(tie.get("primary_total_replay_tolerance_mw"), name="tie_break.primary_total_replay_tolerance_mw")
    objective_tolerance = _nonnegative_number(tie.get("tie_objective_replay_tolerance_native_units"), name="tie_break.tie_objective_replay_tolerance_native_units")
    quadratic_binding = {
        "objective_form": str(objective_form),
        **({
            "eta_dimensionless": eta,
            "normalizer_definition": "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST",
            "rho_quadratic": "FORBIDDEN",
            "strict_convexity_rule": "unique_allocation_x_on_nonempty_closed_convex_certified_tau_H_face",
        } if new_policy else {
            "kappa_dimensionless": kappa,
            "normalizer_definition": "MAX_OF_SUM_ALLOCATION_BOX_UPPER_MW_AND_1_MW",
            "normalizer_floor_mw": normalizer_floor_mw,
            "strict_convexity_rule": "unique_x_and_rho_primal_minimizer_on_nonempty_closed_convex_certified_tau_H_face",
        }),
        "solver_objective_rescaling": "FORBIDDEN",
        "numerical_epsilon_tie_break": "FORBIDDEN",
    }
    binding_hash = canonical_hash(
        {
            "objective_kind": OBJECTIVE_MAX_EXPORT_TIE_STAGE2,
            "constraint_constants": dict(constraint_constants),
            "tie_weights": weights,
            "canonical_linear_weight_scale": linear_scale,
            "normalized_decision_l2_optimality_enclosure_max": normalized_enclosure_cap,
            "strict_convex_quadratic": quadratic_binding,
            "hierarchy_tolerance": {
                "formula_id": "ABSOLUTE_PLUS_RELATIVE_TOTAL_SCALE",
                "total_scale_definition": "SUM_ALLOCATION_BOX_UPPER_MW",
                "absolute_mw": absolute_mw,
                "relative_total_scale": relative_total_scale,
                "stage1_gap_max": stage1_gap_max,
            },
        }
    )
    return {
        "weights": weights,
        "canonical_linear_weight_scale": linear_scale,
        "normalized_decision_l2_optimality_enclosure_max": normalized_enclosure_cap,
        "canonical_tie_objective_id": str(objective_id),
        "canonical_tie_objective_form": str(objective_form),
        "strict_convex_quadratic": quadratic_binding,
        "hierarchy_tolerance": {
            "formula_id": "ABSOLUTE_PLUS_RELATIVE_TOTAL_SCALE",
            "total_scale_definition": "SUM_ALLOCATION_BOX_UPPER_MW",
            "absolute_mw": absolute_mw,
            "relative_total_scale": relative_total_scale,
            "stage1_gap_max": stage1_gap_max,
        },
        "primary_total_replay_tolerance_mw": primary_total_tolerance,
        "tie_objective_replay_tolerance_native_units": objective_tolerance,
        "formulation_policy_binding_hash": binding_hash,
    }


def _hierarchy_tolerance_for_task(tie: Mapping[str, Any], *, runtime_data: Mapping[str, Any]) -> dict[str, float | str]:
    """Evaluate static ``tau_H=a_H+r_H*S_T`` for one immutable source task."""
    hierarchy = tie["hierarchy_tolerance"]
    capacity = np.asarray(runtime_data["capacity"], dtype=float)
    raw = np.asarray(runtime_data["raw"], dtype=float)
    total_scale = float(math.fsum(np.minimum(capacity, raw).tolist()))
    if not math.isfinite(total_scale) or total_scale < 0.0:
        raise SourceConicKKTError("hierarchy total scale is invalid")
    tau_h = float(hierarchy["absolute_mw"]) + float(hierarchy["relative_total_scale"]) * total_scale
    if not math.isfinite(tau_h) or tau_h < 0.0:
        raise SourceConicKKTError("hierarchy tolerance is invalid")
    return {
        "formula_id": str(hierarchy["formula_id"]),
        "total_scale_definition": str(hierarchy["total_scale_definition"]),
        "total_scale_st_mw": total_scale,
        "tau_h_mw": tau_h,
        "stage1_gap_max_mw": float(hierarchy["stage1_gap_max"]),
    }


def _tie_regularization_for_task(
    tie: Mapping[str, Any],
    *,
    runtime_data: Mapping[str, Any],
) -> dict[str, float | str]:
    """Evaluate the frozen Stage-2 regularizer for one numerical atom.

    The active V12.3 policy is ``wbar'x + eta/(2*S_T)||x||^2`` with the exact
    pre-solve scale ``S_T=sum_i min(C_i,r_i)``.  The historical V12.2 policy
    is retained only for replay of explicitly historical artifacts; it also
    has a separate rho quadratic.  Neither branch is a solver scaling or an
    epsilon perturbation.
    """
    quadratic = tie.get("strict_convex_quadratic")
    if not isinstance(quadratic, Mapping):
        raise SourceConicKKTError("TIE_BREAK_STRICT_CONVEX_QUADRATIC_REQUIRED")
    capacity = np.asarray(runtime_data["capacity"], dtype=float)
    raw = np.asarray(runtime_data["raw"], dtype=float)
    total_scale = float(math.fsum(np.minimum(capacity, raw).tolist()))
    if not math.isfinite(total_scale) or total_scale < 0.0:
        raise SourceConicKKTError("tie regularizer total scale is invalid")
    if quadratic.get("normalizer_definition") == "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST":
        eta = _nonnegative_number(
            quadratic.get("eta_dimensionless"),
            name="tie regularizer eta",
            strictly_positive=True,
        )
        if quadratic.get("rho_quadratic") != "FORBIDDEN" or total_scale <= 0.0:
            raise SourceConicKKTError("active tie regularizer scale or rho policy is invalid")
        x_qdiag = eta / (2.0 * total_scale)
        if not math.isfinite(x_qdiag) or x_qdiag <= 0.0:
            raise SourceConicKKTError("active tie regularizer coefficient is invalid")
        return {
            "objective_form": "centered_registry_index_weighted_x_plus_eta_over_2_ST_times_l2_x_squared",
            "eta_dimensionless": eta,
            "normalizer_definition": "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST",
            "allocation_upper_total_mw": total_scale,
            "normalizer_mw": total_scale,
            "x_quadratic_coefficient_mw_inverse": x_qdiag,
            "rho_quadratic_coefficient_mw": 0.0,
            "x_strong_convexity_modulus_mw_inverse": 2.0 * x_qdiag,
            "rho_strong_convexity_modulus_mw": 0.0,
            "normalized_decision_strong_convexity_modulus_mw": eta / total_scale,
            "rho_quadratic": "FORBIDDEN",
        }
    kappa = _nonnegative_number(
        quadratic.get("kappa_dimensionless"),
        name="tie regularizer kappa",
        strictly_positive=True,
    )
    floor = _nonnegative_number(
        quadratic.get("normalizer_floor_mw"),
        name="tie regularizer normalizer floor",
        strictly_positive=True,
    )
    if quadratic.get("normalizer_definition") != "MAX_OF_SUM_ALLOCATION_BOX_UPPER_MW_AND_1_MW":
        raise SourceConicKKTError("tie regularizer normalizer definition is invalid")
    normalizer = max(total_scale, floor)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise SourceConicKKTError("tie regularizer normalizer is invalid")
    x_qdiag = kappa / (2.0 * normalizer)
    rho_qdiag = kappa * normalizer / 2.0
    if not math.isfinite(x_qdiag) or x_qdiag <= 0.0 or not math.isfinite(rho_qdiag) or rho_qdiag <= 0.0:
        raise SourceConicKKTError("tie regularizer coefficients are invalid")
    return {
        "objective_form": "sum_i_participant_index_i_times_x_i_plus_kappa_times_S_ref_over_2_times_sum_i_x_i_over_S_ref_squared_plus_rho_squared",
        "kappa_dimensionless": kappa,
        "normalizer_definition": "MAX_OF_SUM_ALLOCATION_BOX_UPPER_MW_AND_1_MW",
        "normalizer_floor_mw": floor,
        "allocation_upper_total_mw": total_scale,
        "normalizer_mw": normalizer,
        "x_quadratic_coefficient_mw_inverse": x_qdiag,
        "rho_quadratic_coefficient_mw": rho_qdiag,
        "x_strong_convexity_modulus_mw_inverse": 2.0 * x_qdiag,
        "rho_strong_convexity_modulus_mw": 2.0 * rho_qdiag,
        "normalized_decision_strong_convexity_modulus_mw": kappa * normalizer,
    }


def _stage1_primal_dual_certificate(
    *,
    kkt: Mapping[str, Any],
    objective_value: float | None,
    hierarchy_tolerance: Mapping[str, Any],
    primary_total_replay_tolerance_mw: float,
) -> dict[str, Any]:
    """Check ``P1 <= T* <= U1`` using an independently reconstructed dual."""
    dual = kkt.get("dual_objective")
    if not isinstance(dual, Mapping) or dual.get("available") is not True:
        return {"pass": False, "status": "STAGE1_DUAL_OBJECTIVE_UNAVAILABLE"}
    try:
        allocation = _finite_vector(kkt.get("primal_allocation_mw"), name="stage1 primal allocation", length=PARTICIPANT_COUNT)
        p1 = float(math.fsum(allocation))
        u1 = _finite_number(dual.get("max_export_certified_upper_u1_mw"), name="stage1 U1")
        lower_before_guard = _finite_number(
            dual.get("minimization_dual_lower_bound_before_numerical_guard_native_units"),
            name="stage1 dual lower bound before numerical guard",
        )
        dual_guard = _nonnegative_number(
            dual.get("dual_lower_bound_numerical_safety_margin_native_units"),
            name="stage1 dual numerical safety guard",
        )
        lower_after_guard = _finite_number(
            dual.get("minimization_dual_lower_bound_native_units"),
            name="stage1 dual lower bound after numerical guard",
        )
        recorded_p1 = _finite_number(dual.get("max_export_primal_total_p1_mw"), name="stage1 recorded P1")
        recorded_gap = _finite_number(dual.get("u1_minus_p1_mw"), name="stage1 recorded U1-P1")
        tau1 = _nonnegative_number(hierarchy_tolerance.get("stage1_gap_max_mw"), name="stage1_gap_max_mw")
        tau_h = _nonnegative_number(hierarchy_tolerance.get("tau_h_mw"), name="tau_h_mw")
    except SourceConicKKTError as exc:
        return {"pass": False, "status": "STAGE1_DUAL_OBJECTIVE_INVALID", "failure_reason": str(exc)}
    objective_replay_gap = math.inf if objective_value is None else abs(float(objective_value) + p1)
    recorded_replay_gap = abs(recorded_p1 - p1)
    gap = u1 - p1
    ordering_violation = max(0.0, p1 - u1)
    try:
        native_residuals = kkt["residuals"]["primal_by_native_unit"]
        native_tolerances = kkt["tolerances"]["original_unit_primal"]
        primal_feasible = bool(
            _nonnegative_number(native_residuals.get("MW"), name="stage1 primal MW residual")
            <= _nonnegative_number(native_tolerances.get("allocation_box_mw"), name="stage1 primal MW tolerance")
            and _nonnegative_number(native_residuals.get("MVA"), name="stage1 primal MVA residual")
            <= _nonnegative_number(native_tolerances.get("branch_soc_mva"), name="stage1 primal MVA tolerance")
            and _nonnegative_number(native_residuals.get("pu"), name="stage1 primal pu residual")
            <= _nonnegative_number(native_tolerances.get("voltage_pu"), name="stage1 primal pu tolerance")
            and _nonnegative_number(native_residuals.get("dimensionless"), name="stage1 primal rho residual")
            <= _nonnegative_number(native_tolerances.get("rho_dimensionless"), name="stage1 primal rho tolerance")
        )
    except (KeyError, TypeError, SourceConicKKTError):
        primal_feasible = False
    dual_reconstruction_replay_gap = abs(lower_after_guard - (lower_before_guard - dual_guard))
    upper_replay_gap = abs(u1 + lower_after_guard)
    pass_value = bool(
        primal_feasible
        and
        objective_replay_gap <= primary_total_replay_tolerance_mw
        and recorded_replay_gap <= primary_total_replay_tolerance_mw
        and abs(recorded_gap - gap) <= primary_total_replay_tolerance_mw
        and dual_reconstruction_replay_gap <= primary_total_replay_tolerance_mw
        and upper_replay_gap <= primary_total_replay_tolerance_mw
        and ordering_violation <= primary_total_replay_tolerance_mw
        and gap <= tau1
        and tau_h + primary_total_replay_tolerance_mw >= gap
    )
    return {
        "pass": pass_value,
        "status": "P1_TSTAR_U1_CERTIFIED" if pass_value else "P1_TSTAR_U1_CERTIFICATE_FAIL_CLOSED",
        "p1_primal_total_mw": p1,
        "u1_certified_upper_mw": u1,
        "t_star_interval_mw": {"lower_from_primal_p1_mw": p1, "upper_from_dual_u1_mw": u1},
        "u1_minus_p1_mw": gap,
        "p1_exceeds_u1_violation_mw": ordering_violation,
        "p1_source_primal_feasible": primal_feasible,
        "minimization_dual_lower_bound_before_numerical_guard_native_units": lower_before_guard,
        "dual_lower_bound_numerical_safety_margin_native_units": dual_guard,
        "minimization_dual_lower_bound_native_units": lower_after_guard,
        "dual_reconstruction_replay_gap_native_units": dual_reconstruction_replay_gap,
        "max_export_upper_replay_gap_mw": upper_replay_gap,
        "tau1_stage1_gap_max_mw": tau1,
        "tau_h_mw": tau_h,
        "tau_h_covers_observed_stage1_gap": tau_h + primary_total_replay_tolerance_mw >= gap,
        "objective_replay_gap_mw": objective_replay_gap,
        "recorded_p1_replay_gap_mw": recorded_replay_gap,
        "recorded_gap_replay_gap_mw": abs(recorded_gap - gap),
        "total_scale_st_mw": hierarchy_tolerance.get("total_scale_st_mw"),
        "hierarchy_tolerance_formula_id": hierarchy_tolerance.get("formula_id"),
        "hierarchy_tolerance_total_scale_definition": hierarchy_tolerance.get("total_scale_definition"),
    }


def _minimization_primal_dual_certificate(
    *,
    kkt: Mapping[str, Any],
    solver_objective_value: float | None,
    objective_kind: str,
    objective_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify a minimization objective by an explicit projected-cone dual.

    ``MAX_EXPORT_TIE_STAGE2`` is the registered strictly-convex canonical QP
    on the certified ``tau_H`` face.  ``FAIRNESS_QP`` is the convex
    minimization form of the fairness objective.  Both therefore use a global
    *lower* bound reconstructed from the original-unit Lagrangian and never
    interpret a solver-reported primal objective as a global bound.
    """
    if objective_kind not in {OBJECTIVE_MAX_EXPORT_TIE_STAGE2, OBJECTIVE_FAIRNESS_QP}:
        return {"pass": False, "status": "MINIMIZATION_CERTIFICATE_OBJECTIVE_KIND_INVALID"}
    dual = kkt.get("dual_objective")
    if not isinstance(dual, Mapping) or dual.get("available") is not True:
        return {"pass": False, "status": "MINIMIZATION_DUAL_OBJECTIVE_UNAVAILABLE"}
    bound_key = (
        "tie_objective_certified_lower_bound_native_units"
        if objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2
        else "fairness_qp_certified_lower_bound_native_units"
    )
    try:
        primal = _finite_number(kkt.get("primal_objective_native_units"), name="minimization primal objective")
        lower = _finite_number(dual.get(bound_key), name="minimization dual lower bound")
        lower_before_guard = _finite_number(
            dual.get("minimization_dual_lower_bound_before_numerical_guard_native_units"),
            name="minimization dual lower bound before numerical guard",
        )
        guard = _nonnegative_number(
            dual.get("dual_lower_bound_numerical_safety_margin_native_units"),
            name="minimization dual numerical safety guard",
        )
        lower_after_guard = _finite_number(
            dual.get("minimization_dual_lower_bound_native_units"),
            name="minimization dual lower bound after numerical guard",
        )
    except SourceConicKKTError as exc:
        return {"pass": False, "status": "MINIMIZATION_DUAL_OBJECTIVE_INVALID", "failure_reason": str(exc)}
    lower_replay_gap = abs(lower_after_guard - (lower_before_guard - guard))
    bound_replay_gap = abs(lower - lower_after_guard)
    ordering_violation = max(0.0, lower - primal)
    gap = primal - lower
    solver_replay_gap = math.inf if solver_objective_value is None else abs(float(solver_objective_value) - primal)
    passed = bool(
        lower_replay_gap == 0.0
        and bound_replay_gap == 0.0
        and ordering_violation == 0.0
        and math.isfinite(gap)
        and gap >= 0.0
    )
    result = {
        "pass": passed,
        "status": "MINIMIZATION_PRIMAL_DUAL_CERTIFIED" if passed else "MINIMIZATION_PRIMAL_DUAL_CERTIFICATE_FAIL_CLOSED",
        "objective_kind": objective_kind,
        "objective_sense": "MINIMIZE",
        "certified_objective_bound_orientation": "GLOBAL_DUAL_LOWER_BOUND",
        "objective_gap_orientation": "PRIMAL_MINUS_GLOBAL_DUAL_LOWER_BOUND",
        "primal_objective_value_native_units": primal,
        "certified_objective_bound_native_units": lower,
        "objective_gap_native_units": gap,
        "dual_lower_exceeds_primal_violation_native_units": ordering_violation,
        "minimization_dual_lower_bound_before_numerical_guard_native_units": lower_before_guard,
        "dual_lower_bound_numerical_safety_margin_native_units": guard,
        "minimization_dual_lower_bound_native_units": lower_after_guard,
        "dual_lower_bound_replay_gap_native_units": lower_replay_gap,
        "bound_field_replay_gap_native_units": bound_replay_gap,
        "solver_reported_objective_value_native_units": solver_objective_value,
        "solver_reported_objective_replay_gap_native_units": solver_replay_gap,
    }
    if objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        try:
            gap_cap = _nonnegative_number(
                objective_metadata.get("objective_gap_acceptance_cap_native_units"),
                name="tie objective gap acceptance cap",
                strictly_positive=True,
            )
            enclosure_cap = _nonnegative_number(
                objective_metadata.get("normalized_decision_l2_optimality_enclosure_max"),
                name="tie normalized decision enclosure cap",
                strictly_positive=True,
            )
            normalized_enclosure = _nonnegative_number(
                dual.get("tie_objective_normalized_decision_l2_optimality_enclosure"),
                name="tie normalized decision enclosure",
            )
            result.update(
                {
                    "canonical_linear_index_component_native_units": _finite_number(
                        dual.get("tie_objective_linear_index_component_native_units"),
                        name="tie canonical linear component",
                    ),
                    "strict_convex_quadratic_component_native_units": _finite_number(
                        dual.get("tie_objective_strict_convex_quadratic_component_native_units"),
                        name="tie strict convex quadratic component",
                    ),
                    "allocation_l2_optimality_enclosure_mw": _nonnegative_number(
                        dual.get("tie_objective_allocation_l2_optimality_enclosure_mw"),
                        name="tie allocation l2 enclosure",
                    ),
                    "rho_optimality_enclosure_dimensionless": _nonnegative_number(
                        dual.get("tie_objective_rho_optimality_enclosure_dimensionless"),
                        name="tie rho enclosure",
                    ),
                    "normalized_decision_l2_optimality_enclosure": normalized_enclosure,
                    "objective_gap_acceptance_cap_native_units": gap_cap,
                    "normalized_decision_l2_optimality_enclosure_max": enclosure_cap,
                    "objective_gap_within_acceptance_cap": gap <= gap_cap,
                    "normalized_decision_l2_enclosure_within_acceptance_cap": normalized_enclosure <= enclosure_cap,
                }
            )
            result["pass"] = bool(
                result["pass"]
                and gap <= gap_cap
                and normalized_enclosure <= enclosure_cap
            )
            result["status"] = (
                "MINIMIZATION_PRIMAL_DUAL_CERTIFIED"
                if result["pass"]
                else "MINIMIZATION_PRIMAL_DUAL_CERTIFICATE_FAIL_CLOSED"
            )
        except SourceConicKKTError as exc:
            return {
                **result,
                "pass": False,
                "status": "MINIMIZATION_PRIMAL_DUAL_CERTIFICATE_FAIL_CLOSED",
                "failure_reason": str(exc),
            }
    return result


def _fairness_configuration(policy: Mapping[str, Any], *, constraint_constants: Mapping[str, float]) -> dict[str, Any]:
    fairness = policy.get("fairness")
    if not isinstance(fairness, Mapping):
        raise SourceConicKKTError("FAIRNESS_POLICY_MAPPING_REQUIRED")
    alpha = _nonnegative_number(fairness.get("alpha"), name="fairness.alpha", strictly_positive=True)
    epsilon = _nonnegative_number(fairness.get("epsilon_mw"), name="fairness.epsilon_mw", strictly_positive=True)
    binding_hash = canonical_hash(
        {
            "objective_kind": OBJECTIVE_FAIRNESS_QP,
            "constraint_constants": dict(constraint_constants),
            "alpha": alpha,
            "epsilon_mw": epsilon,
        }
    )
    return {"alpha": alpha, "epsilon_mw": epsilon, "formulation_policy_binding_hash": binding_hash}


def _parent_result_common(
    parent: Mapping[str, Any] | None,
    *,
    expected_objective_kind: str,
    expected_method_id: str,
    expected_task_binding: Mapping[str, Any],
    expected_policy_hash: str,
) -> dict[str, Any]:
    if not isinstance(parent, Mapping):
        raise SourceConicKKTError("PARENT_RESULT_REQUIRED")
    result_hash = _result_self_hash(parent)
    if parent.get("serialization_id") != SERIALIZATION_ID:
        raise SourceConicKKTError("PARENT_RESULT_SERIALIZATION_INVALID")
    if parent.get("status") != "NUMERICALLY_CERTIFIED_CONIC_PROGRAM":
        raise SourceConicKKTError("PARENT_RESULT_NOT_NUMERICALLY_CERTIFIED")
    if parent.get("objective_kind") != expected_objective_kind:
        raise SourceConicKKTError("PARENT_RESULT_OBJECTIVE_KIND_MISMATCH")
    if parent.get("method_id") != expected_method_id:
        raise SourceConicKKTError("PARENT_RESULT_METHOD_ID_MISMATCH")
    if parent.get("solver_status") != "optimal":
        raise SourceConicKKTError("PARENT_RESULT_SOLVER_STATUS_INVALID")
    if parent.get("policy_hash") != expected_policy_hash:
        raise SourceConicKKTError("PARENT_RESULT_POLICY_BINDING_MISMATCH")
    if parent.get("source_only") is not True or parent.get("AC_not_run") is not True or parent.get("candidate_or_evidence_eligible") is not False:
        raise SourceConicKKTError("PARENT_RESULT_SCOPE_INVALID")
    source_binding = parent.get("source_task_binding")
    if not isinstance(source_binding, Mapping) or canonical_hash(source_binding) != canonical_hash(expected_task_binding):
        raise SourceConicKKTError("PARENT_RESULT_SOURCE_TASK_BINDING_MISMATCH")
    if source_binding.get("source_task_binding_hash") != expected_task_binding.get("source_task_binding_hash"):
        raise SourceConicKKTError("PARENT_RESULT_SOURCE_TASK_SELF_BINDING_INVALID")
    expected_atom_binding = source_method_atom_binding(source_task=expected_task_binding, method_id=expected_method_id)
    atom_binding = parent.get("method_atom_binding")
    if not isinstance(atom_binding, Mapping) or canonical_hash(atom_binding) != canonical_hash(expected_atom_binding):
        raise SourceConicKKTError("PARENT_RESULT_METHOD_ATOM_BINDING_MISMATCH")
    if parent.get("method_atom_binding_hash") != expected_atom_binding["method_atom_binding_hash"]:
        raise SourceConicKKTError("PARENT_RESULT_METHOD_ATOM_HASH_MISMATCH")
    solver_adapter = parent.get("solver_adapter")
    solver_settings = parent.get("solver_settings")
    if not isinstance(solver_adapter, Mapping) or not isinstance(solver_adapter.get("id"), str) or not solver_adapter["id"] or not isinstance(solver_adapter.get("version"), str) or not solver_adapter["version"]:
        raise SourceConicKKTError("PARENT_RESULT_SOLVER_ADAPTER_INVALID")
    if not isinstance(solver_settings, Mapping) or parent.get("solver_settings_hash") != canonical_hash(solver_settings):
        raise SourceConicKKTError("PARENT_RESULT_SOLVER_SETTINGS_BINDING_INVALID")
    if not isinstance(parent.get("solver_attempts"), list) or not parent["solver_attempts"]:
        raise SourceConicKKTError("PARENT_RESULT_SOLVER_ATTEMPTS_MISSING")
    kkt = parent.get("kkt")
    if not isinstance(kkt, Mapping) or kkt.get("pass") is not True:
        raise SourceConicKKTError("PARENT_RESULT_KKT_NOT_CERTIFIED")
    allocation = _finite_vector(kkt.get("primal_allocation_mw"), name="parent_result.kkt.primal_allocation_mw", length=PARTICIPANT_COUNT)
    objective_value = _finite_number(parent.get("objective_value"), name="parent_result.objective_value")
    metadata = parent.get("objective_metadata")
    if not isinstance(metadata, Mapping):
        raise SourceConicKKTError("PARENT_RESULT_OBJECTIVE_METADATA_INVALID")
    return {
        "result_hash": result_hash,
        "allocation_mw": allocation,
        "allocation_hash": canonical_hash({"values_mw": allocation}),
        "objective_value": objective_value,
        "objective_metadata": dict(metadata),
        "method_atom_binding_hash": expected_atom_binding["method_atom_binding_hash"],
    }


def _validated_stage1_result(
    stage1_result: Mapping[str, Any] | None,
    *,
    task_binding: Mapping[str, Any],
    policy_hash: str,
    tie: Mapping[str, Any],
    hierarchy_tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _parent_result_common(
        stage1_result,
        expected_objective_kind=OBJECTIVE_MAX_EXPORT_STAGE1,
        expected_method_id=MAX_EXPORT_METHOD_ID,
        expected_task_binding=task_binding,
        expected_policy_hash=policy_hash,
    )
    if summary["objective_metadata"].get("sense") != "MAXIMIZE_TOTAL_EXPORT":
        raise SourceConicKKTError("STAGE1_PARENT_OBJECTIVE_SEMANTICS_INVALID")
    certificate = stage1_result.get("stage1_primal_dual_certificate") if isinstance(stage1_result, Mapping) else None
    if not isinstance(certificate, Mapping):
        raise SourceConicKKTError("STAGE1_PARENT_PRIMAL_DUAL_CERTIFICATE_MISSING")
    replayed = _stage1_primal_dual_certificate(
        kkt=stage1_result["kkt"],
        objective_value=summary["objective_value"],
        hierarchy_tolerance=hierarchy_tolerance,
        primary_total_replay_tolerance_mw=float(tie["primary_total_replay_tolerance_mw"]),
    )
    if replayed.get("pass") is not True:
        raise SourceConicKKTError(f"STAGE1_PARENT_PRIMAL_DUAL_CERTIFICATE_INVALID:{replayed.get('status')}")
    if certificate.get("pass") is not True:
        raise SourceConicKKTError("STAGE1_PARENT_PRIMAL_DUAL_CERTIFICATE_NOT_CERTIFIED")
    # The parent is self-hashed, but a self-hash is not an authorization to
    # replace just one reported dual quantity.  Replaying the complete bound
    # object from its KKT payload makes every recorded P1/U1/tolerance field
    # agree with the source-only reconstruction.
    if canonical_hash(dict(certificate)) != canonical_hash(replayed):
        raise SourceConicKKTError("STAGE1_PARENT_PRIMAL_DUAL_CERTIFICATE_FIELD_MISMATCH")
    return {
        **summary,
        "p1_primal_total_mw": float(replayed["p1_primal_total_mw"]),
        "u1_certified_upper_mw": float(replayed["u1_certified_upper_mw"]),
        "stage1_dual_gap_mw": float(replayed["u1_minus_p1_mw"]),
        "tau1_stage1_gap_max_mw": float(replayed["tau1_stage1_gap_max_mw"]),
        "tau_h_mw": float(replayed["tau_h_mw"]),
        "total_scale_st_mw": float(replayed["total_scale_st_mw"]),
        "hierarchy_tolerance_formula_id": str(replayed["hierarchy_tolerance_formula_id"]),
        "hierarchy_tolerance_total_scale_definition": str(replayed["hierarchy_tolerance_total_scale_definition"]),
    }


def _validated_stage2_result(
    stage2_result: Mapping[str, Any] | None,
    *,
    task_binding: Mapping[str, Any],
    policy_hash: str,
    tie: Mapping[str, Any],
    stage1: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _parent_result_common(
        stage2_result,
        expected_objective_kind=OBJECTIVE_MAX_EXPORT_TIE_STAGE2,
        expected_method_id=MAX_EXPORT_METHOD_ID,
        expected_task_binding=task_binding,
        expected_policy_hash=policy_hash,
    )
    metadata = summary["objective_metadata"]
    if metadata.get("sense") != "MINIMIZE_CERTIFIED_TAU_H_CANONICAL_STRICTLY_CONVEX_TIE_BREAK":
        raise SourceConicKKTError("STAGE2_PARENT_OBJECTIVE_SEMANTICS_INVALID")
    if metadata.get("canonical_tie_policy_binding_hash") != tie["formulation_policy_binding_hash"]:
        raise SourceConicKKTError("STAGE2_PARENT_CANONICAL_TIE_POLICY_BINDING_MISMATCH")
    weights = _finite_signed_vector(metadata.get("weights"), name="parent_stage2.tie_weights", length=PARTICIPANT_COUNT)
    if weights != list(tie["weights"]):
        raise SourceConicKKTError("STAGE2_PARENT_TIE_WEIGHTS_MISMATCH")
    if not math.isclose(
        _nonnegative_number(metadata.get("linear_weight_scale"), name="parent_stage2.linear_weight_scale", strictly_positive=True),
        float(tie["canonical_linear_weight_scale"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise SourceConicKKTError("STAGE2_PARENT_LINEAR_WEIGHT_SCALE_MISMATCH")
    if not math.isclose(
        _nonnegative_number(
            metadata.get("normalized_decision_l2_optimality_enclosure_max"),
            name="parent_stage2.normalized_decision_l2_optimality_enclosure_max",
            strictly_positive=True,
        ),
        float(tie["normalized_decision_l2_optimality_enclosure_max"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise SourceConicKKTError("STAGE2_PARENT_NORMALIZED_DECISION_ENCLOSURE_CAP_MISMATCH")
    regularization = metadata.get("strict_convex_regularization")
    if not isinstance(regularization, Mapping):
        raise SourceConicKKTError("STAGE2_PARENT_STRICT_CONVEX_REGULARIZATION_MISSING")
    expected_quadratic = tie["strict_convex_quadratic"]
    for key in ("kappa_dimensionless", "normalizer_definition", "normalizer_floor_mw"):
        if regularization.get(key) != expected_quadratic.get(key):
            raise SourceConicKKTError(f"STAGE2_PARENT_STRICT_CONVEX_REGULARIZATION_MISMATCH:{key}")
    qdiag = _finite_vector(metadata.get("qdiag"), name="parent_stage2.qdiag", length=PARTICIPANT_COUNT, positive=True)
    rho_qdiag = _nonnegative_number(metadata.get("rho_qdiag"), name="parent_stage2.rho_qdiag", strictly_positive=True)
    x_qdiag = _nonnegative_number(
        regularization.get("x_quadratic_coefficient_mw_inverse"),
        name="parent_stage2.x_quadratic_coefficient",
        strictly_positive=True,
    )
    expected_rho_qdiag = _nonnegative_number(
        regularization.get("rho_quadratic_coefficient_mw"),
        name="parent_stage2.rho_quadratic_coefficient",
        strictly_positive=True,
    )
    if any(abs(value - x_qdiag) > 1.0e-15 for value in qdiag) or abs(rho_qdiag - expected_rho_qdiag) > 1.0e-15:
        raise SourceConicKKTError("STAGE2_PARENT_STRICT_CONVEX_COEFFICIENT_MISMATCH")
    allocation_array = np.asarray(summary["allocation_mw"], dtype=float)
    rho = _nonnegative_number(stage2_result.get("kkt", {}).get("primal_rho") if isinstance(stage2_result, Mapping) and isinstance(stage2_result.get("kkt"), Mapping) else None, name="parent_stage2.primal_rho")
    objective_replay = float(
        np.asarray(weights, dtype=float) @ allocation_array
        + np.asarray(qdiag, dtype=float) @ np.square(allocation_array)
        + rho_qdiag * rho * rho
    )
    if abs(summary["objective_value"] - objective_replay) > float(tie["tie_objective_replay_tolerance_native_units"]):
        raise SourceConicKKTError("STAGE2_PARENT_OBJECTIVE_REPLAY_MISMATCH")
    hierarchy = stage2_result.get("hierarchy_proof") if isinstance(stage2_result, Mapping) else None
    if not isinstance(hierarchy, Mapping):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_MISSING")
    if hierarchy.get("source_task_binding_hash") != task_binding.get("source_task_binding_hash"):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_SOURCE_BINDING_MISMATCH")
    if hierarchy.get("formulation_policy_binding_hash") != tie["formulation_policy_binding_hash"]:
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_POLICY_BINDING_MISMATCH")
    p1 = _nonnegative_number(hierarchy.get("p1_primal_total_mw"), name="parent_stage2.p1_primal_total_mw")
    u1 = _nonnegative_number(hierarchy.get("u1_certified_upper_mw"), name="parent_stage2.u1_certified_upper_mw")
    stage1_gap = _nonnegative_number(hierarchy.get("stage1_dual_gap_mw"), name="parent_stage2.stage1_dual_gap_mw")
    tau_h = _nonnegative_number(hierarchy.get("tau_h_mw"), name="parent_stage2.tau_h_mw")
    if hierarchy.get("primary_face_constraint") != "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H":
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_PRIMARY_FACE_SEMANTICS_MISMATCH")
    if hierarchy.get("hierarchy_tolerance_formula_id") != stage1["hierarchy_tolerance_formula_id"]:
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_TOLERANCE_FORMULA_MISMATCH")
    if hierarchy.get("hierarchy_tolerance_total_scale_definition") != stage1["hierarchy_tolerance_total_scale_definition"]:
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_TOTAL_SCALE_DEFINITION_MISMATCH")
    if abs(p1 - float(stage1["p1_primal_total_mw"])) > float(tie["primary_total_replay_tolerance_mw"]):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_P1_MISMATCH")
    if abs(u1 - float(stage1["u1_certified_upper_mw"])) > float(tie["primary_total_replay_tolerance_mw"]):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_U1_MISMATCH")
    if abs(stage1_gap - float(stage1["stage1_dual_gap_mw"])) > float(tie["primary_total_replay_tolerance_mw"]):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_GAP_MISMATCH")
    if abs(tau_h - float(stage1["tau_h_mw"])) > float(tie["primary_total_replay_tolerance_mw"]):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_TAU_H_MISMATCH")
    primary_floor = _finite_number(hierarchy.get("primary_floor_u1_minus_tau_h_mw"), name="parent_stage2.primary_floor_u1_minus_tau_h_mw")
    if abs(primary_floor - (u1 - tau_h)) > float(tie["primary_total_replay_tolerance_mw"]):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_FLOOR_MISMATCH")
    raw_attestation = stage2_result.get("hierarchy_attestation") if isinstance(stage2_result, Mapping) else None
    if not isinstance(raw_attestation, Mapping):
        raise SourceConicKKTError("STAGE2_PARENT_FULL_HIERARCHY_ATTESTATION_MISSING")
    try:
        _hash_bound_mapping(raw_attestation)
    except SourceConicKKTError as exc:
        raise SourceConicKKTError(f"STAGE2_PARENT_FULL_HIERARCHY_ATTESTATION_INVALID:{exc}") from exc
    source_attestation = stage2_result.get("source_hierarchy_attestation") if isinstance(stage2_result, Mapping) else None
    if not isinstance(source_attestation, Mapping) or canonical_hash(dict(source_attestation)) != canonical_hash(dict(raw_attestation)):
        raise SourceConicKKTError("STAGE2_PARENT_SOURCE_HIERARCHY_ATTESTATION_ALIAS_MISMATCH")
    if (
        raw_attestation.get("stage1_result_hash") != stage1["result_hash"]
        or raw_attestation.get("source_task_binding_hash") != task_binding.get("source_task_binding_hash")
        or raw_attestation.get("primary_face_mode") != "DUAL_UPPER_TOLERANCE_PRIMARY_FACE"
        or raw_attestation.get("primary_face_constraint") != "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H"
    ):
        raise SourceConicKKTError("STAGE2_PARENT_FULL_HIERARCHY_ATTESTATION_PARENT_CHAIN_MISMATCH")
    for key, expected in (
        ("p1_primal_total_mw", p1),
        ("u1_certified_upper_mw", u1),
        ("stage1_dual_gap_mw", stage1_gap),
        ("tau_h_mw", tau_h),
        ("primary_floor_u1_minus_tau_h_mw", primary_floor),
    ):
        if abs(_finite_number(raw_attestation.get(key), name=f"parent_stage2.raw_attestation.{key}") - expected) > float(tie["primary_total_replay_tolerance_mw"]):
            raise SourceConicKKTError(f"STAGE2_PARENT_FULL_HIERARCHY_ATTESTATION_{key.upper()}_MISMATCH")
    if metadata.get("primary_face_constraint") != "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H":
        raise SourceConicKKTError("STAGE2_PARENT_METADATA_PRIMARY_FACE_SEMANTICS_MISMATCH")
    for key, expected in (
        ("p1_primal_total_mw", p1),
        ("u1_certified_upper_mw", u1),
        ("stage1_dual_gap_mw", stage1_gap),
        ("tau_h_mw", tau_h),
        ("primary_floor_u1_minus_tau_h_mw", primary_floor),
    ):
        if abs(_finite_number(metadata.get(key), name=f"parent_stage2.metadata.{key}") - expected) > float(tie["primary_total_replay_tolerance_mw"]):
            raise SourceConicKKTError(f"STAGE2_PARENT_METADATA_{key.upper()}_MISMATCH")
    if not isinstance(hierarchy.get("stage1_result_hash"), str) or not hierarchy.get("stage1_result_hash"):
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_HASH_MISSING")
    if hierarchy.get("stage1_result_hash") != stage1["result_hash"]:
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_RESULT_MISMATCH")
    if hierarchy.get("p1_allocation_hash") != stage1["allocation_hash"]:
        raise SourceConicKKTError("STAGE2_PARENT_HIERARCHY_STAGE1_ALLOCATION_MISMATCH")
    total = float(math.fsum(summary["allocation_mw"]))
    if total + float(tie["primary_total_replay_tolerance_mw"]) < primary_floor:
        raise SourceConicKKTError("STAGE2_PARENT_PRIMARY_FACE_VIOLATION")
    return {**summary, "total_mw": total, "hierarchy_proof": dict(hierarchy)}


def _attested_tie_hierarchy(
    hierarchy: Mapping[str, Any] | None,
    *,
    source_task_binding_hash: str,
    stage1: Mapping[str, Any],
    tie: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(hierarchy, Mapping):
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_PROOF_REQUIRED"}
    try:
        attestation_hash = _hash_bound_mapping(hierarchy)
    except SourceConicKKTError as exc:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": str(exc)}
    if hierarchy.get("serialization_id") != HIERARCHY_ATTESTATION_SCHEMA_ID:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_ATTESTATION_SCHEMA_INVALID", "attestation_hash": attestation_hash}
    if hierarchy.get("proof_complete") is not True or hierarchy.get("primary_face_mode") != "DUAL_UPPER_TOLERANCE_PRIMARY_FACE":
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "DUAL_UPPER_PRIMARY_FACE_NOT_ATTESTED", "attestation_hash": attestation_hash}
    if hierarchy.get("primary_face_constraint") != "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H":
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_PRIMARY_FACE_SEMANTICS_INVALID", "attestation_hash": attestation_hash}
    if hierarchy.get("source_task_binding_hash") != source_task_binding_hash:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_SOURCE_TASK_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if hierarchy.get("formulation_policy_binding_hash") != tie["formulation_policy_binding_hash"]:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_FORMULATION_POLICY_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if hierarchy.get("stage1_result_hash") != stage1["result_hash"]:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_STAGE1_RESULT_HASH_MISMATCH", "attestation_hash": attestation_hash}
    try:
        p1 = _nonnegative_number(hierarchy.get("p1_primal_total_mw"), name="p1_primal_total_mw")
        u1 = _nonnegative_number(hierarchy.get("u1_certified_upper_mw"), name="u1_certified_upper_mw")
        stage1_gap = _nonnegative_number(hierarchy.get("stage1_dual_gap_mw"), name="stage1_dual_gap_mw")
        tau1 = _nonnegative_number(hierarchy.get("tau1_stage1_gap_max_mw"), name="tau1_stage1_gap_max_mw")
        tau_h = _nonnegative_number(hierarchy.get("tau_h_mw"), name="tau_h_mw")
        total_scale = _nonnegative_number(hierarchy.get("total_scale_st_mw"), name="total_scale_st_mw")
        allocation = _finite_vector(hierarchy.get("p1_allocation_mw"), name="p1_allocation_mw", length=PARTICIPANT_COUNT)
    except SourceConicKKTError as exc:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": str(exc), "attestation_hash": attestation_hash}
    allocation_hash = canonical_hash({"values_mw": allocation})
    tolerance = float(tie["primary_total_replay_tolerance_mw"])
    if hierarchy.get("p1_allocation_hash") != allocation_hash or allocation_hash != stage1["allocation_hash"]:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "P1_ALLOCATION_PARENT_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if hierarchy.get("hierarchy_tolerance_formula_id") != stage1["hierarchy_tolerance_formula_id"]:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_TOLERANCE_FORMULA_MISMATCH", "attestation_hash": attestation_hash}
    if hierarchy.get("hierarchy_tolerance_total_scale_definition") != stage1["hierarchy_tolerance_total_scale_definition"]:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_TOTAL_SCALE_DEFINITION_MISMATCH", "attestation_hash": attestation_hash}
    if abs(p1 - float(stage1["p1_primal_total_mw"])) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "P1_STAGE1_PARENT_MISMATCH", "attestation_hash": attestation_hash}
    if abs(u1 - float(stage1["u1_certified_upper_mw"])) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "U1_STAGE1_PARENT_MISMATCH", "attestation_hash": attestation_hash}
    if abs(stage1_gap - float(stage1["stage1_dual_gap_mw"])) > tolerance or abs(tau1 - float(stage1["tau1_stage1_gap_max_mw"])) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "STAGE1_DUAL_GAP_PARENT_MISMATCH", "attestation_hash": attestation_hash}
    if abs(tau_h - float(stage1["tau_h_mw"])) > tolerance or abs(total_scale - float(stage1["total_scale_st_mw"])) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "HIERARCHY_TOLERANCE_PARENT_MISMATCH", "attestation_hash": attestation_hash}
    if abs(math.fsum(allocation) - p1) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "P1_ALLOCATION_REPLAY_MISMATCH", "attestation_hash": attestation_hash}
    if u1 + tolerance < p1 or stage1_gap > tau1 + tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "P1_TSTAR_U1_BOUND_INVALID", "attestation_hash": attestation_hash}
    if tau_h + tolerance < stage1_gap:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "TAU_H_DOES_NOT_COVER_STAGE1_GAP", "attestation_hash": attestation_hash}
    primary_floor = u1 - tau_h
    declared_floor = _finite_number(hierarchy.get("primary_floor_u1_minus_tau_h_mw"), name="primary_floor_u1_minus_tau_h_mw")
    if abs(declared_floor - primary_floor) > tolerance:
        return False, {"status": "BLOCKED_HIERARCHY_PROOF_INCOMPLETE", "reason": "U1_MINUS_TAU_H_FLOOR_MISMATCH", "attestation_hash": attestation_hash}
    return True, {
        "attestation_hash": attestation_hash,
        "p1_primal_total_mw": p1,
        "u1_certified_upper_mw": u1,
        "stage1_dual_gap_mw": stage1_gap,
        "tau1_stage1_gap_max_mw": tau1,
        "tau_h_mw": tau_h,
        "total_scale_st_mw": total_scale,
        "hierarchy_tolerance_formula_id": stage1["hierarchy_tolerance_formula_id"],
        "hierarchy_tolerance_total_scale_definition": stage1["hierarchy_tolerance_total_scale_definition"],
        "primary_floor_u1_minus_tau_h_mw": primary_floor,
        "primary_face_constraint": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
        "p1_allocation_mw": allocation,
        "p1_allocation_hash": allocation_hash,
        "source_task_binding_hash": source_task_binding_hash,
        "formulation_policy_binding_hash": tie["formulation_policy_binding_hash"],
        "weights": list(tie["weights"]),
        "stage1_result_hash": stage1["result_hash"],
        "stage1_parent_result_validated": True,
    }


def _attested_fairness_reference(
    reference: Mapping[str, Any] | None,
    *,
    source_task_binding_hash: str,
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    tie: Mapping[str, Any],
    fairness: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(reference, Mapping):
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "FAIRNESS_REFERENCE_PROOF_REQUIRED"}
    try:
        attestation_hash = _hash_bound_mapping(reference)
    except SourceConicKKTError as exc:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": str(exc)}
    if reference.get("serialization_id") != FAIRNESS_REFERENCE_ATTESTATION_SCHEMA_ID:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_ATTESTATION_SCHEMA_INVALID", "attestation_hash": attestation_hash}
    if reference.get("proof_complete") is not True or reference.get("reference_mode") != "CERTIFIED_DETERMINISTIC_MAX_EXPORT_REFERENCE":
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_PROOF_NOT_COMPLETE", "attestation_hash": attestation_hash}
    if reference.get("source_task_binding_hash") != source_task_binding_hash:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_SOURCE_TASK_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("formulation_policy_binding_hash") != fairness["formulation_policy_binding_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_FORMULATION_POLICY_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_stage2_formulation_policy_binding_hash") != tie["formulation_policy_binding_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_TIE_BREAK_POLICY_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_stage2_result_hash") != stage2["result_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE2_RESULT_HASH_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_stage1_result_hash") != stage1["result_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE1_RESULT_HASH_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_method_id") != MAX_EXPORT_METHOD_ID:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_METHOD_ID_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_method_atom_binding_hash") != stage2["method_atom_binding_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_METHOD_ATOM_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if reference.get("reference_stage1_method_atom_binding_hash") != stage1["method_atom_binding_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE1_METHOD_ATOM_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    try:
        allocation = _finite_vector(reference.get("reference_allocation_mw"), name="reference_allocation_mw", length=PARTICIPANT_COUNT)
        total = _nonnegative_number(reference.get("reference_total_mw"), name="reference_total_mw")
    except SourceConicKKTError as exc:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": str(exc), "attestation_hash": attestation_hash}
    allocation_hash = canonical_hash({"values_mw": allocation})
    if reference.get("reference_allocation_hash") != allocation_hash or allocation_hash != stage2["allocation_hash"]:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_ALLOCATION_PARENT_BINDING_MISMATCH", "attestation_hash": attestation_hash}
    if abs(total - float(stage2["total_mw"])) > float(tie["primary_total_replay_tolerance_mw"]):
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_TOTAL_STAGE2_PARENT_MISMATCH", "attestation_hash": attestation_hash}
    if abs(math.fsum(allocation) - total) > float(tie["primary_total_replay_tolerance_mw"]):
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_TOTAL_ALLOCATION_REPLAY_MISMATCH", "attestation_hash": attestation_hash}
    stage2_hierarchy = stage2.get("hierarchy_proof")
    if not isinstance(stage2_hierarchy, Mapping):
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE2_HIERARCHY_MISSING", "attestation_hash": attestation_hash}
    try:
        primary_floor = _finite_number(
            stage2_hierarchy.get("primary_floor_u1_minus_tau_h_mw"),
            name="reference_stage2.primary_floor_u1_minus_tau_h_mw",
        )
        u1 = _nonnegative_number(stage2_hierarchy.get("u1_certified_upper_mw"), name="reference_stage2.u1_certified_upper_mw")
        tau_h = _nonnegative_number(stage2_hierarchy.get("tau_h_mw"), name="reference_stage2.tau_h_mw")
    except SourceConicKKTError as exc:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": str(exc), "attestation_hash": attestation_hash}
    if abs(primary_floor - (u1 - tau_h)) > float(tie["primary_total_replay_tolerance_mw"]):
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE2_PRIMARY_FLOOR_INVALID", "attestation_hash": attestation_hash}
    if total + float(tie["primary_total_replay_tolerance_mw"]) < primary_floor:
        return False, {"status": "BLOCKED_REFERENCE_PROOF_INCOMPLETE", "reason": "REFERENCE_STAGE2_PRIMARY_FACE_VIOLATION", "attestation_hash": attestation_hash}
    return True, {
        "attestation_hash": attestation_hash,
        "reference_allocation_mw": allocation,
        "reference_allocation_hash": allocation_hash,
        "reference_total_mw": total,
        "reference_stage2_result_hash": stage2["result_hash"],
        "reference_stage1_result_hash": stage1["result_hash"],
        "reference_stage2_formulation_policy_binding_hash": tie["formulation_policy_binding_hash"],
        "reference_method_id": MAX_EXPORT_METHOD_ID,
        "reference_method_atom_binding_hash": stage2["method_atom_binding_hash"],
        "reference_stage1_method_atom_binding_hash": stage1["method_atom_binding_hash"],
        "reference_stage2_primary_floor_u1_minus_tau_h_mw": primary_floor,
        "reference_stage2_certified_upper_u1_mw": u1,
        "reference_stage2_tau_h_mw": tau_h,
        "reference_stage2_primary_face_constraint": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
        "reference_stage2_lexicographic_semantics": "CERTIFIED_TAU_H_PRIMARY_FACE_WITH_CANONICAL_STRICT_CONVEX_TIE_BREAK",
        "source_task_binding_hash": source_task_binding_hash,
        "formulation_policy_binding_hash": fairness["formulation_policy_binding_hash"],
        "alpha": fairness["alpha"],
        "epsilon_mw": fairness["epsilon_mw"],
        "reference_stage2_parent_result_validated": True,
    }


def _linear_descriptor(*, constraint_id: str, family: str, index: int | None, constraint: Any, constant: np.ndarray, x_coeff: np.ndarray, rho_coeff: np.ndarray, units: str) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id, "kind": "LINEAR_NONPOSITIVE", "family": family, "index": index,
        "constraint": constraint, "constant": np.asarray(constant, dtype=float), "x_coeff": np.asarray(x_coeff, dtype=float),
        "rho_coeff": np.asarray(rho_coeff, dtype=float), "units": units,
    }


def _soc_descriptor(*, constraint_id: str, family: str, index: int | None, constraint: Any, t_constant: float, t_x: np.ndarray, t_rho: float, z_constant: np.ndarray, z_x: np.ndarray, z_rho: np.ndarray, units: Mapping[str, str]) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id, "kind": "SECOND_ORDER_CONE", "family": family, "index": index,
        "constraint": constraint, "t_constant": float(t_constant), "t_x": np.asarray(t_x, dtype=float), "t_rho": float(t_rho),
        "z_constant": np.asarray(z_constant, dtype=float), "z_x": np.asarray(z_x, dtype=float), "z_rho": np.asarray(z_rho, dtype=float),
        "units": dict(units),
    }


def build_original_unit_source_conic_program(
    *,
    runtime: Mapping[str, Any],
    capacity_mw: Sequence[float],
    raw_request_mw: Sequence[float],
    objective_kind: str,
    policy_mapping: Mapping[str, Any],
    primary_floor_mw: float | None = None,
    tie_weights: Sequence[float] | None = None,
    fairness_qdiag: Sequence[float] | None = None,
    fairness_linear: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build the original-unit SOCP with explicit visible linear/SOC cones."""
    if objective_kind not in OBJECTIVE_KINDS:
        raise SourceConicKKTError("unknown source conic objective")
    policy = validate_external_policy(policy_mapping)
    data = _runtime_data(runtime, capacity_mw, raw_request_mw)
    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise SourceConicKKTError(f"CVXPY_UNAVAILABLE:{type(exc).__name__}") from exc
    n = PARTICIPANT_COUNT
    capacity, raw, rates = data["capacity"], data["raw"], data["rates"]
    base, matrices, margin = data["base"], data["matrices"], data["margin"]
    constants = policy["constraint_constants"]
    af, lf = _components(margin, "branch_from_mva", len(rates))
    at, lt = _components(margin, "branch_to_mva", len(rates))
    al, ll = _components(margin, "voltage_lower_pu", len(base["voltage_pu"]))
    au, lu = _components(margin, "voltage_upper_pu", len(base["voltage_pu"]), allow_none=True)
    x = cp.Variable(n, name="allocation_mw")
    rho = cp.Variable(name="rho")
    constraints: list[Any] = []
    descriptors: list[dict[str, Any]] = []

    def add_linear(constraint_id: str, family: str, index: int | None, expression: Any, constant: np.ndarray, x_coeff: np.ndarray, rho_coeff: np.ndarray, units: str) -> None:
        constraint = expression <= 0.0
        constraints.append(constraint)
        descriptors.append(_linear_descriptor(constraint_id=constraint_id, family=family, index=index, constraint=constraint, constant=constant, x_coeff=x_coeff, rho_coeff=rho_coeff, units=units))

    def add_soc(constraint_id: str, family: str, index: int | None, t: Any, z: Any, t_constant: float, t_x: np.ndarray, t_rho: float, z_constant: np.ndarray, z_x: np.ndarray, z_rho: np.ndarray, units: Mapping[str, str]) -> None:
        constraint = cp.SOC(t, z)
        constraints.append(constraint)
        descriptors.append(_soc_descriptor(constraint_id=constraint_id, family=family, index=index, constraint=constraint, t_constant=t_constant, t_x=t_x, t_rho=t_rho, z_constant=z_constant, z_x=z_x, z_rho=z_rho, units=units))

    eye = np.eye(n, dtype=float)
    zeros = np.zeros(n, dtype=float)
    add_linear("X_LOWER", "x_lower", None, -x, np.zeros(n), -eye, np.zeros(n), "MW")
    add_linear("X_UPPER_RAW_REQUEST", "x_upper_raw_request", None, x - np.asarray(raw), -np.asarray(raw), eye, np.zeros(n), "MW")
    add_linear("X_UPPER_CAPACITY", "x_upper_capacity", None, x - np.asarray(capacity), -np.asarray(capacity), eye, np.zeros(n), "MW")
    add_linear("RHO_LOWER", "rho_lower", None, -rho, np.asarray([0.0]), np.zeros((1, n)), np.asarray([-1.0]), "dimensionless")
    add_linear("RHO_UPPER", "rho_upper", None, rho - float(data["rho_max"]), np.asarray([-float(data["rho_max"])]), np.zeros((1, n)), np.asarray([1.0]), "dimensionless")
    add_soc(
        "RHO_SOC", "rho_soc", None, rho, cp.multiply(1.0 / np.asarray(capacity), x),
        t_constant=0.0, t_x=zeros, t_rho=1.0, z_constant=np.zeros(n), z_x=np.diag(1.0 / np.asarray(capacity)), z_rho=np.zeros(n),
        units={"t": "dimensionless", "z": "dimensionless"},
    )

    def affine(field: str, index: int) -> Any:
        return float(base[field][index]) + np.asarray(matrices[field][index], dtype=float) @ x

    for index, rate in enumerate(rates):
        p_row, q_row = np.asarray(matrices["branch_p_from_mw"][index], dtype=float), np.asarray(matrices["branch_q_from_mvar"][index], dtype=float)
        t_constant = float(rate) - constants["branch_buffer_mva"] - float(af[index] or 0.0)
        add_soc(
            f"BRANCH_FROM_{index}", "branch_from", index,
            t_constant - float(lf[index]) * rho,
            cp.hstack([affine("branch_p_from_mw", index), affine("branch_q_from_mvar", index)]),
            t_constant=t_constant, t_x=zeros, t_rho=-float(lf[index]),
            z_constant=np.asarray([float(base["branch_p_from_mw"][index]), float(base["branch_q_from_mvar"][index])]),
            z_x=np.vstack([p_row, q_row]), z_rho=np.zeros(2), units={"t": "MVA", "z": "MW_or_MVAr"},
        )
        p_row, q_row = np.asarray(matrices["branch_p_to_mw"][index], dtype=float), np.asarray(matrices["branch_q_to_mvar"][index], dtype=float)
        t_constant = float(rate) - constants["branch_buffer_mva"] - float(at[index] or 0.0)
        add_soc(
            f"BRANCH_TO_{index}", "branch_to", index,
            t_constant - float(lt[index]) * rho,
            cp.hstack([affine("branch_p_to_mw", index), affine("branch_q_to_mvar", index)]),
            t_constant=t_constant, t_x=zeros, t_rho=-float(lt[index]),
            z_constant=np.asarray([float(base["branch_p_to_mw"][index]), float(base["branch_q_to_mvar"][index])]),
            z_x=np.vstack([p_row, q_row]), z_rho=np.zeros(2), units={"t": "MVA", "z": "MW_or_MVAr"},
        )
    for index, value in enumerate(base["voltage_pu"]):
        row = np.asarray(matrices["voltage_pu"][index], dtype=float)
        # lower + buffer + a + l*rho - V(x) <= 0
        lower_constant = float(data["lower"]) + constants["voltage_buffer_pu"] + float(al[index] or 0.0) - float(value)
        add_linear(
            f"VOLTAGE_LOWER_{index}", "voltage_lower", index,
            lower_constant - row @ x + float(ll[index]) * rho,
            np.asarray([lower_constant]), -row.reshape(1, n), np.asarray([float(ll[index])]), "pu",
        )
        if au[index] is None:
            # V(x) - Vmax + guard + buffer <= 0; inactive is not zero margin.
            upper_constant = float(value) - float(data["upper"]) + constants["upper_voltage_guard_pu"] + constants["voltage_buffer_pu"]
            add_linear(
                f"VOLTAGE_UPPER_INACTIVE_{index}", "voltage_upper_inactive", index,
                upper_constant + row @ x,
                np.asarray([upper_constant]), row.reshape(1, n), np.asarray([0.0]), "pu",
            )
        else:
            upper_constant = float(value) + float(au[index] or 0.0) - float(data["upper"]) + constants["voltage_buffer_pu"]
            add_linear(
                f"VOLTAGE_UPPER_{index}", "voltage_upper", index,
                upper_constant + row @ x + float(lu[index]) * rho,
                np.asarray([upper_constant]), row.reshape(1, n), np.asarray([float(lu[index])]), "pu",
            )

    primary_face_floor: float | None = None
    if objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        if primary_floor_mw is None:
            raise SourceConicKKTError("certified U1-minus-tauH primary floor is required")
        primary_face_floor = _finite_number(primary_floor_mw, name="primary_floor_mw")
        add_linear(
            "PRIMARY_FACE",
            "primary_face",
            None,
            primary_face_floor - cp.sum(x),
            np.asarray([primary_face_floor]),
            -np.ones((1, n)),
            np.asarray([0.0]),
            "MW",
        )
    elif objective_kind == OBJECTIVE_FAIRNESS_QP and primary_floor_mw is not None:
        raise SourceConicKKTError(
            "fairness uses the same-context ME allocation as an objective reference, "
            "not as an additional primary-face constraint"
        )

    objective_gradient_x: np.ndarray
    objective_gradient_rho = 0.0
    objective_metadata: dict[str, Any] = {"objective_kind": objective_kind}
    if objective_kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        objective = cp.Minimize(-cp.sum(x))
        objective_gradient_x = -np.ones(n, dtype=float)
        objective_metadata["sense"] = "MAXIMIZE_TOTAL_EXPORT"
    elif objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        tie = _tie_configuration(policy["policy"], constraint_constants=policy["constraint_constants"])
        weights = list(tie["weights"])
        if tie_weights is not None and _finite_signed_vector(tie_weights, name="tie_weights", length=n) != weights:
            raise SourceConicKKTError("tie-stage caller weights do not equal registered canonical weights")
        regularization = _tie_regularization_for_task(tie, runtime_data=data)
        x_qdiag = float(regularization["x_quadratic_coefficient_mw_inverse"])
        rho_qdiag = float(regularization["rho_quadratic_coefficient_mw"])
        objective = cp.Minimize(
            np.asarray(weights) @ x
            + x_qdiag * cp.sum_squares(x)
            + rho_qdiag * cp.square(rho)
        )
        objective_gradient_x = np.asarray(weights, dtype=float)
        objective_metadata.update(
            {
                "sense": "MINIMIZE_CERTIFIED_TAU_H_CANONICAL_STRICTLY_CONVEX_TIE_BREAK",
                "weights": weights,
                "linear_weight_scale": float(tie["canonical_linear_weight_scale"]),
                "canonical_tie_policy_binding_hash": str(tie["formulation_policy_binding_hash"]),
                "objective_gap_acceptance_cap_native_units": float(tie["tie_objective_replay_tolerance_native_units"]),
                "normalized_decision_l2_optimality_enclosure_max": float(tie["normalized_decision_l2_optimality_enclosure_max"]),
                "qdiag": [x_qdiag] * n,
                "linear": weights,
                "rho_qdiag": rho_qdiag,
                "strict_convex_regularization": regularization,
                "primary_floor_mw": primary_face_floor,
                "primary_face_constraint": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
                "lexicographic_semantics": "CERTIFIED_TAU_H_PRIMARY_FACE_WITH_CANONICAL_STRICT_CONVEX_TIE_BREAK",
            }
        )
    else:
        if fairness_qdiag is None or fairness_linear is None:
            raise SourceConicKKTError("fairness qdiag and linear terms are required")
        qdiag = _finite_vector(fairness_qdiag, name="fairness_qdiag", length=n, positive=True)
        linear = _finite_signed_vector(fairness_linear, name="fairness_linear", length=n)
        objective = cp.Minimize(cp.sum(cp.multiply(np.asarray(qdiag), cp.square(x))) + np.asarray(linear) @ x)
        objective_gradient_x = np.zeros(n, dtype=float)  # populated at primal replay
        objective_metadata.update(
            {
                "sense": "MINIMIZE_REGISTERED_FAIRNESS_QP",
                "qdiag": qdiag,
                "linear": linear,
                "primary_floor_mw": primary_face_floor,
                "primary_face_constraint": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
                "lexicographic_semantics": "CERTIFIED_TAU_H_PRIMARY_FACE_WITH_CANONICAL_STRICT_CONVEX_TIE_BREAK",
            }
        )
    problem = cp.Problem(objective, constraints)
    return {
        "problem": problem, "x": x, "rho": rho, "descriptors": descriptors,
        "objective_kind": objective_kind, "objective_gradient_x": objective_gradient_x,
        "objective_gradient_rho": objective_gradient_rho, "objective_metadata": objective_metadata,
        "runtime_data": data, "policy": policy,
    }


def _linear_dual_record(descriptor: Mapping[str, Any], *, x: np.ndarray, rho: float) -> tuple[dict[str, Any], np.ndarray, float, float, float, float]:
    dual_raw = descriptor["constraint"].dual_value
    expr = np.asarray(descriptor["constant"], dtype=float) + np.asarray(descriptor["x_coeff"], dtype=float) @ x + np.asarray(descriptor["rho_coeff"], dtype=float) * rho
    dual = np.asarray(dual_raw, dtype=float).reshape(-1) if dual_raw is not None else np.asarray([math.nan])
    if dual.size == 1 and expr.size > 1:
        dual = np.repeat(dual, expr.size)
    if dual.size != expr.size or not np.all(np.isfinite(dual)) or not np.all(np.isfinite(expr)):
        record = {"constraint_id": descriptor["constraint_id"], "family": descriptor["family"], "index": descriptor["index"], "units": descriptor["units"], "dual": None, "primal_expression": None, "valid": False}
        return record, np.full(PARTICIPANT_COUNT, math.nan), math.nan, math.inf, math.inf, math.inf
    x_coeff = np.asarray(descriptor["x_coeff"], dtype=float)
    rho_coeff = np.asarray(descriptor["rho_coeff"], dtype=float).reshape(-1)
    grad_x = x_coeff.T @ dual
    grad_rho = float(rho_coeff @ dual)
    primal = max(0.0, float(np.max(expr)))
    dual_negative = max(0.0, -float(np.min(dual)))
    complementarity = float(np.max(np.abs(dual * expr)))
    record = {
        "constraint_id": descriptor["constraint_id"], "family": descriptor["family"], "index": descriptor["index"], "units": descriptor["units"],
        "dual": dual.tolist(), "primal_expression": expr.tolist(), "primal_violation_inf": primal,
        "dual_negative_inf": dual_negative, "complementarity_inf": complementarity, "valid": True,
    }
    return record, grad_x, grad_rho, primal, dual_negative, complementarity


def _soc_dual_record(descriptor: Mapping[str, Any], *, x: np.ndarray, rho: float) -> tuple[dict[str, Any], np.ndarray, float, float, float, float]:
    dual_raw = descriptor["constraint"].dual_value
    try:
        if not isinstance(dual_raw, (list, tuple)) or len(dual_raw) != 2:
            raise ValueError("SOC dual shape")
        dual_t = float(np.asarray(dual_raw[0], dtype=float).reshape(-1)[0])
        dual_z = np.asarray(dual_raw[1], dtype=float).reshape(-1)
        t = float(descriptor["t_constant"]) + float(np.asarray(descriptor["t_x"], dtype=float) @ x) + float(descriptor["t_rho"]) * rho
        z = np.asarray(descriptor["z_constant"], dtype=float) + np.asarray(descriptor["z_x"], dtype=float) @ x + np.asarray(descriptor["z_rho"], dtype=float) * rho
        if len(dual_z) != len(z) or not math.isfinite(dual_t) or not np.all(np.isfinite(dual_z)) or not math.isfinite(t) or not np.all(np.isfinite(z)):
            raise ValueError("non-finite SOC primal/dual")
    except (TypeError, ValueError, IndexError):
        record = {"constraint_id": descriptor["constraint_id"], "family": descriptor["family"], "index": descriptor["index"], "units": descriptor["units"], "dual_cone": None, "primal_cone": None, "valid": False}
        return record, np.full(PARTICIPANT_COUNT, math.nan), math.nan, math.inf, math.inf, math.inf
    primal_violation = max(0.0, float(np.linalg.norm(z)) - t)
    dual_violation = max(0.0, float(np.linalg.norm(dual_z)) - dual_t, -dual_t)
    complementarity = abs(dual_t * t + float(dual_z @ z))
    # CVXPY's SOC dual convention corresponds to a Lagrangian contribution
    # -<dual, (t,z)> for (t,z) in the cone.  This sign is verified by explicit
    # primal/dual cone complementarity below rather than inferred from a norm
    # surrogate multiplier.
    grad_x = -(dual_t * np.asarray(descriptor["t_x"], dtype=float) + np.asarray(descriptor["z_x"], dtype=float).T @ dual_z)
    grad_rho = -float(dual_t * float(descriptor["t_rho"]) + dual_z @ np.asarray(descriptor["z_rho"], dtype=float))
    record = {
        "constraint_id": descriptor["constraint_id"], "family": descriptor["family"], "index": descriptor["index"], "units": descriptor["units"],
        "primal_cone": {"t": t, "z": z.tolist(), "norm_z": float(np.linalg.norm(z)), "slack": t - float(np.linalg.norm(z)), "primal_violation_inf": primal_violation},
        "dual_cone": {"t": dual_t, "z": dual_z.tolist(), "norm_z": float(np.linalg.norm(dual_z)), "dual_cone_violation_inf": dual_violation},
        "complementarity": dual_t * t + float(dual_z @ z), "complementarity_inf": complementarity, "valid": True,
    }
    return record, grad_x, grad_rho, primal_violation, dual_violation, complementarity


def _project_lorentz_cone(t: float, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Euclidean projection onto ``{(t,z): ||z|| <= t}``."""
    norm_z = float(np.linalg.norm(z))
    if norm_z <= t:
        return float(t), np.asarray(z, dtype=float)
    if norm_z <= -t:
        return 0.0, np.zeros_like(z, dtype=float)
    projected_t = 0.5 * (norm_z + t)
    return projected_t, (projected_t / norm_z) * np.asarray(z, dtype=float)


def _reconstruct_original_unit_linear_dual(program: Mapping[str, Any]) -> dict[str, Any]:
    """Construct a conservative dual lower bound from visible primal duals.

    All linear multipliers are projected to the non-negative orthant and all
    SOC multipliers are projected to their Lorentz cones.  The resulting dual
    vector is exactly cone-feasible.  Any remaining stationarity residual is
    minimized explicitly over the declared allocation/rho box, yielding a
    valid lower bound even when a floating-point solver has not made the raw
    stationarity equation exact.
    """
    kind = program["objective_kind"]
    objective_qdiag: np.ndarray | None = None
    objective_rho_qdiag = 0.0
    if kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        objective_gradient_x = -np.ones(PARTICIPANT_COUNT, dtype=float)
        objective_gradient_rho = 0.0
    elif kind in {OBJECTIVE_MAX_EXPORT_TIE_STAGE2, OBJECTIVE_FAIRNESS_QP}:
        # Both post-stage-one objectives are separable convex QPs.  Unlike a
        # linear LP objective, their dual Lagrangians are minimized
        # analytically over the explicit allocation/rho boxes below; no
        # solver-reported objective value is promoted to a global bound.
        metadata = program.get("objective_metadata")
        if not isinstance(metadata, Mapping):
            return {"available": False, "status": "CONVEX_QP_OBJECTIVE_METADATA_INVALID"}
        try:
            objective_qdiag = np.asarray(
                _finite_vector(
                    metadata.get("qdiag"),
                    name="stage2 qdiag" if kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2 else "fairness qdiag",
                    length=PARTICIPANT_COUNT,
                    positive=True,
                ),
                dtype=float,
            )
            objective_gradient_x = np.asarray(
                _finite_signed_vector(
                    metadata.get("linear"),
                    name="stage2 linear" if kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2 else "fairness linear",
                    length=PARTICIPANT_COUNT,
                ),
                dtype=float,
            )
            if kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
                objective_rho_qdiag = _nonnegative_number(
                    metadata.get("rho_qdiag"),
                    name="stage2 rho qdiag",
                    strictly_positive=True,
                )
        except SourceConicKKTError:
            return {"available": False, "status": "CONVEX_QP_OBJECTIVE_COEFFICIENTS_INVALID"}
        objective_gradient_rho = 0.0
    else:  # pragma: no cover - objective kinds are validated by the builder
        return {"available": False, "status": "UNKNOWN_OBJECTIVE"}
    if objective_gradient_x.shape != (PARTICIPANT_COUNT,) or not np.all(np.isfinite(objective_gradient_x)):
        return {"available": False, "status": "OBJECTIVE_GRADIENT_INVALID"}

    dual_constant = 0.0
    stationarity_x = objective_gradient_x.copy()
    stationarity_rho = objective_gradient_rho
    raw_dual_violation_inf = 0.0
    linear_terms: list[dict[str, Any]] = []
    soc_terms: list[dict[str, Any]] = []
    try:
        for descriptor in program["descriptors"]:
            if descriptor["kind"] == "LINEAR_NONPOSITIVE":
                raw = descriptor["constraint"].dual_value
                constant = np.asarray(descriptor["constant"], dtype=float).reshape(-1)
                raw_dual = np.asarray(raw, dtype=float).reshape(-1)
                if raw_dual.size == 1 and constant.size > 1:
                    raw_dual = np.repeat(raw_dual, constant.size)
                if raw_dual.size != constant.size or not np.all(np.isfinite(raw_dual)):
                    raise ValueError(f"linear dual invalid:{descriptor['constraint_id']}")
                projected = np.maximum(raw_dual, 0.0)
                x_coeff = np.asarray(descriptor["x_coeff"], dtype=float)
                rho_coeff = np.asarray(descriptor["rho_coeff"], dtype=float).reshape(-1)
                contribution = float(projected @ constant)
                dual_constant += contribution
                stationarity_x += x_coeff.T @ projected
                stationarity_rho += float(rho_coeff @ projected)
                raw_dual_violation_inf = max(raw_dual_violation_inf, max(0.0, -float(np.min(raw_dual))))
                linear_terms.append(
                    {
                        "constraint_id": descriptor["constraint_id"],
                        "raw_dual": raw_dual.tolist(),
                        "projected_nonnegative_dual": projected.tolist(),
                        "projection_inf": float(np.max(np.abs(projected - raw_dual))),
                        "dual_objective_constant_contribution": contribution,
                    }
                )
            elif descriptor["kind"] == "SECOND_ORDER_CONE":
                raw = descriptor["constraint"].dual_value
                if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                    raise ValueError(f"SOC dual invalid:{descriptor['constraint_id']}")
                raw_t = float(np.asarray(raw[0], dtype=float).reshape(-1)[0])
                raw_z = np.asarray(raw[1], dtype=float).reshape(-1)
                z_constant = np.asarray(descriptor["z_constant"], dtype=float).reshape(-1)
                if raw_z.size != z_constant.size or not math.isfinite(raw_t) or not np.all(np.isfinite(raw_z)):
                    raise ValueError(f"SOC dual invalid:{descriptor['constraint_id']}")
                projected_t, projected_z = _project_lorentz_cone(raw_t, raw_z)
                raw_violation = max(0.0, float(np.linalg.norm(raw_z)) - raw_t, -raw_t)
                raw_dual_violation_inf = max(raw_dual_violation_inf, raw_violation)
                t_constant = float(descriptor["t_constant"])
                t_x = np.asarray(descriptor["t_x"], dtype=float)
                t_rho = float(descriptor["t_rho"])
                z_x = np.asarray(descriptor["z_x"], dtype=float)
                z_rho = np.asarray(descriptor["z_rho"], dtype=float)
                contribution = -float(projected_t * t_constant + projected_z @ z_constant)
                dual_constant += contribution
                stationarity_x -= projected_t * t_x + z_x.T @ projected_z
                stationarity_rho -= float(projected_t * t_rho + projected_z @ z_rho)
                soc_terms.append(
                    {
                        "constraint_id": descriptor["constraint_id"],
                        "raw_dual_cone": {"t": raw_t, "z": raw_z.tolist()},
                        "projected_dual_cone": {"t": projected_t, "z": projected_z.tolist()},
                        "projection_inf": max(abs(projected_t - raw_t), float(np.max(np.abs(projected_z - raw_z)))),
                        "raw_dual_cone_violation_inf": raw_violation,
                        "dual_objective_constant_contribution": contribution,
                    }
                )
            else:
                raise ValueError(f"unknown descriptor kind:{descriptor['kind']}")
    except (TypeError, ValueError, IndexError):
        return {"available": False, "status": "DUAL_VECTOR_UNAVAILABLE_OR_INVALID"}

    runtime = program["runtime_data"]
    upper = np.minimum(np.asarray(runtime["capacity"], dtype=float), np.asarray(runtime["raw"], dtype=float))
    rho_max = float(runtime["rho_max"])
    if upper.shape != (PARTICIPANT_COUNT,) or not np.all(np.isfinite(upper)) or np.any(upper < 0.0) or not math.isfinite(rho_max) or rho_max < 0.0:
        return {"available": False, "status": "DUAL_BOUND_BOX_INVALID"}
    # For LP objectives the Lagrangian is affine in x.  For the fairness QP,
    # it is separable convex quadratic ``q_i*x_i^2 + a_i*x_i`` after adding
    # the projected cone/linear dual terms.  In both cases we take the exact
    # infimum over the explicit primal box rather than treating an approximate
    # stationarity residual as if it were zero.
    if objective_qdiag is None:
        box_correction_x = np.minimum(0.0, stationarity_x) * upper
        x_box_minimizers = np.where(stationarity_x < 0.0, upper, 0.0)
        x_infimum_mode = "AFFINE_BOX_INFIMUM"
    else:
        x_box_minimizers = np.clip(-stationarity_x / (2.0 * objective_qdiag), 0.0, upper)
        box_correction_x = objective_qdiag * np.square(x_box_minimizers) + stationarity_x * x_box_minimizers
        x_infimum_mode = "SEPARABLE_CONVEX_QP_BOX_INFIMUM"
    if objective_rho_qdiag > 0.0:
        rho_box_minimizer = min(max(-stationarity_rho / (2.0 * objective_rho_qdiag), 0.0), rho_max)
        box_correction_rho = objective_rho_qdiag * rho_box_minimizer * rho_box_minimizer + stationarity_rho * rho_box_minimizer
        rho_infimum_mode = "CONVEX_QUADRATIC_BOX_INFIMUM"
    else:
        rho_box_minimizer = rho_max if stationarity_rho < 0.0 else 0.0
        box_correction_rho = min(0.0, stationarity_rho) * rho_max
        rho_infimum_mode = "AFFINE_BOX_INFIMUM"
    box_correction = float(np.sum(box_correction_x) + box_correction_rho)
    raw_lower_bound = float(dual_constant + box_correction)
    # This is not a policy acceptance tolerance.  It is a deterministic,
    # scale-aware rounding guard on the reconstructed floating-point dual
    # arithmetic itself.  We move the lower bound *down* by the guard, so the
    # max-export upper bound derived from it is conservative.  The individual
    # ingredients are preserved below so a verifier can recompute the guard.
    numerical_scale = 1.0 + abs(dual_constant) + abs(box_correction)
    numerical_scale += float(np.sum(np.abs(stationarity_x) * upper))
    if objective_qdiag is not None:
        numerical_scale += float(np.sum(np.abs(objective_qdiag) * np.square(upper)))
    numerical_scale += abs(float(objective_rho_qdiag)) * rho_max * rho_max
    numerical_scale += abs(float(stationarity_rho)) * rho_max
    numerical_safety_margin = float(128.0 * np.finfo(float).eps * numerical_scale)
    lower_bound = float(raw_lower_bound - numerical_safety_margin)
    projected_stationarity_inf = max(float(np.max(np.abs(stationarity_x))), abs(float(stationarity_rho)))
    result: dict[str, Any] = {
        "available": True,
        "status": "PROJECTED_CONE_DUAL_BOX_LOWER_BOUND",
        "objective_kind": kind,
        "dual_feasibility_enforced_by_projection": True,
        "raw_dual_feasibility_violation_inf": raw_dual_violation_inf,
        "linear_terms": linear_terms,
        "soc_terms": soc_terms,
        "dual_objective_constant_native_units": dual_constant,
        "projected_stationarity": {
            "x": stationarity_x.tolist(),
            "rho": float(stationarity_rho),
            "inf_norm": projected_stationarity_inf,
        },
        "box_stationarity_infimum_correction_native_units": box_correction,
        "box_stationarity_infimum_x_terms_native_units": box_correction_x.tolist(),
        "box_stationarity_infimum_rho_term_native_units": float(box_correction_rho),
        "box_infimum_rho_mode": rho_infimum_mode,
        "box_infimum_rho_minimizer_dimensionless": float(rho_box_minimizer),
        "box_infimum_mode": x_infimum_mode,
        "box_infimum_x_minimizer_mw": x_box_minimizers.tolist(),
        "minimization_dual_lower_bound_before_numerical_guard_native_units": raw_lower_bound,
        "dual_lower_bound_numerical_safety_margin_native_units": numerical_safety_margin,
        "minimization_dual_lower_bound_native_units": lower_bound,
        "allocation_box_upper_mw": upper.tolist(),
        "rho_box_upper_dimensionless": rho_max,
    }
    if kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        result["max_export_certified_upper_u1_mw"] = -lower_bound
    elif kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        result["tie_objective_certified_lower_bound_native_units"] = lower_bound
    elif kind == OBJECTIVE_FAIRNESS_QP:
        result["fairness_qp_certified_lower_bound_native_units"] = lower_bound
    return result


def replay_original_unit_kkt(program: Mapping[str, Any]) -> dict[str, Any]:
    """Replay primal/dual/KKT quantities from explicit original-unit cones."""
    x_value = program["x"].value
    rho_value = program["rho"].value
    if x_value is None or rho_value is None:
        return {"pass": False, "failure_reason": "PRIMAL_VARIABLE_VALUE_MISSING"}
    x = np.asarray(x_value, dtype=float).reshape(-1)
    rho = float(rho_value)
    if len(x) != PARTICIPANT_COUNT or not np.all(np.isfinite(x)) or not math.isfinite(rho):
        return {"pass": False, "failure_reason": "PRIMAL_VARIABLE_VALUE_INVALID"}
    objective_gradient = np.asarray(program["objective_gradient_x"], dtype=float).copy()
    rho_objective_gradient = float(program["objective_gradient_rho"])
    if program["objective_kind"] in {OBJECTIVE_MAX_EXPORT_TIE_STAGE2, OBJECTIVE_FAIRNESS_QP}:
        meta = program["objective_metadata"]
        objective_gradient = 2.0 * np.asarray(meta["qdiag"], dtype=float) * x + np.asarray(meta["linear"], dtype=float)
        rho_objective_gradient = 2.0 * float(meta.get("rho_qdiag", 0.0)) * rho
    stationarity_x = objective_gradient
    stationarity_rho = rho_objective_gradient
    linear_records: list[dict[str, Any]] = []
    soc_records: list[dict[str, Any]] = []
    primal_inf = dual_inf = complementarity_inf = 0.0
    primal_by_native_unit = {"MW": 0.0, "MVA": 0.0, "pu": 0.0, "dimensionless": 0.0}
    valid = True
    for descriptor in program["descriptors"]:
        if descriptor["kind"] == "LINEAR_NONPOSITIVE":
            record, gx, gr, primal, dual, comp = _linear_dual_record(descriptor, x=x, rho=rho)
            linear_records.append(record)
            native_unit = str(descriptor["units"])
        else:
            record, gx, gr, primal, dual, comp = _soc_dual_record(descriptor, x=x, rho=rho)
            soc_records.append(record)
            native_unit = str((descriptor["units"] or {}).get("t"))
        if not (np.all(np.isfinite(gx)) and math.isfinite(gr)):
            valid = False
        else:
            stationarity_x = stationarity_x + gx
            stationarity_rho += gr
        primal_inf = max(primal_inf, primal)
        if native_unit in primal_by_native_unit:
            primal_by_native_unit[native_unit] = max(primal_by_native_unit[native_unit], primal)
        else:
            valid = False
        dual_inf = max(dual_inf, dual)
        complementarity_inf = max(complementarity_inf, comp)
    stationarity_inf = max(float(np.max(np.abs(stationarity_x))), abs(float(stationarity_rho))) if valid else math.inf
    tolerances = program["policy"]["kkt_tolerances"]
    native_tolerances = program["policy"]["original_unit_primal_tolerances"]
    native_primal_pass = bool(
        primal_by_native_unit["MW"] <= native_tolerances["allocation_box_mw"]
        and primal_by_native_unit["MVA"] <= native_tolerances["branch_soc_mva"]
        and primal_by_native_unit["pu"] <= native_tolerances["voltage_pu"]
        and primal_by_native_unit["dimensionless"] <= native_tolerances["rho_dimensionless"]
    )
    passed = bool(
        valid
        and native_primal_pass
        and dual_inf <= tolerances["dual_cone_inf"]
        and stationarity_inf <= tolerances["stationarity_inf"]
        and complementarity_inf <= tolerances["complementarity_inf"]
    )
    dual_objective = _reconstruct_original_unit_linear_dual(program)
    objective_kind = program["objective_kind"]
    if objective_kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        primal_objective = float(math.fsum(x.tolist()))
    elif objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        metadata = program["objective_metadata"]
        linear_component = float(np.asarray(metadata["linear"], dtype=float) @ x)
        x_quadratic_component = float(np.asarray(metadata["qdiag"], dtype=float) @ np.square(x))
        rho_quadratic_component = float(metadata["rho_qdiag"]) * rho * rho
        primal_objective = linear_component + x_quadratic_component + rho_quadratic_component
    else:
        metadata = program["objective_metadata"]
        primal_objective = float(
            np.asarray(metadata["qdiag"], dtype=float) @ np.square(x)
            + np.asarray(metadata["linear"], dtype=float) @ x
        )
    if objective_kind == OBJECTIVE_MAX_EXPORT_STAGE1 and dual_objective.get("available") is True:
        upper = float(dual_objective["max_export_certified_upper_u1_mw"])
        dual_objective = {
            **dual_objective,
            "max_export_primal_total_p1_mw": primal_objective,
            "u1_minus_p1_mw": upper - primal_objective,
            "p1_exceeds_u1_violation_mw": max(0.0, primal_objective - upper),
        }
    elif objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2 and dual_objective.get("available") is True:
        lower = float(dual_objective["tie_objective_certified_lower_bound_native_units"])
        metadata = program["objective_metadata"]
        regularization = metadata.get("strict_convex_regularization")
        if not isinstance(regularization, Mapping):
            return {
                "pass": False,
                "failure_reason": "STAGE2_STRICT_CONVEX_REGULARIZATION_METADATA_INVALID",
            }
        x_modulus = float(regularization["x_strong_convexity_modulus_mw_inverse"])
        rho_modulus = float(regularization["rho_strong_convexity_modulus_mw"])
        normalized_modulus = float(regularization["normalized_decision_strong_convexity_modulus_mw"])
        normalizer = float(regularization["normalizer_mw"])
        gap = max(0.0, primal_objective - lower)
        allocation_l2_enclosure = math.sqrt(2.0 * gap / x_modulus)
        rho_enclosure = math.sqrt(2.0 * gap / rho_modulus)
        normalized_decision_enclosure = math.sqrt(2.0 * gap / normalized_modulus)
        dual_objective = {
            **dual_objective,
            "tie_objective_primal_value_native_units": primal_objective,
            "tie_objective_primal_minus_dual_lower_bound_native_units": primal_objective - lower,
            "tie_objective_dual_lower_exceeds_primal_violation_native_units": max(0.0, lower - primal_objective),
            "tie_objective_linear_index_component_native_units": linear_component,
            "tie_objective_x_quadratic_component_native_units": x_quadratic_component,
            "tie_objective_rho_quadratic_component_native_units": rho_quadratic_component,
            "tie_objective_strict_convex_quadratic_component_native_units": x_quadratic_component + rho_quadratic_component,
            "tie_objective_allocation_l2_optimality_enclosure_mw": allocation_l2_enclosure,
            "tie_objective_rho_optimality_enclosure_dimensionless": rho_enclosure,
            "tie_objective_normalized_decision_l2_optimality_enclosure": normalized_decision_enclosure,
            "tie_objective_normalizer_mw": normalizer,
        }
    elif objective_kind == OBJECTIVE_FAIRNESS_QP and dual_objective.get("available") is True:
        lower = float(dual_objective["fairness_qp_certified_lower_bound_native_units"])
        dual_objective = {
            **dual_objective,
            "fairness_qp_primal_value_native_units": primal_objective,
            "fairness_qp_primal_minus_dual_lower_bound_native_units": primal_objective - lower,
            "fairness_qp_dual_lower_exceeds_primal_violation_native_units": max(0.0, lower - primal_objective),
        }
    return {
        "pass": passed,
        "primal_allocation_mw": x.tolist(), "primal_rho": rho,
        "primal_objective_native_units": primal_objective,
        "linear_duals": linear_records, "soc_dual_cones": soc_records,
        "stationarity": {"x": stationarity_x.tolist(), "rho": float(stationarity_rho), "inf_norm": stationarity_inf},
        "residuals": {
            "primal_inf": primal_inf,
            "primal_by_native_unit": primal_by_native_unit,
            "dual_cone_inf": dual_inf,
            "complementarity_inf": complementarity_inf,
        },
        "tolerances": {"kkt": dict(tolerances), "original_unit_primal": dict(native_tolerances)},
        "dual_objective": dual_objective,
    }


def _solver_version(adapter_id: str) -> str:
    try:
        if adapter_id == "CLARABEL":
            import clarabel

            return str(getattr(clarabel, "__version__", "unknown"))
        if adapter_id == "SCS":
            import scs

            return str(getattr(scs, "__version__", "unknown"))
    except Exception:  # pragma: no cover - adapter is still recorded fail-closed
        return "unavailable"
    return "unknown"


def _run_solver_attempt(program: Mapping[str, Any], *, solver: Any, adapter_id: str, settings: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, float | None]:
    """Solve once and replay KKT without changing any policy threshold."""
    problem = program["problem"]
    version = _solver_version(adapter_id)
    settings_payload = dict(settings)
    attempt = {
        "solver_adapter_id": adapter_id,
        "solver_version": version,
        "solver_settings": settings_payload,
        "solver_settings_hash": canonical_hash(settings_payload),
        "solver_status": None,
        "exception": None,
        "kkt_pass": False,
        "objective_value": None,
    }
    try:
        problem.solve(solver=solver, **settings_payload)
        solver_status = str(problem.status)
        kkt = replay_original_unit_kkt(program)
        objective_value = float(problem.value) if problem.value is not None and math.isfinite(float(problem.value)) else None
        attempt.update({"solver_status": solver_status, "kkt_pass": kkt.get("pass") is True, "objective_value": objective_value})
        return attempt, kkt, solver_status, objective_value
    except Exception as exc:
        attempt.update({"solver_status": f"EXCEPTION:{type(exc).__name__}", "exception": f"{type(exc).__name__}:{exc}"})
        return attempt, {"pass": False, "failure_reason": f"SOLVER_EXCEPTION:{type(exc).__name__}"}, str(attempt["solver_status"]), None


def _solve_with_strict_fallback(program: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, float | None, list[dict[str, Any]]]:
    """Try Clarabel, then SCS only if Clarabel lacks a passing KKT replay."""
    import cvxpy as cp

    clarabel_settings = {
        "verbose": False,
        "max_iter": 3000,
        "tol_gap_abs": 1.0e-12,
        "tol_gap_rel": 1.0e-12,
        "tol_feas": 1.0e-12,
    }
    first, first_kkt, first_status, first_objective = _run_solver_attempt(
        program,
        solver=cp.CLARABEL,
        adapter_id="CLARABEL",
        settings=clarabel_settings,
    )
    attempts = [first]
    if first_status == "optimal" and first_kkt.get("pass") is True:
        return first, first_kkt, first_status, first_objective, attempts
    # The fallback retains the *same* original-unit program and invokes the
    # same KKT replay/tolerances.  It cannot repair a result by relaxing a
    # policy acceptance threshold.
    scs_settings = {
        "verbose": False,
        "eps": 1.0e-9,
        "max_iters": 500000,
    }
    second, second_kkt, second_status, second_objective = _run_solver_attempt(
        program,
        solver=cp.SCS,
        adapter_id="SCS",
        settings=scs_settings,
    )
    attempts.append(second)
    return second, second_kkt, second_status, second_objective, attempts


def _blocked(
    *,
    objective_kind: str,
    reason: str,
    policy_mapping: Mapping[str, Any] | None,
    hierarchy: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    stage1_result: Mapping[str, Any] | None = None,
    reference_stage1_result: Mapping[str, Any] | None = None,
    reference_stage2_result: Mapping[str, Any] | None = None,
    blocker_detail: str | None = None,
) -> dict[str, Any]:
    result = {
        "serialization_id": SERIALIZATION_ID,
        "status": reason,
        "objective_kind": objective_kind,
        "policy_hash": None,
        "hierarchy_proof_hash": hierarchy.get("attestation_hash") if isinstance(hierarchy, Mapping) else None,
        "reference_proof_hash": reference.get("attestation_hash") if isinstance(reference, Mapping) else None,
        "stage1_parent_result_hash": stage1_result.get("result_hash") if isinstance(stage1_result, Mapping) else None,
        "reference_stage1_parent_result_hash": reference_stage1_result.get("result_hash") if isinstance(reference_stage1_result, Mapping) else None,
        "reference_stage2_parent_result_hash": reference_stage2_result.get("result_hash") if isinstance(reference_stage2_result, Mapping) else None,
        "blocker_detail": blocker_detail,
        "source_only": True,
        "AC_not_run": True,
        "candidate_or_evidence_eligible": False,
        "certification_scope": "SOURCE_ONLY_NUMERICAL_DIAGNOSTIC_NO_EVIDENCE_PROMOTION",
        "result_hash": None,
    }
    if isinstance(policy_mapping, Mapping):
        try:
            result["policy_hash"] = _policy_hash(policy_mapping)
        except SourceConicKKTError:
            result["policy_hash"] = None
    result["result_hash"] = canonical_hash(result)
    return result


def certify_source_conic_kkt(
    *,
    runtime: Mapping[str, Any],
    capacity_mw: Sequence[float],
    raw_request_mw: Sequence[float],
    objective_kind: str,
    policy_mapping: Mapping[str, Any] | None,
    hierarchy_proof: Mapping[str, Any] | None = None,
    stage1_result: Mapping[str, Any] | None = None,
    fairness_reference_proof: Mapping[str, Any] | None = None,
    reference_stage1_result: Mapping[str, Any] | None = None,
    reference_stage2_result: Mapping[str, Any] | None = None,
    claimed_allocation_mw: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Independently solve and replay a source-only original-unit KKT system.

    ``MAX_EXPORT_STAGE1`` can be certified from this program alone.  The tie
    stage requires the exact in-memory stage-one result; fairness requires the
    exact in-memory stage-one and certified stage-two max-export results.
    Their fixed-schema
    attestations bind those parent result hashes, allocations, objective
    semantics, policy formulation, and source task.  A static policy never
    contains a result-dependent allowlist.
    """
    if objective_kind not in OBJECTIVE_KINDS:
        return _blocked(objective_kind=objective_kind, reason="BLOCKED_UNKNOWN_OBJECTIVE", policy_mapping=policy_mapping)
    try:
        policy = validate_external_policy(policy_mapping)
    except SourceConicKKTError as exc:
        return _blocked(objective_kind=objective_kind, reason=f"BLOCKED_POLICY_MAPPING:{exc}", policy_mapping=policy_mapping, hierarchy=hierarchy_proof, reference=fairness_reference_proof)
    try:
        data = _runtime_data(runtime, capacity_mw, raw_request_mw)
        task_binding = source_task_binding(runtime=runtime, capacity_mw=capacity_mw, raw_request_mw=raw_request_mw)
        method_id = METHOD_ID_BY_OBJECTIVE[objective_kind]
        method_atom = source_method_atom_binding(source_task=task_binding, method_id=method_id)
    except SourceConicKKTError as exc:
        return _blocked(objective_kind=objective_kind, reason=f"BLOCKED_RUNTIME:{exc}", policy_mapping=policy_mapping, hierarchy=hierarchy_proof, reference=fairness_reference_proof)
    primary_floor: float | None = None
    weights: list[float] | None = None
    fairness_qdiag: list[float] | None = None
    fairness_linear: list[float] | None = None
    hierarchy_summary: dict[str, Any] | None = None
    reference_summary: dict[str, Any] | None = None
    tie: dict[str, Any] | None = None
    hierarchy_tolerance: dict[str, float | str] | None = None
    if objective_kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        try:
            tie = _tie_configuration(policy["policy"], constraint_constants=policy["constraint_constants"])
            hierarchy_tolerance = _hierarchy_tolerance_for_task(tie, runtime_data=data)
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_STATIC_TIE_POLICY_INVALID",
                policy_mapping=policy_mapping,
                blocker_detail=str(exc),
            )
    elif objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2:
        try:
            tie = _tie_configuration(policy["policy"], constraint_constants=policy["constraint_constants"])
            hierarchy_tolerance = _hierarchy_tolerance_for_task(tie, runtime_data=data)
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_STATIC_TIE_POLICY_INVALID",
                policy_mapping=policy_mapping,
                hierarchy=hierarchy_proof,
                stage1_result=stage1_result,
                blocker_detail=str(exc),
            )
        try:
            stage1_summary = _validated_stage1_result(
                stage1_result,
                task_binding=task_binding,
                policy_hash=policy["policy_hash"],
                tie=tie,
                hierarchy_tolerance=hierarchy_tolerance,
            )
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_STAGE1_PARENT_RESULT_INVALID",
                policy_mapping=policy_mapping,
                hierarchy=hierarchy_proof,
                stage1_result=stage1_result,
                blocker_detail=str(exc),
            )
        hierarchy_ok, hierarchy_summary = _attested_tie_hierarchy(
            hierarchy_proof,
            source_task_binding_hash=task_binding["source_task_binding_hash"],
            stage1=stage1_summary,
            tie=tie,
        )
        if not hierarchy_ok:
            return _blocked(
                objective_kind=objective_kind,
                reason=str(hierarchy_summary["status"]),
                policy_mapping=policy_mapping,
                hierarchy=hierarchy_proof,
                stage1_result=stage1_result,
                blocker_detail=str(hierarchy_summary.get("reason")),
            )
        primary_primal = evaluate_original_unit_source_primal(
            runtime=runtime,
            capacity_mw=data["capacity"],
            raw_request_mw=data["raw"],
            policy_mapping=policy["policy"],
            allocation_mw=hierarchy_summary["p1_allocation_mw"],
        )
        if primary_primal["pass"] is not True:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_PRIMARY_ALLOCATION_SOURCE_PRIMAL_FAIL",
                policy_mapping=policy_mapping,
                hierarchy=hierarchy_proof,
                stage1_result=stage1_result,
            )
        hierarchy_summary = {**hierarchy_summary, "primary_allocation_source_primal": primary_primal}
        primary_floor, weights = float(hierarchy_summary["primary_floor_u1_minus_tau_h_mw"]), list(hierarchy_summary["weights"])
    elif objective_kind == OBJECTIVE_FAIRNESS_QP:
        try:
            tie = _tie_configuration(policy["policy"], constraint_constants=policy["constraint_constants"])
            hierarchy_tolerance = _hierarchy_tolerance_for_task(tie, runtime_data=data)
            fairness = _fairness_configuration(policy["policy"], constraint_constants=policy["constraint_constants"])
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_STATIC_FAIRNESS_OR_TIE_POLICY_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail=str(exc),
            )
        try:
            reference_stage1_summary = _validated_stage1_result(
                reference_stage1_result,
                task_binding=task_binding,
                policy_hash=policy["policy_hash"],
                tie=tie,
                hierarchy_tolerance=hierarchy_tolerance,
            )
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_STAGE1_PARENT_RESULT_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail=str(exc),
            )
        try:
            stage2_summary = _validated_stage2_result(
                reference_stage2_result,
                task_binding=task_binding,
                policy_hash=policy["policy_hash"],
                tie=tie,
                stage1=reference_stage1_summary,
            )
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_STAGE2_PARENT_RESULT_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail=str(exc),
            )
        raw_reference_hierarchy = (
            reference_stage2_result.get("hierarchy_attestation")
            if isinstance(reference_stage2_result, Mapping)
            else None
        )
        if not isinstance(raw_reference_hierarchy, Mapping):
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_STAGE2_HIERARCHY_ATTESTATION_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail="REFERENCE_STAGE2_FULL_HIERARCHY_ATTESTATION_REQUIRED",
            )
        try:
            raw_hierarchy_hash = _hash_bound_mapping(raw_reference_hierarchy)
            raw_p1 = _nonnegative_number(raw_reference_hierarchy.get("p1_primal_total_mw"), name="reference hierarchy P1")
            raw_u1 = _nonnegative_number(raw_reference_hierarchy.get("u1_certified_upper_mw"), name="reference hierarchy U1")
            raw_tau_h = _nonnegative_number(raw_reference_hierarchy.get("tau_h_mw"), name="reference hierarchy tau_H")
            raw_floor = _finite_number(raw_reference_hierarchy.get("primary_floor_u1_minus_tau_h_mw"), name="reference hierarchy primary floor")
        except SourceConicKKTError as exc:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_STAGE2_HIERARCHY_ATTESTATION_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail=str(exc),
            )
        hierarchy_tolerance = float(tie["primary_total_replay_tolerance_mw"])
        if (
            raw_reference_hierarchy.get("stage1_result_hash") != reference_stage1_summary["result_hash"]
            or raw_reference_hierarchy.get("source_task_binding_hash") != task_binding["source_task_binding_hash"]
            or raw_reference_hierarchy.get("primary_face_mode") != "DUAL_UPPER_TOLERANCE_PRIMARY_FACE"
            or raw_reference_hierarchy.get("primary_face_constraint") != "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H"
            or abs(raw_p1 - float(reference_stage1_summary["p1_primal_total_mw"])) > hierarchy_tolerance
            or abs(raw_u1 - float(reference_stage1_summary["u1_certified_upper_mw"])) > hierarchy_tolerance
            or abs(raw_floor - (raw_u1 - raw_tau_h)) > hierarchy_tolerance
        ):
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_STAGE2_HIERARCHY_ATTESTATION_INVALID",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail="REFERENCE_STAGE2_HIERARCHY_ATTESTATION_PARENT_CHAIN_MISMATCH",
            )
        reference_ok, reference_summary = _attested_fairness_reference(
            fairness_reference_proof,
            source_task_binding_hash=task_binding["source_task_binding_hash"],
            stage1=reference_stage1_summary,
            stage2=stage2_summary,
            tie=tie,
            fairness=fairness,
        )
        if not reference_ok:
            return _blocked(
                objective_kind=objective_kind,
                reason=str(reference_summary["status"]),
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
                blocker_detail=str(reference_summary.get("reason")),
            )
        reference_summary = {
            **reference_summary,
            "reference_stage2_hierarchy_attestation_hash": raw_hierarchy_hash,
            "reference_stage2_hierarchy_attestation": dict(raw_reference_hierarchy),
        }
        reference = np.asarray(reference_summary["reference_allocation_mw"], dtype=float)
        if np.any(reference > np.minimum(np.asarray(data["capacity"]), np.asarray(data["raw"])) + 1.0e-12):
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_OUTSIDE_SOURCE_BOX",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
            )
        reference_primal = evaluate_original_unit_source_primal(
            runtime=runtime,
            capacity_mw=data["capacity"],
            raw_request_mw=data["raw"],
            policy_mapping=policy["policy"],
            allocation_mw=reference_summary["reference_allocation_mw"],
        )
        if reference_primal["pass"] is not True:
            return _blocked(
                objective_kind=objective_kind,
                reason="BLOCKED_REFERENCE_SOURCE_PRIMAL_FAIL",
                policy_mapping=policy_mapping,
                reference=fairness_reference_proof,
                reference_stage1_result=reference_stage1_result,
                reference_stage2_result=reference_stage2_result,
            )
        d = np.asarray(data["raw"], dtype=float) + float(reference_summary["epsilon_mw"])
        phi = float(np.mean(reference / d))
        alpha = float(reference_summary["alpha"])
        fairness_qdiag = (alpha / (d * d)).tolist()
        fairness_linear = (-(1.0 + 2.0 * alpha * phi / d)).tolist()
        reference_summary = {
            **reference_summary,
            "reference_fraction": phi,
            "reference_source_primal": reference_primal,
            "fairness_allocation_primary_face_constraint": "NONE",
            "fairness_allocation_semantics": (
                "FAIRNESS_TRADEOFF_OVER_PROXY_FEASIBLE_SET_WITH_SAME_CONTEXT_ME_REFERENCE"
            ),
        }
    try:
        program = build_original_unit_source_conic_program(
            runtime=runtime, capacity_mw=data["capacity"], raw_request_mw=data["raw"], objective_kind=objective_kind,
            policy_mapping=policy["policy"], primary_floor_mw=primary_floor, tie_weights=weights,
            fairness_qdiag=fairness_qdiag, fairness_linear=fairness_linear,
        )
    except SourceConicKKTError as exc:
        return _blocked(objective_kind=objective_kind, reason=f"BLOCKED_CONIC_BUILD:{exc}", policy_mapping=policy_mapping, hierarchy=hierarchy_proof, reference=fairness_reference_proof)
    if objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2 and isinstance(hierarchy_summary, Mapping):
        # Preserve the actual primary face in the result rather than leaving a
        # future verifier to infer it from the stage-two allocation.
        program["objective_metadata"].update(
            {
                "primary_face_constraint": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
                "p1_primal_total_mw": hierarchy_summary["p1_primal_total_mw"],
                "u1_certified_upper_mw": hierarchy_summary["u1_certified_upper_mw"],
                "stage1_dual_gap_mw": hierarchy_summary["stage1_dual_gap_mw"],
                "tau_h_mw": hierarchy_summary["tau_h_mw"],
                "hierarchy_tolerance_total_scale_definition": hierarchy_summary["hierarchy_tolerance_total_scale_definition"],
                "primary_floor_u1_minus_tau_h_mw": hierarchy_summary["primary_floor_u1_minus_tau_h_mw"],
                "lexicographic_semantics": "CERTIFIED_TAU_H_PRIMARY_FACE_WITH_CANONICAL_STRICT_CONVEX_TIE_BREAK",
            }
        )
    if objective_kind == OBJECTIVE_FAIRNESS_QP and isinstance(reference_summary, Mapping):
        # The certified Stage-2 allocation is the same-context reference in
        # the fairness objective.  The FA allocation itself remains a
        # total-export/fairness tradeoff over the proxy-feasible set.
        program["objective_metadata"].update(
            {
                "reference_stage1_result_hash": reference_summary["reference_stage1_result_hash"],
                "reference_stage2_result_hash": reference_summary["reference_stage2_result_hash"],
                "reference_stage2_method_atom_binding_hash": reference_summary["reference_method_atom_binding_hash"],
                "p1_primal_total_mw": reference_stage1_summary["p1_primal_total_mw"],
                "u1_certified_upper_mw": reference_summary["reference_stage2_certified_upper_u1_mw"],
                "tau_h_mw": reference_summary["reference_stage2_tau_h_mw"],
                "reference_primary_floor_u1_minus_tau_h_mw": reference_summary[
                    "reference_stage2_primary_floor_u1_minus_tau_h_mw"
                ],
                "reference_primary_face_constraint": reference_summary[
                    "reference_stage2_primary_face_constraint"
                ],
                "allocation_primary_face_constraint": "NONE",
                "allocation_semantics": (
                    "FAIRNESS_TRADEOFF_OVER_PROXY_FEASIBLE_SET_WITH_SAME_CONTEXT_ME_REFERENCE"
                ),
            }
        )
    try:
        selected_attempt, kkt, solver_status, objective_value, solver_attempts = _solve_with_strict_fallback(program)
    except Exception as exc:  # pragma: no cover - import/adapter construction failure
        return _blocked(
            objective_kind=objective_kind,
            reason=f"BLOCKED_CONIC_SOLVER:{type(exc).__name__}",
            policy_mapping=policy_mapping,
            hierarchy=hierarchy_proof,
            reference=fairness_reference_proof,
            stage1_result=stage1_result,
            reference_stage1_result=reference_stage1_result,
            reference_stage2_result=reference_stage2_result,
            blocker_detail=str(exc),
        )
    claimed_replay: dict[str, Any] | None = None
    if claimed_allocation_mw is not None and kkt.get("primal_allocation_mw") is not None:
        try:
            claimed = _finite_vector(claimed_allocation_mw, name="claimed_allocation_mw", length=PARTICIPANT_COUNT)
            gap = max(abs(left - right) for left, right in zip(claimed, kkt["primal_allocation_mw"]))
            claimed_replay = {"provided": True, "allocation_inf_gap_mw": gap, "pass": gap <= policy["kkt_tolerances"]["execution_replay_inf_mw"]}
        except SourceConicKKTError as exc:
            claimed_replay = {"provided": True, "pass": False, "failure_reason": str(exc)}
    else:
        claimed_replay = {"provided": False, "pass": None}
    stage1_primal_dual_certificate: dict[str, Any] | None = None
    minimization_primal_dual_certificate: dict[str, Any] | None = None
    if objective_kind == OBJECTIVE_MAX_EXPORT_STAGE1:
        # ``tie`` and ``hierarchy_tolerance`` were constructed from the static
        # policy before the solve.  They contain no result-derived selection.
        assert tie is not None and hierarchy_tolerance is not None
        stage1_primal_dual_certificate = _stage1_primal_dual_certificate(
            kkt=kkt,
            objective_value=objective_value,
            hierarchy_tolerance=hierarchy_tolerance,
            primary_total_replay_tolerance_mw=float(tie["primary_total_replay_tolerance_mw"]),
        )
        objective_certificate: dict[str, Any] = {
            "pass": stage1_primal_dual_certificate.get("pass") is True,
            "status": stage1_primal_dual_certificate.get("status"),
            "objective_kind": objective_kind,
            "objective_sense": "MAXIMIZE",
            "certified_objective_bound_orientation": "GLOBAL_DUAL_UPPER_BOUND",
            "objective_gap_orientation": "GLOBAL_DUAL_UPPER_BOUND_MINUS_PRIMAL",
            "primal_objective_value_native_units": stage1_primal_dual_certificate.get("p1_primal_total_mw"),
            "certified_objective_bound_native_units": stage1_primal_dual_certificate.get("u1_certified_upper_mw"),
            "objective_gap_native_units": stage1_primal_dual_certificate.get("u1_minus_p1_mw"),
            "solver_reported_objective_value_native_units": objective_value,
            "solver_reported_objective_replay_gap_native_units": stage1_primal_dual_certificate.get("objective_replay_gap_mw"),
        }
    else:
        minimization_primal_dual_certificate = _minimization_primal_dual_certificate(
            kkt=kkt,
            solver_objective_value=objective_value,
            objective_kind=objective_kind,
            objective_metadata=program["objective_metadata"],
        )
        objective_certificate = dict(minimization_primal_dual_certificate)
    try:
        objective_value_native_units = _finite_number(
            objective_certificate.get("primal_objective_value_native_units"),
            name="emitted primal objective",
        )
        certified_objective_bound_native_units = _finite_number(
            objective_certificate.get("certified_objective_bound_native_units"),
            name="emitted certified objective bound",
        )
        objective_gap_native_units = _nonnegative_number(
            objective_certificate.get("objective_gap_native_units"),
            name="emitted objective gap",
        )
    except SourceConicKKTError:
        # Keep the result serializable and fail closed below.  The explicit
        # ``None`` values make it impossible for a downstream adapter to
        # silently fall back to the raw solver objective as a purported bound.
        objective_value_native_units = None
        certified_objective_bound_native_units = None
        objective_gap_native_units = None
    certified = bool(
        solver_status == "optimal"
        and kkt.get("pass") is True
        and claimed_replay.get("pass") is not False
        and (stage1_primal_dual_certificate is None or stage1_primal_dual_certificate.get("pass") is True)
        and (minimization_primal_dual_certificate is None or minimization_primal_dual_certificate.get("pass") is True)
    )
    original_unit_system = {
        "allocation_unit": "MW", "branch_soc_t_unit": "MVA", "branch_soc_z_units": ["MW", "MVAr"],
        "voltage_unit": "pu", "rho_unit": "dimensionless", "constraint_count": len(program["descriptors"]),
        "soc_constraint_count": sum(item["kind"] == "SECOND_ORDER_CONE" for item in program["descriptors"]),
        "linear_constraint_count": sum(item["kind"] == "LINEAR_NONPOSITIVE" for item in program["descriptors"]),
        "constraint_manifest": [{key: item[key] for key in ("constraint_id", "kind", "family", "index", "units")} for item in program["descriptors"]],
    }
    formulation_hash = canonical_hash({"original_unit_system": original_unit_system, "source_task_binding": task_binding})
    primal_dual_pair_hash = canonical_hash(
        {
            "objective_kind": objective_kind,
            "primal_allocation_mw": kkt.get("primal_allocation_mw"),
            "primal_rho": kkt.get("primal_rho"),
            "objective_certificate": objective_certificate,
            "dual_objective": kkt.get("dual_objective"),
        }
    )
    dual_variable_hash = canonical_hash(
        {
            "linear_duals": kkt.get("linear_duals"),
            "soc_dual_cones": kkt.get("soc_dual_cones"),
            "dual_objective": kkt.get("dual_objective"),
        }
    )
    hierarchy_attestation = dict(hierarchy_proof) if objective_kind == OBJECTIVE_MAX_EXPORT_TIE_STAGE2 and isinstance(hierarchy_proof, Mapping) else None
    fairness_parent_chain: dict[str, Any] | None = None
    if objective_kind == OBJECTIVE_FAIRNESS_QP and isinstance(reference_summary, Mapping):
        raw_stage2_attestation = (
            reference_stage2_result.get("hierarchy_attestation")
            if isinstance(reference_stage2_result, Mapping)
            else None
        )
        fairness_parent_chain = {
            "reference_stage1_result_hash": reference_summary["reference_stage1_result_hash"],
            "reference_stage2_result_hash": reference_summary["reference_stage2_result_hash"],
            "reference_stage1_method_atom_binding_hash": reference_summary["reference_stage1_method_atom_binding_hash"],
            "reference_stage2_method_atom_binding_hash": reference_summary["reference_method_atom_binding_hash"],
            "source_task_binding_hash": task_binding["source_task_binding_hash"],
            "reference_stage2_hierarchy_attestation_hash": (
                raw_stage2_attestation.get("attestation_hash") if isinstance(raw_stage2_attestation, Mapping) else None
            ),
            "reference_primary_floor_u1_minus_tau_h_mw": reference_summary[
                "reference_stage2_primary_floor_u1_minus_tau_h_mw"
            ],
            "reference_primary_face_constraint": reference_summary[
                "reference_stage2_primary_face_constraint"
            ],
            "allocation_primary_face_constraint": "NONE",
            "allocation_semantics": (
                "FAIRNESS_TRADEOFF_OVER_PROXY_FEASIBLE_SET_WITH_SAME_CONTEXT_ME_REFERENCE"
            ),
        }
    result = {
        "serialization_id": SERIALIZATION_ID,
        "status": "NUMERICALLY_CERTIFIED_CONIC_PROGRAM" if certified else "KKT_SCAFFOLD_FAIL_CLOSED",
        "objective_kind": objective_kind,
        "method_id": method_id,
        "q_mode": data["q_mode"],
        "policy_id": policy["policy"]["policy_id"],
        "policy_hash": policy["policy_hash"],
        "runtime_bindings": {
            "envelope_hash": runtime.get("envelope_hash"), "proxy_model_hash": runtime.get("proxy_model_hash"),
            "synthetic_rate_policy_hash": runtime.get("synthetic_rate_policy_hash"), "effective_eq067_hash": runtime.get("effective_eq067_hash"),
        },
        "source_task_binding": task_binding,
        "method_atom_binding": method_atom,
        "method_atom_binding_hash": method_atom["method_atom_binding_hash"],
        "original_unit_system": original_unit_system,
        "formulation_hash": formulation_hash,
        "solver_status": solver_status,
        "solver_adapter": {
            "id": selected_attempt["solver_adapter_id"],
            "version": selected_attempt["solver_version"],
        },
        "solver_adapter_id": selected_attempt["solver_adapter_id"],
        "solver_version": selected_attempt["solver_version"],
        "solver_settings": selected_attempt["solver_settings"],
        "solver_settings_hash": selected_attempt["solver_settings_hash"],
        "solver_attempts": solver_attempts,
        "objective_value": objective_value,
        "solver_objective_value_native_units": objective_value,
        "allocation_mw": kkt.get("primal_allocation_mw"),
        "objective_value_native_units": objective_value_native_units,
        "certified_objective_bound_native_units": certified_objective_bound_native_units,
        "objective_gap_native_units": objective_gap_native_units,
        "certified_objective_bound_orientation": objective_certificate.get("certified_objective_bound_orientation"),
        "objective_gap_orientation": objective_certificate.get("objective_gap_orientation"),
        "objective_certificate": objective_certificate,
        "objective_bound_semantics": {
            "objective_sense": objective_certificate.get("objective_sense"),
            "bound_orientation": objective_certificate.get("certified_objective_bound_orientation"),
            "gap_orientation": objective_certificate.get("objective_gap_orientation"),
            "source": "PROJECTED_ORIGINAL_UNIT_CONE_DUAL_WITH_EXPLICIT_BOX_INFIMUM",
            "fairness_equivalence": (
                "CONCAVE_MAXIMIZATION_EQUIVALENT_CONVEX_MINIMIZATION"
                if objective_kind == OBJECTIVE_FAIRNESS_QP
                else None
            ),
        },
        "objective_metadata": program["objective_metadata"],
        "hierarchy_proof": hierarchy_summary,
        "hierarchy_attestation": hierarchy_attestation,
        # This alias makes the source provenance explicit for callers that
        # preserve the emitter payload before the ledger adapter emits its
        # own `source_hierarchy_attestation` field.  It is the exact original
        # self-hashed attestation, not a reconstructed summary.
        "source_hierarchy_attestation": hierarchy_attestation,
        "fairness_reference_proof": reference_summary,
        "fairness_reference_attestation": dict(fairness_reference_proof) if objective_kind == OBJECTIVE_FAIRNESS_QP and isinstance(fairness_reference_proof, Mapping) else None,
        "fairness_parent_chain": fairness_parent_chain,
        "stage1_primal_dual_certificate": stage1_primal_dual_certificate,
        "minimization_primal_dual_certificate": minimization_primal_dual_certificate,
        "parent_result_bindings": {
            "stage1_result_hash": hierarchy_summary.get("stage1_result_hash") if isinstance(hierarchy_summary, Mapping) else None,
            "reference_stage1_result_hash": reference_summary.get("reference_stage1_result_hash") if isinstance(reference_summary, Mapping) else None,
            "reference_stage2_result_hash": reference_summary.get("reference_stage2_result_hash") if isinstance(reference_summary, Mapping) else None,
        },
        "kkt": kkt,
        "primal_dual_pair_hash": primal_dual_pair_hash,
        "dual_variable_hash": dual_variable_hash,
        "claimed_execution_replay": claimed_replay,
        "source_only": True,
        "AC_not_run": True,
        "candidate_or_evidence_eligible": False,
        "certification_scope": "SOURCE_ONLY_NUMERICAL_DIAGNOSTIC_NO_EVIDENCE_PROMOTION",
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    return result


__all__ = [
    "FAIRNESS_METHOD_ID",
    "FAIRNESS_REFERENCE_ATTESTATION_SCHEMA_ID",
    "HIERARCHY_ATTESTATION_SCHEMA_ID",
    "MAX_EXPORT_METHOD_ID",
    "OBJECTIVE_FAIRNESS_QP",
    "OBJECTIVE_KINDS",
    "OBJECTIVE_MAX_EXPORT_STAGE1",
    "OBJECTIVE_MAX_EXPORT_TIE_STAGE2",
    "SERIALIZATION_ID",
    "SOURCE_METHOD_ATOM_BINDING_SCHEMA_ID",
    "SOURCE_TASK_BINDING_SCHEMA_ID",
    "SourceConicKKTError",
    "build_original_unit_source_conic_program",
    "certify_source_conic_kkt",
    "evaluate_original_unit_source_primal",
    "replay_original_unit_kkt",
    "source_method_atom_binding",
    "source_task_binding",
    "validate_external_policy",
]
