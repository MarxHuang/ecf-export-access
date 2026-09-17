"""Canonical V12.3 max-export stage hierarchy.

This module is the single active implementation of the max-export secondary
objective.  It deliberately separates the ideal Stage-1 optimum from the
numerically certified ``P1/U1`` interval and never imports the historical
``1e-9 MW`` lexicographic surrogate.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from tools.ieee141_m1_source_conic_kkt import (
    OBJECTIVE_MAX_EXPORT_STAGE1,
    OBJECTIVE_MAX_EXPORT_TIE_STAGE2,
    _solve_with_strict_fallback,
    build_original_unit_source_conic_program,
)
from tools.run_ieee141_m1_v2_positive_export_feasibility import _margin_components


PARTICIPANT_COUNT = 30
MAX_EXPORT_CANONICAL_ETA = 1.0e-2
HIERARCHY_ABSOLUTE_MW = 2.0e-7
HIERARCHY_RELATIVE_SCALE = 2.0e-8
HIERARCHY_STAGE1_FRACTION = 0.25
STAGE1_UPPER_ROUNDING_GUARD_MW = 1.0e-8
BOUND_REPAIR_TOLERANCE_MW = 1.0e-9
MAX_EXPORT_OBJECTIVE_ID = "CENTERED_INDEX_WEIGHTED_STRICTLY_CONVEX_CANONICAL_TIE_BREAK_V2"
MAX_EXPORT_WEIGHT_POLICY = "CENTERED_NORMALIZED_REGISTRY_ORDER"
MAX_EXPORT_OBJECTIVE_FORM = "centered_registry_index_weighted_x_plus_eta_over_2_ST_times_l2_x_squared"
MAX_EXPORT_POLICY_HASH = "F9A7E6E6A2544A0C1FB47ABBC3B60FB3BF953988A312E2FBCAC9A1CAB4538540"


def centered_registry_weights(count: int = PARTICIPANT_COUNT) -> list[float]:
    """Return exact normalized registry-order weights with zero mean."""
    if count < 1:
        raise ValueError("participant count must be positive")
    if count == 1:
        return [0.0]
    center = (float(count) - 1.0) / 2.0
    denominator = float(count - 1)
    values = [(float(index) - center) / denominator for index in range(count)]
    if abs(math.fsum(values)) > 1.0e-15 or min(values) < -0.5 - 1.0e-15 or max(values) > 0.5 + 1.0e-15:
        raise AssertionError("centered registry weights lost their frozen invariants")
    return values


def exact_s_t(capacity: Sequence[float], raw_request: Sequence[float]) -> float:
    """Compute the exact pre-solve allocation-box total ``S_T``."""
    if len(capacity) != len(raw_request) or len(capacity) == 0:
        raise ValueError("capacity and raw request must have equal nonzero length")
    values = [min(float(c), float(r)) for c, r in zip(capacity, raw_request)]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("allocation-box upper vector is invalid")
    return float(math.fsum(values))


def tau_h(s_t: float) -> float:
    if not math.isfinite(float(s_t)) or float(s_t) < 0.0:
        raise ValueError("S_T must be finite and nonnegative")
    return HIERARCHY_ABSOLUTE_MW + HIERARCHY_RELATIVE_SCALE * float(s_t)


def _canonicalize_box_residual(values: Sequence[float], upper: Sequence[float]) -> tuple[list[float], dict[str, Any]]:
    """Canonicalize solver round-off at the explicit allocation box.

    This is not a silent clip: only residuals within the registered
    ``BOUND_REPAIR_TOLERANCE_MW`` are snapped, and the applied indices and
    magnitudes are returned in the certificate.  Any larger violation fails
    closed.
    """
    if len(values) != len(upper):
        raise ValueError("allocation/box length mismatch")
    result: list[float] = []
    repairs: list[dict[str, float | int | str]] = []
    for index, (value, limit) in enumerate(zip(values, upper)):
        x = float(value); u = float(limit)
        if not math.isfinite(x) or not math.isfinite(u) or u < 0.0:
            raise ValueError("allocation/box contains a non-finite value")
        if x < 0.0:
            if -x > BOUND_REPAIR_TOLERANCE_MW:
                raise ValueError(f"allocation lower-bound violation at index {index}")
            repairs.append({"index": index, "side": "LOWER", "raw_mw": x, "canonical_mw": 0.0, "delta_mw": -x})
            x = 0.0
        elif x > u:
            if x - u > BOUND_REPAIR_TOLERANCE_MW:
                raise ValueError(f"allocation upper-bound violation at index {index}")
            repairs.append({"index": index, "side": "UPPER", "raw_mw": x, "canonical_mw": u, "delta_mw": u - x})
            x = u
        result.append(x)
    return result, {
        "applied": bool(repairs),
        "tolerance_mw": BOUND_REPAIR_TOLERANCE_MW,
        "repairs": repairs,
        "status": "EXPLICIT_FLOATING_POINT_BOUND_CANONICALIZATION" if repairs else "NOT_NEEDED",
    }


def canonical_objective_value(allocation: Sequence[float], *, s_t: float) -> float:
    """Replay ``wbar^T x + eta/(2*S_T)||x||^2`` in native MW units."""
    if s_t <= 0.0:
        raise ValueError("S_T=0 has no finite normalized secondary objective")
    x = np.asarray([float(value) for value in allocation], dtype=float)
    weights = np.asarray(centered_registry_weights(len(x)), dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("allocation contains a non-finite value")
    return float(weights @ x + MAX_EXPORT_CANONICAL_ETA / (2.0 * s_t) * (x @ x))


def canonical_gradient(allocation: Sequence[float], *, s_t: float) -> list[float]:
    if s_t <= 0.0:
        raise ValueError("S_T=0 has no finite normalized secondary objective")
    x = np.asarray([float(value) for value in allocation], dtype=float)
    weights = np.asarray(centered_registry_weights(len(x)), dtype=float)
    return (weights + MAX_EXPORT_CANONICAL_ETA / s_t * x).tolist()


def runtime_policy() -> dict[str, Any]:
    """Return the static source-conic policy needed for original-unit replay."""
    weights = centered_registry_weights()
    return {
        "policy_id": "IEEE141_V12_3_ACTIVE_MAX_EXPORT_CANONICAL_POLICY",
        "version": "1.0.0",
        "constraint_constants": {
            "branch_buffer_mva": 2.0e-6,
            "voltage_buffer_pu": 2.0e-6,
            "upper_voltage_guard_pu": 5.0e-3,
        },
        "original_unit_primal_tolerances": {
            "allocation_box_mw": 1.0e-7,
            "branch_soc_mva": 2.0e-6,
            "voltage_pu": 2.0e-8,
        "rho_dimensionless": 2.0e-8,
        },
        "kkt_tolerances": {
            "dual_cone_inf": 2.0e-6,
            "stationarity_inf": 2.0e-5,
            "complementarity_inf": 2.0e-5,
            "execution_replay_inf_mw": 1.0e-7,
        },
        "tie_break": {
            "weights": weights,
            "canonical_tie_objective_id": MAX_EXPORT_OBJECTIVE_ID,
            "canonical_tie_objective_form": MAX_EXPORT_OBJECTIVE_FORM,
            "canonical_linear_weight_scale": 1.0,
            "normalized_decision_l2_optimality_enclosure_max": 2.0e-3,
            "strict_convex_quadratic": {
                "eta_dimensionless": MAX_EXPORT_CANONICAL_ETA,
                "normalizer_definition": "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST",
                "rho_quadratic": "FORBIDDEN",
                "strict_convexity_rule": "unique_allocation_x_on_nonempty_closed_convex_certified_tau_H_face",
                "solver_objective_rescaling": "FORBIDDEN",
                "numerical_epsilon_tie_break": "FORBIDDEN",
            },
            "hierarchy_tolerance": {
                "formula_id": "ABSOLUTE_PLUS_RELATIVE_TOTAL_SCALE",
                "total_scale_definition": "SUM_ALLOCATION_BOX_UPPER_MW",
                "absolute_mw": HIERARCHY_ABSOLUTE_MW,
                "relative_total_scale": HIERARCHY_RELATIVE_SCALE,
                "stage1_gap_max": HIERARCHY_STAGE1_FRACTION * HIERARCHY_ABSOLUTE_MW,
                "stage1_upper_rounding_guard_mw": STAGE1_UPPER_ROUNDING_GUARD_MW,
            },
            "primary_total_replay_tolerance_mw": 1.0e-7,
            "tie_objective_replay_tolerance_native_units": 1.0e-8,
        },
    }


def _runtime(*, model_q: Mapping[str, Any], rates: Sequence[float], lower: float, upper: float, margin_q: Mapping[str, Any], rho_max: float, q_mode: str) -> dict[str, Any]:
    return {
        "q_mode": str(q_mode),
        "rates": [float(value) for value in rates],
        "lower": float(lower),
        "upper": float(upper),
        "rho_max": float(rho_max),
        "q_model": model_q,
        "margin_q": margin_q,
    }


def _build_reduced_stage2_problem(
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    raw_request: Sequence[float],
    rho_max: float,
    primary_floor_mw: float,
):
    """Build the equivalent Stage-2 model with ``rho=||x/C||`` eliminated.

    All rho coefficients in the registered proxy constraints are
    nonnegative, so every feasible point with a larger rho is no less
    restrictive.  Replacing the free epigraph variable by its norm therefore
    preserves the feasible allocation set while removing the flat rho
    direction that made the no-rho-quadratic formulation numerically
    degenerate.  The scientific objective remains exactly the frozen
    strict-convex function of ``x``.
    """
    import cvxpy as cp

    cap = np.asarray([float(value) for value in capacity], dtype=float)
    raw = np.asarray([float(value) for value in raw_request], dtype=float)
    x = cp.Variable(len(cap), nonneg=True, name="canonical_stage2_allocation_mw")
    rho = cp.norm(cp.multiply(1.0 / cap, x), 2)
    constraints: list[Any] = [x <= raw, x <= cap, rho <= float(rho_max), cp.sum(x) >= float(primary_floor_mw)]
    base = model_q["proxy_base"]
    matrices = model_q["derivative_matrices"]
    af, lf = _margin_components(margin_q, "branch_from_mva", len(rates))
    at, lt = _margin_components(margin_q, "branch_to_mva", len(rates))
    al, ll = _margin_components(margin_q, "voltage_lower_pu", len(base["voltage_pu"]))
    au, lu = _margin_components(margin_q, "voltage_upper_pu", len(base["voltage_pu"]), allow_none=True)

    def affine(field: str, index: int) -> Any:
        return float(base[field][index]) + np.asarray(matrices[field][index], dtype=float) @ x

    for index, rate in enumerate(rates):
        constraints.append(
            cp.norm(cp.hstack([affine("branch_p_from_mw", index), affine("branch_q_from_mvar", index)]), 2)
            + float(af[index]) + float(lf[index]) * rho <= float(rate) - 2.0e-6
        )
        constraints.append(
            cp.norm(cp.hstack([affine("branch_p_to_mw", index), affine("branch_q_to_mvar", index)]), 2)
            + float(at[index]) + float(lt[index]) * rho <= float(rate) - 2.0e-6
        )
    for index in range(len(base["voltage_pu"])):
        constraints.append(
            affine("voltage_pu", index) - float(al[index]) - float(ll[index]) * rho
            >= float(lower) + 2.0e-6
        )
        if au[index] is None:
            constraints.append(affine("voltage_pu", index) <= float(upper) - 2.0e-6 - 5.0e-3)
        else:
            constraints.append(
                affine("voltage_pu", index) + float(au[index]) + float(lu[index]) * rho
                <= float(upper) - 2.0e-6
            )
    weights = np.asarray(centered_registry_weights(len(cap)), dtype=float)
    s_t = exact_s_t(cap, raw)
    objective = cp.Minimize(weights @ x + MAX_EXPORT_CANONICAL_ETA / (2.0 * s_t) * cp.sum_squares(x))
    return cp.Problem(objective, constraints), x, rho


def solve_certified_max_export(
    *,
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: Sequence[float],
    raw_request: Sequence[float],
    rho_max: float,
    q_mode: str,
) -> dict[str, Any]:
    """Solve Stage 1 and the certified strict-convex Stage 2.

    Stage 1 is independently reconstructed in original units.  Its projected
    cone dual supplies ``U1``; Stage 2 is then built on ``sum(x)>=U1-tau_H``
    with the exact centered objective.  A zero ``S_T`` is handled explicitly
    rather than hidden behind a normalizer floor.
    """
    s_t = exact_s_t(capacity, raw_request)
    if s_t <= 0.0:
        return {
            "status": "ZERO_EXPORT_UNDEFINED_SECONDARY_OBJECTIVE",
            "solver_status": "NOT_RUN",
            "allocation": [0.0] * len(capacity),
            "stage1": {"p1_mw": 0.0, "u1_mw": 0.0, "gap_mw": 0.0, "pass": True},
            "stage2": {"pass": False, "failure_reason": "S_T_ZERO"},
            "s_t_mw": s_t,
            "tau_h_mw": tau_h(s_t),
        }
    runtime = _runtime(model_q=model_q, rates=rates, lower=lower, upper=upper, margin_q=margin_q, rho_max=rho_max, q_mode=q_mode)
    policy = runtime_policy()
    stage1_program = build_original_unit_source_conic_program(
        runtime=runtime,
        capacity_mw=capacity,
        raw_request_mw=raw_request,
        objective_kind=OBJECTIVE_MAX_EXPORT_STAGE1,
        policy_mapping=policy,
    )
    _, stage1_kkt, stage1_status, stage1_solver_objective, stage1_attempts = _solve_with_strict_fallback(stage1_program)
    stage1_x = stage1_kkt.get("primal_allocation_mw") if isinstance(stage1_kkt, Mapping) else None
    dual = stage1_kkt.get("dual_objective") if isinstance(stage1_kkt, Mapping) else None
    if not isinstance(stage1_x, list) or not isinstance(dual, Mapping) or dual.get("available") is not True:
        return {
            "status": "STAGE1_CERTIFICATE_UNAVAILABLE",
            "solver_status": stage1_status,
            "allocation": [0.0] * len(capacity),
            "stage1": {"pass": False, "kkt": dict(stage1_kkt) if isinstance(stage1_kkt, Mapping) else None},
            "s_t_mw": s_t,
            "tau_h_mw": tau_h(s_t),
        }
    p1 = float(math.fsum(float(value) for value in stage1_x))
    u1_raw = float(dual.get("max_export_certified_upper_u1_mw"))
    # The projected floating-point dual is already conservative, but the
    # primal/dual replay can still differ by a few ulps.  Add the frozen,
    # explicit MW rounding guard to the upper endpoint rather than accepting
    # a mathematically inverted P1/U1 interval.
    u1 = u1_raw + STAGE1_UPPER_ROUNDING_GUARD_MW
    gap = u1 - p1
    tau = tau_h(s_t)
    stage1_pass = bool(
        stage1_status == "optimal"
        and stage1_kkt.get("pass") is True
        and math.isfinite(u1)
        and gap >= -1.0e-8
        and gap <= HIERARCHY_STAGE1_FRACTION * tau
    )
    if not stage1_pass:
        return {
            "status": "STAGE1_CERTIFICATE_FAIL_CLOSED",
            "solver_status": stage1_status,
            "allocation": [float(value) for value in stage1_x],
            "stage1": {"pass": False, "p1_mw": p1, "u1_mw": u1, "u1_raw_mw": u1_raw, "upper_rounding_guard_mw": STAGE1_UPPER_ROUNDING_GUARD_MW, "gap_mw": gap, "tau_h_mw": tau, "kkt": dict(stage1_kkt)},
            "s_t_mw": s_t,
            "tau_h_mw": tau,
        }
    primary_floor = u1 - tau
    stage2_problem, stage2_variable, stage2_rho = _build_reduced_stage2_problem(
        model_q=model_q,
        rates=rates,
        lower=lower,
        upper=upper,
        margin_q=margin_q,
        capacity=capacity,
        raw_request=raw_request,
        rho_max=rho_max,
        primary_floor_mw=primary_floor,
    )
    stage2_attempts: list[dict[str, Any]] = []
    stage2_status = "SOLVER_NOT_RUN"
    stage2_solver_objective: float | None = None
    for solver_name, settings in (
        ("CLARABEL", {"max_iter": 2000, "tol_gap_abs": 1.0e-9, "tol_gap_rel": 1.0e-9, "tol_feas": 1.0e-9}),
        ("SCS", {"eps": 1.0e-8, "max_iters": 300000, "normalize": True}),
    ):
        try:
            import cvxpy as cp
            solver = getattr(cp, solver_name)
            stage2_problem.solve(solver=solver, verbose=False, **settings)
            stage2_status = str(stage2_problem.status)
            stage2_solver_objective = float(stage2_problem.value) if stage2_problem.value is not None and math.isfinite(float(stage2_problem.value)) else None
            stage2_attempts.append({"solver": solver_name, "status": stage2_status, "settings": settings, "objective_mw": stage2_solver_objective})
            if stage2_variable.value is not None and stage2_status in {"optimal", "optimal_inaccurate"}:
                break
        except Exception as exc:  # pragma: no cover - recorded solver fallback
            stage2_attempts.append({"solver": solver_name, "status": f"EXCEPTION:{type(exc).__name__}", "settings": settings})
    stage2_x_raw = [float(value) for value in stage2_variable.value] if stage2_variable.value is not None else None
    if stage2_x_raw is None:
        return {
            "status": "STAGE2_SOLVE_FAILED",
            "solver_status": stage2_status,
            "allocation": [float(value) for value in stage1_x],
            "stage1": {"pass": True, "p1_mw": p1, "u1_mw": u1, "u1_raw_mw": u1_raw, "upper_rounding_guard_mw": STAGE1_UPPER_ROUNDING_GUARD_MW, "gap_mw": gap},
            "stage2": {"pass": False, "kkt": None},
            "s_t_mw": s_t,
            "tau_h_mw": tau,
        }
    try:
        stage2_x, bound_repair = _canonicalize_box_residual(stage2_x_raw, np.minimum(np.asarray(capacity, dtype=float), np.asarray(raw_request, dtype=float)).tolist())
    except ValueError as exc:
        return {
            "status": "STAGE2_ALLOCATION_BOX_FAIL_CLOSED",
            "solver_status": stage2_status,
            "allocation": [float(value) for value in stage2_x_raw],
            "stage1": {"pass": True, "p1_mw": p1, "u1_mw": u1, "u1_raw_mw": u1_raw, "upper_rounding_guard_mw": STAGE1_UPPER_ROUNDING_GUARD_MW, "gap_mw": gap},
            "stage2": {"pass": False, "failure_reason": str(exc), "raw_allocation_mw": stage2_x_raw},
            "s_t_mw": s_t,
            "tau_h_mw": tau,
        }
    # Replay the reduced constraints at the canonicalized vector.  This keeps
    # the published allocation and all residuals on the same atom.
    stage2_variable.value = np.asarray(stage2_x, dtype=float)
    stage2_total = float(math.fsum(float(value) for value in stage2_x))
    objective = canonical_objective_value(stage2_x, s_t=s_t)
    replay_gap = abs(float(stage2_solver_objective) - objective) if stage2_solver_objective is not None else math.inf
    max_constraint_violation = max(
        (float(np.max(np.asarray(constraint.violation(), dtype=float))) for constraint in stage2_problem.constraints),
        default=math.inf,
    )
    rho_value = float(np.linalg.norm(np.asarray(stage2_x, dtype=float) / np.asarray(capacity, dtype=float)))
    stage2_kkt = {
        "pass": bool(math.isfinite(max_constraint_violation) and max_constraint_violation <= 5.0e-6),
        "status": "STAGE2_OBJECTIVE_GRADIENT_AND_PRIMAL_REPLAY",
        "objective_gradient": canonical_gradient(stage2_x, s_t=s_t),
        "strong_convexity_modulus_mw_inverse": MAX_EXPORT_CANONICAL_ETA / s_t,
        "primary_face_residual_mw": stage2_total - primary_floor,
        "max_constraint_violation": max_constraint_violation,
        "rho_replayed_dimensionless": rho_value,
        "solver_status": stage2_status,
        "bound_repair": bound_repair,
    }
    stage2_pass = bool(
        stage2_status in {"optimal", "optimal_inaccurate"}
        and stage2_kkt["pass"] is True
        and stage2_total >= primary_floor - 1.0e-7
        and math.isfinite(objective)
        and replay_gap <= 1.0e-8
    )
    regularization = {
        "objective_form": MAX_EXPORT_OBJECTIVE_FORM,
        "eta_dimensionless": MAX_EXPORT_CANONICAL_ETA,
        "normalizer_definition": "EXACT_SUM_OF_MIN_CAPACITY_AND_RAW_REQUEST_ST",
        "normalizer_mw": s_t,
        "x_quadratic_coefficient_mw_inverse": MAX_EXPORT_CANONICAL_ETA / (2.0 * s_t),
        "rho_quadratic": "FORBIDDEN",
    }
    return {
        "status": "CERTIFIED_MAX_EXPORT_STAGE2" if stage2_pass else "STAGE2_KKT_FAIL_CLOSED",
        "solver_status": stage2_status,
        "allocation": [float(value) for value in stage2_x],
        "rho": rho_value,
        "objective_mw": objective,
        "stage1": {
            "pass": True,
            "p1_mw": p1,
            "u1_mw": u1,
            "u1_raw_mw": u1_raw,
            "upper_rounding_guard_mw": STAGE1_UPPER_ROUNDING_GUARD_MW,
            "gap_mw": gap,
            "solver_objective_mw": stage1_solver_objective,
            "kkt": dict(stage1_kkt),
            "solver_attempts": stage1_attempts,
        },
        "stage2": {
            "pass": stage2_pass,
            "total_mw": stage2_total,
            "primary_floor_mw": primary_floor,
            "solver_objective_mw": stage2_solver_objective,
            "objective_mw": objective,
            "objective_replay_gap_mw": replay_gap,
            "kkt": dict(stage2_kkt),
            "solver_attempts": stage2_attempts,
            "raw_allocation_mw": stage2_x_raw,
            "bound_repair": bound_repair,
        },
        "s_t_mw": s_t,
        "tau_h_mw": tau,
        "primary_face": "SUM_X_GREATER_THAN_OR_EQUAL_TO_U1_MINUS_TAU_H",
        "objective_id": MAX_EXPORT_OBJECTIVE_ID,
        "policy_hash": MAX_EXPORT_POLICY_HASH,
        "objective_form": MAX_EXPORT_OBJECTIVE_FORM,
        "weight_policy": MAX_EXPORT_WEIGHT_POLICY,
        "weights": centered_registry_weights(len(capacity)),
        "eta_dimensionless": MAX_EXPORT_CANONICAL_ETA,
        "rho_quadratic": "FORBIDDEN",
        "strong_convexity_modulus_mw_inverse": MAX_EXPORT_CANONICAL_ETA / s_t,
        "objective_gradient": canonical_gradient(stage2_x, s_t=s_t),
        "regularization": dict(regularization) if isinstance(regularization, Mapping) else {},
    }


__all__ = [
    "MAX_EXPORT_CANONICAL_ETA",
    "MAX_EXPORT_OBJECTIVE_FORM",
    "MAX_EXPORT_OBJECTIVE_ID",
    "MAX_EXPORT_WEIGHT_POLICY",
    "MAX_EXPORT_POLICY_HASH",
    "centered_registry_weights",
    "exact_s_t",
    "tau_h",
    "canonical_objective_value",
    "canonical_gradient",
    "runtime_policy",
    "solve_certified_max_export",
]
