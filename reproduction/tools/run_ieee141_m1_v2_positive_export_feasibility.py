"""Development-only proxy feasibility probe for the non-empty domain gate.

This tool does not run AC, calibration, evaluation, or mitigation.  It solves
one small nonlinear proxy problem per train reporter and Q mode using the
already-generated state-dependent margin candidate.  The exploratory trust
region is explicit and is never promoted to a frozen policy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from r4r.serialization import canonical_dumps, canonical_hash
from r4r.synthetic_rate import load_synthetic_rate_policy
from tools.build_ieee141_m1_v2_directed_probe_spec import _proxy_values
from tools.ieee141_m1_candidate_guard import validate_diagnostic_unified_eq067_candidate


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _rho(values: list[float], capacity: list[float]) -> float:
    return math.sqrt(math.fsum((float(value) / float(limit)) ** 2 for value, limit in zip(values, capacity)))


def _constraint_slacks(
    model_q: dict[str, Any],
    rates: list[float],
    lower: float,
    upper: float,
    margin_q: dict[str, Any],
    capacity: list[float],
    radius: float,
    allocation: list[float],
    *,
    upper_guard_pu: float = 0.0,
) -> list[float]:
    """Return the registered proxy/intersection inequalities at ``allocation``.

    ``voltage_lower_pu`` is defined as ``V_proxy - V_AC``.  Consequently a
    conservative lower-voltage screen is ``V_proxy - (a_lower+L_lower*rho)
    >= Vmin``; adding the margin to the left-hand side would make the domain
    less safe as the radius grows.
    """
    if not math.isfinite(float(upper_guard_pu)) or float(upper_guard_pu) < 0.0:
        raise ValueError("upper_guard_pu must be finite and non-negative")
    pred = _proxy_values(model_q, allocation)
    rho = _rho(allocation, capacity)
    branch_from_a, branch_from_l = _margin_components(margin_q, "branch_from_mva", len(rates))
    branch_to_a, branch_to_l = _margin_components(margin_q, "branch_to_mva", len(rates))
    lower_a, lower_l = _margin_components(margin_q, "voltage_lower_pu", len(pred["voltage_pu"]))
    upper_a, upper_l = _margin_components(margin_q, "voltage_upper_pu", len(pred["voltage_pu"]), allow_none=True)
    result: list[float] = [float(radius) - rho]
    for i, rate in enumerate(rates):
        result.append(float(rate) - math.hypot(pred["branch_p_from_mw"][i], pred["branch_q_from_mvar"][i]) - branch_from_a[i] - branch_from_l[i] * rho)
        result.append(float(rate) - math.hypot(pred["branch_p_to_mw"][i], pred["branch_q_to_mvar"][i]) - branch_to_a[i] - branch_to_l[i] * rho)
    for i, voltage in enumerate(pred["voltage_pu"]):
        result.append(float(voltage) - float(lower) - lower_a[i] - lower_l[i] * rho)
        if upper_a[i] is not None:
            result.append(float(upper) - float(voltage) - float(upper_a[i]) - upper_l[i] * rho)
        else:
            # An inactive affine upper-voltage residual is not evidence that
            # the physical upper limit is irrelevant.  Keep an explicit
            # applicability guard so the allocation domain stays away from
            # the edge where an uncalibrated residual family could matter.
            result.append(float(upper) - float(voltage) - float(upper_guard_pu))
    return result


def _margin_components(margin_q: dict[str, Any], key: str, count: int, *, allow_none: bool = False) -> tuple[list[float | None], list[float]]:
    """Return only the registered post-safety effective coefficients.

    The shared guard has already checked the paired ``fit_*`` and
    ``effective_*`` arrays and their 1.25 relationship.  An effective-only
    reader prevents double safety scaling and cannot fall back to a legacy
    common-Q99 slope.
    """
    intercept_key = f"{key}_effective_intercept"
    slope_key = f"{key}_effective_slope"
    raw_intercept = margin_q.get(intercept_key)
    raw_slope = margin_q.get(slope_key)
    if not isinstance(raw_slope, list) or len(raw_slope) != count:
        raise ValueError(f"margin {key} effective slope shape is invalid")
    if not isinstance(raw_intercept, list) or len(raw_intercept) != count:
        raise ValueError(f"margin {key} effective intercept shape is invalid")
    intercept: list[float | None] = []
    slope: list[float] = []
    for a, l in zip(raw_intercept, raw_slope):
        if a is None and allow_none:
            if l is not None:
                raise ValueError(f"margin {key} inactive effective coefficients must be null/null")
            intercept.append(None); slope.append(0.0); continue
        if not isinstance(a, (int, float)) or not math.isfinite(float(a)):
            raise ValueError(f"margin {key} intercept contains nonfinite value")
        if not isinstance(l, (int, float)) or not math.isfinite(float(l)) or float(l) < 0.0:
            raise ValueError(f"margin {key} slope contains invalid value")
        intercept.append(float(a)); slope.append(float(l))
    return intercept, slope  # type: ignore[return-value]


def _positive_ray_regression(model_q: dict[str, Any], rates: list[float], lower: float, upper: float, margin_q: dict[str, Any], capacity: list[float], radius: float) -> list[dict[str, Any]]:
    """Check deterministic participant-wise positive rays independently."""
    rows: list[dict[str, Any]] = []
    for index, limit in enumerate(capacity):
        for scale in (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2):
            x = [0.0] * len(capacity)
            x[index] = min(float(limit) * scale, float(radius) * float(limit))
            if sum(x) <= 1.0e-4:
                continue
            slacks = _constraint_slacks(model_q, rates, lower, upper, margin_q, capacity, radius, x)
            rows.append({"participant_index": index, "scale": scale, "total_export_mw": sum(x), "minimum_constraint_slack": min(slacks), "feasible": min(slacks) >= -2.0e-7})
    return rows


def _solve_one(model_q: dict[str, Any], rates: list[float], lower: float, upper: float, margin_q: dict[str, Any], capacity: list[float], radius: float) -> dict[str, Any]:
    try:
        import numpy as np
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"status": "SOLVER_UNAVAILABLE", "failure_reason": type(exc).__name__, "max_feasible_total_export_mw": None}

    n = len(capacity); cap = np.asarray(capacity, dtype=float)
    if n != 30 or not np.all(np.isfinite(cap)) or np.any(cap <= 0.0):
        raise ValueError("positive-export domain requires 30 finite positive capacities")
    x_var = cp.Variable(n, nonneg=True, name="allocation_mw")
    t_var = cp.Variable(nonneg=True, name="rho")
    constraints = [x_var <= cap, t_var <= float(radius), cp.norm(cp.multiply(1.0 / cap, x_var), 2) <= t_var]
    base = model_q["proxy_base"]; mats = model_q["derivative_matrices"]
    branch_from_a, branch_from_l = _margin_components(margin_q, "branch_from_mva", len(rates))
    branch_to_a, branch_to_l = _margin_components(margin_q, "branch_to_mva", len(rates))
    lower_a, lower_l = _margin_components(margin_q, "voltage_lower_pu", len(base["voltage_pu"]))
    upper_a, upper_l = _margin_components(margin_q, "voltage_upper_pu", len(base["voltage_pu"]), allow_none=True)
    affine = lambda key, i: float(base[key][i]) + np.asarray(mats[key][i], dtype=float) @ x_var
    for i, rate in enumerate(rates):
        constraints.append(cp.norm(cp.hstack([affine("branch_p_from_mw", i), affine("branch_q_from_mvar", i)]), 2) + branch_from_a[i] + branch_from_l[i] * t_var <= float(rate))
        constraints.append(cp.norm(cp.hstack([affine("branch_p_to_mw", i), affine("branch_q_to_mvar", i)]), 2) + branch_to_a[i] + branch_to_l[i] * t_var <= float(rate))
    for i in range(len(base["voltage_pu"])):
        constraints.append(affine("voltage_pu", i) - lower_a[i] - lower_l[i] * t_var >= float(lower))
        if upper_a[i] is not None:
            constraints.append(affine("voltage_pu", i) + upper_a[i] + upper_l[i] * t_var <= float(upper))
        else:
            constraints.append(affine("voltage_pu", i) <= float(upper))
    problem = cp.Problem(cp.Maximize(cp.sum(x_var)), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception as exc:
        return {"status": "SOLVER_UNAVAILABLE", "failure_reason": type(exc).__name__, "solver_id": "clarabel", "max_feasible_total_export_mw": None}
    solver_status = str(problem.status)
    raw_x = x_var.value
    x = [float(v) for v in raw_x] if raw_x is not None else [0.0] * n
    t = float(t_var.value) if t_var.value is not None else 0.0
    residual = np.asarray(_constraint_slacks(model_q, rates, lower, upper, margin_q, capacity, radius, x), dtype=float)
    solver_feasible = solver_status in {"optimal", "optimal_inaccurate"} and float(np.min(residual)) >= -2.0e-7 and abs(t - _rho(x, capacity)) <= 2.0e-5
    rays = _positive_ray_regression(model_q, rates, lower, upper, margin_q, capacity, radius)
    positive = solver_feasible and sum(x) > 1.0e-4
    zero_feasible = solver_feasible and not positive
    dual_certificate = bool(zero_feasible and problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and len(constraints) > 0)
    return {
        "status": "FEASIBLE_POSITIVE_EXPORT" if positive else ("FEASIBLE_ZERO_EXPORT_ONLY" if zero_feasible and dual_certificate else "POSITIVE_EXPORT_FEASIBILITY_UNRESOLVED"),
        "solver_id": "clarabel",
        "solver_status": solver_status,
        "solver_success": bool(solver_feasible),
        "solver_message": "explicit SOCP with independent raw-inequality replay",
        "solver_attempts": [{"solver_id": "clarabel", "solver_status": solver_status, "total_export_mw": sum(x), "minimum_constraint_slack": float(np.min(residual)) if len(residual) else None, "feasible": bool(solver_feasible)}],
        "positive_ray_regression": rays,
        "dual_certificate": dual_certificate,
        "allocation_mw": x,
        "allocation_hash": canonical_hash({"values_mw": x}),
        "max_feasible_total_export_mw": float(sum(x)) if positive else (0.0 if zero_feasible and dual_certificate else None),
        "rho": _rho(x, capacity),
        "minimum_constraint_slack": float(min(residual)) if len(residual) else None,
        "auxiliary_norm_t": t,
        "trust_region_radius": float(radius),
    }


def run_positive_export_feasibility(*, policy_path: Path, proxy_model_path: Path, panel_path: Path, margin_path: Path, output_path: Path, radius: float = 1.0) -> dict[str, Any]:
    policy_payload = _load(policy_path); policy = load_synthetic_rate_policy(policy_payload)
    model = _load(proxy_model_path); panel = _load(panel_path); margin = _load(margin_path)
    if panel.get("status") != "DECLARED_NOT_EXECUTED" or margin.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("feasibility probe requires diagnostic panel and unfrozen margin candidate")
    if any(payload.get("policy_hash") != policy.policy_hash.to_json() for payload in (panel, model, margin)):
        raise ValueError("positive-export inputs are not bound to the same RATE policy")
    if panel.get("proxy_model_hash") != model.get("result_hash"):
        raise ValueError("positive-export panel/proxy binding mismatch")
    # The component reader below retains a legacy compatibility fallback for
    # historical diagnostic replays.  It must never be reachable from this
    # active domain probe: require the explicit Q-specific EQ067 candidate
    # before constructing any proxy constraint.
    validate_diagnostic_unified_eq067_candidate(margin)
    if radius <= 0.0 or not math.isfinite(radius): raise ValueError("radius must be positive and finite")
    capacities: dict[str, list[float]] = {}
    for row in panel.get("rows", []):
        if isinstance(row, dict) and isinstance(row.get("availability_vector_mw"), list):
            reporter = str(row["reporter_id"]); values = [float(v) for v in row["availability_vector_mw"]]
            if reporter in capacities and capacities[reporter] != values: raise ValueError("capacity changes within reporter")
            capacities[reporter] = values
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    rows: list[dict[str, Any]] = []
    for q_mode in ("Q0", "Q95"):
        margin_q = (margin.get("q_state_dependent_margin_candidate") or {}).get(q_mode)
        if not isinstance(margin_q, dict): raise ValueError(f"missing margin candidate for {q_mode}")
        for reporter in panel.get("train_reporter_ids", []):
            result = _solve_one(model["q_models"][q_mode], rates, float(policy_payload["voltage_lower_limit_pu"]), float(policy_payload["voltage_upper_limit_pu"]), margin_q, capacities[str(reporter)], radius)
            rows.append({"q_mode": q_mode, "reporter_id": reporter, "availability_vector_mw": capacities[str(reporter)], **result})
    positive = [row for row in rows if row.get("status") == "FEASIBLE_POSITIVE_EXPORT"]
    result: dict[str, Any] = {
        "serialization_id": "ieee141_m1_v2_positive_export_feasibility.v1",
        "status": "DIAGNOSTIC_ONLY",
        "policy_hash": policy.policy_hash.to_json(), "network_hash": policy.network_hash.to_json(), "proxy_model_hash": model.get("result_hash"), "panel_hash": panel.get("result_hash"), "margin_candidate_hash": margin.get("result_hash"),
        "exploratory_trust_region": {"radius": float(radius), "status": "NOT_FROZEN"},
        "rows": rows, "positive_export_exists": bool(positive), "non_empty_domain_status": "POSITIVE_EXPORT_FOUND" if positive else ("POSITIVE_EXPORT_FEASIBILITY_UNRESOLVED" if any(row.get("status") in {"POSITIVE_EXPORT_FEASIBILITY_UNRESOLVED", "SOLVER_UNAVAILABLE"} for row in rows) else "ZERO_EXPORT_FEASIBLE_BUT_POSITIVE_NOT_PROVEN"),
        "evaluation_results_read": False, "mitigation_results_read": False, "rate_policy_mutated": False, "diagnostic_only": True, "primary_evidence_eligible": False, "t3_eligible": False, "t4_eligible": False, "t5_eligible": False, "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result); output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8"); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "proxy-model", "panel", "margin", "output"): parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=1.0); args = parser.parse_args()
    print(canonical_dumps(run_positive_export_feasibility(policy_path=args.policy, proxy_model_path=args.proxy_model, panel_path=args.panel, margin_path=args.margin, output_path=args.output, radius=args.radius))); return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["run_positive_export_feasibility"]
