"""Audit legacy M1 allocations against the frozen state-dependent margin.

The common finite-difference M1 pilot uses a scalar Q99 margin and is retained
as a diagnostic only.  This tool replays the *same immutable allocations* with
the exact two-ended MVA/voltage observables and checks the registered
Q-specific affine margin ``a_k + L_k rho(x)``.  It never changes an allocation,
RATE policy, calibration artifact, or mitigation result.  The output is a
diagnostic bridge used to decide whether the state-dependent M1 contract can
replace the legacy extrapolating pilot on the primary path.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Mapping

from r4r.serialization import canonical_dumps, canonical_hash
from r4r.synthetic_rate import load_synthetic_rate_policy
from tools.ieee141_m1_candidate_guard import validate_diagnostic_unified_eq067_candidate
from tools.run_ieee141_m1_v2_directed_probe_ac_truth import _solve_one


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mva(p: list[float], q: list[float]) -> list[float]:
    return [math.hypot(float(a), float(b)) for a, b in zip(p, q)]


def _components(margin_q: Mapping[str, Any], key: str, count: int, *, allow_none: bool = False) -> tuple[list[float | None], list[float]]:
    raw_a = margin_q.get(f"{key}_effective_intercept")
    raw_l = margin_q.get(f"{key}_effective_slope")
    if not isinstance(raw_l, list) or len(raw_l) != count:
        raise ValueError(f"{key} effective slope shape mismatch")
    if not isinstance(raw_a, list) or len(raw_a) != count:
        raise ValueError(f"{key} effective intercept shape mismatch")
    intercept: list[float | None] = []
    slope: list[float] = []
    for a, l in zip(raw_a, raw_l):
        if a is None and allow_none:
            intercept.append(None)
            # An inactive residual family is represented by a pair of nulls,
            # not by a dimensionful zero slope.  Normalize the slope only for
            # arithmetic while preserving the explicit inactive marker in
            # the intercept vector.
            if l is not None:
                raise ValueError(f"{key} inactive effective coefficients must be null/null")
            slope.append(0.0)
            continue
        elif isinstance(a, (int, float)) and math.isfinite(float(a)):
            intercept.append(float(a))
        else:
            raise ValueError(f"{key} contains an invalid intercept")
        if not isinstance(l, (int, float)) or not math.isfinite(float(l)) or float(l) < 0.0:
            raise ValueError(f"{key} contains an invalid slope")
        slope.append(float(l))
    return intercept, slope


def _rho(allocation: list[float], capacity: list[float]) -> float:
    if len(allocation) != len(capacity) or any(float(c) <= 0.0 for c in capacity):
        raise ValueError("allocation/capacity shape or positivity mismatch")
    return math.sqrt(math.fsum((float(x) / float(c)) ** 2 for x, c in zip(allocation, capacity)))


def _margin_a_l(margin_q: Mapping[str, Any], count_branch: int, count_voltage: int) -> dict[str, tuple[list[float | None], list[float]]]:
    return {
        "branch_from": _components(margin_q, "branch_from_mva", count_branch),
        "branch_to": _components(margin_q, "branch_to_mva", count_branch),
        "voltage_lower": _components(margin_q, "voltage_lower_pu", count_voltage),
        "voltage_upper": _components(margin_q, "voltage_upper_pu", count_voltage, allow_none=True),
    }


def _screen_state_margin(
    *,
    predicted: Mapping[str, list[float]],
    actual: Mapping[str, list[float]],
    rates: list[float],
    lower: float,
    upper: float,
    margin_q: Mapping[str, Any],
    capacity: list[float],
    tolerance_mva: float,
    tolerance_pu: float,
    upper_guard_pu: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(float(upper_guard_pu)) or float(upper_guard_pu) < 0.0:
        raise ValueError("upper_guard_pu must be finite and non-negative")
    n_branch = len(rates)
    n_voltage = len(predicted["voltage_pu"])
    components = _margin_a_l(margin_q, n_branch, n_voltage)
    rho = _rho([float(v) for v in predicted.get("allocation_mw", [])], capacity) if "allocation_mw" in predicted else None
    # The proxy/AC values do not carry allocation in their typed maps; caller
    # supplies rho through the temporary field below.
    if rho is None:
        raise ValueError("internal state-margin screen missing allocation rho")
    (af, lf), (at, lt) = components["branch_from"], components["branch_to"]
    (al, ll), (au, lu) = components["voltage_lower"], components["voltage_upper"]
    pred_from = _mva(predicted["branch_p_from_mw"], predicted["branch_q_from_mvar"])
    pred_to = _mva(predicted["branch_p_to_mw"], predicted["branch_q_to_mvar"])
    actual_from = _mva(actual["branch_p_from_mw"], actual["branch_q_from_mvar"])
    actual_to = _mva(actual["branch_p_to_mw"], actual["branch_q_to_mvar"])
    branch_margin_from = [float(a or 0.0) + float(l) * rho for a, l in zip(af, lf)]
    branch_margin_to = [float(a or 0.0) + float(l) * rho for a, l in zip(at, lt)]
    lower_margin = [float(a or 0.0) + float(l) * rho for a, l in zip(al, ll)]
    upper_margin = [None if a is None else float(a) + float(l) * rho for a, l in zip(au, lu)]
    proxy_branch_pass = all(
        float(value) + margin <= float(rate) + tolerance_mva
        for value, margin, rate in zip(pred_from, branch_margin_from, rates)
    ) and all(
        float(value) + margin <= float(rate) + tolerance_mva
        for value, margin, rate in zip(pred_to, branch_margin_to, rates)
    )
    proxy_lower_pass = all(float(value) - margin >= lower - tolerance_pu for value, margin in zip(predicted["voltage_pu"], lower_margin))
    proxy_upper_pass = all(
        float(value) <= upper - float(upper_guard_pu) + tolerance_pu if margin is None else float(value) + margin <= upper + tolerance_pu
        for value, margin in zip(predicted["voltage_pu"], upper_margin)
    )
    upper_applicability_pass = all(
        float(value) <= upper - float(upper_guard_pu) + tolerance_pu
        for value, margin in zip(predicted["voltage_pu"], upper_margin)
        if margin is None
    )
    actual_branch_pass = all(max(float(a), float(b)) <= float(rate) + tolerance_mva for a, b, rate in zip(actual_from, actual_to, rates))
    actual_lower_pass = all(float(value) >= lower - tolerance_pu for value in actual["voltage_pu"])
    actual_upper_pass = all(float(value) <= upper + tolerance_pu for value in actual["voltage_pu"])
    branch_residual_from = [float(a) - float(p) for a, p in zip(actual_from, pred_from)]
    branch_residual_to = [float(a) - float(p) for a, p in zip(actual_to, pred_to)]
    lower_residual = [max(0.0, float(p) - float(a)) for p, a in zip(predicted["voltage_pu"], actual["voltage_pu"])]
    upper_residual = [max(0.0, float(a) - float(p)) for p, a in zip(predicted["voltage_pu"], actual["voltage_pu"])]
    envelope_pass = (
        all(residual <= margin + tolerance_mva for residual, margin in zip(branch_residual_from, branch_margin_from))
        and all(residual <= margin + tolerance_mva for residual, margin in zip(branch_residual_to, branch_margin_to))
        and all(residual <= margin + tolerance_pu for residual, margin in zip(lower_residual, lower_margin))
        and all(margin is None or residual <= float(margin) + tolerance_pu for residual, margin in zip(upper_residual, upper_margin))
    )
    failure_reasons: list[str] = []
    if not proxy_branch_pass or not proxy_lower_pass or not proxy_upper_pass:
        failure_reasons.append("M1_STATE_DEPENDENT_PROXY_DOMAIN_FAIL")
    if not envelope_pass:
        failure_reasons.append("M1_STATE_DEPENDENT_AC_RESIDUAL_EXCEEDANCE")
    if not actual_branch_pass:
        failure_reasons.append("M1_ACTUAL_BRANCH_EXCEEDANCE")
    if not actual_lower_pass or not actual_upper_pass:
        failure_reasons.append("M1_ACTUAL_VOLTAGE_EXCEEDANCE")
    return {
        "rho": rho,
        "proxy_branch_pass": proxy_branch_pass,
        "proxy_voltage_lower_pass": proxy_lower_pass,
        "proxy_voltage_upper_pass": proxy_upper_pass,
        "upper_voltage_applicability_guard_pu": float(upper_guard_pu),
        "upper_voltage_applicability_pass": upper_applicability_pass,
        "upper_voltage_applicability_status": (
            "INACTIVE_WITHIN_DEVELOPMENT_METHOD_DOMAIN"
            if any(value is None for value in upper_margin) and upper_applicability_pass
            else "ACTIVE_AFFINE_ENVELOPE"
            if not any(value is None for value in upper_margin)
            else "INACTIVE_OUTSIDE_DEVELOPMENT_METHOD_DOMAIN"
        ),
        "proxy_joint_pass": proxy_branch_pass and proxy_lower_pass and proxy_upper_pass,
        "ac_branch_pass": actual_branch_pass,
        "ac_voltage_lower_pass": actual_lower_pass,
        "ac_voltage_upper_pass": actual_upper_pass,
        "ac_joint_pass": actual_branch_pass and actual_lower_pass and actual_upper_pass,
        "envelope_pass": envelope_pass,
        "branch_residual_from_max_mva": max([0.0, *branch_residual_from]),
        "branch_residual_to_max_mva": max([0.0, *branch_residual_to]),
        "voltage_lower_residual_max_pu": max([0.0, *lower_residual]),
        "voltage_upper_residual_max_pu": max([0.0, *upper_residual]),
        "branch_margin_from_max_mva": max(branch_margin_from, default=0.0),
        "branch_margin_to_max_mva": max(branch_margin_to, default=0.0),
        "voltage_lower_margin_max_pu": max(lower_margin, default=0.0),
        "voltage_upper_margin_max_pu": None if all(value is None for value in upper_margin) else max(float(value) for value in upper_margin if value is not None),
        "failure_reasons": failure_reasons,
    }


def audit_state_dependent_realized_allocations(*, case_path: Path, policy_path: Path, proxy_model_path: Path, margin_path: Path, split_path: Path, q0_grid_path: Path, q95_grid_path: Path, output_path: Path, workers: int = 12) -> dict[str, Any]:
    policy_payload = _load(policy_path)
    policy = load_synthetic_rate_policy(policy_payload)
    model = _load(proxy_model_path)
    margin = _load(margin_path)
    split = _load(split_path)
    if model.get("policy_hash") != policy.policy_hash.to_json() or margin.get("policy_hash") != policy.policy_hash.to_json():
        raise ValueError("state-dependent audit inputs are not policy-bound")
    # Keep the realized-allocation replay on the same repaired contract as the
    # active method runner.  Without this precondition, its local affine
    # reader could silently interpret a legacy scalar Q99 margin as a
    # state-dependent envelope.
    validate_diagnostic_unified_eq067_candidate(margin)
    margin_payload = margin.get("q_state_dependent_margin_candidate")
    if not isinstance(margin_payload, dict):
        raise ValueError("margin candidate lacks Q-specific payload")
    evaluation_ids = [str(item) for item in split.get("evaluation_scenario_ids", [])]
    development_ids = [str(item) for item in split.get("development_scenario_ids", [])]
    holdout_ids = [str(item) for item in split.get("holdout_scenario_ids", [])]
    all_ids = development_ids + holdout_ids + evaluation_ids
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    lower = float(policy_payload["voltage_lower_limit_pu"]); upper = float(policy_payload["voltage_upper_limit_pu"])
    tol_mva = float(policy_payload["branch_tolerance_mva"]); tol_pu = float(policy_payload["voltage_tolerance_pu"])
    tasks: list[tuple[Any, ...]] = []
    metadata: list[dict[str, Any]] = []
    grid_hashes: dict[str, str] = {}
    for q_mode, grid_path in (("Q0", q0_grid_path), ("Q95", q95_grid_path)):
        payload = _load(grid_path)
        if payload.get("q_mode") != q_mode or payload.get("synthetic_rate_policy_hash") != policy.policy_hash.to_json():
            raise ValueError(f"{q_mode} grid policy/Q binding failed")
        grid = payload.get("grid"); batches = grid.get("batches") if isinstance(grid, dict) else None
        if not isinstance(batches, list) or {str(batch.get("reference_batch", {}).get("scenario_id", "")).removesuffix("__reference") for batch in batches} != set(all_ids):
            raise ValueError(f"{q_mode} grid is not the frozen full reporter grid")
        grid_hashes[q_mode] = str(payload.get("grid_hash"))
        by_id = {str(batch["reference_batch"]["scenario_id"]).removesuffix("__reference"): batch for batch in batches}
        for scenario_id in all_ids:
            batch = by_id[scenario_id]
            pair_capacity = {str(item["method_id"]): item["capacity_vector"] for item in batch.get("method_pairs", [])}
            for side_key in ("reference_batch", "reported_batch"):
                side = batch[side_key]
                for run in sorted(side.get("method_runs", []), key=lambda item: str(item.get("method_id"))):
                    method_id = str(run["method_id"])
                    allocation = [float(value) for value in run["proxy_audit"]["allocation_mw"]]
                    capacity_payload = pair_capacity.get(method_id)
                    if not isinstance(capacity_payload, dict) or not isinstance(capacity_payload.get("values_mw"), list):
                        raise ValueError("method run lacks typed capacity vector")
                    probe = {
                        "target_id": f"STATE_DEPENDENT::{q_mode}::{scenario_id}::{side_key}::{method_id}",
                        "target_kind": "STATE_DEPENDENT_REALIZED_ALLOCATION",
                        "status": "FIXED_ALLOCATION_REPLAY",
                        "q_mode": q_mode,
                        "reporter_id": scenario_id,
                        "allocation_side": side_key.removesuffix("_batch"),
                        "method_id": method_id,
                        "allocation_mw": allocation,
                        "allocation_hash": run["proxy_audit"].get("allocation_hash"),
                    }
                    tasks.append((str(case_path), policy_payload, model, probe))
                    metadata.append({"q_mode": q_mode, "scenario_id": scenario_id, "side": side_key.removesuffix("_batch"), "method_id": method_id, "allocation_mw": allocation, "capacity": [float(value) for value in capacity_payload["values_mw"]], "capacity_vector_hash": capacity_payload.get("capacity_vector_hash"), "source_grid_hash": payload.get("grid_hash")})
    if workers <= 1:
        solved = [_solve_one(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            solved = list(executor.map(_solve_one, tasks))
    rows: list[dict[str, Any]] = []
    for meta, result in zip(metadata, solved):
        q_margin = margin_payload[meta["q_mode"]]
        predicted = dict(result["predicted_values"]); actual = dict(result["actual_values"])
        predicted["allocation_mw"] = list(meta["allocation_mw"])
        state = _screen_state_margin(predicted=predicted, actual=actual, rates=rates, lower=lower, upper=upper, margin_q=q_margin, capacity=meta["capacity"], tolerance_mva=tol_mva, tolerance_pu=tol_pu)
        rows.append({**meta, "allocation_hash": result.get("allocation_hash"), "solution_hash": result.get("solution_hash"), "solver_success": result.get("solver_success"), "solver_status": result.get("solver_status"), "actual_screen": result.get("actual_screen"), "state_dependent_screen": state})
    rows.sort(key=lambda row: (row["q_mode"], all_ids.index(row["scenario_id"]), row["side"], row["method_id"]))
    summary: dict[str, Any] = {"row_count": len(rows), "proxy_joint_pass_rate": sum(bool(row["state_dependent_screen"]["proxy_joint_pass"]) for row in rows) / max(1, len(rows)), "envelope_pass_rate": sum(bool(row["state_dependent_screen"]["envelope_pass"]) for row in rows) / max(1, len(rows)), "ac_joint_pass_rate": sum(bool(row["state_dependent_screen"]["ac_joint_pass"]) for row in rows) / max(1, len(rows)), "state_dependent_joint_pass_rate": sum(bool(row["state_dependent_screen"]["proxy_joint_pass"] and row["state_dependent_screen"]["envelope_pass"] and row["state_dependent_screen"]["ac_joint_pass"]) for row in rows) / max(1, len(rows)), "by_q_mode": {}}
    for q_mode in ("Q0", "Q95"):
        q_rows = [row for row in rows if row["q_mode"] == q_mode]
        summary["by_q_mode"][q_mode] = {"row_count": len(q_rows), "proxy_joint_pass_rate": sum(bool(row["state_dependent_screen"]["proxy_joint_pass"]) for row in q_rows) / max(1, len(q_rows)), "envelope_pass_rate": sum(bool(row["state_dependent_screen"]["envelope_pass"]) for row in q_rows) / max(1, len(q_rows)), "ac_joint_pass_rate": sum(bool(row["state_dependent_screen"]["ac_joint_pass"]) for row in q_rows) / max(1, len(q_rows))}
    result: dict[str, Any] = {"serialization_id": "ieee141_m1_state_dependent_realized_allocation_audit.v1", "status": "DIAGNOSTIC_STATE_DEPENDENT_REALIZED_AUDIT_COMPLETE", "policy_hash": policy.policy_hash.to_json(), "network_hash": policy.network_hash.to_json(), "proxy_model_hash": model.get("result_hash"), "margin_candidate_hash": margin.get("result_hash"), "split_hash": canonical_hash({key: value for key, value in split.items() if key not in {"result_hash"}}), "q_grid_hashes": grid_hashes, "scenario_ids": all_ids, "rows": rows, "summary": summary, "evaluation_results_read": False, "mitigation_results_read": False, "rate_policy_mutated": False, "allocation_resolved": False, "diagnostic_only": True, "primary_evidence_eligible": False, "t3_eligible": False, "t4_eligible": False, "t5_eligible": False, "result_hash": None}
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join(["# State-dependent M1 realized-allocation audit (diagnostic)", "", f"- Result hash: `{result['result_hash']}`", f"- Rows: `{len(rows)}`", f"- State-dependent joint pass rate: `{summary['state_dependent_joint_pass_rate']}`", "- This replay does not alter RATE, allocations, calibration, or mitigation and is not T3/T4 evidence."]) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "policy", "proxy-model", "margin", "split", "q0-grid", "q95-grid", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    result = audit_state_dependent_realized_allocations(case_path=args.case, policy_path=args.policy, proxy_model_path=args.proxy_model, margin_path=args.margin, split_path=args.split, q0_grid_path=args.q0_grid, q95_grid_path=args.q95_grid, output_path=args.output, workers=args.workers)
    print(canonical_dumps({"status": result["status"], "summary": result["summary"], "result_hash": result["result_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_state_dependent_realized_allocations"]
