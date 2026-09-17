"""Run AC truth only on an already frozen M1 V2 directed probe spec.

The directed probe spec is consumed as an immutable input.  This runner does
not choose, move, clip, or re-solve a probe and does not write RATE, proxy,
evaluation, calibration, or mitigation artifacts.  Its output is a
development-only diagnostic comparison of proxy versus exact AC truth using
the two-ended apparent-power screen.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from r4r.ac_solver import solve_ac_power_flow
from r4r.injection_adapter import build_net_injections_from_operating_point_spec
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.synthetic_rate import load_synthetic_rate_policy
from r4r.types import QAssumption
from tools.diagnostic_ieee141_context import build_ieee141_diagnostic_input_context
from tools.run_ieee141_reporter_scan import _q0, _q95


FIELDS = ("branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar", "voltage_pu")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mva(p: Sequence[float], q: Sequence[float]) -> list[float]:
    return [math.hypot(float(pv), float(qv)) for pv, qv in zip(p, q)]


def _proxy_values(model_q: Mapping[str, Any], allocation: Sequence[float]) -> dict[str, list[float]]:
    base = model_q["proxy_base"]
    matrices = model_q["derivative_matrices"]
    return {
        field: [
            float(value) + math.fsum(float(a) * float(b) for a, b in zip(row, allocation))
            for value, row in zip(base[field], matrices[field])
        ]
        for field in FIELDS
    }


def _screen(values: Mapping[str, Sequence[float]], rates: Sequence[float], lower: float, upper: float, tolerance_mva: float, tolerance_pu: float) -> dict[str, Any]:
    from_mva = _mva(values["branch_p_from_mw"], values["branch_q_from_mvar"])
    to_mva = _mva(values["branch_p_to_mw"], values["branch_q_to_mvar"])
    branch_mva = [max(left, right) for left, right in zip(from_mva, to_mva)]
    excess = [value - float(rate) for value, rate in zip(branch_mva, rates)]
    branch_excess = max([0.0, *excess])
    voltage = [float(value) for value in values["voltage_pu"]]
    lower_excess = max([0.0, *(float(lower) - value for value in voltage)])
    upper_excess = max([0.0, *(value - float(upper) for value in voltage)])
    return {
        "branch_mva_from": from_mva,
        "branch_mva_to": to_mva,
        "branch_mva_max": branch_mva,
        "branch_excess_mva": branch_excess,
        "branch_offending_indices": [index + 1 for index, value in enumerate(excess) if value > float(tolerance_mva)],
        "branch_pass": branch_excess <= float(tolerance_mva),
        "voltage_lower_excess_pu": lower_excess,
        "voltage_upper_excess_pu": upper_excess,
        "voltage_pass": lower_excess <= float(tolerance_pu) and upper_excess <= float(tolerance_pu),
        "joint_pass": branch_excess <= float(tolerance_mva) and lower_excess <= float(tolerance_pu) and upper_excess <= float(tolerance_pu),
    }


def _solve_one(task: tuple[Any, ...]) -> dict[str, Any]:
    case_path, policy_payload, proxy_model, probe = task
    q_mode = str(probe["q_mode"])
    q_spec = _q0() if q_mode == "Q0" else _q95()
    context = build_ieee141_diagnostic_input_context(
        case_path,
        rating_mw_by_bus={bus_id: 1.0 for bus_id in (8, 9, 12, 13, 17, 20, 21, 23, 26, 27, 29, 32, 34, 35, 36, 37, 39, 41, 44, 48, 49, 51, 52, 53, 56, 58, 59, 61, 62, 64)},
        availability_mw_by_bus={bus_id: 1.0 for bus_id in (8, 9, 12, 13, 17, 20, 21, 23, 26, 27, 29, 32, 34, 35, 36, 37, 39, 41, 44, 48, 49, 51, 52, 53, 56, 58, 59, 61, 62, 64)},
        reactive_specification=q_spec,
    )
    allocation = [float(value) for value in probe.get("allocation_mw", [])]
    injection = build_net_injections_from_operating_point_spec(
        context.parsed_case,
        context.selection,
        allocation,
        context.operating_point,
        q_spec,
        formal_execution=False,
    )
    ac_input = {
        "network_hash": str(policy_payload["network_hash"]),
        "q_mode": q_mode,
        "p_injection_mw": list(injection.p_mw.values),
        "q_injection_mvar": list(injection.q_mvar.values),
    }
    solution = solve_ac_power_flow(context.parsed_case, p_injection_mw=injection.p_mw, q_injection_mvar=injection.q_mvar)
    actual = {
        "branch_p_from_mw": list(solution.branch_p_from_mw.values),
        "branch_q_from_mvar": list(solution.branch_q_from_mvar.values),
        "branch_p_to_mw": list(solution.branch_p_to_mw.values),
        "branch_q_to_mvar": list(solution.branch_q_to_mvar.values),
        "voltage_pu": list(solution.bus_voltage_pu.values),
    }
    predicted = _proxy_values(proxy_model["q_models"][q_mode], allocation)
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    lower = float(policy_payload["voltage_lower_limit_pu"])
    upper = float(policy_payload["voltage_upper_limit_pu"])
    tolerance_mva = float(policy_payload["branch_tolerance_mva"])
    tolerance_pu = float(policy_payload["voltage_tolerance_pu"])
    predicted_screen = _screen(predicted, rates, lower, upper, tolerance_mva, tolerance_pu)
    actual_screen = _screen(actual, rates, lower, upper, tolerance_mva, tolerance_pu)
    actual_mva = actual_screen["branch_mva_max"]
    predicted_mva = predicted_screen["branch_mva_max"]
    branch_residual = [float(actual_value) - float(predicted_value) for actual_value, predicted_value in zip(actual_mva, predicted_mva)]
    voltage_residual = [float(actual_value) - float(predicted_value) for actual_value, predicted_value in zip(actual["voltage_pu"], predicted["voltage_pu"])]
    target_branch_index = probe.get("target_branch_index")
    target_voltage_index = probe.get("target_voltage_index")
    return {
        "target_id": probe["target_id"],
        "target_kind": probe["target_kind"],
        "target_status": probe["status"],
        "q_mode": q_mode,
        "reporter_id": probe.get("reporter_id"),
        "allocation_side": probe.get("allocation_side"),
        "method_id": probe.get("method_id"),
        "allocation_hash": probe.get("allocation_hash"),
        "allocation_mw": allocation,
        "lambda_scale": probe.get("lambda_scale"),
        "target_branch_index": target_branch_index,
        "target_branch_id": probe.get("target_branch_id"),
        "target_voltage_index": target_voltage_index,
        "target_voltage_bus_id": probe.get("target_voltage_bus_id"),
        "solver_success": bool(solution.solver_result.solve_success),
        "solver_status": solution.solver_result.raw_status.value,
        "solver_failure_reasons": [reason.value for reason in solution.solver_result.failure_reasons.values],
        "solution_hash": solution.solution_hash.to_json(),
        "ac_input": ac_input,
        "ac_input_hash": canonical_hash(ac_input),
        # Preserve the raw observable vectors so a read-only verifier can
        # recompute the two-ended MVA screen and directional residuals rather
        # than trusting post-processed flags and arrays.
        "predicted_values": predicted,
        "actual_values": actual,
        "predicted_values_hash": canonical_hash(predicted),
        "actual_values_hash": canonical_hash(actual),
        "solver_evidence": {
            "raw_primal_residual": list(solution.solver_result.raw_primal_residual.values),
            "scaled_primal_residual": list(solution.solver_result.scaled_primal_residual.values),
            "residual_norm": float(solution.solver_result.residual_norm.value),
            "tolerance_reference": solution.solver_result.tolerance_reference.value,
            "bus_p_mismatch_mw": list(solution.bus_p_mismatch_mw.values),
            "bus_q_mismatch_mvar": list(solution.bus_q_mismatch_mvar.values),
            "global_p_balance_residual_mw": float(solution.global_p_balance_residual_mw.value),
            "global_q_balance_residual_mvar": float(solution.global_q_balance_residual_mvar.value),
            "slack_p_mw": float(solution.slack_p_mw.value),
            "slack_q_mvar": float(solution.slack_q_mvar.value),
            "total_active_loss_mw": float(solution.total_active_loss_mw.value),
            "total_reactive_loss_mvar": float(solution.total_reactive_loss_mvar.value),
        },
        "predicted_screen": predicted_screen,
        "actual_screen": actual_screen,
        "branch_false_safe": bool(predicted_screen["branch_pass"] and not actual_screen["branch_pass"]),
        "voltage_false_safe": bool(predicted_screen["voltage_pass"] and not actual_screen["voltage_pass"]),
        "joint_false_safe": bool(predicted_screen["joint_pass"] and not actual_screen["joint_pass"]),
        "branch_residual_max_signed_mva": max(branch_residual, default=0.0),
        "branch_residual_from_mva": [float(a) - float(p) for a, p in zip(actual_screen["branch_mva_from"], predicted_screen["branch_mva_from"])],
        "branch_residual_to_mva": [float(a) - float(p) for a, p in zip(actual_screen["branch_mva_to"], predicted_screen["branch_mva_to"])],
        "voltage_residual_pu": voltage_residual,
        "branch_underestimate_max_mva": max([0.0, *branch_residual]),
        "branch_overestimate_max_mva": max([0.0, *(-value for value in branch_residual)]),
        "voltage_residual_max_signed_pu": max(voltage_residual, default=0.0),
        "voltage_underestimate_max_pu": max([0.0, *voltage_residual]),
        "voltage_overestimate_max_pu": max([0.0, *(-value for value in voltage_residual)]),
        # Directional safety residuals: lower-bound screening is endangered
        # when the proxy voltage is above AC truth; upper-bound screening is
        # endangered when AC truth is above the proxy.
        "voltage_lower_safety_residual_max_pu": max([0.0, *(float(predicted_value) - float(actual_value) for predicted_value, actual_value in zip(predicted["voltage_pu"], actual["voltage_pu"]))]),
        "voltage_upper_safety_residual_max_pu": max([0.0, *(float(actual_value) - float(predicted_value) for predicted_value, actual_value in zip(predicted["voltage_pu"], actual["voltage_pu"]))]),
        "target_branch_predicted_ratio": predicted_mva[target_branch_index] / rates[target_branch_index] if isinstance(target_branch_index, int) else None,
        "target_branch_actual_ratio": actual_mva[target_branch_index] / rates[target_branch_index] if isinstance(target_branch_index, int) else None,
        "target_voltage_predicted_pu": predicted["voltage_pu"][target_voltage_index] if isinstance(target_voltage_index, int) else None,
        "target_voltage_actual_pu": actual["voltage_pu"][target_voltage_index] if isinstance(target_voltage_index, int) else None,
    }


def run_directed_probe_ac_truth(*, case_path: Path, policy_path: Path, proxy_model_path: Path, probe_spec_path: Path, output_path: Path, workers: int = 6) -> dict[str, Any]:
    policy_payload = _load(policy_path)
    policy = load_synthetic_rate_policy(policy_payload)
    proxy_model = _load(proxy_model_path)
    probe_spec = _load(probe_spec_path)
    if proxy_model.get("policy_hash") != policy.policy_hash.to_json() or probe_spec.get("policy_hash") != policy.policy_hash.to_json():
        raise ValueError("policy binding mismatch")
    if proxy_model.get("status") != "DIAGNOSTIC_ONLY" or probe_spec.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("AC truth runner requires diagnostic-only inputs")
    if probe_spec.get("ac_truth_read") is not False or probe_spec.get("evaluation_results_read") is not False or probe_spec.get("mitigation_results_read") is not False:
        raise ValueError("probe specification feedback flags are not closed")
    declared_probes = [probe for probe in probe_spec.get("probes", []) if isinstance(probe, Mapping)]
    probes = [
        probe for probe in declared_probes
        if str(probe.get("status", "")).startswith("TARGET_REACHED")
        or str(probe.get("status", "")) == "TARGET_EXTREMUM_FROZEN_WITHIN_DOMAIN"
    ]
    if not probes:
        raise ValueError("no fixed directed probes are available for AC truth")
    tasks = [(case_path, policy_payload, proxy_model, probe) for probe in probes]
    if workers <= 1:
        rows = [_solve_one(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_solve_one, tasks))
    result: dict[str, Any] = {
        "serialization_id": "ieee141_m1_v2_directed_probe_ac_truth.v1",
        "status": "DIAGNOSTIC_ONLY",
        "policy_hash": policy.policy_hash.to_json(),
        "policy_version": policy.version.to_json(),
        "network_hash": policy.network_hash.to_json(),
        "proxy_model_hash": proxy_model.get("result_hash"),
        "probe_spec_hash": probe_spec.get("result_hash"),
        "declared_probe_count": len(declared_probes),
        "ac_run_probe_count": len(rows),
        "unreachable_target_ids": [str(probe.get("target_id")) for probe in declared_probes if str(probe.get("target_id")) not in {str(row.get("target_id")) for row in rows}],
        "rows": rows,
        "screen_semantics": "TWO_ENDED_MVA_MAX_WITH_VOLTAGE_BOUNDS",
        "fresh_ac_replay_required": True,
        "ac_input_serialization": "network_hash+q_mode+bus_aligned_PQ.v1",
        "evaluation_results_read": False,
        "mitigation_results_read": False,
        "rate_policy_mutated": False,
        "probe_spec_mutated": False,
        "allocation_resolved": False,
        "calibration_read": False,
        "diagnostic_only": True,
        "primary_evidence_eligible": False,
        "t3_eligible": False,
        "t4_eligible": False,
        "t5_eligible": False,
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join([
        "# M1 V2 fixed-probe AC truth diagnostic",
        "",
        "- Status: `DIAGNOSTIC_ONLY`",
        f"- RATE policy hash: `{result['policy_hash']}`",
        f"- Probe specification hash: `{result['probe_spec_hash']}`",
        f"- Declared target probes: `{result['declared_probe_count']}`; AC-run fixed probes: `{result['ac_run_probe_count']}`",
        "- AC screen: two-ended MVA maximum plus voltage lower/upper bounds.",
        "- Target selection and RATE are immutable; no AC result is fed back into them.",
        "- This artifact does not authorize calibration, T3, T4, T5, or evaluation.",
    ]) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--proxy-model", type=Path, required=True)
    parser.add_argument("--probe-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(canonical_dumps(run_directed_probe_ac_truth(case_path=args.case, policy_path=args.policy, proxy_model_path=args.proxy_model, probe_spec_path=args.probe_spec, output_path=args.output, workers=args.workers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_directed_probe_ac_truth"]
