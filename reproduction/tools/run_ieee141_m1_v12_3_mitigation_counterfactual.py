"""Run the fresh V12.3 six-mode matched-Q mitigation counterfactual.

The evaluation-only branch is entered only after the fresh T4 attestation.
RATE, network, Q, reporter/scenario, allocation method, and calibration
envelope are hash-bound and unchanged.  Only the registered mitigation mode
transforms the unilateral reported request and/or adds an inward proxy-domain
reserve.  Every mode is re-solved and re-screened on both matched-Q sides.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from r4r.metrics import normalized_jain, raw_jain
from r4r.serialization import canonical_dumps, canonical_hash
from tools.ieee141_rate_v4_design_ac_runner import _run_ac_pair, _selection_for_profile
from tools.run_ieee141_m1_state_dependent_method_execution import (
    BRANCH_BUFFER_MVA,
    VOLTAGE_BUFFER_PU,
    _solve_method,
)
from tools.run_ieee141_m1_v2_directed_probe_ac_truth import _screen
from r4r.network_parser import parse_matpower_case


MODES = ("None", "CF", "MR_005", "MR_010", "C_005", "C_010")
METHODS = ("weighted_proportional", "equal_kw_reduction", "flat_level", "max_export_lp", "fairness_qp")
SIDES = ("reference", "reported")
STRATEGIC_SCORE = 0.70
MODE_CONFIG = {
    "None": {"strategic_score": 1.0, "margin_ratio": 0.0, "request_scope": "IDENTITY"},
    "CF": {"strategic_score": STRATEGIC_SCORE, "margin_ratio": 0.0, "request_scope": "REPORTED_REPORTER_ONLY"},
    "MR_005": {"strategic_score": 1.0, "margin_ratio": 0.05, "request_scope": "IDENTITY"},
    "MR_010": {"strategic_score": 1.0, "margin_ratio": 0.10, "request_scope": "IDENTITY"},
    "C_005": {"strategic_score": STRATEGIC_SCORE, "margin_ratio": 0.05, "request_scope": "REPORTED_REPORTER_ONLY"},
    "C_010": {"strategic_score": STRATEGIC_SCORE, "margin_ratio": 0.10, "request_scope": "REPORTED_REPORTER_ONLY"},
}
DELIVERY_CAP_POLICY = "REFERENCE_REQUEST_AT_REPORTER"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return dict(value)


def _self_hash(value: Mapping[str, Any], *, label: str) -> str:
    body = dict(value)
    declared = body.get("result_hash")
    body["result_hash"] = None
    if not isinstance(declared, str) or canonical_hash(body) != declared:
        raise ValueError(f"{label} result hash is invalid")
    return declared


def _json_safe(value: Any) -> Any:
    """Encode non-finite diagnostic scalars as explicit nulls."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _margin_components(margin: Mapping[str, Any], key: str, count: int) -> tuple[list[float | None], list[float]]:
    intercept = margin.get(f"{key}_effective_intercept")
    slope = margin.get(f"{key}_effective_slope")
    if not isinstance(intercept, list) or not isinstance(slope, list) or len(intercept) != count or len(slope) != count:
        raise ValueError(f"margin component {key} shape is invalid")
    return list(intercept), list(slope)


def _mitigated_margin(
    base: Mapping[str, Any],
    model_q: Mapping[str, Any],
    rates: Sequence[float],
    lower: float,
    upper: float,
    ratio: float,
) -> dict[str, Any]:
    """Apply the registered MR/C inward reserve to the proxy domain.

    The reserve is a fraction of *remaining calibrated headroom at the
    zero-allocation origin*, rather than a fraction of RATE itself.  This is
    important for a synthetic RATE policy: adding ``kappa * RATE`` can make
    the origin infeasible and would turn a mitigation counterfactual into an
    empty-domain test.  The transform is computed only from the frozen proxy
    baseline, frozen RATE, and registered safety buffers; it never reads an
    evaluation allocation or an AC result.
    """
    result = copy.deepcopy(dict(base))
    if ratio <= 0.0:
        result["mitigation_margin_ratio"] = 0.0
        result["mitigation_margin_formula"] = "identity"
        return result
    branch_from_a, branch_from_l = _margin_components(base, "branch_from_mva", len(rates))
    branch_to_a, branch_to_l = _margin_components(base, "branch_to_mva", len(rates))
    lower_a, lower_l = _margin_components(base, "voltage_lower_pu", 141)
    upper_a, upper_l = _margin_components(base, "voltage_upper_pu", 141)
    proxy_base = model_q["proxy_base"]
    branch_from_headroom = [
        max(
            float(rate)
            - BRANCH_BUFFER_MVA
            - math.hypot(float(proxy_base["branch_p_from_mw"][i]), float(proxy_base["branch_q_from_mvar"][i]))
            - float(a),
            0.0,
        )
        for i, (a, rate) in enumerate(zip(branch_from_a, rates))
    ]
    branch_to_headroom = [
        max(
            float(rate)
            - BRANCH_BUFFER_MVA
            - math.hypot(float(proxy_base["branch_p_to_mw"][i]), float(proxy_base["branch_q_to_mvar"][i]))
            - float(a),
            0.0,
        )
        for i, (a, rate) in enumerate(zip(branch_to_a, rates))
    ]
    result["branch_from_mva_effective_intercept"] = [float(a) + ratio * branch_from_headroom[i] for i, a in enumerate(branch_from_a)]
    result["branch_from_mva_effective_slope"] = [float(value) for value in branch_from_l]
    result["branch_to_mva_effective_intercept"] = [float(a) + ratio * branch_to_headroom[i] for i, a in enumerate(branch_to_a)]
    result["branch_to_mva_effective_slope"] = [float(value) for value in branch_to_l]
    lower_headroom = [
        max(
            float(proxy_base["voltage_pu"][i])
            - float(lower)
            - VOLTAGE_BUFFER_PU
            - float(a),
            0.0,
        )
        for i, a in enumerate(lower_a)
    ]
    result["voltage_lower_pu_effective_intercept"] = [float(a) + ratio * lower_headroom[i] for i, a in enumerate(lower_a)]
    result["voltage_lower_pu_effective_slope"] = [float(value) for value in lower_l]
    # Inactive upper residuals remain inactive: there is no calibrated upper
    # family to tighten, and activating one here would silently change the
    # applicability domain rather than apply the registered MR transform.
    upper_headroom = [
        None
        if a is None
        else max(
            float(upper)
            - VOLTAGE_BUFFER_PU
            - float(proxy_base["voltage_pu"][i])
            - float(a),
            0.0,
        )
        for i, a in enumerate(upper_a)
    ]
    result["voltage_upper_pu_effective_intercept"] = [
        None if a is None else float(a) + ratio * upper_headroom[i]
        for i, a in enumerate(upper_a)
    ]
    result["voltage_upper_pu_effective_slope"] = [None if l is None else float(l) for l in upper_l]
    result["mitigation_margin_ratio"] = float(ratio)
    result["mitigation_margin_formula"] = (
        "branch reserve += kappa*max(RATE_A-BRANCH_BUFFER_MVA-"
        "||S_proxy,0||-a_base,0); lower/active-upper voltage reserve += "
        "kappa*max(available_proxy_headroom,0); inactive upper residuals remain null"
    )
    result["mitigation_margin_headroom_reference"] = "FROZEN_PROXY_BASE_AT_ZERO_ALLOCATION"
    return result


def _transformed_request(request: Sequence[float], reporter_index: int, *, side: str, mode: str) -> list[float]:
    values = [float(value) for value in request]
    config = MODE_CONFIG[mode]
    if side == "reported" and config["request_scope"] == "REPORTED_REPORTER_ONLY":
        values[reporter_index] *= float(config["strategic_score"])
    return values


def _actual_values(primary: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        "branch_p_from_mw": list(primary["branch_p_from_mw"]),
        "branch_q_from_mvar": list(primary["branch_q_from_mvar"]),
        "branch_p_to_mw": list(primary["branch_p_to_mw"]),
        "branch_q_to_mvar": list(primary["branch_q_to_mvar"]),
        "voltage_pu": list(primary["bus_voltage_pu"]),
    }


def _run_side(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        case_path, policy, source, execution_input, scenario, row, mode, side,
    ) = task
    q_mode = row["q_mode"]
    method_id = row["method_id"]
    config = MODE_CONFIG[mode]
    reporter_index = int(scenario["reporter_index"])
    request = _transformed_request(scenario[f"{side}_request_mw"], reporter_index, side=side, mode=mode)
    rates = [float(item["rate_a_mva"]) for item in policy["branch_ratings"]]
    lower = float(source["physical_limits"]["voltage_lower_limit_pu"])
    upper = float(source["physical_limits"]["voltage_upper_limit_pu"])
    margin = _mitigated_margin(row["margin_q"], source["q_models"][q_mode], rates, lower, upper, float(config["margin_ratio"]))
    method_execution = _json_safe(_solve_method((method_id, source["q_models"][q_mode], rates, lower, upper, margin, scenario["capacity_mw"], request, 3.5, q_mode)))
    record: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"], "source_role": scenario["source_role"], "source_root_id": scenario["source_root_id"],
        "q_mode": q_mode, "method_id": method_id, "mode": mode, "side": side, "reporter_id": scenario["reporter_id"], "reporter_index": reporter_index,
        "strategic_score": float(config["strategic_score"]), "margin_ratio": float(config["margin_ratio"]),
        "raw_request_mw": list(scenario[f"{side}_request_mw"]), "transformed_request_mw": request,
        "transformed_request_hash": canonical_hash({"participant_ids": source["participant_ids"], "values_mw": request}),
        "margin_hash": canonical_hash(margin), "margin_transformation": margin.get("mitigation_margin_formula", "identity"),
        "method_execution": method_execution,
        "solver_valid": bool(method_execution.get("solver_success")),
        "ac_valid": False,
        "actual_screen": None,
    }
    if not method_execution.get("solver_success"):
        record["failure_reason"] = method_execution.get("failure_reason") or "MITIGATION_SOLVER_FAIL"
        return record
    parsed = parse_matpower_case(Path(case_path))
    selection = _selection_for_profile(parsed, capacity_mw=scenario["capacity_mw"])
    allocation = method_execution["admitted_allocation_mw"]
    ac = _run_ac_pair(
        parsed=parsed, selection=selection, allocation_mw=allocation, q_mode=q_mode,
        q95_specification=execution_input["q95_specification"], execution=execution_input["anchor_execution"],
    )
    actual = _actual_values(ac["primary"])
    screen = _screen(actual, rates, lower, upper, float(source["physical_limits"]["branch_tolerance_mva"]), float(source["physical_limits"]["voltage_tolerance_pu"]))
    record.update({
        "allocation_mw": list(allocation), "allocation_hash": canonical_hash({"participant_ids": source["participant_ids"], "values_mw": list(allocation)}),
        "ac_result_hash": ac["result_hash"], "primary_solution_hash": ac["primary_ac_solution_hash"], "independent_solution_hash": ac["independent_ac_solution_hash"],
        "ac_crosscheck_status": ac["crosscheck_status"], "ac_crosscheck_metrics": ac["crosscheck_metrics"], "actual_values": actual,
        "actual_values_hash": canonical_hash(actual), "actual_screen": screen,
        "ac_valid": bool(ac["ac_converged_both"] and ac["crosscheck_status"] == "PASS" and screen["joint_pass"]),
        "failure_reason": None if (ac["ac_converged_both"] and ac["crosscheck_status"] == "PASS" and screen["joint_pass"]) else "MITIGATION_AC_FAIL",
    })
    return record


def _jain_json(result: Any) -> dict[str, Any]:
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    return {"value": result.value.value if result.value is not None else None, "status": status, "participant_count": result.participant_count}


def _pair_record(reference: Mapping[str, Any], reported: Mapping[str, Any], scenario: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    ref = [float(value) for value in reference.get("allocation_mw", [])]
    rep = [float(value) for value in reported.get("allocation_mw", [])]
    reporter = int(scenario["reporter_index"])
    capacity = scenario["capacity_mw"]
    if len(ref) != len(capacity) or len(rep) != len(capacity):
        return {
            "pair_id": f"{scenario['scenario_id']}::{reference.get('q_mode')}::{reference.get('method_id')}::{reference.get('mode')}",
            "scenario_id": scenario["scenario_id"], "q_mode": reference.get("q_mode"), "method_id": reference.get("method_id"), "mode": reference.get("mode"), "reporter_id": scenario["reporter_id"],
            "reference_solve_attempted": True, "reported_solve_attempted": True,
            "reference_solver_valid": bool(reference.get("solver_valid")), "reported_solver_valid": bool(reported.get("solver_valid")),
            "reference_ac_valid": False, "reported_ac_valid": False, "matched_q_pair_pass": False,
            "reference_allocation_hash": reference.get("allocation_hash"), "reported_allocation_hash": reported.get("allocation_hash"),
            "reference_total_export_mw": 0.0, "reported_total_export_mw": 0.0, "request_attenuation_mw": 0.0, "allocation_reduction_mw": 0.0,
            "sg_mw": 0.0, "ol_minus_reporter_mw": 0.0, "ud_mw": 0.0, "redistribution_mw": 0.0,
            "reference_screen_metrics": {"max_mva_excess": 0.0, "total_mva_excess": 0.0, "voltage_lower_excess_pu": 0.0, "voltage_upper_excess_pu": 0.0, "joint_pass": False},
            "reported_screen_metrics": {"max_mva_excess": 0.0, "total_mva_excess": 0.0, "voltage_lower_excess_pu": 0.0, "voltage_upper_excess_pu": 0.0, "joint_pass": False},
            "failure_reason": "MITIGATION_SOLVER_FAIL", "metrics_valid": False,
        }
    delta = [after - before for before, after in zip(ref, rep)]
    delivery_cap = float(scenario["reference_request_mw"][reporter])
    sg = max(delta[reporter], 0.0)
    ol = sum(max(-value, 0.0) for index, value in enumerate(delta) if index != reporter)
    ud = max(rep[reporter] - delivery_cap, 0.0)
    redistribution = 0.5 * sum(abs(value) for value in delta)
    ref_screen = reference.get("actual_screen") or {}
    rep_screen = reported.get("actual_screen") or {}
    def screen_metrics(screen: Mapping[str, Any]) -> dict[str, Any]:
        branch = list(screen.get("branch_mva_max", []))
        excess = [max(float(value), 0.0) for value in screen.get("branch_mva_max", [])]
        rates = [float(item["rate_a_mva"]) for item in _CURRENT_POLICY["branch_ratings"]]
        signed = [max(value - rate, 0.0) for value, rate in zip(branch, rates)]
        return {"max_mva_excess": max(signed, default=0.0), "total_mva_excess": math.fsum(signed), "violating_branch_count": sum(value > float(source["physical_limits"]["branch_tolerance_mva"]) for value in signed), "voltage_lower_excess_pu": float(screen.get("voltage_lower_excess_pu", 0.0)), "voltage_upper_excess_pu": float(screen.get("voltage_upper_excess_pu", 0.0)), "joint_pass": bool(screen.get("joint_pass"))}
    return {
        "pair_id": f"{scenario['scenario_id']}::{reference['q_mode']}::{reference['method_id']}::{reference['mode']}",
        "scenario_id": scenario["scenario_id"], "q_mode": reference["q_mode"], "method_id": reference["method_id"], "mode": reference["mode"], "reporter_id": scenario["reporter_id"],
        "reference_solve_attempted": True, "reported_solve_attempted": True,
        "reference_solver_valid": bool(reference.get("solver_valid")), "reported_solver_valid": bool(reported.get("solver_valid")),
        "reference_ac_valid": bool(reference.get("ac_valid")), "reported_ac_valid": bool(reported.get("ac_valid")),
        "matched_q_pair_pass": bool(reference.get("solver_valid") and reported.get("solver_valid") and reference.get("ac_valid") and reported.get("ac_valid") and reference["q_mode"] == reported["q_mode"]),
        "reference_allocation_hash": reference.get("allocation_hash"), "reported_allocation_hash": reported.get("allocation_hash"),
        "reference_total_export_mw": math.fsum(ref), "reported_total_export_mw": math.fsum(rep),
        # Attenuation is defined on the strategic reported side itself.  It is
        # not the (different) gap between an honest reference request and a
        # still-strategic request after filtering; the latter can legitimately
        # remain above the reference when the registered score is 0.70.
        "request_attenuation_mw": math.fsum(float(a) - float(b) for a, b in zip(reported["raw_request_mw"], reported["transformed_request_mw"])),
        "allocation_reduction_mw": math.fsum(ref) - math.fsum(rep),
        "sg_mw": sg, "ol_minus_reporter_mw": ol, "ud_mw": ud, "redistribution_mw": redistribution,
        "delivery_cap_mw": delivery_cap,
        "jain_raw_reference": _jain_json(raw_jain(ref)), "jain_raw_reported": _jain_json(raw_jain(rep)),
        "jain_norm_reference": _jain_json(normalized_jain(ref, capacity)), "jain_norm_reported": _jain_json(normalized_jain(rep, capacity)),
        "reference_screen_metrics": screen_metrics(ref_screen), "reported_screen_metrics": screen_metrics(rep_screen),
        "physical_excess_reduction_mva": screen_metrics(ref_screen)["max_mva_excess"] - screen_metrics(rep_screen)["max_mva_excess"],
        "total_physical_excess_reduction_mva": screen_metrics(ref_screen)["total_mva_excess"] - screen_metrics(rep_screen)["total_mva_excess"],
    }


_CURRENT_POLICY: dict[str, Any] = {}


def run_mitigation(*, case_path: Path, policy_path: Path, source_path: Path, execution_input_path: Path, execution_path: Path, t4_path: Path, output_path: Path, workers: int = 8) -> dict[str, Any]:
    global _CURRENT_POLICY
    policy = _load_yaml(policy_path)
    _CURRENT_POLICY = policy
    source = _load_json(source_path)
    execution_input = _load_json(execution_input_path)
    execution = _load_json(execution_path)
    t4 = _load_json(t4_path)
    source_hash = _self_hash(source, label="source envelope")
    execution_hash = _self_hash(execution, label="method execution")
    t4_hash = _self_hash(t4, label="T4 gate")
    if not bool(t4.get("t4_eligible")) or t4.get("status") != "FRESH_V12_3_T4_GATE_PASS" or t4.get("execution_hash") != execution_hash or t4.get("source_envelope_hash") != source_hash:
        raise ValueError("mitigation requires the matching fresh T4 attestation")
    source_by_id = {str(item["scenario_id"]): item for item in source["scenarios"]}
    evaluation_ids = list(source["scenario_split"]["evaluation"])
    if not evaluation_ids:
        raise ValueError("evaluation split is empty")
    base_rows = [row for row in execution["rows"] if row["scenario_id"] in evaluation_ids]
    base_by_key = {(row["scenario_id"], row["q_mode"], row["method_id"], row["side"]): row for row in base_rows}
    tasks: list[tuple[Any, ...]] = []
    for scenario_id in evaluation_ids:
        scenario = source_by_id[scenario_id]
        for q_mode in source["q_roster"]:
            for method_id in METHODS:
                base_ref = base_by_key[(scenario_id, q_mode, method_id, "reference")]
                for mode in MODES:
                    for side in SIDES:
                        row = dict(base_ref)
                        row["q_mode"] = q_mode
                        row["method_id"] = method_id
                        tasks.append((str(case_path), policy, source, execution_input, scenario, row, mode, side))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_init_policy, initargs=(policy,)) as executor:
            side_records = list(executor.map(_run_side, tasks, chunksize=max(1, len(tasks) // max(1, workers * 4))))
    else:
        side_records = [_run_side(task) for task in tasks]
    by_pair: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in side_records:
        key = (record["scenario_id"], record["q_mode"], record["method_id"], record["mode"], record["side"])
        by_pair[key] = record
    pair_records: list[dict[str, Any]] = []
    for scenario_id in evaluation_ids:
        scenario = source_by_id[scenario_id]
        for q_mode in source["q_roster"]:
            for method_id in METHODS:
                for mode in MODES:
                    pair_records.append(_pair_record(by_pair[(scenario_id, q_mode, method_id, mode, "reference")], by_pair[(scenario_id, q_mode, method_id, mode, "reported")], scenario, source))
    transformation_pass = all(record.get("transformed_request_hash") and record.get("margin_hash") for record in side_records)
    solver_pass = all(record.get("solver_valid") for record in side_records)
    metrics_pass = all(record.get("jain_raw_reference", {}).get("status") in {"DEFINED", "UNDEFINED_ALL_ZERO", "UNDEFINED_EMPTY"} for record in pair_records) and all(record.get("redistribution_mw") is not None for record in pair_records)
    ac_pass = all(record.get("matched_q_pair_pass") for record in pair_records)
    none_pairs = {(row["scenario_id"], row["q_mode"], row["method_id"]): row for row in pair_records if row["mode"] == "None"}
    chain_pass = True
    for pair in pair_records:
        baseline = none_pairs[(pair["scenario_id"], pair["q_mode"], pair["method_id"])]
        if pair["mode"] == "None" and pair["reference_allocation_hash"] != baseline["reference_allocation_hash"]:
            chain_pass = False
        if pair["mode"] != "None" and pair["mode"] in {"CF", "C_005", "C_010"} and pair["request_attenuation_mw"] < -1.0e-10:
            chain_pass = False
    gates = {"G014": transformation_pass, "G015": solver_pass, "G016": metrics_pass, "G017": ac_pass, "G018": chain_pass}
    summary_by_mode: list[dict[str, Any]] = []
    for mode in MODES:
        group = [pair for pair in pair_records if pair["mode"] == mode]
        summary_by_mode.append({
            "mode": mode, "pair_count": len(group), "pair_pass_count": sum(bool(item["matched_q_pair_pass"]) for item in group),
            "mean_reference_total_export_mw": math.fsum(item["reference_total_export_mw"] for item in group) / len(group),
            "mean_reported_total_export_mw": math.fsum(item["reported_total_export_mw"] for item in group) / len(group),
            "mean_allocation_reduction_mw": math.fsum(item["allocation_reduction_mw"] for item in group) / len(group),
            "mean_request_attenuation_mw": math.fsum(item["request_attenuation_mw"] for item in group) / len(group),
            "mean_sg_mw": math.fsum(item["sg_mw"] for item in group) / len(group), "mean_ol_minus_reporter_mw": math.fsum(item["ol_minus_reporter_mw"] for item in group) / len(group),
            "mean_ud_mw": math.fsum(item["ud_mw"] for item in group) / len(group), "mean_redistribution_mw": math.fsum(item["redistribution_mw"] for item in group) / len(group),
            "max_reference_mva_excess": max(item["reference_screen_metrics"]["max_mva_excess"] for item in group), "max_reported_mva_excess": max(item["reported_screen_metrics"]["max_mva_excess"] for item in group),
            "total_reference_mva_excess": math.fsum(item["reference_screen_metrics"]["total_mva_excess"] for item in group), "total_reported_mva_excess": math.fsum(item["reported_screen_metrics"]["total_mva_excess"] for item in group),
            "max_voltage_lower_excess_pu": max(max(item["reference_screen_metrics"]["voltage_lower_excess_pu"], item["reported_screen_metrics"]["voltage_lower_excess_pu"]) for item in group),
            "max_voltage_upper_excess_pu": max(max(item["reference_screen_metrics"]["voltage_upper_excess_pu"], item["reported_screen_metrics"]["voltage_upper_excess_pu"]) for item in group),
        })
    result: dict[str, Any] = {
        "serialization_id": "ieee141_m1_v12_3_mitigation_counterfactual.v1",
        "status": "FRESH_V12_3_T5_COUNTERFACTUAL_COMPLETE" if all(gates.values()) else "FRESH_V12_3_T5_COUNTERFACTUAL_BLOCKED",
        "policy_hash": source["policy_hash"], "source_envelope_hash": source_hash, "method_execution_hash": execution_hash, "t4_gate_hash": t4_hash,
        "scope": "EVALUATION_ONLY_AFTER_RATE_FREEZE", "scenario_ids": evaluation_ids, "mode_roster": list(MODES), "method_roster": list(METHODS), "q_roster": list(source["q_roster"]),
        "invariants": {"rate_policy_unchanged": True, "network_unchanged": True, "q_unchanged": True, "reporter_mapping_source_bound": True, "allocation_method_unchanged": True, "calibration_margin_base_unchanged": True, "ac_implementation_unchanged": True},
        "mode_policy": MODE_CONFIG,
        "gate_results": gates,
        "summary_by_mode": summary_by_mode,
        "pair_records": pair_records,
        "side_records": side_records,
        "input_read_assertions": {"historical_outputs_read": False, "old_mitigation_results_read": False, "calibration_refit": False, "holdout_feedback_used": False, "rate_policy_mutated": False},
        "diagnostic_only": True, "primary_evidence_eligible": False, "t5_eligible": bool(all(gates.values())), "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join([
        "# IEEE-141 V12.3 T5 mitigation counterfactual", "", f"- Status: `{result['status']}`", f"- Scope: `{result['scope']}`", f"- Result hash: `{result['result_hash']}`",
        f"- Gates: `{gates}`", "- T5 is a counterfactual diagnostic branch; manuscript eligibility remains an independent claim/run gate.",
    ]) + "\n", encoding="utf-8")
    return result


def _init_policy(policy: Mapping[str, Any]) -> None:
    global _CURRENT_POLICY
    _CURRENT_POLICY = dict(policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execution-input", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run_mitigation(case_path=args.case, policy_path=args.policy, source_path=args.source, execution_input_path=args.execution_input, execution_path=args.execution, t4_path=args.t4, output_path=args.output, workers=max(1, args.workers))
    print(canonical_dumps({"status": result["status"], "gate_results": result["gate_results"], "t5_eligible": result["t5_eligible"], "result_hash": result["result_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_mitigation"]
