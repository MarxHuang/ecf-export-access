"""Run the frozen simultaneous multi-reporter IEEE-141 experiment family."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from r4r.metrics import normalized_jain, raw_jain
from r4r.network_parser import parse_matpower_case
from r4r.pair_metrics import compute_coalition_access_transfer_metrics
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.types import Identifier, IdentifierVector
from tools.build_ieee141_rate_v4_execution_input import rate_v4_anchor_execution, rate_v4_q95_specification
from tools.ieee141_rate_v4_design_ac_runner import _run_ac_pair, _selection_for_profile
from tools.run_ieee141_m1_state_dependent_method_execution import _solve_method
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import (
    METHODS, MODES, MODE_CONFIG, _mitigated_margin,
)
from tools.run_ieee141_m1_v2_directed_probe_ac_truth import _screen


SERIALIZATION_ID = "v12_3_simultaneous_multi_reporter_experiment.v1"
SIDES = ("reference", "reported")
_WORKER: dict[str, Any] = {}


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


def _self_hash(value: Mapping[str, Any], *, field: str, label: str) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    if not isinstance(declared, str) or canonical_hash(body) != declared:
        raise ValueError(f"{label} {field} is invalid")
    return declared


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _jain_json(result: Any) -> dict[str, Any]:
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    return {
        "value": result.value.value if result.value is not None else None,
        "status": status,
        "participant_count": result.participant_count,
    }


def _init_worker(
    case_path: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER.clear()
    _WORKER.update({
        "parsed": parse_matpower_case(Path(case_path)),
        "policy": dict(policy),
        "source": dict(source),
        "margins": dict(margins),
        "rates": [float(item["rate_a_mva"]) for item in policy["branch_ratings"]],
        "q95_specification": rate_v4_q95_specification(),
        "anchor_execution": rate_v4_anchor_execution(),
    })


def _transform_request(scenario: Mapping[str, Any], *, side: str, mode: str) -> list[float]:
    values = [float(value) for value in scenario[f"{side}_request_mw"]]
    if side == "reported" and mode in {"CF", "C_005", "C_010"}:
        score = float(MODE_CONFIG[mode]["strategic_score"])
        participant_ids = list(scenario["participant_ids"])
        for coalition_id in scenario["coalition_ids"]:
            values[participant_ids.index(coalition_id)] *= score
    return values


def _screen_summary(screen: Mapping[str, Any], actual: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    policy = _WORKER["policy"]
    rates = _WORKER["rates"]
    tolerance = float(_WORKER["source"]["physical_limits"]["branch_tolerance_mva"])
    branch = [float(value) for value in screen.get("branch_mva_max", [])]
    signed = [max(value - rate, 0.0) for value, rate in zip(branch, rates)]
    loading_ratio = [value / rate for value, rate in zip(branch, rates)]
    critical_index = max(range(len(branch)), key=lambda index: loading_ratio[index]) if branch else None
    voltage = [float(value) for value in actual["voltage_pu"]]
    return {
        "joint_pass": bool(screen.get("joint_pass")),
        "branch_pass": bool(screen.get("branch_pass")),
        "voltage_pass": bool(screen.get("voltage_pass")),
        "max_mva_excess": max(signed, default=0.0),
        "total_mva_excess": math.fsum(signed),
        "violating_branch_count": sum(value > tolerance for value in signed),
        "maximum_branch_mva": max(branch, default=0.0),
        "maximum_loading_ratio": max(loading_ratio, default=0.0),
        "critical_branch_index": critical_index,
        "critical_branch_id": policy["branch_ratings"][critical_index]["branch_id"] if critical_index is not None else None,
        "critical_branch_loading_mva": branch[critical_index] if critical_index is not None else None,
        "critical_branch_rate_mva": rates[critical_index] if critical_index is not None else None,
        "minimum_voltage_pu": min(voltage, default=None),
        "maximum_voltage_pu": max(voltage, default=None),
        "voltage_lower_excess_pu": float(screen.get("voltage_lower_excess_pu", 0.0)),
        "voltage_upper_excess_pu": float(screen.get("voltage_upper_excess_pu", 0.0)),
    }


def _run_side_with_request(
    scenario: Mapping[str, Any],
    *,
    q_mode: str,
    method_id: str,
    mode: str,
    side: str,
    request: Sequence[float],
    margin_ratio: float,
    fairness_alpha_mw: float | None = None,
) -> dict[str, Any]:
    """Solve and AC-check one side for an explicitly supplied request.

    Keeping the transformed request explicit lets independent benchmark runners
    reuse the frozen solver and AC path without adding their policies to the six
    registered mitigation modes.
    """

    source = _WORKER["source"]
    policy = _WORKER["policy"]
    rates = _WORKER["rates"]
    lower = float(source["physical_limits"]["voltage_lower_limit_pu"])
    upper = float(source["physical_limits"]["voltage_upper_limit_pu"])
    request = [float(value) for value in request]
    base_margin = _WORKER["margins"][f"{q_mode}::{method_id}"]
    margin = _mitigated_margin(
        base_margin, source["q_models"][q_mode], rates, lower, upper,
        float(margin_ratio),
    )
    solve_task: tuple[Any, ...] = (
        method_id, source["q_models"][q_mode], rates, lower, upper, margin,
        scenario["capacity_mw"], request, 3.5, q_mode,
    )
    if fairness_alpha_mw is not None:
        solve_task = (*solve_task, float(fairness_alpha_mw))
    method_execution = _json_safe(_solve_method(solve_task))
    solver_summary = {
        key: method_execution.get(key)
        for key in (
            "status", "solver_id", "solver_status", "solver_success", "failure_reason",
            "objective_value", "minimum_constraint_slack", "rho", "rho_max", "constraint_count",
        )
    }
    record: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "coalition_size": len(scenario["coalition_ids"]),
        "spatial_pattern": scenario["spatial_pattern"],
        "coalition_ids": list(scenario["coalition_ids"]),
        "q_mode": q_mode,
        "method_id": method_id,
        "mode": mode,
        "side": side,
        "raw_request_mw": list(scenario[f"{side}_request_mw"]),
        "transformed_request_mw": request,
        "transformed_request_hash": canonical_hash({"participant_ids": scenario["participant_ids"], "values_mw": request}),
        "margin_hash": canonical_hash(margin),
        "method_execution_hash": canonical_hash(method_execution),
        "solver": solver_summary,
        "solver_valid": bool(method_execution.get("solver_success")),
        "ac_numerical_valid": False,
        "physical_screen_pass": False,
        "screen": None,
    }
    if fairness_alpha_mw is not None:
        record["fairness_alpha_mw"] = float(fairness_alpha_mw)
    if not method_execution.get("solver_success"):
        return record
    allocation = [float(value) for value in method_execution["admitted_allocation_mw"]]
    parsed = _WORKER["parsed"]
    selection = _selection_for_profile(parsed, capacity_mw=scenario["capacity_mw"])
    ac = _run_ac_pair(
        parsed=parsed,
        selection=selection,
        allocation_mw=allocation,
        q_mode=q_mode,
        q95_specification=_WORKER["q95_specification"],
        execution=_WORKER["anchor_execution"],
    )
    primary = ac["primary"]
    actual = {
        "branch_p_from_mw": list(primary["branch_p_from_mw"]),
        "branch_q_from_mvar": list(primary["branch_q_from_mvar"]),
        "branch_p_to_mw": list(primary["branch_p_to_mw"]),
        "branch_q_to_mvar": list(primary["branch_q_to_mvar"]),
        "voltage_pu": list(primary["bus_voltage_pu"]),
    }
    screen = _screen(
        actual, rates, lower, upper,
        float(source["physical_limits"]["branch_tolerance_mva"]),
        float(source["physical_limits"]["voltage_tolerance_pu"]),
    )
    numerical = bool(ac["ac_converged_both"] and ac["crosscheck_status"] == "PASS")
    record.update({
        "allocation_mw": allocation,
        "allocation_hash": canonical_hash({"participant_ids": scenario["participant_ids"], "values_mw": allocation}),
        "ac_result_hash": ac["result_hash"],
        "primary_solution_hash": ac["primary_ac_solution_hash"],
        "independent_solution_hash": ac["independent_ac_solution_hash"],
        "crosscheck_status": ac["crosscheck_status"],
        "crosscheck_metrics": ac["crosscheck_metrics"],
        "actual_values_hash": canonical_hash(actual),
        "ac_numerical_valid": numerical,
        "physical_screen_pass": bool(numerical and screen["joint_pass"]),
        "screen": _screen_summary(screen, actual),
    })
    return record


def _run_side(task: tuple[dict[str, Any], str, str, str, str]) -> dict[str, Any]:
    scenario, q_mode, method_id, mode, side = task
    return _run_side_with_request(
        scenario,
        q_mode=q_mode,
        method_id=method_id,
        mode=mode,
        side=side,
        request=_transform_request(scenario, side=side, mode=mode),
        margin_ratio=float(MODE_CONFIG[mode]["margin_ratio"]),
    )


def _pair_record(
    reference: Mapping[str, Any], reported: Mapping[str, Any], scenario: Mapping[str, Any], participant_ids: Sequence[str],
) -> dict[str, Any]:
    base = {
        "pair_id": f"{scenario['scenario_id']}::{reference['q_mode']}::{reference['method_id']}::{reference['mode']}",
        "scenario_id": scenario["scenario_id"],
        "coalition_size": len(scenario["coalition_ids"]),
        "spatial_pattern": scenario["spatial_pattern"],
        "coalition_ids": list(scenario["coalition_ids"]),
        "coalition_bus_ids": list(scenario["coalition_bus_ids"]),
        "mean_pairwise_distance_edges": float(scenario["mean_pairwise_distance_edges"]),
        "q_mode": reference["q_mode"],
        "method_id": reference["method_id"],
        "mode": reference["mode"],
        "reference_solver_valid": bool(reference.get("solver_valid")),
        "reported_solver_valid": bool(reported.get("solver_valid")),
        "reference_ac_numerical_valid": bool(reference.get("ac_numerical_valid")),
        "reported_ac_numerical_valid": bool(reported.get("ac_numerical_valid")),
        "reference_physical_pass": bool(reference.get("physical_screen_pass")),
        "reported_physical_pass": bool(reported.get("physical_screen_pass")),
        "matched_q_pair_pass": False,
        "metrics_valid": False,
    }
    ref = [float(value) for value in reference.get("allocation_mw", [])]
    rep = [float(value) for value in reported.get("allocation_mw", [])]
    if len(ref) != len(participant_ids) or len(rep) != len(participant_ids):
        return base
    coalition_ids = IdentifierVector(Identifier(item) for item in scenario["coalition_ids"])
    coalition_indices = [list(participant_ids).index(item) for item in scenario["coalition_ids"]]
    delivery_caps = [float(scenario["reference_request_mw"][index]) for index in coalition_indices]
    metrics = compute_coalition_access_transfer_metrics(
        IdentifierVector(Identifier(item) for item in participant_ids), ref, rep, coalition_ids, delivery_caps,
    )
    metric_json = metrics.to_json()
    ref_screen = reference["screen"]
    rep_screen = reported["screen"]
    matched_pass = bool(
        reference.get("ac_numerical_valid") and reported.get("ac_numerical_valid")
        and reference.get("physical_screen_pass") and reported.get("physical_screen_pass")
        and reference["q_mode"] == reported["q_mode"]
    )
    base.update({
        "matched_q_pair_pass": matched_pass,
        "metrics_valid": True,
        "reference_allocation_hash": reference["allocation_hash"],
        "reported_allocation_hash": reported["allocation_hash"],
        "reference_total_export_mw": math.fsum(ref),
        "reported_total_export_mw": math.fsum(rep),
        "total_export_change_mw": math.fsum(rep) - math.fsum(ref),
        "raw_request_inflation_mw": math.fsum(
            float(scenario["reported_request_mw"][index]) - float(scenario["reference_request_mw"][index])
            for index in coalition_indices
        ),
        "request_attenuation_mw": math.fsum(
            float(raw) - float(transformed)
            for raw, transformed in zip(reported["raw_request_mw"], reported["transformed_request_mw"])
        ),
        "coalition_metrics": metric_json,
        "coalition_metric_hash": metrics.metric_hash.to_json(),
        "jain_raw_reference": _jain_json(raw_jain(ref)),
        "jain_raw_reported": _jain_json(raw_jain(rep)),
        "jain_norm_reference": _jain_json(normalized_jain(ref, scenario["capacity_mw"])),
        "jain_norm_reported": _jain_json(normalized_jain(rep, scenario["capacity_mw"])),
        "reference_screen": ref_screen,
        "reported_screen": rep_screen,
        "max_mva_excess_change_mva": float(rep_screen["max_mva_excess"]) - float(ref_screen["max_mva_excess"]),
        "total_mva_excess_change_mva": float(rep_screen["total_mva_excess"]) - float(ref_screen["total_mva_excess"]),
    })
    return base


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return math.fsum(values) / len(values)


def _summarize(group: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dimensions,
        "pair_count": len(group),
        "pair_pass_count": sum(bool(row["matched_q_pair_pass"]) for row in group),
        "pair_pass_rate": sum(bool(row["matched_q_pair_pass"]) for row in group) / len(group),
        "mean_raw_request_inflation_mw": _mean(group, ("raw_request_inflation_mw",)),
        "mean_request_attenuation_mw": _mean(group, ("request_attenuation_mw",)),
        "mean_coalition_gain_mw": _mean(group, ("coalition_metrics", "coalition_gain_mw")),
        "mean_coalition_loss_mw": _mean(group, ("coalition_metrics", "coalition_loss_mw")),
        "mean_coalition_net_change_mw": _mean(group, ("coalition_metrics", "coalition_net_change_mw")),
        "mean_noncoalition_loss_mw": _mean(group, ("coalition_metrics", "noncoalition_loss_mw")),
        "mean_noncoalition_gain_mw": _mean(group, ("coalition_metrics", "noncoalition_gain_mw")),
        "mean_coalition_undeliverable_mw": _mean(group, ("coalition_metrics", "coalition_undeliverable_mw")),
        "mean_redistribution_mw": _mean(group, ("coalition_metrics", "redistribution_mw")),
        "mean_reference_total_export_mw": _mean(group, ("reference_total_export_mw",)),
        "mean_reported_total_export_mw": _mean(group, ("reported_total_export_mw",)),
        "mean_total_export_change_mw": _mean(group, ("total_export_change_mw",)),
        "maximum_reported_loading_ratio": max(float(row["reported_screen"]["maximum_loading_ratio"]) for row in group),
        "maximum_reported_mva_excess": max(float(row["reported_screen"]["max_mva_excess"]) for row in group),
        "total_reported_mva_excess": math.fsum(float(row["reported_screen"]["total_mva_excess"]) for row in group),
        "mean_jain_norm_change": _mean(group, ("jain_norm_reported", "value")) - _mean(group, ("jain_norm_reference", "value")),
    }


def _write_pair_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "scenario_id", "coalition_size", "spatial_pattern", "coalition_ids", "coalition_bus_ids",
        "mean_pairwise_distance_edges", "q_mode", "method_id", "mode", "matched_q_pair_pass",
        "raw_request_inflation_mw", "request_attenuation_mw", "reference_total_export_mw",
        "reported_total_export_mw", "total_export_change_mw", "coalition_gain_mw", "coalition_loss_mw",
        "coalition_net_change_mw", "noncoalition_gain_mw", "noncoalition_loss_mw",
        "noncoalition_net_change_mw", "coalition_undeliverable_mw", "redistribution_mw",
        "jain_raw_reference", "jain_raw_reported", "jain_norm_reference", "jain_norm_reported",
        "reference_maximum_loading_ratio", "reported_maximum_loading_ratio", "reference_max_mva_excess",
        "reported_max_mva_excess", "reference_total_mva_excess", "reported_total_mva_excess",
        "reference_minimum_voltage_pu", "reported_minimum_voltage_pu",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            metric = pair["coalition_metrics"]
            writer.writerow({
                "scenario_id": pair["scenario_id"], "coalition_size": pair["coalition_size"],
                "spatial_pattern": pair["spatial_pattern"], "coalition_ids": ";".join(pair["coalition_ids"]),
                "coalition_bus_ids": ";".join(str(item) for item in pair["coalition_bus_ids"]),
                "mean_pairwise_distance_edges": pair["mean_pairwise_distance_edges"], "q_mode": pair["q_mode"],
                "method_id": pair["method_id"], "mode": pair["mode"], "matched_q_pair_pass": int(pair["matched_q_pair_pass"]),
                "raw_request_inflation_mw": pair["raw_request_inflation_mw"], "request_attenuation_mw": pair["request_attenuation_mw"],
                "reference_total_export_mw": pair["reference_total_export_mw"], "reported_total_export_mw": pair["reported_total_export_mw"],
                "total_export_change_mw": pair["total_export_change_mw"],
                "coalition_gain_mw": metric["coalition_gain_mw"], "coalition_loss_mw": metric["coalition_loss_mw"],
                "coalition_net_change_mw": metric["coalition_net_change_mw"], "noncoalition_gain_mw": metric["noncoalition_gain_mw"],
                "noncoalition_loss_mw": metric["noncoalition_loss_mw"], "noncoalition_net_change_mw": metric["noncoalition_net_change_mw"],
                "coalition_undeliverable_mw": metric["coalition_undeliverable_mw"], "redistribution_mw": metric["redistribution_mw"],
                "jain_raw_reference": pair["jain_raw_reference"]["value"], "jain_raw_reported": pair["jain_raw_reported"]["value"],
                "jain_norm_reference": pair["jain_norm_reference"]["value"], "jain_norm_reported": pair["jain_norm_reported"]["value"],
                "reference_maximum_loading_ratio": pair["reference_screen"]["maximum_loading_ratio"],
                "reported_maximum_loading_ratio": pair["reported_screen"]["maximum_loading_ratio"],
                "reference_max_mva_excess": pair["reference_screen"]["max_mva_excess"],
                "reported_max_mva_excess": pair["reported_screen"]["max_mva_excess"],
                "reference_total_mva_excess": pair["reference_screen"]["total_mva_excess"],
                "reported_total_mva_excess": pair["reported_screen"]["total_mva_excess"],
                "reference_minimum_voltage_pu": pair["reference_screen"]["minimum_voltage_pu"],
                "reported_minimum_voltage_pu": pair["reported_screen"]["minimum_voltage_pu"],
            })


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def run_experiment(
    *, case_path: Path, policy_path: Path, source_path: Path, manifest_path: Path,
    execution_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = _load_yaml(policy_path)
    source = _load_json(source_path)
    manifest = _load_json(manifest_path)
    execution = _load_json(execution_path)
    source_hash = _self_hash(source, field="result_hash", label="source envelope")
    manifest_hash = _self_hash(manifest, field="manifest_hash", label="multi-reporter manifest")
    execution_hash = _self_hash(execution, field="result_hash", label="method execution")
    if manifest["source_envelope_hash"] != source_hash or execution["source_envelope_hash"] != source_hash:
        raise ValueError("source binding mismatch")
    if policy.get("policy_hash") != source.get("policy_hash") or policy.get("network_hash") != source.get("network_hash"):
        raise ValueError("RATE policy binding mismatch")
    if tuple(manifest["method_roster"]) != tuple(METHODS) or tuple(manifest["mitigation_mode_roster"]) != tuple(MODES):
        raise ValueError("method or mitigation roster mismatch")
    source_scenario_id = manifest["source_scenario_id"]
    margin_rows = [
        row for row in execution["rows"]
        if row["scenario_id"] == source_scenario_id and row["side"] == "reference"
    ]
    margins: dict[str, Mapping[str, Any]] = {}
    for q_mode in manifest["q_roster"]:
        for method_id in METHODS:
            matches = [row for row in margin_rows if row["q_mode"] == q_mode and row["method_id"] == method_id]
            if len(matches) != 1:
                raise ValueError(f"missing frozen margin for {q_mode}/{method_id}")
            margins[f"{q_mode}::{method_id}"] = matches[0]["margin_q"]
    tasks = [
        (scenario, q_mode, method_id, mode, side)
        for scenario in manifest["scenarios"]
        for q_mode in manifest["q_roster"]
        for method_id in METHODS
        for mode in MODES
        for side in SIDES
    ]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(case_path), policy, source, margins),
        ) as executor:
            side_records = list(executor.map(_run_side, tasks, chunksize=max(1, len(tasks) // (workers * 8))))
    else:
        _init_worker(str(case_path), policy, source, margins)
        side_records = [_run_side(task) for task in tasks]
    side_by_key = {
        (row["scenario_id"], row["q_mode"], row["method_id"], row["mode"], row["side"]): row
        for row in side_records
    }
    pairs = []
    for scenario in manifest["scenarios"]:
        for q_mode in manifest["q_roster"]:
            for method_id in METHODS:
                for mode in MODES:
                    pairs.append(_pair_record(
                        side_by_key[(scenario["scenario_id"], q_mode, method_id, mode, "reference")],
                        side_by_key[(scenario["scenario_id"], q_mode, method_id, mode, "reported")],
                        scenario,
                        source["participant_ids"],
                    ))
    numerical_ac_complete = all(
        row.get("ac_numerical_valid") for row in side_records if row.get("solver_valid")
    ) and all(row.get("solver_valid") for row in side_records)
    chain_integrity = True
    for row in side_records:
        expected = _transform_request(
            next(item for item in manifest["scenarios"] if item["scenario_id"] == row["scenario_id"]),
            side=row["side"], mode=row["mode"],
        )
        if row["transformed_request_mw"] != expected:
            chain_integrity = False
    gates = {
        "DESIGN_BINDING": bool(
            manifest["source_envelope_hash"] == source_hash
            and manifest["policy_hash"] == policy["policy_hash"]
            and manifest["network_hash"] == policy["network_hash"]
        ),
        "SOLVER_COMPLETENESS": all(row.get("solver_valid") for row in side_records),
        "AC_NUMERICAL_COMPLETENESS": numerical_ac_complete,
        "METRIC_COMPLETENESS": all(row.get("metrics_valid") for row in pairs),
        "MITIGATION_CHAIN_INTEGRITY": chain_integrity,
        "RESULT_FEEDBACK_EXCLUDED": all(value is False for value in manifest["input_read_assertions"].values()),
    }
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(int(pair["coalition_size"]), pair["spatial_pattern"], pair["mode"])].append(pair)
        by_mode[pair["mode"]].append(pair)
    summaries = [
        _summarize(grouped[(size, pattern, mode)], {"coalition_size": size, "spatial_pattern": pattern, "mode": mode})
        for size in manifest["coalition_sizes"]
        for pattern in manifest["spatial_patterns"]
        for mode in MODES
    ]
    mode_summaries = [_summarize(by_mode[mode], {"mode": mode}) for mode in MODES]
    result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": "SIMULTANEOUS_MULTI_REPORTER_EXPERIMENT_COMPLETE" if all(gates.values()) else "SIMULTANEOUS_MULTI_REPORTER_EXPERIMENT_BLOCKED",
        "family_scope": manifest["family_scope"],
        "manifest_hash": manifest_hash,
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "q95_specification_hash": canonical_hash(rate_v4_q95_specification()),
        "ac_execution_hash": canonical_hash(rate_v4_anchor_execution()),
        "scenario_count": len(manifest["scenarios"]),
        "side_record_count": len(side_records),
        "pair_record_count": len(pairs),
        "worker_count": workers,
        "method_roster": list(METHODS),
        "q_roster": list(manifest["q_roster"]),
        "mitigation_mode_roster": list(MODES),
        "mode_policy": {
            mode: {
                **MODE_CONFIG[mode],
                "request_scope": "REPORTED_COALITION_ONLY" if mode in {"CF", "C_005", "C_010"} else "IDENTITY",
            }
            for mode in MODES
        },
        "gate_results": gates,
        "physical_outcomes_are_reported_not_required_for_experiment_completion": True,
        "matched_q_pair_pass_count": sum(bool(row["matched_q_pair_pass"]) for row in pairs),
        "summary_by_coalition_pattern_mode": summaries,
        "summary_by_mode": mode_summaries,
        "pair_records": pairs,
        "side_records": side_records,
        "input_read_assertions": {
            "historical_outputs_read": False,
            "rate_policy_refit": False,
            "calibration_refit": False,
            "coalition_selected_from_results": False,
            "evaluation_results_written_back_to_manifest": False,
        },
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_pair_csv(output_path.with_name("MULTI_REPORTER_PAIR_DATA.csv"), pairs)
    _write_summary_csv(output_path.with_name("MULTI_REPORTER_SUMMARY.csv"), summaries)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Simultaneous multi-reporter experiment", "",
            f"- Status: `{result['status']}`",
            f"- Scenarios: `{result['scenario_count']}`; paired rows: `{result['pair_record_count']}`.",
            f"- Matched-Q physical pair passes: `{result['matched_q_pair_pass_count']}/{result['pair_record_count']}`.",
            f"- Gates: `{gates}`",
            f"- Result hash: `{result['result_hash']}`",
            "- Physical pass/fail is an experimental outcome, not a prerequisite for numerical completeness.",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    result = run_experiment(
        case_path=args.case, policy_path=args.policy, source_path=args.source,
        manifest_path=args.manifest, execution_path=args.execution,
        output_path=args.output, workers=max(1, args.workers),
    )
    print(canonical_dumps({
        "status": result["status"], "gate_results": result["gate_results"],
        "pair_record_count": result["pair_record_count"],
        "matched_q_pair_pass_count": result["matched_q_pair_pass_count"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_experiment", "main", "_run_side_with_request"]
