"""Run a post-freeze request-multiplier sensitivity on the V12.3 case study.

The sweep holds the feeder, participant placement, capacities, calibrated
network margins, branch ratings, reactive-power model, and allocation methods
fixed.  Only one participant's request multiplier changes.  The result is a
diagnostic sensitivity analysis; it is not used to select or refit any physical
limit or allocation parameter.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from r4r.metrics import normalized_jain, raw_jain
from r4r.pair_metrics import PAPER_REDISTRIBUTION_DEFINITION_ID, compute_access_transfer_metrics
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.types import Identifier, IdentifierVector
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import METHODS
from tools.run_v12_3_multi_reporter_experiment import (
    _init_worker,
    _run_side,
)


SERIALIZATION_ID = "v12_3_request_multiplier_sensitivity.v1"
GAMMA_GRID = tuple([round(1.0 + 0.1 * index, 2) for index in range(11)] + [1.74])


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return dict(value)


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    if not isinstance(declared, str) or canonical_hash(body) != declared:
        raise ValueError(f"invalid {label} {field}")
    return declared


def _scenario(
    source_scenario: Mapping[str, Any],
    participant_ids: Sequence[str],
    participant_bus_ids: Sequence[int],
    gamma: float,
) -> dict[str, Any]:
    reporter_index = int(source_scenario["reporter_index"])
    reference = [float(value) for value in source_scenario["reference_request_mw"]]
    reported = list(reference)
    reported[reporter_index] = gamma * reference[reporter_index]
    reporter_id = str(source_scenario["reporter_id"])
    return {
        "scenario_id": f"V12_3_REQUEST_MULTIPLIER_GAMMA_{gamma:.2f}".replace(".", "P"),
        "gamma": gamma,
        "spatial_pattern": "SINGLE_REQUEST_CHANGE",
        "participant_ids": list(participant_ids),
        "coalition_ids": [reporter_id],
        "coalition_bus_ids": [int(participant_bus_ids[reporter_index])],
        "mean_pairwise_distance_edges": 0.0,
        "capacity_mw": [float(value) for value in source_scenario["capacity_mw"]],
        "reference_request_mw": reference,
        "reported_request_mw": reported,
    }


def _write_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "gamma", "q_mode", "method_id", "matched_q_pair_pass",
        "reference_total_export_mw", "reported_total_export_mw",
        "total_export_change_mw", "redistribution_mw",
        "changed_participant_gain_mw", "other_participant_loss_mw",
        "normalized_jain_change", "reported_maximum_loading_ratio",
        "reported_minimum_voltage_pu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            metrics = pair["access_metrics"]
            writer.writerow({
                "gamma": pair["gamma"],
                "q_mode": pair["q_mode"],
                "method_id": pair["method_id"],
                "matched_q_pair_pass": int(bool(pair["matched_q_pair_pass"])),
                "reference_total_export_mw": pair["reference_total_export_mw"],
                "reported_total_export_mw": pair["reported_total_export_mw"],
                "total_export_change_mw": pair["total_export_change_mw"],
                "redistribution_mw": metrics["redistribution_mw"],
                "changed_participant_gain_mw": metrics["sg_mw"],
                "other_participant_loss_mw": metrics["ol_minus_reporter_mw"],
                "normalized_jain_change": (
                    float(pair["jain_norm_reported"]["value"])
                    - float(pair["jain_norm_reference"]["value"])
                ),
                "reported_maximum_loading_ratio": pair["reported_screen"]["maximum_loading_ratio"],
                "reported_minimum_voltage_pu": pair["reported_screen"]["minimum_voltage_pu"],
            })


def _jain_json(result: Any) -> dict[str, Any]:
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    return {
        "value": result.value.value if result.value is not None else None,
        "status": status,
        "participant_count": result.participant_count,
    }


def _single_pair_record(
    reference: Mapping[str, Any],
    reported: Mapping[str, Any],
    scenario: Mapping[str, Any],
    participant_ids: Sequence[str],
) -> dict[str, Any]:
    pair: dict[str, Any] = {
        "pair_id": f"{scenario['scenario_id']}::{reference['q_mode']}::{reference['method_id']}",
        "scenario_id": scenario["scenario_id"],
        "q_mode": reference["q_mode"],
        "method_id": reference["method_id"],
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
        return pair
    reporter_id = str(scenario["coalition_ids"][0])
    reporter_index = list(participant_ids).index(reporter_id)
    metrics = compute_access_transfer_metrics(
        IdentifierVector(Identifier(item) for item in participant_ids),
        ref,
        rep,
        Identifier(reporter_id),
        float(scenario["reference_request_mw"][reporter_index]),
        redistribution_definition_id=Identifier(PAPER_REDISTRIBUTION_DEFINITION_ID),
    )
    matched_pass = bool(
        reference.get("ac_numerical_valid")
        and reported.get("ac_numerical_valid")
        and reference.get("physical_screen_pass")
        and reported.get("physical_screen_pass")
        and reference["q_mode"] == reported["q_mode"]
    )
    pair.update({
        "matched_q_pair_pass": matched_pass,
        "metrics_valid": True,
        "reference_allocation_hash": reference["allocation_hash"],
        "reported_allocation_hash": reported["allocation_hash"],
        "reference_total_export_mw": math.fsum(ref),
        "reported_total_export_mw": math.fsum(rep),
        "total_export_change_mw": math.fsum(rep) - math.fsum(ref),
        "access_metrics": metrics.to_json(),
        "access_metric_hash": metrics.metric_hash.to_json(),
        "jain_raw_reference": _jain_json(raw_jain(ref)),
        "jain_raw_reported": _jain_json(raw_jain(rep)),
        "jain_norm_reference": _jain_json(normalized_jain(ref, scenario["capacity_mw"])),
        "jain_norm_reported": _jain_json(normalized_jain(rep, scenario["capacity_mw"])),
        "reference_screen": reference["screen"],
        "reported_screen": reported["screen"],
    })
    return pair


def run_sensitivity(
    *,
    case_path: Path,
    policy_path: Path,
    source_path: Path,
    execution_path: Path,
    output_path: Path,
    workers: int,
) -> dict[str, Any]:
    policy = _load_yaml(policy_path)
    source = _load_json(source_path)
    execution = _load_json(execution_path)
    source_hash = _self_hash(source, "result_hash", "source envelope")
    execution_hash = _self_hash(execution, "result_hash", "method execution")
    if execution.get("source_envelope_hash") != source_hash:
        raise ValueError("method-execution/source binding mismatch")
    if policy.get("policy_hash") != source.get("policy_hash"):
        raise ValueError("branch-rating/source binding mismatch")

    evaluation_ids = list(source["scenario_split"]["evaluation"])
    if len(evaluation_ids) != 1:
        raise ValueError("expected exactly one frozen evaluation scenario")
    source_scenario = next(
        item for item in source["scenarios"] if item["scenario_id"] == evaluation_ids[0]
    )
    participant_ids = list(source["participant_ids"])
    participant_bus_ids = [int(value) for value in source["participant_bus_ids"]]
    scenarios = [
        _scenario(source_scenario, participant_ids, participant_bus_ids, gamma)
        for gamma in sorted(set(GAMMA_GRID))
    ]

    margin_rows = [
        row for row in execution["rows"]
        if row["scenario_id"] == evaluation_ids[0] and row["side"] == "reference"
    ]
    margins: dict[str, Mapping[str, Any]] = {}
    for q_mode in source["q_roster"]:
        for method_id in METHODS:
            matches = [
                row for row in margin_rows
                if row["q_mode"] == q_mode and row["method_id"] == method_id
            ]
            if len(matches) != 1:
                raise ValueError(f"missing frozen margin for {q_mode}/{method_id}")
            margins[f"{q_mode}::{method_id}"] = matches[0]["margin_q"]

    tasks = [
        (scenario, q_mode, method_id, "None", side)
        for scenario in scenarios
        for q_mode in source["q_roster"]
        for method_id in METHODS
        for side in ("reference", "reported")
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_worker,
        initargs=(str(case_path), policy, source, margins),
    ) as executor:
        side_records = list(
            executor.map(_run_side, tasks, chunksize=max(1, len(tasks) // (max(1, workers) * 8)))
        )

    by_key = {
        (row["scenario_id"], row["q_mode"], row["method_id"], row["side"]): row
        for row in side_records
    }
    pairs: list[dict[str, Any]] = []
    gamma_by_scenario = {scenario["scenario_id"]: scenario["gamma"] for scenario in scenarios}
    for scenario in scenarios:
        for q_mode in source["q_roster"]:
            for method_id in METHODS:
                pair = _single_pair_record(
                    by_key[(scenario["scenario_id"], q_mode, method_id, "reference")],
                    by_key[(scenario["scenario_id"], q_mode, method_id, "reported")],
                    scenario,
                    participant_ids,
                )
                pair["gamma"] = gamma_by_scenario[pair["scenario_id"]]
                pairs.append(pair)

    gates = {
        "SOURCE_AND_RATE_BINDING": bool(
            execution.get("source_envelope_hash") == source_hash
            and policy.get("policy_hash") == source.get("policy_hash")
        ),
        "SOLVER_COMPLETENESS": all(row.get("solver_valid") for row in side_records),
        "AC_NUMERICAL_COMPLETENESS": all(row.get("ac_numerical_valid") for row in side_records),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(pair.get("matched_q_pair_pass") for pair in pairs),
        "FULL_FACTORIAL_POINT_INCLUDED": any(abs(float(pair["gamma"]) - 1.74) < 1e-12 for pair in pairs),
        "NO_LIMIT_OR_PARAMETER_REFIT": True,
    }
    result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": "REQUEST_MULTIPLIER_SENSITIVITY_COMPLETE" if all(gates.values()) else "REQUEST_MULTIPLIER_SENSITIVITY_BLOCKED",
        "scientific_role": "POST_FREEZE_DIAGNOSTIC_SENSITIVITY",
        "interpretation": "The sweep tests dependence on request magnitude; it does not estimate behavioral prevalence.",
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "source_scenario_id": evaluation_ids[0],
        "reporter_id": source_scenario["reporter_id"],
        "gamma_grid": [float(value) for value in sorted(set(GAMMA_GRID))],
        "full_factorial_gamma": 1.74,
        "full_factorial_gamma_provenance": "1.20 pivot multiplier times 1.45 report-scale endpoint",
        "method_roster": list(METHODS),
        "q_roster": list(source["q_roster"]),
        "pair_count": len(pairs),
        "matched_q_pair_pass_count": sum(bool(pair["matched_q_pair_pass"]) for pair in pairs),
        "gate_results": gates,
        "input_read_assertions": {
            "historical_outputs_read": False,
            "branch_ratings_refit": False,
            "calibrated_margins_refit": False,
            "gamma_selected_from_ac_results": False,
        },
        "pairs": pairs,
        "side_records": side_records,
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), pairs)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Request-multiplier sensitivity", "",
            f"- Status: `{result['status']}`",
            f"- Request multipliers: `{result['gamma_grid']}`",
            f"- Paired AC comparisons: `{result['matched_q_pair_pass_count']}/{result['pair_count']}` pass.",
            "- The frozen 1.74 point equals the 1.20 pivot multiplier times the 1.45 report-scale endpoint.",
            "- No branch rating, calibrated margin, or allocation parameter is refitted from this sweep.",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "fresh_source_design"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument("--policy", type=Path, default=root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml")
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "request_multiplier_sensitivity_v12_3" / "REQUEST_MULTIPLIER_SENSITIVITY_RESULTS.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    result = run_sensitivity(
        case_path=args.case,
        policy_path=args.policy,
        source_path=args.source,
        execution_path=args.execution,
        output_path=args.output,
        workers=max(1, args.workers),
    )
    print(canonical_dumps({
        "status": result["status"],
        "pair_count": result["pair_count"],
        "matched_q_pair_pass_count": result["matched_q_pair_pass_count"],
        "gates": result["gate_results"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
