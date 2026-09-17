"""Build a location-robust request-to-allocation response surface.

The experiment changes one participant request at a time for every one of the
30 frozen participants.  It preserves the feeder, participant capacities,
allocation rules, calibrated proxy margins, synthetic branch ratings, and
matched-Q AC screen.  The output is an allocation response surface for a
subsequent endogenous request-choice experiment; it is not itself a behavioral
model and it does not use any behavioral outcome to refit physical inputs.
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

from r4r.serialization import canonical_dumps, canonical_hash
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import METHODS
from tools.run_v12_3_multi_reporter_experiment import (
    _init_worker,
    _run_side,
    _run_side_with_request,
)
from tools.run_v12_3_request_multiplier_sensitivity import (
    GAMMA_GRID,
    _single_pair_record,
)


SERIALIZATION_ID = "r4r.location_robust_request_response.v1"


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


def _validated_hash(
    value: Mapping[str, Any], *, field: str, label: str,
) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    actual = canonical_hash(body)
    if not isinstance(declared, str) or declared != actual:
        raise ValueError(f"{label} self-hash mismatch")
    return actual


def _extract_margins(
    execution: Mapping[str, Any], source: Mapping[str, Any], source_scenario_id: str,
) -> dict[str, Mapping[str, Any]]:
    rows = [
        row for row in execution["rows"]
        if row["scenario_id"] == source_scenario_id and row["side"] == "reference"
    ]
    margins: dict[str, Mapping[str, Any]] = {}
    for q_mode in source["q_roster"]:
        for method_id in METHODS:
            matches = [
                row for row in rows
                if row["q_mode"] == q_mode and row["method_id"] == method_id
            ]
            if len(matches) != 1:
                raise ValueError(f"missing frozen margin for {q_mode}/{method_id}")
            margins[f"{q_mode}::{method_id}"] = matches[0]["margin_q"]
    return margins


def _scenario(
    *, source: Mapping[str, Any], source_scenario: Mapping[str, Any],
    reporter_index: int, gamma: float,
) -> dict[str, Any]:
    participant_ids = [str(value) for value in source["participant_ids"]]
    participant_bus_ids = [int(value) for value in source["participant_bus_ids"]]
    reference = [float(value) for value in source_scenario["reference_request_mw"]]
    reported = list(reference)
    reported[reporter_index] *= float(gamma)
    reporter_id = participant_ids[reporter_index]
    return {
        "scenario_id": (
            f"V14_{reporter_id.upper()}_GAMMA_{gamma:.2f}".replace(".", "P")
        ),
        "source_scenario_id": source_scenario["scenario_id"],
        "gamma": float(gamma),
        "reporter_id": reporter_id,
        "reporter_index": int(reporter_index),
        "reporter_bus_id": int(participant_bus_ids[reporter_index]),
        "spatial_pattern": "SINGLE_REQUEST_LOCATION_ROBUSTNESS",
        "participant_ids": participant_ids,
        "participant_bus_ids": participant_bus_ids,
        "coalition_ids": [reporter_id],
        "coalition_bus_ids": [int(participant_bus_ids[reporter_index])],
        "mean_pairwise_distance_edges": 0.0,
        "capacity_mw": [float(value) for value in source_scenario["capacity_mw"]],
        "reference_request_mw": reference,
        "reported_request_mw": reported,
    }


def _compact_side(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "scenario_id", "coalition_size", "spatial_pattern", "coalition_ids",
        "q_mode", "method_id", "mode", "side", "raw_request_mw",
        "transformed_request_mw", "transformed_request_hash", "margin_hash",
        "method_execution_hash", "solver", "solver_valid", "ac_numerical_valid",
        "physical_screen_pass", "screen", "allocation_mw", "allocation_hash",
        "ac_result_hash", "actual_values_hash", "crosscheck_status",
        "crosscheck_metrics", "primary_solution_hash", "independent_solution_hash",
    )
    return {field: record.get(field) for field in fields if field in record}


def _reference_scenario(
    source: Mapping[str, Any], source_scenario: Mapping[str, Any],
) -> dict[str, Any]:
    scenario = _scenario(
        source=source, source_scenario=source_scenario,
        reporter_index=0, gamma=1.0,
    )
    scenario["scenario_id"] = "V14_COMMON_REFERENCE"
    return scenario


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "reporter_id", "reporter_bus_id", "gamma", "q_mode", "method_id",
        "matched_q_pair_pass", "reference_total_export_mw",
        "reported_total_export_mw", "total_export_change_mw",
        "redistribution_mw", "reporter_access_gain_mw",
        "other_participant_access_loss_mw", "reported_maximum_loading_ratio",
        "reported_minimum_voltage_pu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["access_metrics"]
            writer.writerow({
                "reporter_id": row["reporter_id"],
                "reporter_bus_id": row["reporter_bus_id"],
                "gamma": row["gamma"],
                "q_mode": row["q_mode"],
                "method_id": row["method_id"],
                "matched_q_pair_pass": int(bool(row["matched_q_pair_pass"])),
                "reference_total_export_mw": row["reference_total_export_mw"],
                "reported_total_export_mw": row["reported_total_export_mw"],
                "total_export_change_mw": row["total_export_change_mw"],
                "redistribution_mw": metrics["redistribution_mw"],
                "reporter_access_gain_mw": metrics["sg_mw"],
                "other_participant_access_loss_mw": metrics["ol_minus_reporter_mw"],
                "reported_maximum_loading_ratio": (
                    row["reported_screen"]["maximum_loading_ratio"]
                ),
                "reported_minimum_voltage_pu": (
                    row["reported_screen"]["minimum_voltage_pu"]
                ),
            })


def run_response_surface(
    *, case_path: Path, policy_path: Path, source_path: Path,
    execution_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = _load_yaml(policy_path)
    source = _load_json(source_path)
    execution = _load_json(execution_path)
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    execution_hash = _validated_hash(
        execution, field="result_hash", label="method execution",
    )
    if execution.get("source_envelope_hash") != source_hash:
        raise ValueError("method execution is not bound to the source envelope")
    if policy.get("policy_hash") != source.get("policy_hash"):
        raise ValueError("RATE policy is not bound to the source envelope")

    evaluation_ids = list(source["scenario_split"]["evaluation"])
    if len(evaluation_ids) != 1:
        raise ValueError("expected exactly one frozen evaluation scenario")
    source_scenario = next(
        row for row in source["scenarios"] if row["scenario_id"] == evaluation_ids[0]
    )
    participant_ids = [str(value) for value in source["participant_ids"]]
    margins = _extract_margins(execution, source, evaluation_ids[0])

    common_reference_scenario = _reference_scenario(source, source_scenario)
    _init_worker(str(case_path), policy, source, margins)
    reference_records: dict[str, dict[str, Any]] = {}
    for q_mode in source["q_roster"]:
        for method_id in METHODS:
            record = _run_side_with_request(
                common_reference_scenario,
                q_mode=q_mode,
                method_id=method_id,
                mode="LOCATION_ROBUST_REFERENCE",
                side="reference",
                request=common_reference_scenario["reference_request_mw"],
                margin_ratio=0.0,
            )
            reference_records[f"{q_mode}::{method_id}"] = _compact_side(record)
    if not all(
        row.get("solver_valid") and row.get("ac_numerical_valid")
        for row in reference_records.values()
    ):
        raise ValueError("common reference construction failed")

    scenarios = [
        _scenario(
            source=source,
            source_scenario=source_scenario,
            reporter_index=index,
            gamma=gamma,
        )
        for index in range(len(participant_ids))
        for gamma in sorted(set(float(value) for value in GAMMA_GRID))
    ]
    tasks = [
        (scenario, q_mode, method_id, "None", "reported")
        for scenario in scenarios
        for q_mode in source["q_roster"]
        for method_id in METHODS
    ]
    workers = max(1, int(workers))
    reported_records: list[dict[str, Any]] = []
    if workers == 1:
        _init_worker(str(case_path), policy, source, margins)
        for task in tasks:
            reported_records.append(_compact_side(_run_side(task)))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(case_path), policy, source, margins),
        ) as executor:
            for record in executor.map(
                _run_side,
                tasks,
                chunksize=max(1, len(tasks) // (workers * 10)),
            ):
                reported_records.append(_compact_side(record))

    reported_by_key = {
        (row["scenario_id"], row["q_mode"], row["method_id"]): row
        for row in reported_records
    }
    if len(reported_by_key) != len(reported_records):
        raise ValueError("duplicate reported response record")
    pairs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for q_mode in source["q_roster"]:
            for method_id in METHODS:
                pair = _single_pair_record(
                    reference_records[f"{q_mode}::{method_id}"],
                    reported_by_key[(scenario["scenario_id"], q_mode, method_id)],
                    scenario,
                    participant_ids,
                )
                pair.update({
                    "gamma": float(scenario["gamma"]),
                    "reporter_id": scenario["reporter_id"],
                    "reporter_index": int(scenario["reporter_index"]),
                    "reporter_bus_id": int(scenario["reporter_bus_id"]),
                })
                pairs.append(pair)

    expected_pairs = (
        len(participant_ids) * len(set(GAMMA_GRID))
        * len(source["q_roster"]) * len(METHODS)
    )
    gates = {
        "SOURCE_RATE_EXECUTION_BINDING": bool(
            execution.get("source_envelope_hash") == source_hash
            and policy.get("policy_hash") == source.get("policy_hash")
        ),
        "ALL_PARTICIPANT_LOCATIONS_INCLUDED": {
            str(row["reporter_id"]) for row in pairs
        } == set(participant_ids),
        "RESPONSE_GRID_COMPLETE": len(pairs) == expected_pairs,
        "SOLVER_COMPLETENESS": all(
            row.get("solver_valid") for row in reported_records
        ),
        "AC_NUMERICAL_COMPLETENESS": all(
            row.get("ac_numerical_valid") for row in reported_records
        ),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(
            row.get("matched_q_pair_pass") for row in pairs
        ),
        "NO_LIMIT_MARGIN_OR_RULE_REFIT": True,
        "BEHAVIORAL_OUTCOME_NOT_USED_IN_RESPONSE_BUILD": True,
    }
    result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": (
            "LOCATION_ROBUST_REQUEST_RESPONSE_COMPLETE"
            if all(gates.values())
            else "LOCATION_ROBUST_REQUEST_RESPONSE_BLOCKED"
        ),
        "scientific_role": "PRE_BEHAVIOR_REQUEST_TO_ALLOCATION_RESPONSE_SURFACE",
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "source_scenario_id": evaluation_ids[0],
        "reporter_roster": participant_ids,
        "gamma_grid": sorted(set(float(value) for value in GAMMA_GRID)),
        "q_roster": list(source["q_roster"]),
        "method_roster": list(METHODS),
        "worker_count": workers,
        "pair_count": len(pairs),
        "matched_q_pair_pass_count": sum(
            bool(row["matched_q_pair_pass"]) for row in pairs
        ),
        "reference_records": reference_records,
        "reported_side_records": reported_records,
        "pairs": pairs,
        "gate_results": gates,
        "input_read_assertions": {
            "historical_behavioral_outputs_read": False,
            "behavioral_results_used_to_select_reporters": False,
            "behavioral_results_used_to_select_gamma_grid": False,
            "rate_or_margin_refit": False,
        },
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), pairs)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Location-robust request response surface",
            "",
            f"- Status: `{result['status']}`",
            f"- Reporters: `{len(participant_ids)}`; gamma values: `{len(result['gamma_grid'])}`.",
            f"- Matched-Q pairs: `{result['matched_q_pair_pass_count']}/{result['pair_count']}` pass.",
            "- The response surface precedes and is independent of the endogenous request-choice calculation.",
            "- No physical limit, calibrated margin, or allocation-rule parameter is refitted.",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = (
        root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT"
        / "fresh_source_design"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=root / "data" / "raw" / "case141.m",
    )
    parser.add_argument(
        "--policy", type=Path,
        default=(
            root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT"
            / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml"
        ),
    )
    parser.add_argument(
        "--source", type=Path,
        default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json",
    )
    parser.add_argument(
        "--execution", type=Path,
        default=fresh / "V12_3_METHOD_EXECUTION_V8.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            root / "outputs" / "endogenous_request_choice_v14"
            / "LOCATION_ROBUST_REQUEST_RESPONSE.json"
        ),
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(20, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args(argv)
    result = run_response_surface(
        case_path=args.case,
        policy_path=args.policy,
        source_path=args.source,
        execution_path=args.execution,
        output_path=args.output,
        workers=args.workers,
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


__all__ = ["SERIALIZATION_ID", "run_response_surface"]
