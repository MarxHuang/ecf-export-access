"""Run dynamic horizontal controls against the frozen V12.3 feedback paths.

The experiment treats the recursively generated beta=1 path as the proposed
feedback endpoint.  At every interval, a uniform control receives exactly the
same aggregate request attenuation but distributes it over all participants.
A last-delivery control caps each new request by its own preceding metered
delivery.  Both controls retain the frozen network, allocation rule, Q model,
calibrated margin, and AC implementation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from r4r.metrics import normalized_jain, raw_jain
from r4r.serialization import canonical_dumps, canonical_hash
import tools.run_v12_3_dynamic_feedback_experiment as dynamic


DESIGN_SERIALIZATION_ID = "r4r.behavior_coupled.dynamic_horizontal_control_design.v2"
RESULT_SERIALIZATION_ID = "r4r.behavior_coupled.dynamic_horizontal_control_results.v3"
NO_UPDATE = "NO_UPDATE_BETA_000"
DYNAMIC_CF = "DYNAMIC_CF_BETA_100"
MATCHED_UNIFORM = "MATCHED_UNIFORM_BY_INTERVAL"
LAST_DELIVERY = "LAST_DELIVERY_CAP"
MODE_ROSTER = (NO_UPDATE, DYNAMIC_CF, MATCHED_UNIFORM, LAST_DELIVERY)


def _defined_value(metric: Any) -> float | None:
    value = metric.to_json()
    return float(value["value"]) if value.get("status") == "DEFINED" else None


def matched_uniform_request(
    capacity_limited_request_mw: Sequence[float],
    cf_filtered_request_mw: Sequence[float],
) -> list[float]:
    """Uniformly distribute CF attenuation over the admissible request base."""

    base = [float(value) for value in capacity_limited_request_mw]
    filtered = [float(value) for value in cf_filtered_request_mw]
    if len(base) != len(filtered):
        raise ValueError("capacity-limited and CF-filtered request vectors must have equal length")
    total = math.fsum(base)
    attenuation = math.fsum(r - f for r, f in zip(base, filtered))
    if attenuation < -1.0e-10 or attenuation > total + 1.0e-10:
        raise ValueError("CF attenuation lies outside the aggregate request")
    if total <= 0.0:
        return list(base)
    factor = min(1.0, max(0.0, (total - attenuation) / total))
    return [factor * value for value in base]


def last_delivery_request(
    capacity_limited_request_mw: Sequence[float],
    preceding_delivery_mw: Sequence[float],
) -> list[float]:
    """Cap each admissible current request by preceding metered delivery."""

    base = [float(value) for value in capacity_limited_request_mw]
    delivery = [float(value) for value in preceding_delivery_mw]
    if len(base) != len(delivery):
        raise ValueError("capacity-limited request and preceding delivery vectors must have equal length")
    if any(value < -1.0e-12 for value in base + delivery):
        raise ValueError("requests and metered delivery must be nonnegative")
    return [min(request, prior) for request, prior in zip(base, delivery)]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def freeze_design(
    *, dynamic_design_path: Path, dynamic_result_path: Path, output_path: Path,
) -> dict[str, Any]:
    source_design = _load(dynamic_design_path)
    source_result = _load(dynamic_result_path)
    design_hash = dynamic._validated_hash(
        source_design, field="design_hash", label="dynamic design",
    )
    result_hash = dynamic._validated_hash(
        source_result, field="result_hash", label="dynamic result",
    )
    if source_result.get("design_hash") != design_hash:
        raise ValueError("dynamic result is not bound to the frozen design")
    if source_result.get("status") != "DYNAMIC_FEEDBACK_EXPERIMENT_COMPLETE":
        raise ValueError("dynamic source result is incomplete")
    if not all(source_result.get("gate_results", {}).values()):
        raise ValueError("dynamic source result contains an open gate")

    design: dict[str, Any] = {
        "serialization_id": DESIGN_SERIALIZATION_ID,
        "status": "DYNAMIC_HORIZONTAL_CONTROL_DESIGN_FROZEN",
        "source_dynamic_design_hash": design_hash,
        "source_dynamic_result_hash": result_hash,
        "source_envelope_hash": source_result["source_envelope_hash"],
        "method_execution_hash": source_result["method_execution_hash"],
        "policy_hash": source_result["policy_hash"],
        "network_hash": source_result["network_hash"],
        "horizon": source_design["horizon"],
        "scenario_ids": [row["scenario_id"] for row in source_design["scenarios"]],
        "profile_common_scenario_ids": list(source_design["profile_robustness_scenario_ids"]),
        "schedule_ids": [row["schedule_id"] for row in source_design["schedules"]],
        "method_roster": list(source_design["method_roster"]),
        "q_roster": list(source_design["q_roster"]),
        "mode_roster": list(MODE_ROSTER),
        "control_definitions": {
            DYNAMIC_CF: {
                "definition": "recursively generated delivery-consistency filter at the predeclared beta=1 endpoint",
                "selection_role": "PREDECLARED_FULL_REPLACEMENT_ENDPOINT_NOT_RESULT_SELECTED",
            },
            MATCHED_UNIFORM: {
                "definition": "uniformly scale every interval's capacity-limited request vector to match the aggregate attenuation of the beta=1 CF path in that same interval",
                "purpose": "separate targeting from aggregate request reduction",
            },
            LAST_DELIVERY: {
                "definition": "cap each capacity-limited current request by the same participant's preceding metered delivery, initialized by the frozen capability benchmark",
                "purpose": "conservative causal persistence control",
            },
        },
        "input_read_assertions": {
            "control_outcomes_used_to_choose_modes": False,
            "control_outcomes_used_to_choose_scenarios": False,
            "control_outcomes_used_to_choose_beta": False,
            "control_outcomes_used_to_refit_rate_or_margin": False,
            "historical_forbidden_outputs_read": False,
        },
        "design_hash": None,
    }
    design["design_hash"] = canonical_hash(design)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(design) + "\n", encoding="utf-8")
    return design


def _solve_reported(
    scenario: Mapping[str, Any], *, q_mode: str, method_id: str, mode: str,
    raw_request: Sequence[float], transformed_request: Sequence[float],
) -> dict[str, Any]:
    step_scenario = copy.deepcopy(dict(scenario))
    step_scenario["reported_request_mw"] = [float(value) for value in raw_request]
    attempts: list[dict[str, Any]] = []
    candidate: dict[str, Any] = {}
    for attempt in range(1, 4):
        candidate = dynamic._run_side_with_request(
            step_scenario,
            q_mode=q_mode,
            method_id=method_id,
            mode=mode,
            side="reported",
            request=[float(value) for value in transformed_request],
            margin_ratio=0.0,
        )
        attempts.append({
            "attempt": attempt,
            "solver_valid": bool(candidate.get("solver_valid")),
            "solver_status": candidate.get("solver", {}).get("solver_status"),
            "failure_reason": candidate.get("solver", {}).get("failure_reason"),
        })
        if candidate.get("solver_valid"):
            break
    candidate["raw_request_mw"] = [float(value) for value in raw_request]
    candidate["transformed_request_mw"] = [float(value) for value in transformed_request]
    candidate["transformed_request_hash"] = canonical_hash({
        "participant_ids": list(dynamic._DYNAMIC_WORKER["participant_ids"]),
        "values_mw": candidate["transformed_request_mw"],
    })
    candidate["identical_problem_solve_attempts"] = attempts
    return candidate


def _step_record(
    *, trajectory_id: str, interval: int, gamma: float, mode: str,
    reference: Mapping[str, Any], reported: Mapping[str, Any],
    scenario: Mapping[str, Any], raw_request: Sequence[float],
    capacity_limited_request: Sequence[float],
    filtered_request: Sequence[float], preceding_delivery: Sequence[float] | None,
) -> dict[str, Any]:
    participants = list(dynamic._DYNAMIC_WORKER["participant_ids"])
    pair = dynamic._feedback_pair_record(reference, reported, scenario, participants)
    allocation = [float(value) for value in reported.get("allocation_mw", [])]
    delivery, ratios = dynamic._delivery_ratio(
        allocation, scenario["reference_request_mw"],
        tolerance=dynamic.AUTHORIZATION_TOLERANCE_MW,
    ) if pair.get("metrics_valid") else ([], [])
    participant_index = {item: index for index, item in enumerate(participants)}
    coalition_indices = [participant_index[item] for item in scenario["coalition_ids"]]
    delivery_outcomes = dynamic._delivery_outcome_metrics(
        reference["allocation_mw"], allocation,
        scenario["reference_request_mw"], coalition_indices,
    ) if pair.get("metrics_valid") else {}
    metrics = pair.get("coalition_metrics", {})
    jr = pair.get("jain_raw_reference", {})
    jp = pair.get("jain_raw_reported", {})
    nr = pair.get("jain_norm_reference", {})
    np = pair.get("jain_norm_reported", {})
    return {
        "trajectory_id": trajectory_id,
        "interval": int(interval),
        "gamma": float(gamma),
        "mismatch_active": bool(abs(float(gamma) - 1.0) > 1.0e-12),
        "mode": mode,
        "raw_request_mw": [float(value) for value in raw_request],
        "capacity_limited_request_mw": [
            float(value) for value in capacity_limited_request
        ],
        "filtered_request_mw": [float(value) for value in filtered_request],
        "preceding_delivery_mw": (
            [float(value) for value in preceding_delivery]
            if preceding_delivery is not None else None
        ),
        "request_attenuation_mw": math.fsum(
            float(base) - float(filtered)
            for base, filtered in zip(capacity_limited_request, filtered_request)
        ),
        "allocation_mw": allocation,
        "delivery_mw": delivery,
        "delivery_ratio": ratios,
        **delivery_outcomes,
        "reference_total_export_mw": pair.get("reference_total_export_mw"),
        "reported_total_export_mw": pair.get("reported_total_export_mw"),
        "total_export_change_mw": pair.get("total_export_change_mw"),
        "coalition_gain_mw": metrics.get("coalition_gain_mw"),
        "coalition_undeliverable_mw": metrics.get("coalition_undeliverable_mw"),
        "noncoalition_loss_mw": metrics.get("noncoalition_loss_mw"),
        "redistribution_mw": metrics.get("redistribution_mw"),
        "jain_raw_change": (
            float(jp["value"]) - float(jr["value"])
            if jp.get("status") == jr.get("status") == "DEFINED" else None
        ),
        "jain_norm_change": (
            float(np["value"]) - float(nr["value"])
            if np.get("status") == nr.get("status") == "DEFINED" else None
        ),
        "solver_valid": bool(reported.get("solver_valid")),
        "ac_numerical_valid": bool(reported.get("ac_numerical_valid")),
        "matched_q_pair_pass": bool(pair.get("matched_q_pair_pass")),
        "maximum_loading_ratio": reported.get("screen", {}).get("maximum_loading_ratio"),
        "minimum_voltage_pu": reported.get("screen", {}).get("minimum_voltage_pu"),
        "max_mva_excess": reported.get("screen", {}).get("max_mva_excess"),
        "allocation_hash": reported.get("allocation_hash"),
        "method_execution_hash": reported.get("method_execution_hash"),
        "ac_result_hash": reported.get("ac_result_hash"),
        "identical_problem_solve_attempt_count": len(
            reported.get("identical_problem_solve_attempts", [])
        ),
    }


def _summary(
    *, scenario: Mapping[str, Any], schedule_id: str, q_mode: str,
    method_id: str, mode: str, steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in steps if row.get(field) is not None]

    def mean(field: str) -> float:
        rows = values(field)
        return math.fsum(rows) / len(rows) if rows else 0.0

    result = {
        "trajectory_id": "::".join([
            scenario["scenario_id"], schedule_id, mode, q_mode, method_id,
        ]),
        "scenario_id": scenario["scenario_id"],
        "coalition_size": len(scenario["coalition_ids"]),
        "spatial_pattern": scenario["spatial_pattern"],
        "schedule_id": schedule_id,
        "mode": mode,
        "q_mode": q_mode,
        "method_id": method_id,
        "interval_count": len(steps),
        "valid_interval_count": sum(
            bool(row["solver_valid"] and row["ac_numerical_valid"]) for row in steps
        ),
        "pair_pass_count": sum(bool(row["matched_q_pair_pass"]) for row in steps),
        "mean_request_attenuation_mw": mean("request_attenuation_mw"),
        "mean_redistribution_mw": mean("redistribution_mw"),
        "mean_coalition_undeliverable_mw": mean("coalition_undeliverable_mw"),
        "mean_coalition_gain_mw": mean("coalition_gain_mw"),
        "mean_noncoalition_loss_mw": mean("noncoalition_loss_mw"),
        "mean_reported_total_export_mw": mean("reported_total_export_mw"),
        "mean_total_export_change_mw": mean("total_export_change_mw"),
        "mean_reference_total_delivery_mw": mean("reference_total_delivery_mw"),
        "mean_reported_total_delivery_mw": mean("reported_total_delivery_mw"),
        "mean_total_delivery_change_mw": mean("total_delivery_change_mw"),
        "mean_system_delivery_loss_mw": mean("system_delivery_loss_mw"),
        "mean_reported_total_idle_award_mw": mean("reported_total_idle_award_mw"),
        "mean_total_idle_award_change_mw": mean("total_idle_award_change_mw"),
        "mean_coalition_net_delivery_change_mw": mean("coalition_net_delivery_change_mw"),
        "mean_noncoalition_net_delivery_loss_mw": mean("noncoalition_net_delivery_loss_mw"),
        "mean_delivery_utilization_ratio": mean("delivery_utilization_ratio"),
        "mean_jain_raw_change": mean("jain_raw_change"),
        "mean_jain_norm_change": mean("jain_norm_change"),
        "maximum_loading_ratio": max(values("maximum_loading_ratio"), default=0.0),
        "minimum_voltage_pu": min(values("minimum_voltage_pu"), default=None),
        "maximum_mva_excess": max(values("max_mva_excess"), default=0.0),
        "steps": list(steps),
    }
    result["trajectory_hash"] = canonical_hash(result)
    return result


def _control_task(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario, beta_one, schedule = task
    q_mode = beta_one["q_mode"]
    method_id = beta_one["method_id"]
    reference = dynamic._DYNAMIC_WORKER["reference_records"][f"{q_mode}::{method_id}"]
    uniform_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    last_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    uniform_steps: list[dict[str, Any]] = []
    last_steps: list[dict[str, Any]] = []
    preceding_delivery = [float(value) for value in scenario["reference_request_mw"]]

    for source_step in beta_one["steps"]:
        interval = int(source_step["interval"])
        gamma = float(source_step["gamma"])
        raw = [float(value) for value in source_step["raw_request_mw"]]
        base = [
            float(value) for value in source_step["capacity_limited_request_mw"]
        ]
        cf_filtered = [float(value) for value in source_step["filtered_request_mw"]]

        uniform_request = matched_uniform_request(base, cf_filtered)
        uniform_key = tuple(uniform_request)
        if uniform_key not in uniform_cache:
            uniform_cache[uniform_key] = _solve_reported(
                scenario, q_mode=q_mode, method_id=method_id,
                mode=MATCHED_UNIFORM, raw_request=raw,
                transformed_request=uniform_request,
            )
        uniform_reported = copy.deepcopy(uniform_cache[uniform_key])
        uniform_reported["raw_request_mw"] = raw
        uniform_reported["transformed_request_mw"] = uniform_request
        uniform_id = "::".join([
            scenario["scenario_id"], schedule["schedule_id"],
            MATCHED_UNIFORM, q_mode, method_id,
        ])
        uniform_steps.append(_step_record(
            trajectory_id=uniform_id, interval=interval, gamma=gamma,
            mode=MATCHED_UNIFORM, reference=reference, reported=uniform_reported,
            scenario=scenario, raw_request=raw,
            capacity_limited_request=base,
            filtered_request=uniform_request, preceding_delivery=None,
        ))

        previous = list(preceding_delivery)
        last_request = last_delivery_request(base, previous)
        last_key = tuple(last_request)
        if last_key not in last_cache:
            last_cache[last_key] = _solve_reported(
                scenario, q_mode=q_mode, method_id=method_id,
                mode=LAST_DELIVERY, raw_request=raw,
                transformed_request=last_request,
            )
        last_reported = copy.deepcopy(last_cache[last_key])
        last_reported["raw_request_mw"] = raw
        last_reported["transformed_request_mw"] = last_request
        last_id = "::".join([
            scenario["scenario_id"], schedule["schedule_id"],
            LAST_DELIVERY, q_mode, method_id,
        ])
        last_step = _step_record(
            trajectory_id=last_id, interval=interval, gamma=gamma,
            mode=LAST_DELIVERY, reference=reference, reported=last_reported,
            scenario=scenario, raw_request=raw,
            capacity_limited_request=base, filtered_request=last_request,
            preceding_delivery=previous,
        )
        last_steps.append(last_step)
        preceding_delivery = list(last_step["delivery_mw"])

    return [
        _summary(
            scenario=scenario, schedule_id=schedule["schedule_id"],
            q_mode=q_mode, method_id=method_id,
            mode=MATCHED_UNIFORM, steps=uniform_steps,
        ),
        _summary(
            scenario=scenario, schedule_id=schedule["schedule_id"],
            q_mode=q_mode, method_id=method_id,
            mode=LAST_DELIVERY, steps=last_steps,
        ),
    ]


def _existing_trajectory(
    source_row: Mapping[str, Any], scenario: Mapping[str, Any],
    reference: Mapping[str, Any], mode: str,
) -> dict[str, Any]:
    reference_allocation = [float(value) for value in reference["allocation_mw"]]
    raw_reference = _defined_value(raw_jain(reference_allocation))
    norm_reference = _defined_value(normalized_jain(
        reference_allocation, scenario["capacity_mw"],
    ))
    steps: list[dict[str, Any]] = []
    trajectory_id = "::".join([
        scenario["scenario_id"], source_row["schedule_id"], mode,
        source_row["q_mode"], source_row["method_id"],
    ])
    for source_step in source_row["steps"]:
        allocation = [float(value) for value in source_step["allocation_mw"]]
        raw_value = _defined_value(raw_jain(allocation))
        norm_value = _defined_value(normalized_jain(allocation, scenario["capacity_mw"]))
        steps.append({
            "trajectory_id": trajectory_id,
            "interval": int(source_step["interval"]),
            "gamma": float(source_step["gamma"]),
            "mismatch_active": bool(source_step["mismatch_active"]),
            "mode": mode,
            "raw_request_mw": list(source_step["raw_request_mw"]),
            "capacity_limited_request_mw": list(
                source_step["capacity_limited_request_mw"]
            ),
            "filtered_request_mw": list(source_step["filtered_request_mw"]),
            "preceding_delivery_mw": None,
            "request_attenuation_mw": float(source_step["request_attenuation_mw"]),
            "allocation_mw": allocation,
            "delivery_mw": list(source_step["delivery_mw"]),
            "delivery_ratio": list(source_step["delivery_ratio"]),
            "reference_total_delivery_mw": source_step["reference_total_delivery_mw"],
            "reported_total_delivery_mw": source_step["reported_total_delivery_mw"],
            "total_delivery_change_mw": source_step["total_delivery_change_mw"],
            "system_delivery_loss_mw": source_step["system_delivery_loss_mw"],
            "reference_total_idle_award_mw": source_step["reference_total_idle_award_mw"],
            "reported_total_idle_award_mw": source_step["reported_total_idle_award_mw"],
            "total_idle_award_change_mw": source_step["total_idle_award_change_mw"],
            "coalition_net_delivery_change_mw": source_step["coalition_net_delivery_change_mw"],
            "noncoalition_net_delivery_loss_mw": source_step["noncoalition_net_delivery_loss_mw"],
            "delivery_utilization_ratio": source_step["delivery_utilization_ratio"],
            "reference_total_export_mw": source_step["reference_total_export_mw"],
            "reported_total_export_mw": source_step["reported_total_export_mw"],
            "total_export_change_mw": source_step["total_export_change_mw"],
            "coalition_gain_mw": source_step["coalition_gain_mw"],
            "coalition_undeliverable_mw": source_step["coalition_undeliverable_mw"],
            "noncoalition_loss_mw": source_step["noncoalition_loss_mw"],
            "redistribution_mw": source_step["redistribution_mw"],
            "jain_raw_change": (
                raw_value - raw_reference
                if raw_value is not None and raw_reference is not None else None
            ),
            "jain_norm_change": (
                norm_value - norm_reference
                if norm_value is not None and norm_reference is not None else None
            ),
            "solver_valid": bool(source_step["solver_valid"]),
            "ac_numerical_valid": bool(source_step["ac_numerical_valid"]),
            "matched_q_pair_pass": bool(source_step["matched_q_pair_pass"]),
            "maximum_loading_ratio": source_step["maximum_loading_ratio"],
            "minimum_voltage_pu": source_step["minimum_voltage_pu"],
            "max_mva_excess": source_step["max_mva_excess"],
            "allocation_hash": source_step["allocation_hash"],
            "method_execution_hash": source_step["method_execution_hash"],
            "ac_result_hash": source_step["ac_result_hash"],
            "identical_problem_solve_attempt_count": source_step.get(
                "identical_problem_solve_attempt_count", 0,
            ),
        })
    return _summary(
        scenario=scenario, schedule_id=source_row["schedule_id"],
        q_mode=source_row["q_mode"], method_id=source_row["method_id"],
        mode=mode, steps=steps,
    )


def _aggregate(
    rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    def mean(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return math.fsum(values) / len(values) if values else 0.0

    return {
        **dimensions,
        "trajectory_count": len(rows),
        "interval_count": sum(int(row["interval_count"]) for row in rows),
        "pair_pass_rate": (
            sum(int(row["pair_pass_count"]) for row in rows)
            / sum(int(row["interval_count"]) for row in rows)
        ),
        "mean_request_attenuation_mw": mean("mean_request_attenuation_mw"),
        "mean_redistribution_mw": mean("mean_redistribution_mw"),
        "mean_coalition_undeliverable_mw": mean("mean_coalition_undeliverable_mw"),
        "mean_coalition_gain_mw": mean("mean_coalition_gain_mw"),
        "mean_noncoalition_loss_mw": mean("mean_noncoalition_loss_mw"),
        "mean_reported_total_export_mw": mean("mean_reported_total_export_mw"),
        "mean_total_export_change_mw": mean("mean_total_export_change_mw"),
        "mean_reference_total_delivery_mw": mean("mean_reference_total_delivery_mw"),
        "mean_reported_total_delivery_mw": mean("mean_reported_total_delivery_mw"),
        "mean_total_delivery_change_mw": mean("mean_total_delivery_change_mw"),
        "mean_system_delivery_loss_mw": mean("mean_system_delivery_loss_mw"),
        "mean_reported_total_idle_award_mw": mean("mean_reported_total_idle_award_mw"),
        "mean_total_idle_award_change_mw": mean("mean_total_idle_award_change_mw"),
        "mean_coalition_net_delivery_change_mw": mean("mean_coalition_net_delivery_change_mw"),
        "mean_noncoalition_net_delivery_loss_mw": mean("mean_noncoalition_net_delivery_loss_mw"),
        "mean_delivery_utilization_ratio": mean("mean_delivery_utilization_ratio"),
        "mean_jain_raw_change": mean("mean_jain_raw_change"),
        "mean_jain_norm_change": mean("mean_jain_norm_change"),
        "maximum_loading_ratio": max(float(row["maximum_loading_ratio"]) for row in rows),
        "minimum_voltage_pu": min(float(row["minimum_voltage_pu"]) for row in rows),
        "maximum_mva_excess": max(float(row["maximum_mva_excess"]) for row in rows),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_controls(
    *, case_path: Path, policy_path: Path, source_path: Path,
    execution_path: Path, dynamic_design_path: Path, dynamic_result_path: Path,
    control_design_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = dynamic._load_yaml(policy_path)
    source = dynamic._load_json(source_path)
    execution = dynamic._load_json(execution_path)
    dynamic_design = _load(dynamic_design_path)
    dynamic_result = _load(dynamic_result_path)
    control_design = _load(control_design_path)
    source_hash = dynamic._validated_hash(source, field="result_hash", label="source envelope")
    execution_hash = dynamic._validated_hash(execution, field="result_hash", label="method execution")
    dynamic_design_hash = dynamic._validated_hash(dynamic_design, field="design_hash", label="dynamic design")
    dynamic_result_hash = dynamic._validated_hash(dynamic_result, field="result_hash", label="dynamic result")
    control_design_hash = dynamic._validated_hash(control_design, field="design_hash", label="control design")
    if control_design.get("status") != "DYNAMIC_HORIZONTAL_CONTROL_DESIGN_FROZEN":
        raise ValueError("dynamic horizontal control design is not frozen")
    if control_design.get("source_dynamic_result_hash") != dynamic_result_hash:
        raise ValueError("control design is not bound to the current dynamic result")
    if dynamic_result.get("design_hash") != dynamic_design_hash:
        raise ValueError("dynamic result/design binding mismatch")
    if dynamic_result.get("source_envelope_hash") != source_hash:
        raise ValueError("dynamic result/source binding mismatch")
    if dynamic_result.get("method_execution_hash") != execution_hash:
        raise ValueError("dynamic result/execution binding mismatch")
    if dynamic_result.get("status") != "DYNAMIC_FEEDBACK_EXPERIMENT_COMPLETE":
        raise ValueError("dynamic source result is incomplete")

    source_scenario_id = dynamic_design["scenarios"][0]["source_scenario_id"]
    margins = dynamic._extract_margins(execution, source, source_scenario_id)
    references = dynamic._build_reference_records(
        case_path=case_path, policy=policy, source=source,
        margins=margins, design=dynamic_design,
    )
    scenario_by_id = {row["scenario_id"]: row for row in dynamic_design["scenarios"]}
    schedule_by_id = {row["schedule_id"]: row for row in dynamic_design["schedules"]}
    beta_zero = {
        (row["scenario_id"], row["schedule_id"], row["q_mode"], row["method_id"]): row
        for row in dynamic_result["trajectory_summaries"]
        if row["policy_id"] == "DYNAMIC_BETA_000"
    }
    beta_one = {
        (row["scenario_id"], row["schedule_id"], row["q_mode"], row["method_id"]): row
        for row in dynamic_result["trajectory_summaries"]
        if row["policy_id"] == "DYNAMIC_BETA_100"
    }
    if set(beta_zero) != set(beta_one):
        raise ValueError("beta=0 and beta=1 source trajectory rosters differ")

    tasks = [
        (scenario_by_id[key[0]], beta_one[key], schedule_by_id[key[1]])
        for key in sorted(beta_one)
    ]
    workers = max(1, int(workers))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=dynamic._init_dynamic_worker,
            initargs=(
                str(case_path), policy, source, margins, references,
                list(source["participant_ids"]),
            ),
        ) as executor:
            nested = list(executor.map(
                _control_task, tasks,
                chunksize=max(1, len(tasks) // (workers * 6)),
            ))
    else:
        dynamic._init_dynamic_worker(
            str(case_path), policy, source, margins, references,
            list(source["participant_ids"]),
        )
        nested = [_control_task(task) for task in tasks]
    new_rows = [row for group in nested for row in group]

    existing_rows: list[dict[str, Any]] = []
    for key in sorted(beta_zero):
        scenario = scenario_by_id[key[0]]
        reference = references[f"{key[2]}::{key[3]}"]
        existing_rows.append(_existing_trajectory(beta_zero[key], scenario, reference, NO_UPDATE))
        existing_rows.append(_existing_trajectory(beta_one[key], scenario, reference, DYNAMIC_CF))
    trajectories = existing_rows + new_rows
    steps = [step for row in trajectories for step in row["steps"]]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    common_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    common_ids = set(control_design["profile_common_scenario_ids"])
    for row in trajectories:
        grouped[(row["schedule_id"], row["mode"])].append(row)
        if row["scenario_id"] in common_ids:
            common_grouped[(row["schedule_id"], row["mode"])].append(row)
    aggregate = [
        _aggregate(rows, {"schedule_id": key[0], "mode": key[1], "scope": "AVAILABLE_FROZEN_SCENARIOS"})
        for key, rows in sorted(grouped.items())
    ]
    profile_common = [
        _aggregate(rows, {"schedule_id": key[0], "mode": key[1], "scope": "COMMON_THREE_SCENARIOS"})
        for key, rows in sorted(common_grouped.items())
    ]

    uniform_by_id = {row["trajectory_id"].replace(MATCHED_UNIFORM, DYNAMIC_CF): row for row in new_rows if row["mode"] == MATCHED_UNIFORM}
    cf_by_id = {row["trajectory_id"]: row for row in existing_rows if row["mode"] == DYNAMIC_CF}
    attenuation_match = all(
        math.isclose(
            float(u_step["request_attenuation_mw"]),
            float(c_step["request_attenuation_mw"]),
            rel_tol=0.0, abs_tol=2.0e-12,
        )
        for cf_id, cf_row in cf_by_id.items()
        for uniform_row in [uniform_by_id[cf_id]]
        for c_step, u_step in zip(cf_row["steps"], uniform_row["steps"])
    )
    last_causal = all(
        all(
            float(filtered) <= float(previous) + 1.0e-12
            and float(filtered) <= float(base) + 1.0e-12
            for base, previous, filtered in zip(
                step["capacity_limited_request_mw"], step["preceding_delivery_mw"],
                step["filtered_request_mw"],
            )
        )
        for row in new_rows if row["mode"] == LAST_DELIVERY
        for step in row["steps"]
    )
    gates = {
        "FROZEN_SOURCE_BINDING": bool(
            control_design["source_dynamic_design_hash"] == dynamic_design_hash
            and control_design["source_dynamic_result_hash"] == dynamic_result_hash
            and dynamic_result["source_envelope_hash"] == source_hash
            and dynamic_result["method_execution_hash"] == execution_hash
        ),
        "DESIGN_FROZEN_BEFORE_CONTROL_EXECUTION": control_design["status"] == "DYNAMIC_HORIZONTAL_CONTROL_DESIGN_FROZEN",
        "SOURCE_DYNAMIC_GATES_CLOSED": all(dynamic_result["gate_results"].values()),
        "TRAJECTORY_ROSTER_COMPLETE": len(trajectories) == 4 * len(beta_one),
        "INTERVAL_ROSTER_COMPLETE": len(steps) == 4 * len(beta_one) * int(control_design["horizon"]),
        "NEW_SOLVER_COMPLETENESS": all(step["solver_valid"] for row in new_rows for step in row["steps"]),
        "NEW_AC_NUMERICAL_COMPLETENESS": all(step["ac_numerical_valid"] for row in new_rows for step in row["steps"]),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(step["matched_q_pair_pass"] for step in steps),
        "DELIVERY_ACCOUNTING_CLOSED": all(
            abs(math.fsum(float(value) for value in step["delivery_mw"])
                - float(step["reported_total_delivery_mw"])) <= 2.0e-12
            and abs(
                float(step["reported_total_delivery_mw"])
                + float(step["reported_total_idle_award_mw"])
                - float(step["reported_total_export_mw"])
            ) <= 2.0e-9
            for step in steps
        ),
        "MATCHED_UNIFORM_INTERVAL_ATTENUATION": attenuation_match,
        "LAST_DELIVERY_CAUSAL_CAP": last_causal,
        "RESULT_FEEDBACK_EXCLUDED": all(
            value is False for value in control_design["input_read_assertions"].values()
        ),
    }
    result: dict[str, Any] = {
        "serialization_id": RESULT_SERIALIZATION_ID,
        "status": "DYNAMIC_HORIZONTAL_CONTROLS_COMPLETE" if all(gates.values()) else "DYNAMIC_HORIZONTAL_CONTROLS_BLOCKED",
        "scientific_role": "MULTI_INTERVAL_CAUSAL_HORIZONTAL_CONTROL_EVIDENCE",
        "design_hash": control_design_hash,
        "source_dynamic_result_hash": dynamic_result_hash,
        "source_dynamic_design_hash": dynamic_design_hash,
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "worker_count": workers,
        "mode_roster": list(MODE_ROSTER),
        "trajectory_count": len(trajectories),
        "interval_record_count": len(steps),
        "new_control_trajectory_count": len(new_rows),
        "new_control_interval_count": sum(len(row["steps"]) for row in new_rows),
        "gate_results": gates,
        "aggregate_summary": aggregate,
        "profile_common_summary": profile_common,
        "trajectory_summaries": trajectories,
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")

    trajectory_fields = [
        "trajectory_id", "scenario_id", "coalition_size", "spatial_pattern",
        "schedule_id", "mode", "q_mode", "method_id", "interval_count",
        "valid_interval_count", "pair_pass_count", "mean_request_attenuation_mw",
        "mean_redistribution_mw", "mean_coalition_undeliverable_mw",
        "mean_coalition_gain_mw", "mean_noncoalition_loss_mw",
        "mean_reported_total_export_mw", "mean_total_export_change_mw",
        "mean_reference_total_delivery_mw", "mean_reported_total_delivery_mw",
        "mean_total_delivery_change_mw", "mean_system_delivery_loss_mw",
        "mean_reported_total_idle_award_mw", "mean_total_idle_award_change_mw",
        "mean_coalition_net_delivery_change_mw", "mean_noncoalition_net_delivery_loss_mw",
        "mean_delivery_utilization_ratio",
        "mean_jain_raw_change", "mean_jain_norm_change", "maximum_loading_ratio",
        "minimum_voltage_pu", "maximum_mva_excess", "trajectory_hash",
    ]
    step_fields = [
        "trajectory_id", "interval", "gamma", "mismatch_active", "mode",
        "request_attenuation_mw", "reference_total_export_mw",
        "reported_total_export_mw", "total_export_change_mw",
        "reference_total_delivery_mw", "reported_total_delivery_mw",
        "total_delivery_change_mw", "system_delivery_loss_mw",
        "reported_total_idle_award_mw", "total_idle_award_change_mw",
        "coalition_net_delivery_change_mw", "noncoalition_net_delivery_loss_mw",
        "delivery_utilization_ratio", "coalition_gain_mw",
        "coalition_undeliverable_mw", "noncoalition_loss_mw", "redistribution_mw",
        "jain_raw_change", "jain_norm_change", "solver_valid",
        "ac_numerical_valid", "matched_q_pair_pass", "maximum_loading_ratio",
        "minimum_voltage_pu", "max_mva_excess", "allocation_hash",
        "method_execution_hash", "ac_result_hash",
    ]
    _write_csv(output_path.with_name("DYNAMIC_HORIZONTAL_CONTROL_TRAJECTORIES.csv"), trajectories, trajectory_fields)
    _write_csv(output_path.with_name("DYNAMIC_HORIZONTAL_CONTROL_INTERVALS.csv"), steps, step_fields)
    _write_csv(output_path.with_name("DYNAMIC_HORIZONTAL_CONTROL_AGGREGATES.csv"), aggregate + profile_common, list((aggregate + profile_common)[0]))
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Dynamic horizontal controls",
            "",
            f"- Status: `{result['status']}`",
            f"- Four-mode trajectories: `{len(trajectories)}`; interval records: `{len(steps)}`.",
            f"- Newly solved control trajectories: `{len(new_rows)}`.",
            f"- Gates: `{gates}`",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "fresh_source_design"
    dynamic_out = root / "outputs" / "dynamic_feedback_v12_3"
    out = root / "outputs" / "dynamic_horizontal_controls_v12_3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-design", action="store_true")
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument("--policy", type=Path, default=root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml")
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument("--dynamic-design", type=Path, default=dynamic_out / "DYNAMIC_FEEDBACK_DESIGN_MANIFEST.json")
    parser.add_argument("--dynamic-result", type=Path, default=dynamic_out / "DYNAMIC_FEEDBACK_RESULTS.json")
    parser.add_argument("--control-design", type=Path, default=out / "DYNAMIC_HORIZONTAL_CONTROL_DESIGN.json")
    parser.add_argument("--output", type=Path, default=out / "DYNAMIC_HORIZONTAL_CONTROL_RESULTS.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    if args.freeze_design:
        design = freeze_design(
            dynamic_design_path=args.dynamic_design,
            dynamic_result_path=args.dynamic_result,
            output_path=args.control_design,
        )
        print(canonical_dumps({
            "status": design["status"], "design_hash": design["design_hash"],
        }))
        return 0
    result = run_controls(
        case_path=args.case, policy_path=args.policy, source_path=args.source,
        execution_path=args.execution, dynamic_design_path=args.dynamic_design,
        dynamic_result_path=args.dynamic_result, control_design_path=args.control_design,
        output_path=args.output, workers=max(1, args.workers),
    )
    print(canonical_dumps({
        "status": result["status"], "trajectory_count": result["trajectory_count"],
        "interval_record_count": result["interval_record_count"],
        "gates": result["gate_results"], "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DYNAMIC_CF", "LAST_DELIVERY", "MATCHED_UNIFORM", "MODE_ROSTER",
    "NO_UPDATE", "freeze_design", "last_delivery_request",
    "matched_uniform_request", "run_controls",
]
