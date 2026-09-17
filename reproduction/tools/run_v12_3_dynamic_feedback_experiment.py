"""Run the post-freeze V12.3 multi-interval metered-feedback experiment.

The design is frozen before execution.  It reuses the current five allocation
rules, calibrated proxy margins, synthetic branch ratings, Q0/Q95 models, and
the two-ended MVA AC check.  No historical dynamic output is read.
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

import yaml

from r4r.serialization import canonical_dumps, canonical_hash
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import METHODS
from tools.run_v12_3_multi_reporter_experiment import (
    _init_worker,
    _pair_record,
    _run_side_with_request,
)
from tools.run_v12_3_request_multiplier_sensitivity import _single_pair_record


DESIGN_SERIALIZATION_ID = "r4r.behavior_coupled.dynamic_feedback_design.v2"
RESULT_SERIALIZATION_ID = "r4r.behavior_coupled.dynamic_feedback_results.v3"
HORIZON = 12
REPORT_MULTIPLIER = 1.74
BETA_GRID = (0.0, 0.10, 0.25, 0.50, 1.0)
FIXED_SCORE = 0.70
AUTHORIZATION_TOLERANCE_MW = 1.0e-9
SCHEDULES = {
    "PERSISTENT_MISMATCH": (REPORT_MULTIPLIER,) * HORIZON,
    "INTERMITTENT_MISMATCH": (REPORT_MULTIPLIER, REPORT_MULTIPLIER, 1.0, 1.0) * 3,
    "ONSET_AND_RECOVERY": (1.0,) * 3 + (REPORT_MULTIPLIER,) * 6 + (1.0,) * 3,
}
PROFILE_ROBUSTNESS_SCENARIOS = (
    "DF_SINGLE_P021",
    "MR_K05_LOCAL",
    "MR_K05_DISPERSED",
)

_DYNAMIC_WORKER: dict[str, Any] = {}


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


def _validated_hash(value: Mapping[str, Any], *, field: str, label: str) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    if not isinstance(declared, str) or canonical_hash(body) != declared:
        raise ValueError(f"{label} {field} is invalid")
    return declared


def _mean(values: Sequence[float]) -> float:
    return math.fsum(float(value) for value in values) / len(values) if values else 0.0


def _policy_roster() -> list[dict[str, Any]]:
    roster = [
        {
            "policy_id": f"DYNAMIC_BETA_{int(round(beta * 100)):03d}",
            "policy_type": "DYNAMIC_RECURSION",
            "beta": float(beta),
            "initial_score": 1.0,
        }
        for beta in BETA_GRID
    ]
    roster.append({
        "policy_id": "FIXED_SCORE_070",
        "policy_type": "FIXED_SCORE",
        "beta": None,
        "initial_score": FIXED_SCORE,
    })
    return roster


def _single_scenario(source: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_ids = list(source["scenario_split"]["evaluation"])
    if len(evaluation_ids) != 1:
        raise ValueError("the dynamic design requires exactly one frozen single-reporter evaluation scenario")
    source_scenario = next(
        item for item in source["scenarios"] if item["scenario_id"] == evaluation_ids[0]
    )
    reporter_index = int(source_scenario["reporter_index"])
    participant_ids = list(source["participant_ids"])
    participant_bus_ids = [int(value) for value in source["participant_bus_ids"]]
    return {
        "scenario_id": "DF_SINGLE_P021",
        "source_scenario_id": source_scenario["scenario_id"],
        "source_envelope_hash": source["result_hash"],
        "participant_ids": participant_ids,
        "participant_bus_ids": participant_bus_ids,
        "capacity_mw": [float(value) for value in source_scenario["capacity_mw"]],
        "reference_request_mw": [float(value) for value in source_scenario["reference_request_mw"]],
        "reported_request_mw": [float(value) for value in source_scenario["reported_request_mw"]],
        "coalition_ids": [participant_ids[reporter_index]],
        "coalition_bus_ids": [participant_bus_ids[reporter_index]],
        "spatial_pattern": "SINGLE",
        "mean_pairwise_distance_edges": 0.0,
        "minimum_pairwise_distance_edges": 0,
        "maximum_pairwise_distance_edges": 0,
        "report_multiplier": REPORT_MULTIPLIER,
        "selection_method_id": "FROZEN_SINGLE_REPORTER_EVALUATION_SCENARIO",
    }


def _normalize_multi_scenario(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["capacity_mw"] = [float(item) for item in value["capacity_mw"]]
    result["reference_request_mw"] = [float(item) for item in value["reference_request_mw"]]
    result["reported_request_mw"] = [float(item) for item in value["reported_request_mw"]]
    result["participant_bus_ids"] = [int(item) for item in value["participant_bus_ids"]]
    result["coalition_bus_ids"] = [int(item) for item in value["coalition_bus_ids"]]
    return result


def freeze_design(
    *, source_path: Path, multi_manifest_path: Path, policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a result-blind dynamic experiment design."""

    source = _load_json(source_path)
    multi = _load_json(multi_manifest_path)
    policy = _load_yaml(policy_path)
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    multi_hash = _validated_hash(multi, field="manifest_hash", label="multi-reporter manifest")
    if multi.get("source_envelope_hash") != source_hash:
        raise ValueError("multi-reporter design is not bound to the current source envelope")
    if policy.get("policy_hash") != source.get("policy_hash"):
        raise ValueError("RATE policy is not bound to the current source envelope")
    if tuple(multi.get("method_roster", ())) != tuple(METHODS):
        raise ValueError("multi-reporter method roster differs from the current five-rule roster")

    scenarios = [_single_scenario(source)] + [
        _normalize_multi_scenario(item) for item in multi["scenarios"]
    ]
    scenario_ids = [item["scenario_id"] for item in scenarios]
    missing = sorted(set(PROFILE_ROBUSTNESS_SCENARIOS) - set(scenario_ids))
    if missing:
        raise ValueError(f"profile-robustness scenarios are unavailable: {missing}")
    schedule_rows = [
        {
            "schedule_id": schedule_id,
            "horizon": HORIZON,
            "report_multiplier_by_interval": [float(value) for value in values],
            "changed_interval_count": sum(abs(value - 1.0) > 1.0e-12 for value in values),
        }
        for schedule_id, values in SCHEDULES.items()
    ]
    design: dict[str, Any] = {
        "serialization_id": DESIGN_SERIALIZATION_ID,
        "status": "DYNAMIC_FEEDBACK_DESIGN_FROZEN",
        "scientific_role": "POST_FREEZE_MULTI_INTERVAL_EVALUATION",
        "source_envelope_hash": source_hash,
        "multi_reporter_manifest_hash": multi_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "method_roster": list(METHODS),
        "q_roster": list(source["q_roster"]),
        "horizon": HORIZON,
        "report_multiplier": REPORT_MULTIPLIER,
        "authorization_tolerance_mw": AUTHORIZATION_TOLERANCE_MW,
        "delivery_policy": "CAP_AUTHORIZED_EXPORT_AT_FROZEN_REFERENCE_REQUEST",
        "score_update": "chi_i(t+1)=(1-beta)*chi_i(t)+beta*d_i(t)",
        "capacity_limited_request": "bar_request_i(t)=min(raw_request_i(t),capacity_i)",
        "score_application": "filtered_request_i(t)=min(bar_request_i(t),chi_i(t)*capacity_i)",
        "feedback_policies": _policy_roster(),
        "schedules": schedule_rows,
        "persistent_cross_scenario_ids": scenario_ids,
        "profile_robustness_scenario_ids": list(PROFILE_ROBUSTNESS_SCENARIOS),
        "scenarios": scenarios,
        "design_rationale": {
            "persistent_cross_scenario": "All frozen single- and simultaneous-request scenarios test the beta response under repeated mismatch.",
            "profile_robustness": "Single, local-five, and dispersed-five cases test intermittent mismatch and recovery without selecting cases from dynamic outcomes.",
            "beta_grid": "The grid spans no update, slow/intermediate updating, and full replacement; it is a sensitivity grid rather than an estimated behavioral parameter.",
            "fixed_score_comparator": "The fixed 0.70 score connects the multi-interval experiment to the existing state-conditioned CF comparison.",
        },
        "input_read_assertions": {
            "historical_dynamic_outputs_read": False,
            "dynamic_results_used_to_select_scenarios": False,
            "dynamic_results_used_to_select_beta": False,
            "dynamic_results_used_to_refit_rate_or_margin": False,
            "mitigation_results_used_to_select_schedule": False,
        },
        "design_hash": None,
    }
    design["design_hash"] = canonical_hash(design)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(design) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Frozen multi-interval feedback design",
            "",
            f"- Scenarios: `{len(scenarios)}`; horizon: `{HORIZON}` intervals.",
            f"- Dynamic beta grid: `{list(BETA_GRID)}`; fixed-score comparator: `{FIXED_SCORE}`.",
            f"- Schedules: `{list(SCHEDULES)}`.",
            "- All scenario and parameter choices precede execution of this experiment.",
            f"- Design hash: `{design['design_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return design


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


def _reference_scenario(design: Mapping[str, Any]) -> dict[str, Any]:
    scenario = copy.deepcopy(dict(design["scenarios"][0]))
    scenario["scenario_id"] = "DF_REFERENCE"
    scenario["coalition_ids"] = list(design["scenarios"][0]["coalition_ids"])
    scenario["coalition_bus_ids"] = list(design["scenarios"][0]["coalition_bus_ids"])
    return scenario


def _build_reference_records(
    *, case_path: Path, policy: Mapping[str, Any], source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]], design: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _init_worker(str(case_path), policy, source, margins)
    scenario = _reference_scenario(design)
    request = list(scenario["reference_request_mw"])
    result: dict[str, dict[str, Any]] = {}
    for q_mode in design["q_roster"]:
        for method_id in design["method_roster"]:
            record = _run_side_with_request(
                scenario,
                q_mode=q_mode,
                method_id=method_id,
                mode="DYNAMIC_REFERENCE",
                side="reference",
                request=request,
                margin_ratio=0.0,
            )
            result[f"{q_mode}::{method_id}"] = record
    return result


def _init_dynamic_worker(
    case_path: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    participant_ids: Sequence[str],
) -> None:
    _init_worker(case_path, policy, source, margins)
    _DYNAMIC_WORKER.clear()
    _DYNAMIC_WORKER.update({
        "reference_records": copy.deepcopy(dict(reference_records)),
        "participant_ids": list(participant_ids),
    })


def _raw_request(
    scenario: Mapping[str, Any], participant_index: Mapping[str, int], gamma: float,
) -> list[float]:
    values = [float(value) for value in scenario["reference_request_mw"]]
    for participant_id in scenario["coalition_ids"]:
        values[participant_index[participant_id]] *= float(gamma)
    return values


def _capacity_limited_request(
    raw_request: Sequence[float], capacity: Sequence[float],
) -> list[float]:
    if len(raw_request) != len(capacity):
        raise ValueError("request and capacity vectors must have equal length")
    return [min(float(request), float(limit)) for request, limit in zip(raw_request, capacity)]


def _credibility_capped_request(
    capacity_limited_request: Sequence[float], capacity: Sequence[float],
    scores: Sequence[float],
) -> tuple[list[float], list[float]]:
    if not (
        len(capacity_limited_request) == len(capacity) == len(scores)
    ):
        raise ValueError("request, capacity, and score vectors must have equal length")
    boundary = [float(score) * float(limit) for score, limit in zip(scores, capacity)]
    effective = [
        min(float(request), float(limit))
        for request, limit in zip(capacity_limited_request, boundary)
    ]
    return effective, boundary


def _initial_scores(
    participant_ids: Sequence[str], coalition_ids: Sequence[str], policy: Mapping[str, Any],
) -> list[float]:
    coalition = set(coalition_ids)
    if policy["policy_type"] == "FIXED_SCORE":
        return [float(policy["initial_score"]) if item in coalition else 1.0 for item in participant_ids]
    return [1.0 for _ in participant_ids]


def _delivery_ratio(
    allocation: Sequence[float], reference_request: Sequence[float], *, tolerance: float,
) -> tuple[list[float], list[float]]:
    delivery = [min(float(x), float(cap)) for x, cap in zip(allocation, reference_request)]
    ratio = [
        1.0 if float(x) <= tolerance else min(1.0, max(0.0, float(y) / float(x)))
        for x, y in zip(allocation, delivery)
    ]
    return delivery, ratio


def _delivery_outcome_metrics(
    reference_allocation: Sequence[float],
    reported_allocation: Sequence[float],
    capability: Sequence[float],
    coalition_indices: Sequence[int],
) -> dict[str, float]:
    """Measure realized delivery and idle authorization for one paired allocation.

    Request, allocation, and delivery remain distinct.  The capability vector is
    fixed across the pair, and delivery is recomputed from each allocation rather
    than inferred from the submitted request.
    """

    if not (
        len(reference_allocation) == len(reported_allocation) == len(capability)
    ):
        raise ValueError("allocation and capability vectors must have equal length")
    reference_delivery = [
        min(float(x), float(cap))
        for x, cap in zip(reference_allocation, capability)
    ]
    reported_delivery = [
        min(float(x), float(cap))
        for x, cap in zip(reported_allocation, capability)
    ]
    coalition = set(int(index) for index in coalition_indices)
    outside = [index for index in range(len(capability)) if index not in coalition]
    reference_total = math.fsum(reference_delivery)
    reported_total = math.fsum(reported_delivery)
    reported_authorized = math.fsum(float(value) for value in reported_allocation)
    reference_idle = math.fsum(
        max(float(x) - float(y), 0.0)
        for x, y in zip(reference_allocation, reference_delivery)
    )
    reported_idle = math.fsum(
        max(float(x) - float(y), 0.0)
        for x, y in zip(reported_allocation, reported_delivery)
    )
    coalition_change = math.fsum(
        reported_delivery[index] - reference_delivery[index]
        for index in coalition
    )
    outside_loss = math.fsum(
        reference_delivery[index] - reported_delivery[index]
        for index in outside
    )
    return {
        "reference_total_delivery_mw": reference_total,
        "reported_total_delivery_mw": reported_total,
        "total_delivery_change_mw": reported_total - reference_total,
        "system_delivery_loss_mw": max(reference_total - reported_total, 0.0),
        "reference_total_idle_award_mw": reference_idle,
        "reported_total_idle_award_mw": reported_idle,
        "total_idle_award_change_mw": reported_idle - reference_idle,
        "coalition_net_delivery_change_mw": coalition_change,
        "noncoalition_net_delivery_loss_mw": outside_loss,
        "delivery_utilization_ratio": (
            reported_total / reported_authorized
            if reported_authorized > AUTHORIZATION_TOLERANCE_MW else 1.0
        ),
    }


def _feedback_pair_record(
    reference: Mapping[str, Any], reported: Mapping[str, Any],
    scenario: Mapping[str, Any], participant_ids: Sequence[str],
) -> dict[str, Any]:
    """Return one metric shape for single- and multi-reporter cases."""

    if len(scenario["coalition_ids"]) >= 2:
        return _pair_record(reference, reported, scenario, participant_ids)
    pair = _single_pair_record(reference, reported, scenario, participant_ids)
    metrics = pair.get("access_metrics", {})
    pair["coalition_metrics"] = {
        "coalition_gain_mw": metrics.get("sg_mw"),
        "coalition_loss_mw": 0.0,
        "coalition_net_change_mw": metrics.get("sg_mw"),
        "coalition_undeliverable_mw": metrics.get("ud_mw"),
        "noncoalition_gain_mw": 0.0,
        "noncoalition_loss_mw": metrics.get("ol_minus_reporter_mw"),
        "noncoalition_net_change_mw": (
            -float(metrics["ol_minus_reporter_mw"])
            if metrics.get("ol_minus_reporter_mw") is not None else None
        ),
        "redistribution_mw": metrics.get("redistribution_mw"),
        "status": metrics.get("status"),
    }
    return pair


def _updated_scores(
    scores: Sequence[float], ratios: Sequence[float], policy: Mapping[str, Any],
    coalition_ids: Sequence[str], participant_ids: Sequence[str],
) -> list[float]:
    coalition = set(coalition_ids)
    if policy["policy_type"] == "FIXED_SCORE":
        fixed = float(policy["initial_score"])
        return [fixed if item in coalition else 1.0 for item in participant_ids]
    beta = float(policy["beta"])
    return [
        (1.0 - beta) * float(score) + beta * float(ratio)
        for score, ratio in zip(scores, ratios)
    ]


def _trajectory_task(task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]) -> dict[str, Any]:
    scenario, schedule, policy, q_mode, method_id = task
    participant_ids = list(_DYNAMIC_WORKER["participant_ids"])
    participant_index = {item: index for index, item in enumerate(participant_ids)}
    coalition_indices = [participant_index[item] for item in scenario["coalition_ids"]]
    reference = _DYNAMIC_WORKER["reference_records"][f"{q_mode}::{method_id}"]
    scores = _initial_scores(participant_ids, scenario["coalition_ids"], policy)
    solve_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []

    for interval, gamma in enumerate(schedule["report_multiplier_by_interval"], start=1):
        raw = _raw_request(scenario, participant_index, float(gamma))
        capacity_limited = _capacity_limited_request(raw, scenario["capacity_mw"])
        filtered, credibility_boundary = _credibility_capped_request(
            capacity_limited, scenario["capacity_mw"], scores,
        )
        key = tuple(filtered)
        step_scenario = copy.deepcopy(dict(scenario))
        step_scenario["reported_request_mw"] = raw
        if key not in solve_cache:
            # The conic stage-two certificate can occasionally return an
            # inaccurate status under heavy process-level concurrency.  Repeat
            # the identical mathematical problem before declaring the state
            # invalid; no parameter, tolerance, request, or constraint changes.
            solve_attempts: list[dict[str, Any]] = []
            for attempt in range(1, 4):
                candidate = _run_side_with_request(
                    step_scenario,
                    q_mode=q_mode,
                    method_id=method_id,
                    mode=policy["policy_id"],
                    side="reported",
                    request=filtered,
                    margin_ratio=0.0,
                )
                solve_attempts.append({
                    "attempt": attempt,
                    "solver_valid": bool(candidate.get("solver_valid")),
                    "solver_status": candidate.get("solver", {}).get("solver_status"),
                    "failure_reason": candidate.get("solver", {}).get("failure_reason"),
                    "method_execution_hash": candidate.get("method_execution_hash"),
                })
                if candidate.get("solver_valid"):
                    break
            candidate["identical_problem_solve_attempts"] = solve_attempts
            solve_cache[key] = candidate
        reported = copy.deepcopy(solve_cache[key])
        reported["raw_request_mw"] = raw
        reported["capacity_limited_request_mw"] = capacity_limited
        reported["transformed_request_mw"] = filtered
        reported["transformed_request_hash"] = canonical_hash({
            "participant_ids": participant_ids,
            "values_mw": filtered,
        })
        pair = _feedback_pair_record(reference, reported, step_scenario, participant_ids)
        allocation = [float(value) for value in reported.get("allocation_mw", [])]
        if len(allocation) != len(participant_ids) or not pair.get("metrics_valid"):
            delivery = []
            ratios = []
            score_after = list(scores)
            delivery_outcomes = {
                "reference_total_delivery_mw": None,
                "reported_total_delivery_mw": None,
                "total_delivery_change_mw": None,
                "system_delivery_loss_mw": None,
                "reference_total_idle_award_mw": None,
                "reported_total_idle_award_mw": None,
                "total_idle_award_change_mw": None,
                "coalition_net_delivery_change_mw": None,
                "noncoalition_net_delivery_loss_mw": None,
                "delivery_utilization_ratio": None,
            }
        else:
            delivery, ratios = _delivery_ratio(
                allocation,
                scenario["reference_request_mw"],
                tolerance=AUTHORIZATION_TOLERANCE_MW,
            )
            delivery_outcomes = _delivery_outcome_metrics(
                reference["allocation_mw"],
                allocation,
                scenario["reference_request_mw"],
                coalition_indices,
            )
            score_after = _updated_scores(
                scores, ratios, policy, scenario["coalition_ids"], participant_ids,
            )
        coalition_metrics = pair.get("coalition_metrics", {})
        jain_reference = pair.get("jain_norm_reference", {})
        jain_reported = pair.get("jain_norm_reported", {})
        step = {
            "trajectory_id": None,
            "interval": interval,
            "gamma": float(gamma),
            "mismatch_active": bool(abs(float(gamma) - 1.0) > 1.0e-12),
            "score_before": [float(value) for value in scores],
            "score_after": [float(value) for value in score_after],
            "coalition_score_before_mean": _mean([scores[index] for index in coalition_indices]),
            "coalition_score_after_mean": _mean([score_after[index] for index in coalition_indices]),
            "coalition_score_before_min": min(scores[index] for index in coalition_indices),
            "coalition_score_after_min": min(score_after[index] for index in coalition_indices),
            "raw_request_mw": raw,
            "capacity_limited_request_mw": capacity_limited,
            "credibility_boundary_mw": credibility_boundary,
            "filtered_request_mw": filtered,
            "request_attenuation_mw": math.fsum(capacity_limited) - math.fsum(filtered),
            "submitted_excess_above_capacity_mw": math.fsum(raw) - math.fsum(capacity_limited),
            "allocation_mw": allocation,
            "allocation_hash": reported.get("allocation_hash"),
            "delivery_mw": delivery,
            "delivery_ratio": ratios,
            **delivery_outcomes,
            "mean_coalition_delivery_ratio": _mean([ratios[index] for index in coalition_indices]) if ratios else None,
            "reference_total_export_mw": pair.get("reference_total_export_mw"),
            "reported_total_export_mw": pair.get("reported_total_export_mw"),
            "total_export_change_mw": pair.get("total_export_change_mw"),
            "coalition_gain_mw": coalition_metrics.get("coalition_gain_mw"),
            "coalition_undeliverable_mw": coalition_metrics.get("coalition_undeliverable_mw"),
            "noncoalition_loss_mw": coalition_metrics.get("noncoalition_loss_mw"),
            "redistribution_mw": coalition_metrics.get("redistribution_mw"),
            "jain_norm_change": (
                float(jain_reported["value"]) - float(jain_reference["value"])
                if jain_reported.get("status") == "DEFINED" and jain_reference.get("status") == "DEFINED"
                else None
            ),
            "solver_valid": bool(reported.get("solver_valid")),
            "ac_numerical_valid": bool(reported.get("ac_numerical_valid")),
            "matched_q_pair_pass": bool(pair.get("matched_q_pair_pass")),
            "maximum_loading_ratio": reported.get("screen", {}).get("maximum_loading_ratio") if reported.get("screen") else None,
            "minimum_voltage_pu": reported.get("screen", {}).get("minimum_voltage_pu") if reported.get("screen") else None,
            "max_mva_excess": reported.get("screen", {}).get("max_mva_excess") if reported.get("screen") else None,
            "method_execution_hash": reported.get("method_execution_hash"),
            "ac_result_hash": reported.get("ac_result_hash"),
            "identical_problem_solve_attempt_count": len(reported.get("identical_problem_solve_attempts", [])),
            "identical_problem_solve_attempts": reported.get("identical_problem_solve_attempts", []),
        }
        steps.append(step)
        scores = score_after

    trajectory_id = "::".join([
        scenario["scenario_id"], schedule["schedule_id"], policy["policy_id"], q_mode, method_id,
    ])
    for step in steps:
        step["trajectory_id"] = trajectory_id
    valid_steps = [step for step in steps if step["solver_valid"] and step["ac_numerical_valid"]]
    def values(key: str) -> list[float]:
        return [float(step[key]) for step in valid_steps if step.get(key) is not None]
    changed_steps = [step for step in valid_steps if step["mismatch_active"]]
    recovery_steps = [step for step in valid_steps if not step["mismatch_active"]]
    summary = {
        "trajectory_id": trajectory_id,
        "scenario_id": scenario["scenario_id"],
        "coalition_size": len(scenario["coalition_ids"]),
        "spatial_pattern": scenario["spatial_pattern"],
        "schedule_id": schedule["schedule_id"],
        "policy_id": policy["policy_id"],
        "policy_type": policy["policy_type"],
        "beta": policy["beta"],
        "q_mode": q_mode,
        "method_id": method_id,
        "interval_count": len(steps),
        "valid_interval_count": len(valid_steps),
        "matched_q_pair_pass_count": sum(bool(step["matched_q_pair_pass"]) for step in steps),
        "mean_redistribution_mw": _mean(values("redistribution_mw")),
        "mean_coalition_undeliverable_mw": _mean(values("coalition_undeliverable_mw")),
        "mean_coalition_gain_mw": _mean(values("coalition_gain_mw")),
        "mean_noncoalition_loss_mw": _mean(values("noncoalition_loss_mw")),
        "mean_reported_total_export_mw": _mean(values("reported_total_export_mw")),
        "mean_total_export_change_mw": _mean(values("total_export_change_mw")),
        "mean_reference_total_delivery_mw": _mean(values("reference_total_delivery_mw")),
        "mean_reported_total_delivery_mw": _mean(values("reported_total_delivery_mw")),
        "mean_total_delivery_change_mw": _mean(values("total_delivery_change_mw")),
        "mean_system_delivery_loss_mw": _mean(values("system_delivery_loss_mw")),
        "mean_reported_total_idle_award_mw": _mean(values("reported_total_idle_award_mw")),
        "mean_total_idle_award_change_mw": _mean(values("total_idle_award_change_mw")),
        "mean_coalition_net_delivery_change_mw": _mean(values("coalition_net_delivery_change_mw")),
        "mean_noncoalition_net_delivery_loss_mw": _mean(values("noncoalition_net_delivery_loss_mw")),
        "mean_delivery_utilization_ratio": _mean(values("delivery_utilization_ratio")),
        "mean_request_attenuation_mw": _mean(values("request_attenuation_mw")),
        "mean_jain_norm_change": _mean(values("jain_norm_change")),
        "mean_changed_interval_redistribution_mw": _mean([
            float(step["redistribution_mw"]) for step in changed_steps if step.get("redistribution_mw") is not None
        ]),
        "mean_recovery_interval_redistribution_mw": _mean([
            float(step["redistribution_mw"]) for step in recovery_steps if step.get("redistribution_mw") is not None
        ]),
        "final_coalition_score_mean": _mean([scores[index] for index in coalition_indices]),
        "minimum_coalition_score": min(
            step["coalition_score_after_min"] for step in steps
        ),
        "maximum_loading_ratio": max(values("maximum_loading_ratio"), default=0.0),
        "minimum_voltage_pu": min(values("minimum_voltage_pu"), default=None),
        "maximum_mva_excess": max(values("max_mva_excess"), default=0.0),
        "unique_solve_count": len(solve_cache),
        "steps": steps,
    }
    # The comparison deltas are attached only after every no-update trajectory
    # is available.  A provisional null keeps the serialized shape stable; the
    # final self-hash is computed in ``_attach_no_update_deltas`` after those
    # fields have been added.
    summary["trajectory_hash"] = None
    return summary


def _attach_no_update_deltas(trajectories: list[dict[str, Any]]) -> None:
    baseline = {
        (row["scenario_id"], row["schedule_id"], row["q_mode"], row["method_id"]): row
        for row in trajectories if row["policy_id"] == "DYNAMIC_BETA_000"
    }
    fields = (
        "mean_redistribution_mw",
        "mean_coalition_undeliverable_mw",
        "mean_coalition_gain_mw",
        "mean_noncoalition_loss_mw",
        "mean_reported_total_export_mw",
        "mean_reported_total_delivery_mw",
        "mean_reported_total_idle_award_mw",
        "mean_coalition_net_delivery_change_mw",
        "mean_noncoalition_net_delivery_loss_mw",
        "mean_request_attenuation_mw",
        "mean_jain_norm_change",
    )
    for row in trajectories:
        key = (row["scenario_id"], row["schedule_id"], row["q_mode"], row["method_id"])
        base = baseline[key]
        for field in fields:
            row[f"{field}_change_vs_no_update"] = float(row[field]) - float(base[field])
        row["redistribution_reduction_vs_no_update_mw"] = (
            float(base["mean_redistribution_mw"]) - float(row["mean_redistribution_mw"])
        )
        row["undeliverable_reduction_vs_no_update_mw"] = (
            float(base["mean_coalition_undeliverable_mw"])
            - float(row["mean_coalition_undeliverable_mw"])
        )
        row["export_recovery_vs_no_update_mw"] = (
            float(row["mean_reported_total_export_mw"])
            - float(base["mean_reported_total_export_mw"])
        )
        row["delivery_recovery_vs_no_update_mw"] = (
            float(row["mean_reported_total_delivery_mw"])
            - float(base["mean_reported_total_delivery_mw"])
        )
        row["idle_award_reduction_vs_no_update_mw"] = (
            float(base["mean_reported_total_idle_award_mw"])
            - float(row["mean_reported_total_idle_award_mw"])
        )
        row["trajectory_hash"] = None
        row["trajectory_hash"] = canonical_hash(row)


def _aggregate(
    rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = (
        "mean_redistribution_mw",
        "mean_coalition_undeliverable_mw",
        "mean_coalition_gain_mw",
        "mean_noncoalition_loss_mw",
        "mean_reported_total_export_mw",
        "mean_total_export_change_mw",
        "mean_reference_total_delivery_mw",
        "mean_reported_total_delivery_mw",
        "mean_total_delivery_change_mw",
        "mean_system_delivery_loss_mw",
        "mean_reported_total_idle_award_mw",
        "mean_total_idle_award_change_mw",
        "mean_coalition_net_delivery_change_mw",
        "mean_noncoalition_net_delivery_loss_mw",
        "mean_delivery_utilization_ratio",
        "mean_request_attenuation_mw",
        "mean_jain_norm_change",
        "final_coalition_score_mean",
        "minimum_coalition_score",
        "redistribution_reduction_vs_no_update_mw",
        "undeliverable_reduction_vs_no_update_mw",
        "export_recovery_vs_no_update_mw",
        "delivery_recovery_vs_no_update_mw",
        "idle_award_reduction_vs_no_update_mw",
    )
    result = dict(dimensions)
    result["trajectory_count"] = len(rows)
    result["interval_count"] = sum(int(row["interval_count"]) for row in rows)
    result["matched_q_pair_pass_rate"] = (
        sum(int(row["matched_q_pair_pass_count"]) for row in rows) / result["interval_count"]
        if result["interval_count"] else 0.0
    )
    for field in numeric:
        result[field] = _mean([float(row[field]) for row in rows])
    result["maximum_loading_ratio"] = max(float(row["maximum_loading_ratio"]) for row in rows)
    result["minimum_voltage_pu"] = min(float(row["minimum_voltage_pu"]) for row in rows)
    result["maximum_mva_excess"] = max(float(row["maximum_mva_excess"]) for row in rows)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *, case_path: Path, policy_path: Path, source_path: Path, execution_path: Path,
    design_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = _load_yaml(policy_path)
    source = _load_json(source_path)
    execution = _load_json(execution_path)
    design = _load_json(design_path)
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    execution_hash = _validated_hash(execution, field="result_hash", label="method execution")
    design_hash = _validated_hash(design, field="design_hash", label="dynamic design")
    if design.get("status") != "DYNAMIC_FEEDBACK_DESIGN_FROZEN":
        raise ValueError("dynamic design is not frozen")
    if design.get("source_envelope_hash") != source_hash or execution.get("source_envelope_hash") != source_hash:
        raise ValueError("source binding mismatch")
    if design.get("policy_hash") != policy.get("policy_hash"):
        raise ValueError("RATE policy binding mismatch")
    if tuple(design.get("method_roster", ())) != tuple(METHODS):
        raise ValueError("dynamic method roster mismatch")

    source_scenario_id = design["scenarios"][0]["source_scenario_id"]
    margins = _extract_margins(execution, source, source_scenario_id)
    reference_records = _build_reference_records(
        case_path=case_path, policy=policy, source=source, margins=margins, design=design,
    )
    if not all(row.get("solver_valid") and row.get("ac_numerical_valid") for row in reference_records.values()):
        raise ValueError("dynamic reference allocation/AC construction failed")

    schedule_by_id = {row["schedule_id"]: row for row in design["schedules"]}
    policy_rows = list(design["feedback_policies"])
    tasks: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]] = []
    for scenario in design["scenarios"]:
        schedule_ids = ["PERSISTENT_MISMATCH"]
        if scenario["scenario_id"] in design["profile_robustness_scenario_ids"]:
            schedule_ids.extend(["INTERMITTENT_MISMATCH", "ONSET_AND_RECOVERY"])
        for schedule_id in schedule_ids:
            for feedback_policy in policy_rows:
                for q_mode in design["q_roster"]:
                    for method_id in design["method_roster"]:
                        tasks.append((scenario, schedule_by_id[schedule_id], feedback_policy, q_mode, method_id))

    workers = max(1, int(workers))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_dynamic_worker,
            initargs=(
                str(case_path), policy, source, margins, reference_records,
                list(source["participant_ids"]),
            ),
        ) as executor:
            trajectories = list(executor.map(
                _trajectory_task,
                tasks,
                chunksize=max(1, len(tasks) // (workers * 6)),
            ))
    else:
        _init_dynamic_worker(
            str(case_path), policy, source, margins, reference_records,
            list(source["participant_ids"]),
        )
        trajectories = [_trajectory_task(task) for task in tasks]
    _attach_no_update_deltas(trajectories)

    by_schedule_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_persistent_scenario_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_schedule_policy[(row["schedule_id"], row["policy_id"])].append(row)
        if row["schedule_id"] == "PERSISTENT_MISMATCH":
            by_persistent_scenario_policy[(row["scenario_id"], row["policy_id"])].append(row)
    aggregate_summary = [
        _aggregate(rows, {"schedule_id": schedule_id, "policy_id": policy_id})
        for (schedule_id, policy_id), rows in sorted(by_schedule_policy.items())
    ]
    persistent_scenario_summary = [
        _aggregate(rows, {"scenario_id": scenario_id, "policy_id": policy_id})
        for (scenario_id, policy_id), rows in sorted(by_persistent_scenario_policy.items())
    ]

    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    beta_zero_rows = [row for row in trajectories if row["policy_id"] == "DYNAMIC_BETA_000"]
    fixed_rows = [row for row in trajectories if row["policy_id"] == "FIXED_SCORE_070"]
    gates = {
        "SOURCE_RATE_EXECUTION_BINDING": bool(
            design["source_envelope_hash"] == source_hash
            and design["policy_hash"] == policy["policy_hash"]
            and execution["source_envelope_hash"] == source_hash
        ),
        "DESIGN_FROZEN_BEFORE_EXECUTION": design["status"] == "DYNAMIC_FEEDBACK_DESIGN_FROZEN",
        "FULL_FACTORIAL_COMPLETENESS": len(trajectories) == len(tasks),
        "SOLVER_COMPLETENESS": all(step["solver_valid"] for step in steps),
        "AC_NUMERICAL_COMPLETENESS": all(step["ac_numerical_valid"] for step in steps),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(step["matched_q_pair_pass"] for step in steps),
        "BETA_ZERO_NO_UPDATE_INVARIANT": all(
            abs(float(step["coalition_score_after_mean"]) - 1.0) <= 1.0e-12
            for row in beta_zero_rows for step in row["steps"]
        ),
        "FIXED_SCORE_070_INVARIANT": all(
            abs(float(step["coalition_score_after_mean"]) - FIXED_SCORE) <= 1.0e-12
            for row in fixed_rows for step in row["steps"]
        ),
        "CREDIBILITY_CAP_NONBYPASS": all(
            all(
                float(effective) <= float(base) + 2.0e-12
                and float(effective) <= float(boundary) + 2.0e-12
                for effective, base, boundary in zip(
                    step["filtered_request_mw"],
                    step["capacity_limited_request_mw"],
                    step["credibility_boundary_mw"],
                )
            )
            for step in steps
        ),
        "DELIVERY_ACCOUNTING_CLOSED": all(
            abs(math.fsum(float(value) for value in step["delivery_mw"])
                - float(step["reported_total_delivery_mw"])) <= 2.0e-12
            and abs(
                math.fsum(
                    max(float(x) - float(y), 0.0)
                    for x, y in zip(step["allocation_mw"], step["delivery_mw"])
                ) - float(step["reported_total_idle_award_mw"])
            ) <= 2.0e-12
            and abs(
                float(step["reported_total_delivery_mw"])
                + float(step["reported_total_idle_award_mw"])
                - float(step["reported_total_export_mw"])
            ) <= 2.0e-9
            for step in steps
        ),
        "REFERENCE_ALLOCATION_FULLY_DELIVERABLE": all(
            abs(float(step["reference_total_delivery_mw"])
                - float(step["reference_total_export_mw"])) <= 2.0e-9
            for step in steps
        ),
        "FULL_DELIVERY_STATE_HAS_NO_FALSE_ATTENUATION": all(
            abs(float(step["request_attenuation_mw"])) <= 2.0e-12
            for step in steps
            if all(abs(float(value) - 1.0) <= 1.0e-12 for value in step["score_before"])
        ),
        "ONE_INTERVAL_CAUSAL_STATE_TRANSITION": all(
            all(
                abs(float(current) - float(previous)) <= 2.0e-12
                for current, previous in zip(
                    row["steps"][index]["score_before"],
                    row["steps"][index - 1]["score_after"],
                )
            )
            for row in trajectories
            for index in range(1, len(row["steps"]))
        ),
        "RESULT_FEEDBACK_EXCLUDED": all(value is False for value in design["input_read_assertions"].values()),
    }
    result: dict[str, Any] = {
        "serialization_id": RESULT_SERIALIZATION_ID,
        "status": "DYNAMIC_FEEDBACK_EXPERIMENT_COMPLETE" if all(gates.values()) else "DYNAMIC_FEEDBACK_EXPERIMENT_BLOCKED",
        "scientific_role": "MULTI_INTERVAL_METERED_DELIVERY_FEEDBACK_EVIDENCE",
        "design_hash": design_hash,
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "worker_count": workers,
        "scenario_count": len(design["scenarios"]),
        "trajectory_count": len(trajectories),
        "interval_record_count": len(steps),
        "method_roster": list(design["method_roster"]),
        "q_roster": list(design["q_roster"]),
        "feedback_policies": policy_rows,
        "schedules": list(design["schedules"]),
        "reference_record_hashes": {
            key: canonical_hash(value) for key, value in sorted(reference_records.items())
        },
        "gate_results": gates,
        "aggregate_summary": aggregate_summary,
        "persistent_scenario_summary": persistent_scenario_summary,
        "trajectory_summaries": trajectories,
        "input_read_assertions": dict(design["input_read_assertions"]),
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")

    trajectory_fields = [
        "trajectory_id", "scenario_id", "coalition_size", "spatial_pattern", "schedule_id",
        "policy_id", "policy_type", "beta", "q_mode", "method_id", "interval_count",
        "valid_interval_count", "matched_q_pair_pass_count", "mean_redistribution_mw",
        "mean_coalition_undeliverable_mw", "mean_coalition_gain_mw", "mean_noncoalition_loss_mw",
        "mean_reported_total_export_mw", "mean_total_export_change_mw",
        "mean_reference_total_delivery_mw", "mean_reported_total_delivery_mw",
        "mean_total_delivery_change_mw", "mean_system_delivery_loss_mw",
        "mean_reported_total_idle_award_mw", "mean_total_idle_award_change_mw",
        "mean_coalition_net_delivery_change_mw", "mean_noncoalition_net_delivery_loss_mw",
        "mean_delivery_utilization_ratio", "mean_request_attenuation_mw",
        "mean_jain_norm_change", "final_coalition_score_mean", "minimum_coalition_score",
        "redistribution_reduction_vs_no_update_mw", "undeliverable_reduction_vs_no_update_mw",
        "export_recovery_vs_no_update_mw", "delivery_recovery_vs_no_update_mw",
        "idle_award_reduction_vs_no_update_mw", "maximum_loading_ratio", "minimum_voltage_pu",
        "maximum_mva_excess", "unique_solve_count", "trajectory_hash",
    ]
    step_fields = [
        "trajectory_id", "interval", "gamma", "mismatch_active", "coalition_score_before_mean",
        "coalition_score_after_mean", "coalition_score_before_min", "coalition_score_after_min",
        "request_attenuation_mw", "reference_total_export_mw", "reported_total_export_mw",
        "total_export_change_mw", "reference_total_delivery_mw", "reported_total_delivery_mw",
        "total_delivery_change_mw", "system_delivery_loss_mw",
        "reported_total_idle_award_mw", "total_idle_award_change_mw",
        "coalition_net_delivery_change_mw", "noncoalition_net_delivery_loss_mw",
        "delivery_utilization_ratio", "coalition_gain_mw", "coalition_undeliverable_mw",
        "noncoalition_loss_mw", "redistribution_mw", "jain_norm_change", "solver_valid",
        "ac_numerical_valid", "matched_q_pair_pass", "maximum_loading_ratio", "minimum_voltage_pu",
        "max_mva_excess", "allocation_hash", "method_execution_hash", "ac_result_hash",
    ]
    aggregate_fields = list(aggregate_summary[0].keys()) if aggregate_summary else []
    _write_csv(output_path.with_name("DYNAMIC_FEEDBACK_TRAJECTORY_SUMMARY.csv"), trajectories, trajectory_fields)
    _write_csv(output_path.with_name("DYNAMIC_FEEDBACK_INTERVAL_DATA.csv"), steps, step_fields)
    _write_csv(output_path.with_name("DYNAMIC_FEEDBACK_AGGREGATE_SUMMARY.csv"), aggregate_summary, aggregate_fields)
    _write_csv(
        output_path.with_name("DYNAMIC_FEEDBACK_PERSISTENT_SCENARIO_SUMMARY.csv"),
        persistent_scenario_summary,
        list(persistent_scenario_summary[0].keys()) if persistent_scenario_summary else [],
    )
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Multi-interval metered-delivery feedback experiment",
            "",
            f"- Status: `{result['status']}`",
            f"- Trajectories: `{len(trajectories)}`; interval records: `{len(steps)}`.",
            f"- Methods: `{list(design['method_roster'])}`; power-factor settings: `{list(design['q_roster'])}`.",
            f"- Gates: `{gates}`",
            "- Beta is treated as a sensitivity parameter; no beta value is estimated from these outcomes.",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "fresh_source_design"
    out = root / "outputs" / "dynamic_feedback_v12_3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-design", action="store_true")
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument("--policy", type=Path, default=root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml")
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument("--multi-manifest", type=Path, default=root / "outputs" / "multi_reporter_v12_3" / "MULTI_REPORTER_SCENARIO_MANIFEST.json")
    parser.add_argument("--design", type=Path, default=out / "DYNAMIC_FEEDBACK_DESIGN_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=out / "DYNAMIC_FEEDBACK_RESULTS.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    if args.freeze_design:
        design = freeze_design(
            source_path=args.source,
            multi_manifest_path=args.multi_manifest,
            policy_path=args.policy,
            output_path=args.design,
        )
        print(canonical_dumps({
            "status": design["status"],
            "scenario_count": len(design["scenarios"]),
            "design_hash": design["design_hash"],
        }))
        return 0
    result = run_experiment(
        case_path=args.case,
        policy_path=args.policy,
        source_path=args.source,
        execution_path=args.execution,
        design_path=args.design,
        output_path=args.output,
        workers=max(1, args.workers),
    )
    print(canonical_dumps({
        "status": result["status"],
        "trajectory_count": result["trajectory_count"],
        "interval_record_count": result["interval_record_count"],
        "gates": result["gate_results"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BETA_GRID",
    "FIXED_SCORE",
    "HORIZON",
    "SCHEDULES",
    "_capacity_limited_request",
    "_credibility_capped_request",
    "_delivery_outcome_metrics",
    "freeze_design",
    "run_experiment",
]
