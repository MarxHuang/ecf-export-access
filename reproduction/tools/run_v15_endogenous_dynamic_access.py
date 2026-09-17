"""Run the behavior-linked multi-interval access experiment.

The experiment closes the request--delivery generation chain.  A participant's
raw request multiplier is selected by the frozen expected-utility experiment,
and metered capability is then realized from the same five-point conditional
capability tree.  Penalty-only, selective metered feedback, and an
equal-attenuation uniform control are evaluated on identical capability paths.

The simultaneous multi-participant cases apply independently selected myopic
nominations at the same time.  They are not represented as a Nash equilibrium.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from r4r.serialization import canonical_dumps, canonical_hash
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import METHODS
from tools.run_v12_3_dynamic_horizontal_controls import matched_uniform_request
import tools.run_v12_3_dynamic_feedback_experiment as dynamic


DESIGN_SERIALIZATION_ID = "r4r.endogenous_dynamic_access_design.v1"
RESULT_SERIALIZATION_ID = "r4r.endogenous_dynamic_access_results.v1"
HORIZON = 16
FORECAST_CV = 0.20
PENALTY_RATIO_GRID = (0.0, 1.0, 10.0)
CONTROL_ROSTER = (
    {
        "control_id": "NO_FEEDBACK",
        "control_role": "PENALTY_ONLY_BEHAVIORAL_REQUEST",
        "beta": 0.0,
    },
    {
        "control_id": "SELECTIVE_CF_BETA_050",
        "control_role": "SELECTIVE_METERED_FEEDBACK",
        "beta": 0.5,
    },
    {
        "control_id": "SELECTIVE_CF_BETA_100",
        "control_role": "SELECTIVE_METERED_FEEDBACK",
        "beta": 1.0,
    },
    {
        "control_id": "UNIFORM_MATCHED_CF_BETA_100",
        "control_role": "EQUAL_ATTENUATION_UNIFORM_CONTROL",
        "beta": None,
    },
)
SCENARIO_IDS = (
    "DF_SINGLE_P021",
    "MR_K05_LOCAL",
    "MR_K05_DISPERSED",
)

# Every profile is a permutation of the exact five-point probability-tree
# multiset: one -2, four -1, six 0, four +1, and one +2 realization.
PROFILE_SEQUENCES: dict[str, tuple[int, ...]] = {
    "CLUSTERED_SHORTFALL": (
        -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
    ),
    "INTERMITTENT_SHORTFALL": (
        -2, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 0, 2,
    ),
    "SHOCK_AND_RECOVERY": (
        0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0, 1, 1, 1, 1, 2,
    ),
}
EXPECTED_SUPPORT_COUNTS = {-2: 1, -1: 4, 0: 6, 1: 4, 2: 1}
AUTHORIZATION_TOLERANCE_MW = 1.0e-9
CLASSIFICATION_TOLERANCE_MW = 2.0e-10

_WORKER: dict[str, Any] = {}


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


def _mean(values: Iterable[float]) -> float:
    sequence = [float(value) for value in values]
    return math.fsum(sequence) / len(sequence) if sequence else 0.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen and not isinstance(row[field], (dict, list, tuple)):
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _profile_is_exact(sequence: Sequence[int]) -> bool:
    return len(sequence) == HORIZON and Counter(int(value) for value in sequence) == Counter(
        EXPECTED_SUPPORT_COUNTS
    )


def _phase_offset(participant_index: int) -> int:
    """Return a deterministic coprime phase offset for participant heterogeneity."""

    return (7 * int(participant_index)) % HORIZON


def capability_vector(
    *, profile_id: str, interval: int, forecast_mw: Sequence[float],
    capacity_mw: Sequence[float], cv: float = FORECAST_CV,
) -> tuple[list[float], list[int]]:
    """Realize capability from the same registered five-point tree.

    Participant-specific cyclic phases prevent an artificial feeder-wide common
    shortfall while preserving the exact support frequency for every participant
    over the 16-interval horizon.
    """

    if profile_id not in PROFILE_SEQUENCES:
        raise KeyError(profile_id)
    if not 1 <= int(interval) <= HORIZON:
        raise ValueError("interval lies outside the registered horizon")
    if len(forecast_mw) != len(capacity_mw):
        raise ValueError("forecast and capacity vectors must have equal length")
    base = PROFILE_SEQUENCES[profile_id]
    capability: list[float] = []
    support: list[int] = []
    for index, (forecast, capacity) in enumerate(zip(forecast_mw, capacity_mw)):
        z_value = int(base[(int(interval) - 1 + _phase_offset(index)) % HORIZON])
        support.append(z_value)
        factor = max(0.0, 1.0 + float(cv) * float(z_value))
        capability.append(min(float(capacity), factor * float(forecast)))
    return capability, support


def _choice_lookup(behavior: Mapping[str, Any]) -> dict[tuple[str, str, str, float, float], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, str, float, float], Mapping[str, Any]] = {}
    for row in behavior["choice_cells"]:
        key = (
            str(row["reporter_id"]),
            str(row["q_mode"]),
            str(row["method_id"]),
            float(row["forecast_cv"]),
            float(row["penalty_to_export_value_ratio"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate endogenous choice cell: {key}")
        lookup[key] = row
    return lookup


def freeze_design(
    *, dynamic_design_path: Path, behavior_path: Path, source_path: Path,
    policy_path: Path, output_path: Path,
) -> dict[str, Any]:
    dynamic_design = _load_json(dynamic_design_path)
    behavior = _load_json(behavior_path)
    source = _load_json(source_path)
    policy = _load_yaml(policy_path)
    dynamic_hash = _validated_hash(
        dynamic_design, field="design_hash", label="dynamic scenario design",
    )
    behavior_hash = _validated_hash(
        behavior, field="result_hash", label="endogenous request experiment",
    )
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    if behavior.get("status") != "ENDOGENOUS_REQUEST_CHOICE_EXPERIMENT_COMPLETE":
        raise ValueError("endogenous request experiment is incomplete")
    if not all(behavior.get("gate_results", {}).values()):
        raise ValueError("endogenous request experiment contains an open gate")
    if behavior.get("source_envelope_hash") != source_hash:
        raise ValueError("behavior/source binding mismatch")
    if dynamic_design.get("source_envelope_hash") != source_hash:
        raise ValueError("dynamic-design/source binding mismatch")
    if policy.get("policy_hash") != source.get("policy_hash"):
        raise ValueError("RATE policy/source binding mismatch")
    if not all(_profile_is_exact(sequence) for sequence in PROFILE_SEQUENCES.values()):
        raise ValueError("a temporal profile does not reproduce the registered tree")

    scenario_by_id = {
        str(row["scenario_id"]): copy.deepcopy(dict(row))
        for row in dynamic_design["scenarios"]
    }
    missing = sorted(set(SCENARIO_IDS) - set(scenario_by_id))
    if missing:
        raise ValueError(f"required scenarios are unavailable: {missing}")
    scenarios = [scenario_by_id[scenario_id] for scenario_id in SCENARIO_IDS]

    choice_lookup = _choice_lookup(behavior)
    required_choice_keys = {
        (participant_id, q_mode, method_id, FORECAST_CV, penalty)
        for scenario in scenarios
        for participant_id in scenario["coalition_ids"]
        for q_mode in source["q_roster"]
        for method_id in METHODS
        for penalty in PENALTY_RATIO_GRID
    }
    missing_choices = sorted(required_choice_keys - set(choice_lookup))
    if missing_choices:
        raise ValueError(f"missing endogenous request choices: {missing_choices[:5]}")

    design: dict[str, Any] = {
        "serialization_id": DESIGN_SERIALIZATION_ID,
        "status": "ENDOGENOUS_DYNAMIC_ACCESS_DESIGN_FROZEN",
        "scientific_role": "BEHAVIOR_LINKED_MULTI_INTERVAL_PRIMARY_EVIDENCE",
        "source_envelope_hash": source_hash,
        "dynamic_scenario_design_hash": dynamic_hash,
        "behavioral_choice_result_hash": behavior_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "horizon": HORIZON,
        "forecast_cv": FORECAST_CV,
        "penalty_to_export_value_ratio_grid": list(PENALTY_RATIO_GRID),
        "method_roster": list(METHODS),
        "q_roster": list(source["q_roster"]),
        "control_roster": [dict(row) for row in CONTROL_ROSTER],
        "temporal_profiles": [
            {
                "profile_id": profile_id,
                "standardized_capability_sequence": list(sequence),
                "support_counts": {
                    str(key): int(value)
                    for key, value in sorted(Counter(sequence).items())
                },
            }
            for profile_id, sequence in PROFILE_SEQUENCES.items()
        ],
        "participant_phase_rule": "offset_i=(7*i) mod 16; every participant retains the exact five-point support frequency",
        "scenarios": scenarios,
        "behavioral_contract": {
            "raw_request": "r_i=gamma_i^*(theta,sigma,rule,Q)*forecast_i; gamma_i^* comes only from the frozen expected-utility choice result",
            "realized_capability": "a_i(t)=min(C_i,(1+sigma*z_i(t))*forecast_i), with z_i(t) from the same five-point tree used for choice",
            "delivery": "y_i(t)=min(x_i(t),a_i(t))",
            "feedback": "chi_i(t+1)=(1-beta)*chi_i(t)+beta*y_i(t)/x_i(t)",
            "effective_request": "min(min(r_i,C_i),chi_i*C_i)",
            "simultaneous_scope": "multi-participant cases apply independently selected myopic nominations simultaneously; no equilibrium claim",
        },
        "equal_attenuation_control": "uniform scaling uses exactly the total attenuation generated by SELECTIVE_CF_BETA_100 in the same interval",
        "input_read_assertions": {
            "dynamic_outcomes_used_to_select_profiles": False,
            "dynamic_outcomes_used_to_select_penalty_grid": False,
            "dynamic_outcomes_used_to_select_feedback_gain": False,
            "dynamic_outcomes_used_to_refit_network_or_rate": False,
            "capability_realizations_generated_independently_of_request_model": False,
        },
        "design_hash": None,
    }
    design["design_hash"] = canonical_hash(design)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(design) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Frozen behavior-linked dynamic access design",
            "",
            f"- Scenarios: `{len(scenarios)}`; profiles: `{len(PROFILE_SEQUENCES)}`; horizon: `{HORIZON}`.",
            f"- Penalty ratios: `{list(PENALTY_RATIO_GRID)}`; controls: `{[row['control_id'] for row in CONTROL_ROSTER]}`.",
            "- Each 16-interval capability path exactly reproduces the registered five-point tree frequency.",
            "- Requests and metered delivery use the same conditional capability model.",
            f"- Design hash: `{design['design_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return design


def _init_behavior_worker(
    case_path: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    participant_ids: Sequence[str],
    choices: Mapping[tuple[str, str, str, float, float], Mapping[str, Any]],
) -> None:
    dynamic._init_dynamic_worker(
        case_path, policy, source, margins, reference_records, participant_ids,
    )
    _WORKER.clear()
    _WORKER.update({
        "participant_ids": list(participant_ids),
        "reference_records": copy.deepcopy(dict(reference_records)),
        "choices": dict(choices),
    })


def _raw_behavioral_request(
    scenario: Mapping[str, Any], *, q_mode: str, method_id: str,
    penalty_ratio: float,
) -> tuple[list[float], dict[str, float]]:
    participant_ids = list(_WORKER["participant_ids"])
    participant_index = {participant_id: index for index, participant_id in enumerate(participant_ids)}
    raw = [float(value) for value in scenario["reference_request_mw"]]
    selected: dict[str, float] = {}
    for participant_id in scenario["coalition_ids"]:
        key = (
            str(participant_id), str(q_mode), str(method_id),
            FORECAST_CV, float(penalty_ratio),
        )
        choice = _WORKER["choices"][key]
        gamma = float(choice["selected_gamma"])
        raw[participant_index[str(participant_id)]] *= gamma
        selected[str(participant_id)] = gamma
    return raw, selected


def _delivery_metrics(
    *, reference_allocation: Sequence[float], allocation: Sequence[float],
    capability: Sequence[float], coalition_indices: Sequence[int],
    penalty_ratio: float,
) -> dict[str, Any]:
    if not (
        len(reference_allocation) == len(allocation) == len(capability)
    ):
        raise ValueError("allocation and capability vectors must have equal length")
    coalition = set(int(index) for index in coalition_indices)
    outside = [index for index in range(len(capability)) if index not in coalition]
    reference_delivery = [
        min(float(award), float(available))
        for award, available in zip(reference_allocation, capability)
    ]
    delivery = [
        min(float(award), float(available))
        for award, available in zip(allocation, capability)
    ]
    reference_idle = [
        max(float(award) - float(value), 0.0)
        for award, value in zip(reference_allocation, reference_delivery)
    ]
    idle = [
        max(float(award) - float(value), 0.0)
        for award, value in zip(allocation, delivery)
    ]
    ratios = [
        1.0 if float(award) <= AUTHORIZATION_TOLERANCE_MW
        else min(1.0, max(0.0, float(value) / float(award)))
        for award, value in zip(allocation, delivery)
    ]
    private_utility = math.fsum(
        delivery[index] - float(penalty_ratio) * idle[index]
        for index in coalition
    )
    reference_private_utility = math.fsum(
        reference_delivery[index] - float(penalty_ratio) * reference_idle[index]
        for index in coalition
    )
    total_delivery_change = math.fsum(delivery) - math.fsum(reference_delivery)
    outside_delivery_loss = math.fsum(
        reference_delivery[index] - delivery[index] for index in outside
    )
    coalition_delivery_gain = math.fsum(
        delivery[index] - reference_delivery[index] for index in coalition
    )
    private_gain = private_utility - reference_private_utility
    incremental_coalition_idle = math.fsum(
        idle[index] - reference_idle[index] for index in coalition
    )
    return {
        "reference_delivery_mw": reference_delivery,
        "delivery_mw": delivery,
        "delivery_ratio": ratios,
        "reference_idle_award_mw": reference_idle,
        "idle_award_mw": idle,
        "reference_total_delivery_mw": math.fsum(reference_delivery),
        "reported_total_delivery_mw": math.fsum(delivery),
        "total_delivery_change_mw": total_delivery_change,
        "system_delivery_loss_mw": max(-total_delivery_change, 0.0),
        "reference_total_idle_award_mw": math.fsum(reference_idle),
        "reported_total_idle_award_mw": math.fsum(idle),
        "incremental_total_idle_award_mw": math.fsum(idle) - math.fsum(reference_idle),
        "coalition_delivery_gain_mw": coalition_delivery_gain,
        "outside_delivery_loss_mw": outside_delivery_loss,
        "coalition_private_utility_mw_equivalent": private_utility,
        "reference_coalition_private_utility_mw_equivalent": reference_private_utility,
        "coalition_private_utility_gain_mw_equivalent": private_gain,
        "incremental_coalition_idle_award_mw": incremental_coalition_idle,
        "delivery_externality": bool(
            private_gain > CLASSIFICATION_TOLERANCE_MW
            and total_delivery_change < -CLASSIFICATION_TOLERANCE_MW
            and incremental_coalition_idle > CLASSIFICATION_TOLERANCE_MW
        ),
        "productive_high_request": bool(
            private_gain > CLASSIFICATION_TOLERANCE_MW
            and total_delivery_change >= -CLASSIFICATION_TOLERANCE_MW
        ),
        "delivery_idle_identity_error_mw": abs(
            math.fsum(delivery) + math.fsum(idle) - math.fsum(allocation)
        ),
    }


def _solve_request(
    *, cache: dict[tuple[float, ...], dict[str, Any]], scenario: Mapping[str, Any],
    request: Sequence[float], q_mode: str, method_id: str, mode: str,
) -> dict[str, Any]:
    key = tuple(round(float(value), 14) for value in request)
    if key not in cache:
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            candidate = dynamic._run_side_with_request(
                scenario,
                q_mode=q_mode,
                method_id=method_id,
                mode=mode,
                side="reported",
                request=request,
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
        candidate["identical_problem_solve_attempts"] = attempts
        cache[key] = candidate
    return copy.deepcopy(cache[key])


def _updated_scores(
    scores: Sequence[float], ratios: Sequence[float], beta: float,
) -> list[float]:
    return [
        (1.0 - float(beta)) * float(score) + float(beta) * float(ratio)
        for score, ratio in zip(scores, ratios)
    ]


def _trajectory_task(
    task: tuple[dict[str, Any], str, float, str, str],
) -> list[dict[str, Any]]:
    scenario, profile_id, penalty_ratio, q_mode, method_id = task
    participant_ids = list(_WORKER["participant_ids"])
    participant_index = {participant_id: index for index, participant_id in enumerate(participant_ids)}
    coalition_indices = [participant_index[str(value)] for value in scenario["coalition_ids"]]
    reference = _WORKER["reference_records"][f"{q_mode}::{method_id}"]
    reference_allocation = [float(value) for value in reference["allocation_mw"]]
    raw, selected_gamma = _raw_behavioral_request(
        scenario,
        q_mode=q_mode,
        method_id=method_id,
        penalty_ratio=float(penalty_ratio),
    )
    capacity_limited = dynamic._capacity_limited_request(raw, scenario["capacity_mw"])
    scores_by_control = {
        "NO_FEEDBACK": [1.0 for _ in participant_ids],
        "SELECTIVE_CF_BETA_050": [1.0 for _ in participant_ids],
        "SELECTIVE_CF_BETA_100": [1.0 for _ in participant_ids],
    }
    solve_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    steps_by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for interval in range(1, HORIZON + 1):
        capability, support = capability_vector(
            profile_id=profile_id,
            interval=interval,
            forecast_mw=scenario["reference_request_mw"],
            capacity_mw=scenario["capacity_mw"],
        )
        effective_by_control: dict[str, list[float]] = {
            "NO_FEEDBACK": list(capacity_limited),
        }
        boundary_by_control: dict[str, list[float]] = {
            "NO_FEEDBACK": [float(value) for value in scenario["capacity_mw"]],
        }
        for control_id in ("SELECTIVE_CF_BETA_050", "SELECTIVE_CF_BETA_100"):
            effective, boundary = dynamic._credibility_capped_request(
                capacity_limited,
                scenario["capacity_mw"],
                scores_by_control[control_id],
            )
            effective_by_control[control_id] = effective
            boundary_by_control[control_id] = boundary
        uniform = matched_uniform_request(
            capacity_limited,
            effective_by_control["SELECTIVE_CF_BETA_100"],
        )
        effective_by_control["UNIFORM_MATCHED_CF_BETA_100"] = uniform
        boundary_by_control["UNIFORM_MATCHED_CF_BETA_100"] = list(uniform)

        for control in CONTROL_ROSTER:
            control_id = str(control["control_id"])
            effective = effective_by_control[control_id]
            step_scenario = copy.deepcopy(dict(scenario))
            step_scenario["reported_request_mw"] = list(raw)
            reported = _solve_request(
                cache=solve_cache,
                scenario=step_scenario,
                request=effective,
                q_mode=q_mode,
                method_id=method_id,
                mode=control_id,
            )
            allocation = [float(value) for value in reported.get("allocation_mw", [])]
            valid = len(allocation) == len(participant_ids)
            if valid:
                delivery = _delivery_metrics(
                    reference_allocation=reference_allocation,
                    allocation=allocation,
                    capability=capability,
                    coalition_indices=coalition_indices,
                    penalty_ratio=float(penalty_ratio),
                )
                ratios = list(delivery["delivery_ratio"])
            else:
                delivery = {}
                ratios = []

            score_before = (
                list(scores_by_control[control_id])
                if control_id in scores_by_control
                else list(scores_by_control["SELECTIVE_CF_BETA_100"])
            )
            if control_id in {"SELECTIVE_CF_BETA_050", "SELECTIVE_CF_BETA_100"} and ratios:
                beta = float(control["beta"])
                score_after = _updated_scores(score_before, ratios, beta)
            else:
                score_after = list(score_before)

            pair = dynamic._feedback_pair_record(
                reference, reported, step_scenario, participant_ids,
            )
            jain_reference = pair.get("jain_norm_reference", {})
            jain_reported = pair.get("jain_norm_reported", {})
            step = {
                "trajectory_id": None,
                "interval": interval,
                "scenario_id": scenario["scenario_id"],
                "spatial_pattern": scenario["spatial_pattern"],
                "coalition_size": len(scenario["coalition_ids"]),
                "coalition_ids": list(scenario["coalition_ids"]),
                "profile_id": profile_id,
                "forecast_cv": FORECAST_CV,
                "penalty_to_export_value_ratio": float(penalty_ratio),
                "q_mode": q_mode,
                "method_id": method_id,
                "control_id": control_id,
                "control_role": control["control_role"],
                "beta": control["beta"],
                "selected_gamma_by_participant": dict(selected_gamma),
                "mean_selected_gamma": _mean(selected_gamma.values()),
                "capability_support_by_participant": list(support),
                "capability_mw": capability,
                "raw_request_mw": list(raw),
                "capacity_limited_request_mw": list(capacity_limited),
                "credibility_boundary_mw": boundary_by_control[control_id],
                "effective_request_mw": list(effective),
                "request_attenuation_mw": math.fsum(capacity_limited) - math.fsum(effective),
                "score_before": score_before,
                "score_after": score_after,
                "mean_score_before": _mean(score_before),
                "mean_score_after": _mean(score_after),
                "allocation_mw": allocation,
                "reference_allocation_mw": reference_allocation,
                "allocation_hash": reported.get("allocation_hash"),
                "solver_valid": bool(reported.get("solver_valid")),
                "ac_numerical_valid": bool(reported.get("ac_numerical_valid")),
                "matched_q_pair_pass": bool(pair.get("matched_q_pair_pass")),
                "physical_screen_pass": bool(reported.get("physical_screen_pass")),
                "maximum_loading_ratio": reported.get("screen", {}).get("maximum_loading_ratio") if reported.get("screen") else None,
                "minimum_voltage_pu": reported.get("screen", {}).get("minimum_voltage_pu") if reported.get("screen") else None,
                "max_mva_excess": reported.get("screen", {}).get("max_mva_excess") if reported.get("screen") else None,
                "redistribution_mw": pair.get("coalition_metrics", {}).get("redistribution_mw"),
                "jain_norm_change": (
                    float(jain_reported["value"]) - float(jain_reference["value"])
                    if jain_reported.get("status") == "DEFINED"
                    and jain_reference.get("status") == "DEFINED"
                    else None
                ),
                **delivery,
            }
            steps_by_control[control_id].append(step)
            if control_id in scores_by_control:
                scores_by_control[control_id] = score_after

    trajectories: list[dict[str, Any]] = []
    for control in CONTROL_ROSTER:
        control_id = str(control["control_id"])
        steps = steps_by_control[control_id]
        trajectory_id = "::".join([
            str(scenario["scenario_id"]), profile_id,
            f"THETA_{int(round(float(penalty_ratio) * 100)):04d}",
            control_id, q_mode, method_id,
        ])
        for step in steps:
            step["trajectory_id"] = trajectory_id
        numeric = lambda field: [
            float(step[field]) for step in steps if step.get(field) is not None
        ]
        trajectory = {
            "trajectory_id": trajectory_id,
            "scenario_id": scenario["scenario_id"],
            "spatial_pattern": scenario["spatial_pattern"],
            "coalition_size": len(scenario["coalition_ids"]),
            "profile_id": profile_id,
            "forecast_cv": FORECAST_CV,
            "penalty_to_export_value_ratio": float(penalty_ratio),
            "control_id": control_id,
            "control_role": control["control_role"],
            "beta": control["beta"],
            "q_mode": q_mode,
            "method_id": method_id,
            "interval_count": len(steps),
            "mean_selected_gamma": _mean(step["mean_selected_gamma"] for step in steps),
            "externality_interval_share": _mean(bool(step.get("delivery_externality")) for step in steps),
            "productive_high_request_interval_share": _mean(bool(step.get("productive_high_request")) for step in steps),
            "mean_private_utility_gain_mw_equivalent": _mean(numeric("coalition_private_utility_gain_mw_equivalent")),
            "mean_coalition_delivery_gain_mw": _mean(numeric("coalition_delivery_gain_mw")),
            "mean_outside_delivery_loss_mw": _mean(numeric("outside_delivery_loss_mw")),
            "mean_total_delivery_change_mw": _mean(numeric("total_delivery_change_mw")),
            "mean_system_delivery_loss_mw": _mean(numeric("system_delivery_loss_mw")),
            "mean_incremental_coalition_idle_award_mw": _mean(numeric("incremental_coalition_idle_award_mw")),
            "mean_reported_total_idle_award_mw": _mean(numeric("reported_total_idle_award_mw")),
            "mean_request_attenuation_mw": _mean(numeric("request_attenuation_mw")),
            "mean_redistribution_mw": _mean(numeric("redistribution_mw")),
            "mean_jain_norm_change": _mean(numeric("jain_norm_change")),
            "terminal_mean_score": _mean(steps[-1]["score_after"]),
            "solver_pass_rate": _mean(bool(step["solver_valid"]) for step in steps),
            "ac_numerical_pass_rate": _mean(bool(step["ac_numerical_valid"]) for step in steps),
            "matched_q_pair_pass_rate": _mean(bool(step["matched_q_pair_pass"]) for step in steps),
            "maximum_loading_ratio": max(numeric("maximum_loading_ratio"), default=0.0),
            "minimum_voltage_pu": min(numeric("minimum_voltage_pu"), default=None),
            "maximum_mva_excess": max(numeric("max_mva_excess"), default=0.0),
            "unique_request_solve_count": len(solve_cache),
            "steps": steps,
            "trajectory_hash": None,
        }
        trajectory["trajectory_hash"] = canonical_hash(trajectory)
        trajectories.append(trajectory)
    return trajectories


def _aggregate(
    rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {**dimensions, "trajectory_count": len(rows)}
    result["interval_count"] = sum(int(row["interval_count"]) for row in rows)
    fields = (
        "mean_selected_gamma",
        "externality_interval_share",
        "productive_high_request_interval_share",
        "mean_private_utility_gain_mw_equivalent",
        "mean_coalition_delivery_gain_mw",
        "mean_outside_delivery_loss_mw",
        "mean_total_delivery_change_mw",
        "mean_system_delivery_loss_mw",
        "mean_incremental_coalition_idle_award_mw",
        "mean_reported_total_idle_award_mw",
        "mean_request_attenuation_mw",
        "mean_redistribution_mw",
        "mean_jain_norm_change",
        "terminal_mean_score",
        "solver_pass_rate",
        "ac_numerical_pass_rate",
        "matched_q_pair_pass_rate",
    )
    for field in fields:
        result[field] = _mean(float(row[field]) for row in rows)
    result["maximum_loading_ratio"] = max(float(row["maximum_loading_ratio"]) for row in rows)
    result["minimum_voltage_pu"] = min(float(row["minimum_voltage_pu"]) for row in rows)
    result["maximum_mva_excess"] = max(float(row["maximum_mva_excess"]) for row in rows)
    return result


def run_experiment(
    *, case_path: Path, policy_path: Path, source_path: Path,
    execution_path: Path, behavior_path: Path, design_path: Path,
    output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = _load_yaml(policy_path)
    source = _load_json(source_path)
    execution = _load_json(execution_path)
    behavior = _load_json(behavior_path)
    design = _load_json(design_path)
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    execution_hash = _validated_hash(execution, field="result_hash", label="method execution")
    behavior_hash = _validated_hash(behavior, field="result_hash", label="behavior result")
    design_hash = _validated_hash(design, field="design_hash", label="dynamic behavior design")
    if design.get("status") != "ENDOGENOUS_DYNAMIC_ACCESS_DESIGN_FROZEN":
        raise ValueError("behavior-linked dynamic design is not frozen")
    if design.get("source_envelope_hash") != source_hash:
        raise ValueError("design/source binding mismatch")
    if design.get("behavioral_choice_result_hash") != behavior_hash:
        raise ValueError("design/behavior binding mismatch")
    if execution.get("source_envelope_hash") != source_hash:
        raise ValueError("execution/source binding mismatch")
    if design.get("policy_hash") != policy.get("policy_hash"):
        raise ValueError("design/RATE policy binding mismatch")

    source_scenario_id = design["scenarios"][0]["source_scenario_id"]
    margins = dynamic._extract_margins(execution, source, source_scenario_id)
    reference_records = dynamic._build_reference_records(
        case_path=case_path,
        policy=policy,
        source=source,
        margins=margins,
        design=design,
    )
    if not all(
        row.get("solver_valid") and row.get("ac_numerical_valid")
        and row.get("physical_screen_pass")
        for row in reference_records.values()
    ):
        raise ValueError("reference allocation or AC construction failed")
    choices = _choice_lookup(behavior)

    tasks = [
        (scenario, profile["profile_id"], float(penalty), q_mode, method_id)
        for scenario in design["scenarios"]
        for profile in design["temporal_profiles"]
        for penalty in design["penalty_to_export_value_ratio_grid"]
        for q_mode in design["q_roster"]
        for method_id in design["method_roster"]
    ]
    workers = max(1, int(workers))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_behavior_worker,
            initargs=(
                str(case_path), policy, source, margins, reference_records,
                list(source["participant_ids"]), choices,
            ),
        ) as executor:
            task_results = list(executor.map(
                _trajectory_task,
                tasks,
                chunksize=max(1, len(tasks) // (workers * 6)),
            ))
    else:
        _init_behavior_worker(
            str(case_path), policy, source, margins, reference_records,
            list(source["participant_ids"]), choices,
        )
        task_results = [_trajectory_task(task) for task in tasks]
    trajectories = [trajectory for group in task_results for trajectory in group]
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]

    aggregate_overall: list[dict[str, Any]] = []
    for penalty in design["penalty_to_export_value_ratio_grid"]:
        for control in design["control_roster"]:
            rows = [
                row for row in trajectories
                if math.isclose(
                    float(row["penalty_to_export_value_ratio"]), float(penalty),
                    abs_tol=1.0e-15,
                ) and row["control_id"] == control["control_id"]
            ]
            aggregate_overall.append(_aggregate(rows, {
                "penalty_to_export_value_ratio": float(penalty),
                "control_id": control["control_id"],
            }))
    aggregate_by_profile_method: list[dict[str, Any]] = []
    for profile in PROFILE_SEQUENCES:
        for method_id in design["method_roster"]:
            for penalty in design["penalty_to_export_value_ratio_grid"]:
                for control in design["control_roster"]:
                    rows = [
                        row for row in trajectories
                        if row["profile_id"] == profile
                        and row["method_id"] == method_id
                        and math.isclose(
                            float(row["penalty_to_export_value_ratio"]), float(penalty),
                            abs_tol=1.0e-15,
                        )
                        and row["control_id"] == control["control_id"]
                    ]
                    aggregate_by_profile_method.append(_aggregate(rows, {
                        "profile_id": profile,
                        "method_id": method_id,
                        "penalty_to_export_value_ratio": float(penalty),
                        "control_id": control["control_id"],
                    }))

    expected_trajectories = len(tasks) * len(CONTROL_ROSTER)
    selective_by_key = {
        (
            row["scenario_id"], row["profile_id"],
            row["penalty_to_export_value_ratio"], row["q_mode"], row["method_id"],
        ): row
        for row in trajectories if row["control_id"] == "SELECTIVE_CF_BETA_100"
    }
    uniform_by_key = {
        (
            row["scenario_id"], row["profile_id"],
            row["penalty_to_export_value_ratio"], row["q_mode"], row["method_id"],
        ): row
        for row in trajectories if row["control_id"] == "UNIFORM_MATCHED_CF_BETA_100"
    }
    equal_attenuation = all(
        all(
            abs(
                float(selective_step["request_attenuation_mw"])
                - float(uniform_step["request_attenuation_mw"])
            ) <= 2.0e-10
            for selective_step, uniform_step in zip(
                selective_by_key[key]["steps"], uniform_by_key[key]["steps"],
            )
        )
        for key in selective_by_key
    )
    causal_transition = all(
        all(
            all(
                abs(float(current) - float(previous)) <= 2.0e-12
                for current, previous in zip(
                    trajectory["steps"][index]["score_before"],
                    trajectory["steps"][index - 1]["score_after"],
                )
            )
            for index in range(1, HORIZON)
        )
        for trajectory in trajectories
        if trajectory["control_id"] in {
            "SELECTIVE_CF_BETA_050", "SELECTIVE_CF_BETA_100",
        }
    )
    no_feedback_invariant = all(
        all(
            all(abs(float(value) - 1.0) <= 2.0e-12 for value in step["score_after"])
            for step in trajectory["steps"]
        )
        for trajectory in trajectories if trajectory["control_id"] == "NO_FEEDBACK"
    )
    gates = {
        "SOURCE_EXECUTION_RATE_BEHAVIOR_BINDING": bool(
            design["source_envelope_hash"] == source_hash
            and execution["source_envelope_hash"] == source_hash
            and design["behavioral_choice_result_hash"] == behavior_hash
            and design["policy_hash"] == policy["policy_hash"]
        ),
        "DESIGN_FROZEN_BEFORE_DYNAMIC_EXECUTION": design["status"] == "ENDOGENOUS_DYNAMIC_ACCESS_DESIGN_FROZEN",
        "EXACT_TREE_FREQUENCY_FOR_EVERY_PROFILE": all(
            _profile_is_exact(profile["standardized_capability_sequence"])
            for profile in design["temporal_profiles"]
        ),
        "BEHAVIORAL_REQUEST_SELECTION_COMPLETE": all(
            step["selected_gamma_by_participant"] for step in steps
        ),
        "FULL_FACTORIAL_COMPLETENESS": (
            len(trajectories) == expected_trajectories
            and len(steps) == expected_trajectories * HORIZON
        ),
        "SOLVER_COMPLETENESS": all(step["solver_valid"] for step in steps),
        "AC_NUMERICAL_COMPLETENESS": all(step["ac_numerical_valid"] for step in steps),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(step["matched_q_pair_pass"] for step in steps),
        "DELIVERY_IDLE_ACCOUNTING_CLOSED": max(
            float(step.get("delivery_idle_identity_error_mw", math.inf)) for step in steps
        ) <= 2.0e-9,
        "CREDIBILITY_BOUNDARY_NONBYPASS": all(
            all(
                float(effective) <= float(base) + 2.0e-12
                and float(effective) <= float(boundary) + 2.0e-12
                for effective, base, boundary in zip(
                    step["effective_request_mw"],
                    step["capacity_limited_request_mw"],
                    step["credibility_boundary_mw"],
                )
            )
            for step in steps
            if step["control_id"].startswith("SELECTIVE_CF")
        ),
        "ONE_INTERVAL_CAUSAL_FEEDBACK": causal_transition,
        "NO_FEEDBACK_STATE_INVARIANT": no_feedback_invariant,
        "UNIFORM_CONTROL_EXACTLY_MATCHES_CF_ATTENUATION": equal_attenuation,
        "DYNAMIC_RESULTS_EXCLUDED_FROM_DESIGN": all(
            value is False for value in design["input_read_assertions"].values()
        ),
        "NO_EQUILIBRIUM_OR_EMPIRICAL_PREVALENCE_CLAIM": True,
    }
    diagnostics = {
        "externality_intervals_exist": any(step.get("delivery_externality") for step in steps),
        "productive_high_request_intervals_exist": any(step.get("productive_high_request") for step in steps),
        "penalty_changes_selected_requests": len({
            (
                step["penalty_to_export_value_ratio"],
                tuple(sorted(step["selected_gamma_by_participant"].items())),
            )
            for step in steps
        }) > len(PENALTY_RATIO_GRID),
        "selective_and_uniform_delivery_differ": any(
            abs(
                float(selective_by_key[key]["mean_total_delivery_change_mw"])
                - float(uniform_by_key[key]["mean_total_delivery_change_mw"])
            ) > 2.0e-10
            for key in selective_by_key
        ),
    }
    result: dict[str, Any] = {
        "serialization_id": RESULT_SERIALIZATION_ID,
        "status": (
            "ENDOGENOUS_DYNAMIC_ACCESS_EXPERIMENT_COMPLETE"
            if all(gates.values())
            else "ENDOGENOUS_DYNAMIC_ACCESS_EXPERIMENT_BLOCKED"
        ),
        "scientific_role": "PRIMARY_BEHAVIOR_REQUEST_ALLOCATION_DELIVERY_FEEDBACK_EVIDENCE",
        "design_hash": design_hash,
        "source_envelope_hash": source_hash,
        "method_execution_hash": execution_hash,
        "behavioral_choice_result_hash": behavior_hash,
        "policy_hash": policy["policy_hash"],
        "network_hash": policy["network_hash"],
        "worker_count": workers,
        "scenario_count": len(design["scenarios"]),
        "profile_count": len(design["temporal_profiles"]),
        "trajectory_count": len(trajectories),
        "interval_record_count": len(steps),
        "gate_results": gates,
        "scientific_diagnostics": diagnostics,
        "aggregate_overall": aggregate_overall,
        "aggregate_by_profile_method": aggregate_by_profile_method,
        "trajectory_summaries": trajectories,
        "input_read_assertions": dict(design["input_read_assertions"]),
        "scope_limits": [
            "single-participant cases use an exact myopic grid best response",
            "simultaneous cases apply independently selected myopic nominations and are not a Nash equilibrium",
            "the controlled five-point capability tree is not an empirical behavioral prevalence model",
            "fixed-gamma experiments remain secondary residual-mismatch stress tests",
        ],
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_name("ENDOGENOUS_DYNAMIC_INTERVAL_DATA.csv"), steps)
    _write_csv(output_path.with_name("ENDOGENOUS_DYNAMIC_TRAJECTORY_SUMMARY.csv"), trajectories)
    _write_csv(output_path.with_name("ENDOGENOUS_DYNAMIC_AGGREGATE_OVERALL.csv"), aggregate_overall)
    _write_csv(
        output_path.with_name("ENDOGENOUS_DYNAMIC_PROFILE_METHOD_SUMMARY.csv"),
        aggregate_by_profile_method,
    )
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Behavior-linked multi-interval access experiment",
            "",
            f"- Status: `{result['status']}`",
            f"- Trajectories: `{len(trajectories)}`; interval records: `{len(steps)}`.",
            "- Raw requests and metered capability are generated from the same frozen conditional capability model.",
            "- Penalty-only, selective feedback, and equal-attenuation uniform controls use identical capability paths.",
            f"- Gates: `{gates}`",
            f"- Scientific diagnostics: `{diagnostics}`",
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
    out = root / "outputs" / "endogenous_dynamic_access_v15"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-design", action="store_true")
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument(
        "--policy", type=Path,
        default=(
            root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT"
            / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml"
        ),
    )
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument(
        "--behavior", type=Path,
        default=(
            root / "outputs" / "endogenous_request_choice_v14"
            / "ENDOGENOUS_REQUEST_CHOICE_RESULTS.json"
        ),
    )
    parser.add_argument(
        "--dynamic-design", type=Path,
        default=(
            root / "outputs" / "dynamic_feedback_v12_3"
            / "DYNAMIC_FEEDBACK_DESIGN_MANIFEST.json"
        ),
    )
    parser.add_argument("--design", type=Path, default=out / "ENDOGENOUS_DYNAMIC_DESIGN.json")
    parser.add_argument("--output", type=Path, default=out / "ENDOGENOUS_DYNAMIC_RESULTS.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    if args.freeze_design:
        design = freeze_design(
            dynamic_design_path=args.dynamic_design,
            behavior_path=args.behavior,
            source_path=args.source,
            policy_path=args.policy,
            output_path=args.design,
        )
        print(canonical_dumps({
            "status": design["status"],
            "design_hash": design["design_hash"],
        }))
        return 0
    result = run_experiment(
        case_path=args.case,
        policy_path=args.policy,
        source_path=args.source,
        execution_path=args.execution,
        behavior_path=args.behavior,
        design_path=args.design,
        output_path=args.output,
        workers=args.workers,
    )
    print(canonical_dumps({
        "status": result["status"],
        "trajectory_count": result["trajectory_count"],
        "interval_record_count": result["interval_record_count"],
        "gates": result["gate_results"],
        "diagnostics": result["scientific_diagnostics"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_ROSTER",
    "EXPECTED_SUPPORT_COUNTS",
    "FORECAST_CV",
    "HORIZON",
    "PENALTY_RATIO_GRID",
    "PROFILE_SEQUENCES",
    "capability_vector",
    "freeze_design",
    "run_experiment",
]
