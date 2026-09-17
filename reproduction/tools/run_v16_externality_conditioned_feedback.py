"""Run the V16 behavior--allocation--delivery--feedback experiment.

V16 keeps the V14 endogenous request choice and the frozen IEEE-141 physical
envelope.  It replaces unconditional boundary learning with a post-event,
same-realization replay gate.  A metered ceiling is reduced only when removing
idle awards releases access to other participants and increases total delivered
export.  Replay delivery is evaluated through an explicit verified
available-export telemetry interface.  The registered controlled-experiment
assumption sets that telemetry equal to simulation truth componentwise.  The
prior delivery-ratio filter is retained as an ablation baseline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from r4r.externality_feedback import (
    EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID,
    apply_access_ceiling,
    assess_replay,
    fulfillment_replay_request,
    ideal_verified_available_export_telemetry,
    metered_delivery,
    update_access_ceiling,
)
from r4r.serialization import canonical_dumps, canonical_hash
from tools.run_ieee141_m1_v12_3_mitigation_counterfactual import METHODS
from tools.run_v12_3_dynamic_horizontal_controls import matched_uniform_request
import tools.run_v12_3_dynamic_feedback_experiment as dynamic
import tools.run_v15_endogenous_dynamic_access as v15


DESIGN_SERIALIZATION_ID = "r4r.externality_conditioned_feedback_design.v1"
RESULT_SERIALIZATION_ID = "r4r.externality_conditioned_feedback_results.v1"
HORIZON = v15.HORIZON
FORECAST_CV = v15.FORECAST_CV
PENALTY_RATIO_GRID = v15.PENALTY_RATIO_GRID
PROFILE_SEQUENCES = v15.PROFILE_SEQUENCES
SCENARIO_IDS = v15.SCENARIO_IDS
AUTHORIZATION_TOLERANCE_MW = v15.AUTHORIZATION_TOLERANCE_MW
CLASSIFICATION_TOLERANCE_MW = v15.CLASSIFICATION_TOLERANCE_MW
REPLAY_TOLERANCE_MW = 2.0e-10

CONTROL_ROSTER = (
    {
        "control_id": "NO_FEEDBACK",
        "control_role": "PENALTY_ONLY_BEHAVIORAL_REQUEST",
        "beta": 0.0,
    },
    {
        "control_id": "DELIVERY_RATIO_CF_BETA_050",
        "control_role": "DELIVERY_ONLY_RATIO_FILTER_ABLATION",
        "beta": 0.5,
    },
    {
        "control_id": "EXTERNALITY_CF_BETA_025",
        "control_role": "COUNTERFACTUAL_EXTERNALITY_CONDITIONED_FEEDBACK",
        "beta": 0.25,
    },
    {
        "control_id": "EXTERNALITY_CF_BETA_050",
        "control_role": "COUNTERFACTUAL_EXTERNALITY_CONDITIONED_FEEDBACK",
        "beta": 0.5,
    },
    {
        "control_id": "EXTERNALITY_CF_BETA_100",
        "control_role": "COUNTERFACTUAL_EXTERNALITY_CONDITIONED_FEEDBACK",
        "beta": 1.0,
    },
    {
        "control_id": "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
        "control_role": "EQUAL_ATTENUATION_UNIFORM_CONTROL",
        "beta": None,
    },
)
EXTERNALITY_CONTROLS = tuple(
    row["control_id"]
    for row in CONTROL_ROSTER
    if row["control_role"] == "COUNTERFACTUAL_EXTERNALITY_CONDITIONED_FEEDBACK"
)


def _freeze_design(
    *, dynamic_design_path: Path, behavior_path: Path, source_path: Path,
    policy_path: Path, output_path: Path,
) -> dict[str, Any]:
    dynamic_design = v15._load_json(dynamic_design_path)
    behavior = v15._load_json(behavior_path)
    source = v15._load_json(source_path)
    policy = v15._load_yaml(policy_path)
    dynamic_hash = v15._validated_hash(
        dynamic_design, field="design_hash", label="dynamic scenario design",
    )
    behavior_hash = v15._validated_hash(
        behavior, field="result_hash", label="endogenous request experiment",
    )
    source_hash = v15._validated_hash(source, field="result_hash", label="source envelope")
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
    if not all(v15._profile_is_exact(sequence) for sequence in PROFILE_SEQUENCES.values()):
        raise ValueError("a temporal profile does not reproduce the registered capability tree")

    scenario_by_id = {
        str(row["scenario_id"]): copy.deepcopy(dict(row))
        for row in dynamic_design["scenarios"]
    }
    missing = sorted(set(SCENARIO_IDS) - set(scenario_by_id))
    if missing:
        raise ValueError(f"required scenarios are unavailable: {missing}")
    scenarios = [scenario_by_id[scenario_id] for scenario_id in SCENARIO_IDS]
    choices = v15._choice_lookup(behavior)
    required = {
        (participant_id, q_mode, method_id, FORECAST_CV, penalty)
        for scenario in scenarios
        for participant_id in scenario["coalition_ids"]
        for q_mode in source["q_roster"]
        for method_id in METHODS
        for penalty in PENALTY_RATIO_GRID
    }
    missing_choices = sorted(required - set(choices))
    if missing_choices:
        raise ValueError(f"missing endogenous request choices: {missing_choices[:5]}")

    design: dict[str, Any] = {
        "serialization_id": DESIGN_SERIALIZATION_ID,
        "status": "EXTERNALITY_CONDITIONED_FEEDBACK_DESIGN_FROZEN",
        "scientific_role": "PRIMARY_BEHAVIOR_ALLOCATION_DELIVERY_EXTERNALITY_FEEDBACK_EVIDENCE",
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
                "evidence_role": "PRIMARY_TEMPORAL_ORDER_ROBUSTNESS",
            }
            for profile_id, sequence in PROFILE_SEQUENCES.items()
        ],
        "participant_phase_rule": (
            "offset_i=(7*i) mod 16; every participant retains the exact "
            "five-point support frequency"
        ),
        "scenarios": scenarios,
        "behavioral_contract": {
            "choice": (
                "gamma_i^* is the smallest maximizer of expected delivered value "
                "minus theta times idle-award consequence on the frozen uniform grid"
            ),
            "raw_request": (
                "r_i=gamma_i^*(theta,sigma,rule,Q)*forecast_i; unchanged participants "
                "retain capability-consistent reference requests"
            ),
            "realized_capability": (
                "a_i(t)=min(C_i,(1+sigma*z_i(t))*forecast_i), using the same "
                "conditional tree as the request choice"
            ),
            "delivery": "y_i(t)=min(x_i(t),a_i(t)); no strategic withholding is modeled",
            "simultaneous_scope": (
                "changed participants apply independent myopic choices simultaneously; "
                "no equilibrium or empirical prevalence claim"
            ),
        },
        "externality_feedback_contract": {
            "idle_set": "I_t={i:x_i(t)-y_i(t)>epsilon_x}",
            "replay_request": (
                "r_i^rep(t)=min(r_i^eff(t),y_i(t)) for i in I_t and r_i^eff(t) otherwise"
            ),
            "same_realization_replay": (
                "x^rep=Pi(r^rep) and y_i^rep=min(x_i^rep,a_hat_i^tel(t)); the "
                "completed allocation is never rewritten"
            ),
            "available_export_telemetry": (
                "post-event site-level telemetry a_hat_i^tel(t) reports the maximum "
                "net export available at the connection point for the completed interval"
            ),
            "telemetry_assumption": (
                f"{EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID}: "
                "a_hat_i^tel(t)=a_i(t) componentwise in the controlled experiment; "
                "measurement error, latency, and dropout are outside the tested scope"
            ),
            "gate": (
                "g_t=1 only when the replay is solver/AC/physical valid, releases "
                "positive outsider allocation, raises outsider delivery, and raises "
                "total feeder delivery"
            ),
            "mw_ceiling_update": (
                "c_i(t+1)=(1-beta)c_i(t)+beta*y_i(t) for gated idle participants; "
                "the recovery target is C_i otherwise"
            ),
            "effective_request": "min(r_i(t+1),C_i,c_i(t+1))",
            "online_information": (
                "submitted request, frozen allocation model, registered capacity, "
                "completed allocation, metered delivery, and verified post-event "
                "available-export telemetry; the private F_i and s_i are not learned online"
            ),
        },
        "ablation_contract": {
            "delivery_ratio_filter": (
                "legacy chi update from y/x is retained only as a delivery-only ablation"
            ),
            "equal_attenuation_uniform": (
                "uniform scaling removes exactly the total request removed by "
                "EXTERNALITY_CF_BETA_050 in the same paired interval"
            ),
        },
        "negative_invariants": [
            "full delivery leaves the replay gate closed",
            "mismatch without outsider access release leaves the replay gate closed",
            "released access without deliverable recovery leaves the replay gate closed",
            "a feeder-wide shortfall with no outsider set leaves the replay gate closed",
            "a raw request cannot bypass the active MW ceiling",
        ],
        "input_read_assertions": {
            "v15_dynamic_outcomes_used_to_select_v16_controls": False,
            "v16_outcomes_used_to_select_penalty_grid": False,
            "v16_outcomes_used_to_select_feedback_gain_grid": False,
            "v16_outcomes_used_to_refit_behavior_network_or_rate": False,
            "private_distribution_required_by_online_feedback": False,
            "simulation_truth_used_directly_by_feedback_interface": False,
            "verified_telemetry_equals_simulation_truth_in_primary_experiment": True,
        },
        "design_hash": None,
    }
    design["design_hash"] = canonical_hash(design)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(design) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Frozen V16 behavior and externality-feedback design",
            "",
            f"- Scenarios: `{len(scenarios)}`; profiles: `{len(PROFILE_SEQUENCES)}`; horizon: `{HORIZON}`.",
            f"- Controls: `{[row['control_id'] for row in CONTROL_ROSTER]}`.",
            "- Request choice and delivery use one conditional capability model.",
            (
                "- Replay reads verified available-export telemetry; under "
                f"`{EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID}` it equals simulation truth exactly."
            ),
            "- The proposed feedback opens only after a physically valid same-realization replay verifies usable outsider release.",
            "- The V15 delivery-ratio rule is retained only as an ablation.",
            f"- Design hash: `{design['design_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return design


def _init_worker(
    case_path: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    participant_ids: Sequence[str],
    choices: Mapping[tuple[str, str, str, float, float], Mapping[str, Any]],
) -> None:
    v15._init_behavior_worker(
        case_path, policy, source, margins, reference_records, participant_ids, choices,
    )


def _externality_replay(
    *,
    cache: dict[tuple[float, ...], dict[str, Any]],
    scenario: Mapping[str, Any],
    effective_request: Sequence[float],
    reported: Mapping[str, Any],
    allocation: Sequence[float],
    delivery: Sequence[float],
    available_export_telemetry: Sequence[float],
    q_mode: str,
    method_id: str,
    control_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    replay_request, idle_indices = fulfillment_replay_request(
        effective_request,
        allocation,
        delivery,
        tolerance_mw=AUTHORIZATION_TOLERANCE_MW,
    )
    if not idle_indices:
        assessment = assess_replay(
            allocation_mw=allocation,
            delivery_mw=delivery,
            replay_allocation_mw=allocation,
            replay_delivery_mw=delivery,
            idle_indices=(),
            replay_valid=True,
            tolerance_mw=REPLAY_TOLERANCE_MW,
        )
        return {
            **assessment.to_json(),
            "replay_attempted": False,
            "replay_request_mw": list(replay_request),
            "replay_solver_valid": None,
            "replay_ac_numerical_valid": None,
            "replay_physical_screen_pass": None,
            "replay_allocation_mw": list(allocation),
            "replay_delivery_mw": list(delivery),
            "replay_allocation_hash": reported.get("allocation_hash"),
            "replay_ac_result_hash": reported.get("ac_result_hash"),
        }, None

    replay_record = v15._solve_request(
        cache=cache,
        scenario=scenario,
        request=replay_request,
        q_mode=q_mode,
        method_id=method_id,
        mode=f"{control_id}_SAME_REALIZATION_REPLAY",
    )
    replay_allocation = [float(value) for value in replay_record.get("allocation_mw", [])]
    vector_valid = len(replay_allocation) == len(allocation)
    replay_valid = bool(
        vector_valid
        and replay_record.get("solver_valid")
        and replay_record.get("ac_numerical_valid")
        and replay_record.get("physical_screen_pass")
    )
    replay_delivery = (
        metered_delivery(replay_allocation, available_export_telemetry)
        if vector_valid
        else [float(value) for value in delivery]
    )
    assessment = assess_replay(
        allocation_mw=allocation,
        delivery_mw=delivery,
        replay_allocation_mw=(replay_allocation if vector_valid else allocation),
        replay_delivery_mw=replay_delivery,
        idle_indices=idle_indices,
        replay_valid=replay_valid,
        tolerance_mw=REPLAY_TOLERANCE_MW,
    )
    return {
        **assessment.to_json(),
        "replay_attempted": True,
        "replay_request_mw": list(replay_request),
        "replay_solver_valid": bool(replay_record.get("solver_valid")),
        "replay_ac_numerical_valid": bool(replay_record.get("ac_numerical_valid")),
        "replay_physical_screen_pass": bool(replay_record.get("physical_screen_pass")),
        "replay_allocation_mw": replay_allocation,
        "replay_delivery_mw": replay_delivery,
        "replay_allocation_hash": replay_record.get("allocation_hash"),
        "replay_ac_result_hash": replay_record.get("ac_result_hash"),
    }, replay_record


def _trajectory_task(
    task: tuple[dict[str, Any], str, float, str, str],
) -> list[dict[str, Any]]:
    scenario, profile_id, penalty_ratio, q_mode, method_id = task
    participant_ids = list(v15._WORKER["participant_ids"])
    participant_index = {
        participant_id: index for index, participant_id in enumerate(participant_ids)
    }
    coalition_indices = [
        participant_index[str(value)] for value in scenario["coalition_ids"]
    ]
    capacity = [float(value) for value in scenario["capacity_mw"]]
    reference = v15._WORKER["reference_records"][f"{q_mode}::{method_id}"]
    reference_allocation = [float(value) for value in reference["allocation_mw"]]
    raw, selected_gamma = v15._raw_behavioral_request(
        scenario,
        q_mode=q_mode,
        method_id=method_id,
        penalty_ratio=float(penalty_ratio),
    )
    capacity_limited = dynamic._capacity_limited_request(raw, capacity)
    ratio_scores = [1.0 for _ in participant_ids]
    ceilings_by_control = {
        control_id: list(capacity) for control_id in EXTERNALITY_CONTROLS
    }
    solve_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    steps_by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for interval in range(1, HORIZON + 1):
        capability, support = v15.capability_vector(
            profile_id=profile_id,
            interval=interval,
            forecast_mw=scenario["reference_request_mw"],
            capacity_mw=capacity,
        )
        available_export_telemetry = ideal_verified_available_export_telemetry(
            capability
        )
        telemetry_truth_max_abs_error_mw = max(
            (
                abs(float(telemetry) - float(truth))
                for telemetry, truth in zip(available_export_telemetry, capability)
            ),
            default=0.0,
        )
        ratio_effective, ratio_boundary = dynamic._credibility_capped_request(
            capacity_limited, capacity, ratio_scores,
        )
        effective_by_control: dict[str, list[float]] = {
            "NO_FEEDBACK": list(capacity_limited),
            "DELIVERY_RATIO_CF_BETA_050": ratio_effective,
        }
        boundary_by_control: dict[str, list[float]] = {
            "NO_FEEDBACK": list(capacity),
            "DELIVERY_RATIO_CF_BETA_050": ratio_boundary,
        }
        for control_id in EXTERNALITY_CONTROLS:
            effective_by_control[control_id] = apply_access_ceiling(
                capacity_limited, capacity, ceilings_by_control[control_id],
            )
            boundary_by_control[control_id] = list(ceilings_by_control[control_id])
        uniform = matched_uniform_request(
            capacity_limited, effective_by_control["EXTERNALITY_CF_BETA_050"],
        )
        effective_by_control["UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050"] = uniform
        boundary_by_control["UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050"] = list(uniform)

        for control in CONTROL_ROSTER:
            control_id = str(control["control_id"])
            effective = effective_by_control[control_id]
            step_scenario = copy.deepcopy(dict(scenario))
            step_scenario["reported_request_mw"] = list(raw)
            reported = v15._solve_request(
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
                delivery_metrics = v15._delivery_metrics(
                    reference_allocation=reference_allocation,
                    allocation=allocation,
                    capability=capability,
                    coalition_indices=coalition_indices,
                    penalty_ratio=float(penalty_ratio),
                )
                delivery = list(delivery_metrics["delivery_mw"])
                ratios = list(delivery_metrics["delivery_ratio"])
            else:
                delivery_metrics = {}
                delivery = []
                ratios = []

            replay_metrics: dict[str, Any] = {
                "idle_indices": [],
                "total_idle_award_mw": 0.0,
                "outsider_allocation_gain_mw": 0.0,
                "outsider_delivery_gain_mw": 0.0,
                "total_delivery_gain_mw": 0.0,
                "allocation_redistribution_mw": 0.0,
                "gate_triggered": False,
                "gate_reason": "NOT_APPLICABLE_TO_CONTROL",
                "replay_attempted": False,
                "replay_request_mw": list(effective),
                "replay_solver_valid": None,
                "replay_ac_numerical_valid": None,
                "replay_physical_screen_pass": None,
                "replay_allocation_mw": list(allocation),
                "replay_delivery_mw": list(delivery),
                "replay_allocation_hash": reported.get("allocation_hash"),
                "replay_ac_result_hash": reported.get("ac_result_hash"),
            }
            ceiling_before = list(boundary_by_control[control_id])
            ceiling_after = list(ceiling_before)
            ceiling_target = list(capacity)

            if control_id == "DELIVERY_RATIO_CF_BETA_050" and ratios:
                ratio_scores = v15._updated_scores(
                    ratio_scores, ratios, float(control["beta"]),
                )
                ceiling_after = [
                    score * cap for score, cap in zip(ratio_scores, capacity)
                ]
                ceiling_target = [ratio * cap for ratio, cap in zip(ratios, capacity)]
            elif control_id in EXTERNALITY_CONTROLS and valid:
                replay_metrics, _ = _externality_replay(
                    cache=solve_cache,
                    scenario=step_scenario,
                    effective_request=effective,
                    reported=reported,
                    allocation=allocation,
                    delivery=delivery,
                    available_export_telemetry=available_export_telemetry,
                    q_mode=q_mode,
                    method_id=method_id,
                    control_id=control_id,
                )
                ceiling_after, ceiling_target = update_access_ceiling(
                    ceiling_before_mw=ceilings_by_control[control_id],
                    registered_capacity_mw=capacity,
                    delivery_mw=delivery,
                    idle_indices=replay_metrics["idle_indices"],
                    gate_triggered=bool(replay_metrics["gate_triggered"]),
                    beta=float(control["beta"]),
                )
                ceilings_by_control[control_id] = list(ceiling_after)

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
                "mean_selected_gamma": v15._mean(selected_gamma.values()),
                "capability_support_by_participant": list(support),
                "capability_mw": capability,
                "available_export_telemetry_mw": list(available_export_telemetry),
                "telemetry_assumption_id": EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID,
                "telemetry_truth_max_abs_error_mw": telemetry_truth_max_abs_error_mw,
                "raw_request_mw": list(raw),
                "capacity_limited_request_mw": list(capacity_limited),
                "access_ceiling_before_mw": ceiling_before,
                "access_ceiling_target_mw": ceiling_target,
                "access_ceiling_after_mw": ceiling_after,
                "effective_request_mw": list(effective),
                "request_attenuation_mw": math.fsum(capacity_limited) - math.fsum(effective),
                "mean_access_ceiling_ratio_before": v15._mean(
                    before / cap if cap > 0.0 else 1.0
                    for before, cap in zip(ceiling_before, capacity)
                ),
                "mean_access_ceiling_ratio_after": v15._mean(
                    after / cap if cap > 0.0 else 1.0
                    for after, cap in zip(ceiling_after, capacity)
                ),
                "allocation_mw": allocation,
                "reference_allocation_mw": reference_allocation,
                "allocation_hash": reported.get("allocation_hash"),
                "solver_valid": bool(reported.get("solver_valid")),
                "ac_numerical_valid": bool(reported.get("ac_numerical_valid")),
                "matched_q_pair_pass": bool(pair.get("matched_q_pair_pass")),
                "physical_screen_pass": bool(reported.get("physical_screen_pass")),
                "maximum_loading_ratio": (
                    reported.get("screen", {}).get("maximum_loading_ratio")
                    if reported.get("screen") else None
                ),
                "minimum_voltage_pu": (
                    reported.get("screen", {}).get("minimum_voltage_pu")
                    if reported.get("screen") else None
                ),
                "max_mva_excess": (
                    reported.get("screen", {}).get("max_mva_excess")
                    if reported.get("screen") else None
                ),
                "redistribution_mw": pair.get("coalition_metrics", {}).get("redistribution_mw"),
                "jain_norm_change": (
                    float(jain_reported["value"]) - float(jain_reference["value"])
                    if jain_reported.get("status") == "DEFINED"
                    and jain_reference.get("status") == "DEFINED"
                    else None
                ),
                **delivery_metrics,
                **{f"replay_{key}" if key in {
                    "idle_indices", "total_idle_award_mw", "outsider_allocation_gain_mw",
                    "outsider_delivery_gain_mw", "total_delivery_gain_mw",
                    "allocation_redistribution_mw", "gate_triggered", "gate_reason",
                } else key: value for key, value in replay_metrics.items()},
            }
            steps_by_control[control_id].append(step)

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
        replay_steps = [step for step in steps if step.get("replay_attempted")]
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
            "mean_selected_gamma": v15._mean(step["mean_selected_gamma"] for step in steps),
            "externality_interval_share": v15._mean(bool(step.get("delivery_externality")) for step in steps),
            "productive_high_request_interval_share": v15._mean(bool(step.get("productive_high_request")) for step in steps),
            "mean_private_utility_gain_mw_equivalent": v15._mean(numeric("coalition_private_utility_gain_mw_equivalent")),
            "mean_coalition_delivery_gain_mw": v15._mean(numeric("coalition_delivery_gain_mw")),
            "mean_outside_delivery_loss_mw": v15._mean(numeric("outside_delivery_loss_mw")),
            "mean_total_delivery_change_mw": v15._mean(numeric("total_delivery_change_mw")),
            "mean_system_delivery_loss_mw": v15._mean(numeric("system_delivery_loss_mw")),
            "mean_incremental_coalition_idle_award_mw": v15._mean(numeric("incremental_coalition_idle_award_mw")),
            "mean_reported_total_idle_award_mw": v15._mean(numeric("reported_total_idle_award_mw")),
            "mean_request_attenuation_mw": v15._mean(numeric("request_attenuation_mw")),
            "mean_redistribution_mw": v15._mean(numeric("redistribution_mw")),
            "mean_jain_norm_change": v15._mean(numeric("jain_norm_change")),
            "terminal_mean_score": float(steps[-1]["mean_access_ceiling_ratio_after"]),
            "externality_gate_interval_share": v15._mean(
                bool(step.get("replay_gate_triggered")) for step in steps
            ),
            "replay_attempt_rate": len(replay_steps) / len(steps),
            "replay_valid_rate": (
                v15._mean(bool(step.get("replay_solver_valid"))
                          and bool(step.get("replay_ac_numerical_valid"))
                          and bool(step.get("replay_physical_screen_pass"))
                          for step in replay_steps)
                if replay_steps else 1.0
            ),
            "mean_replay_outsider_allocation_gain_mw": v15._mean(numeric("replay_outsider_allocation_gain_mw")),
            "mean_replay_outsider_delivery_gain_mw": v15._mean(numeric("replay_outsider_delivery_gain_mw")),
            "mean_replay_total_delivery_gain_mw": v15._mean(numeric("replay_total_delivery_gain_mw")),
            "mean_replay_redistribution_mw": v15._mean(numeric("replay_allocation_redistribution_mw")),
            "solver_pass_rate": v15._mean(bool(step["solver_valid"]) for step in steps),
            "ac_numerical_pass_rate": v15._mean(bool(step["ac_numerical_valid"]) for step in steps),
            "matched_q_pair_pass_rate": v15._mean(bool(step["matched_q_pair_pass"]) for step in steps),
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
    result = v15._aggregate(rows, dimensions)
    for field in (
        "externality_gate_interval_share",
        "replay_attempt_rate",
        "replay_valid_rate",
        "mean_replay_outsider_allocation_gain_mw",
        "mean_replay_outsider_delivery_gain_mw",
        "mean_replay_total_delivery_gain_mw",
        "mean_replay_redistribution_mw",
    ):
        result[field] = v15._mean(float(row[field]) for row in rows)
    return result


def _ceiling_transition_valid(step: Mapping[str, Any], capacity: Sequence[float]) -> bool:
    if step["control_id"] not in EXTERNALITY_CONTROLS:
        return True
    beta = float(step["beta"])
    return all(
        abs(
            float(after)
            - ((1.0 - beta) * float(before) + beta * float(target))
        ) <= 2.0e-12
        and 0.0 <= float(after) <= float(cap) + 2.0e-12
        for before, target, after, cap in zip(
            step["access_ceiling_before_mw"],
            step["access_ceiling_target_mw"],
            step["access_ceiling_after_mw"],
            capacity,
        )
    )


def run_experiment(
    *, case_path: Path, policy_path: Path, source_path: Path,
    execution_path: Path, behavior_path: Path, design_path: Path,
    output_path: Path, workers: int,
) -> dict[str, Any]:
    policy = v15._load_yaml(policy_path)
    source = v15._load_json(source_path)
    execution = v15._load_json(execution_path)
    behavior = v15._load_json(behavior_path)
    design = v15._load_json(design_path)
    source_hash = v15._validated_hash(source, field="result_hash", label="source envelope")
    execution_hash = v15._validated_hash(execution, field="result_hash", label="method execution")
    behavior_hash = v15._validated_hash(behavior, field="result_hash", label="behavior result")
    design_hash = v15._validated_hash(design, field="design_hash", label="V16 design")
    if design.get("status") != "EXTERNALITY_CONDITIONED_FEEDBACK_DESIGN_FROZEN":
        raise ValueError("V16 design is not frozen")
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
    choices = v15._choice_lookup(behavior)
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
            initializer=_init_worker,
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
        _init_worker(
            str(case_path), policy, source, margins, reference_records,
            list(source["participant_ids"]), choices,
        )
        task_results = [_trajectory_task(task) for task in tasks]
    trajectories = [trajectory for group in task_results for trajectory in group]
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]

    aggregate_overall = []
    for penalty in design["penalty_to_export_value_ratio_grid"]:
        for control in design["control_roster"]:
            rows = [
                row for row in trajectories
                if math.isclose(
                    float(row["penalty_to_export_value_ratio"]), float(penalty),
                    abs_tol=1.0e-15,
                )
                and row["control_id"] == control["control_id"]
            ]
            aggregate_overall.append(_aggregate(rows, {
                "penalty_to_export_value_ratio": float(penalty),
                "control_id": control["control_id"],
            }))
    aggregate_by_profile_method = []
    for profile_id in PROFILE_SEQUENCES:
        for method_id in design["method_roster"]:
            for penalty in design["penalty_to_export_value_ratio_grid"]:
                for control in design["control_roster"]:
                    rows = [
                        row for row in trajectories
                        if row["profile_id"] == profile_id
                        and row["method_id"] == method_id
                        and math.isclose(
                            float(row["penalty_to_export_value_ratio"]), float(penalty),
                            abs_tol=1.0e-15,
                        )
                        and row["control_id"] == control["control_id"]
                    ]
                    aggregate_by_profile_method.append(_aggregate(rows, {
                        "profile_id": profile_id,
                        "method_id": method_id,
                        "penalty_to_export_value_ratio": float(penalty),
                        "control_id": control["control_id"],
                    }))

    pair_key = lambda row: (
        row["scenario_id"], row["profile_id"],
        row["penalty_to_export_value_ratio"], row["q_mode"], row["method_id"],
    )
    ecf = {
        pair_key(row): row for row in trajectories
        if row["control_id"] == "EXTERNALITY_CF_BETA_050"
    }
    uniform = {
        pair_key(row): row for row in trajectories
        if row["control_id"] == "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050"
    }
    equal_attenuation = all(
        all(
            abs(float(left["request_attenuation_mw"])
                - float(right["request_attenuation_mw"])) <= 2.0e-10
            for left, right in zip(ecf[key]["steps"], uniform[key]["steps"])
        )
        for key in ecf
    )
    capacity_by_scenario = {
        row["scenario_id"]: [float(value) for value in row["capacity_mw"]]
        for row in design["scenarios"]
    }
    attempted_replays = [step for step in steps if step.get("replay_attempted")]
    ecf_steps = [step for step in steps if step["control_id"] in EXTERNALITY_CONTROLS]
    gate_steps = [step for step in ecf_steps if step.get("replay_gate_triggered")]
    closed_gate_steps = [step for step in ecf_steps if not step.get("replay_gate_triggered")]
    transitions = all(
        all(
            all(abs(float(a) - float(b)) <= 2.0e-12 for a, b in zip(
                trajectory["steps"][index]["access_ceiling_before_mw"],
                trajectory["steps"][index - 1]["access_ceiling_after_mw"],
            ))
            for index in range(1, HORIZON)
        )
        for trajectory in trajectories
        if trajectory["control_id"] in EXTERNALITY_CONTROLS
        or trajectory["control_id"] == "DELIVERY_RATIO_CF_BETA_050"
    )
    gates = {
        "SOURCE_EXECUTION_RATE_BEHAVIOR_BINDING": bool(
            design["source_envelope_hash"] == source_hash
            and execution["source_envelope_hash"] == source_hash
            and design["behavioral_choice_result_hash"] == behavior_hash
            and design["policy_hash"] == policy["policy_hash"]
        ),
        "DESIGN_FROZEN_BEFORE_EXECUTION": design["status"] == "EXTERNALITY_CONDITIONED_FEEDBACK_DESIGN_FROZEN",
        "EXACT_TREE_FREQUENCY_FOR_EVERY_PROFILE": all(
            v15._profile_is_exact(profile["standardized_capability_sequence"])
            for profile in design["temporal_profiles"]
        ),
        "FULL_FACTORIAL_COMPLETENESS": (
            len(trajectories) == len(tasks) * len(CONTROL_ROSTER)
            and len(steps) == len(tasks) * len(CONTROL_ROSTER) * HORIZON
        ),
        "SOLVER_COMPLETENESS": all(step["solver_valid"] for step in steps),
        "AC_NUMERICAL_COMPLETENESS": all(step["ac_numerical_valid"] for step in steps),
        "MATCHED_Q_PHYSICAL_COMPLETENESS": all(step["matched_q_pair_pass"] for step in steps),
        "REPLAY_SOLVER_AC_PHYSICAL_COMPLETENESS": bool(attempted_replays) and all(
            step.get("replay_solver_valid")
            and step.get("replay_ac_numerical_valid")
            and step.get("replay_physical_screen_pass")
            for step in attempted_replays
        ),
        "DELIVERY_IDLE_ACCOUNTING_CLOSED": max(
            float(step.get("delivery_idle_identity_error_mw", math.inf)) for step in steps
        ) <= 2.0e-9,
        "EXACT_VERIFIED_TELEMETRY_BINDING": all(
            step.get("telemetry_assumption_id")
            == EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID
            and float(step.get("telemetry_truth_max_abs_error_mw", math.inf)) == 0.0
            and step.get("available_export_telemetry_mw") == step.get("capability_mw")
            for step in steps
        ),
        "EXTERNALITY_GATE_REQUIRES_USABLE_OUTSIDER_RELEASE": bool(gate_steps) and all(
            float(step["replay_outsider_allocation_gain_mw"]) > REPLAY_TOLERANCE_MW
            and float(step["replay_outsider_delivery_gain_mw"]) > REPLAY_TOLERANCE_MW
            and float(step["replay_total_delivery_gain_mw"]) > REPLAY_TOLERANCE_MW
            for step in gate_steps
        ),
        "NO_IDLE_OR_NO_RELEASE_NEVER_OPENS_GATE": bool(closed_gate_steps) and all(
            not step.get("replay_gate_triggered")
            for step in ecf_steps
            if not step.get("replay_idle_indices")
            or float(step.get("replay_outsider_allocation_gain_mw", 0.0)) <= REPLAY_TOLERANCE_MW
            or float(step.get("replay_outsider_delivery_gain_mw", 0.0)) <= REPLAY_TOLERANCE_MW
            or float(step.get("replay_total_delivery_gain_mw", 0.0)) <= REPLAY_TOLERANCE_MW
        ),
        "MW_CEILING_TRANSITION_IDENTITY": all(
            _ceiling_transition_valid(step, capacity_by_scenario[step["scenario_id"]])
            for step in steps
        ),
        "ONE_INTERVAL_CAUSAL_TRANSITION": transitions,
        "ACCESS_CEILING_NONBYPASS": all(
            all(
                float(effective) <= float(base) + 2.0e-12
                and float(effective) <= float(boundary) + 2.0e-12
                for effective, base, boundary in zip(
                    step["effective_request_mw"],
                    step["capacity_limited_request_mw"],
                    step["access_ceiling_before_mw"],
                )
            )
            for step in steps
            if step["control_id"] in EXTERNALITY_CONTROLS
        ),
        "UNIFORM_CONTROL_EXACTLY_MATCHES_PRIMARY_ATTENUATION": equal_attenuation,
        "DYNAMIC_RESULTS_EXCLUDED_FROM_DESIGN": all(
            value is False for value in design["input_read_assertions"].values()
        ),
        "NO_WITHHOLDING_EQUILIBRIUM_OR_EMPIRICAL_PREVALENCE_CLAIM": True,
    }
    diagnostics = {
        "productive_high_request_intervals_exist": any(
            step.get("productive_high_request") for step in steps
        ),
        "delivery_externality_intervals_exist": any(
            step.get("delivery_externality") for step in steps
        ),
        "externality_replay_gate_opens": bool(gate_steps),
        "externality_replay_gate_closes": bool(closed_gate_steps),
        "mismatch_without_gate_exists": any(
            step.get("replay_idle_indices") and not step.get("replay_gate_triggered")
            for step in ecf_steps
        ),
        "ceiling_recovery_observed": any(
            any(float(after) > float(before) + 2.0e-12 for before, after in zip(
                step["access_ceiling_before_mw"], step["access_ceiling_after_mw"],
            ))
            for step in ecf_steps
        ),
        "proposed_and_delivery_only_ablation_differ": any(
            trajectory["control_id"] == "EXTERNALITY_CF_BETA_050"
            and any(
                abs(float(a) - float(b)) > 2.0e-10
                for a, b in zip(
                    trajectory["steps"][index]["effective_request_mw"],
                    next(
                        row for row in trajectories
                        if pair_key(row) == pair_key(trajectory)
                        and row["control_id"] == "DELIVERY_RATIO_CF_BETA_050"
                    )["steps"][index]["effective_request_mw"],
                )
            )
            for trajectory in trajectories
            for index in range(HORIZON)
            if trajectory["control_id"] == "EXTERNALITY_CF_BETA_050"
        ),
    }
    result: dict[str, Any] = {
        "serialization_id": RESULT_SERIALIZATION_ID,
        "status": (
            "EXTERNALITY_CONDITIONED_FEEDBACK_EXPERIMENT_COMPLETE"
            if all(gates.values())
            else "EXTERNALITY_CONDITIONED_FEEDBACK_EXPERIMENT_BLOCKED"
        ),
        "scientific_role": "PRIMARY_BEHAVIOR_ALLOCATION_DELIVERY_EXTERNALITY_FEEDBACK_EVIDENCE",
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
        "replay_attempt_count": len(attempted_replays),
        "replay_gate_open_count": len(gate_steps),
        "gate_results": gates,
        "scientific_diagnostics": diagnostics,
        "aggregate_overall": aggregate_overall,
        "aggregate_by_profile_method": aggregate_by_profile_method,
        "trajectory_summaries": trajectories,
        "input_read_assertions": dict(design["input_read_assertions"]),
        "scope_limits": [
            "request choice is a myopic structural model, not an empirical prevalence estimate",
            "delivery equals available capability up to the award; strategic withholding is outside scope",
            (
                "verified post-event available-export telemetry equals simulation truth "
                "componentwise; telemetry error, latency, and dropout are outside scope"
            ),
            "simultaneous requests are independent myopic choices, not a Nash equilibrium",
            "the replay is a next-interval operational diagnostic and never rewrites the completed interval",
            "the synthetic branch ratings certify only the frozen scenario-defined operating limits",
        ],
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    v15._write_csv(output_path.with_name("V16_INTERVAL_DATA.csv"), steps)
    v15._write_csv(output_path.with_name("V16_TRAJECTORY_SUMMARY.csv"), trajectories)
    v15._write_csv(output_path.with_name("V16_AGGREGATE_OVERALL.csv"), aggregate_overall)
    v15._write_csv(
        output_path.with_name("V16_PROFILE_METHOD_SUMMARY.csv"),
        aggregate_by_profile_method,
    )
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# V16 externality-conditioned feedback experiment",
            "",
            f"- Status: `{result['status']}`",
            f"- Trajectories: `{len(trajectories)}`; intervals: `{len(steps)}`.",
            f"- Replays attempted: `{len(attempted_replays)}`; gates opened: `{len(gate_steps)}`.",
            "- The gate requires outsider access release and same-realization delivery recovery.",
            "- The previous delivery-ratio filter is retained only as an ablation.",
            f"- Gates: `{gates}`",
            f"- Diagnostics: `{diagnostics}`",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "fresh_source_design"
    out = root / "outputs" / "externality_conditioned_feedback_v16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-design", action="store_true")
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument(
        "--policy", type=Path,
        default=root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml",
    )
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument(
        "--behavior", type=Path,
        default=root / "outputs" / "endogenous_request_choice_v14" / "ENDOGENOUS_REQUEST_CHOICE_RESULTS.json",
    )
    parser.add_argument(
        "--dynamic-design", type=Path,
        default=root / "outputs" / "dynamic_feedback_v12_3" / "DYNAMIC_FEEDBACK_DESIGN_MANIFEST.json",
    )
    parser.add_argument("--design", type=Path, default=out / "V16_DESIGN.json")
    parser.add_argument("--output", type=Path, default=out / "V16_RESULTS.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    if args.freeze_design:
        design = _freeze_design(
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
        "replay_attempt_count": result["replay_attempt_count"],
        "replay_gate_open_count": result["replay_gate_open_count"],
        "gates": result["gate_results"],
        "diagnostics": result["scientific_diagnostics"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_ROSTER",
    "EXTERNALITY_CONTROLS",
    "_freeze_design",
    "run_experiment",
]
