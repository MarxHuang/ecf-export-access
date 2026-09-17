"""Endogenize prosumer requests under private capability uncertainty.

The participant chooses a request multiplier before its deliverable capability
is realized.  A five-point probability tree links the capability forecast and
the realized output; request and delivery are therefore not independent random
curves.  For each frozen request-to-allocation response cell, the participant
maximizes expected delivered-energy value net of a non-delivery charge.

The experiment is a myopic best response with other requests fixed.  It does
not claim a Nash equilibrium, truthful reporting, or an empirically calibrated
behavioral distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from r4r.serialization import canonical_dumps, canonical_hash


SERIALIZATION_ID = "r4r.endogenous_request_choice_under_uncertainty.v1"
STANDARDIZED_SUPPORT = (-2.0, -1.0, 0.0, 1.0, 2.0)
STANDARDIZED_PROBABILITY = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)
FORECAST_CV_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
PENALTY_RATIO_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0)
BEHAVIOR_GAMMA_GRID = tuple(round(1.0 + 0.1 * index, 1) for index in range(11))
REALIZED_CV_MULTIPLIERS = (0.5, 1.0, 1.5)
REALIZED_BIAS_GRID = (-0.10, 0.0, 0.10)
TOL = 2.0e-10


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def _tree(
    *, forecast_mw: float, capacity_mw: float, cv: float, bias: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    if forecast_mw < 0 or capacity_mw < 0 or cv < 0:
        raise ValueError("forecast, capacity, and coefficient of variation must be nonnegative")
    values = []
    for z, probability in zip(STANDARDIZED_SUPPORT, STANDARDIZED_PROBABILITY):
        factor = max(0.0, 1.0 + float(bias) + float(cv) * float(z))
        values.append((min(float(capacity_mw), factor * float(forecast_mw)), probability))
    return tuple(values)


def _expected_outcome(
    *, allocation: Sequence[float], reference_allocation: Sequence[float],
    capability_trees: Sequence[Sequence[tuple[float, float]]],
    reporter_index: int, penalty_ratio: float,
) -> dict[str, float]:
    if not (len(allocation) == len(reference_allocation) == len(capability_trees)):
        raise ValueError("allocation and capability-tree vectors must have equal length")
    outside = [index for index in range(len(allocation)) if index != reporter_index]
    delivery: list[float] = []
    reference_delivery: list[float] = []
    for award, reference_award, tree in zip(
        allocation, reference_allocation, capability_trees
    ):
        if not math.isclose(
            math.fsum(float(probability) for _, probability in tree),
            1.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("every capability tree must be normalized")
        delivery.append(math.fsum(
            float(probability) * min(float(award), float(available))
            for available, probability in tree
        ))
        reference_delivery.append(math.fsum(
            float(probability) * min(float(reference_award), float(available))
            for available, probability in tree
        ))
    idle = [
        max(float(award) - float(realized), 0.0)
        for award, realized in zip(allocation, delivery)
    ]
    reference_idle = [
        max(float(award) - float(realized), 0.0)
        for award, realized in zip(reference_allocation, reference_delivery)
    ]
    own_delivery_gain = delivery[reporter_index] - reference_delivery[reporter_index]
    outside_delivery_loss = math.fsum(
        reference_delivery[index] - delivery[index] for index in outside
    )
    total_delivery = math.fsum(delivery)
    reference_total_delivery = math.fsum(reference_delivery)
    own_idle = idle[reporter_index]
    reference_own_idle = reference_idle[reporter_index]
    private_utility = delivery[reporter_index] - float(penalty_ratio) * own_idle
    reference_private_utility = (
        reference_delivery[reporter_index]
        - float(penalty_ratio) * reference_own_idle
    )
    accumulator: dict[str, float] = {
        "expected_private_utility_mw_equivalent": private_utility,
        "expected_reference_private_utility_mw_equivalent": reference_private_utility,
        "expected_private_utility_gain_mw_equivalent": (
            private_utility - reference_private_utility
        ),
        "expected_own_delivery_mw": delivery[reporter_index],
        "expected_reference_own_delivery_mw": reference_delivery[reporter_index],
        "expected_own_delivery_gain_mw": own_delivery_gain,
        "expected_own_idle_award_mw": own_idle,
        "expected_reference_own_idle_award_mw": reference_own_idle,
        "expected_incremental_own_idle_award_mw": own_idle - reference_own_idle,
        "expected_outside_delivery_loss_mw": outside_delivery_loss,
        "expected_total_delivery_mw": total_delivery,
        "expected_reference_total_delivery_mw": reference_total_delivery,
        "expected_total_delivery_change_mw": total_delivery - reference_total_delivery,
        "expected_system_delivery_loss_mw": max(
            reference_total_delivery - total_delivery, 0.0
        ),
        "expected_total_idle_award_mw": math.fsum(idle),
        "expected_reference_total_idle_award_mw": math.fsum(reference_idle),
    }
    accumulator["expected_external_delivery_loss_mw"] = (
        accumulator["expected_outside_delivery_loss_mw"]
        - accumulator["expected_own_delivery_gain_mw"]
    )
    accumulator["expected_delivery_plus_idle_identity_error_mw"] = abs(
        accumulator["expected_total_delivery_mw"]
        + accumulator["expected_total_idle_award_mw"]
        - math.fsum(float(value) for value in allocation)
    )
    return dict(accumulator)


def _classify_choice(row: Mapping[str, Any]) -> str:
    gamma = float(row["selected_gamma"])
    private_gain = float(row["expected_private_utility_gain_mw_equivalent"])
    system_change = float(row["expected_total_delivery_change_mw"])
    idle_change = float(row["expected_incremental_own_idle_award_mw"])
    if gamma <= 1.0 + TOL:
        return "REFERENCE_REQUEST"
    if private_gain <= TOL:
        return "HIGH_REQUEST_WITHOUT_STRICT_PRIVATE_GAIN"
    if idle_change > TOL and system_change < -TOL:
        return "ENDOGENOUS_PRIVATE_GAIN_WITH_DELIVERY_EXTERNALITY"
    if system_change >= -TOL:
        return "PRODUCTIVE_HIGH_REQUEST"
    return "HIGH_REQUEST_WITH_SYSTEM_LOSS_NO_IDLE_INCREMENT"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_choice_rows(
    rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {**dimensions, "cell_count": len(rows)}
    result.update({
        "mean_selected_gamma": _mean(row["selected_gamma"] for row in rows),
        "inflated_request_cell_share": _mean(
            float(row["selected_gamma"]) > 1.0 + TOL for row in rows
        ),
        "strict_private_gain_cell_share": _mean(
            float(row["expected_private_utility_gain_mw_equivalent"]) > TOL
            for row in rows
        ),
        "endogenous_externality_cell_share": _mean(
            row["choice_regime"]
            == "ENDOGENOUS_PRIVATE_GAIN_WITH_DELIVERY_EXTERNALITY"
            for row in rows
        ),
        "productive_high_request_cell_share": _mean(
            row["choice_regime"] == "PRODUCTIVE_HIGH_REQUEST"
            for row in rows
        ),
        "mean_private_utility_gain_mw_equivalent": _mean(
            row["expected_private_utility_gain_mw_equivalent"] for row in rows
        ),
        "mean_own_delivery_gain_mw": _mean(
            row["expected_own_delivery_gain_mw"] for row in rows
        ),
        "mean_outside_delivery_loss_mw": _mean(
            row["expected_outside_delivery_loss_mw"] for row in rows
        ),
        "mean_total_delivery_change_mw": _mean(
            row["expected_total_delivery_change_mw"] for row in rows
        ),
        "mean_system_delivery_loss_mw": _mean(
            row["expected_system_delivery_loss_mw"] for row in rows
        ),
        "mean_incremental_own_idle_award_mw": _mean(
            row["expected_incremental_own_idle_award_mw"] for row in rows
        ),
    })
    return result


def analyze(
    *, response_path: Path, source_path: Path, output_path: Path,
) -> dict[str, Any]:
    response = _load(response_path)
    source = _load(source_path)
    response_hash = _validated_hash(
        response, field="result_hash", label="location-robust response",
    )
    source_hash = _validated_hash(source, field="result_hash", label="source envelope")
    if response.get("status") != "LOCATION_ROBUST_REQUEST_RESPONSE_COMPLETE":
        raise ValueError("location-robust request response is incomplete")
    if response.get("source_envelope_hash") != source_hash:
        raise ValueError("response/source binding mismatch")
    if not all(response.get("gate_results", {}).values()):
        raise ValueError("location-robust response contains an open gate")

    evaluation_ids = list(source["scenario_split"]["evaluation"])
    source_scenario = next(
        row for row in source["scenarios"] if row["scenario_id"] == evaluation_ids[0]
    )
    participant_ids = [str(value) for value in source["participant_ids"]]
    capacity = [float(value) for value in source_scenario["capacity_mw"]]
    reference_capability = [
        float(value) for value in source_scenario["reference_request_mw"]
    ]
    reference_records = response["reference_records"]
    side_lookup = {
        (row["scenario_id"], row["q_mode"], row["method_id"]): row
        for row in response["reported_side_records"]
    }
    pair_lookup = {
        (row["reporter_id"], float(row["gamma"]), row["q_mode"], row["method_id"]): row
        for row in response["pairs"]
    }
    if len(pair_lookup) != len(response["pairs"]):
        raise ValueError("duplicate response pair")

    outcome_rows: list[dict[str, Any]] = []
    cell_rows: dict[tuple[str, str, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for pair in response["pairs"]:
        reporter_id = str(pair["reporter_id"])
        reporter_index = participant_ids.index(reporter_id)
        gamma = float(pair["gamma"])
        if not any(math.isclose(gamma, candidate, abs_tol=1.0e-12)
                   for candidate in BEHAVIOR_GAMMA_GRID):
            # The source response surface also carries a legacy 1.74 stress
            # point.  It remains available for conditional sensitivity, but a
            # nonuniform extra point must not receive an artificial advantage
            # in the endogenous best-response choice set.
            continue
        q_mode = str(pair["q_mode"])
        method_id = str(pair["method_id"])
        scenario_id = str(pair["scenario_id"])
        allocation = [
            float(value) for value in side_lookup[
                (scenario_id, q_mode, method_id)
            ]["allocation_mw"]
        ]
        reference_allocation = [
            float(value)
            for value in reference_records[f"{q_mode}::{method_id}"]["allocation_mw"]
        ]
        for cv in FORECAST_CV_GRID:
            capability_trees = tuple(
                _tree(forecast_mw=forecast, capacity_mw=cap, cv=cv)
                for forecast, cap in zip(reference_capability, capacity)
            )
            for penalty_ratio in PENALTY_RATIO_GRID:
                outcome = _expected_outcome(
                    allocation=allocation,
                    reference_allocation=reference_allocation,
                    capability_trees=capability_trees,
                    reporter_index=reporter_index,
                    penalty_ratio=penalty_ratio,
                )
                row = {
                    "reporter_id": reporter_id,
                    "reporter_bus_id": int(pair["reporter_bus_id"]),
                    "q_mode": q_mode,
                    "method_id": method_id,
                    "forecast_cv": float(cv),
                    "penalty_to_export_value_ratio": float(penalty_ratio),
                    "gamma": gamma,
                    "matched_q_pair_pass": bool(pair["matched_q_pair_pass"]),
                    **outcome,
                }
                outcome_rows.append(row)
                cell_rows[(reporter_id, q_mode, method_id, cv, penalty_ratio)].append(row)

    choice_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    for key, rows in sorted(cell_rows.items()):
        reporter_id, q_mode, method_id, cv, penalty_ratio = key
        ordered = sorted(rows, key=lambda row: float(row["gamma"]))
        maximum = max(float(row["expected_private_utility_mw_equivalent"]) for row in ordered)
        selected = min(
            (
                row for row in ordered
                if float(row["expected_private_utility_mw_equivalent"])
                >= maximum - TOL
            ),
            key=lambda row: float(row["gamma"]),
        )
        choice = {
            "reporter_id": reporter_id,
            "reporter_bus_id": int(selected["reporter_bus_id"]),
            "q_mode": q_mode,
            "method_id": method_id,
            "forecast_cv": float(cv),
            "penalty_to_export_value_ratio": float(penalty_ratio),
            "selected_gamma": float(selected["gamma"]),
            "selection_rule": "SMALLEST_GAMMA_WITHIN_TOLERANCE_OF_EXPECTED_UTILITY_MAXIMUM",
            **{
                field: selected[field]
                for field in selected
                if field.startswith("expected_")
            },
        }
        choice["choice_regime"] = _classify_choice(choice)
        choice_rows.append(choice)
        for previous, current in zip(ordered, ordered[1:]):
            marginal_rows.append({
                "reporter_id": reporter_id,
                "q_mode": q_mode,
                "method_id": method_id,
                "forecast_cv": float(cv),
                "penalty_to_export_value_ratio": float(penalty_ratio),
                "gamma_from": float(previous["gamma"]),
                "gamma_to": float(current["gamma"]),
                "private_utility_increment_mw_equivalent": (
                    float(current["expected_private_utility_mw_equivalent"])
                    - float(previous["expected_private_utility_mw_equivalent"])
                ),
                "own_delivery_increment_mw": (
                    float(current["expected_own_delivery_mw"])
                    - float(previous["expected_own_delivery_mw"])
                ),
                "system_delivery_increment_mw": (
                    float(current["expected_total_delivery_mw"])
                    - float(previous["expected_total_delivery_mw"])
                ),
                "private_positive_system_negative": bool(
                    float(current["expected_private_utility_mw_equivalent"])
                    - float(previous["expected_private_utility_mw_equivalent"]) > TOL
                    and float(current["expected_total_delivery_mw"])
                    - float(previous["expected_total_delivery_mw"]) < -TOL
                ),
            })

    aggregate_rows = [
        _aggregate_choice_rows(
            [
                row for row in choice_rows
                if math.isclose(float(row["forecast_cv"]), cv, abs_tol=1.0e-15)
                and math.isclose(
                    float(row["penalty_to_export_value_ratio"]), penalty,
                    abs_tol=1.0e-15,
                )
            ],
            {
                "forecast_cv": float(cv),
                "penalty_to_export_value_ratio": float(penalty),
            },
        )
        for cv in FORECAST_CV_GRID
        for penalty in PENALTY_RATIO_GRID
    ]
    by_method_rows = [
        _aggregate_choice_rows(
            [
                row for row in choice_rows
                if row["method_id"] == method_id
                and math.isclose(float(row["forecast_cv"]), cv, abs_tol=1.0e-15)
                and math.isclose(
                    float(row["penalty_to_export_value_ratio"]), penalty,
                    abs_tol=1.0e-15,
                )
            ],
            {
                "method_id": method_id,
                "forecast_cv": float(cv),
                "penalty_to_export_value_ratio": float(penalty),
            },
        )
        for method_id in response["method_roster"]
        for cv in FORECAST_CV_GRID
        for penalty in PENALTY_RATIO_GRID
    ]

    # Evaluate selected requests under a distribution shift.  These outcomes
    # never feed back into request selection.
    shifted_accumulator: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for choice in choice_rows:
        reporter_id = str(choice["reporter_id"])
        reporter_index = participant_ids.index(reporter_id)
        q_mode = str(choice["q_mode"])
        method_id = str(choice["method_id"])
        gamma = float(choice["selected_gamma"])
        pair = pair_lookup[(reporter_id, gamma, q_mode, method_id)]
        allocation = [
            float(value) for value in side_lookup[
                (pair["scenario_id"], q_mode, method_id)
            ]["allocation_mw"]
        ]
        reference_allocation = [
            float(value)
            for value in reference_records[f"{q_mode}::{method_id}"]["allocation_mw"]
        ]
        for cv_multiplier in REALIZED_CV_MULTIPLIERS:
            for bias in REALIZED_BIAS_GRID:
                trees = tuple(
                    _tree(
                        forecast_mw=forecast,
                        capacity_mw=cap,
                        cv=float(choice["forecast_cv"]) * float(cv_multiplier),
                        bias=float(bias),
                    )
                    for forecast, cap in zip(reference_capability, capacity)
                )
                outcome = _expected_outcome(
                    allocation=allocation,
                    reference_allocation=reference_allocation,
                    capability_trees=trees,
                    reporter_index=reporter_index,
                    penalty_ratio=float(choice["penalty_to_export_value_ratio"]),
                )
                shifted_accumulator[(
                    float(choice["forecast_cv"]),
                    float(choice["penalty_to_export_value_ratio"]),
                    method_id,
                    float(cv_multiplier),
                    float(bias),
                )].append(outcome)
    shifted_rows = []
    for key, rows in sorted(shifted_accumulator.items()):
        assumed_cv, penalty, method_id, cv_multiplier, bias = key
        shifted_rows.append({
            "assumed_forecast_cv": assumed_cv,
            "penalty_to_export_value_ratio": penalty,
            "method_id": method_id,
            "realized_cv_multiplier": cv_multiplier,
            "realized_mean_bias": bias,
            "cell_count": len(rows),
            "mean_private_utility_gain_mw_equivalent": _mean(
                row["expected_private_utility_gain_mw_equivalent"] for row in rows
            ),
            "mean_total_delivery_change_mw": _mean(
                row["expected_total_delivery_change_mw"] for row in rows
            ),
            "mean_system_delivery_loss_mw": _mean(
                row["expected_system_delivery_loss_mw"] for row in rows
            ),
            "mean_incremental_own_idle_award_mw": _mean(
                row["expected_incremental_own_idle_award_mw"] for row in rows
            ),
        })

    expected_outcomes = (
        len(participant_ids) * len(response["q_roster"])
        * len(response["method_roster"]) * len(FORECAST_CV_GRID)
        * len(PENALTY_RATIO_GRID) * len(BEHAVIOR_GAMMA_GRID)
    )
    expected_choices = (
        len(participant_ids) * len(response["q_roster"])
        * len(response["method_roster"]) * len(FORECAST_CV_GRID)
        * len(PENALTY_RATIO_GRID)
    )
    tree_mean = math.fsum(
        z * p for z, p in zip(STANDARDIZED_SUPPORT, STANDARDIZED_PROBABILITY)
    )
    tree_variance = math.fsum(
        z * z * p for z, p in zip(STANDARDIZED_SUPPORT, STANDARDIZED_PROBABILITY)
    )
    low_penalty_rows = [
        row for row in choice_rows
        if math.isclose(
            float(row["penalty_to_export_value_ratio"]), min(PENALTY_RATIO_GRID),
            abs_tol=1.0e-15,
        )
    ]
    high_penalty_rows = [
        row for row in choice_rows
        if math.isclose(
            float(row["penalty_to_export_value_ratio"]), max(PENALTY_RATIO_GRID),
            abs_tol=1.0e-15,
        )
    ]
    low_penalty_externality_share = _mean(
        row["choice_regime"]
        == "ENDOGENOUS_PRIVATE_GAIN_WITH_DELIVERY_EXTERNALITY"
        for row in low_penalty_rows
    )
    high_penalty_externality_share = _mean(
        row["choice_regime"]
        == "ENDOGENOUS_PRIVATE_GAIN_WITH_DELIVERY_EXTERNALITY"
        for row in high_penalty_rows
    )
    gates = {
        "RESPONSE_AND_SOURCE_BINDING": response["source_envelope_hash"] == source_hash,
        "ALL_LOCATION_RESPONSE_GATES_CLOSED": all(response["gate_results"].values()),
        "PROBABILITY_TREE_NORMALIZED": math.isclose(
            math.fsum(STANDARDIZED_PROBABILITY), 1.0, abs_tol=1.0e-15
        ),
        "PROBABILITY_TREE_ZERO_MEAN_UNIT_VARIANCE": (
            abs(tree_mean) <= 1.0e-15 and abs(tree_variance - 1.0) <= 1.0e-15
        ),
        "EXPECTED_OUTCOME_GRID_COMPLETE": len(outcome_rows) == expected_outcomes,
        "ENDOGENOUS_CHOICE_GRID_COMPLETE": len(choice_rows) == expected_choices,
        "ALL_REPORTER_LOCATIONS_INCLUDED": {
            row["reporter_id"] for row in choice_rows
        } == set(participant_ids),
        "UTILITY_SELECTION_IS_DETERMINISTIC": all(
            row["selection_rule"]
            == "SMALLEST_GAMMA_WITHIN_TOLERANCE_OF_EXPECTED_UTILITY_MAXIMUM"
            for row in choice_rows
        ),
        "DELIVERY_IDLE_ACCOUNTING_CLOSED": max(
            float(row["expected_delivery_plus_idle_identity_error_mw"])
            for row in outcome_rows
        ) <= 2.0e-9,
        "MATCHED_Q_AC_EVIDENCE_CARRIED_FORWARD": all(
            bool(row["matched_q_pair_pass"]) for row in outcome_rows
        ),
        "PRODUCTIVE_HIGH_REQUEST_REGION_IDENTIFIED": any(
            row["choice_regime"] == "PRODUCTIVE_HIGH_REQUEST"
            for row in choice_rows
        ),
        "PRIVATE_GAIN_SYSTEM_LOSS_REGION_IDENTIFIED": any(
            row["choice_regime"]
            == "ENDOGENOUS_PRIVATE_GAIN_WITH_DELIVERY_EXTERNALITY"
            for row in choice_rows
        ),
        "STRONG_CONSEQUENCE_BOUNDARY_REDUCES_EXTERNALITY_SHARE": (
            max(PENALTY_RATIO_GRID) >= 10.0
            and high_penalty_externality_share < low_penalty_externality_share
        ),
        "REFERENCE_AND_INFLATED_CHOICES_BOTH_PRESENT": (
            any(float(row["selected_gamma"]) <= 1.0 + TOL for row in choice_rows)
            and any(float(row["selected_gamma"]) > 1.0 + TOL for row in choice_rows)
        ),
        "BEHAVIOR_CHOICE_GRID_IS_UNIFORM": (
            list(BEHAVIOR_GAMMA_GRID)
            == [round(1.0 + 0.1 * index, 1) for index in range(11)]
            and set(BEHAVIOR_GAMMA_GRID).issubset({
                float(value) for value in response["gamma_grid"]
            })
        ),
        "OUT_OF_SAMPLE_SHIFT_EXCLUDED_FROM_SELECTION": True,
        "NO_EMPIRICAL_PREVALENCE_OR_EQUILIBRIUM_CLAIM": True,
    }
    result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": (
            "ENDOGENOUS_REQUEST_CHOICE_EXPERIMENT_COMPLETE"
            if all(gates.values())
            else "ENDOGENOUS_REQUEST_CHOICE_EXPERIMENT_BLOCKED"
        ),
        "scientific_role": "BEHAVIOR_GENERATION_AND_DELIVERY_EXTERNALITY_EVIDENCE",
        "response_result_hash": response_hash,
        "source_envelope_hash": source_hash,
        "behavioral_contract": {
            "timing": "private capability distribution -> submitted request -> network allocation -> realized capability -> metered delivery",
            "availability_tree": "Q_i=min(C_i,(1+sigma*z)qhat_i), z in {-2,-1,0,1,2}, probabilities {1,4,6,4,1}/16",
            "request_choice": "gamma maximizes E[min(x_i(gamma),Q_i)-theta*(x_i(gamma)-Q_i)^+] on the frozen grid",
            "delivery": "y_j=min(x_j,Q_j); every participant uses the same forecast-centered capability-tree family while only the focal participant changes its request",
            "system_externality": "the reporter ignores delivery changes induced through other participants' allocations",
            "theta": "non-delivery penalty divided by delivered-export value",
        },
        "scope_limits": [
            "myopic one-participant best response with other requests fixed; all participants retain controlled delivery uncertainty",
            "controlled uncertainty tree, not an empirical prevalence estimate",
            "not a Nash-equilibrium or incentive-compatibility claim",
            "out-of-sample distribution shifts evaluate but never select requests",
            "fixed-gamma multi-participant trajectories remain conditional stress tests",
        ],
        "standardized_support": list(STANDARDIZED_SUPPORT),
        "standardized_probability": list(STANDARDIZED_PROBABILITY),
        "forecast_cv_grid": list(FORECAST_CV_GRID),
        "penalty_to_export_value_ratio_grid": list(PENALTY_RATIO_GRID),
        "realized_cv_multipliers": list(REALIZED_CV_MULTIPLIERS),
        "realized_bias_grid": list(REALIZED_BIAS_GRID),
        "reporter_roster": participant_ids,
        "q_roster": list(response["q_roster"]),
        "method_roster": list(response["method_roster"]),
        "gamma_grid": list(BEHAVIOR_GAMMA_GRID),
        "source_response_gamma_grid": list(response["gamma_grid"]),
        "gate_results": gates,
        "choice_cells": choice_rows,
        "choice_aggregate": aggregate_rows,
        "choice_by_method": by_method_rows,
        "marginal_externality_cells": marginal_rows,
        "out_of_sample_shift_summary": shifted_rows,
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_name("ENDOGENOUS_REQUEST_CHOICE_CELLS.csv"), choice_rows)
    _write_csv(output_path.with_name("ENDOGENOUS_REQUEST_CHOICE_AGGREGATE.csv"), aggregate_rows)
    _write_csv(output_path.with_name("ENDOGENOUS_REQUEST_CHOICE_BY_METHOD.csv"), by_method_rows)
    _write_csv(output_path.with_name("ENDOGENOUS_MARGINAL_EXTERNALITY_CELLS.csv"), marginal_rows)
    _write_csv(output_path.with_name("ENDOGENOUS_OUT_OF_SAMPLE_SHIFT_SUMMARY.csv"), shifted_rows)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# Endogenous request choice under private capability uncertainty",
            "",
            f"- Status: `{result['status']}`",
            f"- Reporter locations: `{len(participant_ids)}`.",
            f"- Endogenous choice cells: `{len(choice_rows)}`.",
            "- Requests maximize an exact expected payoff over the uniform 1.0:0.1:2.0 choice grid; the legacy 1.74 response point is excluded from selection.",
            "- Realized capability uses the same forecast-linked probability tree as the request choice.",
            "- No claim is made about empirical prevalence, Nash equilibrium, or incentive compatibility.",
            f"- Gates: `{gates}`",
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
        "--response", type=Path,
        default=(
            root / "outputs" / "endogenous_request_choice_v14"
            / "LOCATION_ROBUST_REQUEST_RESPONSE.json"
        ),
    )
    parser.add_argument(
        "--source", type=Path,
        default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            root / "outputs" / "endogenous_request_choice_v14"
            / "ENDOGENOUS_REQUEST_CHOICE_RESULTS.json"
        ),
    )
    args = parser.parse_args(argv)
    result = analyze(
        response_path=args.response,
        source_path=args.source,
        output_path=args.output,
    )
    print(canonical_dumps({
        "status": result["status"],
        "choice_cell_count": len(result["choice_cells"]),
        "gates": result["gate_results"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORECAST_CV_GRID",
    "PENALTY_RATIO_GRID",
    "STANDARDIZED_PROBABILITY",
    "STANDARDIZED_SUPPORT",
    "_classify_choice",
    "_expected_outcome",
    "_tree",
    "analyze",
]
