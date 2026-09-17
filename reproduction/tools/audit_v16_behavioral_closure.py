"""Audit the behavioral closure behind the V16 dynamic experiment.

This audit performs no allocation, optimization, or AC solve.  It binds the
frozen V14 request-choice surface to the frozen V16 dynamic records and checks
that request and delivery are generated from one conditional capability model.
It also derives two manuscript-facing comparisons:

* the consequence-only behavioral benchmark versus nominal ECF; and
* the stepwise consequence ratio that would equate a participant's private
  payoff increment with the feeder-delivery increment.

The latter is a finite-difference diagnostic, not a tariff recommendation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from r4r.serialization.hashing import canonical_hash


SERIALIZATION_ID = "r4r.v16_behavioral_closure_audit.v1"
TOLERANCE_MW = 2.0e-10
PRIMARY_CONTROL = "EXTERNALITY_CF_BETA_050"
CONSEQUENCE_ONLY_CONTROL = "NO_FEEDBACK"
PAIR_KEYS = (
    "scenario_id",
    "profile_id",
    "penalty_to_export_value_ratio",
    "q_mode",
    "method_id",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validated_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload[field] = None
    actual = canonical_hash(payload)
    if expected != actual:
        raise ValueError(f"{label} {field} mismatch: {expected} != {actual}")
    return expected


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty evidence table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _all_close(left: Iterable[float], right: Iterable[float], tolerance: float = 2.0e-12) -> bool:
    a = list(left)
    b = list(right)
    return len(a) == len(b) and all(
        abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b)
    )


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return math.fsum(materialized) / len(materialized) if materialized else math.nan


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
            raise ValueError(f"duplicate V14 choice cell: {key}")
        lookup[key] = row
    return lookup


def _stepwise_consequence_equivalent(
    behavior: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return material externality steps and robust summary rows.

    For a fixed request step, the own idle-award increment is recovered from
    the difference between the theta=0 and theta=1 private-payoff increments.
    The stepwise consequence-equivalent ratio is

        - outside-delivery increment / own-idle increment.

    Only steps with positive own delivery, positive idle increment, and a
    negative feeder-delivery increment above the registered numerical tolerance
    are included.
    """

    frame = pd.DataFrame(behavior["marginal_externality_cells"])
    keys = [
        "reporter_id", "q_mode", "method_id", "forecast_cv",
        "gamma_from", "gamma_to",
    ]
    theta0 = frame[frame["penalty_to_export_value_ratio"] == 0.0].set_index(keys)
    theta1 = frame[frame["penalty_to_export_value_ratio"] == 1.0].set_index(keys)
    joined = theta0.join(
        theta1[["private_utility_increment_mw_equivalent"]],
        rsuffix="_theta1",
        how="inner",
    ).reset_index()
    joined["idle_award_increment_mw"] = (
        joined["own_delivery_increment_mw"]
        - joined["private_utility_increment_mw_equivalent_theta1"]
    )
    joined["outside_delivery_increment_mw"] = (
        joined["system_delivery_increment_mw"]
        - joined["own_delivery_increment_mw"]
    )
    selected = joined[
        (joined["own_delivery_increment_mw"] > TOLERANCE_MW)
        & (joined["system_delivery_increment_mw"] < -TOLERANCE_MW)
        & (joined["idle_award_increment_mw"] > TOLERANCE_MW)
    ].copy()
    selected["stepwise_consequence_equivalent_ratio"] = (
        -selected["outside_delivery_increment_mw"]
        / selected["idle_award_increment_mw"]
    )
    selected = selected[
        selected["stepwise_consequence_equivalent_ratio"].map(math.isfinite)
        & (selected["stepwise_consequence_equivalent_ratio"] > 0.0)
    ].copy()
    selected["own_delivery_increment_kw"] = 1000.0 * selected["own_delivery_increment_mw"]
    selected["outside_delivery_decrement_kw"] = -1000.0 * selected["outside_delivery_increment_mw"]
    selected["feeder_delivery_increment_kw"] = 1000.0 * selected["system_delivery_increment_mw"]
    selected["idle_award_increment_kw"] = 1000.0 * selected["idle_award_increment_mw"]

    raw_fields = [
        "reporter_id", "q_mode", "method_id", "forecast_cv", "gamma_from", "gamma_to",
        "own_delivery_increment_kw", "outside_delivery_decrement_kw",
        "feeder_delivery_increment_kw", "idle_award_increment_kw",
        "stepwise_consequence_equivalent_ratio",
    ]
    raw_rows = selected[raw_fields].sort_values(keys).to_dict(orient="records")

    def summary(scope: str, value: str, cell: pd.DataFrame) -> dict[str, Any]:
        ratio = cell["stepwise_consequence_equivalent_ratio"]
        return {
            "scope": scope,
            "value": value,
            "step_count": int(len(cell)),
            "lower_quartile_ratio": float(ratio.quantile(0.25)),
            "median_ratio": float(ratio.median()),
            "upper_quartile_ratio": float(ratio.quantile(0.75)),
            "fifth_percentile_ratio": float(ratio.quantile(0.05)),
            "ninety_fifth_percentile_ratio": float(ratio.quantile(0.95)),
            "mean_idle_award_increment_kw": float(cell["idle_award_increment_kw"].mean()),
            "mean_outside_delivery_decrement_kw": float(cell["outside_delivery_decrement_kw"].mean()),
        }

    summary_rows = [summary("overall", "all", selected)]
    for field in ("method_id", "q_mode", "forecast_cv"):
        for value, cell in selected.groupby(field, sort=True):
            summary_rows.append(summary(field, str(value), cell))
    return raw_rows, summary_rows


def _stage_comparison(trajectories: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = {
        "mean_selected_gamma": "mean_selected_gamma",
        "changed_set_delivery_gain_kw": "mean_coalition_delivery_gain_mw",
        "outside_delivery_loss_kw": "mean_outside_delivery_loss_mw",
        "signed_feeder_delivery_change_kw": "mean_total_delivery_change_mw",
        "positive_part_feeder_loss_kw": "mean_system_delivery_loss_mw",
        "externality_interval_percent": "externality_interval_share",
        "productive_high_request_interval_percent": "productive_high_request_interval_share",
        "idle_award_kw": "mean_reported_total_idle_award_mw",
        "request_attenuation_kw": "mean_request_attenuation_mw",
    }
    selected = trajectories[
        trajectories["control_id"].isin([CONSEQUENCE_ONLY_CONTROL, PRIMARY_CONTROL])
    ]
    rows: list[dict[str, Any]] = []
    for (theta, control_id), cell in selected.groupby(
        ["penalty_to_export_value_ratio", "control_id"], sort=True,
    ):
        row: dict[str, Any] = {
            "penalty_to_export_value_ratio": float(theta),
            "control_id": str(control_id),
            "stage_label": (
                "CONSEQUENCE_ONLY_NO_METERED_FEEDBACK"
                if control_id == CONSEQUENCE_ONLY_CONTROL
                else "CONSEQUENCE_PLUS_NOMINAL_ECF"
            ),
            "trajectory_count": int(len(cell)),
        }
        for output, source in fields.items():
            value = float(cell[source].mean())
            if output.endswith("_kw"):
                value *= 1000.0
            elif output.endswith("_percent"):
                value *= 100.0
            row[output] = value
        rows.append(row)

    by_key = {
        (float(row["penalty_to_export_value_ratio"]), str(row["control_id"])): row
        for row in rows
    }
    zero = by_key[(0.0, CONSEQUENCE_ONLY_CONTROL)]
    contrasts: list[dict[str, Any]] = []
    for theta in sorted({float(row["penalty_to_export_value_ratio"]) for row in rows}):
        consequence = by_key[(theta, CONSEQUENCE_ONLY_CONTROL)]
        ecf = by_key[(theta, PRIMARY_CONTROL)]
        contrasts.append({
            "comparison": "CONSEQUENCE_ONLY_VS_ZERO_CONSEQUENCE",
            "penalty_to_export_value_ratio": theta,
            "comparator_penalty_to_export_value_ratio": 0.0,
            "feeder_delivery_improvement_kw": (
                consequence["signed_feeder_delivery_change_kw"]
                - zero["signed_feeder_delivery_change_kw"]
            ),
            "changed_set_delivery_change_kw": (
                consequence["changed_set_delivery_gain_kw"]
                - zero["changed_set_delivery_gain_kw"]
            ),
            "outside_delivery_loss_reduction_kw": (
                zero["outside_delivery_loss_kw"]
                - consequence["outside_delivery_loss_kw"]
            ),
            "externality_share_reduction_percentage_points": (
                zero["externality_interval_percent"]
                - consequence["externality_interval_percent"]
            ),
        })
        contrasts.append({
            "comparison": "NOMINAL_ECF_VS_SAME_CONSEQUENCE_ONLY",
            "penalty_to_export_value_ratio": theta,
            "comparator_penalty_to_export_value_ratio": theta,
            "feeder_delivery_improvement_kw": (
                ecf["signed_feeder_delivery_change_kw"]
                - consequence["signed_feeder_delivery_change_kw"]
            ),
            "changed_set_delivery_change_kw": (
                ecf["changed_set_delivery_gain_kw"]
                - consequence["changed_set_delivery_gain_kw"]
            ),
            "outside_delivery_loss_reduction_kw": (
                consequence["outside_delivery_loss_kw"]
                - ecf["outside_delivery_loss_kw"]
            ),
            "externality_share_reduction_percentage_points": (
                consequence["externality_interval_percent"]
                - ecf["externality_interval_percent"]
            ),
        })
    return rows, contrasts


def audit(
    *, behavior_path: Path, design_path: Path, result_path: Path,
    trajectory_path: Path, output_path: Path,
) -> dict[str, Any]:
    behavior = _load(behavior_path)
    design = _load(design_path)
    result = _load(result_path)
    behavior_hash = _validated_hash(behavior, "result_hash", "V14 behavior")
    design_hash = _validated_hash(design, "design_hash", "V16 design")
    result_hash = _validated_hash(result, "result_hash", "V16 result")
    if behavior.get("status") != "ENDOGENOUS_REQUEST_CHOICE_EXPERIMENT_COMPLETE":
        raise ValueError("V14 endogenous request-choice experiment is incomplete")
    if design.get("status") != "EXTERNALITY_CONDITIONED_FEEDBACK_DESIGN_FROZEN":
        raise ValueError("V16 design is not frozen")
    if result.get("status") != "EXTERNALITY_CONDITIONED_FEEDBACK_EXPERIMENT_COMPLETE":
        raise ValueError("V16 experiment is incomplete")
    if not all(result.get("gate_results", {}).values()):
        raise ValueError("V16 source experiment contains an open gate")

    choices = _choice_lookup(behavior)
    scenarios = {str(row["scenario_id"]): row for row in design["scenarios"]}
    participant_ids = [str(value) for value in design["scenarios"][0]["participant_ids"]]
    participant_index = {participant_id: index for index, participant_id in enumerate(participant_ids)}

    raw_request_identity = True
    choice_identity = True
    capability_identity = True
    delivery_identity = True
    control_raw_request_identity = True
    checked_behavior_steps = 0
    checked_delivery_steps = 0
    raw_by_pair: dict[tuple[Any, ...], dict[int, tuple[float, ...]]] = defaultdict(dict)

    for trajectory in result["trajectory_summaries"]:
        pair = tuple(trajectory[field] for field in PAIR_KEYS)
        control_id = str(trajectory["control_id"])
        scenario = scenarios[str(trajectory["scenario_id"])]
        reference = [float(value) for value in scenario["reference_request_mw"]]
        capacity = [float(value) for value in scenario["capacity_mw"]]
        coalition = [str(value) for value in scenario["coalition_ids"]]
        theta = float(trajectory["penalty_to_export_value_ratio"])
        q_mode = str(trajectory["q_mode"])
        method_id = str(trajectory["method_id"])
        for step in trajectory["steps"]:
            allocation = [float(value) for value in step["allocation_mw"]]
            capability = [float(value) for value in step["capability_mw"]]
            delivery = [float(value) for value in step["delivery_mw"]]
            delivery_identity &= _all_close(
                delivery,
                [min(award, available) for award, available in zip(allocation, capability)],
            )
            checked_delivery_steps += 1
            raw = tuple(float(value) for value in step["raw_request_mw"])
            interval = int(step["interval"])
            if interval in raw_by_pair[pair]:
                control_raw_request_identity &= _all_close(raw, raw_by_pair[pair][interval])
            else:
                raw_by_pair[pair][interval] = raw
            if control_id != CONSEQUENCE_ONLY_CONTROL:
                continue
            selected = {str(key): float(value) for key, value in step["selected_gamma_by_participant"].items()}
            expected_raw = list(reference)
            for participant_id in coalition:
                key = (participant_id, q_mode, method_id, float(design["forecast_cv"]), theta)
                choice = choices[key]
                gamma = float(choice["selected_gamma"])
                choice_identity &= (
                    participant_id in selected
                    and abs(selected[participant_id] - gamma) <= 2.0e-12
                )
                expected_raw[participant_index[participant_id]] *= gamma
            raw_request_identity &= _all_close(raw, expected_raw)

            support = [int(value) for value in step["capability_support_by_participant"]]
            expected_capability = [
                min(cap, max(0.0, 1.0 + float(design["forecast_cv"]) * z) * forecast)
                for cap, forecast, z in zip(capacity, reference, support)
            ]
            capability_identity &= _all_close(capability, expected_capability)
            checked_behavior_steps += 1

    trajectories = pd.read_csv(trajectory_path, encoding="utf-8-sig")
    stage_rows, contrast_rows = _stage_comparison(trajectories)
    equivalent_rows, equivalent_summary = _stepwise_consequence_equivalent(behavior)
    overall_equivalent = next(
        row for row in equivalent_summary if row["scope"] == "overall"
    )
    method_medians = [
        float(row["median_ratio"])
        for row in equivalent_summary if row["scope"] == "method_id"
    ]

    consequence_rows = [
        row for row in stage_rows if row["control_id"] == CONSEQUENCE_ONLY_CONTROL
    ]
    consequence_rows.sort(key=lambda row: float(row["penalty_to_export_value_ratio"]))
    gamma_values = [float(row["mean_selected_gamma"]) for row in consequence_rows]
    gates = {
        "SOURCE_HASHES_AND_STATUSES_VALID": bool(
            design["behavioral_choice_result_hash"] == behavior_hash
            and result["behavioral_choice_result_hash"] == behavior_hash
            and result["design_hash"] == design_hash
        ),
        "SELECTED_GAMMA_MATCHES_FROZEN_EXPECTED_PAYOFF_CHOICE": choice_identity,
        "RAW_REQUEST_EQUALS_SELECTED_GAMMA_TIMES_SHARED_FORECAST": raw_request_identity,
        "REALIZED_CAPABILITY_EQUALS_SHARED_CONDITIONAL_TREE": capability_identity,
        "METERED_DELIVERY_EQUALS_MINIMUM_OF_AWARD_AND_CAPABILITY": delivery_identity,
        "RAW_REQUEST_IS_IDENTICAL_ACROSS_PAIRED_CONTROLS": control_raw_request_identity,
        "CONSEQUENCE_ONLY_CONTROL_HAS_NO_METERED_ATTENUATION": bool(
            trajectories[
                trajectories["control_id"] == CONSEQUENCE_ONLY_CONTROL
            ]["mean_request_attenuation_mw"].abs().max() <= 2.0e-12
        ),
        "CONSEQUENCE_RESPONSE_IS_MONOTONE_ON_TESTED_DYNAMIC_LEVELS": all(
            left + 2.0e-12 >= right for left, right in zip(gamma_values, gamma_values[1:])
        ) and gamma_values[0] > gamma_values[-1] + 2.0e-12,
        "PRODUCTIVE_AND_EXTERNALITY_REGIMES_BOTH_RETAINED": bool(
            trajectories["productive_high_request_interval_share"].gt(0.0).any()
            and trajectories["externality_interval_share"].gt(0.0).any()
        ),
        "STEPWISE_EXTERNALITY_EQUIVALENT_RATIO_IS_DEFINED_ON_MATERIAL_CELLS": bool(
            equivalent_rows
            and float(overall_equivalent["lower_quartile_ratio"]) > 0.0
            and float(overall_equivalent["upper_quartile_ratio"])
            > float(overall_equivalent["lower_quartile_ratio"])
        ),
        "STEPWISE_EQUIVALENT_RATIO_VARIES_ACROSS_ALLOCATION_RULES": bool(
            len(method_medians) == len(design["method_roster"])
            and max(method_medians) - min(method_medians) > 1.0e-6
        ),
        "NO_NEW_SOLVER_OR_AC_EXECUTION": True,
    }
    diagnostics = {
        "checked_behavior_step_count": checked_behavior_steps,
        "checked_delivery_step_count": checked_delivery_steps,
        "stepwise_externality_cell_count": len(equivalent_rows),
        "stepwise_equivalent_ratio_median": float(overall_equivalent["median_ratio"]),
        "stepwise_equivalent_ratio_lower_quartile": float(overall_equivalent["lower_quartile_ratio"]),
        "stepwise_equivalent_ratio_upper_quartile": float(overall_equivalent["upper_quartile_ratio"]),
        "stepwise_equivalent_method_median_range": [min(method_medians), max(method_medians)],
        "interpretation": (
            "A fixed own-shortfall consequence can alter endogenous requests, but the "
            "stepwise ratio that matches the cross-participant delivery effect varies "
            "with the allocation response. ECF is evaluated only for residual realized "
            "mismatch and does not replace settlement."
        ),
    }

    audit_result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": (
            "V16_BEHAVIORAL_CLOSURE_AUDIT_COMPLETE"
            if all(gates.values())
            else "V16_BEHAVIORAL_CLOSURE_AUDIT_BLOCKED"
        ),
        "scientific_role": "BEHAVIOR_REQUEST_DELIVERY_AND_SETTLEMENT_COMPLEMENTARITY_EVIDENCE",
        "behavior_result_hash": behavior_hash,
        "design_hash": design_hash,
        "experiment_result_hash": result_hash,
        "gate_results": gates,
        "diagnostics": diagnostics,
        "stage_comparison": stage_rows,
        "stage_contrasts": contrast_rows,
        "stepwise_consequence_equivalent_summary": equivalent_summary,
        "scope_limits": [
            "The request choice is a myopic expected-payoff response, not a Nash equilibrium or field-prevalence estimate.",
            "The consequence ratio is normalized by delivered-export value and is not a proposed tariff.",
            "The stepwise consequence-equivalent ratio is a finite-difference diagnostic and can be unstable when the idle-award increment approaches zero.",
            "Delivery is capability limited; strategic withholding is not modeled.",
            "The audit performs no new allocation, optimization, AC solve, or parameter selection.",
        ],
        "audit_hash": None,
    }
    audit_result["audit_hash"] = canonical_hash(audit_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_path.with_name("V16_SETTLEMENT_FEEDBACK_STAGE_COMPARISON.csv"), stage_rows)
    _write_csv(output_path.with_name("V16_SETTLEMENT_FEEDBACK_STAGE_CONTRASTS.csv"), contrast_rows)
    _write_csv(output_path.with_name("V16_STEPWISE_CONSEQUENCE_EQUIVALENT.csv"), equivalent_rows)
    _write_csv(output_path.with_name("V16_STEPWISE_CONSEQUENCE_EQUIVALENT_SUMMARY.csv"), equivalent_summary)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# V16 behavioral-closure audit",
            "",
            f"- Status: `{audit_result['status']}`.",
            f"- Behavior-linked steps checked: `{checked_behavior_steps}`.",
            f"- Delivery identities checked: `{checked_delivery_steps}`.",
            "- Request choice, raw request, capability realization, and metered delivery are bound to one conditional capability model.",
            "- The no-feedback rows at positive consequence ratios are the consequence-only behavioral benchmark.",
            f"- Stepwise externality cells: `{len(equivalent_rows)}`; consequence-equivalent ratio median `{overall_equivalent['median_ratio']:.3f}`, IQR `{overall_equivalent['lower_quartile_ratio']:.3f}--{overall_equivalent['upper_quartile_ratio']:.3f}`.",
            "- The ratio is diagnostic rather than a tariff recommendation; nominal ECF addresses residual realized mismatch after the ex-ante consequence response.",
            "",
        ]),
        encoding="utf-8",
    )
    return audit_result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    dynamic = root / "outputs" / "externality_conditioned_feedback_v16"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior", type=Path,
        default=root / "outputs" / "endogenous_request_choice_v14" / "ENDOGENOUS_REQUEST_CHOICE_RESULTS.json",
    )
    parser.add_argument("--design", type=Path, default=dynamic / "V16_DESIGN.json")
    parser.add_argument("--result", type=Path, default=dynamic / "V16_RESULTS.json")
    parser.add_argument("--trajectories", type=Path, default=dynamic / "V16_TRAJECTORY_SUMMARY.csv")
    parser.add_argument("--output", type=Path, default=dynamic / "V16_BEHAVIORAL_CLOSURE_AUDIT.json")
    args = parser.parse_args(argv)
    result = audit(
        behavior_path=args.behavior,
        design_path=args.design,
        result_path=args.result,
        trajectory_path=args.trajectories,
        output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "audit_hash": result["audit_hash"],
        "gate_results": result["gate_results"],
    }))
    return 0 if result["status"] == "V16_BEHAVIORAL_CLOSURE_AUDIT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
