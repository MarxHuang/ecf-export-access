"""Derive paired evidence contrasts from the frozen V16 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from r4r.serialization import canonical_dumps, canonical_hash


SERIALIZATION_ID = "r4r.externality_conditioned_feedback_evidence.v1"
CONTROL_IDS = (
    "NO_FEEDBACK",
    "DELIVERY_RATIO_CF_BETA_050",
    "EXTERNALITY_CF_BETA_025",
    "EXTERNALITY_CF_BETA_050",
    "EXTERNALITY_CF_BETA_100",
    "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
)
PAIR_KEYS = [
    "scenario_id",
    "profile_id",
    "penalty_to_export_value_ratio",
    "q_mode",
    "method_id",
]
PRIMARY_CONTROL = "EXTERNALITY_CF_BETA_050"
LEGACY_ABLATION = "DELIVERY_RATIO_CF_BETA_050"
UNIFORM_CONTROL = "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050"
REPLAY_TOLERANCE_MW = 2.0e-10


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validated_hash(value: Mapping[str, Any], *, field: str, label: str) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    actual = canonical_hash(body)
    if not isinstance(declared, str) or declared != actual:
        raise ValueError(f"{label} self-hash mismatch")
    return actual


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(frame: pd.DataFrame, field: str) -> float:
    return float(frame[field].mean())


def _bool_series(frame: pd.DataFrame, field: str) -> pd.Series:
    return frame[field].astype(str).str.lower().eq("true")


def analyze(
    *, result_path: Path, trajectory_path: Path, interval_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source = _load(result_path)
    source_hash = _validated_hash(
        source, field="result_hash", label="V16 experiment",
    )
    if source.get("status") != "EXTERNALITY_CONDITIONED_FEEDBACK_EXPERIMENT_COMPLETE":
        raise ValueError("V16 experiment is incomplete")
    if not all(source.get("gate_results", {}).values()):
        raise ValueError("V16 experiment contains an open hard gate")
    trajectories = pd.read_csv(trajectory_path, encoding="utf-8-sig")
    intervals = pd.read_csv(interval_path, encoding="utf-8-sig")

    duplicate_count = int(trajectories.duplicated(PAIR_KEYS + ["control_id"]).sum())
    control_sets = trajectories.groupby(PAIR_KEYS)["control_id"].agg(set)
    complete_control_cells = bool(
        len(control_sets) > 0
        and all(values == set(CONTROL_IDS) for values in control_sets)
    )
    pivot_fields = [
        "mean_total_delivery_change_mw",
        "mean_system_delivery_loss_mw",
        "externality_interval_share",
        "productive_high_request_interval_share",
        "mean_reported_total_idle_award_mw",
        "mean_request_attenuation_mw",
        "mean_redistribution_mw",
        "mean_jain_norm_change",
        "externality_gate_interval_share",
        "mean_replay_total_delivery_gain_mw",
    ]
    wide = trajectories.pivot(
        index=PAIR_KEYS, columns="control_id", values=pivot_fields,
    ).reset_index()

    penalty_rows: list[dict[str, Any]] = []
    no_feedback = trajectories[trajectories["control_id"] == "NO_FEEDBACK"]
    for penalty, group in no_feedback.groupby("penalty_to_export_value_ratio"):
        penalty_rows.append({
            "penalty_to_export_value_ratio": float(penalty),
            "trajectory_count": int(len(group)),
            "mean_selected_gamma": _mean(group, "mean_selected_gamma"),
            "externality_interval_share": _mean(group, "externality_interval_share"),
            "productive_high_request_interval_share": _mean(
                group, "productive_high_request_interval_share",
            ),
            "mean_private_utility_gain_kw_equivalent": 1000.0 * _mean(
                group, "mean_private_utility_gain_mw_equivalent",
            ),
            "mean_coalition_delivery_gain_kw": 1000.0 * _mean(
                group, "mean_coalition_delivery_gain_mw",
            ),
            "mean_outside_delivery_loss_kw": 1000.0 * _mean(
                group, "mean_outside_delivery_loss_mw",
            ),
            "mean_signed_feeder_delivery_change_kw": 1000.0 * _mean(
                group, "mean_total_delivery_change_mw",
            ),
            "mean_positive_part_delivery_loss_kw": 1000.0 * _mean(
                group, "mean_system_delivery_loss_mw",
            ),
        })

    control_rows: list[dict[str, Any]] = []
    for penalty in sorted(trajectories["penalty_to_export_value_ratio"].unique()):
        group = wide[wide[("penalty_to_export_value_ratio", "")] == float(penalty)]
        for control_id in CONTROL_IDS[1:]:
            control_rows.append({
                "penalty_to_export_value_ratio": float(penalty),
                "control_id": control_id,
                "pair_count": int(len(group)),
                "delivery_improvement_vs_no_feedback_kw": 1000.0 * float((
                    group[("mean_total_delivery_change_mw", control_id)]
                    - group[("mean_total_delivery_change_mw", "NO_FEEDBACK")]
                ).mean()),
                "positive_part_loss_reduction_kw": 1000.0 * float((
                    group[("mean_system_delivery_loss_mw", "NO_FEEDBACK")]
                    - group[("mean_system_delivery_loss_mw", control_id)]
                ).mean()),
                "externality_share_reduction_percentage_points": 100.0 * float((
                    group[("externality_interval_share", "NO_FEEDBACK")]
                    - group[("externality_interval_share", control_id)]
                ).mean()),
                "idle_award_reduction_kw": 1000.0 * float((
                    group[("mean_reported_total_idle_award_mw", "NO_FEEDBACK")]
                    - group[("mean_reported_total_idle_award_mw", control_id)]
                ).mean()),
                "mean_request_attenuation_kw": 1000.0 * float(
                    group[("mean_request_attenuation_mw", control_id)].mean()
                ),
                "mean_redistribution_change_kw": 1000.0 * float((
                    group[("mean_redistribution_mw", control_id)]
                    - group[("mean_redistribution_mw", "NO_FEEDBACK")]
                ).mean()),
                "mean_normalized_jain_change_from_reference": float(
                    group[("mean_jain_norm_change", control_id)].mean()
                ),
                "gate_interval_share": float(
                    group[("externality_gate_interval_share", control_id)].mean()
                ),
            })

    paired_rows: list[dict[str, Any]] = []
    paired_distribution_rows: list[dict[str, Any]] = []
    for penalty in sorted(trajectories["penalty_to_export_value_ratio"].unique()):
        group = wide[wide[("penalty_to_export_value_ratio", "")] == float(penalty)]
        for comparator, comparison_id in (
            (LEGACY_ABLATION, "PRIMARY_VS_DELIVERY_ONLY_ABLATION"),
            (UNIFORM_CONTROL, "PRIMARY_VS_EQUAL_ATTENUATION_UNIFORM"),
        ):
            paired_rows.append({
                "penalty_to_export_value_ratio": float(penalty),
                "comparison_id": comparison_id,
                "primary_control_id": PRIMARY_CONTROL,
                "comparator_control_id": comparator,
                "pair_count": int(len(group)),
                "primary_delivery_advantage_kw": 1000.0 * float((
                    group[("mean_total_delivery_change_mw", PRIMARY_CONTROL)]
                    - group[("mean_total_delivery_change_mw", comparator)]
                ).mean()),
                "primary_positive_part_loss_advantage_kw": 1000.0 * float((
                    group[("mean_system_delivery_loss_mw", comparator)]
                    - group[("mean_system_delivery_loss_mw", PRIMARY_CONTROL)]
                ).mean()),
                "primary_externality_share_advantage_percentage_points": 100.0 * float((
                    group[("externality_interval_share", comparator)]
                    - group[("externality_interval_share", PRIMARY_CONTROL)]
                ).mean()),
                "primary_idle_award_advantage_kw": 1000.0 * float((
                    group[("mean_reported_total_idle_award_mw", comparator)]
                    - group[("mean_reported_total_idle_award_mw", PRIMARY_CONTROL)]
                ).mean()),
                "primary_redistribution_reduction_kw": 1000.0 * float((
                    group[("mean_redistribution_mw", comparator)]
                    - group[("mean_redistribution_mw", PRIMARY_CONTROL)]
                ).mean()),
                "attenuation_difference_kw": 1000.0 * float((
                    group[("mean_request_attenuation_mw", PRIMARY_CONTROL)]
                    - group[("mean_request_attenuation_mw", comparator)]
                ).mean()),
                "maximum_absolute_attenuation_mismatch_w": 1.0e6 * float((
                    group[("mean_request_attenuation_mw", PRIMARY_CONTROL)]
                    - group[("mean_request_attenuation_mw", comparator)]
                ).abs().max()),
            })
            delivery_difference = 1000.0 * (
                group[("mean_total_delivery_change_mw", PRIMARY_CONTROL)]
                - group[("mean_total_delivery_change_mw", comparator)]
            )
            tolerance_kw = 1.0e-6
            paired_distribution_rows.append({
                "penalty_to_export_value_ratio": float(penalty),
                "comparison_id": comparison_id,
                "pair_count": int(len(delivery_difference)),
                "mean_primary_delivery_advantage_kw": float(delivery_difference.mean()),
                "median_primary_delivery_advantage_kw": float(delivery_difference.median()),
                "lower_quartile_primary_delivery_advantage_kw": float(delivery_difference.quantile(0.25)),
                "upper_quartile_primary_delivery_advantage_kw": float(delivery_difference.quantile(0.75)),
                "primary_win_share": float((delivery_difference > tolerance_kw).mean()),
                "tie_share": float((delivery_difference.abs() <= tolerance_kw).mean()),
                "primary_loss_share": float((delivery_difference < -tolerance_kw).mean()),
            })

    beta_rows: list[dict[str, Any]] = []
    for penalty in sorted(trajectories["penalty_to_export_value_ratio"].unique()):
        group = trajectories[
            (trajectories["penalty_to_export_value_ratio"] == float(penalty))
            & trajectories["control_id"].isin([
                "EXTERNALITY_CF_BETA_025",
                "EXTERNALITY_CF_BETA_050",
                "EXTERNALITY_CF_BETA_100",
            ])
        ]
        for control_id, cell in group.groupby("control_id"):
            beta_rows.append({
                "penalty_to_export_value_ratio": float(penalty),
                "control_id": str(control_id),
                "beta": float(cell["beta"].iloc[0]),
                "trajectory_count": int(len(cell)),
                "mean_signed_feeder_delivery_change_kw": 1000.0 * _mean(
                    cell, "mean_total_delivery_change_mw",
                ),
                "externality_interval_share": _mean(
                    cell, "externality_interval_share",
                ),
                "mean_request_attenuation_kw": 1000.0 * _mean(
                    cell, "mean_request_attenuation_mw",
                ),
                "gate_interval_share": _mean(
                    cell, "externality_gate_interval_share",
                ),
                "terminal_mean_access_ceiling_ratio": _mean(
                    cell, "terminal_mean_score",
                ),
            })

    dimension_rows: list[dict[str, Any]] = []
    nominal = wide[wide[("penalty_to_export_value_ratio", "")] == 0.0]
    for dimension in ("method_id", "profile_id", "scenario_id", "q_mode"):
        for value, group in nominal.groupby((dimension, "")):
            dimension_rows.append({
                "dimension": dimension,
                "value": str(value),
                "pair_count": int(len(group)),
                "primary_delivery_improvement_vs_no_feedback_kw": 1000.0 * float((
                    group[("mean_total_delivery_change_mw", PRIMARY_CONTROL)]
                    - group[("mean_total_delivery_change_mw", "NO_FEEDBACK")]
                ).mean()),
                "primary_advantage_over_delivery_only_kw": 1000.0 * float((
                    group[("mean_total_delivery_change_mw", PRIMARY_CONTROL)]
                    - group[("mean_total_delivery_change_mw", LEGACY_ABLATION)]
                ).mean()),
                "primary_advantage_over_uniform_kw": 1000.0 * float((
                    group[("mean_total_delivery_change_mw", PRIMARY_CONTROL)]
                    - group[("mean_total_delivery_change_mw", UNIFORM_CONTROL)]
                ).mean()),
                "externality_share_reduction_percentage_points": 100.0 * float((
                    group[("externality_interval_share", "NO_FEEDBACK")]
                    - group[("externality_interval_share", PRIMARY_CONTROL)]
                ).mean()),
                "gate_interval_share": float(
                    group[("externality_gate_interval_share", PRIMARY_CONTROL)].mean()
                ),
            })

    ecf_intervals = intervals[
        intervals["control_id"].isin([
            "EXTERNALITY_CF_BETA_025",
            "EXTERNALITY_CF_BETA_050",
            "EXTERNALITY_CF_BETA_100",
        ])
    ].copy()
    gate_reason_rows = []
    for (control_id, reason), group in ecf_intervals.groupby(
        ["control_id", "replay_gate_reason"], dropna=False,
    ):
        gate_reason_rows.append({
            "control_id": str(control_id),
            "gate_reason": str(reason),
            "interval_count": int(len(group)),
            "interval_share_within_control": float(
                len(group) / len(ecf_intervals[ecf_intervals["control_id"] == control_id])
            ),
            "mean_idle_award_kw": 1000.0 * _mean(
                group, "replay_total_idle_award_mw",
            ),
            "mean_outsider_access_release_kw": 1000.0 * _mean(
                group, "replay_outsider_allocation_gain_mw",
            ),
            "mean_outsider_delivery_recovery_kw": 1000.0 * _mean(
                group, "replay_outsider_delivery_gain_mw",
            ),
            "mean_total_delivery_recovery_kw": 1000.0 * _mean(
                group, "replay_total_delivery_gain_mw",
            ),
        })

    interval_wide = intervals.pivot(
        index=PAIR_KEYS + ["interval"],
        columns="control_id",
        values="request_attenuation_mw",
    )
    attenuation_mismatch = float((
        interval_wide[PRIMARY_CONTROL] - interval_wide[UNIFORM_CONTROL]
    ).abs().max())
    gate_mask = _bool_series(ecf_intervals, "replay_gate_triggered")
    gate_rows = ecf_intervals[gate_mask]
    no_gate_rows = ecf_intervals[~gate_mask]
    gate_materiality_rows: list[dict[str, Any]] = []
    for threshold_kw in (0.0, 0.1, 1.0, 5.0):
        threshold_mw = threshold_kw / 1000.0
        gate_materiality_rows.append({
            "total_delivery_recovery_threshold_kw": threshold_kw,
            "gate_interval_count": int(len(gate_rows)),
            "gate_interval_share_meeting_threshold": float(
                (gate_rows["replay_total_delivery_gain_mw"] > threshold_mw).mean()
            ) if len(gate_rows) else 0.0,
            "median_gate_total_delivery_recovery_kw": (
                1000.0 * float(gate_rows["replay_total_delivery_gain_mw"].median())
                if len(gate_rows) else 0.0
            ),
            "lower_quartile_gate_total_delivery_recovery_kw": (
                1000.0 * float(gate_rows["replay_total_delivery_gain_mw"].quantile(0.25))
                if len(gate_rows) else 0.0
            ),
            "upper_quartile_gate_total_delivery_recovery_kw": (
                1000.0 * float(gate_rows["replay_total_delivery_gain_mw"].quantile(0.75))
                if len(gate_rows) else 0.0
            ),
        })

    classification_sensitivity_rows: list[dict[str, Any]] = []
    for control_id in ("NO_FEEDBACK", PRIMARY_CONTROL):
        control = intervals[intervals["control_id"] == control_id]
        for threshold_kw in (0.0, 0.1, 1.0, 5.0):
            threshold_mw = threshold_kw / 1000.0
            classified = (
                (control["coalition_private_utility_gain_mw_equivalent"] > threshold_mw)
                & (control["total_delivery_change_mw"] < -threshold_mw)
                & (control["incremental_coalition_idle_award_mw"] > threshold_mw)
            )
            classification_sensitivity_rows.append({
                "control_id": control_id,
                "materiality_threshold_kw": threshold_kw,
                "interval_count": int(len(control)),
                "externality_interval_share": float(classified.mean()),
            })
    gates = {
        "SOURCE_RESULT_SELF_HASH_VALID": bool(source_hash),
        "SOURCE_HARD_GATES_ALL_CLOSED": all(source["gate_results"].values()),
        "TRAJECTORY_COUNT_MATCHES_SOURCE": len(trajectories) == int(source["trajectory_count"]),
        "INTERVAL_COUNT_MATCHES_SOURCE": len(intervals) == int(source["interval_record_count"]),
        "NO_DUPLICATE_TRAJECTORY_CONTROL_KEYS": duplicate_count == 0,
        "EVERY_FACTOR_CELL_HAS_SIX_CONTROLS": complete_control_cells,
        "ALL_METHODS_Q_PROFILES_SCENARIOS_PRESENT": (
            len(set(trajectories["method_id"])) == 5
            and len(set(trajectories["q_mode"])) == 2
            and len(set(trajectories["profile_id"])) == 3
            and len(set(trajectories["scenario_id"])) == 3
        ),
        "INTERVAL_LEVEL_EQUAL_ATTENUATION_IDENTITY": attenuation_mismatch <= 2.0e-10,
        "GATE_OPEN_AND_CLOSED_REGIMES_PRESENT": len(gate_rows) > 0 and len(no_gate_rows) > 0,
        "MISMATCH_WITHOUT_GATE_IS_OBSERVED": bool(
            source.get("scientific_diagnostics", {}).get("mismatch_without_gate_exists")
        ),
        "PROPOSED_CONTROL_DIFFERS_FROM_DELIVERY_ONLY_ABLATION": bool(
            source.get("scientific_diagnostics", {}).get(
                "proposed_and_delivery_only_ablation_differ"
            )
        ),
        "EVERY_OPEN_GATE_HAS_POSITIVE_USABLE_RELEASE": bool(
            len(gate_rows) > 0
            and (gate_rows["replay_outsider_allocation_gain_mw"] > REPLAY_TOLERANCE_MW).all()
            and (gate_rows["replay_outsider_delivery_gain_mw"] > REPLAY_TOLERANCE_MW).all()
            and (gate_rows["replay_total_delivery_gain_mw"] > REPLAY_TOLERANCE_MW).all()
        ),
        "PRODUCTIVE_AND_EXTERNALITY_REGIMES_BOTH_PRESENT": bool(
            _bool_series(intervals, "productive_high_request").any()
            and _bool_series(intervals, "delivery_externality").any()
        ),
        "BETA_SENSITIVITY_HAS_THREE_PREDECLARED_LEVELS": set(beta_rows[index]["beta"] for index in range(len(beta_rows))) == {0.25, 0.5, 1.0},
    }
    result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": (
            "EXTERNALITY_CONDITIONED_FEEDBACK_EVIDENCE_COMPLETE"
            if all(gates.values())
            else "EXTERNALITY_CONDITIONED_FEEDBACK_EVIDENCE_BLOCKED"
        ),
        "source_result_hash": source_hash,
        "gate_results": gates,
        "penalty_behavior_effect": penalty_rows,
        "control_contrast": control_rows,
        "paired_primary_contrast": paired_rows,
        "paired_primary_distribution": paired_distribution_rows,
        "beta_sensitivity": beta_rows,
        "dimension_contrast_theta_0": dimension_rows,
        "gate_reason_summary": gate_reason_rows,
        "gate_materiality_sensitivity": gate_materiality_rows,
        "externality_classification_sensitivity": classification_sensitivity_rows,
        "interpretation_boundaries": [
            "the private capability model structures behavior but is not empirically estimated",
            "metered delivery is non-discretionary capability realization; strategic withholding is not claimed",
            "the replay diagnoses an observed allocation externality and only changes the next interval",
            "the delivery-ratio filter is an ablation, not the proposed mechanism",
            "equal attenuation separates targeting from the amount of total request reduction",
            "signed delivery change and positive-part delivery loss answer different questions",
        ],
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_name("V16_PENALTY_BEHAVIOR_EFFECT.csv"), penalty_rows)
    _write_csv(output_path.with_name("V16_CONTROL_CONTRAST.csv"), control_rows)
    _write_csv(output_path.with_name("V16_PAIRED_PRIMARY_CONTRAST.csv"), paired_rows)
    _write_csv(
        output_path.with_name("V16_PAIRED_PRIMARY_DISTRIBUTION.csv"),
        paired_distribution_rows,
    )
    _write_csv(output_path.with_name("V16_BETA_SENSITIVITY.csv"), beta_rows)
    _write_csv(output_path.with_name("V16_DIMENSION_CONTRAST_THETA_0.csv"), dimension_rows)
    _write_csv(output_path.with_name("V16_GATE_REASON_SUMMARY.csv"), gate_reason_rows)
    _write_csv(
        output_path.with_name("V16_GATE_MATERIALITY_SENSITIVITY.csv"),
        gate_materiality_rows,
    )
    _write_csv(
        output_path.with_name("V16_EXTERNALITY_CLASSIFICATION_SENSITIVITY.csv"),
        classification_sensitivity_rows,
    )
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# V16 behavior and counterfactual-feedback evidence",
            "",
            f"- Status: `{result['status']}`",
            f"- Source result hash: `{source_hash}`",
            "- Every primary comparison is paired by scenario, temporal profile, consequence ratio, Q setting, and allocation rule.",
            "- The equal-attenuation identity is checked at every interval.",
            "- Gate-open rows must show positive outsider access release and same-realization delivery recovery.",
            f"- Gates: `{gates}`",
            f"- Result hash: `{result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "externality_conditioned_feedback_v16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=out / "V16_RESULTS.json")
    parser.add_argument("--trajectories", type=Path, default=out / "V16_TRAJECTORY_SUMMARY.csv")
    parser.add_argument("--intervals", type=Path, default=out / "V16_INTERVAL_DATA.csv")
    parser.add_argument("--output", type=Path, default=out / "V16_EVIDENCE_SUMMARY.json")
    args = parser.parse_args(argv)
    result = analyze(
        result_path=args.result,
        trajectory_path=args.trajectories,
        interval_path=args.intervals,
        output_path=args.output,
    )
    print(canonical_dumps({
        "status": result["status"],
        "gates": result["gate_results"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
