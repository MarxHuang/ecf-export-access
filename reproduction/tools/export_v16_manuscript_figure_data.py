"""Export publication-facing CSV data for the V16 manuscript figures."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from r4r.serialization import canonical_dumps, canonical_hash


PRIMARY_CONTROL = "EXTERNALITY_CF_BETA_050"
CONTROL_ORDER = (
    "NO_FEEDBACK",
    "DELIVERY_RATIO_CF_BETA_050",
    "EXTERNALITY_CF_BETA_025",
    PRIMARY_CONTROL,
    "EXTERNALITY_CF_BETA_100",
    "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
)
METHOD_ORDER = (
    "weighted_proportional",
    "equal_kw_reduction",
    "flat_level",
    "max_export_lp",
    "fairness_qp",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    declared = body.get(field)
    body[field] = None
    actual = canonical_hash(body)
    if not isinstance(declared, str) or declared != actual:
        raise ValueError(f"invalid {field}")
    return actual


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
    return path


def _sequence(value: Any) -> list[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("serialized vector is not a sequence")
    return [float(item) for item in value]


def _mapping(value: Any) -> dict[str, float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError("serialized mapping is not an object")
    return {str(key): float(item) for key, item in value.items()}


def _mean(values: Iterable[float]) -> float:
    sequence = [float(value) for value in values]
    return math.fsum(sequence) / len(sequence) if sequence else 0.0


def _utility_curve(behavior: Mapping[str, Any]) -> pd.DataFrame:
    reporter = "p021"
    q_mode = "Q0"
    method = "weighted_proportional"
    forecast_cv = 0.20
    penalties = (0.0, 1.0, 10.0)
    rows: list[dict[str, Any]] = []
    choices = {
        (
            str(row["reporter_id"]), str(row["q_mode"]), str(row["method_id"]),
            float(row["forecast_cv"]), float(row["penalty_to_export_value_ratio"]),
        ): row
        for row in behavior["choice_cells"]
    }
    marginals: dict[tuple[float, float], Mapping[str, Any]] = {}
    for row in behavior["marginal_externality_cells"]:
        if (
            str(row["reporter_id"]) == reporter
            and str(row["q_mode"]) == q_mode
            and str(row["method_id"]) == method
            and math.isclose(float(row["forecast_cv"]), forecast_cv, abs_tol=1.0e-12)
            and float(row["penalty_to_export_value_ratio"]) in penalties
        ):
            marginals[(
                float(row["penalty_to_export_value_ratio"]),
                float(row["gamma_to"]),
            )] = row
    for penalty in penalties:
        choice = choices[(reporter, q_mode, method, forecast_cv, penalty)]
        reference = float(choice["expected_reference_private_utility_mw_equivalent"])
        utility = reference
        rows.append({
            "reporter_id": reporter,
            "q_mode": q_mode,
            "method_id": method,
            "forecast_cv": forecast_cv,
            "penalty_to_export_value_ratio": penalty,
            "gamma": 1.0,
            "expected_private_utility_gain_kw_equivalent": 0.0,
            "selected_gamma": float(choice["selected_gamma"]),
        })
        for gamma in [round(1.0 + 0.1 * index, 1) for index in range(1, 11)]:
            marginal = marginals[(penalty, gamma)]
            utility += float(marginal["private_utility_increment_mw_equivalent"])
            rows.append({
                "reporter_id": reporter,
                "q_mode": q_mode,
                "method_id": method,
                "forecast_cv": forecast_cv,
                "penalty_to_export_value_ratio": penalty,
                "gamma": gamma,
                "expected_private_utility_gain_kw_equivalent": 1000.0 * (utility - reference),
                "selected_gamma": float(choice["selected_gamma"]),
            })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("penalty_to_export_value_ratio")[
        "expected_private_utility_gain_kw_equivalent"
    ]
    lower = grouped.transform("min")
    upper = grouped.transform("max")
    span = upper - lower
    frame["within_consequence_payoff_score"] = (
        frame["expected_private_utility_gain_kw_equivalent"] - lower
    ) / span.where(span > 1.0e-12, 1.0)
    return frame


def _behavior_tables(behavior: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    aggregate = pd.DataFrame(behavior["choice_aggregate"])
    aggregate["externality_cell_percent"] = 100.0 * aggregate["endogenous_externality_cell_share"]
    aggregate["productive_high_request_percent"] = 100.0 * aggregate["productive_high_request_cell_share"]
    aggregate["reference_or_other_percent"] = 100.0 * (
        1.0
        - aggregate["endogenous_externality_cell_share"]
        - aggregate["productive_high_request_cell_share"]
    )
    aggregate["mean_own_delivery_gain_kw"] = 1000.0 * aggregate["mean_own_delivery_gain_mw"]
    aggregate["mean_outside_delivery_loss_kw"] = 1000.0 * aggregate["mean_outside_delivery_loss_mw"]
    aggregate["mean_signed_feeder_delivery_change_kw"] = 1000.0 * aggregate["mean_total_delivery_change_mw"]

    by_method = pd.DataFrame(behavior["choice_by_method"])
    by_method["mean_own_delivery_gain_kw"] = 1000.0 * by_method["mean_own_delivery_gain_mw"]
    by_method["mean_outside_delivery_loss_kw"] = 1000.0 * by_method["mean_outside_delivery_loss_mw"]
    by_method["mean_signed_feeder_delivery_change_kw"] = 1000.0 * by_method["mean_total_delivery_change_mw"]
    method_nominal = by_method[
        (by_method["forecast_cv"] == 0.20)
        & (by_method["penalty_to_export_value_ratio"] == 0.0)
    ].copy()

    tree = pd.DataFrame({
        "standardized_capability_node": behavior["standardized_support"],
        "probability": behavior["standardized_probability"],
    })
    tree["capability_multiplier_at_sigma_020"] = (
        1.0 + 0.20 * tree["standardized_capability_node"]
    )
    return {
        "Fig03_behavior_surface": aggregate,
        "Fig03_behavior_method_theta0_sigma020": method_nominal,
        "Fig03_representative_utility_curve": _utility_curve(behavior),
        "Fig03_capability_tree": tree,
    }


def _trace_table(
    intervals: pd.DataFrame, design: Mapping[str, Any],
) -> pd.DataFrame:
    subset = intervals[
        (intervals["scenario_id"] == "MR_K05_LOCAL")
        & (intervals["profile_id"] == "SHOCK_AND_RECOVERY")
        & (intervals["penalty_to_export_value_ratio"] == 0.0)
        & (intervals["q_mode"] == "Q0")
        & (intervals["method_id"] == "weighted_proportional")
        & intervals["control_id"].isin([
            "NO_FEEDBACK", "DELIVERY_RATIO_CF_BETA_050", PRIMARY_CONTROL,
            "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
        ])
    ].copy()
    participants = [str(item) for item in design["scenarios"][0]["participant_ids"]]
    coalition = next(
        scenario["coalition_ids"] for scenario in design["scenarios"]
        if scenario["scenario_id"] == "MR_K05_LOCAL"
    )
    indices = [participants.index(str(item)) for item in coalition]
    records: list[dict[str, Any]] = []
    for row in subset.to_dict(orient="records"):
        raw = _sequence(row["raw_request_mw"])
        effective = _sequence(row["effective_request_mw"])
        allocation = _sequence(row["allocation_mw"])
        delivery = _sequence(row["delivery_mw"])
        capability = _sequence(row["capability_mw"])
        ceiling_before = _sequence(row["access_ceiling_before_mw"])
        ceiling_after = _sequence(row["access_ceiling_after_mw"])
        record = {
            "interval": int(row["interval"]),
            "control_id": str(row["control_id"]),
            "changed_participant_count": len(indices),
            "changed_raw_request_kw": 1000.0 * math.fsum(raw[index] for index in indices),
            "changed_effective_request_kw": 1000.0 * math.fsum(effective[index] for index in indices),
            "changed_allocation_kw": 1000.0 * math.fsum(allocation[index] for index in indices),
            "changed_capability_kw": 1000.0 * math.fsum(capability[index] for index in indices),
            "changed_delivery_kw": 1000.0 * math.fsum(delivery[index] for index in indices),
            "feeder_delivery_kw": 1000.0 * math.fsum(delivery),
            "feeder_delivery_change_kw": 1000.0 * float(row["total_delivery_change_mw"]),
            "idle_authorization_kw": 1000.0 * float(row["reported_total_idle_award_mw"]),
            "request_attenuation_kw": 1000.0 * float(row["request_attenuation_mw"]),
            "mean_changed_ceiling_before_ratio": _mean(
                ceiling_before[index] / max(raw[index], 1.0e-15) for index in indices
            ),
            "mean_changed_ceiling_after_ratio": _mean(
                ceiling_after[index] / max(raw[index], 1.0e-15) for index in indices
            ),
            "replay_gate_triggered": str(row.get("replay_gate_triggered", "")).lower() == "true",
            "replay_outsider_access_release_kw": 1000.0 * float(
                row.get("replay_outsider_allocation_gain_mw", 0.0)
            ),
            "replay_outsider_delivery_recovery_kw": 1000.0 * float(
                row.get("replay_outsider_delivery_gain_mw", 0.0)
            ),
            "replay_feeder_delivery_recovery_kw": 1000.0 * float(
                row.get("replay_total_delivery_gain_mw", 0.0)
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def _dynamic_tables(
    intervals: pd.DataFrame,
    trace_intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    evidence: Mapping[str, Any],
    closure_audit: Mapping[str, Any],
    design: Mapping[str, Any],
    delivery_audit: Mapping[str, Any] | None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "Fig04_representative_causal_trace": _trace_table(trace_intervals, design),
        "Fig05_penalty_behavior_effect": pd.DataFrame(evidence["penalty_behavior_effect"]),
        "Fig05_settlement_feedback_stage": pd.DataFrame(
            closure_audit["stage_comparison"]
        ),
        "Fig05_control_contrast": pd.DataFrame(evidence["control_contrast"]),
        "Fig05_externality_classification_sensitivity": pd.DataFrame(
            evidence["externality_classification_sensitivity"]
        ),
        "Fig06_dimension_robustness_theta0": pd.DataFrame(
            evidence["dimension_contrast_theta_0"]
        ),
        "Fig07_beta_sensitivity": pd.DataFrame(evidence["beta_sensitivity"]),
        "Fig07_paired_primary_distribution": pd.DataFrame(
            evidence["paired_primary_distribution"]
        ),
        "Fig07_gate_reason_summary": pd.DataFrame(evidence["gate_reason_summary"]),
        "Fig07_gate_materiality_sensitivity": pd.DataFrame(
            evidence["gate_materiality_sensitivity"]
        ),
    }
    gate_materiality = intervals[
        intervals["control_id"] == PRIMARY_CONTROL
    ][[
        "trajectory_id", "interval", "penalty_to_export_value_ratio",
        "replay_gate_triggered", "replay_gate_reason",
        "replay_total_delivery_gain_mw", "replay_outsider_delivery_gain_mw",
    ]].copy()
    gate_materiality["replay_total_delivery_recovery_kw"] = (
        1000.0 * gate_materiality["replay_total_delivery_gain_mw"]
    )
    gate_materiality["replay_outsider_delivery_recovery_kw"] = (
        1000.0 * gate_materiality["replay_outsider_delivery_gain_mw"]
    )
    tables["Fig07_gate_materiality_distribution"] = gate_materiality

    nominal = trajectories[
        trajectories["penalty_to_export_value_ratio"] == 0.0
    ].copy()
    nominal["delivery_change_kw"] = 1000.0 * nominal["mean_total_delivery_change_mw"]
    nominal["jain_change_percentage_points"] = 100.0 * nominal["mean_jain_norm_change"]
    nominal["minimum_voltage_margin_mpu"] = 1000.0 * (
        nominal["minimum_voltage_pu"] - 0.95
    )
    tables["Fig08_trajectory_physical_fairness"] = nominal[[
        "trajectory_id", "scenario_id", "profile_id", "q_mode", "method_id",
        "control_id", "delivery_change_kw", "jain_change_percentage_points",
        "minimum_voltage_margin_mpu", "maximum_loading_ratio", "maximum_mva_excess",
        "matched_q_pair_pass_rate",
    ]].copy()
    physical = pd.DataFrame([{
        "allocation_interval_count": int(trajectories["interval_count"].sum()),
        "allocation_matched_q_pass_count": int(round(
            (trajectories["interval_count"] * trajectories["matched_q_pair_pass_rate"]).sum()
        )),
        "allocation_minimum_voltage_pu": float(trajectories["minimum_voltage_pu"].min()),
        "allocation_maximum_loading_ratio": float(trajectories["maximum_loading_ratio"].max()),
        "allocation_maximum_mva_excess": float(trajectories["maximum_mva_excess"].max()),
        "delivery_state_count": (
            int(delivery_audit["unique_delivery_state_count"]) if delivery_audit else None
        ),
        "delivery_minimum_voltage_pu": (
            float(delivery_audit["minimum_voltage_pu"]) if delivery_audit else None
        ),
        "delivery_maximum_loading_ratio": (
            float(delivery_audit["maximum_loading_ratio"]) if delivery_audit else None
        ),
        "delivery_maximum_mva_excess": (
            float(delivery_audit["maximum_mva_excess"]) if delivery_audit else None
        ),
        "delivery_ac_audit_complete": bool(
            delivery_audit and all(delivery_audit["gate_results"].values())
        ),
    }])
    tables["Fig08_physical_validity_summary"] = physical
    return tables


def export(root: Path, output_dir: Path) -> dict[str, Any]:
    behavior_path = root / "outputs" / "endogenous_request_choice_v14" / "ENDOGENOUS_REQUEST_CHOICE_RESULTS.json"
    dynamic_dir = root / "outputs" / "externality_conditioned_feedback_v16"
    behavior = _load(behavior_path)
    design = _load(dynamic_dir / "V16_DESIGN.json")
    result = _load(dynamic_dir / "V16_RESULTS.json")
    evidence = _load(dynamic_dir / "V16_EVIDENCE_SUMMARY.json")
    closure_audit = _load(dynamic_dir / "V16_BEHAVIORAL_CLOSURE_AUDIT.json")
    behavior_hash = _self_hash(behavior, "result_hash")
    design_hash = _self_hash(design, "design_hash")
    result_hash = _self_hash(result, "result_hash")
    evidence_hash = _self_hash(evidence, "result_hash")
    closure_audit_hash = _self_hash(closure_audit, "audit_hash")
    if not (
        all(behavior["gate_results"].values())
        and all(result["gate_results"].values())
        and all(evidence["gate_results"].values())
        and all(closure_audit["gate_results"].values())
    ):
        raise ValueError("behavior or dynamic evidence contains an open gate")
    delivery_audit_path = dynamic_dir / "V16_REALIZED_DELIVERY_AC_AUDIT.json"
    delivery_audit = _load(delivery_audit_path) if delivery_audit_path.exists() else None
    if delivery_audit is not None:
        _self_hash(delivery_audit, "result_hash")
        if not all(delivery_audit["gate_results"].values()):
            raise ValueError("realized-delivery AC audit contains an open gate")

    intervals = pd.read_csv(dynamic_dir / "V16_INTERVAL_DATA.csv", encoding="utf-8-sig")
    if len(intervals) != int(result["interval_record_count"]):
        raise ValueError("V16 scalar interval table does not match the result count")
    trace_controls = {
        "NO_FEEDBACK", "DELIVERY_RATIO_CF_BETA_050", PRIMARY_CONTROL,
        "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
    }
    trace_steps: list[Mapping[str, Any]] = []
    for trajectory in result["trajectory_summaries"]:
        for step in trajectory["steps"]:
            if (
                step["scenario_id"] == "MR_K05_LOCAL"
                and step["profile_id"] == "SHOCK_AND_RECOVERY"
                and float(step["penalty_to_export_value_ratio"]) == 0.0
                and step["q_mode"] == "Q0"
                and step["method_id"] == "weighted_proportional"
                and step["control_id"] in trace_controls
            ):
                trace_steps.append(step)
    trace_intervals = pd.DataFrame(trace_steps)
    if len(trace_intervals) != 4 * int(design["horizon"]):
        raise ValueError("representative V16 trace is incomplete")
    trajectories = pd.read_csv(
        dynamic_dir / "V16_TRAJECTORY_SUMMARY.csv", encoding="utf-8-sig",
    )
    tables = _behavior_tables(behavior)
    tables["Fig03_stepwise_consequence_equivalent_summary"] = pd.DataFrame(
        closure_audit["stepwise_consequence_equivalent_summary"]
    )
    tables.update(_dynamic_tables(
        intervals, trace_intervals, trajectories, evidence, closure_audit,
        design, delivery_audit,
    ))
    data_dir = output_dir / "data" / "csv"
    paths = [
        _write(frame, data_dir / f"{name}.csv")
        for name, frame in sorted(tables.items())
    ]
    manifest: dict[str, Any] = {
        "serialization_id": "r4r.v16_manuscript_figure_data.v1",
        "status": "V16_MANUSCRIPT_FIGURE_DATA_COMPLETE",
        "behavior_result_hash": behavior_hash,
        "v16_design_hash": design_hash,
        "v16_result_hash": result_hash,
        "v16_evidence_hash": evidence_hash,
        "behavioral_closure_audit_hash": closure_audit_hash,
        "delivery_ac_audit_hash": (
            delivery_audit["result_hash"] if delivery_audit else None
        ),
        "csv_count": len(paths),
        "csv_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "row_count": int(len(tables[path.stem])),
                "sha256": _file_hash(path),
            }
            for path in paths
        ],
        "manifest_hash": None,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    (output_dir / "V16_MANUSCRIPT_FIGURE_DATA_MANIFEST.json").write_text(
        canonical_dumps(manifest) + "\n", encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs" / "manuscript_behavior_externality_v16",
    )
    args = parser.parse_args(argv)
    manifest = export(root, args.output_dir)
    print(canonical_dumps({
        "status": manifest["status"],
        "csv_count": manifest["csv_count"],
        "manifest_hash": manifest["manifest_hash"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
