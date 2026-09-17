"""AC-audit observed and replayed delivery states from the V16 experiment.

The allocation runner verifies every awarded access vector.  This independent
audit closes the remaining physical link by mapping the metered delivery
vectors, and the same-realization replay delivery vectors, into the identical
IEEE-141 AC path.  Duplicate vectors are solved once and then mapped back to
all interval occurrences.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from r4r.serialization import canonical_dumps, canonical_hash
from tools.ieee141_rate_v4_design_ac_runner import _run_ac_pair, _selection_for_profile
from tools.run_ieee141_m1_v2_directed_probe_ac_truth import _screen
import tools.run_v12_3_multi_reporter_experiment as multi
import tools.run_v12_3_dynamic_feedback_experiment as dynamic


SERIALIZATION_ID = "r4r.v16_realized_delivery_ac_audit.v1"
_AUDIT: dict[str, Any] = {}


def _load_json(path: Path) -> dict[str, Any]:
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


def _vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("delivery vector is not serialized as a sequence")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < -1.0e-12 for item in result):
        raise ValueError("delivery vector contains an invalid value")
    return result


def _key(q_mode: str, values: Sequence[float]) -> tuple[str, tuple[float, ...]]:
    return str(q_mode), tuple(round(float(item), 14) for item in values)


def _init_worker(
    case_path: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    margins: Mapping[str, Mapping[str, Any]],
    capacity_mw: Sequence[float],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    multi._init_worker(case_path, policy, source, margins)
    parsed = multi._WORKER["parsed"]
    _AUDIT.clear()
    _AUDIT.update({
        "selection": _selection_for_profile(parsed, capacity_mw=capacity_mw),
    })


def _screen_one(
    task: tuple[str, tuple[float, ...]],
) -> dict[str, Any]:
    q_mode, delivery = task
    worker = multi._WORKER
    source = worker["source"]
    lower = float(source["physical_limits"]["voltage_lower_limit_pu"])
    upper = float(source["physical_limits"]["voltage_upper_limit_pu"])
    ac = _run_ac_pair(
        parsed=worker["parsed"],
        selection=_AUDIT["selection"],
        allocation_mw=delivery,
        q_mode=q_mode,
        q95_specification=worker["q95_specification"],
        execution=worker["anchor_execution"],
    )
    primary = ac["primary"]
    actual = {
        "branch_p_from_mw": list(primary["branch_p_from_mw"]),
        "branch_q_from_mvar": list(primary["branch_q_from_mvar"]),
        "branch_p_to_mw": list(primary["branch_p_to_mw"]),
        "branch_q_to_mvar": list(primary["branch_q_to_mvar"]),
        "voltage_pu": list(primary["bus_voltage_pu"]),
    }
    screen = _screen(
        actual,
        worker["rates"],
        lower,
        upper,
        float(source["physical_limits"]["branch_tolerance_mva"]),
        float(source["physical_limits"]["voltage_tolerance_pu"]),
    )
    numerical = bool(ac["ac_converged_both"] and ac["crosscheck_status"] == "PASS")
    voltage = [float(item) for item in actual["voltage_pu"]]
    branch = [float(item) for item in screen.get("branch_mva_max", [])]
    rates = [float(item) for item in worker["rates"]]
    loading = [flow / rate for flow, rate in zip(branch, rates) if rate > 0.0]
    record: dict[str, Any] = {
        "q_mode": q_mode,
        "delivery_mw": list(delivery),
        "delivery_hash": canonical_hash({"q_mode": q_mode, "values_mw": list(delivery)}),
        "ac_result_hash": ac["result_hash"],
        "primary_solution_hash": ac["primary_ac_solution_hash"],
        "independent_solution_hash": ac["independent_ac_solution_hash"],
        "crosscheck_status": ac["crosscheck_status"],
        "ac_numerical_valid": numerical,
        "physical_screen_pass": bool(numerical and screen["joint_pass"]),
        "minimum_voltage_pu": min(voltage, default=None),
        "maximum_voltage_pu": max(voltage, default=None),
        "maximum_loading_ratio": max(loading, default=0.0),
        "max_mva_excess": float(screen.get("max_mva_excess", 0.0)),
        "total_mva_excess": float(screen.get("total_mva_excess", 0.0)),
        "voltage_lower_excess_pu": float(screen.get("voltage_lower_excess_pu", 0.0)),
        "voltage_upper_excess_pu": float(screen.get("voltage_upper_excess_pu", 0.0)),
        "record_hash": None,
    }
    record["record_hash"] = canonical_hash(record)
    return record


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


def audit(
    *,
    case_path: Path,
    policy_path: Path,
    source_path: Path,
    execution_path: Path,
    design_path: Path,
    result_path: Path,
    interval_path: Path,
    output_path: Path,
    workers: int,
) -> dict[str, Any]:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8-sig"))
    source = _load_json(source_path)
    execution = _load_json(execution_path)
    design = _load_json(design_path)
    result = _load_json(result_path)
    design_hash = _validated_hash(design, field="design_hash", label="V16 design")
    result_hash = _validated_hash(result, field="result_hash", label="V16 result")
    if result.get("status") != "EXTERNALITY_CONDITIONED_FEEDBACK_EXPERIMENT_COMPLETE":
        raise ValueError("V16 result is not complete")
    if result.get("design_hash") != design_hash:
        raise ValueError("V16 result/design binding mismatch")
    capacities = {
        tuple(float(item) for item in scenario["capacity_mw"])
        for scenario in design["scenarios"]
    }
    if len(capacities) != 1:
        raise ValueError("delivery audit requires one registered participant-capacity vector")
    capacity = next(iter(capacities))

    frame = pd.read_csv(interval_path, encoding="utf-8-sig")
    required_scalar_columns = {
        "trajectory_id", "interval", "q_mode", "replay_attempted",
    }
    missing_scalar_columns = required_scalar_columns.difference(frame.columns)
    if missing_scalar_columns:
        raise ValueError(
            "V16 interval table is missing required scalar columns: "
            f"{sorted(missing_scalar_columns)}"
        )
    interval_rows: list[Mapping[str, Any]] = []
    for trajectory in result.get("trajectory_summaries", []):
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            raise ValueError("V16 trajectory is missing its serialized interval steps")
        interval_rows.extend(steps)
    if len(interval_rows) != len(frame):
        raise ValueError(
            "V16 JSON/CSV interval-count mismatch: "
            f"{len(interval_rows)} != {len(frame)}"
        )
    occurrence_roles: dict[tuple[str, tuple[float, ...]], dict[str, int]] = defaultdict(
        lambda: {"OBSERVED_DELIVERY": 0, "REPLAY_DELIVERY": 0}
    )
    for row in interval_rows:
        q_mode = str(row["q_mode"])
        if "delivery_mw" not in row:
            raise ValueError("V16 interval step is missing delivery_mw")
        observed = _key(q_mode, _vector(row["delivery_mw"]))
        occurrence_roles[observed]["OBSERVED_DELIVERY"] += 1
        if bool(row.get("replay_attempted", False)):
            if "replay_delivery_mw" not in row:
                raise ValueError("attempted replay is missing replay_delivery_mw")
            replay = _key(q_mode, _vector(row["replay_delivery_mw"]))
            occurrence_roles[replay]["REPLAY_DELIVERY"] += 1
    tasks = sorted(occurrence_roles, key=lambda item: (item[0], item[1]))

    source_scenario_id = str(design["scenarios"][0]["source_scenario_id"])
    margins = dynamic._extract_margins(
        execution, source, source_scenario_id,
    )

    workers = max(1, int(workers))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(case_path), policy, source, margins, capacity),
        ) as executor:
            screened = list(executor.map(
                _screen_one,
                tasks,
                chunksize=max(1, len(tasks) // (workers * 8)),
            ))
    else:
        _init_worker(str(case_path), policy, source, margins, capacity)
        screened = [_screen_one(task) for task in tasks]

    records: list[dict[str, Any]] = []
    for task, row in zip(tasks, screened):
        counts = occurrence_roles[task]
        records.append({
            **row,
            "observed_occurrence_count": counts["OBSERVED_DELIVERY"],
            "replay_occurrence_count": counts["REPLAY_DELIVERY"],
        })
    observed = [row for row in records if row["observed_occurrence_count"] > 0]
    replay = [row for row in records if row["replay_occurrence_count"] > 0]
    expected_observed = len(frame)
    expected_replay = int(result["replay_attempt_count"])
    gates = {
        "SOURCE_BINDING": bool(
            result["design_hash"] == design_hash
            and result["source_envelope_hash"] == source["result_hash"]
            and design["policy_hash"] == policy["policy_hash"]
        ),
        "OBSERVED_OCCURRENCE_COMPLETENESS": sum(
            int(row["observed_occurrence_count"]) for row in records
        ) == expected_observed,
        "REPLAY_OCCURRENCE_COMPLETENESS": sum(
            int(row["replay_occurrence_count"]) for row in records
        ) == expected_replay,
        "OBSERVED_DELIVERY_AC_NUMERICAL_COMPLETENESS": bool(observed) and all(
            row["ac_numerical_valid"] for row in observed
        ),
        "OBSERVED_DELIVERY_PHYSICAL_COMPLETENESS": bool(observed) and all(
            row["physical_screen_pass"] for row in observed
        ),
        "REPLAY_DELIVERY_AC_NUMERICAL_COMPLETENESS": bool(replay) and all(
            row["ac_numerical_valid"] for row in replay
        ),
        "REPLAY_DELIVERY_PHYSICAL_COMPLETENESS": bool(replay) and all(
            row["physical_screen_pass"] for row in replay
        ),
        "Q0_Q95_BOTH_AUDITED": {row["q_mode"] for row in records} == set(design["q_roster"]),
    }
    audit_result: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": (
            "V16_REALIZED_DELIVERY_AC_AUDIT_COMPLETE"
            if all(gates.values())
            else "V16_REALIZED_DELIVERY_AC_AUDIT_BLOCKED"
        ),
        "v16_design_hash": design_hash,
        "v16_result_hash": result_hash,
        "unique_delivery_state_count": len(records),
        "unique_observed_delivery_state_count": len(observed),
        "unique_replay_delivery_state_count": len(replay),
        "observed_occurrence_count": expected_observed,
        "replay_occurrence_count": expected_replay,
        "minimum_voltage_pu": min(
            (float(row["minimum_voltage_pu"]) for row in records), default=None,
        ),
        "maximum_loading_ratio": max(
            (float(row["maximum_loading_ratio"]) for row in records), default=0.0,
        ),
        "maximum_mva_excess": max(
            (float(row["max_mva_excess"]) for row in records), default=0.0,
        ),
        "gate_results": gates,
        "records": records,
        "result_hash": None,
    }
    audit_result["result_hash"] = canonical_hash(audit_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(audit_result) + "\n", encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), records)
    output_path.with_suffix(".md").write_text(
        "\n".join([
            "# V16 realized-delivery AC audit",
            "",
            f"- Status: `{audit_result['status']}`",
            f"- Unique delivery states: `{len(records)}`.",
            f"- Observed occurrences: `{expected_observed}`; replay occurrences: `{expected_replay}`.",
            f"- Minimum voltage: `{audit_result['minimum_voltage_pu']}` pu.",
            f"- Maximum loading ratio: `{audit_result['maximum_loading_ratio']}`.",
            f"- Gates: `{gates}`",
            f"- Result hash: `{audit_result['result_hash']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    return audit_result


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fresh = root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "fresh_source_design"
    out = root / "outputs" / "externality_conditioned_feedback_v16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=root / "data" / "raw" / "case141.m")
    parser.add_argument(
        "--policy", type=Path,
        default=root / "outputs" / "rate_v4_design" / "V4_FREEZE_DEVELOPMENT" / "CASE141_SYNTHETIC_RATE_POLICY_V4.yaml",
    )
    parser.add_argument("--source", type=Path, default=fresh / "V12_3_FRESH_SOURCE_ENVELOPE_V5.json")
    parser.add_argument("--execution", type=Path, default=fresh / "V12_3_METHOD_EXECUTION_V8.json")
    parser.add_argument("--design", type=Path, default=out / "V16_DESIGN.json")
    parser.add_argument("--result", type=Path, default=out / "V16_RESULTS.json")
    parser.add_argument("--intervals", type=Path, default=out / "V16_INTERVAL_DATA.csv")
    parser.add_argument("--output", type=Path, default=out / "V16_REALIZED_DELIVERY_AC_AUDIT.json")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    result = audit(
        case_path=args.case,
        policy_path=args.policy,
        source_path=args.source,
        execution_path=args.execution,
        design_path=args.design,
        result_path=args.result,
        interval_path=args.intervals,
        output_path=args.output,
        workers=args.workers,
    )
    print(canonical_dumps({
        "status": result["status"],
        "unique_delivery_state_count": result["unique_delivery_state_count"],
        "gates": result["gate_results"],
        "result_hash": result["result_hash"],
    }))
    return 0 if all(result["gate_results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
