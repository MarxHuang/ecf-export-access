"""Build fixed target-directed M1 V2 probes from a frozen proxy only.

This tool never runs AC, never reads point-level residual/pass fields, never
re-solves an allocation, and never reads evaluation or mitigation outputs.  It
uses only the frozen affine proxy model, development source allocation
vectors, frozen RATE_A/voltage bounds, and declared domain limits.  The output
is a diagnostic probe specification; a later AC run must consume its fixed
allocation vectors without feeding results back into this selection step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from r4r.network_parser import parse_matpower_case
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.synthetic_rate import load_synthetic_rate_policy
from tools.split_identity import declared_split_hash, split_hash_is_self_consistent


Q_MODES = ("Q0", "Q95")
METHODS = ("max_export_lp", "weighted_proportional", "equal_kw_reduction", "flat_level", "fairness_qp")
SIDES = ("reference", "reported")
TRAIN_REPORTERS = (
    "ieee141-reporter-scan-01",
    "ieee141-reporter-scan-08",
    "ieee141-reporter-scan-15",
)
TARGET_TOLERANCE = 2.0e-4
MONOTONIC_TOLERANCE = 1.0e-9
PREEMPTION_TOLERANCE = TARGET_TOLERANCE
BISECTION_ITERATIONS = 32
PRIMARY_VOLTAGE_TARGET = 0.0015
FALLBACK_VOLTAGE_TARGET = 0.0035
JOINT_BRANCH_TARGET = 0.95
JOINT_VOLTAGE_TARGET = 0.002


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _allocation_values(run: Mapping[str, Any]) -> tuple[float, ...]:
    profile = run.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("source method run lacks profile")
    allocation = profile.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("source method run lacks allocation")
    values = allocation.get("values_mw")
    if values is None and isinstance(allocation.get("allocation"), Mapping):
        values = allocation["allocation"].get("values_mw")
    if not isinstance(values, list) or len(values) != 30:
        raise ValueError("source allocation must contain 30 participant values")
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and value >= -1.0e-12 for value in values):
        raise ValueError("source allocation is outside the nonnegative domain")
    return values


def _source_method_run(grid: Mapping[str, Any], *, scenario_id: str, side: str, method_id: str) -> Mapping[str, Any]:
    batches = (grid.get("grid") or {}).get("batches")
    if not isinstance(batches, list):
        raise ValueError("source grid has no typed batches")
    for batch in batches:
        if not isinstance(batch, Mapping):
            continue
        side_payload = batch.get(f"{side}_batch")
        if not isinstance(side_payload, Mapping) or side_payload.get("scenario_id") != f"{scenario_id}__{side}":
            continue
        for run in side_payload.get("method_runs", []):
            if isinstance(run, Mapping) and run.get("method_id") == method_id:
                return run
    raise ValueError(f"source method run not found: {scenario_id}/{side}/{method_id}")


def _transform(values: Sequence[float], transform_id: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    if transform_id == "SOURCE_ALLOCATION_SCALE":
        return values
    if transform_id != "TOP10_SHARE45":
        raise ValueError(f"unsupported allocation transform: {transform_id}")
    total = math.fsum(values)
    ranked = tuple(sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)[:10])
    top_sum = math.fsum(values[i] for i in ranked)
    remainder = total - top_sum
    if total <= 0.0 or top_sum <= 0.0 or remainder <= 0.0:
        raise ValueError("participant concentration transform is degenerate")
    top_set = set(ranked)
    return tuple(
        0.45 * total * value / top_sum if i in top_set else 0.55 * total * value / remainder
        for i, value in enumerate(values)
    )


def _dot(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [math.fsum(float(a) * float(b) for a, b in zip(row, vector)) for row in matrix]


def _proxy_values(model_q: Mapping[str, Any], allocation: Sequence[float]) -> dict[str, list[float]]:
    base = model_q["proxy_base"]
    matrices = model_q["derivative_matrices"]
    return {
        field: [float(value) + delta for value, delta in zip(base[field], _dot(matrices[field], allocation))]
        for field in ("branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar", "voltage_pu")
    }


def _directional_delta(model_q: Mapping[str, Any], allocation: Sequence[float]) -> dict[str, list[float]]:
    matrices = model_q["derivative_matrices"]
    return {
        field: _dot(matrices[field], allocation)
        for field in ("branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar", "voltage_pu")
    }


def _directional_values(direction: Mapping[str, Any], scale: float) -> dict[str, list[float]]:
    model_q = direction["model_q"]
    base = model_q["proxy_base"]
    delta = direction["delta_fields"]
    return {
        field: [float(value) + float(scale) * float(slope) for value, slope in zip(base[field], delta[field])]
        for field in ("branch_p_from_mw", "branch_q_from_mvar", "branch_p_to_mw", "branch_q_to_mvar", "voltage_pu")
    }


def _metrics(values: Mapping[str, Sequence[float]], rates: Sequence[float], lower: float, upper: float, *, voltage_indices: Sequence[int]) -> dict[str, Any]:
    mva_from = [math.hypot(p, q) for p, q in zip(values["branch_p_from_mw"], values["branch_q_from_mvar"])]
    mva_to = [math.hypot(p, q) for p, q in zip(values["branch_p_to_mw"], values["branch_q_to_mvar"])]
    branch_mva = [max(left, right) for left, right in zip(mva_from, mva_to)]
    ratios = [mva / rate for mva, rate in zip(branch_mva, rates)]
    lower_slacks = [float(values["voltage_pu"][i]) - lower for i in voltage_indices]
    upper_slacks = [upper - float(values["voltage_pu"][i]) for i in voltage_indices]
    return {
        "branch_ratios": ratios,
        "branch_mva": branch_mva,
        "lower_slacks": lower_slacks,
        "upper_slacks": upper_slacks,
        "min_lower_slack": min(lower_slacks, default=float("inf")),
        "min_upper_slack": min(upper_slacks, default=float("inf")),
        "min_lower_index": voltage_indices[min(range(len(lower_slacks)), key=lambda i: lower_slacks[i])] if lower_slacks else None,
        "min_upper_index": voltage_indices[min(range(len(upper_slacks)), key=lambda i: upper_slacks[i])] if upper_slacks else None,
    }


def _monotonic(values: Sequence[float]) -> str | None:
    increasing = all(right + MONOTONIC_TOLERANCE >= left for left, right in zip(values, values[1:]))
    decreasing = all(right <= left + MONOTONIC_TOLERANCE for left, right in zip(values, values[1:]))
    if increasing:
        return "NONDECREASING"
    if decreasing:
        return "NONINCREASING"
    return None


def _domain_max(allocation: Sequence[float], requested_max: float) -> float:
    positive = [1.0 / value for value in allocation if value > 0.0]
    return min([float(requested_max), *positive], default=float(requested_max))


def _target_crossing(
    *,
    evaluate: Any,
    target: float,
    direction: str,
    lambda_max: float,
) -> dict[str, Any]:
    if lambda_max <= 0.0:
        return {"status": "DOMAIN_EXCLUSION_AVAILABILITY", "lambda_scale": None}
    grid = sorted(set([0.0, lambda_max] + [value for value in (0.25, 0.5, 0.75, 0.9, 1.0, 1.05, 1.1, 1.2) if value <= lambda_max + MONOTONIC_TOLERANCE]))
    values = [float(evaluate(value)) for value in grid]
    monotonic = _monotonic(values)
    if monotonic is None:
        return {"status": "TARGET_NONMONOTONE_NO_VALID_BRACKET", "lambda_scale": None, "grid": grid, "grid_values": values}
    candidates = [(abs(value - target), index, value) for index, value in enumerate(values)]
    best = min(candidates, key=lambda item: (item[0], item[1]))
    if best[0] <= TARGET_TOLERANCE:
        return {"status": "TARGET_REACHED_BY_GRID_POINT", "lambda_scale": grid[best[1]], "metric": best[2], "monotonicity": monotonic, "grid": grid, "grid_values": values}
    low_lambda, high_lambda = grid[0], grid[-1]
    low_value, high_value = values[0], values[-1]
    if not ((min(low_value, high_value) <= target <= max(low_value, high_value))):
        return {"status": "TARGET_UNREACHABLE_WITHIN_DOMAIN", "lambda_scale": grid[best[1]], "metric": best[2], "monotonicity": monotonic, "grid": grid, "grid_values": values}
    for _ in range(BISECTION_ITERATIONS):
        mid = 0.5 * (low_lambda + high_lambda)
        mid_value = float(evaluate(mid))
        if (mid_value < target) == (low_value < target):
            low_lambda, low_value = mid, mid_value
        else:
            high_lambda, high_value = mid, mid_value
    chosen = low_lambda if abs(low_value - target) <= abs(high_value - target) else high_lambda
    metric = low_value if chosen == low_lambda else high_value
    return {"status": "TARGET_REACHED_BY_BISECTION", "lambda_scale": chosen, "metric": metric, "monotonicity": monotonic, "grid": grid, "grid_values": values}


def _joint_crossing(*, evaluate: Any, branch_target: float, voltage_target: float, lambda_max: float, voltage_kind: str) -> dict[str, Any]:
    if lambda_max <= 0.0:
        return {"status": "DOMAIN_EXCLUSION_AVAILABILITY", "lambda_scale": None}
    grid = sorted(set([0.0, lambda_max] + [value for value in (0.25, 0.5, 0.75, 0.9, 1.0, 1.05, 1.1, 1.2) if value <= lambda_max + MONOTONIC_TOLERANCE]))
    rows = [evaluate(value) for value in grid]
    branch_values = [float(row["branch_ratio"]) for row in rows]
    voltage_values = [float(row["voltage_slack"]) for row in rows]
    branch_mono = _monotonic(branch_values)
    voltage_mono = _monotonic(voltage_values)
    ok = [row["branch_ratio"] >= branch_target and row["voltage_slack"] <= voltage_target for row in rows]
    if branch_mono is None or voltage_mono is None or any(left and not right for left, right in zip(ok, ok[1:])):
        return {"status": "TARGET_NONMONOTONE_NO_VALID_BRACKET", "lambda_scale": None, "grid": grid, "grid_values": rows, "branch_monotonicity": branch_mono, "voltage_monotonicity": voltage_mono}
    if not ok[-1]:
        return {"status": "TARGET_UNREACHABLE_WITHIN_DOMAIN", "lambda_scale": grid[-1], "metric": rows[-1], "grid": grid, "grid_values": rows, "branch_monotonicity": branch_mono, "voltage_monotonicity": voltage_mono}
    if ok[0]:
        return {"status": "TARGET_REACHED_BY_GRID_POINT", "lambda_scale": 0.0, "metric": rows[0], "grid": grid, "grid_values": rows, "branch_monotonicity": branch_mono, "voltage_monotonicity": voltage_mono}
    low, high = 0.0, lambda_max
    for _ in range(BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        row = evaluate(mid)
        if row["branch_ratio"] >= branch_target and row["voltage_slack"] <= voltage_target:
            high = mid
        else:
            low = mid
    return {"status": "TARGET_REACHED_BY_BISECTION", "lambda_scale": high, "metric": evaluate(high), "grid": grid, "grid_values": rows, "branch_monotonicity": branch_mono, "voltage_monotonicity": voltage_mono}


def _probe_direction_rows(*, grid_payloads: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Any], model: Mapping[str, Any], policy_payload: Mapping[str, Any], voltage_indices: Sequence[int]) -> Iterable[dict[str, Any]]:
    transform_id = str((manifest.get("allocation_transform") or {}).get("id"))
    for q_mode in Q_MODES:
        grid = grid_payloads[q_mode]
        for reporter_id in TRAIN_REPORTERS:
            for side in SIDES:
                for method_id in METHODS:
                    run = _source_method_run(grid, scenario_id=reporter_id, side=side, method_id=method_id)
                    source_allocation = _allocation_values(run)
                    allocation = _transform(source_allocation, transform_id)
                    source_hash = str((run.get("proxy_audit") or {}).get("allocation_hash"))
                    if len(source_hash) != 64:
                        raise ValueError("source allocation hash missing")
                    yield {
                        "q_mode": q_mode,
                        "reporter_id": reporter_id,
                        "allocation_side": side,
                        "method_id": method_id,
                        "source_allocation_hash": source_hash,
                        "allocation_mw": list(allocation),
                        "allocation_hash": canonical_hash({"values_mw": list(allocation)}),
                        "model_q": model["q_models"][q_mode],
                        "delta_fields": _directional_delta(model["q_models"][q_mode], allocation),
                        "policy_payload": policy_payload,
                        "voltage_indices": voltage_indices,
                    }


def _candidate_probe(*, target_id: str, target_kind: str, direction: Mapping[str, Any], policy_payload: Mapping[str, Any], branch_index: int | None = None, voltage_index: int | None = None, target_value: float, fallback_used: bool = False) -> dict[str, Any]:
    model_q = direction["model_q"]
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    lower = float(policy_payload["voltage_lower_limit_pu"])
    upper = float(policy_payload["voltage_upper_limit_pu"])
    voltage_indices = direction["voltage_indices"]
    allocation = direction["allocation_mw"]
    lambda_max = _domain_max(allocation, 1.2)

    def row_at(scale: float) -> dict[str, Any]:
        values = _directional_values(direction, scale)
        return _metrics(values, rates, lower, upper, voltage_indices=voltage_indices)

    def branch_at(scale: float, index: int) -> float:
        base = model_q["proxy_base"]
        delta = direction["delta_fields"]
        p_from = float(base["branch_p_from_mw"][index]) + float(scale) * float(delta["branch_p_from_mw"][index])
        q_from = float(base["branch_q_from_mvar"][index]) + float(scale) * float(delta["branch_q_from_mvar"][index])
        p_to = float(base["branch_p_to_mw"][index]) + float(scale) * float(delta["branch_p_to_mw"][index])
        q_to = float(base["branch_q_to_mvar"][index]) + float(scale) * float(delta["branch_q_to_mvar"][index])
        return max(math.hypot(p_from, q_from), math.hypot(p_to, q_to)) / rates[index]

    def voltage_at(scale: float, index: int, kind: str) -> float:
        base = model_q["proxy_base"]["voltage_pu"][index]
        slope = direction["delta_fields"]["voltage_pu"][index]
        value = float(base) + float(scale) * float(slope)
        return value - lower if kind == "VOLTAGE_LOWER" else upper - value

    if target_kind == "BRANCH":
        assert branch_index is not None
        crossing = _target_crossing(evaluate=lambda scale: branch_at(scale, branch_index), target=target_value, direction="increasing", lambda_max=lambda_max)
    elif target_kind == "VOLTAGE_LOWER":
        assert voltage_index is not None
        crossing = _target_crossing(evaluate=lambda scale: voltage_at(scale, voltage_index, "VOLTAGE_LOWER"), target=target_value, direction="any", lambda_max=lambda_max)
    elif target_kind == "VOLTAGE_UPPER":
        assert voltage_index is not None
        crossing = _target_crossing(evaluate=lambda scale: voltage_at(scale, voltage_index, "VOLTAGE_UPPER"), target=target_value, direction="any", lambda_max=lambda_max)
    else:
        assert target_kind in {"JOINT_LOWER", "JOINT_UPPER"} and branch_index is not None and voltage_index is not None
        voltage_key = "lower_slacks" if target_kind == "JOINT_LOWER" else "upper_slacks"
        crossing = _joint_crossing(
            evaluate=lambda scale: {
                "branch_ratio": branch_at(scale, branch_index),
                "voltage_slack": voltage_at(scale, voltage_index, "VOLTAGE_LOWER" if target_kind == "JOINT_LOWER" else "VOLTAGE_UPPER"),
            },
            branch_target=JOINT_BRANCH_TARGET,
            voltage_target=JOINT_VOLTAGE_TARGET,
            lambda_max=lambda_max,
            voltage_kind=target_kind,
        )
    scale = crossing.get("lambda_scale")
    if scale is None:
        return {
            "status": crossing.get("status"),
            "target_id": target_id,
            "target_kind": target_kind,
            "q_mode": direction["q_mode"],
            "reporter_id": direction["reporter_id"],
            "allocation_side": direction["allocation_side"],
            "method_id": direction["method_id"],
            "target_branch_index": branch_index,
            "target_voltage_index": voltage_index,
            "target_value": target_value,
            "fallback_used": fallback_used,
            "lambda_domain_max": lambda_max,
            "monotonicity": crossing.get("monotonicity") or {"branch": crossing.get("branch_monotonicity"), "voltage": crossing.get("voltage_monotonicity")},
        }
    metrics = row_at(float(scale))
    status = crossing["status"]
    preemption: dict[str, Any] | None = None
    # A branch target is a valid boundary probe only when every *other*
    # branch and every screened voltage remains within the frozen proxy
    # envelope.  If another constraint fires first, preserve the point as a
    # diagnostic but do not call it a branch-boundary probe.
    if target_kind == "BRANCH" and status in {"TARGET_REACHED_BY_BISECTION", "TARGET_REACHED_BY_GRID_POINT"}:
        other_branch = [
            (ratio, index)
            for index, ratio in enumerate(metrics["branch_ratios"])
            if index != branch_index and ratio > 1.0 + PREEMPTION_TOLERANCE
        ]
        voltage_values = _directional_values(direction, float(scale))["voltage_pu"]
        lower_bad = [
            (float(lower) - float(voltage_values[index]), index)
            for index in voltage_indices
            if float(voltage_values[index]) < lower - PREEMPTION_TOLERANCE
        ]
        upper_bad = [
            (float(voltage_values[index]) - float(upper), index)
            for index in voltage_indices
            if float(voltage_values[index]) > upper + PREEMPTION_TOLERANCE
        ]
        if other_branch or lower_bad or upper_bad:
            status = "TARGET_PREEMPTED_BY_OTHER_CONSTRAINT"
            preemption = {
                "other_branch_index": max(other_branch, default=(None, None))[1],
                "other_branch_ratio": max(other_branch, default=(None, None))[0],
                "lower_voltage_index": max(lower_bad, default=(None, None))[1],
                "lower_voltage_excess_pu": max(lower_bad, default=(None, None))[0],
                "upper_voltage_index": max(upper_bad, default=(None, None))[1],
                "upper_voltage_excess_pu": max(upper_bad, default=(None, None))[0],
            }
    row = {
        "status": status,
        "target_id": target_id,
        "target_kind": target_kind,
        "q_mode": direction["q_mode"],
        "reporter_id": direction["reporter_id"],
        "allocation_side": direction["allocation_side"],
        "method_id": direction["method_id"],
        "source_allocation_hash": direction["source_allocation_hash"],
        "allocation_hash": canonical_hash({"values_mw": [float(scale) * value for value in direction["allocation_mw"]]}),
        "allocation_mw": [float(scale) * value for value in direction["allocation_mw"]],
        "target_branch_index": branch_index,
        "target_voltage_index": voltage_index,
        "target_value": target_value,
        "fallback_used": fallback_used,
        "lambda_scale": float(scale),
        "lambda_domain_max": lambda_max,
        "metric_at_probe": branch_at(float(scale), branch_index) if target_kind == "BRANCH" else voltage_at(float(scale), voltage_index, target_kind) if target_kind in {"VOLTAGE_LOWER", "VOLTAGE_UPPER"} else {"branch_ratio": branch_at(float(scale), branch_index), "voltage_slack": voltage_at(float(scale), voltage_index, "VOLTAGE_LOWER" if target_kind == "JOINT_LOWER" else "VOLTAGE_UPPER")},
        "proxy_screen_summary": {
            "max_branch_ratio": max(metrics["branch_ratios"], default=0.0),
            "min_lower_slack": metrics["min_lower_slack"],
            "min_upper_slack": metrics["min_upper_slack"],
            "limiting_branch_index": max(range(len(metrics["branch_ratios"])), key=lambda i: metrics["branch_ratios"][i]) if metrics["branch_ratios"] else None,
            "limiting_lower_voltage_index": metrics["min_lower_index"],
            "limiting_upper_voltage_index": metrics["min_upper_index"],
        },
        "monotonicity": crossing.get("monotonicity") or {"branch": crossing.get("branch_monotonicity"), "voltage": crossing.get("voltage_monotonicity")},
    }
    if preemption is not None:
        row["preemption"] = preemption
    return row


def _floor_extremum_probe(*, target_id: str, q_mode: str, directions: Sequence[Mapping[str, Any]], policy_payload: Mapping[str, Any], floor_indices: Sequence[int], voltage_indices: Sequence[int]) -> dict[str, Any]:
    """Select one deterministic floor-dominant extremum without AC feedback."""
    candidates: list[dict[str, Any]] = []
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    lower = float(policy_payload["voltage_lower_limit_pu"])
    upper = float(policy_payload["voltage_upper_limit_pu"])
    for direction in directions:
        if direction["q_mode"] != q_mode:
            continue
        scale = _domain_max(direction["allocation_mw"], 1.2)
        values = _directional_values(direction, scale)
        metrics = _metrics(values, rates, lower, upper, voltage_indices=voltage_indices)
        for branch_index in floor_indices:
            candidates.append({
                "status": "TARGET_EXTREMUM_FROZEN_WITHIN_DOMAIN",
                "target_id": target_id,
                "target_kind": "BRANCH_EXTREMUM",
                "q_mode": q_mode,
                "reporter_id": direction["reporter_id"],
                "allocation_side": direction["allocation_side"],
                "method_id": direction["method_id"],
                "source_allocation_hash": direction["source_allocation_hash"],
                "allocation_hash": canonical_hash({"values_mw": [float(scale) * value for value in direction["allocation_mw"]]}),
                "allocation_mw": [float(scale) * value for value in direction["allocation_mw"]],
                "target_branch_index": branch_index,
                "target_value": metrics["branch_ratios"][branch_index],
                "fallback_used": False,
                "lambda_scale": scale,
                "lambda_domain_max": scale,
                "metric_at_probe": metrics["branch_ratios"][branch_index],
                "extremum_selection_rule": "MAX_PROXY_LOADING_RATIO_OVER_FLOOR_DOMINANT_AND_DEV_TRAIN_DIRECTIONS",
                "proxy_screen_summary": {
                    "max_branch_ratio": max(metrics["branch_ratios"], default=0.0),
                    "min_lower_slack": metrics["min_lower_slack"],
                    "min_upper_slack": metrics["min_upper_slack"],
                    "limiting_branch_index": max(range(len(metrics["branch_ratios"])), key=lambda i: metrics["branch_ratios"][i]) if metrics["branch_ratios"] else None,
                    "limiting_lower_voltage_index": metrics["min_lower_index"],
                    "limiting_upper_voltage_index": metrics["min_upper_index"],
                },
                "monotonicity": None,
            })
    if not candidates:
        return {"target_id": target_id, "target_kind": "BRANCH_EXTREMUM", "q_mode": q_mode, "status": "DOMAIN_EXCLUSION_AVAILABILITY", "target_value": None, "selection_pool": "FLOOR_DOMINANT_DEV_TRAIN_ONLY"}
    return max(candidates, key=lambda row: (float(row["metric_at_probe"]), row["reporter_id"], row["allocation_side"], row["method_id"], -int(row["target_branch_index"])))


def build_directed_probe_spec(*, case_path: Path, policy_path: Path, split_path: Path, manifest_path: Path, model_path: Path, q0_grid_path: Path, q95_grid_path: Path, output_path: Path) -> dict[str, Any]:
    policy_payload = _load(policy_path)
    policy = load_synthetic_rate_policy(policy_payload)
    split = _load(split_path)
    if not split_hash_is_self_consistent(split):
        raise ValueError("development/evaluation split identity is missing or not self-consistent")
    manifest = _load(manifest_path)
    model = _load(model_path)
    grids = {"Q0": _load(q0_grid_path), "Q95": _load(q95_grid_path)}
    if manifest.get("policy_hash") != policy.policy_hash.to_json() or model.get("policy_hash") != policy.policy_hash.to_json():
        raise ValueError("policy hash binding mismatch")
    if manifest.get("status") != "DIAGNOSTIC_ONLY" or model.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("manifest/model must be diagnostic-only")
    if model.get("input_field_policy", {}).get("ac_truth_read") is not False:
        raise ValueError("proxy model is not AC-truth-free")
    if tuple(split.get("development_scenario_ids", ())) != tuple(item.value for item in policy.development_scenario_ids.values):
        raise ValueError("development split mismatch")
    if any(grid.get("development_only") is not True or grid.get("evaluation_scenarios_excluded") is not True for grid in grids.values()):
        raise ValueError("directed probes require development-only grids")
    if any(grid.get("grid_hash") != manifest.get("source_grid_hashes", {}).get(q) for q, grid in grids.items()):
        raise ValueError("source grid hash mismatch")
    parsed = parse_matpower_case(case_path, expected_sha256=policy.raw_case_hash.to_json())
    bus_ids = [int(bus.bus_id) for bus in parsed.network.buses]
    reference_indices = [index for index, bus_type in enumerate(parsed.bus_types) if bus_type == 3]
    voltage_indices = [index for index in range(len(bus_ids)) if index not in set(reference_indices)]
    rates = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
    floor_dominant = {
        index: float(row["minimum_mva"]) >= (1.0 + float(row["headroom_ratio"])) * float(row["anchor_mva"])
        for index, row in enumerate(policy_payload["branch_ratings"])
    }
    anchor_branches = [index for index in range(len(rates)) if not floor_dominant[index]]
    directions = list(_probe_direction_rows(grid_payloads=grids, manifest=manifest, model=model, policy_payload=policy_payload, voltage_indices=voltage_indices))
    method_rank = {method: index for index, method in enumerate(METHODS)}

    def select_branch(target_id: str, target: float, q_mode: str | None = None) -> dict[str, Any]:
        candidates = []
        preempted = []
        for direction in directions:
            if q_mode is not None and direction["q_mode"] != q_mode:
                continue
            for index in anchor_branches:
                candidate = _candidate_probe(target_id=target_id, target_kind="BRANCH", direction=direction, policy_payload=policy_payload, branch_index=index, target_value=target)
                if candidate["status"] in {"TARGET_REACHED_BY_BISECTION", "TARGET_REACHED_BY_GRID_POINT"}:
                    candidates.append(candidate)
                elif candidate["status"] == "TARGET_PREEMPTED_BY_OTHER_CONSTRAINT":
                    preempted.append(candidate)
        if not candidates:
            if preempted:
                return min(preempted, key=lambda row: (float(row.get("preemption", {}).get("other_branch_ratio") or float("inf")), row["q_mode"], row["reporter_id"], row["allocation_side"], method_rank.get(row["method_id"], 99), int(row["target_branch_index"])))
            return {"target_id": target_id, "target_kind": "BRANCH", "status": "TARGET_UNREACHABLE_WITHIN_DOMAIN", "target_value": target, "q_mode": q_mode, "selection_pool": "ANCHOR_HEADROOM_DOMINANT_ONLY"}
        return min(candidates, key=lambda row: (0 if row["status"] == "TARGET_REACHED_BY_BISECTION" else 1, abs(float(row.get("metric_at_probe", target)) - target), row["q_mode"], row["reporter_id"], row["allocation_side"], method_rank.get(row["method_id"], 99), int(row["target_branch_index"])))

    def select_voltage(target_id: str, kind: str, q_mode: str | None = None) -> dict[str, Any]:
        attempts = []
        for target, fallback in ((PRIMARY_VOLTAGE_TARGET, False), (FALLBACK_VOLTAGE_TARGET, True)):
            candidates = []
            for direction in directions:
                if q_mode is not None and direction["q_mode"] != q_mode:
                    continue
                for index in voltage_indices:
                    candidate = _candidate_probe(target_id=target_id, target_kind=kind, direction=direction, policy_payload=policy_payload, voltage_index=index, target_value=target, fallback_used=fallback)
                    if candidate["status"] in {"TARGET_REACHED_BY_BISECTION", "TARGET_REACHED_BY_GRID_POINT"}:
                        candidates.append(candidate)
            if candidates:
                return min(candidates, key=lambda row: (0 if row["status"] == "TARGET_REACHED_BY_BISECTION" else 1, abs(float(row.get("metric_at_probe", target)) - target), row["q_mode"], row["reporter_id"], row["allocation_side"], method_rank.get(row["method_id"], 99), int(row["target_voltage_index"])))
            attempts.append({"target": target, "fallback": fallback, "candidate_count": 0})
        return {"target_id": target_id, "target_kind": kind, "status": "TARGET_UNREACHABLE_WITHIN_DOMAIN", "q_mode": q_mode, "target_value": FALLBACK_VOLTAGE_TARGET, "fallback_used": True, "attempts": attempts}

    def select_joint(target_id: str, kind: str) -> dict[str, Any]:
        candidates = []
        for direction in directions:
            if direction["q_mode"] != "Q95":
                continue
            # Joint search is deliberately stratified before bisection.  The
            # full AC screen still retains every branch/bus, but target search
            # uses a deterministic proxy-only candidate pool: branches that
            # can reach the 95% target at a domain endpoint and the 16
            # non-reference buses closest to the declared voltage threshold.
            rates_local = [float(row["rate_a_mva"]) for row in policy_payload["branch_ratings"]]
            lower_local = float(policy_payload["voltage_lower_limit_pu"])
            upper_local = float(policy_payload["voltage_upper_limit_pu"])
            metrics_zero = _metrics(_directional_values(direction, 0.0), rates_local, lower_local, upper_local, voltage_indices=voltage_indices)
            lambda_max = _domain_max(direction["allocation_mw"], 1.2)
            metrics_end = _metrics(_directional_values(direction, lambda_max), rates_local, lower_local, upper_local, voltage_indices=voltage_indices)
            branch_pool = [
                index for index in anchor_branches
                if metrics_zero["branch_ratios"][index] <= JOINT_BRANCH_TARGET + TARGET_TOLERANCE
                and metrics_end["branch_ratios"][index] >= JOINT_BRANCH_TARGET - TARGET_TOLERANCE
            ]
            branch_pool = sorted(
                branch_pool,
                key=lambda index: (abs(metrics_end["branch_ratios"][index] - JOINT_BRANCH_TARGET), index),
            )[:16]
            voltage_key = "lower_slacks" if kind == "JOINT_LOWER" else "upper_slacks"
            voltage_pool = sorted(
                voltage_indices,
                key=lambda index: (
                    min(
                        float(metrics_zero[voltage_key][voltage_indices.index(index)]),
                        float(metrics_end[voltage_key][voltage_indices.index(index)]),
                    ),
                    index,
                ),
            )[:16]
            for branch_index in branch_pool:
                for voltage_index in voltage_pool:
                    candidate = _candidate_probe(target_id=target_id, target_kind=kind, direction=direction, policy_payload=policy_payload, branch_index=branch_index, voltage_index=voltage_index, target_value=JOINT_BRANCH_TARGET)
                    if candidate["status"] in {"TARGET_REACHED_BY_BISECTION", "TARGET_REACHED_BY_GRID_POINT"}:
                        candidates.append(candidate)
        if not candidates:
            return {"target_id": target_id, "target_kind": kind, "status": "TARGET_UNREACHABLE_WITHIN_DOMAIN", "branch_target": JOINT_BRANCH_TARGET, "voltage_slack_target": JOINT_VOLTAGE_TARGET, "selection_pool": "Q95_ANCHOR_HEADROOM_DOMINANT_ONLY"}
        return min(candidates, key=lambda row: (0 if row["status"] == "TARGET_REACHED_BY_BISECTION" else 1, row.get("lambda_scale", float("inf")), row["reporter_id"], row["allocation_side"], method_rank.get(row["method_id"], 99), int(row["target_branch_index"]), int(row["target_voltage_index"])))

    floor_indices = [index for index, is_floor in floor_dominant.items() if is_floor]
    probes = [
        select_branch("BRANCH_95_RATE", 0.95),
        select_branch("BRANCH_99_RATE", 0.99),
        select_branch("Q0_BRANCH_95_RATE", 0.95, "Q0"),
        select_branch("Q95_BRANCH_95_RATE", 0.95, "Q95"),
        select_voltage("VOLTAGE_LOWER_NEAR", "VOLTAGE_LOWER"),
        select_voltage("Q0_VOLTAGE_LOWER_NEAR", "VOLTAGE_LOWER", "Q0"),
        select_voltage("Q95_VOLTAGE_LOWER_NEAR", "VOLTAGE_LOWER", "Q95"),
        select_voltage("VOLTAGE_UPPER_NEAR", "VOLTAGE_UPPER"),
        _floor_extremum_probe(target_id="Q0_FLOOR_DOMINANT_MAX_LOADING", q_mode="Q0", directions=directions, policy_payload=policy_payload, floor_indices=floor_indices, voltage_indices=voltage_indices),
        _floor_extremum_probe(target_id="Q95_FLOOR_DOMINANT_MAX_LOADING", q_mode="Q95", directions=directions, policy_payload=policy_payload, floor_indices=floor_indices, voltage_indices=voltage_indices),
        select_joint("Q95_JOINT_LOWER", "JOINT_LOWER"),
        select_joint("Q95_JOINT_UPPER", "JOINT_UPPER"),
    ]
    for probe in probes:
        branch_index = probe.get("target_branch_index")
        if isinstance(branch_index, int) and 0 <= branch_index < len(policy_payload["branch_ratings"]):
            branch_row = policy_payload["branch_ratings"][branch_index]
            probe["target_branch_id"] = branch_row.get("branch_id")
            probe["target_branch_class"] = branch_row.get("branch_class")
            probe["target_branch_floor_dominant"] = bool(floor_dominant[branch_index])
        voltage_index = probe.get("target_voltage_index")
        if isinstance(voltage_index, int) and 0 <= voltage_index < len(bus_ids):
            probe["target_voltage_bus_id"] = bus_ids[voltage_index]
    result: dict[str, Any] = {
        "serialization_id": "ieee141_m1_v2_directed_probe_spec.v1",
        "status": "DIAGNOSTIC_ONLY",
        "policy_hash": policy.policy_hash.to_json(),
        "policy_version": policy.version.to_json(),
        "network_hash": policy.network_hash.to_json(),
        "split_hash": declared_split_hash(split),
        "proxy_model_hash": model.get("result_hash"),
        "proxy_audit_hash": model.get("source_audit_hash"),
        "source_probe_manifest_hash": manifest.get("result_hash"),
        "development_reporter_ids": list(TRAIN_REPORTERS),
        "q_modes": list(Q_MODES),
        "method_ids": list(METHODS),
        "allocation_sides": list(SIDES),
        "target_definitions": {
            "BRANCH_95_RATE": {"kind": "BRANCH", "target_ratio": 0.95, "branch_pool": "ANCHOR_HEADROOM_DOMINANT_ONLY", "global_preemption_check": True},
            "BRANCH_99_RATE": {"kind": "BRANCH", "target_ratio": 0.99, "branch_pool": "ANCHOR_HEADROOM_DOMINANT_ONLY", "global_preemption_check": True},
            "Q0_BRANCH_95_RATE": {"kind": "BRANCH", "target_ratio": 0.95, "q_mode": "Q0", "branch_pool": "ANCHOR_HEADROOM_DOMINANT_ONLY", "global_preemption_check": True},
            "Q95_BRANCH_95_RATE": {"kind": "BRANCH", "target_ratio": 0.95, "q_mode": "Q95", "branch_pool": "ANCHOR_HEADROOM_DOMINANT_ONLY", "global_preemption_check": True},
            "VOLTAGE_LOWER_NEAR": {"kind": "VOLTAGE_LOWER", "primary_slack": PRIMARY_VOLTAGE_TARGET, "fallback_slack": FALLBACK_VOLTAGE_TARGET, "reference_bus_excluded": True},
            "Q0_VOLTAGE_LOWER_NEAR": {"kind": "VOLTAGE_LOWER", "q_mode": "Q0", "primary_slack": PRIMARY_VOLTAGE_TARGET, "fallback_slack": FALLBACK_VOLTAGE_TARGET, "reference_bus_excluded": True},
            "Q95_VOLTAGE_LOWER_NEAR": {"kind": "VOLTAGE_LOWER", "q_mode": "Q95", "primary_slack": PRIMARY_VOLTAGE_TARGET, "fallback_slack": FALLBACK_VOLTAGE_TARGET, "reference_bus_excluded": True},
            "VOLTAGE_UPPER_NEAR": {"kind": "VOLTAGE_UPPER", "primary_slack": PRIMARY_VOLTAGE_TARGET, "fallback_slack": FALLBACK_VOLTAGE_TARGET, "reference_bus_excluded": True},
            "Q0_FLOOR_DOMINANT_MAX_LOADING": {"kind": "BRANCH_EXTREMUM", "q_mode": "Q0", "selection": "MAX_PROXY_LOADING_RATIO_OVER_FLOOR_DOMINANT_AND_DEV_TRAIN_DIRECTIONS"},
            "Q95_FLOOR_DOMINANT_MAX_LOADING": {"kind": "BRANCH_EXTREMUM", "q_mode": "Q95", "selection": "MAX_PROXY_LOADING_RATIO_OVER_FLOOR_DOMINANT_AND_DEV_TRAIN_DIRECTIONS"},
            "Q95_JOINT_LOWER": {"kind": "JOINT_LOWER", "branch_ratio": JOINT_BRANCH_TARGET, "voltage_slack": JOINT_VOLTAGE_TARGET, "q_mode": "Q95", "reference_bus_excluded": True},
            "Q95_JOINT_UPPER": {"kind": "JOINT_UPPER", "branch_ratio": JOINT_BRANCH_TARGET, "voltage_slack": JOINT_VOLTAGE_TARGET, "q_mode": "Q95", "reference_bus_excluded": True},
        },
        "reference_bus_ids_excluded_from_voltage_target": [bus_ids[index] for index in reference_indices],
        "screened_voltage_indices": list(voltage_indices),
        "floor_dominant_branch_count": sum(floor_dominant.values()),
        "anchor_headroom_dominant_branch_count": len(anchor_branches),
        "selection_rule": "PROXY_ONLY_MONOTONIC_BRACKET_THEN_BISECTION_DETERMINISTIC_TIE_BREAK",
        "domain_policy": {"lambda_requested_max": 1.2, "clipping": False, "allocation_resolve_per_lambda": False, "out_of_domain_status": "DOMAIN_EXCLUSION_AVAILABILITY"},
        "monotonicity_policy": {"tolerance": MONOTONIC_TOLERANCE, "nonmonotone_status": "TARGET_NONMONOTONE_NO_VALID_BRACKET"},
        "probes": probes,
        "evaluation_results_read": False,
        "mitigation_results_read": False,
        "ac_truth_read": False,
        "allocation_resolved": False,
        "rate_policy_mutated": False,
        "diagnostic_only": True,
        "primary_evidence_eligible": False,
        "t3_eligible": False,
        "t4_eligible": False,
        "t5_eligible": False,
        "result_hash": None,
    }
    result["result_hash"] = canonical_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_dumps(result) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join([
        "# M1 V2 proxy-only directed probe specification",
        "",
        "- Status: `DIAGNOSTIC_ONLY`",
        f"- RATE policy hash: `{result['policy_hash']}`",
        f"- Proxy model hash: `{result['proxy_model_hash']}`",
        f"- Probe statuses: `{[(row.get('target_id'), row.get('status')) for row in probes]}`",
        "- Selection consumes only frozen proxy coefficients, development allocation directions, domain bounds, and RATE/voltage policy.",
        "- AC truth, evaluation, mitigation, pass/fail, residual and allocation re-solve fields are not read.",
        "- Floor-dominant branches remain in the eventual AC screen but are excluded from target-search dominance.",
        "",
        "This artifact fixes probe inputs; it does not authorize AC evidence or T3/T4/T5.",
    ]) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proxy-model", type=Path, required=True)
    parser.add_argument("--q0-grid", type=Path, required=True)
    parser.add_argument("--q95-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_dumps(build_directed_probe_spec(case_path=args.case, policy_path=args.policy, split_path=args.split, manifest_path=args.manifest, model_path=args.proxy_model, q0_grid_path=args.q0_grid, q95_grid_path=args.q95_grid, output_path=args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_directed_probe_spec"]
