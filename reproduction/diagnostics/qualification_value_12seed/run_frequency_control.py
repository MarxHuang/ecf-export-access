"""Reproduce a frequency-matched, replay-free feedback diagnostic.

This is intentionally a post hoc mechanism comparison. See DESIGN.md.
The frozen Australian scientific core and inputs are loaded from the archived
80-day evaluation; this script does not change the allocation or AC solvers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
NC_ROOT = HERE.parents[1]
ARCHIVE = NC_ROOT.parents[1] / "ees_transition" / "archive"
REFERENCE = ARCHIVE / "169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908"
ORIGINAL = ARCHIVE / "151_FROZEN_SCENARIO_EVALUATION_20260907"
CAPTURE = ARCHIVE / "157_PROPORTIONAL_SEARCH_AUDIT_20260907" / "v2"
TRAJECTORIES = ARCHIVE / "164_CEILING_TRAJECTORY_REVIEW_20260907"
REFERENCE_PROTOCOL = REFERENCE / "PROTOCOL.json"
RESULTS = HERE / "results_12seed"
EARLIER_RESULTS = HERE / "results"
SEEDS = tuple(range(2401, 2413))
P_UPDATE = 483 / 3091  # Original ECF qualifications / unused-award intervals.
INTERVAL_HOURS = 0.5
NUMERICAL_TOLERANCE_KW = 1e-7


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference():
    path = REFERENCE / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("frozen_same_update_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_draw(seed: int, date: str, interval: int) -> float:
    identifier = f"ECF_RATE_DIAGNOSTIC_V1:{seed}:{date}:{interval}".encode()
    return int.from_bytes(hashlib.sha256(identifier).digest()[:8], "big") / 2**64


def input_manifest() -> dict:
    reference = load_reference()
    protocol = reference.verify_protocol()
    assert len(protocol["dates"]) == 80
    assert protocol["config"]["load_scale"] == 0.3
    assert protocol["beta"] == 0.5
    old_rows = list(csv.DictReader(
        (ORIGINAL / "runs" / "primary" / "AU_EXTERNAL_INTERVAL_RESULTS.csv").open(
            encoding="utf-8-sig", newline=""
        )
    ))
    ecf = [row for row in old_rows if row["control"] == "ECF"]
    assert len(ecf) == 80 * 48
    assert sum(int(row["gate_triggered"]) for row in ecf) == 483
    assert sum(float(row["idle_authorization_kw"]) > 0.002 for row in ecf) == 3091
    assert abs(P_UPDATE - 483 / 3091) < 1e-15
    return {
        "design": "post-hoc frequency-matched information diagnostic",
        "primary_evaluation_dates": protocol["dates"],
        "seeds": list(SEEDS),
        "update_probability_given_unused_award": P_UPDATE,
        "probability_origin": "483/3091 in existing ECF evaluation; no delivery optimization",
        "scientific_source_sha256": sha(ORIGINAL / "scientific_source" / "tools" / "run_ees_australian_external_replay.py"),
        "reference_protocol_sha256": sha(REFERENCE_PROTOCOL),
        "design_sha256": sha(HERE / "DESIGN_12SEED.md"),
        "implementation_sha256": sha(Path(__file__)),
        "frozen_config": protocol["config"],
    }


def day_task(date: str) -> dict:
    reference = load_reference()
    model_api = reference.original()
    config = json.loads(REFERENCE_PROTOCOL.read_text(encoding="utf-8"))["config"]
    captured_path = CAPTURE / "days" / date / "CAPTURED_RAYS.npz"
    trace_path = TRAJECTORIES / "trajectories" / f"{date}.npz"
    with np.load(captured_path) as captured:
        participant = captured["participant"].astype(bool)
        original_index = {
            int(t): j for j, t in enumerate(captured["interval_index"])
            if captured["control"][j] == "NO_FEEDBACK"
        }
        assert set(original_index) == set(range(48))
        raw_mw = np.stack([
            captured["request_kw"][original_index[t]] / 1000 for t in range(48)
        ])
        loads_kw = np.stack([
            captured["load_kw"][original_index[t]] for t in range(48)
        ])
    with np.load(trace_path) as old_trace:
        capacity_mw = np.zeros(len(participant))
        capacity_mw[participant] = old_trace["capacity_mw"]
        original_first_award = old_trace["beta_0.5_award"][0]
        original_first_delivery = old_trace["beta_0.5_delivery"][0]

    connections = model_api.parse_dss_load_connections(config["dss_root"])
    profile, selected = model_api._load_mapping(Path(config["mapping_file"]), connections)
    assert np.array_equal(selected, participant)
    with np.load(Path(config["package"]) / "AUSGRID_EVALUATION_2012_2013.npz") as data:
        day_index = list(map(str, data["dates"])).index(date)
        gross_kw = model_api.PV_SCALE * data["gross_generation_kw"][day_index].astype(float)[:, profile]
        calculated_loads = .3 * data["household_load_kw"][day_index].astype(float)[:, profile]
    assert np.max(np.abs(calculated_loads - loads_kw)) < 1e-9

    earlier = json.loads((EARLIER_RESULTS / "days" / f"{date}.json").read_text(encoding="utf-8"))
    assert earlier["date"] == date
    assert [entry["seed"] for entry in earlier["seeds"]] == list(SEEDS[:3])
    result = {"date": date, "seeds": earlier["seeds"].copy(), "input_sha256": {
        "captured_rays": sha(captured_path), "original_trace": sha(trace_path)
    }}
    assert result["input_sha256"] == earlier["input_sha256"]
    for seed in SEEDS[3:]:
        start = time.perf_counter()
        model_api._WORKER["model"] = model_api.AustralianOpenDSSModel(
            Path(config["dss_root"]), connections,
            load_power_factor=model_api.LOAD_POWER_FACTOR,
        )
        ceiling_mw = capacity_mw.copy()
        intervals = []
        for t in range(48):
            admitted_mw = np.asarray(
                model_api.apply_access_ceiling(raw_mw[t], capacity_mw, ceiling_mw),
                dtype=float,
            )
            outcome = model_api._allocate_and_realize(
                loads_kw[t], gross_kw[t], admitted_mw, participant
            )
            if t == 0:
                assert np.max(np.abs(outcome["allocation_mw"][participant] - original_first_award)) * 1000 <= NUMERICAL_TOLERANCE_KW
                assert np.max(np.abs(outcome["delivery_mw"][participant] - original_first_delivery)) * 1000 <= NUMERICAL_TOLERANCE_KW
            _, idle = model_api.fulfillment_replay_request(
                admitted_mw, outcome["allocation_mw"], outcome["delivery_mw"],
                tolerance_mw=model_api.REPLAY_TOLERANCE_MW,
            )
            draw = deterministic_draw(seed, date, t)
            update = bool(len(idle) and draw < P_UPDATE)
            next_ceiling, target = model_api.update_access_ceiling(
                ceiling_before_mw=ceiling_mw,
                registered_capacity_mw=capacity_mw,
                delivery_mw=outcome["delivery_mw"],
                idle_indices=idle,
                gate_triggered=update,
                beta=.5,
            )
            next_ceiling = np.asarray(next_ceiling, dtype=float)
            target = np.asarray(target, dtype=float)
            assert np.all((next_ceiling >= -1e-12) & (next_ceiling <= capacity_mw + 1e-12))
            assert np.max(np.abs(next_ceiling - (.5 * ceiling_mw + .5 * target))) < 1e-12
            row = model_api._interval_record(
                date=date, interval=t, control="FREQUENCY_MATCHED_RANDOM",
                raw_request_mw=raw_mw[t], admitted_request_mw=admitted_mw,
                outcome=outcome, gate=update,
                gate_reason="SCHEDULED_WITH_UNUSED_AWARD" if update else "NO_SCHEDULED_UPDATE",
            )
            row["unused_award_members"] = len(idle)
            row["scheduled_draw"] = draw
            row["positive_ceiling_reduction_kw"] = float(np.sum(
                np.maximum(ceiling_mw - next_ceiling, 0.0)
            ) * 1000)
            row["replay_power_flow_count"] = 0
            intervals.append(row)
            ceiling_mw = next_ceiling
        assert len(intervals) == 48
        result["seeds"].append({
            "seed": seed,
            "wall_seconds": time.perf_counter() - start,
            "intervals": intervals,
            "summary": {
                "delivered_export_mwh": sum(r["delivered_export_kw"] for r in intervals) * INTERVAL_HOURS / 1000,
                "authorized_export_mwh": sum(r["authorized_export_kw"] for r in intervals) * INTERVAL_HOURS / 1000,
                "idle_authorization_mwh": sum(r["idle_authorization_kw"] for r in intervals) * INTERVAL_HOURS / 1000,
                "source_terminal_exchange_mwh": sum(r["actual_source_terminal_p_kw"] for r in intervals) * INTERVAL_HOURS / 1000,
                "eligible_unused_intervals": sum(r["unused_award_members"] > 0 for r in intervals),
                "scheduled_update_intervals": sum(r["gate_triggered"] for r in intervals),
                "positive_ceiling_reduction_kw_sum": sum(r["positive_ceiling_reduction_kw"] for r in intervals),
                "necessary_power_flow_count": sum(r["power_flow_count"] for r in intervals),
                "all_physical_pair_pass": all(r["physical_pair_pass"] for r in intervals),
                "all_allocation_pair_pass": all(r["allocation_pair_pass"] for r in intervals),
                "minimum_customer_voltage_pu": min(r["actual_voltage_min_pu"] for r in intervals),
                "maximum_customer_voltage_pu": max(r["actual_voltage_max_pu"] for r in intervals),
                "maximum_transformer_loading_pu": max(r["actual_transformer_max_loading_pu"] for r in intervals),
            },
        })
    out = RESULTS / "days" / f"{date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"date": date, "sha256": sha(out), "seconds": sum(x["wall_seconds"] for x in result["seeds"])}


def run(workers: int, limit: int | None):
    manifest = input_manifest()
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / "INPUT_MANIFEST.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dates = manifest["primary_evaluation_dates"][:limit]
    pending = [d for d in dates if not (RESULTS / "days" / f"{d}.json").exists()]
    print(json.dumps({"dates": len(dates), "pending": len(pending), "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(day_task, date): date for date in pending}
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def date_dependent_interval(values: np.ndarray, dates: list[str], seed: int = 2409) -> tuple[float, float]:
    from datetime import date

    day_numbers = np.asarray([date.fromisoformat(d).toordinal() for d in dates])
    distance = np.abs(day_numbers[:, None] - day_numbers[None, :])
    covariance = np.maximum(1 - distance / 8, 0)
    rng = np.random.default_rng(seed)
    multipliers = rng.multivariate_normal(np.zeros(len(dates)), covariance, size=5000)
    multipliers = np.concatenate([multipliers, -multipliers])
    mean = float(np.mean(values))
    replicates = mean + multipliers @ (values - mean) / len(dates)
    return tuple(float(x) for x in np.quantile(replicates, [.025, .975]))


def summarize():
    manifest = input_manifest()
    dates = manifest["primary_evaluation_dates"]
    original_days = list(csv.DictReader(
        (ORIGINAL / "runs" / "primary" / "AU_EXTERNAL_DAY_SUMMARY.csv").open(
            encoding="utf-8-sig", newline=""
        )
    ))
    original_ecf = {r["date"]: r for r in original_days if r["control"] == "ECF"}
    original_no_feedback = {r["date"]: r for r in original_days if r["control"] == "NO_FEEDBACK"}
    assert set(original_ecf) == set(dates) == set(original_no_feedback)
    daily_rows = []
    interval_rows = []
    for d in dates:
        path = RESULTS / "days" / f"{d}.json"
        assert path.is_file(), f"Missing result date: {d}"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert [x["seed"] for x in record["seeds"]] == list(SEEDS)
        for run in record["seeds"]:
            row = {"date": d, "seed": run["seed"], **run["summary"]}
            row["ecf_delivered_export_mwh"] = float(original_ecf[d]["delivered_export_mwh"])
            row["no_feedback_delivered_export_mwh"] = float(original_no_feedback[d]["delivered_export_mwh"])
            row["ecf_minus_random_mwh"] = row["ecf_delivered_export_mwh"] - row["delivered_export_mwh"]
            daily_rows.append(row)
            for interval in run["intervals"]:
                interval_rows.append({"date": d, "seed": run["seed"], **interval})
    assert len(daily_rows) == len(dates) * len(SEEDS)
    assert len(interval_rows) == len(dates) * len(SEEDS) * 48

    def write_csv(path: Path, rows: list[dict]):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(RESULTS / "DAILY_RESULTS.csv", daily_rows)
    write_csv(RESULTS / "INTERVAL_RESULTS.csv", interval_rows)
    daily_by_date = {d: [r for r in daily_rows if r["date"] == d] for d in dates}
    averaged = np.asarray([
        np.mean([r["ecf_minus_random_mwh"] for r in daily_by_date[d]]) for d in dates
    ])
    seed_summaries = {}
    for seed in SEEDS:
        rows = [r for r in daily_rows if r["seed"] == seed]
        diff = np.asarray([r["ecf_minus_random_mwh"] for r in rows])
        seed_summaries[str(seed)] = {
            "random_delivered_mwh": sum(r["delivered_export_mwh"] for r in rows),
            "ecf_minus_random_mwh": float(np.sum(diff)),
            "random_update_count": sum(r["scheduled_update_intervals"] for r in rows),
            "eligible_unused_intervals": sum(r["eligible_unused_intervals"] for r in rows),
            "positive_days": int(np.sum(diff > 1e-10)),
            "negative_days": int(np.sum(diff < -1e-10)),
            "all_physical_pass": all(r["all_physical_pair_pass"] for r in rows),
            "all_allocation_pass": all(r["all_allocation_pair_pass"] for r in rows),
            "power_flow_count": sum(r["necessary_power_flow_count"] for r in rows),
            "source_terminal_exchange_mwh": sum(r["source_terminal_exchange_mwh"] for r in rows),
            "authorized_export_mwh": sum(r["authorized_export_mwh"] for r in rows),
            "idle_authorization_mwh": sum(r["idle_authorization_mwh"] for r in rows),
        }
    longest_block_dates = [d for d in dates if "2013-05-16" <= d <= "2013-06-19"]
    assert len(longest_block_dates) == 35
    remaining = [averaged[j] for j, d in enumerate(dates) if d not in longest_block_dates]
    summary = {
        "case": "post-hoc frequency-matched randomized update",
        "dates": len(dates), "seeds": list(SEEDS), "interval_records": len(interval_rows),
        "update_probability": P_UPDATE,
        "ecf_original_update_count": 483,
        "ecf_delivered_mwh": sum(float(original_ecf[d]["delivered_export_mwh"]) for d in dates),
        "no_feedback_delivered_mwh": sum(float(original_no_feedback[d]["delivered_export_mwh"]) for d in dates),
        "ecf_minus_random_seed_average_mwh": float(np.sum(averaged)),
        "ecf_minus_random_daily_mean_kwh": float(np.mean(averaged) * 1000),
        "ecf_minus_random_95pct_daily_mean_kwh": [x * 1000 for x in date_dependent_interval(averaged, dates)],
        "positive_days_seed_averaged": int(np.sum(averaged > 1e-10)),
        "negative_days_seed_averaged": int(np.sum(averaged < -1e-10)),
        "longest_35_day_block_contrast_mwh": float(np.sum([averaged[j] for j,d in enumerate(dates) if d in longest_block_dates])),
        "remaining_45_day_contrast_mwh": float(np.sum(remaining)),
        "seed_results": seed_summaries,
        "inputs_manifest_sha256": sha(RESULTS / "INPUT_MANIFEST.json"),
    }
    (RESULTS / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "summarize"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "run":
        run(args.workers, args.limit)
    else:
        summarize()
