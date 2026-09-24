"""Verify deposited 12-schedule replay-selection results without rerunning AC."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction/outputs/ees_transition/qualification_value_12seed"
OLD = ROOT / "reproduction/outputs/ees_transition/qualification_value"
CODE = ROOT / "reproduction/diagnostics/qualification_value_12seed"
SEEDS = list(range(2401, 2413))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, label: str, atol: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = json.loads((DATA / "SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    assert sha(DATA / "INPUT_MANIFEST.json") == summary["inputs_manifest_sha256"]
    assert sha(CODE / "DESIGN_12SEED.md") == manifest["design_sha256"]
    assert sha(CODE / "run_frequency_control.py") == manifest["implementation_sha256"]
    assert manifest["seeds"] == summary["seeds"] == SEEDS
    assert len(manifest["primary_evaluation_dates"]) == summary["dates"] == 80
    close(manifest["update_probability_given_unused_award"], 483 / 3091,
          "fixed update probability")

    daily = rows(DATA / "DAILY_RESULTS.csv")
    intervals = rows(DATA / "INTERVAL_RESULTS.csv")
    assert len(daily) == 80 * 12
    assert len(intervals) == summary["interval_records"] == 80 * 48 * 12
    by_day_seed: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in intervals:
        seed = int(row["seed"])
        assert seed in SEEDS
        by_day_seed[row["date"], seed].append(row)
        assert row["allocation_pair_pass"] == row["physical_pair_pass"] == "1"
        assert row["replay_power_flow_count"] == "0"
        close(float(row["authorized_export_kw"]) - float(row["idle_authorization_kw"]),
              float(row["delivered_export_kw"]), "interval energy identity", 1e-6)
    for row in daily:
        date, seed = row["date"], int(row["seed"])
        assert seed in SEEDS
        by_seed[seed].append(row)
        trace = by_day_seed[date, seed]
        assert len(trace) == 48
        assert [int(item["interval_index"]) for item in trace] == list(range(48))
        for interval_key, daily_key in (
            ("delivered_export_kw", "delivered_export_mwh"),
            ("authorized_export_kw", "authorized_export_mwh"),
            ("idle_authorization_kw", "idle_authorization_mwh"),
        ):
            close(math.fsum(float(item[interval_key]) for item in trace) / 2000,
                  float(row[daily_key]), f"{date}:{seed}:{daily_key}")
        assert sum(int(item["gate_triggered"]) for item in trace) == int(
            row["scheduled_update_intervals"])
        assert row["all_physical_pair_pass"] == row["all_allocation_pair_pass"] == "True"
    assert {date for date, _ in by_day_seed} == set(manifest["primary_evaluation_dates"])
    assert len(by_day_seed) == 80 * 12

    contrasts = []
    counts = []
    for seed in SEEDS:
        records = by_seed[seed]
        assert len(records) == 80
        expected = summary["seed_results"][str(seed)]
        for daily_key, summary_key in (
            ("delivered_export_mwh", "random_delivered_mwh"),
            ("authorized_export_mwh", "authorized_export_mwh"),
            ("idle_authorization_mwh", "idle_authorization_mwh"),
        ):
            close(math.fsum(float(item[daily_key]) for item in records),
                  expected[summary_key], f"{seed}:{daily_key}")
        count = sum(int(item["scheduled_update_intervals"]) for item in records)
        assert count == expected["random_update_count"]
        counts.append(count)
        contrast = summary["ecf_delivered_mwh"] - expected["random_delivered_mwh"]
        close(contrast, expected["ecf_minus_random_mwh"], f"{seed}:contrast")
        contrasts.append(contrast)
    close(math.fsum(contrasts) / 12, summary["ecf_minus_random_seed_average_mwh"],
          "12-schedule mean contrast")
    assert all(contrast > 0 for contrast in contrasts)

    # The first three trajectories were retained unchanged from the prior deposit.
    old_daily = {(item["date"], item["seed"]): item for item in rows(OLD / "DAILY_RESULTS.csv")}
    old_intervals = {(item["date"], item["seed"], item["interval_index"]): item
                     for item in rows(OLD / "INTERVAL_RESULTS.csv")}
    for item in daily:
        if int(item["seed"]) <= 2403:
            earlier = old_daily[item["date"], item["seed"]]
            for key in ("delivered_export_mwh", "authorized_export_mwh",
                        "idle_authorization_mwh", "scheduled_update_intervals"):
                close(float(item[key]), float(earlier[key]), f"retained daily {key}")
    for item in intervals:
        if int(item["seed"]) <= 2403:
            earlier = old_intervals[item["date"], item["seed"], item["interval_index"]]
            for key in ("delivered_export_kw", "authorized_export_kw",
                        "idle_authorization_kw", "gate_triggered"):
                close(float(item[key]), float(earlier[key]), f"retained interval {key}")

    print(json.dumps({
        "passed": True,
        "dates": 80,
        "schedules": 12,
        "interval_records": len(intervals),
        "update_count_range": [min(counts), max(counts)],
        "ecf_advantage_mwh_range": [min(contrasts), max(contrasts)],
        "ecf_advantage_mwh_mean": math.fsum(contrasts) / 12,
        "scope": "Saved-result arithmetic and pass flags, not an independent AC rerun",
    }, indent=2))


if __name__ == "__main__":
    main()
