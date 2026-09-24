"""Verify deposited replay-selection results without solving the AC model.

This checks arithmetic and saved screen flags. It does not independently
validate the published network, third-party input data or AC implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction/outputs/ees_transition/qualification_value"
CODE = ROOT / "reproduction/diagnostics/qualification_value"


def close(actual: float, expected: float, label: str, atol: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    summary = json.loads((DATA / "SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    assert sha(DATA / "INPUT_MANIFEST.json") == summary["inputs_manifest_sha256"]
    assert sha(CODE / "run_frequency_control.py") == manifest["implementation_sha256"]
    assert sha(CODE / "DESIGN.md") == manifest["design_sha256"]
    assert manifest["update_probability_given_unused_award"] == 483 / 3091
    seeds = summary["seeds"]
    assert seeds == [2401, 2402, 2403]

    daily = read_csv(DATA / "DAILY_RESULTS.csv")
    intervals = read_csv(DATA / "INTERVAL_RESULTS.csv")
    assert len(daily) == 80 * len(seeds)
    assert len(intervals) == 80 * 48 * len(seeds) == summary["interval_records"]
    by_seed_day: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in intervals:
        seed = int(row["seed"])
        assert seed in seeds
        key = (seed, row["date"])
        by_seed_day[key].append(row)
        assert row["allocation_pair_pass"] == "1"
        assert row["physical_pair_pass"] == "1"
        assert row["actual_converged"] == "1"
        assert row["replay_power_flow_count"] == "0"
        assert int(row["gate_triggered"]) <= int(int(row["unused_award_members"]) > 0)
        close(float(row["authorized_export_kw"]) - float(row["idle_authorization_kw"]),
              float(row["delivered_export_kw"]), "interval energy identity", 1e-6)

    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_day: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in daily:
        seed = int(row["seed"])
        date = row["date"]
        by_seed[seed].append(row)
        by_day[date].append(row)
        trace = by_seed_day[seed, date]
        assert len(trace) == 48
        close(sum(float(r["delivered_export_kw"]) for r in trace) / 2000,
              float(row["delivered_export_mwh"]), "daily delivery")
        close(sum(float(r["authorized_export_kw"]) for r in trace) / 2000,
              float(row["authorized_export_mwh"]), "daily authorization")
        close(sum(float(r["idle_authorization_kw"]) for r in trace) / 2000,
              float(row["idle_authorization_mwh"]), "daily unused authorization")
        assert sum(int(r["gate_triggered"]) for r in trace) == int(row["scheduled_update_intervals"])
        assert row["all_physical_pair_pass"] in ("True", "1")
        assert row["all_allocation_pair_pass"] in ("True", "1")

    assert len(by_day) == 80
    assert all(len(rows) == 3 for rows in by_day.values())
    for seed in seeds:
        rows = by_seed[seed]
        assert len(rows) == 80
        expected = summary["seed_results"][str(seed)]
        for field, key in (("delivered_export_mwh", "random_delivered_mwh"),
                           ("authorized_export_mwh", "authorized_export_mwh"),
                           ("idle_authorization_mwh", "idle_authorization_mwh")):
            close(math.fsum(float(r[field]) for r in rows), expected[key], f"{seed}:{field}")
        assert sum(int(r["scheduled_update_intervals"]) for r in rows) == expected["random_update_count"]
        assert sum(int(r["necessary_power_flow_count"]) for r in rows) == expected["power_flow_count"]

    ecf_totals = []
    random_totals = []
    differences = []
    for date in sorted(by_day):
        rows = by_day[date]
        ecf = [float(r["ecf_delivered_export_mwh"]) for r in rows]
        assert max(ecf) - min(ecf) < 1e-10
        random = math.fsum(float(r["delivered_export_mwh"]) for r in rows) / 3
        ecf_totals.append(ecf[0])
        random_totals.append(random)
        differences.append(ecf[0] - random)
    close(math.fsum(ecf_totals), summary["ecf_delivered_mwh"], "ECF total")
    close(math.fsum(differences), summary["ecf_minus_random_seed_average_mwh"],
          "ECF minus random mean")
    assert sum(d > 1e-10 for d in differences) == summary["positive_days_seed_averaged"]
    assert sum(d < -1e-10 for d in differences) == summary["negative_days_seed_averaged"]
    print(json.dumps({
        "passed": True,
        "days": len(by_day),
        "interval_records": len(intervals),
        "random_update_counts": [summary["seed_results"][str(s)]["random_update_count"]
                                 for s in seeds],
        "ecf_minus_random_mean_mwh": math.fsum(differences),
        "scope": "saved-result arithmetic and flags, not independent AC rerun",
    }, indent=2))


if __name__ == "__main__":
    main()
