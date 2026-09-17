#!/usr/bin/env python3
"""Read-only verifier for Australian network-energy endpoint statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import analyze_ees_australian_external_replay as parent_analysis
import analyze_ees_australian_network_energy_extension as endpoint_analysis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLOAT_TOLERANCE = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(actual: float, expected: float, message: str) -> None:
    scale = max(1.0, abs(actual), abs(expected))
    _require(abs(actual - expected) <= FLOAT_TOLERANCE * scale, message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/australian_network_energy_extension_v1_2",
    )
    return parser.parse_args()


def _verify_metric(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    scalar_keys = (
        "mean_difference_mwh_per_network_day",
        "median_difference_mwh_per_network_day",
        "positive_day_fraction",
        "zero_day_fraction",
        "aggregate_difference_mwh_over_observed_days",
    )
    for key in scalar_keys:
        _close(float(actual[key]), float(expected[key]), f"{label}/{key}")
    _require(
        int(actual["network_day_count"]) == int(expected["network_day_count"]),
        f"{label}/network_day_count",
    )
    for index, value in enumerate(expected["bootstrap_95_ci_mean_mwh_per_network_day"]):
        _close(
            float(actual["bootstrap_95_ci_mean_mwh_per_network_day"][index]),
            float(value),
            f"{label}/ordinary_ci/{index}",
        )
    actual_dependent = actual["date_aware_dependent_wild"]
    expected_dependent = expected["date_aware_dependent_wild"]
    for key in (
        "bootstrap_empirical_center_mwh_per_network_day",
        "target_sample_mean_mwh_per_network_day",
        "center_error_mwh_per_network_day",
        "observation_coverage_fraction",
        "minimum_covariance_eigenvalue",
    ):
        _close(
            float(actual_dependent[key]),
            float(expected_dependent[key]),
            f"{label}/dependent/{key}",
        )
    for index, value in enumerate(
        expected_dependent["bootstrap_95_ci_mean_mwh_per_network_day"]
    ):
        _close(
            float(actual_dependent["bootstrap_95_ci_mean_mwh_per_network_day"][index]),
            float(value),
            f"{label}/dependent_ci/{index}",
        )


def main() -> int:
    args = _parse_args()
    root = args.result_dir.resolve()
    before = _tree_hashes(root)
    stats_path = root / "AU_NETWORK_ENERGY_ENDPOINT_STATISTICS.json"
    manifest_path = root / "AU_NETWORK_ENERGY_ENDPOINT_STATISTICS_SHA256_MANIFEST.json"
    statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _require(
        statistics["statistics_id"]
        == "AUSTRALIAN_NETWORK_ENERGY_ENDPOINT_STATISTICS_V1",
        "wrong statistics id",
    )
    _require(statistics["eligible_network_day_count"] == 80, "wrong day count")
    _require(statistics["paired_unit"] == "network_day", "wrong paired unit")
    _require(
        statistics["bootstrap_replicates"] == parent_analysis.BOOTSTRAP_REPLICATES,
        "wrong bootstrap replicate count",
    )
    _require(
        statistics["dependence_bandwidth_calendar_days"]
        == parent_analysis.DEPENDENCE_BANDWIDTH_DAYS,
        "wrong dependence bandwidth",
    )
    _require(
        statistics["resampling_seed_derivation"] == endpoint_analysis.SEED_DERIVATION,
        "wrong resampling seed derivation",
    )

    input_paths = statistics["input_paths"]
    input_hashes = statistics["input_hashes"]
    _require(set(input_paths) == set(input_hashes), "input path/hash roster mismatch")
    for name, relative in input_paths.items():
        path = REPOSITORY_ROOT / relative
        _require(path.is_file(), f"missing statistics input: {name}")
        _require(_sha256(path) == input_hashes[name], f"statistics input drift: {name}")
    protocol = yaml.safe_load(
        (REPOSITORY_ROOT / input_paths["statistics_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    _require(protocol["version"] == "1.0.3", "wrong statistics protocol version")
    _require(
        tuple(protocol["endpoints"]) == endpoint_analysis.METRICS,
        "statistics protocol endpoint roster mismatch",
    )
    _require(len(manifest["payload"]) == 1, "wrong statistics payload count")
    record = manifest["payload"][0]
    _require(record["path"] == stats_path.name, "wrong statistics payload path")
    _require(record["sha256"] == _sha256(stats_path), "statistics payload hash mismatch")
    _require(record["bytes"] == stats_path.stat().st_size, "statistics payload size mismatch")

    day_path = REPOSITORY_ROOT / input_paths["extension_day_summary"]
    rows = _read_csv(day_path)
    by_key = {(row["date"], row["control"]): row for row in rows}
    dates = sorted({row["date"] for row in rows})
    _require(len(rows) == 320 and len(dates) == 80, "statistics source panel mismatch")
    expected_comparisons = {
        f"ECF_MINUS_{baseline}" for baseline in endpoint_analysis.BASELINES
    }
    _require(
        set(statistics["comparisons"]) == expected_comparisons,
        "statistics comparison roster mismatch",
    )
    for comparison in expected_comparisons:
        _require(
            set(statistics["comparisons"][comparison]["metrics"])
            == set(endpoint_analysis.METRICS),
            f"statistics metric roster mismatch: {comparison}",
        )
    for baseline in endpoint_analysis.BASELINES:
        comparison = f"ECF_MINUS_{baseline}"
        for metric in endpoint_analysis.METRICS:
            values = np.asarray(
                [
                    endpoint_analysis._paired_values(
                        by_key[(date, endpoint_analysis.PRIMARY_CONTROL)],
                        by_key[(date, baseline)],
                        metric,
                    )
                    for date in dates
                ],
                dtype=float,
            )
            expected = {
                **parent_analysis._paired_bootstrap(
                    values,
                    endpoint_analysis._rng_for(comparison, metric, "ordinary"),
                ),
                "date_aware_dependent_wild": parent_analysis._date_aware_dependent_wild_bootstrap(
                    values,
                    dates,
                    endpoint_analysis._rng_for(
                        comparison, metric, "date_aware_dependent_wild"
                    ),
                    bandwidth_days=parent_analysis.DEPENDENCE_BANDWIDTH_DAYS,
                ),
            }
            actual = statistics["comparisons"][comparison]["metrics"][metric]
            _verify_metric(actual, expected, f"{comparison}/{metric}")

    after = _tree_hashes(root)
    _require(before == after, "statistics verifier modified the evidence tree")
    report = {
        "status": "PASS",
        "statistics_id": statistics["statistics_id"],
        "eligible_network_day_count": len(dates),
        "comparison_count": len(endpoint_analysis.BASELINES),
        "metric_count_per_comparison": len(endpoint_analysis.METRICS),
        "input_hash_count": len(input_paths),
        "payload_hash_count": len(manifest["payload"]),
        "statistics_recomputed": True,
        "tree_unchanged": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
