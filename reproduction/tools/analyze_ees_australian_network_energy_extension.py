#!/usr/bin/env python3
"""Compute frozen paired-day statistics for the Australian network endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_ees_australian_external_replay as parent_analysis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CONTROL = "ECF"
BASELINES = ("NO_FEEDBACK", "MATCHED_UNIFORM")
METRICS = (
    "upstream_net_export_energy_mwh",
    "circuit_active_loss_energy_mwh",
    "network_adjustment_difference_mwh",
)
BOOTSTRAP_SEED = parent_analysis.BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = parent_analysis.BOOTSTRAP_REPLICATES
DEPENDENCE_BANDWIDTH_DAYS = parent_analysis.DEPENDENCE_BANDWIDTH_DAYS
SEED_DERIVATION = "sha256(base_seed|comparison|metric|method), first 64 bits"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _rng_for(comparison: str, metric: str, method: str) -> np.random.Generator:
    token = f"{BOOTSTRAP_SEED}|{comparison}|{metric}|{method}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/australian_network_energy_extension_v1_2",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/AUSTRALIAN_NETWORK_ENERGY_ENDPOINT_STATISTICS_V1.yaml",
    )
    return parser.parse_args()


def _paired_values(
    primary: dict[str, str], baseline: dict[str, str], metric: str
) -> float:
    if metric == "network_adjustment_difference_mwh":
        primary_adjustment = float(primary["delivered_export_mwh"]) - float(
            primary["upstream_net_export_energy_mwh"]
        )
        baseline_adjustment = float(baseline["delivered_export_mwh"]) - float(
            baseline["upstream_net_export_energy_mwh"]
        )
        return primary_adjustment - baseline_adjustment
    return float(primary[metric]) - float(baseline[metric])


def main() -> int:
    args = _parse_args()
    result_dir = args.result_dir.resolve()
    protocol = args.protocol.resolve()
    summary_path = result_dir / "AU_NETWORK_ENERGY_EXTENSION_SUMMARY.json"
    day_path = result_dir / "AU_NETWORK_ENERGY_DAY_SUMMARY.csv"
    result_manifest_path = (
        result_dir / "AU_NETWORK_ENERGY_EXTENSION_SHA256_MANIFEST.json"
    )
    output_path = result_dir / "AU_NETWORK_ENERGY_ENDPOINT_STATISTICS.json"
    manifest_path = (
        result_dir / "AU_NETWORK_ENERGY_ENDPOINT_STATISTICS_SHA256_MANIFEST.json"
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["extension_id"] != "AUSTRALIAN_NETWORK_ENERGY_ENDPOINT_EXTENSION_V1_2":
        raise RuntimeError("statistics require the V1.2 endpoint extension")
    if summary["evidence_class"] != "PRIMARY_ENDPOINT_EXTENSION":
        raise RuntimeError("statistics require primary endpoint evidence")
    if summary["run_scope"] != "FULL_EVALUATION":
        raise RuntimeError("statistics require the full evaluation panel")

    rows = _read_csv(day_path)
    by_key = {(row["date"], row["control"]): row for row in rows}
    dates = sorted({row["date"] for row in rows})
    if len(dates) != 80 or len(rows) != 320:
        raise RuntimeError("formal endpoint statistics require 80 days x 4 controls")
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate network-day/control record")

    comparisons: dict[str, Any] = {}
    for baseline in BASELINES:
        label = f"ECF_MINUS_{baseline}"
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            values = np.asarray(
                [
                    _paired_values(
                        by_key[(date, PRIMARY_CONTROL)], by_key[(date, baseline)], metric
                    )
                    for date in dates
                ],
                dtype=float,
            )
            if np.any(~np.isfinite(values)):
                raise RuntimeError(f"non-finite paired values for {label}/{metric}")
            ordinary = parent_analysis._paired_bootstrap(
                values, _rng_for(label, metric, "ordinary")
            )
            dependent = parent_analysis._date_aware_dependent_wild_bootstrap(
                values,
                dates,
                _rng_for(label, metric, "date_aware_dependent_wild"),
                bandwidth_days=DEPENDENCE_BANDWIDTH_DAYS,
            )
            metric_results[metric] = {
                **ordinary,
                "date_aware_dependent_wild": dependent,
            }
        comparisons[label] = {"metrics": metric_results}

    analyzer_path = Path(__file__).resolve()
    parent_analyzer_path = (
        REPOSITORY_ROOT / "tools/analyze_ees_australian_external_replay.py"
    )
    input_paths = {
        "statistics_protocol": protocol.relative_to(REPOSITORY_ROOT).as_posix(),
        "extension_summary": summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "extension_day_summary": day_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "extension_result_manifest": result_manifest_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "endpoint_analyzer": analyzer_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "parent_statistics_implementation": parent_analyzer_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
    }
    payload = {
        "statistics_id": "AUSTRALIAN_NETWORK_ENERGY_ENDPOINT_STATISTICS_V1",
        "source_extension_id": summary["extension_id"],
        "paired_unit": "network_day",
        "eligible_network_day_count": len(dates),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_seed_derivation": SEED_DERIVATION,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "dependence_bandwidth_calendar_days": DEPENDENCE_BANDWIDTH_DAYS,
        "inference_boundary": (
            "Observed eligible-day panel only; intervals are not annual, field, "
            "independent-feeder, economic, or carbon estimates."
        ),
        "comparisons": comparisons,
        "input_paths": input_paths,
        "input_hashes": {
            name: _sha256(REPOSITORY_ROOT / relative)
            for name, relative in input_paths.items()
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "statistics_id": payload["statistics_id"],
        "hash_algorithm": "SHA-256",
        "payload": [
            {
                "path": output_path.name,
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
