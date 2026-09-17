#!/usr/bin/env python3
"""Compute paired network-day statistics for the frozen Australian replay."""

from __future__ import annotations

import argparse
import csv
from datetime import date as calendar_date
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 10000
DEPENDENCE_BANDWIDTH_DAYS = 7
PRIMARY_CONTROL = "ECF"
COMPARATORS = ("NO_FEEDBACK", "MATCHED_UNIFORM", "PERSISTENCE_ONLY")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite {key} in {row.get('date')} {row.get('control')}")
    return value


def _paired_bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise RuntimeError("paired bootstrap requires a finite one-dimensional sample")
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "network_day_count": int(values.size),
        "mean_difference_mwh_per_network_day": float(np.mean(values)),
        "median_difference_mwh_per_network_day": float(np.median(values)),
        "bootstrap_95_ci_mean_mwh_per_network_day": [float(lower), float(upper)],
        "positive_day_fraction": float(np.mean(values > 0.0)),
        "zero_day_fraction": float(np.mean(np.isclose(values, 0.0, atol=1.0e-12))),
        "aggregate_difference_mwh_over_observed_days": float(np.sum(values)),
    }


def _date_aware_dependent_wild_bootstrap(
    values: np.ndarray,
    dates: list[str],
    rng: np.random.Generator,
    *,
    bandwidth_days: int,
) -> dict[str, Any]:
    """Date-aware dependent-wild interval for an irregular daily panel.

    Gaussian multipliers use a Bartlett covariance kernel evaluated at actual
    calendar lags. Every eligible observation enters every replicate, so gaps
    neither disappear from the target mean nor receive implicit zero values.
    Antithetic multiplier pairs make the empirical bootstrap center equal the
    observed paired-day mean up to floating-point rounding.
    """

    if values.ndim != 1 or values.size != len(dates) or values.size < 2:
        raise RuntimeError("dependent-wild bootstrap requires paired ordered days")
    if not isinstance(bandwidth_days, int) or bandwidth_days < 1:
        raise RuntimeError("dependence bandwidth must be a positive integer")
    if BOOTSTRAP_REPLICATES % 2:
        raise RuntimeError("antithetic dependent-wild bootstrap requires an even replicate count")
    parsed = [calendar_date.fromisoformat(value) for value in dates]
    if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
        raise RuntimeError("dependent-wild bootstrap dates must be unique and ordered")
    ordinal = np.asarray([value.toordinal() for value in parsed], dtype=float)
    lag = np.abs(ordinal[:, None] - ordinal[None, :])
    covariance = np.maximum(1.0 - lag / (bandwidth_days + 1.0), 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -1.0e-10:
        raise RuntimeError("date-aware Bartlett covariance is not positive semidefinite")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    half = BOOTSTRAP_REPLICATES // 2
    independent = rng.standard_normal(size=(half, values.size))
    multiplier = independent @ root.T
    multiplier = np.vstack((multiplier, -multiplier))
    observed_mean = float(np.mean(values))
    centered = values - observed_mean
    bootstrap_means = observed_mean + multiplier @ centered / values.size
    empirical_center = float(np.mean(bootstrap_means))
    center_error = empirical_center - observed_mean
    if abs(center_error) > 1.0e-12:
        raise RuntimeError("dependent-wild bootstrap is not centered on the target mean")
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "bootstrap_95_ci_mean_mwh_per_network_day": [float(lower), float(upper)],
        "bootstrap_empirical_center_mwh_per_network_day": empirical_center,
        "target_sample_mean_mwh_per_network_day": observed_mean,
        "center_error_mwh_per_network_day": float(center_error),
        "bandwidth_calendar_days": bandwidth_days,
        "eligible_observation_count": int(values.size),
        "observation_coverage_fraction": 1.0,
        "minimum_covariance_eigenvalue": minimum_eigenvalue,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/australian_external_replay_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "statistics output path; defaults to "
            "<result-dir>/AU_EXTERNAL_PAIRED_STATISTICS.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_output(result_dir: Path, requested_output: Path | None) -> Path:
    """Keep statistics in the analyzed result tree unless explicitly overridden."""
    if requested_output is not None:
        return requested_output.resolve()
    return result_dir / "AU_EXTERNAL_PAIRED_STATISTICS.json"


def main() -> int:
    args = _parse_args()
    result_dir = args.result_dir.resolve()
    output = _resolve_output(result_dir, args.output)
    if output.exists() and not args.overwrite:
        raise RuntimeError("refusing to overwrite paired statistics without --overwrite")
    day_path = result_dir / "AU_EXTERNAL_DAY_SUMMARY.csv"
    interval_path = result_dir / "AU_EXTERNAL_INTERVAL_RESULTS.csv"
    run_summary_path = result_dir / "AU_EXTERNAL_RUN_SUMMARY.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    with day_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_date: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_date.setdefault(row["date"], {})[row["control"]] = row
    expected_controls = {PRIMARY_CONTROL, *COMPARATORS}
    for date, controls in by_date.items():
        if set(controls) != expected_controls:
            raise RuntimeError(f"incomplete paired controls on {date}: {sorted(controls)}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = (
        "delivered_export_mwh",
        "authorized_export_mwh",
        "idle_authorization_mwh",
        "curtailed_available_export_mwh",
    )
    comparisons: dict[str, Any] = {}
    dates = sorted(by_date)
    for comparator in COMPARATORS:
        label = f"{PRIMARY_CONTROL}_MINUS_{comparator}"
        comparisons[label] = {}
        for metric in metrics:
            differences = np.asarray(
                [
                    _float(by_date[date][PRIMARY_CONTROL], metric)
                    - _float(by_date[date][comparator], metric)
                    for date in dates
                ],
                dtype=float,
            )
            comparisons[label][metric] = _paired_bootstrap(differences, rng)
            comparisons[label][metric]["date_aware_dependent_wild_bootstrap"] = (
                _date_aware_dependent_wild_bootstrap(
                    differences,
                    dates,
                    rng,
                    bandwidth_days=DEPENDENCE_BANDWIDTH_DAYS,
                )
            )

    with interval_path.open(newline="", encoding="utf-8") as stream:
        intervals = list(csv.DictReader(stream))
    ecf_intervals = [row for row in intervals if row["control"] == PRIMARY_CONTROL]
    open_gates = [row for row in ecf_intervals if row["gate_triggered"] == "1"]
    gate_recovery = np.asarray(
        [_float(row, "replay_delivery_gain_kw") for row in open_gates], dtype=float
    )
    payload = {
        "case_id": "AU_REAL_DERIVED_REPLAY_V2",
        "status": "PASS",
        "evidence_class": run_summary["evidence_class"],
        "sensitivity_run_id": run_summary.get("sensitivity_run_id"),
        "beta": run_summary["beta"],
        "telemetry_mode": run_summary["telemetry_mode"],
        "telemetry_scale": run_summary["telemetry_scale"],
        "telemetry_dropout_fraction": run_summary["telemetry_dropout_fraction"],
        "statistical_unit": "NETWORK_DAY_TRAJECTORY",
        "paired_network_day_count": len(dates),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "dependence_bandwidth_days": DEPENDENCE_BANDWIDTH_DAYS,
        "dependent_interval_method": (
            "DATE_AWARE_GAUSSIAN_DEPENDENT_WILD_BOOTSTRAP_WITH_BARTLETT_KERNEL"
        ),
        "confidence_interval": "PERCENTILE_PAIRED_NETWORK_DAY_BOOTSTRAP_95_PERCENT",
        "comparisons": comparisons,
        "replay_gate": {
            "eligible_interval_count": len(ecf_intervals),
            "open_count": len(open_gates),
            "open_fraction": len(open_gates) / len(ecf_intervals),
            "median_same_realization_delivery_recovery_kw": (
                float(np.median(gate_recovery)) if gate_recovery.size else None
            ),
            "interquartile_recovery_kw": (
                [float(value) for value in np.quantile(gate_recovery, [0.25, 0.75])]
                if gate_recovery.size
                else None
            ),
        },
        "post_event_signal_diagnostic": {
            "absolute_error_mwh": run_summary["aggregate"][PRIMARY_CONTROL][
                "post_event_signal_absolute_error_mwh"
            ],
            "dropout_count": run_summary["aggregate"][PRIMARY_CONTROL][
                "telemetry_dropout_count"
            ],
        },
        "interpretation_limits": [
            "The paired day sample is one real-derived network replay, not independent networks.",
            "Percentages are replay frequencies, not estimates of Australian field prevalence.",
            "Energy totals cover observed eligible half-hours only and are not annualized.",
        ],
        "input_hashes": {
            "day_summary": _sha256(day_path),
            "interval_results": _sha256(interval_path),
            "run_summary": _sha256(run_summary_path),
            "analysis_tool": _sha256(Path(__file__).resolve()),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_manifest = {
        "case_id": payload["case_id"],
        "hash_algorithm": "SHA-256",
        "payload": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (day_path, interval_path, run_summary_path, output)
        ],
    }
    evidence_manifest_path = result_dir / "AU_EXTERNAL_EVIDENCE_SHA256_MANIFEST.json"
    evidence_manifest_path.write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
