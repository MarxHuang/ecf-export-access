#!/usr/bin/env python3
"""Freeze public Australian time-series and topology inputs for EES replay.

This tool does no allocation, feedback, or power-flow calculation.  It creates
an immutable, hash-addressed input layer from two deliberately separate public
sources:

* Ausgrid household gross-PV and consumption time series; and
* the CSIRO/GridQube real-world-derived Australian MV/LV network model.

The resulting case is a data-calibrated replay, not a field-matched digital
twin.  Train/development/evaluation years are fixed before any outcome is
computed.  Missing mandatory customer-days remain masked and are never
imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from r4r.australian_data import (  # noqa: E402
    AUSGRID_INTERVAL_HOURS,
    deterministic_profile_assignment,
    load_ausgrid_year,
    parse_dss_load_connections,
    sha256_file,
    validate_year_compatibility,
)


CASE_ID = "AU_REAL_DERIVED_REPLAY_V2"
SUPERSEDES_CASE_ID = "AU_REAL_DERIVED_REPLAY_V1"
ASSIGNMENT_SEED = "AU_REAL_DERIVED_REPLAY_V1_PROFILE_ASSIGNMENT"
ADOPTION_LEVELS = (1.0 / 3.0, 0.50, 1.00)
SPLIT_ROLES = ("TRAIN", "DEVELOPMENT", "EVALUATION")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _stable_rank(seed: str, token: str) -> str:
    return hashlib.sha256(f"{seed}|{token}".encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _discover_csvs(root: Path) -> tuple[Path, ...]:
    values = tuple(sorted(root.rglob("*.csv")))
    if len(values) != 3:
        raise RuntimeError(f"expected exactly three Ausgrid CSVs, found {len(values)}")
    return values


def _hash_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": _relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def _write_connection_mapping(
    path: Path,
    connections: tuple[Any, ...],
    assignment: dict[str, int],
) -> None:
    ordered = sorted(
        connections,
        key=lambda item: _stable_rank(ASSIGNMENT_SEED, item.load_name),
    )
    rank = {item.load_name: index + 1 for index, item in enumerate(ordered)}
    population = len(ordered)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "load_name",
                "bus_name",
                "phase",
                "lv_feeder_id",
                "source_file",
                "ausgrid_customer_id",
                "adoption_rank",
                "participant_33pct",
                "participant_50pct",
                "participant_100pct",
            ]
        )
        for item in sorted(connections, key=lambda value: value.load_name):
            item_rank = rank[item.load_name]
            writer.writerow(
                [
                    item.load_name,
                    item.bus_name,
                    item.phase,
                    item.feeder_id,
                    item.source_file,
                    assignment[item.load_name],
                    item_rank,
                    int(item_rank <= round(ADOPTION_LEVELS[0] * population)),
                    int(item_rank <= round(ADOPTION_LEVELS[1] * population)),
                    1,
                ]
            )


def _write_timeseries_archive(path: Path, year: Any, role: str) -> None:
    np.savez_compressed(
        path,
        case_id=np.asarray(CASE_ID),
        split_role=np.asarray(role),
        dates=np.asarray([value.isoformat() for value in year.dates]),
        customer_ids=np.asarray(year.customer_ids, dtype=np.int16),
        generator_capacity_kwp=year.generator_capacity_kwp.astype(np.float32),
        postcodes=year.postcodes.astype(np.int32),
        gross_generation_kw=year.gross_generation_kw.astype(np.float32),
        household_load_kw=year.load_kw.astype(np.float32),
        complete_customer_day=year.complete_customer_day.astype(bool),
        chronology_valid_day=year.chronology_valid_day.astype(bool),
        source_row_present=year.source_row_present.astype(bool),
        source_row_actual=year.source_row_actual.astype(bool),
        controlled_load_expected=year.controlled_load_expected.astype(bool),
        interval_hours=np.asarray(AUSGRID_INTERVAL_HOURS, dtype=np.float64),
        source_sha256=np.asarray(year.source_sha256),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ausgrid-root",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/data/external/ausgrid_solar_home_2010_2013/extracted",
    )
    parser.add_argument(
        "--csiro-data-root",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/data/external/csiro_australian_mv_lv_feeder_v1/000065408v001/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/data/processed/australian_external_replay_v2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the known generated files; never deletes other files",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ausgrid_root = args.ausgrid_root.resolve()
    csiro_root = args.csiro_data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known_outputs = [
        output_dir / "AUSGRID_TRAIN_2010_2011.npz",
        output_dir / "AUSGRID_DEVELOPMENT_2011_2012.npz",
        output_dir / "AUSGRID_EVALUATION_2012_2013.npz",
        output_dir / "CSIRO_LOAD_TO_AUSGRID_PROFILE_MAPPING.csv",
        output_dir / "CASE_INPUT_METADATA.json",
        output_dir / "CASE_INPUT_SHA256_MANIFEST.json",
    ]
    existing = [path for path in known_outputs if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            "refusing to overwrite frozen inputs; pass --overwrite explicitly: "
            + ", ".join(path.name for path in existing)
        )

    years = sorted(
        (load_ausgrid_year(path) for path in _discover_csvs(ausgrid_root)),
        key=lambda value: value.dates[0],
    )
    validate_year_compatibility(years)
    if len(years) != len(SPLIT_ROLES):
        raise RuntimeError("split-role count does not match source-year count")

    archives = known_outputs[:3]
    for archive, year, role in zip(archives, years, SPLIT_ROLES, strict=True):
        _write_timeseries_archive(archive, year, role)

    connections = parse_dss_load_connections(csiro_root)
    assignment = deterministic_profile_assignment(
        connections,
        years[0].customer_ids,
        seed=ASSIGNMENT_SEED,
    )
    mapping_path = known_outputs[3]
    _write_connection_mapping(mapping_path, connections, assignment)

    complete_counts = [int(np.sum(year.complete_customer_day)) for year in years]
    total_counts = [int(year.complete_customer_day.size) for year in years]
    metadata = {
        "case_id": CASE_ID,
        "supersedes_case_id": SUPERSEDES_CASE_ID,
        "supersession_reason": (
            "V2 preserves literal Row Quality=NA flags, applies strict partial-CL "
            "completeness, and excludes DST-transition dates from state recursion."
        ),
        "case_description": (
            "Data-calibrated replay joining a real-world-derived Australian "
            "network model to public Ausgrid household gross-PV and consumption "
            "profiles; the sources are not field matched."
        ),
        "evidence_scope": "REAL_DERIVED_TOPOLOGY_PLUS_PUBLIC_TIME_SERIES_REPLAY",
        "prohibited_descriptions": [
            "field validation",
            "digital twin",
            "historically observed feeder operation",
            "utility-certified thermal validation",
        ],
        "interval_hours": AUSGRID_INTERVAL_HOURS,
        "split_policy": {
            role: {
                **year.to_metadata(),
                "source_path": _relative(year.source_path),
                "archive": _relative(archive),
                "strict_incomplete_customer_days_excluded_not_imputed": (
                    total - complete
                ),
            }
            for role, year, archive, complete, total in zip(
                SPLIT_ROLES,
                years,
                archives,
                complete_counts,
                total_counts,
                strict=True,
            )
        },
        "network": {
            "data_root": _relative(csiro_root),
            "load_connection_count": len(connections),
            "lv_feeder_ids": sorted({item.feeder_id for item in connections}),
            "lv_feeder_count": len({item.feeder_id for item in connections}),
            "native_line_ampacity_available": False,
            "topology_sections_are_not_independent_network_replicates": True,
        },
        "profile_assignment": {
            "seed": ASSIGNMENT_SEED,
            "method": "SHA256_ORDER_WITH_CYCLIC_REUSE_AFTER_ALL_300_PROFILES",
            "mapping": _relative(mapping_path),
            "adoption_levels": list(ADOPTION_LEVELS),
            "adoption_levels_are_scenarios_not_observed_penetration": True,
        },
        "input_hashes": {
            "ausgrid_csv": _hash_records(_discover_csvs(ausgrid_root)),
            "csiro_dss": _hash_records(
                [csiro_root / "Master.dss", *csiro_root.glob("LV/*/Loads.dss")]
            ),
        },
        "build_tool": "tools/prepare_ees_australian_case_inputs.py",
        "quality_policy": {
            "literal_NA_meaning": "estimated_or_substitute_source_row",
            "primary_rows": "blank_quality_or_source_asserted_actual_only",
            "partial_controlled_load": "incomplete_not_zero_filled",
            "dst_transition_dates": "excluded_from_lag_and_state_recursion",
        },
        "build_schema_version": "2.0.0",
    }
    metadata_path = known_outputs[4]
    _json_dump(metadata_path, metadata)

    payload_files = [*archives, mapping_path, metadata_path]
    manifest = {
        "case_id": CASE_ID,
        "hash_algorithm": "SHA-256",
        "payload": _hash_records(payload_files),
    }
    _json_dump(known_outputs[5], manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
