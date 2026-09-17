#!/usr/bin/env python3
"""Read-only verifier for the frozen EES Australian replay inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    REPOSITORY_ROOT
    / "outputs/ees_transition/data/processed/australian_external_replay_v2"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_record_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _verify_hash_record(record: dict[str, Any], *, label: str) -> None:
    _require(set(record) == {"path", "sha256", "bytes"}, f"invalid {label} record")
    expected = str(record["sha256"])
    _require(bool(SHA256_PATTERN.fullmatch(expected)), f"invalid {label} SHA-256")
    path = _resolve_record_path(str(record["path"]))
    _require(path.is_file(), f"missing {label} file: {path}")
    _require(path.stat().st_size == int(record["bytes"]), f"{label} byte count mismatch")
    _require(_sha256(path) == expected, f"{label} hash mismatch: {path}")


def _verify_npz(path: Path, role: str, metadata: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "case_id",
            "split_role",
            "dates",
            "customer_ids",
            "generator_capacity_kwp",
            "postcodes",
            "gross_generation_kw",
            "household_load_kw",
            "complete_customer_day",
            "chronology_valid_day",
            "source_row_present",
            "source_row_actual",
            "controlled_load_expected",
            "interval_hours",
            "source_sha256",
        }
        _require(set(archive.files) == expected, f"{role} archive fields differ")
        _require(str(archive["case_id"]) == "AU_REAL_DERIVED_REPLAY_V2", "case ID mismatch")
        _require(str(archive["split_role"]) == role, f"{role} split tag mismatch")
        customer_ids = archive["customer_ids"]
        dates = archive["dates"]
        generation = archive["gross_generation_kw"]
        load = archive["household_load_kw"]
        complete = archive["complete_customer_day"]
        chronology_valid = archive["chronology_valid_day"]
        present = archive["source_row_present"]
        actual = archive["source_row_actual"]
        cl_expected = archive["controlled_load_expected"]
        _require(np.array_equal(customer_ids, np.arange(1, 301)), f"{role} customer IDs")
        _require(
            generation.shape == (len(dates), 48, 300) and load.shape == generation.shape,
            f"{role} time-series shape mismatch",
        )
        _require(complete.shape == (len(dates), 300), f"{role} completeness shape")
        _require(chronology_valid.shape == (len(dates),), f"{role} chronology shape")
        _require(present.shape == (len(dates), 300, 3), f"{role} presence shape")
        _require(actual.shape == present.shape, f"{role} quality shape")
        _require(cl_expected.shape == (300,), f"{role} CL expectation shape")
        _require(np.all(actual <= present), f"{role} actual rows must be present")
        _require(
            int(np.sum(~chronology_valid)) == 2,
            f"{role} must identify both DST transition dates",
        )
        complete_intervals = np.broadcast_to(complete[:, None, :], generation.shape)
        _require(np.all(np.isfinite(generation[complete_intervals])), f"{role} complete GG invalid")
        _require(np.all(np.isfinite(load[complete_intervals])), f"{role} complete load invalid")
        _require(np.nanmin(generation) >= 0.0 and np.nanmin(load) >= 0.0, f"{role} negatives")
        _require(float(archive["interval_hours"]) == 0.5, f"{role} interval duration")
        _require(len(dates) == int(metadata["day_count"]), f"{role} day count")
        _require(
            int(np.sum(complete)) == int(metadata["mandatory_customer_days_complete"]),
            f"{role} complete customer-day count",
        )
        source_hash = str(archive["source_sha256"])
        _require(source_hash == metadata["source_sha256"], f"{role} source hash binding")


def _verify_mapping(path: Path, expected_loads: int, expected_profiles: set[int]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == expected_loads, "connection mapping row count mismatch")
    _require(len({row["load_name"] for row in rows}) == expected_loads, "duplicate mapped loads")
    _require(
        {int(row["ausgrid_customer_id"]) for row in rows} == expected_profiles,
        "mapped profile roster differs from the registered cohort",
    )
    ranks = sorted(int(row["adoption_rank"]) for row in rows)
    _require(ranks == list(range(1, expected_loads + 1)), "adoption ranks are not a permutation")
    _require(sum(int(row["participant_33pct"]) for row in rows) == round((1.0 / 3.0) * expected_loads), "one-third count")
    _require(sum(int(row["participant_50pct"]) for row in rows) == round(0.50 * expected_loads), "50% count")
    _require(all(row["participant_100pct"] == "1" for row in rows), "100% count")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args()


def main() -> int:
    package = _parse_args().package.resolve()
    before = {
        path.relative_to(package).as_posix(): (_sha256(path), path.stat().st_size)
        for path in package.rglob("*")
        if path.is_file()
    }
    metadata = _load_json(package / "CASE_INPUT_METADATA.json")
    manifest = _load_json(package / "CASE_INPUT_SHA256_MANIFEST.json")
    _require(metadata["case_id"] == "AU_REAL_DERIVED_REPLAY_V2", "metadata case ID")
    _require(metadata["supersedes_case_id"] == "AU_REAL_DERIVED_REPLAY_V1", "supersession binding")
    _require(manifest["case_id"] == metadata["case_id"], "manifest case ID")
    _require(manifest["hash_algorithm"] == "SHA-256", "manifest hash algorithm")

    for record in manifest["payload"]:
        _verify_hash_record(record, label="payload")
    for category, records in metadata["input_hashes"].items():
        for record in records:
            _verify_hash_record(record, label=f"raw {category}")

    split_policy = metadata["split_policy"]
    for role in ("TRAIN", "DEVELOPMENT", "EVALUATION"):
        record = split_policy[role]
        _verify_npz(_resolve_record_path(record["archive"]), role, record)
    mapping = _resolve_record_path(metadata["profile_assignment"]["mapping"])
    profile_variant = metadata.get("profile_variant")
    expected_profiles = (
        {int(value) for value in profile_variant["selected_customer_ids"]}
        if profile_variant
        else set(range(1, 301))
    )
    _verify_mapping(
        mapping,
        int(metadata["network"]["load_connection_count"]),
        expected_profiles,
    )

    after = {
        path.relative_to(package).as_posix(): (_sha256(path), path.stat().st_size)
        for path in package.rglob("*")
        if path.is_file()
    }
    _require(before == after, "verifier modified the verified package")
    print(
        json.dumps(
            {
                "case_id": metadata["case_id"],
                "status": "PASS",
                "payload_file_count": len(manifest["payload"]),
                "verified_tree_unchanged": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
