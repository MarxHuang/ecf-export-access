#!/usr/bin/env python3
"""Run the preregistered Australian external replay with daily parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SCIENTIFIC_CORE_MODULES = {
    "australian_allocation": SOURCE_ROOT / "r4r/australian_allocation.py",
    "australian_data": SOURCE_ROOT / "r4r/australian_data.py",
    "australian_opendss": SOURCE_ROOT / "r4r/australian_opendss.py",
    "australian_policy": SOURCE_ROOT / "r4r/australian_policy.py",
    "externality_feedback": SOURCE_ROOT / "r4r/externality_feedback.py",
    "environment_specification": (
        REPOSITORY_ROOT / "outputs/ees_transition/EES_COMPUTATIONAL_ENVIRONMENT.yaml"
    ),
}

from r4r.australian_allocation import allocate_proportional_ac_prefix  # noqa: E402
from r4r.australian_data import parse_dss_load_connections, sha256_file  # noqa: E402
from r4r.australian_opendss import AustralianOpenDSSModel  # noqa: E402
from r4r.australian_policy import (  # noqa: E402
    deterministic_dropout_mask,
    fit_persistence_upper_residual,
    matched_uniform_request,
    post_event_availability_signal,
    persistence_request,
)
from r4r.externality_feedback import (  # noqa: E402
    apply_access_ceiling,
    assess_replay,
    fulfillment_replay_request,
    metered_delivery,
    update_access_ceiling,
)


CASE_ID = "AU_REAL_DERIVED_REPLAY_V2"
INTERVAL_HOURS = 0.5
PV_SCALE = 5.881304516313253
ADOPTION_COLUMN = "participant_33pct"
LOAD_POWER_FACTOR = 0.95
FORECAST_QUANTILE = 0.90
PRIMARY_BETA = 0.5
REPLAY_TOLERANCE_MW = 2.0e-6
SCALAR_TOLERANCE = 1.0e-5
AUTHORIZATION_VOLTAGE_MAX_PU = 1.07
CONTROLS = ("NO_FEEDBACK", "PERSISTENCE_ONLY", "ECF", "MATCHED_UNIFORM")
SENSITIVITY_RUNS: dict[str, dict[str, Any]] = {
    "BETA_025_EXACT": {"beta": 0.25, "telemetry_mode": "EXACT"},
    "BETA_100_EXACT": {"beta": 1.0, "telemetry_mode": "EXACT"},
    "BETA_050_SCALE_090": {
        "beta": 0.5,
        "telemetry_mode": "MULTIPLICATIVE",
        "telemetry_scale": 0.90,
    },
    "BETA_050_SCALE_110": {
        "beta": 0.5,
        "telemetry_mode": "MULTIPLICATIVE",
        "telemetry_scale": 1.10,
    },
    "BETA_050_LAG_ONE": {"beta": 0.5, "telemetry_mode": "LAG_ONE_INTERVAL"},
    "BETA_050_DROPOUT_010": {
        "beta": 0.5,
        "telemetry_mode": "DETERMINISTIC_DROPOUT",
        "telemetry_dropout_fraction": 0.10,
        "telemetry_dropout_seed": "AU_TELEMETRY_DROPOUT_V1",
    },
}


_WORKER: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _mapping_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["load_name"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise RuntimeError("profile mapping is empty")
    return rows


def _load_mapping(path: Path, connections: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows = _mapping_rows(path)
    missing = [item.load_name for item in connections if item.load_name not in rows]
    if missing:
        raise RuntimeError(f"profile mapping is missing loads: {missing[:5]}")
    profile_index = np.asarray(
        [int(rows[item.load_name]["ausgrid_customer_id"]) - 1 for item in connections],
        dtype=int,
    )
    participant = np.asarray(
        [rows[item.load_name][ADOPTION_COLUMN] == "1" for item in connections],
        dtype=bool,
    )
    return profile_index, participant


def _used_profile_indices(path: Path, connections: tuple[Any, ...]) -> list[int]:
    """Return only profiles attached to the OpenDSS case being replayed."""

    rows = _mapping_rows(path)
    missing = [item.load_name for item in connections if item.load_name not in rows]
    if missing:
        raise RuntimeError(f"profile mapping is missing loads: {missing[:5]}")
    return sorted({int(rows[item.load_name]["ausgrid_customer_id"]) - 1 for item in connections})


def _worker_init(config: dict[str, str]) -> None:
    data_root = Path(config["dss_root"])
    package = Path(config["package"])
    connections = parse_dss_load_connections(data_root)
    source_profile_index, participant = _load_mapping(
        Path(config["mapping_file"]), connections
    )
    used_profile_index, profile_index = np.unique(
        source_profile_index, return_inverse=True
    )
    evaluation = np.load(package / "AUSGRID_EVALUATION_2012_2013.npz", allow_pickle=False)
    development = np.load(package / "AUSGRID_DEVELOPMENT_2011_2012.npz", allow_pickle=False)
    training = np.load(package / "AUSGRID_TRAIN_2010_2011.npz", allow_pickle=False)
    split = config["split"]
    if split == "EVALUATION":
        current = evaluation
        previous_year = development
    elif split == "DEVELOPMENT":
        current = development
        previous_year = training
    else:
        raise RuntimeError(f"unsupported split: {split}")

    load_scale = float(config["load_scale"])
    training_available = np.maximum(
        PV_SCALE * training["gross_generation_kw"][:, :, used_profile_index]
        - load_scale * training["household_load_kw"][:, :, used_profile_index],
        0.0,
    )
    upper_residual = fit_persistence_upper_residual(
        training_available,
        training["complete_customer_day"][:, used_profile_index]
        & training["chronology_valid_day"][:, None],
        quantile=FORECAST_QUANTILE,
    )
    capacity_profile_kw = (
        PV_SCALE
        * evaluation["generator_capacity_kwp"][used_profile_index].astype(float)
    )
    capacity_connection_mw = capacity_profile_kw[profile_index] / 1000.0
    capacity_connection_mw = np.where(participant, capacity_connection_mw, 0.0)
    model = AustralianOpenDSSModel(
        data_root,
        connections,
        load_power_factor=LOAD_POWER_FACTOR,
    )
    _WORKER.update(
        {
            "model": model,
            "current": current,
            "previous_year": previous_year,
            "profile_index": profile_index,
            "used_profile_index": used_profile_index,
            "participant": participant,
            "capacity_profile_kw": capacity_profile_kw,
            "capacity_connection_mw": capacity_connection_mw,
            "load_scale": load_scale,
            "upper_residual": upper_residual,
            "beta": float(config["beta"]),
            "telemetry_mode": config["telemetry_mode"],
            "telemetry_scale": float(config["telemetry_scale"]),
            "telemetry_dropout_fraction": float(
                config["telemetry_dropout_fraction"]
            ),
            "telemetry_dropout_seed": config["telemetry_dropout_seed"],
        }
    )


def _allocate_and_realize(
    load_kw: np.ndarray,
    gross_pv_kw: np.ndarray,
    request_mw: np.ndarray,
    participant: np.ndarray,
) -> dict[str, Any]:
    model: AustralianOpenDSSModel = _WORKER["model"]
    allocation = allocate_proportional_ac_prefix(
        model,
        load_kw,
        request_mw * 1000.0,
        participant,
        scalar_tolerance=SCALAR_TOLERANCE,
        authorization_voltage_max_pu=AUTHORIZATION_VOLTAGE_MAX_PU,
    )
    actual_screen, delivered_kw, available_kw = model.solve(
        load_kw,
        gross_pv_kw,
        allocation.allocation_kw,
        participant,
    )
    return {
        "allocation": allocation,
        "actual_screen": actual_screen,
        "allocation_mw": allocation.allocation_kw / 1000.0,
        "delivery_mw": delivered_kw / 1000.0,
        "available_mw": available_kw / 1000.0,
        "power_flow_count": allocation.power_flow_count + 1,
    }


def _interval_record(
    *,
    date: str,
    interval: int,
    control: str,
    raw_request_mw: np.ndarray,
    admitted_request_mw: np.ndarray,
    outcome: dict[str, Any],
    gate: bool = False,
    gate_reason: str = "NOT_APPLICABLE",
    outsider_access_gain_mw: float = 0.0,
    outsider_delivery_gain_mw: float = 0.0,
    replay_delivery_gain_mw: float = 0.0,
    uniform_factor: float = 1.0,
    extra_power_flows: int = 0,
    post_event_availability_signal_mw: np.ndarray | None = None,
    telemetry_dropout_count: int = 0,
) -> dict[str, Any]:
    allocation = outcome["allocation_mw"]
    delivery = outcome["delivery_mw"]
    available = outcome["available_mw"]
    availability_signal = (
        available
        if post_event_availability_signal_mw is None
        else np.asarray(post_event_availability_signal_mw, dtype=float)
    )
    if availability_signal.shape != available.shape:
        raise RuntimeError("post-event availability signal does not align")
    screen = outcome["actual_screen"]
    authorization_screen = outcome["allocation"].authorized_export_screen
    return {
        "date": date,
        "interval_index": interval,
        # OBSERVABILITY_BEGIN: record existing solve outputs without altering decisions.
        "actual_source_terminal_p_kw": screen.source_terminal_p_kw,
        "actual_circuit_loss_kw": screen.circuit_loss_kw,
        "actual_converged": int(screen.converged),
        "allocation_status": outcome["allocation"].status,
        "allocation_pair_pass": int(outcome["allocation"].pair_pass),
        "load_only_pair_pass": int(outcome["allocation"].load_only_screen.physical_pair_pass),
        "load_only_voltage_min_pu": outcome["allocation"].load_only_screen.customer_voltage_min_pu,
        "load_only_voltage_max_pu": outcome["allocation"].load_only_screen.customer_voltage_max_pu,
        "authorized_converged": int(authorization_screen.converged),
        "authorized_voltage_min_pu": authorization_screen.customer_voltage_min_pu,
        # OBSERVABILITY_END
        "interval_end_hour": (interval + 1) * INTERVAL_HOURS,
        "control": control,
        "raw_request_kw": float(np.sum(raw_request_mw) * 1000.0),
        "admitted_request_kw": float(np.sum(admitted_request_mw) * 1000.0),
        "authorized_export_kw": float(np.sum(allocation) * 1000.0),
        "delivered_export_kw": float(np.sum(delivery) * 1000.0),
        "available_export_kw": float(np.sum(available) * 1000.0),
        "post_event_availability_signal_kw": float(
            np.sum(availability_signal) * 1000.0
        ),
        "post_event_signal_absolute_error_kw": float(
            np.sum(np.abs(availability_signal - available)) * 1000.0
        ),
        "telemetry_dropout_count": int(telemetry_dropout_count),
        "idle_authorization_kw": float(np.sum(np.maximum(allocation - delivery, 0.0)) * 1000.0),
        "curtailed_available_export_kw": float(np.sum(np.maximum(available - delivery, 0.0)) * 1000.0),
        "request_scale": outcome["allocation"].request_scale,
        "gate_triggered": int(gate),
        "gate_reason": gate_reason,
        "outsider_access_gain_kw": outsider_access_gain_mw * 1000.0,
        "outsider_delivery_gain_kw": outsider_delivery_gain_mw * 1000.0,
        "replay_delivery_gain_kw": replay_delivery_gain_mw * 1000.0,
        "uniform_factor": uniform_factor,
        "actual_voltage_min_pu": screen.customer_voltage_min_pu,
        "actual_voltage_max_pu": screen.customer_voltage_max_pu,
        "actual_voltage_violation_count": screen.customer_undervoltage_count + screen.customer_overvoltage_count,
        "actual_transformer_max_loading_pu": screen.transformer_max_loading_pu,
        "actual_transformer_overload_count": screen.transformer_overload_count,
        "authorized_voltage_max_pu": authorization_screen.customer_voltage_max_pu,
        "authorized_transformer_max_loading_pu": authorization_screen.transformer_max_loading_pu,
        "physical_pair_pass": int(outcome["allocation"].pair_pass and screen.physical_pair_pass),
        "power_flow_count": outcome["power_flow_count"] + extra_power_flows,
    }


def _run_day(day_index: int) -> dict[str, Any]:
    current = _WORKER["current"]
    previous_year = _WORKER["previous_year"]
    profile_index: np.ndarray = _WORKER["profile_index"]
    used_profile_index: np.ndarray = _WORKER["used_profile_index"]
    participant: np.ndarray = _WORKER["participant"]
    capacity_profile_kw: np.ndarray = _WORKER["capacity_profile_kw"]
    capacity_mw: np.ndarray = _WORKER["capacity_connection_mw"]
    upper_residual: np.ndarray = _WORKER["upper_residual"]
    load_scale = float(_WORKER["load_scale"])
    beta = float(_WORKER["beta"])
    telemetry_mode = str(_WORKER["telemetry_mode"])
    telemetry_scale = float(_WORKER["telemetry_scale"])
    telemetry_dropout_fraction = float(_WORKER["telemetry_dropout_fraction"])
    telemetry_dropout_seed = str(_WORKER["telemetry_dropout_seed"])

    if day_index == 0:
        previous_load = (
            load_scale
            * previous_year["household_load_kw"][-1][:, used_profile_index]
        )
        previous_pv = previous_year["gross_generation_kw"][-1][
            :, used_profile_index
        ]
    else:
        previous_load = (
            load_scale
            * current["household_load_kw"][day_index - 1][:, used_profile_index]
        )
        previous_pv = current["gross_generation_kw"][day_index - 1][
            :, used_profile_index
        ]
    previous_available = np.maximum(PV_SCALE * previous_pv - previous_load, 0.0)
    raw_profile_kw = persistence_request(
        previous_available,
        capacity_profile_kw,
        upper_residual_kw=upper_residual,
    )
    persistence_profile_kw = persistence_request(
        previous_available,
        capacity_profile_kw,
    )
    raw_connection_mw = raw_profile_kw[:, profile_index] / 1000.0
    persistence_connection_mw = persistence_profile_kw[:, profile_index] / 1000.0
    raw_connection_mw[:, ~participant] = 0.0
    persistence_connection_mw[:, ~participant] = 0.0

    ceiling_mw = capacity_mw.copy()
    records: list[dict[str, Any]] = []
    participant_delivery = {control: np.zeros_like(capacity_mw) for control in CONTROLS}
    participant_authorization = {control: np.zeros_like(capacity_mw) for control in CONTROLS}
    date = str(current["dates"][day_index])
    previous_interval_available_mw: np.ndarray | None = None

    for interval in range(48):
        load_kw = (
            load_scale
            * current["household_load_kw"][
                day_index, interval, used_profile_index
            ].astype(float)[profile_index]
        )
        gross_pv_kw = (
            PV_SCALE
            * current["gross_generation_kw"][
                day_index, interval, used_profile_index
            ].astype(float)[profile_index]
        )
        raw_mw = raw_connection_mw[interval]

        no_feedback = _allocate_and_realize(load_kw, gross_pv_kw, raw_mw, participant)
        records.append(
            _interval_record(
                date=date,
                interval=interval,
                control="NO_FEEDBACK",
                raw_request_mw=raw_mw,
                admitted_request_mw=raw_mw,
                outcome=no_feedback,
            )
        )
        participant_delivery["NO_FEEDBACK"] += no_feedback["delivery_mw"] * INTERVAL_HOURS
        participant_authorization["NO_FEEDBACK"] += no_feedback["allocation_mw"] * INTERVAL_HOURS

        persistence_mw = persistence_connection_mw[interval]
        persistence = _allocate_and_realize(load_kw, gross_pv_kw, persistence_mw, participant)
        records.append(
            _interval_record(
                date=date,
                interval=interval,
                control="PERSISTENCE_ONLY",
                raw_request_mw=raw_mw,
                admitted_request_mw=persistence_mw,
                outcome=persistence,
            )
        )
        participant_delivery["PERSISTENCE_ONLY"] += persistence["delivery_mw"] * INTERVAL_HOURS
        participant_authorization["PERSISTENCE_ONLY"] += persistence["allocation_mw"] * INTERVAL_HOURS

        admitted_ecf_mw = np.asarray(
            apply_access_ceiling(raw_mw, capacity_mw, ceiling_mw), dtype=float
        )
        ecf = _allocate_and_realize(load_kw, gross_pv_kw, admitted_ecf_mw, participant)
        replay_request_mw, idle = fulfillment_replay_request(
            admitted_ecf_mw,
            ecf["allocation_mw"],
            ecf["delivery_mw"],
            tolerance_mw=REPLAY_TOLERANCE_MW,
        )
        replay_request_mw = np.asarray(replay_request_mw, dtype=float)
        replay = _allocate_and_realize(load_kw, gross_pv_kw, replay_request_mw, participant)
        dropout_mask = None
        if telemetry_mode == "DETERMINISTIC_DROPOUT":
            dropout_mask = deterministic_dropout_mask(
                len(capacity_mw),
                telemetry_dropout_fraction,
                seed=telemetry_dropout_seed,
                date=date,
                interval_index=interval,
            )
            dropout_mask &= participant
        availability_signal_mw = post_event_availability_signal(
            ecf["available_mw"],
            ecf["delivery_mw"],
            mode=telemetry_mode,
            scale=telemetry_scale,
            previous_available_mw=previous_interval_available_mw,
            dropout_mask=dropout_mask,
        )
        replay_delivery_mw = np.asarray(
            metered_delivery(replay["allocation_mw"], availability_signal_mw),
            dtype=float,
        )
        replay_valid = bool(
            replay["allocation"].pair_pass and replay["actual_screen"].physical_pair_pass
        )
        assessment = assess_replay(
            allocation_mw=ecf["allocation_mw"],
            delivery_mw=ecf["delivery_mw"],
            replay_allocation_mw=replay["allocation_mw"],
            replay_delivery_mw=replay_delivery_mw,
            idle_indices=idle,
            replay_valid=replay_valid,
            tolerance_mw=REPLAY_TOLERANCE_MW,
        )
        ceiling_mw, _ = update_access_ceiling(
            ceiling_before_mw=ceiling_mw,
            registered_capacity_mw=capacity_mw,
            delivery_mw=ecf["delivery_mw"],
            idle_indices=idle,
            gate_triggered=assessment.gate_triggered,
            beta=beta,
        )
        ceiling_mw = np.asarray(ceiling_mw, dtype=float)
        records.append(
            _interval_record(
                date=date,
                interval=interval,
                control="ECF",
                raw_request_mw=raw_mw,
                admitted_request_mw=admitted_ecf_mw,
                outcome=ecf,
                gate=assessment.gate_triggered,
                gate_reason=assessment.gate_reason,
                outsider_access_gain_mw=assessment.outsider_allocation_gain_mw,
                outsider_delivery_gain_mw=assessment.outsider_delivery_gain_mw,
                replay_delivery_gain_mw=assessment.total_delivery_gain_mw,
                extra_power_flows=replay["power_flow_count"],
                post_event_availability_signal_mw=availability_signal_mw,
                telemetry_dropout_count=(
                    int(np.count_nonzero(dropout_mask))
                    if dropout_mask is not None
                    else 0
                ),
            )
        )
        previous_interval_available_mw = np.asarray(
            ecf["available_mw"], dtype=float
        ).copy()
        participant_delivery["ECF"] += ecf["delivery_mw"] * INTERVAL_HOURS
        participant_authorization["ECF"] += ecf["allocation_mw"] * INTERVAL_HOURS

        uniform_mw, uniform_factor = matched_uniform_request(raw_mw, admitted_ecf_mw)
        uniform = _allocate_and_realize(load_kw, gross_pv_kw, uniform_mw, participant)
        records.append(
            _interval_record(
                date=date,
                interval=interval,
                control="MATCHED_UNIFORM",
                raw_request_mw=raw_mw,
                admitted_request_mw=uniform_mw,
                outcome=uniform,
                uniform_factor=uniform_factor,
            )
        )
        participant_delivery["MATCHED_UNIFORM"] += uniform["delivery_mw"] * INTERVAL_HOURS
        participant_authorization["MATCHED_UNIFORM"] += uniform["allocation_mw"] * INTERVAL_HOURS

    summaries: list[dict[str, Any]] = []
    for control in CONTROLS:
        selected = [record for record in records if record["control"] == control]
        delivery_energy = participant_delivery[control]
        authorization_energy = participant_authorization[control]
        active_energy = delivery_energy[participant]
        summaries.append(
            {
                "date": date,
                "control": control,
                "delivered_export_mwh": float(np.sum(delivery_energy)),
                "authorized_export_mwh": float(np.sum(authorization_energy)),
                "idle_authorization_mwh": float(np.sum(np.maximum(authorization_energy - delivery_energy, 0.0))),
                "available_export_mwh": float(sum(record["available_export_kw"] for record in selected) * INTERVAL_HOURS / 1000.0),
                "curtailed_available_export_mwh": float(sum(record["curtailed_available_export_kw"] for record in selected) * INTERVAL_HOURS / 1000.0),
                "post_event_signal_absolute_error_mwh": float(
                    sum(
                        record["post_event_signal_absolute_error_kw"]
                        for record in selected
                    )
                    * INTERVAL_HOURS
                    / 1000.0
                ),
                "telemetry_dropout_count": int(
                    sum(record["telemetry_dropout_count"] for record in selected)
                ),
                "raw_request_mwh": float(sum(record["raw_request_kw"] for record in selected) * INTERVAL_HOURS / 1000.0),
                "admitted_request_mwh": float(sum(record["admitted_request_kw"] for record in selected) * INTERVAL_HOURS / 1000.0),
                "gate_count": int(sum(record["gate_triggered"] for record in selected)),
                "all_interval_physical_pair_pass": int(all(record["physical_pair_pass"] for record in selected)),
                "maximum_actual_voltage_pu": float(max(record["actual_voltage_max_pu"] for record in selected)),
                "minimum_actual_voltage_pu": float(min(record["actual_voltage_min_pu"] for record in selected)),
                "maximum_transformer_loading_pu": float(max(record["actual_transformer_max_loading_pu"] for record in selected)),
                "power_flow_count": int(sum(record["power_flow_count"] for record in selected)),
                "participant_delivery_p10_mwh": float(np.quantile(active_energy, 0.10)),
                "participant_delivery_median_mwh": float(np.median(active_energy)),
                "participant_delivery_p90_mwh": float(np.quantile(active_energy, 0.90)),
            }
        )
    return {"date": date, "interval_records": records, "day_summaries": summaries}


def _eligible_days(
    package: Path, split: str, connections: tuple[Any, ...], mapping_file: Path
) -> list[int]:
    if split == "EVALUATION":
        current_name = "AUSGRID_EVALUATION_2012_2013.npz"
        previous_name = "AUSGRID_DEVELOPMENT_2011_2012.npz"
    elif split == "DEVELOPMENT":
        current_name = "AUSGRID_DEVELOPMENT_2011_2012.npz"
        previous_name = "AUSGRID_TRAIN_2010_2011.npz"
    else:
        raise RuntimeError(f"unsupported split: {split}")
    used_profiles = _used_profile_indices(
        mapping_file, connections
    )
    with np.load(package / current_name, allow_pickle=False) as current:
        complete = np.all(current["complete_customer_day"][:, used_profiles], axis=1)
        chronology_valid = current["chronology_valid_day"].astype(bool)
    with np.load(package / previous_name, allow_pickle=False) as previous:
        previous_last_complete = bool(
            np.all(previous["complete_customer_day"][-1, used_profiles])
        )
        previous_last_chronology_valid = bool(previous["chronology_valid_day"][-1])
    eligible: list[int] = []
    for day in range(len(complete)):
        previous_complete = previous_last_complete if day == 0 else bool(complete[day - 1])
        previous_chronology_valid = (
            previous_last_chronology_valid
            if day == 0
            else bool(chronology_valid[day - 1])
        )
        if (
            bool(complete[day])
            and previous_complete
            and bool(chronology_valid[day])
            and previous_chronology_valid
        ):
            eligible.append(day)
    return eligible


def _registered_eligible_days(
    package: Path,
    split: str,
    connections: tuple[Any, ...],
    mapping_file: Path,
    path: Path | None,
) -> list[int]:
    """Use a pre-frozen date panel, while rejecting an invalid substitution.

    Spatial robustness changes the public network section only.  It must not
    silently change the Ausgrid dates because a smaller section happens to
    admit more complete profiles.  The registered file therefore selects a
    subset of the section-valid dates by ISO calendar date.
    """

    eligible = _eligible_days(package, split, connections, mapping_file)
    if path is None:
        return eligible
    if not path.is_file():
        raise RuntimeError(f"registered eligible-date file is missing: {path}")
    requested = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not requested or len(requested) != len(set(requested)):
        raise RuntimeError("registered eligible-date file must contain unique ISO dates")
    if requested != sorted(requested):
        raise RuntimeError("registered eligible dates must be sorted")
    names = (
        "AUSGRID_EVALUATION_2012_2013.npz"
        if split == "EVALUATION"
        else "AUSGRID_DEVELOPMENT_2011_2012.npz"
    )
    with np.load(package / names, allow_pickle=False) as current:
        dates = [str(value) for value in current["dates"]]
    index_by_date = {date: index for index, date in enumerate(dates)}
    missing = [date for date in requested if date not in index_by_date]
    if missing:
        raise RuntimeError(f"registered eligible dates are absent from split: {missing[:3]}")
    requested_indices = [index_by_date[date] for date in requested]
    invalid = sorted(set(requested_indices) - set(eligible))
    if invalid:
        raise RuntimeError(
            "registered eligible dates fail this section's source-quality contract: "
            f"{[dates[index] for index in invalid[:3]]}"
        )
    return requested_indices


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_day_summary(path: Path) -> list[dict[str, Any]]:
    """Recover a completed day table after a post-solve finalization failure."""

    float_fields = {
        "delivered_export_mwh",
        "authorized_export_mwh",
        "idle_authorization_mwh",
        "available_export_mwh",
        "curtailed_available_export_mwh",
        "post_event_signal_absolute_error_mwh",
        "raw_request_mwh",
        "admitted_request_mwh",
        "maximum_actual_voltage_pu",
        "minimum_actual_voltage_pu",
        "maximum_transformer_loading_pu",
        "participant_delivery_p10_mwh",
        "participant_delivery_median_mwh",
        "participant_delivery_p90_mwh",
    }
    int_fields = {"gate_count", "power_flow_count", "telemetry_dropout_count"}
    bool_fields = {"all_interval_physical_pair_pass"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    converted: list[dict[str, Any]] = []
    for row in rows:
        value: dict[str, Any] = dict(row)
        for key in float_fields:
            value[key] = float(row[key])
        for key in int_fields:
            value[key] = int(row[key])
        for key in bool_fields:
            value[key] = row[key] in {"1", "True", "true"}
        converted.append(value)
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/ees_transition/data/processed/australian_external_replay_v2",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        help=(
            "frozen load-to-Australian-profile map; defaults to the parent "
            "CSIRO replay map"
        ),
    )
    parser.add_argument(
        "--case-id",
        default=CASE_ID,
        help="registered case identifier recorded in the output summary",
    )
    parser.add_argument(
        "--dss-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/ees_transition/data/external/csiro_australian_mv_lv_feeder_v1/000065408v001/data",
    )
    parser.add_argument(
        "--lv-section-id",
        help=(
            "registered LV-section identifier for a spatial robustness replay; "
            "the local OpenDSS case must contain only this section's mapped loads"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/ees_transition/australian_external_replay_v2",
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/AUSTRALIAN_EXTERNAL_CASE_PROTOCOL_V2.yaml",
    )
    parser.add_argument(
        "--load-scale-calibration",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/AUSTRALIAN_LOAD_SCALE_DEVELOPMENT_CALIBRATION_V2.json",
    )
    parser.add_argument(
        "--authorization-margin-freeze",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/AUSTRALIAN_AUTHORIZATION_MARGIN_FREEZE_V2.json",
    )
    parser.add_argument(
        "--sensitivity-protocol",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs/ees_transition/AUSTRALIAN_TELEMETRY_AND_GAIN_SENSITIVITY_PROTOCOL_V1.yaml",
    )
    parser.add_argument(
        "--sensitivity-run-id",
        choices=tuple(SENSITIVITY_RUNS),
        help="select one preregistered gain or telemetry sensitivity",
    )
    parser.add_argument(
        "--evidence-class",
        choices=(
            "PRIMARY_EXTERNAL_EVALUATION",
            "SENSITIVITY",
            "SPATIAL_ROBUSTNESS",
            "DEVELOPMENT_OR_DIAGNOSTIC",
        ),
    )
    parser.add_argument(
        "--split", choices=("DEVELOPMENT", "EVALUATION"), default="EVALUATION"
    )
    parser.add_argument("--limit-days", type=int)
    parser.add_argument(
        "--eligible-dates-file",
        type=Path,
        help=(
            "frozen newline-delimited ISO-date panel; used to hold chronology "
            "constant across public LV-section replays"
        ),
    )
    parser.add_argument(
        "--resume-summary",
        action="store_true",
        help="finalize complete existing interval/day CSVs without re-solving",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    package = args.package.resolve()
    dss_root = args.dss_root.resolve()
    eligible_dates_file = (
        args.eligible_dates_file.resolve()
        if args.eligible_dates_file is not None
        else None
    )
    output_dir = args.output_dir.resolve()
    load_scale_calibration_path = args.load_scale_calibration.resolve()
    mapping_file = (
        args.mapping_file.resolve()
        if args.mapping_file is not None
        else package / "CSIRO_LOAD_TO_AUSGRID_PROFILE_MAPPING.csv"
    )
    if not mapping_file.is_file():
        raise RuntimeError(f"frozen mapping file is missing: {mapping_file}")
    connections = parse_dss_load_connections(dss_root)
    if args.lv_section_id is not None:
        discovered_sections = {item.feeder_id for item in connections}
        if discovered_sections != {args.lv_section_id}:
            raise RuntimeError(
                "LV-section identifier does not match the executable OpenDSS case: "
                f"expected={args.lv_section_id}, discovered={sorted(discovered_sections)}"
            )
    load_scale_calibration = json.loads(
        load_scale_calibration_path.read_text(encoding="utf-8")
    )
    if load_scale_calibration.get("status") != "PASS":
        raise RuntimeError("load-scale calibration is not PASS")
    load_scale = float(load_scale_calibration["selected_load_multiplier"])
    sensitivity_settings: dict[str, Any] = {
        "beta": PRIMARY_BETA,
        "telemetry_mode": "EXACT",
        "telemetry_scale": 1.0,
        "telemetry_dropout_fraction": 0.0,
        "telemetry_dropout_seed": "",
    }
    if args.sensitivity_run_id is not None:
        if args.split != "EVALUATION":
            raise RuntimeError("registered gain/telemetry sensitivities use EVALUATION only")
        if args.evidence_class not in (None, "SENSITIVITY"):
            raise RuntimeError("a sensitivity run id requires evidence class SENSITIVITY")
        sensitivity_settings.update(SENSITIVITY_RUNS[args.sensitivity_run_id])
        if not args.sensitivity_protocol.resolve().is_file():
            raise RuntimeError("sensitivity protocol is missing")
        resolved_evidence_class = "SENSITIVITY"
    else:
        resolved_evidence_class = args.evidence_class or (
            "PRIMARY_EXTERNAL_EVALUATION"
            if args.split == "EVALUATION" and args.limit_days is None
            else "DEVELOPMENT_OR_DIAGNOSTIC"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "intervals": output_dir / "AU_EXTERNAL_INTERVAL_RESULTS.csv",
        "days": output_dir / "AU_EXTERNAL_DAY_SUMMARY.csv",
        "summary": output_dir / "AU_EXTERNAL_RUN_SUMMARY.json",
        "manifest": output_dir / "AU_EXTERNAL_RESULT_SHA256_MANIFEST.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite and not args.resume_summary:
        raise RuntimeError("refusing to overwrite results without --overwrite")
    if args.resume_summary:
        if args.limit_days is not None:
            raise RuntimeError("--resume-summary cannot be combined with --limit-days")
        if not outputs["intervals"].is_file() or not outputs["days"].is_file():
            raise RuntimeError("resume requires complete interval and day CSVs")
        days = _read_day_summary(outputs["days"])
        eligible_dates = sorted({str(row["date"]) for row in days})
        if len(days) != len(eligible_dates) * len(CONTROLS):
            raise RuntimeError("existing day summary does not contain paired controls")
        eligible = list(range(len(eligible_dates)))
        worker_count = 0
    else:
        eligible = _registered_eligible_days(
            package, args.split, connections, mapping_file, eligible_dates_file
        )
        if args.limit_days is not None:
            if args.limit_days < 1:
                raise RuntimeError("--limit-days must be positive")
            eligible = eligible[: args.limit_days]
        config = {
            "package": str(package),
            "dss_root": str(dss_root),
            "mapping_file": str(mapping_file),
            "split": args.split,
            "load_scale": repr(load_scale),
            "beta": repr(sensitivity_settings["beta"]),
            "telemetry_mode": str(sensitivity_settings["telemetry_mode"]),
            "telemetry_scale": repr(sensitivity_settings["telemetry_scale"]),
            "telemetry_dropout_fraction": repr(
                sensitivity_settings["telemetry_dropout_fraction"]
            ),
            "telemetry_dropout_seed": str(
                sensitivity_settings["telemetry_dropout_seed"]
            ),
        }
        worker_count = min(args.workers, len(eligible))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=(config,),
        ) as executor:
            results = list(executor.map(_run_day, eligible, chunksize=1))
        results.sort(key=lambda value: value["date"])
        intervals = [record for result in results for record in result["interval_records"]]
        days = [record for result in results for record in result["day_summaries"]]
        _write_csv(outputs["intervals"], intervals)
        _write_csv(outputs["days"], days)

    aggregate: dict[str, Any] = {}
    for control in CONTROLS:
        selected = [row for row in days if row["control"] == control]
        aggregate[control] = {
            "trajectory_count": len(selected),
            "delivered_export_mwh": float(sum(row["delivered_export_mwh"] for row in selected)),
            "authorized_export_mwh": float(sum(row["authorized_export_mwh"] for row in selected)),
            "idle_authorization_mwh": float(sum(row["idle_authorization_mwh"] for row in selected)),
            "available_export_mwh": float(sum(row["available_export_mwh"] for row in selected)),
            "curtailed_available_export_mwh": float(sum(row["curtailed_available_export_mwh"] for row in selected)),
            "post_event_signal_absolute_error_mwh": float(
                sum(row["post_event_signal_absolute_error_mwh"] for row in selected)
            ),
            "telemetry_dropout_count": int(
                sum(row["telemetry_dropout_count"] for row in selected)
            ),
            "gate_count": int(sum(row["gate_count"] for row in selected)),
            "all_trajectory_physical_pair_pass": bool(all(row["all_interval_physical_pair_pass"] for row in selected)),
            "maximum_actual_voltage_pu": float(max(row["maximum_actual_voltage_pu"] for row in selected)),
            "minimum_actual_voltage_pu": float(min(row["minimum_actual_voltage_pu"] for row in selected)),
            "maximum_transformer_loading_pu": float(max(row["maximum_transformer_loading_pu"] for row in selected)),
            "power_flow_count": int(sum(row["power_flow_count"] for row in selected)),
        }
    input_hashes = {
        "case_input_manifest": _sha256(package / "CASE_INPUT_SHA256_MANIFEST.json"),
        "profile_mapping": _sha256(mapping_file),
        "protocol": _sha256(
            args.protocol.resolve()
        ),
        "load_scale_calibration": _sha256(
            load_scale_calibration_path
        ),
        "runner": _sha256(Path(__file__).resolve()),
    }
    input_paths = {
        "case_input_manifest": _repo_relative(
            package / "CASE_INPUT_SHA256_MANIFEST.json"
        ),
        "profile_mapping": _repo_relative(mapping_file),
        "protocol": _repo_relative(args.protocol.resolve()),
        "load_scale_calibration": _repo_relative(load_scale_calibration_path),
        "runner": _repo_relative(Path(__file__).resolve()),
    }
    for name, path in SCIENTIFIC_CORE_MODULES.items():
        input_hashes[name] = _sha256(path)
        input_paths[name] = _repo_relative(path)
    if args.split == "EVALUATION":
        input_hashes["authorization_margin_freeze"] = _sha256(
            args.authorization_margin_freeze.resolve()
        )
        input_paths["authorization_margin_freeze"] = _repo_relative(
            args.authorization_margin_freeze.resolve()
        )
    if args.sensitivity_run_id is not None:
        input_hashes["sensitivity_protocol"] = _sha256(
            args.sensitivity_protocol.resolve()
        )
        input_paths["sensitivity_protocol"] = _repo_relative(
            args.sensitivity_protocol.resolve()
        )
    if eligible_dates_file is not None:
        input_hashes["eligible_dates"] = _sha256(eligible_dates_file)
        input_paths["eligible_dates"] = _repo_relative(eligible_dates_file)
    summary = {
        "case_id": args.case_id,
        "lv_section_id": args.lv_section_id,
        "connection_count": len(connections),
        "evidence_class": resolved_evidence_class,
        "run_scope": (
            "SMOKE"
            if args.limit_days is not None
            else f"FULL_{args.split}"
        ),
        "eligible_network_day_count": len(eligible),
        "interval_count_per_control": len(eligible) * 48,
        "worker_count": worker_count,
        "pv_scale": PV_SCALE,
        "adoption_column": ADOPTION_COLUMN,
        "load_power_factor": LOAD_POWER_FACTOR,
        "load_scale": load_scale,
        "authorization_voltage_max_pu": AUTHORIZATION_VOLTAGE_MAX_PU,
        "forecast_quantile": FORECAST_QUANTILE,
        "beta": float(sensitivity_settings["beta"]),
        "sensitivity_run_id": args.sensitivity_run_id,
        "telemetry_mode": str(sensitivity_settings["telemetry_mode"]),
        "telemetry_scale": float(sensitivity_settings["telemetry_scale"]),
        "telemetry_dropout_fraction": float(
            sensitivity_settings["telemetry_dropout_fraction"]
        ),
        "telemetry_dropout_seed": str(
            sensitivity_settings["telemetry_dropout_seed"]
        ),
        "eligibility_policy": (
            "STRICT_ACTUAL_GC_GG_AND_EXPECTED_CL_WITH_CURRENT_AND_PREVIOUS_"
            "DAY_DST_EXCLUSION"
        ),
        "eligible_dates_file": (
            _repo_relative(eligible_dates_file)
            if eligible_dates_file is not None
            else None
        ),
        "aggregate": aggregate,
        "input_hashes": input_hashes,
        "input_paths": input_paths,
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "case_id": args.case_id,
        "hash_algorithm": "SHA-256",
        "payload": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
            if key != "manifest"
        ],
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
