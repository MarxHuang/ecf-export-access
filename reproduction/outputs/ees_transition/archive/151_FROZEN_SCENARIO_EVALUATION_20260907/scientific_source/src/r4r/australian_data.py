"""Adapters for the public Australian feeder and household time-series case.

This module deliberately keeps the two public sources separate.  The CSIRO /
GridQube package supplies a real-world-derived network model, whereas the
Ausgrid files supply half-hour gross PV and household demand records from a
different customer sample.  Joining them creates a data-calibrated replay, not
a field-matched digital twin.

Ausgrid interval values are energy in kWh over the half-hour ending at the
column label.  They are converted to interval-average kW with the registered
0.5 h duration.  Gross generation is first used to serve household demand;
only the positive residual is available at the connection point for export.
No negative or missing value is silently clipped or imputed by this loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


AUSGRID_INTERVAL_HOURS = 0.5
AUSGRID_CUSTOMER_COUNT = 300
AUSGRID_INTERVAL_COUNT = 48
AUSGRID_CATEGORIES = ("GC", "CL", "GG")
AUSGRID_DST_TRANSITION_DATES = frozenset(
    {
        date(2010, 10, 3),
        date(2011, 4, 3),
        date(2011, 10, 2),
        date(2012, 4, 1),
        date(2012, 10, 7),
        date(2013, 4, 7),
    }
)


class AustralianDataError(ValueError):
    """Raised when a public input violates the declared data contract."""


@dataclass(frozen=True, slots=True)
class AusgridYear:
    """One July--June Ausgrid year in a dense, typed representation.

    Arrays are ordered ``[day, interval, customer]``.  Energy arrays retain
    the source unit.  Derived power arrays are properties so the unit
    conversion cannot diverge between runners.
    """

    source_path: Path
    source_sha256: str
    dates: tuple[date, ...]
    customer_ids: tuple[int, ...]
    generator_capacity_kwp: np.ndarray
    postcodes: np.ndarray
    gross_generation_kwh: np.ndarray
    general_consumption_kwh: np.ndarray
    controlled_load_kwh: np.ndarray
    source_row_present: np.ndarray
    source_row_actual: np.ndarray
    controlled_load_expected: np.ndarray
    chronology_valid_day: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.dates), AUSGRID_INTERVAL_COUNT, len(self.customer_ids))
        for name in (
            "gross_generation_kwh",
            "general_consumption_kwh",
            "controlled_load_kwh",
        ):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != expected:
                raise AustralianDataError(f"{name} must have shape {expected}")
            if not np.issubdtype(value.dtype, np.floating):
                raise AustralianDataError(f"{name} must contain floating-point data")
            finite = np.isfinite(value)
            if np.any(value[finite] < 0.0):
                raise AustralianDataError(f"{name} contains negative interval energy")
        if self.source_row_present.shape != (
            len(self.dates), len(self.customer_ids), len(AUSGRID_CATEGORIES)
        ):
            raise AustralianDataError("source_row_present has an invalid shape")
        if self.source_row_actual.shape != self.source_row_present.shape:
            raise AustralianDataError("source_row_actual has an invalid shape")
        if self.controlled_load_expected.shape != (len(self.customer_ids),):
            raise AustralianDataError("controlled_load_expected has an invalid shape")
        if self.chronology_valid_day.shape != (len(self.dates),):
            raise AustralianDataError("chronology_valid_day has an invalid shape")
        if self.generator_capacity_kwp.shape != (len(self.customer_ids),):
            raise AustralianDataError("generator capacities must align with customers")
        if self.postcodes.shape != (len(self.customer_ids),):
            raise AustralianDataError("postcodes must align with customers")
        if np.any(~np.isfinite(self.generator_capacity_kwp)) or np.any(
            self.generator_capacity_kwp <= 0.0
        ):
            raise AustralianDataError("generator capacities must be finite and positive")

    @property
    def load_kwh(self) -> np.ndarray:
        """Household consumption including controlled load, in kWh."""

        return self.general_consumption_kwh + self.controlled_load_kwh

    @property
    def gross_generation_kw(self) -> np.ndarray:
        return self.gross_generation_kwh / AUSGRID_INTERVAL_HOURS

    @property
    def load_kw(self) -> np.ndarray:
        return self.load_kwh / AUSGRID_INTERVAL_HOURS

    @property
    def available_export_kw(self) -> np.ndarray:
        """Counterfactual connection-point export available before curtailment."""

        return np.maximum(self.gross_generation_kw - self.load_kw, 0.0)

    @property
    def residual_demand_kw(self) -> np.ndarray:
        """Connection-point demand remaining after contemporaneous gross PV."""

        return np.maximum(self.load_kw - self.gross_generation_kw, 0.0)

    @property
    def complete_customer_day(self) -> np.ndarray:
        """Strict primary-profile completeness for each customer-day.

        GC and GG must be present and source-asserted actual.  If a customer
        has any CL row in the source year, CL is treated as an expected channel
        for that customer throughout the year; a missing or non-actual CL row
        therefore makes that customer-day ineligible.  Customers with no CL
        channel anywhere in the year retain the physically meaningful zero.
        """

        gc_index = AUSGRID_CATEGORIES.index("GC")
        cl_index = AUSGRID_CATEGORIES.index("CL")
        gg_index = AUSGRID_CATEGORIES.index("GG")
        gc_ok = self.source_row_present[:, :, gc_index] & self.source_row_actual[:, :, gc_index]
        gg_ok = self.source_row_present[:, :, gg_index] & self.source_row_actual[:, :, gg_index]
        cl_ok = (
            ~self.controlled_load_expected[None, :]
            | (
                self.source_row_present[:, :, cl_index]
                & self.source_row_actual[:, :, cl_index]
            )
        )
        return gc_ok & gg_ok & cl_ok

    @property
    def complete_day(self) -> np.ndarray:
        return np.all(self.complete_customer_day, axis=1)

    def to_metadata(self) -> dict[str, object]:
        complete = self.complete_customer_day
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "date_start": self.dates[0].isoformat(),
            "date_end": self.dates[-1].isoformat(),
            "day_count": len(self.dates),
            "customer_count": len(self.customer_ids),
            "interval_count_per_day": AUSGRID_INTERVAL_COUNT,
            "interval_hours": AUSGRID_INTERVAL_HOURS,
            "mandatory_customer_days_complete": int(np.sum(complete)),
            "mandatory_customer_days_total": int(complete.size),
            "fully_complete_days": int(np.sum(self.complete_day)),
            "chronology_valid_days": int(np.sum(self.chronology_valid_day)),
            "dst_transition_days_excluded_from_state_recursion": int(
                np.sum(~self.chronology_valid_day)
            ),
            "source_rows_present": int(np.sum(self.source_row_present)),
            "source_rows_actual": int(np.sum(self.source_row_actual)),
            "source_rows_nonactual": int(
                np.sum(self.source_row_present & ~self.source_row_actual)
            ),
            "customers_with_controlled_load_channel": int(
                np.sum(self.controlled_load_expected)
            ),
            "gross_generation_kwh": float(np.nansum(self.gross_generation_kwh)),
            "general_consumption_kwh": float(np.nansum(self.general_consumption_kwh)),
            "controlled_load_kwh": float(np.nansum(self.controlled_load_kwh)),
        }


@dataclass(frozen=True, slots=True)
class DSSLoadConnection:
    """One published OpenDSS load premise used by the external replay.

    ``phase`` preserves the legacy primary-phase field used in single-phase
    public cases. ``phases`` retains the declared phase set for a multi-phase
    load. A participant remains one premise with one request, authorization,
    and delivery total; it is never silently split into separate phase-level
    participants.
    """

    load_name: str
    bus_name: str
    phase: int
    feeder_id: str
    source_file: str
    phases: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.load_name or not self.bus_name or not self.feeder_id:
            raise AustralianDataError("DSS connection identifiers cannot be empty")
        if self.phase not in (1, 2, 3):
            raise AustralianDataError("DSS load phase must be 1, 2, or 3")
        declared = self.phases or (self.phase,)
        if not declared or any(value not in (1, 2, 3) for value in declared):
            raise AustralianDataError("DSS load phases must lie in {1, 2, 3}")
        if len(set(declared)) != len(declared):
            raise AustralianDataError("DSS load phases must be unique")
        if self.phase != declared[0]:
            raise AustralianDataError("primary DSS phase must be the first declared phase")
        object.__setattr__(self, "phases", tuple(declared))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_columns(columns: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in columns)
    if len(values) != AUSGRID_INTERVAL_COUNT:
        raise AustralianDataError(
            f"expected {AUSGRID_INTERVAL_COUNT} interval columns, found {len(values)}"
        )
    if values[0] != "0:30" or values[-1] != "0:00":
        raise AustralianDataError("unexpected Ausgrid half-hour column order")
    if len(set(values)) != len(values):
        raise AustralianDataError("Ausgrid interval labels are not unique")
    return values


def load_ausgrid_year(path: str | Path) -> AusgridYear:
    """Read and validate one original Ausgrid yearly CSV.

    The first line is the source notice, not the header.  Mandatory GC/GG rows
    that are absent remain NaN and are exposed through ``complete_customer_day``;
    the loader never fills them.  Optional absent CL rows are represented as
    zero only because the category is not applicable to every customer.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise AustralianDataError(f"Ausgrid source does not exist: {source}")
    # ``NA`` is a literal Ausgrid row-quality code, not a missing value.  The
    # pandas default NA parser would silently erase that distinction.
    frame = pd.read_csv(
        source,
        skiprows=1,
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )
    required = {
        "Customer",
        "Generator Capacity",
        "Postcode",
        "Consumption Category",
        "date",
    }
    missing = required - set(frame.columns)
    if missing:
        raise AustralianDataError(f"Ausgrid source is missing columns: {sorted(missing)}")
    interval_columns = _interval_columns(frame.columns[5:53])
    if "Row Quality" in frame.columns:
        quality = frame["Row Quality"].astype(str).str.strip()
        unsupported = ~quality.isin(("", "NA"))
        if bool(unsupported.any()):
            counts = quality.loc[unsupported].value_counts().to_dict()
            raise AustralianDataError(f"unsupported Ausgrid row-quality codes: {counts}")
        frame = frame.assign(_row_actual=quality.eq(""))
    else:
        # The source notes assert that the selected first-year panel contains
        # complete actual records; the CSV itself has no Row Quality field.
        frame = frame.assign(_row_actual=True)

    frame["Customer"] = pd.to_numeric(frame["Customer"], errors="raise").astype(int)
    customers = tuple(sorted(int(value) for value in frame["Customer"].unique()))
    if customers != tuple(range(1, AUSGRID_CUSTOMER_COUNT + 1)):
        raise AustralianDataError("Ausgrid customer IDs must be the complete 1--300 set")
    categories = set(frame["Consumption Category"].astype(str).str.strip())
    if not categories.issubset(set(AUSGRID_CATEGORIES)) or not {"GC", "GG"}.issubset(categories):
        raise AustralianDataError(f"unexpected or incomplete categories: {sorted(categories)}")

    # The first release uses ``1-Jul-10`` while the later releases use
    # ``1/07/2011``.  ``format="mixed"`` makes that documented source change
    # explicit and avoids locale-dependent inference.
    parsed_dates = pd.to_datetime(
        frame["date"], format="mixed", dayfirst=True, errors="raise"
    )
    frame = frame.assign(_date=parsed_dates.dt.date)
    dates = tuple(sorted(frame["_date"].unique()))
    date_index = {value: index for index, value in enumerate(dates)}
    customer_index = {value: index for index, value in enumerate(customers)}
    capacities_by_customer = frame.groupby("Customer")["Generator Capacity"].nunique()
    postcodes_by_customer = frame.groupby("Customer")["Postcode"].nunique()
    if bool((capacities_by_customer != 1).any()) or bool((postcodes_by_customer != 1).any()):
        raise AustralianDataError("customer capacity or postcode changes within a source year")
    capacity = (
        frame.groupby("Customer")["Generator Capacity"]
        .first()
        .reindex(customers)
        .to_numpy(dtype=float)
    )
    postcode = (
        frame.groupby("Customer")["Postcode"]
        .first()
        .reindex(customers)
        .to_numpy(dtype=int)
    )

    shape = (len(dates), AUSGRID_INTERVAL_COUNT, len(customers))
    cl_customers = set(
        int(value)
        for value in frame.loc[
            frame["Consumption Category"].astype(str).str.strip().eq("CL"),
            "Customer",
        ].unique()
    )
    cl_expected = np.asarray([customer in cl_customers for customer in customers], dtype=bool)
    arrays = {
        "GG": np.full(shape, np.nan, dtype=np.float32),
        "GC": np.full(shape, np.nan, dtype=np.float32),
        "CL": np.zeros(shape, dtype=np.float32),
    }
    arrays["CL"][:, :, cl_expected] = np.nan
    present = np.zeros((len(dates), len(customers), len(AUSGRID_CATEGORIES)), dtype=bool)
    actual = np.zeros_like(present)
    duplicate_keys = frame.duplicated(["_date", "Customer", "Consumption Category"])
    if bool(duplicate_keys.any()):
        duplicate = frame.loc[
            duplicate_keys, ["_date", "Customer", "Consumption Category"]
        ].head(10)
        raise AustralianDataError(f"duplicate customer-day-category rows: {duplicate.to_dict('records')}")

    for category in AUSGRID_CATEGORIES:
        rows = frame.loc[frame["Consumption Category"] == category]
        if rows.empty:
            continue
        values = rows.loc[:, interval_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise AustralianDataError(f"{category} contains missing, non-finite, or negative values")
        day_positions = rows["_date"].map(date_index).to_numpy(dtype=int)
        customer_positions = rows["Customer"].map(customer_index).to_numpy(dtype=int)
        arrays[category][day_positions, :, customer_positions] = values
        present[
            day_positions,
            customer_positions,
            AUSGRID_CATEGORIES.index(category),
        ] = True
        actual[
            day_positions,
            customer_positions,
            AUSGRID_CATEGORIES.index(category),
        ] = rows["_row_actual"].to_numpy(dtype=bool)

    return AusgridYear(
        source_path=source,
        source_sha256=sha256_file(source),
        dates=dates,
        customer_ids=customers,
        generator_capacity_kwp=capacity,
        postcodes=postcode,
        gross_generation_kwh=arrays["GG"],
        general_consumption_kwh=arrays["GC"],
        controlled_load_kwh=arrays["CL"],
        source_row_present=present,
        source_row_actual=actual,
        controlled_load_expected=cl_expected,
        chronology_valid_day=np.asarray(
            [value not in AUSGRID_DST_TRANSITION_DATES for value in dates], dtype=bool
        ),
    )


_LOAD_LINE = re.compile(
    r"^\s*new\s+load\.(?P<name>\S+)\s+.*?\bbus1=(?P<bus>\S+)",
    flags=re.IGNORECASE,
)


def parse_dss_load_connections(data_root: str | Path) -> tuple[DSSLoadConnection, ...]:
    """Parse load premises, declared phase sets, and LV-section provenance.

    The public Australian master model stores loads below ``LV/*/Loads.dss``.
    Its LV-section masters, however, are separately executable OpenDSS cases
    with a local ``Loads.dss``.  Supporting both layouts lets a spatial
    robustness replay use the published section model itself rather than
    cutting loads out of the full feeder in memory.
    """

    root = Path(data_root).resolve()
    if not (root / "Master.dss").is_file():
        raise AustralianDataError("CSIRO data root must contain Master.dss")
    nested_paths = sorted(root.glob("LV/*/Loads.dss"))
    local_path = root / "Loads.dss"
    if nested_paths:
        load_paths = nested_paths
    elif local_path.is_file():
        load_paths = [local_path]
    else:
        load_paths = []
    connections: list[DSSLoadConnection] = []
    for path in load_paths:
        feeder_id = path.parent.name
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "//")):
                continue
            match = _LOAD_LINE.match(line)
            if match is None:
                raise AustralianDataError(f"unparsed DSS load row {path}:{line_number}")
            bus_token = match.group("bus")
            parts = bus_token.split(".")
            if len(parts) < 2:
                raise AustralianDataError(f"DSS load bus omits phase at {path}:{line_number}")
            try:
                conductors = tuple(int(token) for token in parts[1:] if token != "")
            except ValueError as exc:
                raise AustralianDataError(f"invalid DSS phase at {path}:{line_number}") from exc
            phase_match = re.search(r"\bphases\s*=\s*(\d+)\b", line, re.IGNORECASE)
            connection_match = re.search(r"\bconn(?:ection)?\s*=\s*(\w+)", line, re.IGNORECASE)
            connection = connection_match.group(1).lower() if connection_match else "wye"
            # OpenDSS wye terminals have nphase + 1 conductors. The final
            # conductor is neutral, not another phase: it need not be node 0.
            # The published GridQube single-phase loads use bus.phase.4.
            # https://opendss.epri.com/NeutralRules.html
            if phase_match and connection in ("wye", "y", "ln", "star"):
                declared_count = int(phase_match.group(1))
                if len(conductors) == declared_count + 1:
                    phases = conductors[:-1]
                else:
                    phases = conductors
                if len(phases) != declared_count:
                    raise AustralianDataError(
                        f"DSS phase count and explicit bus nodes disagree at {path}:{line_number}"
                    )
            else:
                phases = tuple(node for node in conductors if node != 0)
            if not phases:
                raise AustralianDataError(f"DSS load bus omits a live phase at {path}:{line_number}")
            connections.append(
                DSSLoadConnection(
                    load_name=match.group("name").lower(),
                    bus_name=parts[0].lower(),
                    phase=phases[0],
                    feeder_id=feeder_id,
                    source_file=str(path.relative_to(root)).replace("\\", "/"),
                    phases=phases,
                )
            )
    if not connections:
        raise AustralianDataError("no LV load connections were parsed")
    names = [item.load_name for item in connections]
    if len(set(names)) != len(names):
        raise AustralianDataError("DSS load names are not globally unique")
    return tuple(connections)


def deterministic_profile_assignment(
    connections: Sequence[DSSLoadConnection],
    customer_ids: Sequence[int],
    *,
    seed: str,
) -> dict[str, int]:
    """Assign public profiles to connections without data-dependent selection.

    Each profile appears once before reuse.  The hash-based order is stable
    across platforms and independent of any simulated outcome.
    """

    if not connections or not customer_ids:
        raise AustralianDataError("connections and customer IDs cannot be empty")
    unique_customers = tuple(sorted(set(int(value) for value in customer_ids)))
    if len(unique_customers) != len(customer_ids):
        raise AustralianDataError("customer IDs must be unique")

    def rank(token: str) -> str:
        return hashlib.sha256(f"{seed}|{token}".encode("utf-8")).hexdigest()

    ordered_connections = sorted(connections, key=lambda item: rank(item.load_name))
    ordered_customers = sorted(unique_customers, key=lambda item: rank(f"customer:{item}"))
    return {
        connection.load_name: ordered_customers[index % len(ordered_customers)]
        for index, connection in enumerate(ordered_connections)
    }


def validate_year_compatibility(years: Sequence[AusgridYear]) -> None:
    """Require stable customer metadata across chronological splits."""

    if not years:
        raise AustralianDataError("at least one Ausgrid year is required")
    first = years[0]
    previous = first
    for year in years[1:]:
        if year.customer_ids != first.customer_ids:
            raise AustralianDataError("customer order changes between source years")
        if not np.array_equal(year.generator_capacity_kwp, first.generator_capacity_kwp):
            raise AustralianDataError("generator capacities change between source years")
        if not np.array_equal(year.postcodes, first.postcodes):
            raise AustralianDataError("postcodes change between source years")
        if year.dates[0] <= previous.dates[-1]:
            raise AustralianDataError("Ausgrid years are not in chronological order")
        previous = year


def interval_energy_mwh(power_kw: np.ndarray, *, interval_hours: float = AUSGRID_INTERVAL_HOURS) -> float:
    """Integrate interval-average kW without annualizing unrepresented periods."""

    values = np.asarray(power_kw, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise AustralianDataError("energy integration requires finite nonnegative power")
    if not math.isfinite(interval_hours) or interval_hours <= 0.0:
        raise AustralianDataError("interval duration must be finite and positive")
    return float(np.sum(values) * interval_hours / 1000.0)


def connection_point_power(
    household_load_kw: Sequence[float],
    gross_pv_kw: Sequence[float],
    allocated_export_kw: Sequence[float],
    participant_mask: Sequence[bool],
    *,
    load_power_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map household data and export access to connection-point P/Q.

    Positive P and Q denote consumption.  PV operates at unity power factor;
    household Q is therefore based on gross household load, not residual P.
    Nonparticipants retain their full household load and have no modelled PV.
    """

    load = np.asarray(household_load_kw, dtype=float)
    pv = np.asarray(gross_pv_kw, dtype=float)
    allocation = np.asarray(allocated_export_kw, dtype=float)
    participant = np.asarray(participant_mask, dtype=bool)
    if not (load.shape == pv.shape == allocation.shape == participant.shape):
        raise AustralianDataError("PCC input vectors must have identical shapes")
    if load.ndim != 1 or load.size == 0:
        raise AustralianDataError("PCC input vectors must be nonempty and one-dimensional")
    if np.any(~np.isfinite(load)) or np.any(~np.isfinite(pv)) or np.any(~np.isfinite(allocation)):
        raise AustralianDataError("PCC input vectors must be finite")
    if np.any(load < 0.0) or np.any(pv < 0.0) or np.any(allocation < 0.0):
        raise AustralianDataError("load, PV, and allocation must be nonnegative")
    if not math.isfinite(load_power_factor) or not 0.0 < load_power_factor <= 1.0:
        raise AustralianDataError("load power factor must lie in (0, 1]")

    participant_pv = np.where(participant, pv, 0.0)
    available_export = np.maximum(participant_pv - load, 0.0)
    realized_export = np.minimum(allocation, available_export)
    residual_demand = np.maximum(load - participant_pv, 0.0)
    p_load = residual_demand - realized_export
    q_load = load * math.tan(math.acos(load_power_factor))
    return p_load, q_load, realized_export, available_export


__all__ = [
    "AUSGRID_CATEGORIES",
    "AUSGRID_CUSTOMER_COUNT",
    "AUSGRID_DST_TRANSITION_DATES",
    "AUSGRID_INTERVAL_COUNT",
    "AUSGRID_INTERVAL_HOURS",
    "AustralianDataError",
    "AusgridYear",
    "DSSLoadConnection",
    "connection_point_power",
    "deterministic_profile_assignment",
    "interval_energy_mwh",
    "load_ausgrid_year",
    "parse_dss_load_connections",
    "sha256_file",
    "validate_year_compatibility",
]
