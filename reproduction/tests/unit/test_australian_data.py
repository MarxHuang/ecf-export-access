from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from r4r.australian_data import (
    AUSGRID_CUSTOMER_COUNT,
    AUSGRID_INTERVAL_COUNT,
    AustralianDataError,
    DSSLoadConnection,
    connection_point_power,
    deterministic_profile_assignment,
    interval_energy_mwh,
    load_ausgrid_year,
    parse_dss_load_connections,
)


INTERVAL_LABELS = tuple(
    f"{(index // 2) % 24}:{'30' if index % 2 else '00'}"
    for index in range(1, AUSGRID_INTERVAL_COUNT + 1)
)


def _write_ausgrid_fixture(
    path: Path, *, omit_last_gg: bool = False, first_quality: str = ""
) -> None:
    header = [
        "Customer",
        "Generator Capacity",
        "Postcode",
        "Consumption Category",
        "date",
        *INTERVAL_LABELS,
        "Row Quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("source notice," + "," * (len(header) - 2) + "\n")
        writer = csv.writer(stream)
        writer.writerow(header)
        for customer in range(1, AUSGRID_CUSTOMER_COUNT + 1):
            writer.writerow(
                [customer, 3.0, 2000, "GC", "1/07/2012"]
                + [0.25] * AUSGRID_INTERVAL_COUNT
                + [first_quality if customer == 1 else ""]
            )
            if not (omit_last_gg and customer == AUSGRID_CUSTOMER_COUNT):
                writer.writerow(
                    [customer, 3.0, 2000, "GG", "1/07/2012"]
                    + [0.75] * AUSGRID_INTERVAL_COUNT
                    + [""]
                )


def test_load_ausgrid_year_preserves_energy_and_converts_to_power(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source)

    year = load_ausgrid_year(source)

    assert year.gross_generation_kwh.shape == (1, 48, 300)
    assert np.all(year.complete_customer_day)
    assert year.gross_generation_kw[0, 0, 0] == pytest.approx(1.5)
    assert year.load_kw[0, 0, 0] == pytest.approx(0.5)
    assert year.available_export_kw[0, 0, 0] == pytest.approx(1.0)
    assert year.residual_demand_kw[0, 0, 0] == pytest.approx(0.0)


def test_missing_mandatory_row_is_exposed_not_imputed(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source, omit_last_gg=True)

    year = load_ausgrid_year(source)

    assert not year.complete_customer_day[0, -1]
    assert np.isnan(year.gross_generation_kwh[0, 0, -1])
    assert np.isnan(year.available_export_kw[0, 0, -1])


def test_literal_na_quality_is_not_parsed_as_missing(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source, first_quality="NA")

    year = load_ausgrid_year(source)

    assert year.source_row_present[0, 0, 0]
    assert not year.source_row_actual[0, 0, 0]
    assert not year.complete_customer_day[0, 0]


def test_unsupported_quality_code_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source, first_quality="Estimated")

    with pytest.raises(AustralianDataError, match="unsupported"):
        load_ausgrid_year(source)


def test_dst_transition_day_is_not_valid_for_state_recursion(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source)
    text = source.read_text(encoding="utf-8").replace("1/07/2012", "7/10/2012")
    source.write_text(text, encoding="utf-8")

    year = load_ausgrid_year(source)

    assert np.all(year.complete_customer_day)
    assert not year.chronology_valid_day[0]


def test_partial_controlled_load_channel_is_not_silently_zero_filled(tmp_path: Path) -> None:
    source = tmp_path / "year.csv"
    _write_ausgrid_fixture(source)
    with source.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [1, 3.0, 2000, "CL", "1/07/2012"]
            + [0.1] * AUSGRID_INTERVAL_COUNT
            + [""]
        )
        for customer in range(1, AUSGRID_CUSTOMER_COUNT + 1):
            writer.writerow(
                [customer, 3.0, 2000, "GC", "2/07/2012"]
                + [0.25] * AUSGRID_INTERVAL_COUNT
                + [""]
            )
            writer.writerow(
                [customer, 3.0, 2000, "GG", "2/07/2012"]
                + [0.75] * AUSGRID_INTERVAL_COUNT
                + [""]
            )

    year = load_ausgrid_year(source)

    assert year.controlled_load_expected[0]
    assert year.complete_customer_day[0, 0]
    assert not year.complete_customer_day[1, 0]
    assert np.isnan(year.controlled_load_kwh[1, 0, 0])


def test_dss_parser_and_assignment_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "Master.dss").write_text("clear\n", encoding="utf-8")
    feeder = tmp_path / "LV" / "LV01"
    feeder.mkdir(parents=True)
    (feeder / "Loads.dss").write_text(
        "new load.site_a phases=1 bus1=bus_a.1 kV=0.23 kW=1\n"
        "new load.site_b phases=1 bus1=bus_b.2 kV=0.23 kW=1\n",
        encoding="utf-8",
    )

    connections = parse_dss_load_connections(tmp_path)
    first = deterministic_profile_assignment(connections, [1, 2], seed="fixed")
    second = deterministic_profile_assignment(connections, [1, 2], seed="fixed")

    assert connections == (
        DSSLoadConnection("site_a", "bus_a", 1, "LV01", "LV/LV01/Loads.dss"),
        DSSLoadConnection("site_b", "bus_b", 2, "LV01", "LV/LV01/Loads.dss"),
    )
    assert first == second
    assert sorted(first.values()) == [1, 2]


def test_dss_parser_accepts_an_executable_lv_section_master(tmp_path: Path) -> None:
    (tmp_path / "Master.dss").write_text("clear\n", encoding="utf-8")
    (tmp_path / "Loads.dss").write_text(
        "new load.section_a phases=1 bus1=bus_a.1 kV=0.23 kW=1\n"
        "new load.section_b phases=1 bus1=bus_b.3 kV=0.23 kW=1\n",
        encoding="utf-8",
    )

    connections = parse_dss_load_connections(tmp_path)

    assert connections == (
        DSSLoadConnection("section_a", "bus_a", 1, tmp_path.name, "Loads.dss"),
        DSSLoadConnection("section_b", "bus_b", 3, tmp_path.name, "Loads.dss"),
    )


def test_dss_parser_preserves_a_multiphase_premise_without_splitting_it(tmp_path: Path) -> None:
    (tmp_path / "Master.dss").write_text("clear\n", encoding="utf-8")
    (tmp_path / "Loads.dss").write_text(
        "new load.premise_a phases=3 bus1=bus_a.1.2.3.0 kV=0.433 kW=3\n"
        "new load.premise_b phases=2 bus1=bus_b.2.3.0 kV=0.4 kW=2\n",
        encoding="utf-8",
    )

    connections = parse_dss_load_connections(tmp_path)

    assert len(connections) == 2
    assert connections[0].load_name == "premise_a"
    assert connections[0].phases == (1, 2, 3)
    assert connections[1].load_name == "premise_b"
    assert connections[1].phases == (2, 3)


def test_energy_integration_does_not_annualize() -> None:
    assert interval_energy_mwh(np.array([1.0, 2.0])) == pytest.approx(0.0015)
    with pytest.raises(AustralianDataError, match="nonnegative"):
        interval_energy_mwh(np.array([1.0, -1.0]))


def test_connection_point_power_separates_self_consumption_and_export() -> None:
    p_load, q_load, delivered, available = connection_point_power(
        household_load_kw=[1.0, 2.0, 1.0],
        gross_pv_kw=[3.0, 1.0, 4.0],
        allocated_export_kw=[1.5, 5.0, 5.0],
        participant_mask=[True, True, False],
        load_power_factor=1.0,
    )
    assert available == pytest.approx([2.0, 0.0, 0.0])
    assert delivered == pytest.approx([1.5, 0.0, 0.0])
    assert p_load == pytest.approx([-1.5, 1.0, 1.0])
    assert q_load == pytest.approx([0.0, 0.0, 0.0])


def test_connection_point_power_rejects_allocation_below_zero() -> None:
    with pytest.raises(AustralianDataError, match="nonnegative"):
        connection_point_power([1.0], [2.0], [-1.0], [True], load_power_factor=0.95)


def test_load_reactive_power_is_not_based_on_net_export() -> None:
    _, q_load, _, _ = connection_point_power(
        [2.0], [5.0], [3.0], [True], load_power_factor=0.8
    )
    assert q_load == pytest.approx([1.5])
