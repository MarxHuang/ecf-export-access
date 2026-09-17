"""Fail-closed parser for the frozen MATPOWER case input.

Round 2 is intentionally limited to source parsing and static validation.  This
module never executes MATLAB, calls a power-flow solver, changes topology, or
applies a load-scale scenario.  The case file's literal matrices are read and
the conversion statements that are present in the pinned source are checked
before the typed :class:`NetworkModel` is constructed.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from r4r.errors import ValidationError
from r4r.models.network import Branch, Bus, NetworkModel
from r4r.types import FloatVector, FiniteFloat, Identifier, NonNegativeFloat, ObjectReference, Sha256


class MatpowerParseError(ValidationError):
    """The source is not a supported, statically verifiable MATPOWER case."""


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$")
_MATRIX = re.compile(r"mpc\.(?P<name>bus|gen|branch)\s*=\s*\[(?P<body>.*?)\];", re.S)


@dataclass(frozen=True, slots=True)
class MatpowerMapping:
    """The source-declared units and deterministic conversion policy."""

    mapping_id: str
    source_version: str
    branch_source_unit: str
    load_source_unit: str
    power_factor: FiniteFloat
    base_impedance_ohm: FiniteFloat


@dataclass(frozen=True, slots=True)
class ParsedMatpowerCase:
    """Static case data produced from a pinned MATPOWER source file."""

    network: NetworkModel
    bus_types: tuple[int, ...]
    branch_status: tuple[int, ...]
    p_load_mw: FloatVector
    q_load_mvar: FloatVector
    # Source-declared operating-point fields are preserved as runtime inputs for
    # the authorized AC solver slice.  They are not optimization results.
    bus_vm_pu: FloatVector
    bus_va_deg: FloatVector
    bus_vmax_pu: FloatVector
    bus_vmin_pu: FloatVector
    p_generation_mw: FloatVector
    q_generation_mvar: FloatVector
    branch_b_pu: FloatVector
    mapping: MatpowerMapping
    source_hash: Sha256
    source_path: str


def _float(token: str, *, context: str) -> float:
    token = token.strip()
    if not _NUMBER.fullmatch(token):
        raise MatpowerParseError(f"non-literal numeric token in {context}: {token!r}")
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise MatpowerParseError(f"non-finite numeric token in {context}")
    return value


def _matrix_rows(body: str, *, name: str, columns: int) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for line_no, raw_line in enumerate(body.splitlines(), 1):
        line = raw_line.split("%", 1)[0].strip().rstrip(";").strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) != columns:
            raise MatpowerParseError(
                f"{name} row {line_no} has {len(tokens)} columns; expected {columns}"
            )
        rows.append(tuple(_float(token, context=f"{name} row {line_no}") for token in tokens))
    if not rows:
        raise MatpowerParseError(f"{name} matrix is empty")
    return tuple(rows)


def _integer(value: float, *, context: str) -> int:
    if not value.is_integer():
        raise MatpowerParseError(f"{context} must be an integer")
    return int(value)


def _source_mapping(source: str, base_mva: float, base_kv: float) -> MatpowerMapping:
    version = re.search(r"mpc\.version\s*=\s*'([^']+)'", source)
    if not version or version.group(1) != "2":
        raise MatpowerParseError("only MATPOWER case format version 2 is supported")
    if not re.search(r"mpc\.branch\(:,\s*\[BR_R\s+BR_X\]\)\s*=\s*mpc\.branch\(:,\s*\[BR_R\s+BR_X\]\)\s*/\s*\(Vbase\^2\s*/\s*Sbase\)", source):
        raise MatpowerParseError("branch Ohm-to-pu conversion declaration is missing")
    if not re.search(r"mpc\.bus\(:,\s*\[PD,\s*QD\]\)\s*=\s*mpc\.bus\(:,\s*\[PD,\s*QD\]\)\s*/\s*1e3", source):
        raise MatpowerParseError("load kVA-to-MVA conversion declaration is missing")
    pf_match = re.search(r"\bpf\s*=\s*([0-9.]+)\s*;", source)
    if not pf_match:
        raise MatpowerParseError("power-factor declaration is missing")
    power_factor = _float(pf_match.group(1), context="pf")
    if not 0.0 < power_factor <= 1.0:
        raise MatpowerParseError("power factor must be in (0, 1]")
    zbase = (base_kv * 1e3) ** 2 / (base_mva * 1e6)
    if not math.isfinite(zbase) or zbase <= 0:
        raise MatpowerParseError("base impedance is not positive and finite")
    return MatpowerMapping(
        mapping_id="MATPOWER_V2_CASE141_LITERAL_CONVERSION_V1",
        source_version="2",
        branch_source_unit="ohm",
        load_source_unit="kVA",
        power_factor=FiniteFloat(power_factor),
        base_impedance_ohm=FiniteFloat(zbase),
    )


def _connected(bus_ids: set[int], edges: Iterable[tuple[int, int]]) -> bool:
    adjacency = {bus_id: set() for bus_id in bus_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    stack = [min(bus_ids)]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    return seen == bus_ids


def parse_matpower_case(path: str | Path, *, expected_sha256: str | None = None) -> ParsedMatpowerCase:
    """Parse one pinned MATPOWER case without executing the source file."""

    source_path = Path(path)
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    if expected_sha256 is not None and digest != expected_sha256.upper():
        raise MatpowerParseError("source SHA-256 does not match the pinned input")
    source = raw.decode("utf-8")
    matrices = {match.group("name"): match.group("body") for match in _MATRIX.finditer(source)}
    if set(matrices) != {"bus", "gen", "branch"}:
        raise MatpowerParseError("literal mpc.bus, mpc.gen and mpc.branch matrices are required")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", source)
    if not base_match:
        raise MatpowerParseError("mpc.baseMVA declaration is missing")
    base_mva = _float(base_match.group(1), context="baseMVA")
    buses = _matrix_rows(matrices["bus"], name="bus", columns=13)
    generators = _matrix_rows(matrices["gen"], name="gen", columns=21)
    # The pinned case stores MATPOWER's required branch columns through
    # ANGMAX (13 columns); optional OPF output columns are not present.
    branches = _matrix_rows(matrices["branch"], name="branch", columns=13)
    base_kvs = {row[9] for row in buses}
    if len(base_kvs) != 1:
        raise MatpowerParseError("mixed baseKV values require an explicit mapping decision")
    base_kv = next(iter(base_kvs))
    mapping = _source_mapping(source, base_mva, base_kv)

    bus_ids = [_integer(row[0], context="bus_i") for row in buses]
    if any(bus_id <= 0 for bus_id in bus_ids) or len(set(bus_ids)) != len(bus_ids):
        raise MatpowerParseError("bus IDs must be positive and unique")
    bus_by_id = set(bus_ids)
    bus_types = tuple(_integer(row[1], context="bus type") for row in buses)
    if sum(bus_type == 3 for bus_type in bus_types) != 1:
        raise MatpowerParseError("exactly one MATPOWER reference bus is required")
    if any(bus_type not in {1, 2, 3, 4} for bus_type in bus_types):
        raise MatpowerParseError("unsupported MATPOWER bus type")

    bus_vm_pu = FloatVector(row[7] for row in buses)
    bus_va_deg = FloatVector(row[8] for row in buses)
    bus_vmax_pu = FloatVector(row[11] for row in buses)
    bus_vmin_pu = FloatVector(row[12] for row in buses)
    p_generation = [0.0] * len(buses)
    q_generation = [0.0] * len(buses)
    for row in generators:
        gen_bus = _integer(row[0], context="gen bus")
        if gen_bus not in bus_by_id:
            raise MatpowerParseError("generator references a bus absent from the bus matrix")
        if _integer(row[7], context="gen status") not in {0, 1}:
            raise MatpowerParseError("generator status must be 0 or 1")
        if row[7] == 1:
            index = bus_ids.index(gen_bus)
            p_generation[index] += row[1]
            q_generation[index] += row[2]

    bus_entities = tuple(Bus(bus_id=bus_id, base_kv=FiniteFloat(row[9])) for bus_id, row in zip(bus_ids, buses))
    branch_entities: list[Branch] = []
    branch_status: list[int] = []
    branch_b_pu: list[float] = []
    edges: list[tuple[int, int]] = []
    seen_branch_ids: set[int] = set()
    for branch_id, row in enumerate(branches, 1):
        from_id = _integer(row[0], context="F_BUS")
        to_id = _integer(row[1], context="T_BUS")
        if from_id not in bus_by_id or to_id not in bus_by_id:
            raise MatpowerParseError("branch references a bus absent from the bus matrix")
        if from_id == to_id:
            raise MatpowerParseError("self-loop branch is forbidden")
        if branch_id in seen_branch_ids:
            raise MatpowerParseError("duplicate branch ID")
        seen_branch_ids.add(branch_id)
        status = _integer(row[10], context="BR_STATUS")
        if status != 1:
            raise MatpowerParseError("out-of-service branches require an explicit topology contract")
        branch_status.append(status)
        branch_b_pu.append(row[4])
        edges.append((from_id, to_id))
        branch_entities.append(
            Branch(
                branch_id=branch_id,
                from_bus=ObjectReference("ENT001", Identifier(str(from_id))),
                to_bus=ObjectReference("ENT001", Identifier(str(to_id))),
                r=FiniteFloat(row[2] / mapping.base_impedance_ohm.value),
                x=FiniteFloat(row[3] / mapping.base_impedance_ohm.value),
            )
        )
    if not _connected(bus_by_id, edges):
        raise MatpowerParseError("network graph is disconnected")
    if len(branch_entities) != len(bus_entities) - 1:
        raise MatpowerParseError("case141 static contract requires a radial branch count")

    raw_pd_mva = [row[2] / 1000.0 for row in buses]
    p_load = FloatVector(value * mapping.power_factor.value for value in raw_pd_mva)
    q_factor = math.sin(math.acos(mapping.power_factor.value))
    q_load = FloatVector(value * q_factor for value in raw_pd_mva)
    return ParsedMatpowerCase(
        network=NetworkModel(tuple(bus_entities), tuple(branch_entities), NonNegativeFloat(base_mva)),
        bus_types=bus_types,
        branch_status=tuple(branch_status),
        p_load_mw=p_load,
        q_load_mvar=q_load,
        bus_vm_pu=bus_vm_pu,
        bus_va_deg=bus_va_deg,
        bus_vmax_pu=bus_vmax_pu,
        bus_vmin_pu=bus_vmin_pu,
        p_generation_mw=FloatVector(p_generation),
        q_generation_mvar=FloatVector(q_generation),
        branch_b_pu=FloatVector(branch_b_pu),
        mapping=mapping,
        source_hash=Sha256(digest),
        source_path=str(source_path),
    )
