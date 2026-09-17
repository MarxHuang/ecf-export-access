"""Reusable diagnostic input context for the pinned IEEE 141 case.

This helper materializes the real 141-bus network, the frozen deterministic
30-participant mapping, the canonical 0.70 operating point, and an explicit
Q specification.  It deliberately stops before proxy construction, AC
finite-difference evaluation, optimization, or evidence promotion.  Capacity
values and the Q definition are caller inputs; no historical configuration is
read and no scientific threshold is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from r4r.errors import ValidationError
from r4r.models import OperatingPoint, model_hash
from r4r.network_parser import ParsedMatpowerCase, parse_matpower_case
from r4r.operating_point import build_canonical_operating_point
from r4r.participant_registry import (
    ParticipantSelection,
    build_participant_selection,
    select_deterministic_nonzero_load_buses,
)
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.serialization import canonical_hash
from r4r.types import IdentifierVector, QAssumption, Sha256


IEEE141_CASE_SHA256 = "613C313B92629160C5F250E28BD33B22DF316C81A8C6507F5958B3D23FE1C88E"


@dataclass(frozen=True, slots=True)
class IEEE141DiagnosticInputContext:
    """Hash-bound case inputs for diagnostic AC/proxy experiments."""

    parsed_case: ParsedMatpowerCase
    selection: ParticipantSelection
    operating_point: OperatingPoint
    reactive_specification: ReactiveInjectionSpecification
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "ieee141_diagnostic_input_context.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.parsed_case, ParsedMatpowerCase):
            raise ValidationError("parsed_case must be ParsedMatpowerCase")
        if not isinstance(self.selection, ParticipantSelection):
            raise ValidationError("selection must be ParticipantSelection")
        if not isinstance(self.operating_point, OperatingPoint):
            raise ValidationError("operating_point must be OperatingPoint")
        if not isinstance(self.reactive_specification, ReactiveInjectionSpecification):
            raise ValidationError("reactive_specification must be ReactiveInjectionSpecification")
        if self.operating_point.q_assumption is not self.reactive_specification.q_mode_id:
            raise ValidationError("operating point and reactive specification Q assumptions must match")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("IEEE 141 input context cannot be promoted")

    @property
    def network_hash(self) -> Sha256:
        return self.parsed_case.source_hash

    @property
    def participant_registry_hash(self) -> Sha256:
        return Sha256(self.selection.alignment_hash)

    @property
    def operating_point_hash(self) -> Sha256:
        return Sha256(model_hash(self.operating_point))

    @property
    def participant_ids(self) -> IdentifierVector:
        return self.selection.participant_set.participant_ids

    def to_json(self) -> dict[str, Any]:
        return {
            "network_hash": self.network_hash.to_json(),
            "source_path": self.parsed_case.source_path,
            "participant_ids": self.participant_ids.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "participants": [participant.to_json() for participant in self.selection.participants],
            "reporter_ids": self.selection.reporter_ids.to_json(),
            "bus_to_participant": [
                {"bus_id": bus_id, "participant_id": participant_id.to_json()}
                for bus_id, participant_id in self.selection.bus_to_participant
            ],
            "selection_status": self.selection.scientific_status,
            "selection_declared_complete": self.selection.declared_complete,
            "selection_ordering_rule": self.selection.ordering_rule,
            "co_location_policy": self.selection.co_location_policy,
            "operating_point": self.operating_point.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "reactive_specification": {
                "q_mode_id": self.reactive_specification.q_mode_id.value,
                "power_factor": self.reactive_specification.power_factor,
                "magnitude_rule_id": self.reactive_specification.magnitude_rule_id.to_json(),
                "q_sign_convention": self.reactive_specification.q_sign_convention,
                "positive_p_definition": self.reactive_specification.positive_p_definition,
                "positive_q_definition": self.reactive_specification.positive_q_definition,
                "capability_limit_policy": self.reactive_specification.capability_limit_policy.to_json(),
                "scientific_status": self.reactive_specification.scientific_status,
            },
            "status": self.status,
        }

    @property
    def context_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def build_ieee141_diagnostic_input_context(
    case_path: str | Path,
    *,
    rating_mw_by_bus: Mapping[int, float],
    availability_mw_by_bus: Mapping[int, float],
    reactive_specification: ReactiveInjectionSpecification,
    expected_sha256: str = IEEE141_CASE_SHA256,
    participant_count: int = 30,
) -> IEEE141DiagnosticInputContext:
    """Build explicit diagnostic inputs from the pinned MATPOWER source."""

    if not isinstance(reactive_specification, ReactiveInjectionSpecification):
        raise ValidationError("reactive_specification must be ReactiveInjectionSpecification")
    parsed = parse_matpower_case(case_path, expected_sha256=expected_sha256)
    bus_ids = tuple(bus.bus_id for bus in parsed.network.buses)
    load_by_bus = dict(zip(bus_ids, parsed.p_load_mw.values))
    branch_endpoints = tuple(
        (int(branch.from_bus.object_id.value), int(branch.to_bus.object_id.value))
        for branch in parsed.network.branches
    )
    slack_bus_id = next(bus_id for bus_id, bus_type in zip(bus_ids, parsed.bus_types) if bus_type == 3)
    selected = select_deterministic_nonzero_load_buses(
        network_bus_ids=bus_ids,
        slack_bus_id=slack_bus_id,
        load_mw_by_bus=load_by_bus,
        branch_endpoints=branch_endpoints,
        participant_count=participant_count,
    )
    selection = build_participant_selection(
        selected,
        network_bus_ids=bus_ids,
        slack_bus_id=slack_bus_id,
        load_mw_by_bus=load_by_bus,
        rating_mw_by_bus=rating_mw_by_bus,
        availability_mw_by_bus=availability_mw_by_bus,
        participant_count=participant_count,
        scientific_status="CANDIDATE",
        declared_complete="COMPLETE",
        ordering_rule="stable_participant_id_then_bus_id",
        co_location_policy="ONE_PARTICIPANT_PER_BUS",
    )
    operating_point = build_canonical_operating_point(parsed, q_assumption=reactive_specification.q_mode_id)
    return IEEE141DiagnosticInputContext(
        parsed_case=parsed,
        selection=selection,
        operating_point=operating_point,
        reactive_specification=reactive_specification,
    )


__all__ = [
    "IEEE141_CASE_SHA256",
    "IEEE141DiagnosticInputContext",
    "build_ieee141_diagnostic_input_context",
]
