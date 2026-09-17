"""Scenario and allocation data contracts, without allocation computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier, require_mapping, require_reference
from r4r.types import (
    AllocationSide,
    FloatVector,
    Identifier,
    IdentifierVector,
    ObjectReference,
    ObjectReferenceVector,
    PairStatus,
    RequestKind,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class CapacityScenario(ContractModel):
    capacity_model: Identifier
    parameters: dict[str, Any]
    capacity_mw: FloatVector
    serialization_id = "capacity_scenario.v1"

    def __post_init__(self) -> None:
        require_identifier(self.capacity_model, "capacity_model")
        require_mapping(self.parameters, "parameters")
        if not isinstance(self.capacity_mw, FloatVector):
            raise ValidationError("capacity_mw must be a finite FloatVector")
        if any(value < 0 for value in self.capacity_mw.values):
            raise ValidationError("capacity_mw must be nonnegative")


@dataclass(frozen=True, slots=True)
class RequestScenario(ContractModel):
    scenario_id: Identifier
    request_kind: RequestKind
    request_mw: FloatVector
    reporter_id: Identifier | None
    serialization_id = "request_scenario.v1"

    def __post_init__(self) -> None:
        require_identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.request_kind, RequestKind):
            raise ValidationError("request_kind must be registered")
        if not isinstance(self.request_mw, FloatVector):
            raise ValidationError("request_mw must be a finite FloatVector")
        if any(value < 0 for value in self.request_mw.values):
            raise ValidationError("request_mw must be nonnegative")
        if self.request_kind is RequestKind.REFERENCE and self.reporter_id is not None:
            raise ValidationError("reference request cannot carry a reporter ID")
        if self.request_kind is RequestKind.STRATEGIC_REPORT and self.reporter_id is None:
            raise ValidationError("strategic report requires a reporter ID")
        if self.reporter_id is not None:
            require_identifier(self.reporter_id, "reporter_id")


@dataclass(frozen=True, slots=True)
class ScenarioDefinition(ContractModel):
    scenario_id: Identifier
    operating_point_id: Identifier
    selection_hash: Sha256
    serialization_id = "scenario_definition.v1"

    def __post_init__(self) -> None:
        require_identifier(self.scenario_id, "scenario_id")
        require_identifier(self.operating_point_id, "operating_point_id")


@dataclass(frozen=True, slots=True)
class AllocationVector(ContractModel):
    participant_ids: IdentifierVector
    values_mw: FloatVector
    vector_length: int
    side: AllocationSide
    scenario_id: Identifier
    allocation_hash: Sha256
    request_hash: Sha256 | None = None
    capacity_hash: Sha256 | None = None
    objective_id: Identifier | None = None
    serialization_id = "allocation.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector):
            raise ValidationError("participant_ids must be an IdentifierVector")
        if not isinstance(self.values_mw, FloatVector):
            raise ValidationError("values_mw must be a finite FloatVector")
        if isinstance(self.vector_length, bool) or not isinstance(self.vector_length, int) or self.vector_length < 0:
            raise ValidationError("vector_length must be a nonnegative integer")
        if self.vector_length != len(self.values_mw.values) or self.vector_length != len(self.participant_ids.values):
            raise ValidationError("allocation vector length must match IDs and values")
        if any(value < 0 for value in self.values_mw.values):
            raise ValidationError("allocation values must be nonnegative")
        if not isinstance(self.side, AllocationSide):
            raise ValidationError("allocation side must be reference or reported")
        require_identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.allocation_hash, Sha256):
            raise ValidationError("allocation_hash must be a registered SHA-256 value")
        if self.request_hash is not None and not isinstance(self.request_hash, Sha256):
            raise ValidationError("request_hash must be SHA-256 or null")
        if self.capacity_hash is not None and not isinstance(self.capacity_hash, Sha256):
            raise ValidationError("capacity_hash must be SHA-256 or null")
        if self.objective_id is not None and not isinstance(self.objective_id, Identifier):
            raise ValidationError("objective_id must be an Identifier or null")


@dataclass(frozen=True, slots=True)
class AllocationPairResult(ContractModel):
    reference: ObjectReference
    reported: ObjectReference
    alignment_hash: Sha256
    signed_deviation_mw: FiniteFloat
    absolute_deviation_mw: FiniteFloat
    relative_deviation: FiniteFloat
    pair_status: PairStatus
    metric_references: ObjectReferenceVector
    serialization_id = "allocation_pair.v1"

    def __post_init__(self) -> None:
        require_reference(self.reference, "reference", "ENT015")
        require_reference(self.reported, "reported", "ENT015")
        for metric_reference in self.metric_references.values:
            require_reference(metric_reference, "metric_references", "ENT024")
        if not isinstance(self.pair_status, PairStatus):
            raise ValidationError("pair_status must be registered")
