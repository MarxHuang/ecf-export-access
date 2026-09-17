"""Participant entities and stable participant ordering."""
from __future__ import annotations

from dataclasses import dataclass

from r4r.errors import ReferenceError, ValidationError
from r4r.models.base import ContractModel
from r4r.types import Identifier, IdentifierVector, NonNegativeFloat, ObjectReference


@dataclass(frozen=True, slots=True)
class Participant(ContractModel):
    participant_id: Identifier
    bus_id: ObjectReference
    rating_mw: NonNegativeFloat
    availability_mw: NonNegativeFloat
    serialization_id = "participant.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, Identifier):
            raise ValidationError("participant_id must be Identifier")
        if not isinstance(self.bus_id, ObjectReference) or self.bus_id.entity_id != "ENT001":
            raise ReferenceError("participant bus_id must reference ENT001 Bus")


@dataclass(frozen=True, slots=True)
class ParticipantSet(ContractModel):
    participant_ids: IdentifierVector
    canonical_order: IdentifierVector
    serialization_id = "participant_set.v1"

    def __post_init__(self) -> None:
        if tuple(self.participant_ids.values) != tuple(self.canonical_order.values):
            raise ValidationError("participant IDs and canonical order must match exactly")
