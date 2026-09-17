"""Explicit capacity-vector binding without inventing a capacity model."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from r4r.errors import ContractError, UnfrozenScientificDefinitionError, ValidationError
from r4r.models import CapacityScenario
from r4r.participant_registry import ParticipantSelection
from r4r.serialization import canonical_bytes, canonical_hash
from r4r.types import FloatVector, Identifier, IdentifierVector, Sha256


_STATUSES = {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}
_LOAD_BASES = {"NATIVE_CASE_LOAD", "OPERATING_POINT_LOAD", "EXTERNAL_LOAD_VECTOR"}


@dataclass(frozen=True, slots=True)
class CapacityBinding:
    """A participant-aligned capacity input, not a network-feasible domain."""

    capacity_model: Identifier
    participant_ids: IdentifierVector
    capacity_mw: FloatVector
    binding_hash: str
    capacity_spec_id: Identifier | None = None
    capacity_spec_hash: Sha256 | None = None
    participant_registry_hash: str | None = None
    operating_point_id: Identifier | None = None
    load_basis: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capacity_model, Identifier):
            raise ValidationError("capacity_model must be an Identifier")
        if not isinstance(self.participant_ids, IdentifierVector):
            raise ValidationError("participant_ids must be an IdentifierVector")
        if not isinstance(self.capacity_mw, FloatVector):
            raise ValidationError("capacity_mw must be a finite FloatVector")
        if len(self.participant_ids.values) != len(self.capacity_mw.values):
            raise ValidationError("capacity and participant vectors must align")
        if any(value < 0 for value in self.capacity_mw.values):
            raise ValidationError("capacity values must be nonnegative")
        if len(self.binding_hash) != 64 or any(c not in "0123456789abcdef" for c in self.binding_hash):
            raise ValidationError("binding_hash must be a lowercase SHA-256 digest")
        if self.capacity_spec_id is not None and not isinstance(self.capacity_spec_id, Identifier):
            raise ValidationError("capacity_spec_id must be an Identifier")
        if self.capacity_spec_hash is not None and not isinstance(self.capacity_spec_hash, Sha256):
            raise ValidationError("capacity_spec_hash must be SHA-256")
        if self.participant_registry_hash is not None and (
            len(self.participant_registry_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.participant_registry_hash)
        ):
            raise ValidationError("participant_registry_hash must be lowercase SHA-256")
        if self.operating_point_id is not None and not isinstance(self.operating_point_id, Identifier):
            raise ValidationError("operating_point_id must be an Identifier")
        if self.load_basis is not None and self.load_basis not in _LOAD_BASES:
            raise ValidationError("load_basis is not registered")

    def to_scenario(self) -> CapacityScenario:
        """Expose the binding through the existing typed scenario interface."""
        return CapacityScenario(
            self.capacity_model,
            {
                "participant_ids": self.participant_ids,
                "binding_hash": self.binding_hash,
                "capacity_spec_id": self.capacity_spec_id,
                "capacity_spec_hash": self.capacity_spec_hash,
                "participant_registry_hash": self.participant_registry_hash,
                "operating_point_id": self.operating_point_id,
                "load_basis": self.load_basis,
            },
            self.capacity_mw,
        )


@dataclass(frozen=True, slots=True)
class CapacitySpecification:
    """Scientific metadata for a capacity model, including unresolved state."""

    specification_id: Identifier
    model_family: Identifier
    equation_id: Identifier | None
    scientific_status: str
    participant_registry_hash: str
    operating_point_id: Identifier
    load_basis: str
    load_scale_reference: str
    parameters: Mapping[str, Any]
    zone_assignment_id: Identifier | None = None
    zone_assignment_hash: Sha256 | None = None
    requires_complete_participant_registry: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.specification_id, Identifier) or not isinstance(self.model_family, Identifier):
            raise ValidationError("capacity specification IDs must be Identifiers")
        if self.equation_id is not None and not isinstance(self.equation_id, Identifier):
            raise ValidationError("equation_id must be an Identifier or null")
        if self.scientific_status not in _STATUSES:
            raise ValidationError("scientific_status is not registered")
        if len(self.participant_registry_hash) != 64 or any(c not in "0123456789abcdef" for c in self.participant_registry_hash):
            raise ValidationError("participant_registry_hash must be lowercase SHA-256")
        if not isinstance(self.operating_point_id, Identifier):
            raise ValidationError("operating_point_id must be an Identifier")
        if self.load_basis not in _LOAD_BASES:
            raise ValidationError("load_basis is not registered")
        if not isinstance(self.load_scale_reference, str) or not self.load_scale_reference:
            raise ValidationError("load_scale_reference is required")
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("parameters must be a mapping")
        if self.zone_assignment_id is not None and not isinstance(self.zone_assignment_id, Identifier):
            raise ValidationError("zone_assignment_id must be an Identifier or null")
        if self.zone_assignment_hash is not None and not isinstance(self.zone_assignment_hash, Sha256):
            raise ValidationError("zone_assignment_hash must be SHA-256 or null")
        if not isinstance(self.requires_complete_participant_registry, bool):
            raise ValidationError("requires_complete_participant_registry must be boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "model_family": self.model_family,
            "equation_id": self.equation_id,
            "scientific_status": self.scientific_status,
            "participant_registry_hash": self.participant_registry_hash,
            "operating_point_id": self.operating_point_id,
            "load_basis": self.load_basis,
            "load_scale_reference": self.load_scale_reference,
            "parameters": dict(self.parameters),
            "zone_assignment_id": self.zone_assignment_id,
            "zone_assignment_hash": self.zone_assignment_hash,
            "requires_complete_participant_registry": self.requires_complete_participant_registry,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


class ExplicitCapacityModel:
    """Neutral model that returns only a caller-supplied capacity vector."""

    def evaluate(
        self,
        selection: ParticipantSelection,
        specification: CapacitySpecification,
        capacity_mw: Sequence[float],
        *,
        formal_execution: bool = True,
    ) -> CapacityBinding:
        if specification.model_family.value != "EXPLICIT":
            if specification.scientific_status != "AUTHOR_FROZEN":
                raise UnfrozenScientificDefinitionError(
                    f"capacity model {specification.model_family.value} is not frozen"
                )
            raise ContractError(
                f"no registered implementation for capacity model {specification.model_family.value}"
            )
        if formal_execution and specification.scientific_status != "AUTHOR_FROZEN":
            raise UnfrozenScientificDefinitionError(
                "formal capacity evaluation requires AUTHOR_FROZEN specification"
            )
        if specification.participant_registry_hash != selection.alignment_hash:
            raise ValidationError("capacity specification and participant registry hashes differ")
        if specification.requires_complete_participant_registry and (
            selection.scientific_status != "AUTHOR_FROZEN" or selection.declared_complete != "COMPLETE"
        ):
            raise UnfrozenScientificDefinitionError(
                "capacity evaluation requires a complete AUTHOR_FROZEN participant registry"
            )
        binding = bind_capacity_vector(selection, specification.model_family, capacity_mw)
        return CapacityBinding(
            binding.capacity_model,
            binding.participant_ids,
            binding.capacity_mw,
            binding.binding_hash,
            capacity_spec_id=specification.specification_id,
            capacity_spec_hash=specification.specification_hash,
            participant_registry_hash=selection.alignment_hash,
            operating_point_id=specification.operating_point_id,
            load_basis=specification.load_basis,
        )


def bind_capacity_vector(
    selection: ParticipantSelection,
    capacity_model: Identifier,
    capacity_mw: Sequence[float],
) -> CapacityBinding:
    """Bind explicit capacities to the already-frozen participant order.

    No load scaling, ranking, clipping, zone inference, or network-feasibility
    claim is made. Those choices belong to a later, separately registered
    capacity model and proxy domain.
    """
    if not isinstance(capacity_model, Identifier):
        raise ValidationError("capacity_model must be an Identifier")
    if len(capacity_mw) != len(selection.participants):
        raise ValidationError("capacity vector length must match participant selection")
    values = FloatVector(capacity_mw)
    for value, participant in zip(values.values, selection.participants):
        if value > participant.availability_mw.value or value > participant.rating_mw.value:
            raise ValidationError(
                f"capacity exceeds participant bound for {participant.participant_id.value}"
            )
    ids = selection.participant_set.participant_ids
    payload = {
        "capacity_model": capacity_model.value,
        "participant_ids": ids.to_json(),
        "capacity_mw": values.to_json(),
        "participant_alignment_hash": selection.alignment_hash,
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return CapacityBinding(capacity_model, ids, values, digest)
