"""Explicit request-to-allocation materialization for the interface slice.

This module deliberately does not solve an LP/QP or infer a feasible domain.
It binds a caller-supplied allocation to one admitted request and records the
request/capacity provenance needed by later proxy and AC diagnostics.  Any
upper-bound policy is an explicit argument; no raw capacity box is silently
treated as a network-feasible domain.
"""
from __future__ import annotations

import hashlib
from typing import Literal, Protocol, runtime_checkable

from dataclasses import dataclass

from r4r.errors import ValidationError
from r4r.models import AllocationVector
from r4r.request_contracts import AdmittedRequestVector
from r4r.serialization import canonical_bytes, canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, QAssumption, Sha256


AllocationBoundPolicy = Literal["NONE", "ADMITTED_REQUEST"]


@runtime_checkable
class AllocationBindingProtocol(Protocol):
    """Read-only provenance surface shared by caller and solver bindings."""

    participant_ids: IdentifierVector
    values_mw: FloatVector
    side: AllocationSide
    allocation_hash: Sha256
    request_hash: Sha256 | None
    capacity_hash: Sha256 | None
    participant_registry_hash: Sha256 | None
    capacity_vector_hash: Sha256 | None
    bound_policy: AllocationBoundPolicy
    scenario_id: Identifier


@runtime_checkable
class RuleAllocationBindingProtocol(Protocol):
    """Allocation binding carrying an explicit algebraic rule identity."""

    rule_id: Identifier


@runtime_checkable
class ObjectiveAllocationBindingProtocol(Protocol):
    """Allocation binding carrying an explicit LP/QP objective identity."""

    objective_id: Identifier


@runtime_checkable
class DomainAllocationBindingProtocol(Protocol):
    """Allocation binding exposing the exact feasible-domain identity."""

    domain_hash: Sha256 | None


@runtime_checkable
class DiagnosticProfileBindingProtocol(Protocol):
    """Typed method/provenance envelope required by AC profile audits.

    The allocation object itself remains the object consumed by the injection
    adapter.  This separate envelope prevents an audit from guessing method
    identity through ``getattr`` and makes the scenario/domain/request join
    explicit for every diagnostic profile.
    """

    allocation: AllocationBindingProtocol
    method_id: Identifier
    scenario_id: Identifier
    rule_id: Identifier | None
    objective_id: Identifier | None
    domain_hash: Sha256
    request_hash: Sha256 | None
    admitted_request_hash: Sha256 | None
    participant_ids: IdentifierVector
    values_mw: FloatVector
    side: AllocationSide
    allocation_hash: Sha256
    participant_registry_hash: Sha256 | None


@dataclass(frozen=True, slots=True)
class DiagnosticProfileBinding:
    """Explicit method binding for a complete diagnostic allocation profile.

    ``rule_id`` is used by algebraic allocations and ``objective_id`` by
    solver-produced LP/QP allocations.  Exactly one must be supplied for a
    method-backed profile; a future explicit/manual diagnostic path must use a
    registered rule identifier rather than silently omitting method identity.
    """

    allocation: AllocationBindingProtocol
    method_id: Identifier
    scenario_id: Identifier
    rule_id: Identifier | None
    objective_id: Identifier | None
    domain_hash: Sha256
    request_hash: Sha256 | None = None
    admitted_request_hash: Sha256 | None = None
    serialization_id = "diagnostic_profile_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.allocation, AllocationBindingProtocol):
            raise ValidationError("diagnostic profile requires an allocation binding")
        for name, value in (("method_id", self.method_id), ("scenario_id", self.scenario_id)):
            if not isinstance(value, Identifier):
                raise ValidationError(f"{name} must be an Identifier")
        for name, value in (("rule_id", self.rule_id), ("objective_id", self.objective_id)):
            if value is not None and not isinstance(value, Identifier):
                raise ValidationError(f"{name} must be an Identifier or null")
        if (self.rule_id is None) == (self.objective_id is None):
            raise ValidationError("diagnostic profile must bind exactly one rule_id or objective_id")
        if not isinstance(self.domain_hash, Sha256):
            raise ValidationError("diagnostic profile domain_hash must be SHA-256")
        if self.request_hash is not None and not isinstance(self.request_hash, Sha256):
            raise ValidationError("diagnostic profile request_hash must be SHA-256 or null")
        if self.admitted_request_hash is not None and not isinstance(self.admitted_request_hash, Sha256):
            raise ValidationError("diagnostic profile admitted_request_hash must be SHA-256 or null")
        if self.allocation.scenario_id != self.scenario_id:
            raise ValidationError("diagnostic profile scenario does not match allocation")
        if self.allocation.participant_registry_hash is None:
            raise ValidationError("diagnostic profile requires participant registry provenance")
        if self.request_hash != self.allocation.request_hash:
            raise ValidationError("diagnostic profile request hash does not match allocation")
        actual_rule_id = (
            self.allocation.rule_id
            if isinstance(self.allocation, RuleAllocationBindingProtocol)
            else None
        )
        actual_objective_id = (
            self.allocation.objective_id
            if isinstance(self.allocation, ObjectiveAllocationBindingProtocol)
            else None
        )
        if actual_rule_id is not None or actual_objective_id is not None:
            if (self.rule_id, self.objective_id) != (actual_rule_id, actual_objective_id):
                raise ValidationError("diagnostic profile rule/objective provenance does not match allocation")
        if isinstance(self.allocation, DomainAllocationBindingProtocol):
            actual_domain_hash = self.allocation.domain_hash
            if actual_domain_hash is None:
                raise ValidationError("diagnostic profile allocation lacks feasible-domain provenance")
            if actual_domain_hash != self.domain_hash:
                raise ValidationError("diagnostic profile domain does not match allocation")

    @property
    def participant_ids(self) -> IdentifierVector:
        return self.allocation.participant_ids

    @property
    def values_mw(self) -> FloatVector:
        return self.allocation.values_mw

    @property
    def side(self) -> AllocationSide:
        return self.allocation.side

    @property
    def allocation_hash(self) -> Sha256:
        return self.allocation.allocation_hash

    @property
    def participant_registry_hash(self) -> Sha256 | None:
        return self.allocation.participant_registry_hash

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "allocation": self.allocation.to_json(),
            "method_id": self.method_id.to_json(),
            "scenario_id": self.scenario_id.to_json(),
            "rule_id": self.rule_id.to_json() if self.rule_id else None,
            "objective_id": self.objective_id.to_json() if self.objective_id else None,
            "domain_hash": self.domain_hash.to_json(),
            "request_hash": self.request_hash.to_json() if self.request_hash else None,
            "admitted_request_hash": self.admitted_request_hash.to_json() if self.admitted_request_hash else None,
        }


@dataclass(frozen=True, slots=True)
class AlgebraicAllocationBinding:
    """Typed diagnostic binding from a fixed-regime algebraic result.

    This binding carries both the request-vector hash and the upstream
    ``RequestConstruction`` hash without pretending that a capacity box is a
    network-feasible domain.  It is intentionally diagnostic-only and uses
    the existing ``NONE`` bound policy.
    """

    participant_ids: IdentifierVector
    values_mw: FloatVector
    side: AllocationSide
    scenario_id: Identifier
    allocation_hash: Sha256
    request_hash: Sha256
    source_request_hash: Sha256
    domain_hash: Sha256
    rule_id: Identifier
    participant_registry_hash: Sha256
    capacity_hash: Sha256 | None = None
    capacity_vector_hash: Sha256 | None = None
    bound_policy: AllocationBoundPolicy = "NONE"
    serialization_id = "algebraic_allocation_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("algebraic binding participant_ids must be non-empty")
        if not isinstance(self.values_mw, FloatVector) or len(self.values_mw.values) != len(self.participant_ids.values):
            raise ValidationError("algebraic binding values must align with participant IDs")
        if any(value < 0.0 for value in self.values_mw.values):
            raise ValidationError("algebraic binding values must be nonnegative")
        if not isinstance(self.side, AllocationSide) or not isinstance(self.scenario_id, Identifier):
            raise ValidationError("algebraic binding side and scenario_id are required")
        if not isinstance(self.rule_id, Identifier) or not isinstance(self.domain_hash, Sha256):
            raise ValidationError("algebraic binding rule and domain provenance are required")
        for name, value in (
            ("allocation_hash", self.allocation_hash),
            ("request_hash", self.request_hash),
            ("source_request_hash", self.source_request_hash),
            ("participant_registry_hash", self.participant_registry_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"algebraic binding {name} must be SHA-256")
        if self.capacity_hash is not None or self.capacity_vector_hash is not None:
            raise ValidationError("algebraic binding cannot claim capacity provenance")
        if self.bound_policy != "NONE":
            raise ValidationError("algebraic diagnostic binding must use NONE bound policy")
        expected = Sha256(canonical_hash({
                    "serialization_id": self.serialization_id,
                    "participant_ids": self.participant_ids.to_json(),
                    "values_mw": self.values_mw.to_json(),
                    "side": self.side.value,
                    "scenario_id": self.scenario_id.to_json(),
                    "request_hash": self.request_hash.to_json(),
                    "source_request_hash": self.source_request_hash.to_json(),
                    "domain_hash": self.domain_hash.to_json(),
                    "rule_id": self.rule_id.to_json(),
                    "participant_registry_hash": self.participant_registry_hash.to_json(),
                    "bound_policy": self.bound_policy,
                }))
        if self.allocation_hash != expected:
            raise ValidationError("algebraic binding allocation_hash does not match provenance")

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "participant_ids": self.participant_ids.to_json(),
            "values_mw": self.values_mw.to_json(),
            "side": self.side.value,
            "scenario_id": self.scenario_id.to_json(),
            "allocation_hash": self.allocation_hash.to_json(),
            "request_hash": self.request_hash.to_json(),
            "source_request_hash": self.source_request_hash.to_json(),
            "domain_hash": self.domain_hash.to_json(),
            "rule_id": self.rule_id.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "capacity_hash": None,
            "capacity_vector_hash": None,
            "bound_policy": self.bound_policy,
        }


def _allocation_hash(
    *,
    participant_ids: IdentifierVector,
    allocation_mw: FloatVector,
    side: AllocationSide,
    scenario_id: Identifier,
    request_hash: Sha256 | None,
    capacity_hash: Sha256 | None,
    objective_id: Identifier | None,
    bound_policy: AllocationBoundPolicy,
    participant_registry_hash: Sha256 | None,
    capacity_vector_hash: Sha256 | None,
) -> Sha256:
    payload = {
        "participant_ids": participant_ids.to_json(),
        "values_mw": allocation_mw.to_json(),
        "side": side.value,
        "scenario_id": scenario_id.value,
        "request_hash": request_hash.to_json() if request_hash else None,
        "capacity_hash": capacity_hash.to_json() if capacity_hash else None,
        "objective_id": objective_id.to_json() if objective_id else None,
        "bound_policy": bound_policy,
        "participant_registry_hash": (
            participant_registry_hash.to_json() if participant_registry_hash else None
        ),
        "capacity_vector_hash": (
            capacity_vector_hash.to_json() if capacity_vector_hash else None
        ),
    }
    return Sha256(hashlib.sha256(canonical_bytes(payload)).hexdigest())


@dataclass(frozen=True, slots=True)
class ExplicitCapacityVector:
    """Typed caller-supplied capacity vector for diagnostic bindings."""

    participant_ids: IdentifierVector
    values_mw: FloatVector
    capacity_spec_hash: Sha256
    participant_registry_hash: Sha256
    # The v2 source-to-allocation contract is optional at the base entity
    # boundary so legacy diagnostic unit tests remain readable.  Any formal
    # DEV panel/source-grid path must populate all five fields and therefore
    # receives the stronger typed hash below.
    reporter_id: Identifier | None = None
    q_mode: QAssumption | None = None
    profile_hash: Sha256 | None = None
    source_grid_row_id: Identifier | None = None
    serialization_id = "explicit_capacity_vector.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector):
            raise ValidationError("capacity participant_ids must be an IdentifierVector")
        if not isinstance(self.values_mw, FloatVector):
            raise ValidationError("capacity values must be a finite FloatVector")
        if len(self.participant_ids.values) != len(self.values_mw.values):
            raise ValidationError("capacity vector must align with participant IDs")
        if any(value < 0.0 for value in self.values_mw.values):
            raise ValidationError("capacity values must be nonnegative")
        if not isinstance(self.capacity_spec_hash, Sha256) or not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("capacity provenance hashes must be SHA-256")
        metadata = (self.reporter_id, self.q_mode, self.profile_hash, self.source_grid_row_id)
        if any(value is not None for value in metadata):
            if not isinstance(self.reporter_id, Identifier):
                raise ValidationError("deep capacity binding requires reporter_id")
            if not isinstance(self.q_mode, QAssumption):
                raise ValidationError("deep capacity binding requires Q mode")
            if not isinstance(self.profile_hash, Sha256):
                raise ValidationError("deep capacity binding requires profile_hash")
            if not isinstance(self.source_grid_row_id, Identifier):
                raise ValidationError("deep capacity binding requires source_grid_row_id")

    @property
    def capacity_vector_hash(self) -> Sha256:
        if self.is_deep_bound:
            return Sha256(
                hashlib.sha256(
                    canonical_bytes(
                        {
                            "serialization_id": "explicit_capacity_vector.v2",
                            "capacity_spec_hash": self.capacity_spec_hash.to_json(),
                            "participant_registry_hash": self.participant_registry_hash.to_json(),
                            "participant_ids": self.participant_ids.to_json(),
                            "values_mw": self.values_mw.to_json(),
                            "reporter_id": self.reporter_id.to_json(),
                            "q_mode": self.q_mode.value,
                            "profile_hash": self.profile_hash.to_json(),
                            "source_grid_row_id": self.source_grid_row_id.to_json(),
                        }
                    )
                ).hexdigest()
            )
        return Sha256(
            hashlib.sha256(
                canonical_bytes(
                    {
                        "participant_ids": self.participant_ids.to_json(),
                        "values_mw": self.values_mw.to_json(),
                        "capacity_spec_hash": self.capacity_spec_hash.to_json(),
                        "participant_registry_hash": self.participant_registry_hash.to_json(),
                    }
                )
            ).hexdigest()
        )

    @property
    def is_deep_bound(self) -> bool:
        return all(
            value is not None
            for value in (self.reporter_id, self.q_mode, self.profile_hash, self.source_grid_row_id)
        )

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": "explicit_capacity_vector.v2" if self.is_deep_bound else self.serialization_id,
            "participant_ids": self.participant_ids.to_json(),
            "values_mw": self.values_mw.to_json(),
            "capacity_spec_hash": self.capacity_spec_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "capacity_vector_hash": self.capacity_vector_hash.to_json(),
            "reporter_id": self.reporter_id.to_json() if self.reporter_id else None,
            "q_mode": self.q_mode.value if self.q_mode else None,
            "profile_hash": self.profile_hash.to_json() if self.profile_hash else None,
            "source_grid_row_id": self.source_grid_row_id.to_json() if self.source_grid_row_id else None,
        }


@dataclass(frozen=True, slots=True)
class ExplicitAllocationBinding:
    """Allocation plus the explicit bound policy used to construct it.

    ``AllocationVector`` remains the registered entity.  This wrapper keeps
    the interface-slice policy visible without silently changing the frozen
    Round 1 entity registry; callers can still access the vector fields
    through the read-only properties below.
    """

    allocation: AllocationVector
    bound_policy: AllocationBoundPolicy
    participant_registry_hash: Sha256 | None = None
    capacity_vector_hash: Sha256 | None = None
    capacity_vector: ExplicitCapacityVector | None = None
    serialization_id = "explicit_allocation_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.allocation, AllocationVector):
            raise ValidationError("allocation must be an AllocationVector")
        if self.bound_policy not in {"NONE", "ADMITTED_REQUEST"}:
            raise ValidationError("allocation bound policy is not registered")
        if self.participant_registry_hash is not None and not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be SHA-256 or null")
        if self.capacity_vector_hash is not None and not isinstance(self.capacity_vector_hash, Sha256):
            raise ValidationError("capacity_vector_hash must be SHA-256 or null")
        if self.capacity_vector is not None and not isinstance(self.capacity_vector, ExplicitCapacityVector):
            raise ValidationError("capacity_vector must be an ExplicitCapacityVector or null")
        if self.allocation.objective_id is not None:
            raise ValidationError("caller-supplied allocation bindings cannot claim an optimization objective")
        claims_capacity_provenance = (
            self.bound_policy == "ADMITTED_REQUEST"
            or self.allocation.request_hash is not None
            or self.allocation.capacity_hash is not None
            or self.capacity_vector_hash is not None
        )
        # A registry hash alone is identity provenance, not a capacity or
        # request claim.  It is therefore valid for diagnostic profile
        # bindings with bound_policy=NONE and no capacity vector.
        if claims_capacity_provenance and self.capacity_vector is None:
            raise ValidationError("provenance-bound allocation requires the ExplicitCapacityVector object")
        if self.capacity_vector is not None:
            if self.capacity_vector.participant_ids != self.allocation.participant_ids:
                raise ValidationError("capacity vector participant order does not match allocation binding")
            if self.allocation.capacity_hash != self.capacity_vector.capacity_spec_hash:
                raise ValidationError("capacity vector specification does not match allocation binding")
            if self.participant_registry_hash != self.capacity_vector.participant_registry_hash:
                raise ValidationError("allocation binding registry provenance does not match capacity vector")
            if self.capacity_vector_hash != self.capacity_vector.capacity_vector_hash:
                raise ValidationError("allocation binding capacity provenance does not match capacity vector")
        if self.bound_policy == "ADMITTED_REQUEST":
            if self.participant_registry_hash is None:
                raise ValidationError("ADMITTED_REQUEST binding requires participant registry provenance")
            if self.capacity_vector_hash is None:
                raise ValidationError("ADMITTED_REQUEST binding requires capacity vector provenance")
            if self.allocation.request_hash is None:
                raise ValidationError("ADMITTED_REQUEST binding requires request provenance")
            if self.allocation.capacity_hash is None:
                raise ValidationError("ADMITTED_REQUEST binding requires capacity provenance")
        expected_hash = _allocation_hash(
            participant_ids=self.allocation.participant_ids,
            allocation_mw=self.allocation.values_mw,
            side=self.allocation.side,
            scenario_id=self.allocation.scenario_id,
            request_hash=self.allocation.request_hash,
            capacity_hash=self.allocation.capacity_hash,
            objective_id=self.allocation.objective_id,
            bound_policy=self.bound_policy,
            participant_registry_hash=self.participant_registry_hash,
            capacity_vector_hash=self.capacity_vector_hash,
        )
        if self.allocation.allocation_hash != expected_hash:
            raise ValidationError("allocation hash does not match complete binding provenance")

    @property
    def participant_ids(self) -> IdentifierVector:
        return self.allocation.participant_ids

    @property
    def values_mw(self) -> FloatVector:
        return self.allocation.values_mw

    @property
    def vector_length(self) -> int:
        return self.allocation.vector_length

    @property
    def side(self) -> AllocationSide:
        return self.allocation.side

    @property
    def scenario_id(self) -> Identifier:
        return self.allocation.scenario_id

    @property
    def allocation_hash(self) -> Sha256:
        return self.allocation.allocation_hash

    @property
    def request_hash(self) -> Sha256 | None:
        return self.allocation.request_hash

    @property
    def capacity_hash(self) -> Sha256 | None:
        return self.allocation.capacity_hash

    @property
    def objective_id(self) -> Identifier | None:
        return self.allocation.objective_id

    def to_json(self) -> dict[str, object]:
        return {
            "allocation": self.allocation.to_json(),
            "bound_policy": self.bound_policy,
            "participant_registry_hash": self.participant_registry_hash.to_json() if self.participant_registry_hash else None,
            "capacity_vector_hash": self.capacity_vector_hash.to_json() if self.capacity_vector_hash else None,
            "capacity_vector": self.capacity_vector.to_json() if self.capacity_vector else None,
        }


def build_explicit_allocation(
    *,
    participant_ids: IdentifierVector,
    allocation_mw: FloatVector,
    side: AllocationSide,
    scenario_id: Identifier,
    admitted_request: AdmittedRequestVector | None,
    capacity_hash: Sha256 | None,
    objective_id: Identifier | None = None,
    bound_policy: AllocationBoundPolicy,
    participant_registry_hash: Sha256 | None = None,
    capacity_vector: ExplicitCapacityVector | None = None,
) -> ExplicitAllocationBinding:
    """Bind an explicit allocation without invoking an optimizer.

    ``bound_policy`` is intentionally required so callers cannot accidentally
    turn a request/capacity box into a scientific feasibility claim.  The
    ``ADMITTED_REQUEST`` option checks the allocation against the supplied
    admitted vector; ``NONE`` records no upper-bound claim and is suitable for
    diagnostic construction before a problem specification is frozen.
    """
    if not isinstance(participant_ids, IdentifierVector):
        raise ValidationError("participant_ids must be an IdentifierVector")
    if not isinstance(allocation_mw, FloatVector):
        raise ValidationError("allocation_mw must be a finite FloatVector")
    if not isinstance(side, AllocationSide):
        raise ValidationError("allocation side must be registered")
    if not isinstance(scenario_id, Identifier):
        raise ValidationError("scenario_id must be an Identifier")
    if bound_policy not in {"NONE", "ADMITTED_REQUEST"}:
        raise ValidationError("allocation bound policy is not registered")
    if any(value < 0.0 for value in allocation_mw.values):
        raise ValidationError("allocation values must be nonnegative")

    if capacity_vector is not None:
        if not isinstance(capacity_vector, ExplicitCapacityVector):
            raise ValidationError("capacity_vector must be an ExplicitCapacityVector or null")
        if capacity_vector.participant_ids != participant_ids:
            raise ValidationError("capacity vector participant order does not match allocation")
        if participant_registry_hash is not None and participant_registry_hash != capacity_vector.participant_registry_hash:
            raise ValidationError("capacity vector registry hash does not match allocation")
        participant_registry_hash = capacity_vector.participant_registry_hash
        if capacity_hash is None:
            raise ValidationError("typed capacity vector requires an explicit capacity hash")
        if capacity_hash != capacity_vector.capacity_spec_hash:
            raise ValidationError("capacity hash does not match typed capacity vector")
    elif admitted_request is not None or capacity_hash is not None:
        raise ValidationError(
            "request-bound or capacity-bound allocations require an ExplicitCapacityVector"
        )

    request_hash: Sha256 | None = None
    if admitted_request is not None:
        if not isinstance(admitted_request, AdmittedRequestVector):
            raise ValidationError("admitted_request must be an AdmittedRequestVector or null")
        if admitted_request.source_side is not side:
            raise ValidationError("allocation side does not match admitted request side")
        if admitted_request.participant_ids != participant_ids:
            raise ValidationError("allocation participant order does not match admitted request")
        if capacity_hash is None:
            raise ValidationError("an admitted request requires an explicit capacity hash")
        if capacity_hash != admitted_request.capacity_spec_hash:
            raise ValidationError("capacity hash does not match the admitted request specification")
        request_registry_hash = admitted_request.participant_registry_hash
        if request_registry_hash is None:
            raise ValidationError("request-bound allocation requires request registry provenance")
        if participant_registry_hash is not None and participant_registry_hash != request_registry_hash:
            raise ValidationError("participant registry hash does not match the admitted request")
        participant_registry_hash = request_registry_hash
        request_hash = admitted_request.request_hash
    elif bound_policy == "ADMITTED_REQUEST":
        raise ValidationError("ADMITTED_REQUEST policy requires an admitted request")

    if bound_policy == "ADMITTED_REQUEST":
        assert admitted_request is not None
        if any(
            value > request
            for value, request in zip(
                allocation_mw.values, admitted_request.admitted_values_mw.values
            )
        ):
            raise ValidationError("allocation exceeds the explicit admitted request bound")

    if capacity_hash is not None and not isinstance(capacity_hash, Sha256):
        raise ValidationError("capacity_hash must be SHA-256 or null")
    if objective_id is not None and not isinstance(objective_id, Identifier):
        raise ValidationError("objective_id must be an Identifier or null")
    if objective_id is not None:
        raise ValidationError("caller-supplied allocations cannot claim an optimization objective")

    digest = _allocation_hash(
        participant_ids=participant_ids,
        allocation_mw=allocation_mw,
        side=side,
        scenario_id=scenario_id,
        request_hash=request_hash,
        capacity_hash=capacity_hash,
        objective_id=objective_id,
        bound_policy=bound_policy,
        participant_registry_hash=participant_registry_hash,
        capacity_vector_hash=(
            capacity_vector.capacity_vector_hash if capacity_vector is not None else None
        ),
    )
    bound_allocation = AllocationVector(
        participant_ids=participant_ids,
        values_mw=allocation_mw,
        vector_length=len(participant_ids.values),
        side=side,
        scenario_id=scenario_id,
        allocation_hash=digest,
        request_hash=request_hash,
        capacity_hash=capacity_hash,
        objective_id=objective_id,
    )
    return ExplicitAllocationBinding(
        bound_allocation,
        bound_policy,
        participant_registry_hash,
        capacity_vector.capacity_vector_hash if capacity_vector is not None else None,
        capacity_vector,
    )


__all__ = [
    "AllocationBoundPolicy",
    "AllocationBindingProtocol",
    "RuleAllocationBindingProtocol",
    "ObjectiveAllocationBindingProtocol",
    "DomainAllocationBindingProtocol",
    "DiagnosticProfileBindingProtocol",
    "DiagnosticProfileBinding",
    "AlgebraicAllocationBinding",
    "ExplicitCapacityVector",
    "ExplicitAllocationBinding",
    "build_explicit_allocation",
]
