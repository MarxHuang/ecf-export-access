"""Allocation-to-AC P/Q adaptation with explicit sign and Q semantics.

The adapter only performs vector alignment. It does not choose a participant
list, capacity model, power-factor model, or Q95 sign convention. Those are
scientific inputs supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from r4r.allocation_pipeline import AllocationBindingProtocol
from r4r.errors import ReferenceError, ValidationError
from r4r.models import OperatingPoint
from r4r.network_parser import ParsedMatpowerCase
from r4r.participant_registry import ParticipantSelection
from r4r.reactive_spec import ReactiveInjectionSpecification, build_participant_q
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, QAssumption, Sha256


def _reactive_spec_hash(specification: ReactiveInjectionSpecification) -> Sha256:
    return Sha256(
        canonical_hash(
            {
                "q_mode_id": specification.q_mode_id.value,
                "power_factor": specification.power_factor,
                "magnitude_rule_id": specification.magnitude_rule_id.value,
                "q_sign_convention": specification.q_sign_convention,
                "positive_p_definition": specification.positive_p_definition,
                "positive_q_definition": specification.positive_q_definition,
                "capability_limit_policy": specification.capability_limit_policy.value,
                "scientific_status": specification.scientific_status,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class NetInjectionVectors:
    """Complete bus-aligned net injections for the AC solver."""

    p_mw: FloatVector
    q_mvar: FloatVector
    q_assumption: QAssumption
    allocation_mw: FloatVector
    participant_q_mvar: FloatVector
    baseline_p_mw: FloatVector
    baseline_q_mvar: FloatVector
    delta_p_mw: FloatVector
    delta_q_mvar: FloatVector
    participant_registry_hash: Sha256
    allocation_hash: Sha256
    allocation_side: AllocationSide | None = None
    request_hash: Sha256 | None = None
    source_request_hash: Sha256 | None = None
    capacity_hash: Sha256 | None = None
    bound_policy: str | None = None
    capacity_vector_hash: Sha256 | None = None
    network_hash: Sha256 | None = None
    operating_point_hash: Sha256 | None = None
    q_spec_hash: Sha256 | None = None
    serialization_id = "net_injection_vectors.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.q_assumption, QAssumption):
            raise ValidationError("q_assumption must be a registered Q token")
        if len(self.p_mw.values) != len(self.q_mvar.values):
            raise ValidationError("P/Q vectors must have equal bus shape")
        if len(self.allocation_mw.values) != len(self.participant_q_mvar.values):
            raise ValidationError("allocation and participant-Q vectors must align")
        for name, vector in (
            ("baseline_p_mw", self.baseline_p_mw),
            ("baseline_q_mvar", self.baseline_q_mvar),
            ("delta_p_mw", self.delta_p_mw),
            ("delta_q_mvar", self.delta_q_mvar),
        ):
            if not isinstance(vector, FloatVector) or len(vector.values) != len(self.p_mw.values):
                raise ValidationError(f"{name} must be bus-aligned with p_mw")
        if not isinstance(self.participant_registry_hash, Sha256) or not isinstance(self.allocation_hash, Sha256):
            raise ValidationError("injection provenance hashes must be SHA-256")
        if self.allocation_side is not None and not isinstance(self.allocation_side, AllocationSide):
            raise ValidationError("allocation_side must be registered or null")
        if self.request_hash is not None and not isinstance(self.request_hash, Sha256):
            raise ValidationError("request_hash must be SHA-256 or null")
        if self.source_request_hash is not None and not isinstance(self.source_request_hash, Sha256):
            raise ValidationError("source_request_hash must be SHA-256 or null")
        if self.capacity_hash is not None and not isinstance(self.capacity_hash, Sha256):
            raise ValidationError("capacity_hash must be SHA-256 or null")
        if self.bound_policy is not None and self.bound_policy not in {"NONE", "ADMITTED_REQUEST"}:
            raise ValidationError("bound_policy is not registered")
        if self.capacity_vector_hash is not None and not isinstance(self.capacity_vector_hash, Sha256):
            raise ValidationError("capacity_vector_hash must be SHA-256 or null")
        for name, value in (
            ("network_hash", self.network_hash),
            ("operating_point_hash", self.operating_point_hash),
            ("q_spec_hash", self.q_spec_hash),
        ):
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be SHA-256 or null")


def build_net_injections(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    *,
    q_assumption: QAssumption,
    participant_q_mvar: Sequence[float] | None = None,
    allocation_hash: Sha256 | None = None,
    allocation_binding: AllocationBindingProtocol | None = None,
    operating_point_hash: Sha256 | None = None,
    q_spec_hash: Sha256 | None = None,
) -> NetInjectionVectors:
    """Apply an export allocation to native MATPOWER net injections.

    Net injection follows the AC solver convention (generation minus demand).
    Therefore an export allocation adds ``+x_i`` to active injection. Q is
    unchanged unless an explicit participant-Q vector is supplied. For Q0,
    participant Q must be omitted or identically zero; Q95 requires an
    explicit vector so its still-open scientific sign/model decision cannot be
    silently selected here.
    """
    return _build_net_injections_with_loads(
        parsed,
        selection,
        allocation_mw,
        q_assumption=q_assumption,
        participant_q_mvar=participant_q_mvar,
        p_load=parsed.p_load_mw,
        q_load=parsed.q_load_mvar,
        allocation_hash=allocation_hash,
        allocation_binding=allocation_binding,
        operating_point_hash=operating_point_hash,
        q_spec_hash=q_spec_hash,
    )


def _build_net_injections_with_loads(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    *,
    q_assumption: QAssumption,
    participant_q_mvar: Sequence[float] | None,
    p_load: FloatVector,
    q_load: FloatVector,
    allocation_hash: Sha256 | None = None,
    allocation_binding: AllocationBindingProtocol | None = None,
    operating_point_hash: Sha256 | None = None,
    q_spec_hash: Sha256 | None = None,
) -> NetInjectionVectors:
    """Build injections against an explicitly selected load operating point."""
    if not isinstance(q_assumption, QAssumption):
        raise ValidationError("q_assumption must be a registered Q token")
    n = len(parsed.network.buses)
    if len(p_load.values) != n or len(q_load.values) != n:
        raise ValidationError("operating-point load vectors do not match network bus count")
    if operating_point_hash is not None and not isinstance(operating_point_hash, Sha256):
        raise ValidationError("operating_point_hash must be SHA-256 or null")
    if q_spec_hash is not None and not isinstance(q_spec_hash, Sha256):
        raise ValidationError("q_spec_hash must be SHA-256 or null")
    parsed_bus_ids = tuple(bus.bus_id for bus in parsed.network.buses)
    if tuple(selection.network_bus_ids) != parsed_bus_ids:
        raise ReferenceError("participant registry network bus universe does not match parsed case")
    reference_indices = [index for index, bus_type in enumerate(parsed.bus_types) if bus_type == 3]
    if len(reference_indices) != 1:
        raise ValidationError("parsed MATPOWER case must contain exactly one reference bus")
    parsed_slack_bus_id = parsed_bus_ids[reference_indices[0]]
    if selection.slack_bus_id != parsed_slack_bus_id:
        raise ReferenceError("participant selection slack bus differs from MATPOWER reference bus")
    expected_network_hash = canonical_hash(
        {"network_bus_ids": list(parsed_bus_ids), "slack_bus_id": parsed_slack_bus_id}
    ).lower()
    if selection.network_hash != expected_network_hash:
        raise ValidationError("participant registry network hash does not match parsed case")
    if len(allocation_mw) != len(selection.participants):
        raise ValidationError("allocation length must match participant selection")
    allocation = FloatVector(allocation_mw)
    binding_side: AllocationSide | None = None
    binding_request_hash: Sha256 | None = None
    binding_source_request_hash: Sha256 | None = None
    binding_capacity_hash: Sha256 | None = None
    binding_policy: str | None = None
    binding_capacity_vector_hash: Sha256 | None = None
    if allocation_binding is not None:
        if not isinstance(allocation_binding, AllocationBindingProtocol):
            raise ValidationError("allocation_binding must be a registered allocation binding or null")
        if allocation_binding.participant_ids != selection.participant_set.participant_ids:
            raise ValidationError("allocation binding participant order does not match selection")
        if allocation_binding.values_mw != allocation:
            raise ValidationError("allocation binding values do not match allocation_mw")
        if allocation_hash is not None and allocation_hash != allocation_binding.allocation_hash:
            raise ValidationError("allocation_hash does not match allocation binding")
        if allocation_binding.participant_registry_hash is not None and allocation_binding.participant_registry_hash != Sha256(selection.alignment_hash):
            raise ValidationError("allocation binding registry hash does not match selection")
        allocation_hash_value = allocation_binding.allocation_hash
        binding_side = allocation_binding.side
        binding_request_hash = allocation_binding.request_hash
        binding_source_request_hash = getattr(allocation_binding, "source_request_hash", None)
        binding_capacity_hash = allocation_binding.capacity_hash
        binding_policy = allocation_binding.bound_policy
        binding_capacity_vector_hash = allocation_binding.capacity_vector_hash
    elif allocation_hash is None:
        allocation_hash_value = None
    else:
        allocation_hash_value = allocation_hash
    if any(value < 0 for value in allocation.values):
        raise ValidationError("export allocation must be nonnegative")
    for value, participant in zip(allocation.values, selection.participants):
        if value > participant.availability_mw.value:
            raise ValidationError(
                f"allocation exceeds availability for {participant.participant_id.value}"
            )

    if participant_q_mvar is None:
        q_participant = FloatVector([0.0] * len(selection.participants))
        if q_assumption is QAssumption.Q95:
            raise ValidationError("Q95 requires an explicit participant-Q vector")
    else:
        q_participant = FloatVector(participant_q_mvar)
        if len(q_participant.values) != len(selection.participants):
            raise ValidationError("participant-Q length must match participant selection")
        if q_assumption is QAssumption.Q0 and any(value != 0.0 for value in q_participant.values):
            raise ValidationError("Q0 forbids nonzero participant-Q injection")

    bus_index = {bus.bus_id: index for index, bus in enumerate(parsed.network.buses)}
    baseline_p = [generation - load for generation, load in zip(parsed.p_generation_mw.values, p_load.values)]
    baseline_q = [generation - load for generation, load in zip(parsed.q_generation_mvar.values, q_load.values)]
    p = list(baseline_p)
    q = list(baseline_q)
    for participant, p_export, q_export in zip(selection.participants, allocation.values, q_participant.values):
        bus_id = int(participant.bus_id.object_id.value)
        if bus_id not in bus_index:
            raise ReferenceError(f"participant bus {bus_id} is absent from parsed network")
        index = bus_index[bus_id]
        p[index] += p_export
        q[index] += q_export
    if allocation_hash_value is None:
        allocation_hash_value = Sha256(canonical_hash({"participant_ids": selection.participant_set.participant_ids.to_json(), "values_mw": allocation.to_json()}))
    elif not isinstance(allocation_hash_value, Sha256):
        raise ValidationError("allocation_hash must be SHA-256 or null")
    return NetInjectionVectors(
        p_mw=FloatVector(p),
        q_mvar=FloatVector(q),
        q_assumption=q_assumption,
        allocation_mw=allocation,
        participant_q_mvar=q_participant,
        baseline_p_mw=FloatVector(baseline_p),
        baseline_q_mvar=FloatVector(baseline_q),
        delta_p_mw=FloatVector(a - b for a, b in zip(p, baseline_p)),
        delta_q_mvar=FloatVector(a - b for a, b in zip(q, baseline_q)),
        participant_registry_hash=Sha256(selection.alignment_hash),
        allocation_hash=allocation_hash_value,
        allocation_side=binding_side,
        request_hash=binding_request_hash,
        source_request_hash=binding_source_request_hash,
        capacity_hash=binding_capacity_hash,
        bound_policy=binding_policy,
        capacity_vector_hash=binding_capacity_vector_hash,
        network_hash=Sha256(selection.network_hash),
        operating_point_hash=operating_point_hash,
        q_spec_hash=q_spec_hash,
    )


def build_net_injections_from_operating_point(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    operating_point: OperatingPoint,
    *,
    q_assumption: QAssumption,
    participant_q_mvar: Sequence[float] | None = None,
    allocation_hash: Sha256 | None = None,
    allocation_binding: AllocationBindingProtocol | None = None,
    q_spec_hash: Sha256 | None = None,
) -> NetInjectionVectors:
    """Build injections from a caller-supplied, hashed operating point.

    This is the explicit path for AC diagnostics.  It never falls back to
    native MATPOWER load vectors, so a canonical load scale cannot be lost by
    an implicit default.
    """
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("operating_point must be an OperatingPoint")
    if operating_point.q_assumption is not q_assumption:
        raise ValidationError("operating-point and requested Q assumptions differ")
    return _build_net_injections_with_loads(
        parsed,
        selection,
        allocation_mw,
        q_assumption=q_assumption,
        participant_q_mvar=participant_q_mvar,
        p_load=operating_point.p_load,
        q_load=operating_point.q_load,
        allocation_hash=allocation_hash,
        allocation_binding=allocation_binding,
        operating_point_hash=Sha256(operating_point.object_hash()),
        q_spec_hash=q_spec_hash,
    )


def build_net_injections_from_operating_point_spec(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    operating_point: OperatingPoint,
    specification: ReactiveInjectionSpecification,
    *,
    formal_execution: bool = True,
    allocation_hash: Sha256 | None = None,
    allocation_binding: AllocationBindingProtocol | None = None,
    q_spec_hash: Sha256 | None = None,
) -> NetInjectionVectors:
    """Apply a reactive specification on an explicit operating point."""
    computed_q_spec_hash = _reactive_spec_hash(specification)
    if q_spec_hash is not None:
        if not isinstance(q_spec_hash, Sha256):
            raise ValidationError("q_spec_hash must be SHA-256 or null")
        if q_spec_hash != computed_q_spec_hash:
            raise ValidationError("q_spec_hash does not match the reactive specification")
    participant_q = build_participant_q(
        allocation_mw, specification, formal_execution=formal_execution
    )
    return build_net_injections_from_operating_point(
        parsed,
        selection,
        allocation_mw,
        operating_point,
        q_assumption=specification.q_mode_id,
        participant_q_mvar=participant_q.values,
        allocation_hash=allocation_hash,
        allocation_binding=allocation_binding,
        q_spec_hash=computed_q_spec_hash,
    )


def build_net_injections_from_spec(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation_mw: Sequence[float],
    specification: ReactiveInjectionSpecification,
    *,
    formal_execution: bool = True,
    allocation_hash: Sha256 | None = None,
    allocation_binding: AllocationBindingProtocol | None = None,
    q_spec_hash: Sha256 | None = None,
) -> NetInjectionVectors:
    """Apply an explicit Q specification, preserving its fail-closed status."""
    computed_q_spec_hash = _reactive_spec_hash(specification)
    if q_spec_hash is not None:
        if not isinstance(q_spec_hash, Sha256):
            raise ValidationError("q_spec_hash must be SHA-256 or null")
        if q_spec_hash != computed_q_spec_hash:
            raise ValidationError("q_spec_hash does not match the reactive specification")
    participant_q = build_participant_q(
        allocation_mw, specification, formal_execution=formal_execution
    )
    return build_net_injections(
        parsed,
        selection,
        allocation_mw,
        q_assumption=specification.q_mode_id,
        participant_q_mvar=participant_q.values,
        allocation_hash=allocation_hash,
        allocation_binding=allocation_binding,
        q_spec_hash=computed_q_spec_hash,
    )
