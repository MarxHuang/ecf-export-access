"""Typed active-set diagnostics for an explicit feasible allocation domain.

The result explains whether an allocation is request/capacity bound, network
active, interior, or a degenerate box corner.  It never changes an allocation
and never substitutes for AC feasibility or an evidence gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, Identifier, IdentifierVector, Sha256
from r4r.types.scalars import FiniteFloat


_NETWORK_FAMILIES = {
    "BRANCH_INCREMENT",
    "VOLTAGE_LOWER",
    "VOLTAGE_UPPER",
    "EXPLICIT_LINEAR",
}
_BOUND_STATUSES = {"LOWER_ACTIVE", "UPPER_ACTIVE", "BOTH_ACTIVE", "INTERIOR"}
_REGIMES = {
    "REQUEST_BOUND",
    "CAPACITY_BOUND",
    "REQUEST_CAPACITY_TIE",
    "NETWORK_ACTIVE",
    "MIXED",
    "INTERIOR",
    "DEGENERATE_BOX_CORNER",
}


@dataclass(frozen=True, slots=True)
class DiagnosticAllocationActivityParticipant(ContractModel):
    participant_id: Identifier
    allocation_mw: FiniteFloat
    lower_slack_mw: FiniteFloat
    upper_slack_mw: FiniteFloat
    upper_bound_source: str
    allocation_bound_status: str
    serialization_id = "diagnostic_allocation_activity_participant.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, Identifier):
            raise ValidationError("activity participant_id must be Identifier")
        for name, value in (
            ("allocation_mw", self.allocation_mw),
            ("lower_slack_mw", self.lower_slack_mw),
            ("upper_slack_mw", self.upper_slack_mw),
        ):
            if not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat")
        if self.upper_bound_source not in {"REQUEST_BOUND", "CAPACITY_BOUND", "REQUEST_CAPACITY_TIE", "EXPLICIT_BOUND"}:
            raise ValidationError("activity upper_bound_source is not registered")
        if self.allocation_bound_status not in _BOUND_STATUSES:
            raise ValidationError("activity allocation_bound_status is not registered")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "participant_id": self.participant_id.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "lower_slack_mw": self.lower_slack_mw.to_json(),
            "upper_slack_mw": self.upper_slack_mw.to_json(),
            "upper_bound_source": self.upper_bound_source,
            "allocation_bound_status": self.allocation_bound_status,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticAllocationActivityConstraint(ContractModel):
    constraint_id: Identifier
    constraint_family: str
    orientation: str | None
    row_unit: str
    lhs: FiniteFloat
    rhs: FiniteFloat
    slack: FiniteFloat
    normalized_utilization: FiniteFloat
    active: bool
    violating: bool
    serialization_id = "diagnostic_allocation_activity_constraint.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, Identifier):
            raise ValidationError("activity constraint_id must be Identifier")
        if not isinstance(self.constraint_family, str) or not self.constraint_family:
            raise ValidationError("activity constraint_family is required")
        if not isinstance(self.row_unit, str) or not self.row_unit:
            raise ValidationError("activity row_unit is required")
        for name, value in (
            ("lhs", self.lhs),
            ("rhs", self.rhs),
            ("slack", self.slack),
            ("normalized_utilization", self.normalized_utilization),
        ):
            if not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat")
        if not isinstance(self.active, bool) or not isinstance(self.violating, bool):
            raise ValidationError("activity constraint flags must be boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "constraint_id": self.constraint_id.to_json(),
            "constraint_family": self.constraint_family,
            "orientation": self.orientation,
            "row_unit": self.row_unit,
            "lhs": self.lhs.to_json(),
            "rhs": self.rhs.to_json(),
            "slack": self.slack.to_json(),
            "normalized_utilization": self.normalized_utilization.to_json(),
            "active": self.active,
            "violating": self.violating,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticAllocationActivityResult(ContractModel):
    domain_hash: Sha256
    participant_registry_hash: Sha256
    allocation_hash: Sha256
    participants: tuple[DiagnosticAllocationActivityParticipant, ...]
    constraints: tuple[DiagnosticAllocationActivityConstraint, ...]
    active_constraint_ids: IdentifierVector
    network_active_constraint_ids: IdentifierVector
    request_bound_participant_count: int
    capacity_bound_participant_count: int
    request_capacity_tie_count: int
    network_active_constraint_count: int
    minimum_network_slack: FiniteFloat
    maximum_network_utilization: FiniteFloat
    allocation_regime: str
    feasibility_tolerance: FiniteFloat
    source_allocation_hash: Sha256 | None = None
    diagnostic_only: bool = True
    serialization_id = "diagnostic_allocation_activity_result.v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("domain_hash", self.domain_hash),
            ("participant_registry_hash", self.participant_registry_hash),
            ("allocation_hash", self.allocation_hash),
            ("source_allocation_hash", self.source_allocation_hash),
        ):
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or null")
        if not self.participants or any(not isinstance(item, DiagnosticAllocationActivityParticipant) for item in self.participants):
            raise ValidationError("activity result requires typed participants")
        if not isinstance(self.constraints, tuple) or not self.constraints or any(not isinstance(item, DiagnosticAllocationActivityConstraint) for item in self.constraints):
            raise ValidationError("activity result requires typed constraints")
        if not isinstance(self.active_constraint_ids, IdentifierVector) or not isinstance(self.network_active_constraint_ids, IdentifierVector):
            raise ValidationError("activity constraint IDs must be typed")
        if self.network_active_constraint_count != len(self.network_active_constraint_ids.values):
            raise ValidationError("network_active_constraint_count is not bound to IDs")
        for name, value in (
            ("request_bound_participant_count", self.request_bound_participant_count),
            ("capacity_bound_participant_count", self.capacity_bound_participant_count),
            ("request_capacity_tie_count", self.request_capacity_tie_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a nonnegative integer")
        if not isinstance(self.minimum_network_slack, FiniteFloat) or not isinstance(self.maximum_network_utilization, FiniteFloat):
            raise ValidationError("activity aggregate scalars must be FiniteFloat")
        if self.allocation_regime not in _REGIMES:
            raise ValidationError("activity allocation_regime is not registered")
        if not isinstance(self.feasibility_tolerance, FiniteFloat) or self.feasibility_tolerance.value < 0.0:
            raise ValidationError("activity feasibility tolerance must be nonnegative")
        if self.diagnostic_only is not True:
            raise ValidationError("allocation activity cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "domain_hash": self.domain_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "allocation_hash": self.allocation_hash.to_json(),
            "source_allocation_hash": self.source_allocation_hash.to_json() if self.source_allocation_hash else None,
            "participants": [item.to_json() for item in self.participants],
            "constraints": [item.to_json() for item in self.constraints],
            "active_constraint_ids": self.active_constraint_ids.to_json(),
            "network_active_constraint_ids": self.network_active_constraint_ids.to_json(),
            "request_bound_participant_count": self.request_bound_participant_count,
            "capacity_bound_participant_count": self.capacity_bound_participant_count,
            "request_capacity_tie_count": self.request_capacity_tie_count,
            "network_active_constraint_count": self.network_active_constraint_count,
            "minimum_network_slack": self.minimum_network_slack.to_json(),
            "maximum_network_utilization": self.maximum_network_utilization.to_json(),
            "allocation_regime": self.allocation_regime,
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def activity_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def _bound_source_label(domain: FeasibleAllocationDomain) -> str:
    return {
        "ADMITTED_REQUEST": "REQUEST_BOUND",
        "CAPACITY": "CAPACITY_BOUND",
        "MIN_CAPACITY_REQUEST": "REQUEST_CAPACITY_TIE",
    }.get(domain.upper_bound_source, "EXPLICIT_BOUND")


def build_diagnostic_allocation_activity(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    allocation_mw: FloatVector,
    *,
    allocation_hash: Sha256 | None = None,
    source_allocation_hash: Sha256 | None = None,
) -> DiagnosticAllocationActivityResult:
    """Materialize active-set evidence from the exact domain check."""

    if not isinstance(domain, FeasibleAllocationDomain):
        raise ValidationError("activity builder requires FeasibleAllocationDomain")
    if not isinstance(participant_ids, IdentifierVector) or participant_ids != domain.participant_ids:
        raise ValidationError("activity participant order does not match domain")
    if not isinstance(allocation_mw, FloatVector):
        raise ValidationError("activity allocation must be FloatVector")
    checked = domain.check(participant_ids, allocation_mw)
    tol = max(domain.feasibility_tolerance.value, 1e-12)
    if allocation_hash is not None and not isinstance(allocation_hash, Sha256):
        raise ValidationError("activity allocation_hash must be Sha256")
    if allocation_hash is not None and allocation_hash != checked.allocation_hash:
        raise ValidationError("activity allocation_hash is not the hash of the checked allocation")
    if source_allocation_hash is not None and not isinstance(source_allocation_hash, Sha256):
        raise ValidationError("activity source_allocation_hash must be Sha256")
    domain_allocation_hash = checked.allocation_hash

    bound_source = _bound_source_label(domain)
    participants: list[DiagnosticAllocationActivityParticipant] = []
    for participant_id, value, lower_slack, upper_slack in zip(
        participant_ids.values,
        allocation_mw.values,
        checked.lower_slack_mw.values,
        checked.upper_slack_mw.values,
    ):
        lower_active = abs(lower_slack) <= tol
        upper_active = abs(upper_slack) <= tol
        bound_status = "BOTH_ACTIVE" if lower_active and upper_active else "LOWER_ACTIVE" if lower_active else "UPPER_ACTIVE" if upper_active else "INTERIOR"
        participants.append(
            DiagnosticAllocationActivityParticipant(
                participant_id=participant_id,
                allocation_mw=FiniteFloat(value),
                lower_slack_mw=FiniteFloat(lower_slack),
                upper_slack_mw=FiniteFloat(upper_slack),
                upper_bound_source=bound_source,
                allocation_bound_status=bound_status,
            )
        )

    constraint_rows: list[DiagnosticAllocationActivityConstraint] = []
    for row, slack, violation in zip(domain.constraints, checked.constraint_slack.values, checked.constraint_violation.values):
        lhs = sum(coef * value for coef, value in zip(row.coefficients.values, allocation_mw.values))
        scale = max(abs(row.rhs.value), tol)
        utilization = 1.0 - slack / scale
        constraint_rows.append(
            DiagnosticAllocationActivityConstraint(
                constraint_id=row.constraint_id,
                constraint_family=row.family,
                orientation=row.orientation,
                row_unit=row.row_unit,
                lhs=FiniteFloat(lhs),
                rhs=row.rhs,
                slack=FiniteFloat(slack),
                normalized_utilization=FiniteFloat(utilization),
                active=abs(slack) <= tol,
                violating=violation > tol,
            )
        )

    active_ids = IdentifierVector(item.constraint_id for item in constraint_rows if item.active)
    network_active_ids = IdentifierVector(
        item.constraint_id
        for item in constraint_rows
        if item.active and item.constraint_family in _NETWORK_FAMILIES
    )
    upper_active = [
        item for item in participants
        if item.allocation_bound_status in {"UPPER_ACTIVE", "BOTH_ACTIVE"}
    ]
    request_count = len(upper_active) if bound_source == "REQUEST_BOUND" else 0
    capacity_count = len(upper_active) if bound_source == "CAPACITY_BOUND" else 0
    tie_count = len(upper_active) if bound_source == "REQUEST_CAPACITY_TIE" else 0
    any_box_active = any(item.allocation_bound_status != "INTERIOR" for item in participants)
    all_box_active = all(item.allocation_bound_status != "INTERIOR" for item in participants)
    if network_active_ids.values and any_box_active:
        regime = "MIXED"
    elif network_active_ids.values:
        regime = "NETWORK_ACTIVE"
    elif all_box_active:
        regime = "DEGENERATE_BOX_CORNER"
    elif tie_count:
        regime = "REQUEST_CAPACITY_TIE"
    elif request_count:
        regime = "REQUEST_BOUND"
    elif capacity_count:
        regime = "CAPACITY_BOUND"
    else:
        regime = "INTERIOR"
    network_slacks = [item.slack.value for item in constraint_rows if item.constraint_family in _NETWORK_FAMILIES]
    utilizations = [item.normalized_utilization.value for item in constraint_rows if item.constraint_family in _NETWORK_FAMILIES]
    return DiagnosticAllocationActivityResult(
        domain_hash=domain.domain_hash,
        participant_registry_hash=domain.participant_registry_hash,
        allocation_hash=domain_allocation_hash,
        participants=tuple(participants),
        constraints=tuple(constraint_rows),
        active_constraint_ids=active_ids,
        network_active_constraint_ids=network_active_ids,
        request_bound_participant_count=request_count,
        capacity_bound_participant_count=capacity_count,
        request_capacity_tie_count=tie_count,
        network_active_constraint_count=len(network_active_ids.values),
        minimum_network_slack=FiniteFloat(min(network_slacks) if network_slacks else 0.0),
        maximum_network_utilization=FiniteFloat(max(utilizations) if utilizations else 0.0),
        allocation_regime=regime,
        feasibility_tolerance=domain.feasibility_tolerance,
        source_allocation_hash=source_allocation_hash,
    )


__all__ = [
    "DiagnosticAllocationActivityParticipant",
    "DiagnosticAllocationActivityConstraint",
    "DiagnosticAllocationActivityResult",
    "build_diagnostic_allocation_activity",
]
