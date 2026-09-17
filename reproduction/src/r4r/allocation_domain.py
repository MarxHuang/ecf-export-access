"""Explicit feasible-allocation domain contracts without an optimizer.

The domain is intentionally separate from capacity, request, proxy and solver
implementations.  It requires explicit participant bounds and at least one
constraint row, so a raw capacity box cannot silently become the network
feasible domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, Identifier, IdentifierVector, Sha256
from r4r.types.scalars import FiniteFloat


_STATUSES = {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}
_BOUND_SOURCES = {"CAPACITY", "ADMITTED_REQUEST", "MIN_CAPACITY_REQUEST", "EXPLICIT", "UNRESOLVED"}
_SENSES = {"LESS_EQUAL", "GREATER_EQUAL", "EQUAL"}
_FAMILIES = {
    "PARTICIPANT_LOWER_BOUND",
    "PARTICIPANT_UPPER_BOUND",
    "BRANCH_INCREMENT",
    "VOLTAGE_LOWER",
    "VOLTAGE_UPPER",
    "EXPLICIT_LINEAR",
}
_RESULT_STATUSES = {"VALID_FEASIBLE", "VALID_INFEASIBLE"}


@dataclass(frozen=True, slots=True)
class LinearConstraintRow(ContractModel):
    """One explicit linear row with its physical interpretation."""

    constraint_id: Identifier
    family: str
    coefficients: FloatVector
    rhs: FiniteFloat
    sense: str
    row_unit: str
    element_id: Identifier | None = None
    orientation: str | None = None
    baseline_quantity: str | None = None
    limit_source: str | None = None
    serialization_id = "allocation_constraint_row.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, Identifier):
            raise ValidationError("constraint_id must be Identifier")
        if self.family not in _FAMILIES:
            raise ValidationError("constraint family is not registered")
        if not isinstance(self.coefficients, FloatVector):
            raise ValidationError("coefficients must be FloatVector")
        if self.sense not in _SENSES:
            raise ValidationError("constraint sense is not registered")
        if not isinstance(self.row_unit, str) or not self.row_unit:
            raise ValidationError("row_unit is required")
        if self.element_id is not None and not isinstance(self.element_id, Identifier):
            raise ValidationError("element_id must be Identifier or null")

    def to_json(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id.to_json(),
            "family": self.family,
            "coefficients": self.coefficients.to_json(),
            "rhs": self.rhs.to_json(),
            "sense": self.sense,
            "row_unit": self.row_unit,
            "element_id": self.element_id.to_json() if self.element_id else None,
            "orientation": self.orientation,
            "baseline_quantity": self.baseline_quantity,
            "limit_source": self.limit_source,
        }


@dataclass(frozen=True, slots=True)
class AllocationDomainCheckResult(ContractModel):
    """A pure domain check; it does not solve or promote evidence."""

    domain_hash: Sha256
    participant_registry_hash: Sha256
    allocation_hash: Sha256
    status: str
    diagnostic_only: bool
    lower_slack_mw: FloatVector
    upper_slack_mw: FloatVector
    constraint_ids: IdentifierVector
    constraint_slack: FloatVector
    constraint_violation: FloatVector
    feasibility_tolerance: FiniteFloat
    violating_constraint_ids: IdentifierVector
    failure_reasons: IdentifierVector
    serialization_id = "allocation_domain_check.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.domain_hash, Sha256) or not isinstance(self.participant_registry_hash, Sha256) or not isinstance(self.allocation_hash, Sha256):
            raise ValidationError("domain and participant registry hashes must be Sha256")
        if self.status not in _RESULT_STATUSES:
            raise ValidationError("allocation domain result status is not registered")
        if not isinstance(self.diagnostic_only, bool):
            raise ValidationError("diagnostic_only must be boolean")
        for name in ("lower_slack_mw", "upper_slack_mw", "constraint_slack", "constraint_violation"):
            if not isinstance(getattr(self, name), FloatVector):
                raise ValidationError(f"{name} must be FloatVector")
        if not isinstance(self.constraint_ids, IdentifierVector):
            raise ValidationError("constraint_ids must be an IdentifierVector")
        if len(self.constraint_slack.values) != len(self.constraint_violation.values):
            raise ValidationError("constraint slack and violation vectors must align")
        if len(self.constraint_ids.values) != len(self.constraint_slack.values):
            raise ValidationError("constraint IDs and slack vectors must align")
        if not isinstance(self.feasibility_tolerance, FiniteFloat) or self.feasibility_tolerance.value < 0.0:
            raise ValidationError("feasibility_tolerance must be a nonnegative FiniteFloat")
        if not isinstance(self.violating_constraint_ids, IdentifierVector) or not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure reasons and violating IDs must be IdentifierVector values")

    def to_json(self) -> dict[str, Any]:
        return {
            "domain_hash": self.domain_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "allocation_hash": self.allocation_hash.to_json(),
            "status": self.status,
            "diagnostic_only": self.diagnostic_only,
            "lower_slack_mw": self.lower_slack_mw.to_json(),
            "upper_slack_mw": self.upper_slack_mw.to_json(),
            "constraint_ids": self.constraint_ids.to_json(),
            "constraint_slack": self.constraint_slack.to_json(),
            "constraint_violation": self.constraint_violation.to_json(),
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
            "violating_constraint_ids": self.violating_constraint_ids.to_json(),
            "failure_reasons": self.failure_reasons.to_json(),
        }


@dataclass(frozen=True, slots=True)
class FeasibleAllocationDomain(ContractModel):
    """Explicit bounds and network constraint rows for future LP/QP calls."""

    participant_ids: IdentifierVector
    lower_bounds_mw: FloatVector
    upper_bounds_mw: FloatVector
    upper_bound_source: str
    constraints: tuple[LinearConstraintRow, ...]
    participant_registry_hash: Sha256
    capacity_spec_hash: Sha256 | None
    request_hash: Sha256 | None
    proxy_spec_hash: Sha256 | None
    feasibility_tolerance: FiniteFloat
    scientific_status: str
    unresolved_fields: tuple[str, ...]
    serialization_id = "feasible_allocation_domain.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.lower_bounds_mw, FloatVector) or not isinstance(self.upper_bounds_mw, FloatVector):
            raise ValidationError("allocation bounds must be FloatVector values")
        n = len(self.participant_ids.values)
        if len(self.lower_bounds_mw.values) != n or len(self.upper_bounds_mw.values) != n:
            raise ValidationError("allocation bounds must match participant count")
        if any(value < 0.0 for value in self.lower_bounds_mw.values) or any(value < 0.0 for value in self.upper_bounds_mw.values):
            raise ValidationError("allocation bounds must be nonnegative")
        if any(lower > upper for lower, upper in zip(self.lower_bounds_mw.values, self.upper_bounds_mw.values)):
            raise ValidationError("lower bound cannot exceed upper bound")
        if self.upper_bound_source not in _BOUND_SOURCES:
            raise ValidationError("upper_bound_source is not registered")
        if not self.constraints:
            raise ValidationError("at least one explicit network constraint row is required")
        if any(not isinstance(row, LinearConstraintRow) for row in self.constraints):
            raise ValidationError("constraints must contain LinearConstraintRow values")
        if any(len(row.coefficients.values) != n for row in self.constraints):
            raise ValidationError("constraint coefficients must match participant count")
        if not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be Sha256")
        for name, value in (("capacity_spec_hash", self.capacity_spec_hash), ("request_hash", self.request_hash), ("proxy_spec_hash", self.proxy_spec_hash)):
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or null")
        if self.feasibility_tolerance.value < 0.0:
            raise ValidationError("feasibility_tolerance must be nonnegative")
        if self.scientific_status not in _STATUSES:
            raise ValidationError("scientific_status is not registered")
        if any(not isinstance(field, str) or not field for field in self.unresolved_fields):
            raise ValidationError("unresolved_fields must contain non-empty strings")

    @property
    def executable(self) -> bool:
        source_ready = {
            "CAPACITY": self.capacity_spec_hash is not None,
            "ADMITTED_REQUEST": self.request_hash is not None,
            "MIN_CAPACITY_REQUEST": self.capacity_spec_hash is not None and self.request_hash is not None,
            "EXPLICIT": True,
            "UNRESOLVED": False,
        }[self.upper_bound_source]
        return self.scientific_status == "AUTHOR_FROZEN" and not self.unresolved_fields and source_ready

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "lower_bounds_mw": self.lower_bounds_mw.to_json(),
            "upper_bounds_mw": self.upper_bounds_mw.to_json(),
            "upper_bound_source": self.upper_bound_source,
            "constraints": [row.to_json() for row in self.constraints],
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "capacity_spec_hash": self.capacity_spec_hash.to_json() if self.capacity_spec_hash else None,
            "request_hash": self.request_hash.to_json() if self.request_hash else None,
            "proxy_spec_hash": self.proxy_spec_hash.to_json() if self.proxy_spec_hash else None,
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
            "scientific_status": self.scientific_status,
            "unresolved_fields": list(self.unresolved_fields),
            "executable": self.executable,
        }

    @property
    def domain_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))

    def check(self, participant_ids: IdentifierVector, allocation_mw: FloatVector) -> AllocationDomainCheckResult:
        if participant_ids != self.participant_ids:
            raise ValidationError("allocation participant order does not match domain")
        if len(allocation_mw.values) != len(self.participant_ids.values):
            raise ValidationError("allocation length does not match domain")
        tol = self.feasibility_tolerance.value
        lower_slack = FloatVector(value - bound for value, bound in zip(allocation_mw.values, self.lower_bounds_mw.values))
        upper_slack = FloatVector(bound - value for value, bound in zip(allocation_mw.values, self.upper_bounds_mw.values))
        row_slacks: list[float] = []
        violations: list[float] = []
        violating_ids: list[Identifier] = []
        for row in self.constraints:
            dot = sum(coef * value for coef, value in zip(row.coefficients.values, allocation_mw.values))
            if row.sense == "LESS_EQUAL":
                slack = row.rhs.value - dot
            elif row.sense == "GREATER_EQUAL":
                slack = dot - row.rhs.value
            else:
                slack = -abs(dot - row.rhs.value)
            row_slacks.append(slack)
            violation = max(0.0, -slack)
            violations.append(violation)
            if violation > tol:
                violating_ids.append(row.constraint_id)
        bound_violation = any(value < -tol for value in lower_slack.values) or any(value < -tol for value in upper_slack.values)
        feasible = not bound_violation and not any(value > tol for value in violations)
        if bound_violation and not violating_ids:
            violating_ids.append(Identifier("PARTICIPANT_BOUND"))
        reasons = IdentifierVector([] if feasible else [Identifier("PREDICATE_FAIL")])
        allocation_hash = Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": allocation_mw.to_json()}))
        return AllocationDomainCheckResult(
            domain_hash=self.domain_hash,
            participant_registry_hash=self.participant_registry_hash,
            allocation_hash=allocation_hash,
            status="VALID_FEASIBLE" if feasible else "VALID_INFEASIBLE",
            diagnostic_only=not self.executable,
            lower_slack_mw=lower_slack,
            upper_slack_mw=upper_slack,
            constraint_ids=IdentifierVector(row.constraint_id for row in self.constraints),
            constraint_slack=FloatVector(row_slacks),
            constraint_violation=FloatVector(violations),
            feasibility_tolerance=self.feasibility_tolerance,
            violating_constraint_ids=IdentifierVector(violating_ids),
            failure_reasons=reasons,
        )
