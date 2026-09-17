"""Deterministic algebraic allocation rules over an explicit feasible domain.

The manuscript's proportional, equal-kW and flat/level rules are fixed-regime
rules.  This module evaluates those rules only when the supplied domain is
monotone for the corresponding scalar search: all network rows must be upper
inequalities with nonnegative coefficients, and the participant lower bounds
must be zero.  It never replaces the network domain with a raw request or
capacity box and never infers a missing screen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from r4r.allocation_domain import AllocationDomainCheckResult, FeasibleAllocationDomain
from r4r.allocation_pipeline import AlgebraicAllocationBinding
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.mitigation import MitigatedRequestVector
from r4r.request_generation import RequestConstruction
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, Sha256
from r4r.types.scalars import FiniteFloat


_RULES = {"WEIGHTED_PROPORTIONAL", "EQUAL_KW_REDUCTION", "FLAT_LEVEL"}
_STATUSES = {"VALID_FEASIBLE", "NO_FEASIBLE_ALLOCATION"}
_SCALAR_NAMES = {
    "WEIGHTED_PROPORTIONAL": "rho",
    "EQUAL_KW_REDUCTION": "lambda",
    "FLAT_LEVEL": "mu",
}
_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class BoundedRequestVector:
    """Effective request after an explicit feasible-domain upper bound.

    The vector is distinct from the raw RequestConstruction values, while its
    source hash preserves the request construction that produced the bound.
    When produced by the mitigation adapter, the optional fields make the
    capacity/domain bounding operation a first-class typed join rather than an
    inline ``min`` in a runner.  The two-argument constructor remains valid for
    the generic algebraic tests and older diagnostic callers.
    """

    values_mw: FloatVector
    source_request_hash: Sha256
    participant_ids: IdentifierVector | None = None
    filtered_request_hash: Sha256 | None = None
    admitted_request_hash: Sha256 | None = None
    capacity_spec_hash: Sha256 | None = None
    domain_hash: Sha256 | None = None
    bound_mask: tuple[bool, ...] = ()
    bound_amount_mw: FloatVector | None = None
    diagnostic_only: bool = True
    serialization_id = "bounded_request_vector.v2"

    def __post_init__(self) -> None:
        if not isinstance(self.values_mw, FloatVector) or not isinstance(self.source_request_hash, Sha256):
            raise ValidationError("bounded request requires typed values and source hash")
        hashes = (
            self.filtered_request_hash,
            self.admitted_request_hash,
            self.capacity_spec_hash,
            self.domain_hash,
        )
        if any(value is not None and not isinstance(value, Sha256) for value in hashes):
            raise ValidationError("bounded request provenance hashes must be Sha256 or null")
        if self.participant_ids is not None:
            if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
                raise ValidationError("bounded request participant IDs must be a non-empty IdentifierVector")
            if len(self.values_mw.values) != len(self.participant_ids.values):
                raise ValidationError("bounded request values must align with participant IDs")
            if not all(value is not None for value in hashes):
                raise ValidationError("typed bounded requests require complete request/capacity/domain provenance")
            if len(self.bound_mask) != len(self.values_mw.values):
                raise ValidationError("bounded request mask must align with participant IDs")
            if self.bound_amount_mw is None or len(self.bound_amount_mw.values) != len(self.values_mw.values):
                raise ValidationError("typed bounded requests require aligned bound amounts")
            if any(value < 0.0 for value in self.bound_amount_mw.values):
                raise ValidationError("bounded request amounts must be nonnegative")
            if any(not isinstance(value, bool) for value in self.bound_mask):
                raise ValidationError("bounded request mask must contain booleans")
            for mask, amount in zip(self.bound_mask, self.bound_amount_mw.values):
                if mask != (amount > _TOLERANCE):
                    raise ValidationError("bounded request mask must match positive bound amounts")
        elif any(value is not None for value in hashes) or self.bound_mask or self.bound_amount_mw is not None:
            raise ValidationError("bounded provenance fields require participant IDs")
        if self.diagnostic_only is not True:
            raise ValidationError("bounded requests remain diagnostic-only")

    @property
    def request_hash(self) -> Sha256:
        """Hash of the effective values in the registered participant order."""

        if self.participant_ids is None:
            return Sha256(canonical_hash({"values_mw": self.values_mw.to_json()}))
        return _request_hash(self.participant_ids, self.values_mw)

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "participant_ids": self.participant_ids.to_json() if self.participant_ids else None,
            "values_mw": self.values_mw.to_json(),
            "source_request_hash": self.source_request_hash.to_json(),
            "filtered_request_hash": self.filtered_request_hash.to_json() if self.filtered_request_hash else None,
            "admitted_request_hash": self.admitted_request_hash.to_json() if self.admitted_request_hash else None,
            "capacity_spec_hash": self.capacity_spec_hash.to_json() if self.capacity_spec_hash else None,
            "domain_hash": self.domain_hash.to_json() if self.domain_hash else None,
            "bound_mask": list(self.bound_mask),
            "bound_amount_mw": self.bound_amount_mw.to_json() if self.bound_amount_mw else None,
            "request_hash": self.request_hash.to_json(),
            "diagnostic_only": self.diagnostic_only,
        }


@dataclass(frozen=True, slots=True)
class AlgebraicAllocationResult(ContractModel):
    """One rule output and its explicit domain validation evidence."""

    rule_id: Identifier
    participant_ids: IdentifierVector
    request_mw: FloatVector
    allocation_mw: FloatVector
    scalar_name: str
    scalar_value: FiniteFloat
    domain_hash: Sha256
    request_hash: Sha256
    domain_check: AllocationDomainCheckResult
    participant_registry_hash: Sha256
    active_constraint_ids: IdentifierVector
    status: str
    diagnostic_only: bool = True
    source_request_hash: Sha256 | None = None
    serialization_id = "algebraic_allocation_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, Identifier):
            raise ValidationError("rule_id must be an Identifier")
        if self.rule_id.value not in _RULES:
            raise ValidationError("rule_id is not registered")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        for name, value in (("request_mw", self.request_mw), ("allocation_mw", self.allocation_mw)):
            if not isinstance(value, FloatVector) or len(value.values) != len(self.participant_ids.values):
                raise ValidationError(f"{name} must align with participant_ids")
            if any(not math.isfinite(float(item)) or item < 0.0 for item in value.values):
                raise ValidationError(f"{name} must contain finite nonnegative values")
        if self.scalar_name != _SCALAR_NAMES[self.rule_id.value]:
            raise ValidationError("scalar_name does not match rule_id")
        if not isinstance(self.scalar_value, FiniteFloat) or self.scalar_value.value < 0.0:
            raise ValidationError("scalar_value must be a nonnegative FiniteFloat")
        for name, value in (("domain_hash", self.domain_hash), ("request_hash", self.request_hash)):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be Sha256")
        if self.source_request_hash is not None and not isinstance(self.source_request_hash, Sha256):
            raise ValidationError("source_request_hash must be Sha256 or null")
        if self.request_hash != _request_hash(self.participant_ids, self.request_mw):
            raise ValidationError("request_hash does not match request_mw")
        expected_allocation = _candidate(self.rule_id.value, self.request_mw, self.scalar_value.value)
        if expected_allocation != self.allocation_mw:
            raise ValidationError("allocation_mw does not satisfy the algebraic rule equation")
        if not isinstance(self.domain_check, AllocationDomainCheckResult):
            raise ValidationError("domain_check must be AllocationDomainCheckResult")
        if self.domain_check.domain_hash != self.domain_hash:
            raise ValidationError("domain check does not reference domain_hash")
        if self.domain_check.participant_registry_hash != self.participant_registry_hash:
            raise ValidationError("domain check does not reference participant_registry_hash")
        if self.domain_check.allocation_hash != _allocation_hash(self.participant_ids, self.allocation_mw):
            raise ValidationError("domain check does not reference allocation_mw")
        if self.status not in _STATUSES:
            raise ValidationError("algebraic allocation status is not registered")
        expected_status = "VALID_FEASIBLE" if self.domain_check.status == "VALID_FEASIBLE" else "NO_FEASIBLE_ALLOCATION"
        if self.status != expected_status:
            raise ValidationError("algebraic status does not match domain check")
        if not isinstance(self.active_constraint_ids, IdentifierVector):
            raise ValidationError("active_constraint_ids must be IdentifierVector")
        active_tolerance = max(self.domain_check.feasibility_tolerance.value, _TOLERANCE)
        expected_active = IdentifierVector(
            constraint_id
            for constraint_id, slack in zip(
                self.domain_check.constraint_ids.values,
                self.domain_check.constraint_slack.values,
            )
            if abs(slack) <= active_tolerance
        )
        if self.active_constraint_ids != expected_active:
            raise ValidationError("active_constraint_ids do not match domain slack values")
        if self.diagnostic_only is not True:
            raise ValidationError("algebraic results remain diagnostic-only")

    def to_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id.to_json(),
            "participant_ids": self.participant_ids.to_json(),
            "request_mw": self.request_mw.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "scalar_name": self.scalar_name,
            "scalar_value": self.scalar_value.to_json(),
            "domain_hash": self.domain_hash.to_json(),
            "request_hash": self.request_hash.to_json(),
            "domain_check": self.domain_check.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "active_constraint_ids": self.active_constraint_ids.to_json(),
            "status": self.status,
            "diagnostic_only": self.diagnostic_only,
            "source_request_hash": self.source_request_hash.to_json() if self.source_request_hash else None,
        }

    @property
    def result_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def _allocation_hash(participant_ids: IdentifierVector, values: FloatVector) -> Sha256:
    return Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": values.to_json()}))


def _request_hash(participant_ids: IdentifierVector, request: FloatVector) -> Sha256:
    return Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": request.to_json()}))


def _coerce_request(
    request_mw: Sequence[float] | FloatVector | RequestConstruction | BoundedRequestVector | MitigatedRequestVector,
    source_request_hash: Sha256 | None,
) -> tuple[FloatVector, Sha256 | None]:
    if isinstance(request_mw, RequestConstruction):
        inferred_hash = request_mw.construction_hash
        if source_request_hash is not None and source_request_hash != inferred_hash:
            raise ValidationError("source_request_hash does not match RequestConstruction")
        return request_mw.actual_request_mw, inferred_hash
    if isinstance(request_mw, BoundedRequestVector):
        if source_request_hash is not None and source_request_hash != request_mw.source_request_hash:
            raise ValidationError("source_request_hash does not match BoundedRequestVector")
        return request_mw.values_mw, request_mw.source_request_hash
    if isinstance(request_mw, MitigatedRequestVector):
        if source_request_hash is not None and source_request_hash != request_mw.raw_request_hash:
            raise ValidationError("source_request_hash does not match MitigatedRequestVector")
        return request_mw.filtered_values_mw, request_mw.raw_request_hash
    if isinstance(request_mw, FloatVector):
        if source_request_hash is not None:
            raise ValidationError(
                "source_request_hash requires a typed RequestConstruction; raw vectors cannot claim construction provenance"
            )
        return request_mw, source_request_hash
    if source_request_hash is not None:
        raise ValidationError(
            "source_request_hash requires a typed RequestConstruction; raw vectors cannot claim construction provenance"
        )
    return FloatVector(request_mw), source_request_hash


def _validate_inputs(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    request_mw: Sequence[float] | FloatVector | RequestConstruction | MitigatedRequestVector,
    source_request_hash: Sha256 | None,
) -> tuple[FloatVector, Sha256 | None]:
    if not isinstance(domain, FeasibleAllocationDomain):
        raise ValidationError("domain must be FeasibleAllocationDomain")
    if participant_ids != domain.participant_ids:
        raise ValidationError("participant order does not match feasible domain")
    request, source_hash = _coerce_request(request_mw, source_request_hash)
    if len(request.values) != len(participant_ids.values):
        raise ValidationError("request vector must match participant IDs")
    if any(value < 0.0 for value in request.values):
        raise ValidationError("request vector must be nonnegative")
    if any(lower != 0.0 for lower in domain.lower_bounds_mw.values):
        raise ValidationError("scalar algebraic rules require zero participant lower bounds")
    if any(request_value > upper for request_value, upper in zip(request.values, domain.upper_bounds_mw.values)):
        raise ValidationError("request exceeds explicit participant upper bound; scalar rule would change its semantics")
    if not domain.constraints:
        raise ValidationError("scalar algebraic rules require explicit network constraints")
    for row in domain.constraints:
        if row.sense != "LESS_EQUAL":
            raise ValidationError("scalar algebraic rules require upper-inequality network rows")
        if not any(coefficient > 0.0 for coefficient in row.coefficients.values):
            raise ValidationError("scalar algebraic rules require every network row to be effective; found an ineffective network row")
        if any(coefficient < 0.0 for coefficient in row.coefficients.values):
            raise ValidationError("scalar algebraic rules require nonnegative network coefficients")
        if row.rhs.value < 0.0:
            raise ValidationError("scalar algebraic rules require nonnegative network right-hand sides")
    return request, source_hash


def _active_constraints(domain: FeasibleAllocationDomain, check: AllocationDomainCheckResult) -> IdentifierVector:
    tol = max(domain.feasibility_tolerance.value, _TOLERANCE)
    return IdentifierVector(
        constraint_id
        for constraint_id, slack in zip(check.constraint_ids.values, check.constraint_slack.values)
        if abs(slack) <= tol
    )


def _result(
    rule: str,
    participant_ids: IdentifierVector,
    request: FloatVector,
    allocation: FloatVector,
    scalar: float,
    domain: FeasibleAllocationDomain,
    source_request_hash: Sha256 | None,
) -> AlgebraicAllocationResult:
    check = domain.check(participant_ids, allocation)
    return AlgebraicAllocationResult(
        rule_id=Identifier(rule),
        participant_ids=participant_ids,
        request_mw=request,
        allocation_mw=allocation,
        scalar_name=_SCALAR_NAMES[rule],
        scalar_value=FiniteFloat(scalar),
        domain_hash=domain.domain_hash,
        request_hash=_request_hash(participant_ids, request),
        domain_check=check,
        participant_registry_hash=domain.participant_registry_hash,
        active_constraint_ids=_active_constraints(domain, check),
        status="VALID_FEASIBLE" if check.status == "VALID_FEASIBLE" else "NO_FEASIBLE_ALLOCATION",
        source_request_hash=source_request_hash,
    )


def bind_algebraic_allocation(
    result: AlgebraicAllocationResult,
    *,
    side: AllocationSide,
    scenario_id: Identifier,
    participant_registry_hash: Sha256 | None = None,
) -> AlgebraicAllocationBinding:
    """Materialize one typed algebraic result for the Q/P injection adapter.

    The source construction hash is mandatory here: a raw request vector may
    be used for an isolated rule diagnostic, but it cannot enter the typed
    request-to-AC chain as if its construction provenance were known.
    """

    if not isinstance(result, AlgebraicAllocationResult):
        raise ValidationError("result must be an AlgebraicAllocationResult")
    if result.source_request_hash is None:
        raise ValidationError("typed AC binding requires source_request_hash")
    if not isinstance(side, AllocationSide) or not isinstance(scenario_id, Identifier):
        raise ValidationError("side and scenario_id are required for algebraic binding")
    if participant_registry_hash is not None and not isinstance(participant_registry_hash, Sha256):
        raise ValidationError("participant_registry_hash must be Sha256 or null")
    if participant_registry_hash is not None and participant_registry_hash != result.participant_registry_hash:
        raise ValidationError("participant_registry_hash does not match the allocation domain")
    participant_registry_hash = result.participant_registry_hash
    payload = {
        "serialization_id": "algebraic_allocation_binding.v1",
        "participant_ids": result.participant_ids.to_json(),
        "values_mw": result.allocation_mw.to_json(),
        "side": side.value,
        "scenario_id": scenario_id.to_json(),
        "request_hash": result.request_hash.to_json(),
        "source_request_hash": result.source_request_hash.to_json(),
        "domain_hash": result.domain_hash.to_json(),
        "rule_id": result.rule_id.to_json(),
        "participant_registry_hash": participant_registry_hash.to_json(),
        "bound_policy": "NONE",
    }
    return AlgebraicAllocationBinding(
        participant_ids=result.participant_ids,
        values_mw=result.allocation_mw,
        side=side,
        scenario_id=scenario_id,
        allocation_hash=Sha256(canonical_hash(payload)),
        request_hash=result.request_hash,
        source_request_hash=result.source_request_hash,
        domain_hash=result.domain_hash,
        rule_id=result.rule_id,
        participant_registry_hash=participant_registry_hash,
    )


def _candidate(rule: str, request: FloatVector, scalar: float) -> FloatVector:
    if rule == "WEIGHTED_PROPORTIONAL":
        values = (scalar * value for value in request.values)
    elif rule == "EQUAL_KW_REDUCTION":
        values = (max(0.0, value - scalar) for value in request.values)
    elif rule == "FLAT_LEVEL":
        values = (min(value, scalar) for value in request.values)
    else:  # pragma: no cover - guarded by public dispatcher
        raise ValidationError("rule is not registered")
    return FloatVector(values)


def _monotone_scalar_search(rule: str, domain: FeasibleAllocationDomain, participant_ids: IdentifierVector, request: FloatVector) -> tuple[float, FloatVector]:
    """Find the maximum feasible scalar output for a monotone fixed regime."""
    high = max(request.values, default=0.0) if rule != "WEIGHTED_PROPORTIONAL" else 1.0
    low = 0.0
    high_candidate = _candidate(rule, request, high)
    low_candidate = _candidate(rule, request, low)
    high_feasible = domain.check(participant_ids, high_candidate).status == "VALID_FEASIBLE"
    low_feasible = domain.check(participant_ids, low_candidate).status == "VALID_FEASIBLE"
    if rule != "EQUAL_KW_REDUCTION" and high_feasible:
        return high, high_candidate
    if rule == "EQUAL_KW_REDUCTION":
        if low_feasible:
            return low, low_candidate
        if not high_feasible:
            return high, high_candidate
    elif not low_feasible:
        return low, low_candidate
    # For equal reduction, feasibility improves as lambda increases; for
    # flat-level and proportional rules it improves as the scalar decreases.
    if rule == "EQUAL_KW_REDUCTION":
        lo, hi = low, high
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if domain.check(participant_ids, _candidate(rule, request, mid)).status == "VALID_FEASIBLE":
                hi = mid
            else:
                lo = mid
        scalar = hi
    else:
        lo, hi = low, high
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if domain.check(participant_ids, _candidate(rule, request, mid)).status == "VALID_FEASIBLE":
                lo = mid
            else:
                hi = mid
        scalar = lo
    return scalar, _candidate(rule, request, scalar)


def allocate_algebraic(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    request_mw: Sequence[float] | FloatVector | RequestConstruction | MitigatedRequestVector,
    *,
    rule: str,
    source_request_hash: Sha256 | None = None,
) -> AlgebraicAllocationResult:
    """Evaluate one registered algebraic rule on an explicit network domain.

    ``WEIGHTED_PROPORTIONAL`` computes the common scale directly from every
    nonnegative upper row.  ``EQUAL_KW_REDUCTION`` and ``FLAT_LEVEL`` use a
    deterministic 80-iteration scalar search over the same rows.  A returned
    ``NO_FEASIBLE_ALLOCATION`` record retains the candidate and the failed
    domain check; it is never silently clipped or relabeled as feasible.
    """
    if rule not in _RULES:
        raise ValidationError("rule is not registered")
    request, source_hash = _validate_inputs(domain, participant_ids, request_mw, source_request_hash)
    if rule == "WEIGHTED_PROPORTIONAL":
        scalar = 1.0
        for row in domain.constraints:
            denominator = math.fsum(coefficient * value for coefficient, value in zip(row.coefficients.values, request.values))
            if denominator > 0.0:
                scalar = min(scalar, max(0.0, row.rhs.value / denominator))
        allocation = _candidate(rule, request, scalar)
    else:
        scalar, allocation = _monotone_scalar_search(rule, domain, participant_ids, request)
    return _result(rule, participant_ids, request, allocation, scalar, domain, source_hash)


def allocate_weighted_proportional(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    request_mw: Sequence[float] | FloatVector | RequestConstruction | MitigatedRequestVector,
    *,
    source_request_hash: Sha256 | None = None,
) -> AlgebraicAllocationResult:
    return allocate_algebraic(
        domain, participant_ids, request_mw,
        rule="WEIGHTED_PROPORTIONAL",
        source_request_hash=source_request_hash,
    )


def allocate_equal_kw_reduction(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    request_mw: Sequence[float] | FloatVector | RequestConstruction | MitigatedRequestVector,
    *,
    source_request_hash: Sha256 | None = None,
) -> AlgebraicAllocationResult:
    return allocate_algebraic(
        domain, participant_ids, request_mw,
        rule="EQUAL_KW_REDUCTION",
        source_request_hash=source_request_hash,
    )


def allocate_flat_level(
    domain: FeasibleAllocationDomain,
    participant_ids: IdentifierVector,
    request_mw: Sequence[float] | FloatVector | RequestConstruction | MitigatedRequestVector,
    *,
    source_request_hash: Sha256 | None = None,
) -> AlgebraicAllocationResult:
    return allocate_algebraic(
        domain, participant_ids, request_mw,
        rule="FLAT_LEVEL",
        source_request_hash=source_request_hash,
    )


__all__ = [
    "AlgebraicAllocationBinding",
    "AlgebraicAllocationResult",
    "bind_algebraic_allocation",
    "allocate_algebraic",
    "allocate_equal_kw_reduction",
    "allocate_flat_level",
    "allocate_weighted_proportional",
]
