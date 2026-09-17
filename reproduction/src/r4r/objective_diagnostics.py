"""Pure numerical diagnostics for the registered allocation objectives.

This module deliberately evaluates objective components only.  It does not
construct a feasible domain, invoke an LP/QP solver, select a tie-break
direction, or create evidence.  The returned records therefore remain
diagnostic even when an objective specification is author-frozen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from r4r.errors import ValidationError
from r4r.objective_contracts import FairnessObjectiveSpecification, MaxExportObjectiveSpecification
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, IdentifierVector, Sha256
from r4r.types.scalars import FiniteFloat


def _allocation(values: Sequence[float], participant_ids: IdentifierVector, field: str) -> FloatVector:
    if not isinstance(participant_ids, IdentifierVector) or not participant_ids.values:
        raise ValidationError("participant_ids must be a non-empty IdentifierVector")
    allocation = FloatVector(values)
    if len(allocation.values) != len(participant_ids.values):
        raise ValidationError(f"{field} must align with participant_ids")
    if any(value < 0.0 for value in allocation.values):
        raise ValidationError(f"{field} must be nonnegative")
    return allocation


@dataclass(frozen=True, slots=True)
class MaxExportDiagnosticResult:
    participant_ids: IdentifierVector
    allocation_mw: FloatVector
    total_export_mw: FiniteFloat
    tie_break_value: FiniteFloat | None
    objective_specification_hash: Sha256
    status: str
    diagnostic_only: bool = True
    serialization_id = "max_export_diagnostic_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.allocation_mw, FloatVector) or len(self.allocation_mw.values) != len(self.participant_ids.values):
            raise ValidationError("allocation_mw must align with participant_ids")
        if any(value < 0.0 for value in self.allocation_mw.values):
            raise ValidationError("allocation_mw must be nonnegative")
        if not isinstance(self.total_export_mw, FiniteFloat):
            raise ValidationError("total_export_mw must be FiniteFloat")
        if self.tie_break_value is not None and not isinstance(self.tie_break_value, FiniteFloat):
            raise ValidationError("tie_break_value must be FiniteFloat or null")
        if not isinstance(self.objective_specification_hash, Sha256):
            raise ValidationError("objective_specification_hash must be Sha256")
        if self.status not in {"DEFINED", "TIE_BREAK_UNAVAILABLE"}:
            raise ValidationError("max-export diagnostic status is not registered")
        if self.diagnostic_only is not True:
            raise ValidationError("objective diagnostics cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "total_export_mw": self.total_export_mw.to_json(),
            "tie_break_value": self.tie_break_value.to_json() if self.tie_break_value else None,
            "objective_specification_hash": self.objective_specification_hash.to_json(),
            "status": self.status,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def result_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


@dataclass(frozen=True, slots=True)
class FairnessDiagnosticResult:
    participant_ids: IdentifierVector
    allocation_mw: FloatVector
    request_mw: FloatVector
    reference_allocation_mw: FloatVector
    normalized_fraction: FloatVector
    reference_fraction: FiniteFloat
    total_export_mw: FiniteFloat
    fairness_penalty: FiniteFloat
    objective_value: FiniteFloat
    objective_specification_hash: Sha256
    status: str = "DEFINED"
    diagnostic_only: bool = True
    serialization_id = "fairness_diagnostic_result.v1"

    def __post_init__(self) -> None:
        n = len(self.participant_ids.values) if isinstance(self.participant_ids, IdentifierVector) else 0
        if n == 0:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        for name, value in (("allocation_mw", self.allocation_mw), ("request_mw", self.request_mw), ("reference_allocation_mw", self.reference_allocation_mw), ("normalized_fraction", self.normalized_fraction)):
            if not isinstance(value, FloatVector) or len(value.values) != n:
                raise ValidationError(f"{name} must align with participant_ids")
        if any(value < 0.0 for value in self.allocation_mw.values + self.request_mw.values + self.reference_allocation_mw.values):
            raise ValidationError("allocation, request and reference allocation must be nonnegative")
        for name in ("reference_fraction", "total_export_mw", "fairness_penalty", "objective_value"):
            if not isinstance(getattr(self, name), FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat")
        if self.fairness_penalty.value < 0.0:
            raise ValidationError("fairness_penalty must be nonnegative")
        if not isinstance(self.objective_specification_hash, Sha256):
            raise ValidationError("objective_specification_hash must be Sha256")
        if self.status != "DEFINED" or self.diagnostic_only is not True:
            raise ValidationError("fairness diagnostics must remain defined diagnostic records")

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "request_mw": self.request_mw.to_json(),
            "reference_allocation_mw": self.reference_allocation_mw.to_json(),
            "normalized_fraction": self.normalized_fraction.to_json(),
            "reference_fraction": self.reference_fraction.to_json(),
            "total_export_mw": self.total_export_mw.to_json(),
            "fairness_penalty": self.fairness_penalty.to_json(),
            "objective_value": self.objective_value.to_json(),
            "objective_specification_hash": self.objective_specification_hash.to_json(),
            "status": self.status,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def result_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def evaluate_max_export_diagnostic(
    allocation_mw: Sequence[float],
    specification: MaxExportObjectiveSpecification,
) -> MaxExportDiagnosticResult:
    """Evaluate max-export components without solving or choosing a direction."""
    if not isinstance(specification, MaxExportObjectiveSpecification):
        raise ValidationError("specification must be MaxExportObjectiveSpecification")
    allocation = _allocation(allocation_mw, specification.participant_ids, "allocation_mw")
    tie_break_value = None
    status = "TIE_BREAK_UNAVAILABLE"
    if specification.tie_break.weights is not None:
        tie_break_value = FiniteFloat(math.fsum(
            weight * value
            for weight, value in zip(specification.tie_break.weights.values, allocation.values)
        ))
        status = "DEFINED"
    return MaxExportDiagnosticResult(
        participant_ids=specification.participant_ids,
        allocation_mw=allocation,
        total_export_mw=FiniteFloat(math.fsum(allocation.values)),
        tie_break_value=tie_break_value,
        objective_specification_hash=specification.specification_hash,
        status=status,
    )


def evaluate_fairness_diagnostic(
    allocation_mw: Sequence[float],
    request_mw: Sequence[float],
    reference_allocation_mw: Sequence[float],
    specification: FairnessObjectiveSpecification,
) -> FairnessDiagnosticResult:
    """Evaluate fairness with explicit request and max-export reference vectors."""
    if not isinstance(specification, FairnessObjectiveSpecification):
        raise ValidationError("specification must be FairnessObjectiveSpecification")
    allocation = _allocation(allocation_mw, specification.participant_ids, "allocation_mw")
    request = _allocation(request_mw, specification.participant_ids, "request_mw")
    reference = _allocation(reference_allocation_mw, specification.participant_ids, "reference_allocation_mw")
    denominator = specification.epsilon_mw.value
    fractions = FloatVector(value / (request_value + denominator) for value, request_value in zip(allocation.values, request.values))
    reference_fraction = math.fsum(value / (request_value + denominator) for value, request_value in zip(reference.values, request.values)) / len(reference.values)
    penalty = math.fsum((fraction - reference_fraction) ** 2 for fraction in fractions.values)
    total = math.fsum(allocation.values)
    objective = total - specification.alpha_mw.value * penalty
    return FairnessDiagnosticResult(
        participant_ids=specification.participant_ids,
        allocation_mw=allocation,
        request_mw=request,
        reference_allocation_mw=reference,
        normalized_fraction=fractions,
        reference_fraction=FiniteFloat(reference_fraction),
        total_export_mw=FiniteFloat(total),
        fairness_penalty=FiniteFloat(penalty),
        objective_value=FiniteFloat(objective),
        objective_specification_hash=specification.specification_hash,
    )


__all__ = [
    "FairnessDiagnosticResult",
    "MaxExportDiagnosticResult",
    "evaluate_fairness_diagnostic",
    "evaluate_max_export_diagnostic",
]
