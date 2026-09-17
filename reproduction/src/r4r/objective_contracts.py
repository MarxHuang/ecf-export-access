"""Unactivated objective contracts for the later allocation rounds.

These records make the manuscript objective interfaces explicit without
choosing missing scientific semantics or invoking an optimizer.  A contract is
executable only after its references and all unresolved decisions are closed;
the present active contract intentionally keeps these specifications
candidate-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, FiniteFloat, Identifier, IdentifierVector, ObjectiveUnitPolicy, Sha256


_STATUSES = {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}
_TIE_RULES = {"INDEX_WEIGHTED"}
_REFERENCE_MODES = {"ENDOGENOUS_MAX_EXPORT", "EXTERNAL_REFERENCE"}


def _validate_status(status: str, unresolved_fields: tuple[str, ...]) -> None:
    if status not in _STATUSES:
        raise ValidationError("objective scientific_status is not registered")
    if any(not isinstance(field, str) or not field for field in unresolved_fields):
        raise ValidationError("unresolved_fields must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class TieBreakSpecification(ContractModel):
    """Deterministic second-stage ordering for equal max-export optima."""

    tie_break_id: Identifier
    participant_ids: IdentifierVector
    weights: FloatVector | None
    tolerance_mw: FiniteFloat
    rule_id: str
    scientific_status: str
    unresolved_fields: tuple[str, ...]
    serialization_id = "tie_break_specification.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.tie_break_id, Identifier):
            raise ValidationError("tie_break_id must be Identifier")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.weights is not None:
            if not isinstance(self.weights, FloatVector) or len(self.weights.values) != len(self.participant_ids.values):
                raise ValidationError("tie-break weights must match participant IDs")
            if any(value < 0.0 for value in self.weights.values):
                raise ValidationError("tie-break weights must be nonnegative")
        if self.tolerance_mw.value < 0.0:
            raise ValidationError("tie-break tolerance must be nonnegative")
        if self.rule_id not in _TIE_RULES:
            raise ValidationError("tie-break rule is not registered")
        _validate_status(self.scientific_status, self.unresolved_fields)

    @property
    def executable(self) -> bool:
        return self.scientific_status == "AUTHOR_FROZEN" and not self.unresolved_fields and self.weights is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "tie_break_id": self.tie_break_id.to_json(),
            "participant_ids": self.participant_ids.to_json(),
            "weights": self.weights.to_json() if self.weights else None,
            "tolerance_mw": self.tolerance_mw.to_json(),
            "rule_id": self.rule_id,
            "scientific_status": self.scientific_status,
            "unresolved_fields": list(self.unresolved_fields),
            "executable": self.executable,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class MaxExportObjectiveSpecification(ContractModel):
    """Two-stage max-export objective interface; no solve is performed here."""

    objective_id: Identifier
    participant_ids: IdentifierVector
    feasible_domain_hash: Sha256 | None
    equation_ids: IdentifierVector
    tie_break: TieBreakSpecification
    primary_stage_name: str
    secondary_stage_name: str
    preserve_primary_value: bool
    scientific_status: str
    unresolved_fields: tuple[str, ...]
    serialization_id = "max_export_objective_specification.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.objective_id, Identifier):
            raise ValidationError("objective_id must be Identifier")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.feasible_domain_hash is not None and not isinstance(self.feasible_domain_hash, Sha256):
            raise ValidationError("feasible_domain_hash must be Sha256 or null")
        if not isinstance(self.equation_ids, IdentifierVector) or not self.equation_ids.values:
            raise ValidationError("equation_ids must be a non-empty IdentifierVector")
        if not isinstance(self.tie_break, TieBreakSpecification):
            raise ValidationError("tie_break must be TieBreakSpecification")
        if self.tie_break.participant_ids != self.participant_ids:
            raise ValidationError("tie-break participant order must match objective participant order")
        if not self.primary_stage_name or not self.secondary_stage_name or self.primary_stage_name == self.secondary_stage_name:
            raise ValidationError("two distinct objective stage names are required")
        if not isinstance(self.preserve_primary_value, bool):
            raise ValidationError("preserve_primary_value must be boolean")
        _validate_status(self.scientific_status, self.unresolved_fields)

    @property
    def executable(self) -> bool:
        return (
            self.scientific_status == "AUTHOR_FROZEN"
            and not self.unresolved_fields
            and self.feasible_domain_hash is not None
            and self.tie_break.executable
            and self.preserve_primary_value
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id.to_json(),
            "participant_ids": self.participant_ids.to_json(),
            "feasible_domain_hash": self.feasible_domain_hash.to_json() if self.feasible_domain_hash else None,
            "equation_ids": self.equation_ids.to_json(),
            "tie_break": self.tie_break.to_json(),
            "primary_stage_name": self.primary_stage_name,
            "secondary_stage_name": self.secondary_stage_name,
            "preserve_primary_value": self.preserve_primary_value,
            "scientific_status": self.scientific_status,
            "unresolved_fields": list(self.unresolved_fields),
            "executable": self.executable,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class FairnessObjectiveSpecification(ContractModel):
    """Fairness-aware objective parameters and reference binding."""

    objective_id: Identifier
    participant_ids: IdentifierVector
    equation_ids: IdentifierVector
    alpha_mw: FiniteFloat
    epsilon_mw: FiniteFloat
    reference_mode: str
    max_export_specification_hash: Sha256 | None
    external_reference_allocation_hash: Sha256 | None
    objective_unit_policy: ObjectiveUnitPolicy
    scientific_status: str
    unresolved_fields: tuple[str, ...]
    feasible_domain_hash: Sha256 | None = None
    serialization_id = "fairness_objective_specification.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.objective_id, Identifier) or not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("objective ID and participant IDs are required")
        if not isinstance(self.equation_ids, IdentifierVector) or not self.equation_ids.values:
            raise ValidationError("equation_ids must be a non-empty IdentifierVector")
        if self.alpha_mw.value < 0.0 or self.epsilon_mw.value <= 0.0:
            raise ValidationError("alpha must be nonnegative and epsilon must be positive")
        if self.reference_mode not in _REFERENCE_MODES:
            raise ValidationError("fairness reference mode is not registered")
        if self.max_export_specification_hash is not None and not isinstance(self.max_export_specification_hash, Sha256):
            raise ValidationError("max_export_specification_hash must be Sha256 or null")
        if self.external_reference_allocation_hash is not None and not isinstance(self.external_reference_allocation_hash, Sha256):
            raise ValidationError("external_reference_allocation_hash must be Sha256 or null")
        if self.feasible_domain_hash is not None and not isinstance(self.feasible_domain_hash, Sha256):
            raise ValidationError("feasible_domain_hash must be Sha256 or null")
        if self.reference_mode == "ENDOGENOUS_MAX_EXPORT" and self.external_reference_allocation_hash is not None:
            raise ValidationError("endogenous reference cannot carry an external allocation hash")
        if self.reference_mode == "EXTERNAL_REFERENCE" and self.external_reference_allocation_hash is None:
            raise ValidationError("external reference mode requires an allocation hash")
        if not isinstance(self.objective_unit_policy, ObjectiveUnitPolicy):
            raise ValidationError("objective_unit_policy must be registered")
        _validate_status(self.scientific_status, self.unresolved_fields)

    @property
    def executable(self) -> bool:
        reference_ready = (
            self.max_export_specification_hash is not None
            if self.reference_mode == "ENDOGENOUS_MAX_EXPORT"
            else self.external_reference_allocation_hash is not None
        )
        return self.scientific_status == "AUTHOR_FROZEN" and not self.unresolved_fields and reference_ready

    def to_json(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id.to_json(),
            "participant_ids": self.participant_ids.to_json(),
            "equation_ids": self.equation_ids.to_json(),
            "alpha_mw": self.alpha_mw.to_json(),
            "epsilon_mw": self.epsilon_mw.to_json(),
            "reference_mode": self.reference_mode,
            "max_export_specification_hash": self.max_export_specification_hash.to_json() if self.max_export_specification_hash else None,
            "external_reference_allocation_hash": self.external_reference_allocation_hash.to_json() if self.external_reference_allocation_hash else None,
            "objective_unit_policy": self.objective_unit_policy.value,
            "scientific_status": self.scientific_status,
            "unresolved_fields": list(self.unresolved_fields),
            "feasible_domain_hash": self.feasible_domain_hash.to_json() if self.feasible_domain_hash else None,
            "executable": self.executable,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class NumericalAcceptanceSpecification(ContractModel):
    """Explicit numerical tolerances for future LP/QP acceptance."""

    specification_id: Identifier
    solver_adapter_ids: IdentifierVector
    primal_tolerance: FiniteFloat
    dual_tolerance: FiniteFloat
    kkt_tolerance: FiniteFloat
    objective_tolerance: FiniteFloat
    feasibility_tolerance: FiniteFloat
    scientific_status: str
    unresolved_fields: tuple[str, ...]
    serialization_id = "numerical_acceptance_specification.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.specification_id, Identifier) or not isinstance(self.solver_adapter_ids, IdentifierVector) or not self.solver_adapter_ids.values:
            raise ValidationError("specification ID and solver adapter IDs are required")
        for name in ("primal_tolerance", "dual_tolerance", "kkt_tolerance", "objective_tolerance", "feasibility_tolerance"):
            if getattr(self, name).value < 0.0:
                raise ValidationError(f"{name} must be nonnegative")
        _validate_status(self.scientific_status, self.unresolved_fields)

    @property
    def executable(self) -> bool:
        return self.scientific_status == "AUTHOR_FROZEN" and not self.unresolved_fields

    def to_json(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id.to_json(),
            "solver_adapter_ids": self.solver_adapter_ids.to_json(),
            "primal_tolerance": self.primal_tolerance.to_json(),
            "dual_tolerance": self.dual_tolerance.to_json(),
            "kkt_tolerance": self.kkt_tolerance.to_json(),
            "objective_tolerance": self.objective_tolerance.to_json(),
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
            "scientific_status": self.scientific_status,
            "unresolved_fields": list(self.unresolved_fields),
            "executable": self.executable,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))
