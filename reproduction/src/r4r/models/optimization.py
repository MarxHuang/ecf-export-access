"""Optimization and solver result interfaces; no solver invocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from r4r.errors import ValidationError
from r4r.contracts.registry_loader import registered_failure_reason_ids
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier, require_mapping, require_reference
from r4r.types import (
    FloatVector,
    Identifier,
    IdentifierVector,
    NormalizedSolverStatus,
    ObjectReference,
    ObjectiveUnitPolicy,
    RawSolverStatus,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition(ContractModel):
    objective_id: Identifier
    expression_id: Identifier
    participant_registry_index: tuple[int, ...]
    alpha: FiniteFloat | None
    epsilon: FiniteFloat | None
    reference_solution_id: Identifier
    serialization_id = "objective.v1"

    def __post_init__(self) -> None:
        for name, value in (("objective_id", self.objective_id), ("expression_id", self.expression_id), ("reference_solution_id", self.reference_solution_id)):
            require_identifier(value, name)
        if isinstance(self.participant_registry_index, (str, bytes)):
            raise ValidationError("participant_registry_index must be an integer vector")
        try:
            indices = tuple(self.participant_registry_index)
        except TypeError as exc:
            raise ValidationError("participant_registry_index must be an integer vector") from exc
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in indices):
            raise ValidationError("participant_registry_index must contain positive integers")
        if len(set(indices)) != len(indices):
            raise ValidationError("participant_registry_index must not contain duplicates")
        object.__setattr__(self, "participant_registry_index", indices)


@dataclass(frozen=True, slots=True)
class OptimizationProblem(ContractModel):
    objective_id: Identifier
    constraints: dict[str, Any]
    request: ObjectReference
    serialization_id = "optimization_problem.v1"

    def __post_init__(self) -> None:
        require_identifier(self.objective_id, "objective_id")
        require_mapping(self.constraints, "constraints")
        require_reference(self.request, "request", "ENT008")


@dataclass(frozen=True, slots=True)
class SolverResult(ContractModel):
    solver_name: Identifier
    solver_version: str
    raw_status: RawSolverStatus
    normalized_status: NormalizedSolverStatus
    solver_domain: str = "UNSPECIFIED"
    evidence_scope: str = "UNSPECIFIED"
    objective_components: dict[str, float] = field(default_factory=dict)
    common_objective_value: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    stage1_total: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    # DEC015 records these only for a certified two-stage max-export solve.
    # Generic AC/LP/QP interfaces preserve explicit null rather than inventing
    # numerical hierarchy evidence for a solve that did not use that policy.
    stage1_primal_lower_mw: FiniteFloat | None = None
    stage1_dual_upper_mw: FiniteFloat | None = None
    tau_h_mw: FiniteFloat | None = None
    stage2_primary_floor_mw: FiniteFloat | None = None
    stage2_normalized_decision_enclosure: FiniteFloat | None = None
    stage2_canonical_objective_mw: FiniteFloat | None = None
    stage2_objective_gradient_mw_inverse: FloatVector | None = None
    stage2_strong_convexity_modulus_mw_inverse: FiniteFloat | None = None
    stage2_weight_policy: str | None = None
    stage2_eta_dimensionless: FiniteFloat | None = None
    stage2_s_t_mw: FiniteFloat | None = None
    hierarchy_attestation: Sha256 | None = None
    fairness_reference_attestation: Sha256 | None = None
    tie_break_value: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    fairness_total_component: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    fairness_penalty_component: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    primal_residual: FloatVector = field(default_factory=lambda: FloatVector([]))
    dual_sign_violation: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    stationarity_residual: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    complementarity_residual: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    bound_violation: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    primal_dual_gap: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    iteration_count: int = 0
    solve_attempted: bool = False
    solve_success: bool = False
    evidence_valid: bool = False
    failure_reasons: IdentifierVector = field(default_factory=lambda: IdentifierVector([]))
    objective_unit_policy: ObjectiveUnitPolicy = ObjectiveUnitPolicy.UNDEFINED
    objective_component_units: dict[str, str] = field(default_factory=dict)
    raw_primal_residual: FloatVector = field(default_factory=lambda: FloatVector([]))
    scaled_primal_residual: FloatVector = field(default_factory=lambda: FloatVector([]))
    residual_norm: FiniteFloat = field(default_factory=lambda: FiniteFloat(0.0))
    tolerance_reference: Identifier = field(default_factory=lambda: Identifier("UNREGISTERED"))
    # Numerical provenance is optional for generic optimization interfaces,
    # but is mandatory (and populated) for the AC solver slice.  Keeping the
    # fields here binds the actual SolverResult to the same payload hashes as
    # the typed AC output without inventing defaults for older non-AC callers.
    network_input_hash: Sha256 | None = None
    p_injection_hash: Sha256 | None = None
    q_injection_hash: Sha256 | None = None
    solver_config_hash: Sha256 | None = None
    solver_environment_hash: Sha256 | None = None
    solution_hash: Sha256 | None = None
    serialization_id = "solver_result.v1"

    def __post_init__(self) -> None:
        require_identifier(self.solver_name, "solver_name")
        if not isinstance(self.solver_version, str) or not self.solver_version:
            raise ValidationError("solver_version must be a non-empty string")
        if not isinstance(self.raw_status, RawSolverStatus) or not isinstance(self.normalized_status, NormalizedSolverStatus):
            raise ValidationError("solver statuses must be registered enums")
        if self.solver_domain not in {"AC", "LP", "QP", "UNSPECIFIED"}:
            raise ValidationError("solver_domain must be AC, LP, QP, or UNSPECIFIED")
        if self.evidence_scope not in {"NUMERICAL_IMPLEMENTATION_ONLY", "OPTIMIZATION_RESULT", "UNSPECIFIED"}:
            raise ValidationError("evidence_scope is not registered")
        if isinstance(self.iteration_count, bool) or self.iteration_count < 0:
            raise ValidationError("iteration_count must be a nonnegative integer")
        if not isinstance(self.solve_attempted, bool) or not isinstance(self.solve_success, bool) or not isinstance(self.evidence_valid, bool):
            raise ValidationError("solver booleans must be explicit")
        # The raw and normalized status fields are evidence about one solve
        # attempt, not independent labels.  Keep the unattempted state
        # fail-closed so downstream T1/T2 gates cannot mistake a default or
        # fabricated status for solver evidence.
        if not self.solve_attempted:
            if self.raw_status is not RawSolverStatus.NOT_RUN or self.normalized_status is not NormalizedSolverStatus.NOT_ATTEMPTED:
                raise ValidationError("unattempted solver result must use NOT_RUN/NOT_ATTEMPTED statuses")
            if self.solve_success or self.evidence_valid:
                raise ValidationError("unattempted solver result cannot be successful or valid evidence")
        else:
            if self.raw_status is RawSolverStatus.NOT_RUN or self.normalized_status is NormalizedSolverStatus.NOT_ATTEMPTED:
                raise ValidationError("attempted solver result cannot use NOT_RUN/NOT_ATTEMPTED statuses")
        if self.normalized_status is NormalizedSolverStatus.PASS and not self.solve_success:
            raise ValidationError("normalized PASS requires solve_success=True")
        if self.evidence_valid and not self.solve_success:
            raise ValidationError("valid solver evidence requires solve_success=True")
        if self.evidence_valid and (self.solver_domain == "UNSPECIFIED" or self.evidence_scope == "UNSPECIFIED"):
            raise ValidationError("valid solver evidence requires an explicit domain and evidence scope")
        if not isinstance(self.objective_unit_policy, ObjectiveUnitPolicy):
            raise ValidationError("objective_unit_policy must be registered")
        for name in (
            "stage1_primal_lower_mw", "stage1_dual_upper_mw", "tau_h_mw",
            "stage2_primary_floor_mw", "stage2_normalized_decision_enclosure",
            "stage2_canonical_objective_mw", "stage2_strong_convexity_modulus_mw_inverse",
            "stage2_eta_dimensionless", "stage2_s_t_mw",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat or explicit null")
        if self.stage2_objective_gradient_mw_inverse is not None and not isinstance(self.stage2_objective_gradient_mw_inverse, FloatVector):
            raise ValidationError("stage2_objective_gradient_mw_inverse must be FloatVector or explicit null")
        if self.stage2_weight_policy is not None and self.stage2_weight_policy != "CENTERED_NORMALIZED_REGISTRY_ORDER":
            raise ValidationError("stage2_weight_policy is not registered")
        for name in ("hierarchy_attestation", "fairness_reference_attestation"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or explicit null")
        hierarchy_values = (
            self.stage1_primal_lower_mw, self.stage1_dual_upper_mw, self.tau_h_mw,
            self.stage2_primary_floor_mw, self.stage2_normalized_decision_enclosure,
            self.stage2_canonical_objective_mw, self.stage2_objective_gradient_mw_inverse,
            self.stage2_strong_convexity_modulus_mw_inverse, self.stage2_weight_policy,
            self.stage2_eta_dimensionless, self.stage2_s_t_mw,
            self.hierarchy_attestation,
        )
        if any(value is not None for value in hierarchy_values) and any(value is None for value in hierarchy_values):
            raise ValidationError("certified hierarchy evidence must be complete or explicitly absent")
        if self.fairness_reference_attestation is not None and self.hierarchy_attestation is None:
            raise ValidationError("fairness reference attestation requires hierarchy attestation")
        unknown_reasons = sorted(
            reason.value for reason in self.failure_reasons.values
            if reason.value not in registered_failure_reason_ids()
        )
        if unknown_reasons:
            raise ValidationError("unregistered solver failure reasons: " + ",".join(unknown_reasons))
        for name in (
            "network_input_hash", "p_injection_hash", "q_injection_hash",
            "solver_config_hash", "solver_environment_hash", "solution_hash",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or explicit null")
