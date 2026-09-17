"""Objective-specific LP/QP problem adapters for diagnostic experiments.

These adapters consume a typed network-constrained domain and explicit
objective specifications.  They build solver inputs only; callers must still
use the diagnostic solver functions and cannot promote the returned records.
Tie-break direction is an explicit argument, never inferred from a rule name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.objective_contracts import FairnessObjectiveSpecification, MaxExportObjectiveSpecification
from r4r.request_generation import RequestConstruction
from r4r.serialization import canonical_hash
from r4r.types import FloatMatrix, FloatVector, FiniteFloat, Identifier, IdentifierVector, Sha256
from tools.diagnostic_optimization import (
    DiagnosticConstraintRow,
    DiagnosticLinearProblem,
    DiagnosticQuadraticProblem,
    DiagnosticSolveResult,
    is_solver_emitted_result,
    solve_diagnostic_lp,
)


def _diagnostic_solver_result_hash(result: DiagnosticSolveResult) -> Sha256:
    payload = {
        "problem_hash": result.problem_hash.to_json(),
        "solver_name": result.solver_name.to_json(),
        "raw_status": result.raw_status.value,
        "normalized_status": result.normalized_status.value,
        "variable_ids": result.variable_ids.to_json(),
        "solution": result.solution.to_json() if result.solution else None,
        "objective_value": result.objective_value.to_json() if result.objective_value else None,
        "constraint_residuals": result.constraint_residuals.to_json() if result.constraint_residuals else None,
        "failure_reason": result.failure_reason.to_json() if result.failure_reason else None,
    }
    return Sha256(canonical_hash(payload))


def _check_domain_alignment(domain: FeasibleAllocationDomain, participant_ids: IdentifierVector) -> None:
    if not isinstance(domain, FeasibleAllocationDomain):
        raise ValidationError("domain must be FeasibleAllocationDomain")
    if not isinstance(participant_ids, IdentifierVector) or participant_ids != domain.participant_ids:
        raise ValidationError("objective participant order must match feasible domain")
    if not domain.constraints:
        raise ValidationError("objective problem requires explicit network constraints")


def _check_domain_hash(
    domain: FeasibleAllocationDomain,
    feasible_domain_hash: Sha256 | None,
) -> None:
    """Require an explicit objective/domain provenance binding when supplied.

    Participant order alone is not enough to identify a feasible allocation
    domain: two domains can contain the same participants while differing in
    proxy rows, bounds, or tightened budgets.  A missing hash remains allowed
    for candidate diagnostic specifications, but a supplied hash must match
    the exact domain consumed by the adapter.
    """
    if feasible_domain_hash is not None and feasible_domain_hash != domain.domain_hash:
        raise ValidationError("objective feasible_domain_hash does not match domain")


def _diagnostic_rows(domain: FeasibleAllocationDomain) -> tuple[DiagnosticConstraintRow, ...]:
    return tuple(
        DiagnosticConstraintRow(
            constraint_id=row.constraint_id,
            coefficients=row.coefficients,
            rhs=row.rhs,
            sense=row.sense,
        )
        for row in domain.constraints
    )


def build_diagnostic_max_export_stage1_problem(
    domain: FeasibleAllocationDomain,
    specification: MaxExportObjectiveSpecification,
    *,
    context_hash: Sha256,
) -> DiagnosticLinearProblem:
    """Build the first-stage total-export diagnostic LP."""
    if not isinstance(specification, MaxExportObjectiveSpecification):
        raise ValidationError("specification must be MaxExportObjectiveSpecification")
    if not isinstance(context_hash, Sha256):
        raise ValidationError("context_hash must be Sha256")
    _check_domain_alignment(domain, specification.participant_ids)
    _check_domain_hash(domain, specification.feasible_domain_hash)
    return DiagnosticLinearProblem(
        variable_ids=domain.participant_ids,
        objective=FloatVector([1.0] * len(domain.participant_ids.values)),
        constraints=_diagnostic_rows(domain),
        lower_bounds=domain.lower_bounds_mw,
        upper_bounds=domain.upper_bounds_mw,
        objective_sense="MAXIMIZE",
        context_hash=context_hash,
        objective_specification_hash=specification.specification_hash,
        objective_family="MAX_EXPORT",
        objective_stage="MAX_EXPORT_STAGE1",
    )


@dataclass(frozen=True, slots=True)
class DiagnosticMaxExportStage2ProblemBinding:
    """Typed stage-two max-export problem with primary-stage provenance.

    A bare LP contains no information about which first-stage optimum it is
    preserving.  This wrapper keeps that value and the objective specification
    attached to the LP and rechecks the primary-preservation row before a
    solver result can be bound downstream.
    """

    problem: DiagnosticLinearProblem
    objective_specification: MaxExportObjectiveSpecification
    stage1_problem: DiagnosticLinearProblem
    stage1_solver_result: DiagnosticSolveResult
    tie_break_sense: str
    diagnostic_only: bool = True
    serialization_id = "diagnostic_max_export_stage2_problem_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.problem, DiagnosticLinearProblem):
            raise ValidationError("stage2 binding requires a DiagnosticLinearProblem")
        if not isinstance(self.stage1_problem, DiagnosticLinearProblem):
            raise ValidationError("stage2 binding requires the stage1 DiagnosticLinearProblem")
        if not isinstance(self.stage1_solver_result, DiagnosticSolveResult):
            raise ValidationError("stage2 binding requires the stage1 DiagnosticSolveResult")
        if not isinstance(self.objective_specification, MaxExportObjectiveSpecification):
            raise ValidationError("stage2 binding requires MaxExportObjectiveSpecification")
        if self.stage1_problem.variable_ids != self.problem.variable_ids:
            raise ValidationError("stage1 and stage2 variable order must match")
        if self.objective_specification.participant_ids != self.problem.variable_ids:
            raise ValidationError("objective specification participant order does not match stage2 problem")
        if self.stage1_problem.objective_specification_hash != self.objective_specification.specification_hash:
            raise ValidationError("stage1 problem is not bound to objective specification")
        if self.stage1_problem.objective_family != "MAX_EXPORT" or self.stage1_problem.objective_stage != "MAX_EXPORT_STAGE1":
            raise ValidationError("stage1 problem objective metadata is inconsistent")
        expected_stage1_objective = (1.0,) * len(self.stage1_problem.variable_ids.values)
        if self.stage1_problem.objective.values != expected_stage1_objective or self.stage1_problem.objective_sense != "MAXIMIZE":
            raise ValidationError("stage1 problem objective is inconsistent")
        if self.stage1_solver_result.problem_hash != self.stage1_problem.problem_hash:
            raise ValidationError("stage1 solver result does not belong to stage1 problem")
        if self.stage1_solver_result.variable_ids != self.stage1_problem.variable_ids:
            raise ValidationError("stage1 solver result variable order does not match stage1 problem")
        if self.stage1_solver_result.normalized_status.value != "PASS" or self.stage1_solver_result.solution is None:
            raise ValidationError("stage2 binding requires a successful stage1 solver result")
        if self.stage1_solver_result.result_hash != _diagnostic_solver_result_hash(self.stage1_solver_result):
            raise ValidationError("stage1 solver result hash does not match its contents")
        if not is_solver_emitted_result(self.stage1_solver_result):
            raise ValidationError("stage1 solver result was not emitted by a diagnostic solver")
        stage1_total_mw = math.fsum(self.stage1_solver_result.solution.values)
        if any(value < 0.0 for value in self.stage1_solver_result.solution.values):
            raise ValidationError("stage1 solver result contains a negative allocation")
        if self.stage1_solver_result.objective_value is None or abs(self.stage1_solver_result.objective_value.value - stage1_total_mw) > 1e-10:
            raise ValidationError("stage1 solver result objective does not match its solution")
        if self.tie_break_sense not in {"MINIMIZE", "MAXIMIZE"}:
            raise ValidationError("stage2 tie-break sense is not registered")
        if self.diagnostic_only is not True:
            raise ValidationError("stage2 binding cannot be promoted")
        if self.problem.objective_specification_hash != self.objective_specification.specification_hash:
            raise ValidationError("stage2 problem is not bound to its objective specification")
        if self.problem.objective_family != "MAX_EXPORT" or self.problem.objective_stage != "MAX_EXPORT_STAGE2":
            raise ValidationError("stage2 problem objective metadata is inconsistent")
        weights = self.objective_specification.tie_break.weights
        if weights is None or self.problem.objective.values != weights.values:
            raise ValidationError("stage2 objective coefficients do not match tie-break weights")
        if self.problem.objective_sense != self.tie_break_sense:
            raise ValidationError("stage2 objective sense does not match recorded tie-break sense")
        primary_rows = [
            row for row in self.problem.constraints
            if row.constraint_id == Identifier("MAX_EXPORT_PRIMARY_PRESERVED")
        ]
        if len(primary_rows) != 1:
            raise ValidationError("stage2 problem must contain exactly one primary-preservation row")
        primary = primary_rows[0]
        stage2_base_constraints = tuple(
            row for row in self.problem.constraints
            if row.constraint_id != Identifier("MAX_EXPORT_PRIMARY_PRESERVED")
        )
        if (
            stage2_base_constraints != self.stage1_problem.constraints
            or self.problem.lower_bounds != self.stage1_problem.lower_bounds
            or self.problem.upper_bounds != self.stage1_problem.upper_bounds
            or self.problem.context_hash != self.stage1_problem.context_hash
        ):
            raise ValidationError("stage1 and stage2 problems do not share the same base domain")
        expected_coefficients = (1.0,) * len(self.problem.variable_ids.values)
        expected_rhs = stage1_total_mw - self.objective_specification.tie_break.tolerance_mw.value
        if (
            primary.coefficients.values != expected_coefficients
            or primary.sense != "GREATER_EQUAL"
            or abs(primary.rhs.value - expected_rhs) > 1e-12
        ):
            raise ValidationError("stage2 primary-preservation row is inconsistent")
        replay = solve_diagnostic_lp(self.stage1_problem)
        if replay.normalized_status.value != "PASS" or replay.objective_value is None:
            raise ValidationError("stage1 independent replay did not succeed")
        if abs(replay.objective_value.value - stage1_total_mw) > max(
            1e-9, self.objective_specification.tie_break.tolerance_mw.value
        ):
            raise ValidationError("stage1 solver result does not match independent optimum replay")

    def __getattr__(self, name: str) -> Any:
        # Preserve the diagnostic problem surface for existing solver adapters
        # while keeping the provenance wrapper explicit at binding boundaries.
        return getattr(self.problem, name)

    @property
    def stage1_total_mw(self) -> FiniteFloat:
        """Derived primary optimum; never caller-supplied provenance."""
        return FiniteFloat(math.fsum(self.stage1_solver_result.solution.values))

    def to_json(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_json(),
            "stage1_problem": self.stage1_problem.to_json(),
            "stage1_solver_result": self.stage1_solver_result.to_json(),
            "objective_specification_hash": self.objective_specification.specification_hash.to_json(),
            "stage1_total_mw": FiniteFloat(math.fsum(self.stage1_solver_result.solution.values)).to_json(),
            "tie_break_sense": self.tie_break_sense,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def binding_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def build_diagnostic_max_export_stage2_problem(
    domain: FeasibleAllocationDomain,
    specification: MaxExportObjectiveSpecification,
    *,
    stage1_problem: DiagnosticLinearProblem,
    stage1_solver_result: DiagnosticSolveResult,
    context_hash: Sha256,
    tie_break_sense: str,
) -> DiagnosticMaxExportStage2ProblemBinding:
    """Build the tie-break LP while explicitly preserving stage-one value."""
    if not isinstance(specification, MaxExportObjectiveSpecification):
        raise ValidationError("specification must be MaxExportObjectiveSpecification")
    if not isinstance(context_hash, Sha256):
        raise ValidationError("context_hash must be Sha256")
    if tie_break_sense not in {"MINIMIZE", "MAXIMIZE"}:
        raise ValidationError("tie_break_sense must be explicitly MINIMIZE or MAXIMIZE")
    if not isinstance(stage1_problem, DiagnosticLinearProblem):
        raise ValidationError("stage1_problem must be DiagnosticLinearProblem")
    if not isinstance(stage1_solver_result, DiagnosticSolveResult):
        raise ValidationError("stage1_solver_result must be DiagnosticSolveResult")
    if specification.tie_break.weights is None:
        raise ValidationError("tie-break weights are unresolved")
    _check_domain_alignment(domain, specification.participant_ids)
    _check_domain_hash(domain, specification.feasible_domain_hash)
    if stage1_problem.variable_ids != domain.participant_ids:
        raise ValidationError("stage1 problem participant order must match domain")
    if stage1_solver_result.problem_hash != stage1_problem.problem_hash:
        raise ValidationError("stage1 solver result does not belong to stage1 problem")
    if stage1_solver_result.normalized_status.value != "PASS" or stage1_solver_result.solution is None:
        raise ValidationError("stage1 solver result must be successful")
    if stage1_solver_result.result_hash != _diagnostic_solver_result_hash(stage1_solver_result):
        raise ValidationError("stage1 solver result hash does not match its contents")
    if not is_solver_emitted_result(stage1_solver_result):
        raise ValidationError("stage1 solver result was not emitted by a diagnostic solver")
    expected_stage1_problem = build_diagnostic_max_export_stage1_problem(
        domain, specification, context_hash=stage1_problem.context_hash
    )
    if stage1_problem.problem_hash != expected_stage1_problem.problem_hash:
        raise ValidationError("stage1 problem is not the exact problem built from this domain")
    stage1_total_mw = math.fsum(stage1_solver_result.solution.values)
    primary_row = DiagnosticConstraintRow(
        constraint_id=Identifier("MAX_EXPORT_PRIMARY_PRESERVED"),
        coefficients=FloatVector([1.0] * len(domain.participant_ids.values)),
        rhs=FiniteFloat(float(stage1_total_mw) - specification.tie_break.tolerance_mw.value),
        sense="GREATER_EQUAL",
    )
    problem = DiagnosticLinearProblem(
        variable_ids=domain.participant_ids,
        objective=specification.tie_break.weights,
        constraints=(*_diagnostic_rows(domain), primary_row),
        lower_bounds=domain.lower_bounds_mw,
        upper_bounds=domain.upper_bounds_mw,
        objective_sense=tie_break_sense,
        context_hash=context_hash,
        objective_specification_hash=specification.specification_hash,
        objective_family="MAX_EXPORT",
        objective_stage="MAX_EXPORT_STAGE2",
    )
    return DiagnosticMaxExportStage2ProblemBinding(
        problem=problem,
        objective_specification=specification,
        stage1_problem=stage1_problem,
        stage1_solver_result=stage1_solver_result,
        tie_break_sense=tie_break_sense,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticMaxExportSolveResult(ContractModel):
    """Two-stage max-export execution record kept permanently diagnostic."""

    objective_specification_hash: Sha256
    stage1_solver: DiagnosticSolveResult
    stage2_solver: DiagnosticSolveResult | None
    stage1_total_mw: FiniteFloat | None
    stage2_total_mw: FiniteFloat | None
    status: str
    failure_reason: Identifier | None
    diagnostic_only: bool = True
    serialization_id = "diagnostic_max_export_solve.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.objective_specification_hash, Sha256):
            raise ValidationError("objective_specification_hash must be Sha256")
        if not isinstance(self.stage1_solver, DiagnosticSolveResult):
            raise ValidationError("stage1_solver must be DiagnosticSolveResult")
        if self.stage2_solver is not None and not isinstance(self.stage2_solver, DiagnosticSolveResult):
            raise ValidationError("stage2_solver must be DiagnosticSolveResult or null")
        for name, value in (("stage1_total_mw", self.stage1_total_mw), ("stage2_total_mw", self.stage2_total_mw)):
            if value is not None and not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat or null")
        if self.status not in {"MAX_EXPORT_DIAGNOSTIC", "MAX_EXPORT_UNAVAILABLE"}:
            raise ValidationError("max-export diagnostic status is not registered")
        if self.status == "MAX_EXPORT_DIAGNOSTIC":
            if self.stage2_solver is None or self.stage1_total_mw is None or self.stage2_total_mw is None:
                raise ValidationError("successful max-export diagnostic requires both stages")
            if self.failure_reason is not None:
                raise ValidationError("successful max-export diagnostic cannot carry failure reason")
        elif self.failure_reason is None:
            raise ValidationError("unavailable max-export diagnostic requires failure reason")
        if self.failure_reason is not None and not isinstance(self.failure_reason, Identifier):
            raise ValidationError("failure_reason must be Identifier or null")
        if self.diagnostic_only is not True:
            raise ValidationError("max-export diagnostic results cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "objective_specification_hash": self.objective_specification_hash.to_json(),
            "stage1_solver": self.stage1_solver.to_json(),
            "stage2_solver": self.stage2_solver.to_json() if self.stage2_solver else None,
            "stage1_total_mw": self.stage1_total_mw.to_json() if self.stage1_total_mw else None,
            "stage2_total_mw": self.stage2_total_mw.to_json() if self.stage2_total_mw else None,
            "status": self.status,
            "failure_reason": self.failure_reason.to_json() if self.failure_reason else None,
            "diagnostic_only": self.diagnostic_only,
        }


def solve_diagnostic_max_export(
    domain: FeasibleAllocationDomain,
    specification: MaxExportObjectiveSpecification,
    *,
    context_hash: Sha256,
    tie_break_sense: str,
) -> DiagnosticMaxExportSolveResult:
    """Solve both max-export stages and verify primary-value preservation.

    This helper is intentionally diagnostic-only.  It records a failed second
    stage instead of returning a tie-break allocation that could silently lose
    the stage-one total.
    """
    stage1_problem = build_diagnostic_max_export_stage1_problem(
        domain, specification, context_hash=context_hash
    )
    stage1_solver = solve_diagnostic_lp(stage1_problem)
    if stage1_solver.normalized_status.value != "PASS" or stage1_solver.solution is None:
        return DiagnosticMaxExportSolveResult(
            objective_specification_hash=specification.specification_hash,
            stage1_solver=stage1_solver,
            stage2_solver=None,
            stage1_total_mw=None,
            stage2_total_mw=None,
            status="MAX_EXPORT_UNAVAILABLE",
            failure_reason=Identifier("REFERENCE_SOLVE_FAILED"),
        )
    stage1_total = math.fsum(stage1_solver.solution.values)
    stage2_problem = build_diagnostic_max_export_stage2_problem(
        domain,
        specification,
        stage1_problem=stage1_problem,
        stage1_solver_result=stage1_solver,
        context_hash=context_hash,
        tie_break_sense=tie_break_sense,
    )
    stage2_solver = solve_diagnostic_lp(stage2_problem)
    if stage2_solver.normalized_status.value != "PASS" or stage2_solver.solution is None:
        return DiagnosticMaxExportSolveResult(
            objective_specification_hash=specification.specification_hash,
            stage1_solver=stage1_solver,
            stage2_solver=stage2_solver,
            stage1_total_mw=FiniteFloat(stage1_total),
            stage2_total_mw=None,
            status="MAX_EXPORT_UNAVAILABLE",
            failure_reason=Identifier("REPORTED_SOLVE_FAILED"),
        )
    stage2_total = math.fsum(stage2_solver.solution.values)
    if stage2_total < stage1_total - specification.tie_break.tolerance_mw.value:
        return DiagnosticMaxExportSolveResult(
            objective_specification_hash=specification.specification_hash,
            stage1_solver=stage1_solver,
            stage2_solver=stage2_solver,
            stage1_total_mw=FiniteFloat(stage1_total),
            stage2_total_mw=FiniteFloat(stage2_total),
            status="MAX_EXPORT_UNAVAILABLE",
            failure_reason=Identifier("OBJECTIVE_MISMATCH"),
        )
    return DiagnosticMaxExportSolveResult(
        objective_specification_hash=specification.specification_hash,
        stage1_solver=stage1_solver,
        stage2_solver=stage2_solver,
        stage1_total_mw=FiniteFloat(stage1_total),
        stage2_total_mw=FiniteFloat(stage2_total),
        status="MAX_EXPORT_DIAGNOSTIC",
        failure_reason=None,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticFairnessQPBinding:
    problem: DiagnosticQuadraticProblem
    request_mw: FloatVector
    reference_allocation_mw: FloatVector
    reference_fraction: FiniteFloat
    omitted_constant: FiniteFloat
    reference_allocation_hash: Sha256
    objective_specification_hash: Sha256
    objective_specification: FairnessObjectiveSpecification
    # This is the hash of the admitted-request source object.  It is deliberately
    # not typed as a RequestConstruction hash: a reported side is sourced from a
    # StrategicReportResult and must not be given an independent construction.
    request_source_hash: Sha256 | None = None
    domain_hash: Sha256 | None = None
    request_hash: Sha256 | None = None
    request_vector_hash: Sha256 | None = None
    diagnostic_only: bool = True
    serialization_id = "diagnostic_fairness_qp_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.problem, DiagnosticQuadraticProblem):
            raise ValidationError("problem must be DiagnosticQuadraticProblem")
        n = len(self.problem.variable_ids.values)
        if not isinstance(self.request_mw, FloatVector) or not isinstance(self.reference_allocation_mw, FloatVector):
            raise ValidationError("request and reference vectors must be FloatVector values")
        if len(self.request_mw.values) != n or len(self.reference_allocation_mw.values) != n:
            raise ValidationError("request/reference vectors must match QP variables")
        if not isinstance(self.reference_fraction, FiniteFloat) or not isinstance(self.omitted_constant, FiniteFloat):
            raise ValidationError("fairness scalar components must be FiniteFloat")
        if not isinstance(self.reference_allocation_hash, Sha256) or not isinstance(self.objective_specification_hash, Sha256):
            raise ValidationError("fairness binding hashes must be Sha256")
        if self.request_source_hash is not None and not isinstance(self.request_source_hash, Sha256):
            raise ValidationError("request_source_hash must be Sha256 or null")
        if self.domain_hash is not None and not isinstance(self.domain_hash, Sha256):
            raise ValidationError("domain_hash must be Sha256 or null")
        if self.request_hash is not None and not isinstance(self.request_hash, Sha256):
            raise ValidationError("request_hash must be Sha256 or null")
        if self.request_vector_hash is not None and not isinstance(self.request_vector_hash, Sha256):
            raise ValidationError("request_vector_hash must be Sha256 or null")
        if not isinstance(self.objective_specification, FairnessObjectiveSpecification):
            raise ValidationError("fairness binding requires FairnessObjectiveSpecification")
        if any(value < 0.0 for value in self.request_mw.values + self.reference_allocation_mw.values):
            raise ValidationError("fairness request and reference vectors must be nonnegative")
        if self.objective_specification_hash != self.objective_specification.specification_hash:
            raise ValidationError("fairness binding objective hash does not match specification")
        if self.objective_specification.participant_ids != self.problem.variable_ids:
            raise ValidationError("fairness specification participant order does not match QP variables")
        if self.problem.objective_specification_hash != self.objective_specification_hash:
            raise ValidationError("fairness problem objective hash does not match specification")
        if self.problem.objective_family != "FAIRNESS" or self.problem.objective_stage != "FAIRNESS_QP":
            raise ValidationError("fairness problem objective metadata is inconsistent")
        expected_reference_hash = Sha256(canonical_hash({"participant_ids": self.problem.variable_ids.to_json(), "values_mw": self.reference_allocation_mw.to_json()}))
        if self.reference_allocation_hash != expected_reference_hash:
            raise ValidationError("fairness reference allocation hash does not match reference values")
        if self.objective_specification.reference_mode == "EXTERNAL_REFERENCE":
            if self.objective_specification.external_reference_allocation_hash != self.reference_allocation_hash:
                raise ValidationError("fairness reference does not match external reference specification")
        elif self.objective_specification.reference_mode == "ENDOGENOUS_MAX_EXPORT":
            if self.objective_specification.max_export_specification_hash is None:
                raise ValidationError("endogenous fairness reference requires max-export specification hash")
        else:
            raise ValidationError("fairness reference mode is not registered")
        expected_request_vector_hash = Sha256(canonical_hash({"participant_ids": self.problem.variable_ids.to_json(), "values_mw": self.request_mw.to_json()}))
        if self.request_vector_hash is None:
            object.__setattr__(self, "request_vector_hash", expected_request_vector_hash)
        if self.request_vector_hash != expected_request_vector_hash:
            raise ValidationError("fairness request_vector_hash does not match request_mw")
        denominator = tuple(value + self.objective_specification.epsilon_mw.value for value in self.request_mw.values)
        phi = math.fsum(value / d for value, d in zip(self.reference_allocation_mw.values, denominator)) / n
        alpha = self.objective_specification.alpha_mw.value
        expected_diagonal = tuple(2.0 * alpha / (d * d) for d in denominator)
        expected_linear = tuple(-(1.0 + 2.0 * alpha * phi / d) for d in denominator)
        expected_matrix = tuple(
            tuple(expected_diagonal[i] if i == j else 0.0 for j in range(n))
            for i in range(n)
        )
        if self.problem.quadratic_matrix.values != expected_matrix:
            raise ValidationError("fairness quadratic coefficients do not match specification and reference")
        if self.problem.linear_objective.values != expected_linear:
            raise ValidationError("fairness linear coefficients do not match specification and reference")
        if abs(self.reference_fraction.value - phi) > 1e-12:
            raise ValidationError("fairness reference fraction does not match request and reference")
        expected_constant = alpha * n * phi * phi
        if abs(self.omitted_constant.value - expected_constant) > 1e-12:
            raise ValidationError("fairness omitted constant does not match request and reference")
        if self.diagnostic_only is not True:
            raise ValidationError("fairness QP bindings cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_json(),
            "request_mw": self.request_mw.to_json(),
            "reference_allocation_mw": self.reference_allocation_mw.to_json(),
            "reference_fraction": self.reference_fraction.to_json(),
            "omitted_constant": self.omitted_constant.to_json(),
            "reference_allocation_hash": self.reference_allocation_hash.to_json(),
            "objective_specification_hash": self.objective_specification_hash.to_json(),
            "objective_specification": self.objective_specification.to_json(),
            "request_source_hash": self.request_source_hash.to_json() if self.request_source_hash else None,
            "domain_hash": self.domain_hash.to_json() if self.domain_hash else None,
            "request_hash": self.request_hash.to_json() if self.request_hash else None,
            "request_vector_hash": self.request_vector_hash.to_json() if self.request_vector_hash else None,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def binding_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    @property
    def problem_hash(self) -> Sha256:
        """Expose the underlying QP hash for generic diagnostic consumers."""
        return self.problem.problem_hash

    @property
    def request_construction_hash(self) -> Sha256 | None:
        """Backward-compatible read-only alias for older diagnostic callers."""
        return self.request_source_hash


def build_diagnostic_fairness_qp_problem(
    domain: FeasibleAllocationDomain,
    specification: FairnessObjectiveSpecification,
    *,
    request_mw: Sequence[float] | RequestConstruction,
    reference_allocation_mw: Sequence[float],
    context_hash: Sha256,
    request_hash: Sha256 | None = None,
    request_source_hash: Sha256 | None = None,
    request_construction_hash: Sha256 | None = None,
) -> DiagnosticFairnessQPBinding:
    """Build the QP equivalent of the frozen fairness equation diagnostically."""
    if not isinstance(specification, FairnessObjectiveSpecification):
        raise ValidationError("specification must be FairnessObjectiveSpecification")
    if not isinstance(context_hash, Sha256):
        raise ValidationError("context_hash must be Sha256")
    _check_domain_alignment(domain, specification.participant_ids)
    _check_domain_hash(domain, specification.feasible_domain_hash)
    if request_hash is not None and not isinstance(request_hash, Sha256):
        raise ValidationError("request_hash must be Sha256 or null")
    if request_source_hash is not None and request_construction_hash is not None and request_source_hash != request_construction_hash:
        raise ValidationError("request source hashes disagree")
    if isinstance(request_mw, RequestConstruction):
        request_source_hash = request_mw.construction_hash
        request = request_mw.actual_request_mw
    elif isinstance(request_mw, FloatVector):
        request = request_mw
    else:
        request = FloatVector(request_mw)
        if request_source_hash is not None and not isinstance(request_source_hash, Sha256):
            raise ValidationError("request_source_hash must be Sha256 or null")
    reference = FloatVector(reference_allocation_mw)
    n = len(domain.participant_ids.values)
    if len(request.values) != n or len(reference.values) != n:
        raise ValidationError("request and reference vectors must match participant IDs")
    if any(value < 0.0 for value in request.values + reference.values):
        raise ValidationError("request and reference vectors must be nonnegative")
    request_vector_hash = Sha256(canonical_hash({"participant_ids": domain.participant_ids.to_json(), "values_mw": request.to_json()}))
    denominator = tuple(value + specification.epsilon_mw.value for value in request.values)
    phi = math.fsum(value / d for value, d in zip(reference.values, denominator)) / n
    alpha = specification.alpha_mw.value
    diagonal = tuple(2.0 * alpha / (d * d) for d in denominator)
    linear = tuple(-(1.0 + 2.0 * alpha * phi / d) for d in denominator)
    constant = alpha * n * phi * phi
    reference_hash = Sha256(canonical_hash({"participant_ids": domain.participant_ids.to_json(), "values_mw": reference.to_json()}))
    if specification.reference_mode == "EXTERNAL_REFERENCE":
        if specification.external_reference_allocation_hash != reference_hash:
            raise ValidationError("external fairness reference hash does not match allocation")
    elif specification.reference_mode == "ENDOGENOUS_MAX_EXPORT" and specification.max_export_specification_hash is None:
        raise ValidationError("endogenous fairness reference requires max-export specification hash")
    problem = DiagnosticQuadraticProblem(
        variable_ids=domain.participant_ids,
        quadratic_matrix=FloatMatrix(tuple(tuple(diagonal[i] if i == j else 0.0 for j in range(n)) for i in range(n))),
        linear_objective=FloatVector(linear),
        constraints=_diagnostic_rows(domain),
        lower_bounds=domain.lower_bounds_mw,
        upper_bounds=domain.upper_bounds_mw,
        context_hash=context_hash,
        objective_specification_hash=specification.specification_hash,
        objective_family="FAIRNESS",
        objective_stage="FAIRNESS_QP",
    )
    return DiagnosticFairnessQPBinding(
        problem=problem,
        request_mw=request,
        reference_allocation_mw=reference,
        reference_fraction=FiniteFloat(phi),
        omitted_constant=FiniteFloat(constant),
        reference_allocation_hash=reference_hash,
        objective_specification_hash=specification.specification_hash,
        objective_specification=specification,
        request_source_hash=request_source_hash,
        domain_hash=domain.domain_hash,
        request_hash=request_hash,
        request_vector_hash=request_vector_hash,
    )


__all__ = [
    "DiagnosticMaxExportSolveResult",
    "DiagnosticFairnessQPBinding",
    "DiagnosticMaxExportStage2ProblemBinding",
    "build_diagnostic_fairness_qp_problem",
    "build_diagnostic_max_export_stage1_problem",
    "build_diagnostic_max_export_stage2_problem",
    "solve_diagnostic_max_export",
]
