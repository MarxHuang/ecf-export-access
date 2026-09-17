"""Bind a diagnostic solver result to an allocation without mislabeling it.

The caller-supplied ``build_explicit_allocation`` path deliberately rejects
optimization objective IDs.  This module is the separate diagnostic path for
LP/QP results: it retains the actual problem, objective specification, solver
result and typed capacity object, then verifies all derived hashes before an
allocation can be consumed by later diagnostic adapters.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from r4r.allocation_pipeline import (
    AllocationBoundPolicy,
    ExplicitCapacityVector,
)
from r4r.errors import ValidationError
from r4r.models import AllocationVector
from r4r.objective_contracts import (
    FairnessObjectiveSpecification,
    MaxExportObjectiveSpecification,
)
from r4r.request_contracts import AdmittedRequestVector
from r4r.serialization import canonical_bytes, canonical_hash
from r4r.types import (
    AllocationSide,
    FloatVector,
    Identifier,
    IdentifierVector,
    NormalizedSolverStatus,
    Sha256,
)
from tools.diagnostic_optimization import (
    DiagnosticLinearProblem,
    DiagnosticQuadraticProblem,
    DiagnosticSolveResult,
    is_solver_emitted_result,
)
from tools.diagnostic_objective_problems import (
    DiagnosticFairnessQPBinding,
    DiagnosticMaxExportStage2ProblemBinding,
)


ObjectiveSpecification = MaxExportObjectiveSpecification | FairnessObjectiveSpecification
DiagnosticProblem = DiagnosticLinearProblem | DiagnosticQuadraticProblem
ObjectiveProblemBinding = DiagnosticLinearProblem | DiagnosticMaxExportStage2ProblemBinding | DiagnosticFairnessQPBinding
ADMITTED_REQUEST_BOUND_TOLERANCE_MW = 1e-6
# LP/QP solutions are retained exactly as emitted.  This tolerance is used
# only to distinguish a floating-point boundary residual from a real capacity
# violation; no allocation value is clipped or rewritten.
SOLVER_BOUND_TOLERANCE_MW = 1e-9


def _problem_from_binding(problem: ObjectiveProblemBinding) -> DiagnosticLinearProblem | DiagnosticQuadraticProblem:
    return problem.problem if isinstance(
        problem, (DiagnosticMaxExportStage2ProblemBinding, DiagnosticFairnessQPBinding)
    ) else problem


def _problem_binding_hash(problem: ObjectiveProblemBinding) -> Sha256:
    """Return the hash of the full typed problem binding, not only its LP/QP."""
    if isinstance(problem, (DiagnosticMaxExportStage2ProblemBinding, DiagnosticFairnessQPBinding)):
        return problem.binding_hash
    return problem.problem_hash


def _validate_objective_problem_binding(
    problem: ObjectiveProblemBinding,
    objective_specification: ObjectiveSpecification,
) -> None:
    """Prove that the typed objective specification matches the actual problem."""
    actual_problem = _problem_from_binding(problem)
    if actual_problem.variable_ids != objective_specification.participant_ids:
        raise ValidationError("objective, problem and allocation participant order do not match")
    if actual_problem.objective_specification_hash != objective_specification.specification_hash:
        raise ValidationError("diagnostic problem is not bound to this objective specification")
    if isinstance(objective_specification, MaxExportObjectiveSpecification):
        if isinstance(problem, DiagnosticMaxExportStage2ProblemBinding):
            if problem.objective_specification != objective_specification:
                raise ValidationError("stage2 problem wrapper objective does not match specification")
            return
        if not isinstance(problem, DiagnosticLinearProblem):
            raise ValidationError("max-export objective requires a diagnostic linear problem")
        if problem.objective_family != "MAX_EXPORT":
            raise ValidationError("max-export problem objective family is not registered")
        if problem.objective_stage == "MAX_EXPORT_STAGE1":
            expected = (1.0,) * len(objective_specification.participant_ids.values)
            if problem.objective.values != expected or problem.objective_sense != "MAXIMIZE":
                raise ValidationError("MAX_EXPORT_STAGE1 problem objective coefficients are inconsistent")
        elif problem.objective_stage == "MAX_EXPORT_STAGE2":
            raise ValidationError("MAX_EXPORT_STAGE2 requires its typed problem binding")
        else:
            raise ValidationError("max-export problem objective stage is not registered")
    elif isinstance(objective_specification, FairnessObjectiveSpecification):
        if not isinstance(problem, DiagnosticFairnessQPBinding):
            raise ValidationError("fairness objective requires its typed QP binding")
        if problem.objective_specification != objective_specification:
            raise ValidationError("fairness QP wrapper objective does not match specification")


def _solver_result_hash(result: DiagnosticSolveResult) -> Sha256:
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


def _allocation_hash(
    *,
    participant_ids: IdentifierVector,
    values_mw: tuple[float, ...],
    side: AllocationSide,
    scenario_id: Identifier,
    request_hash: Sha256 | None,
    capacity_hash: Sha256,
    objective_id: Identifier,
    bound_policy: AllocationBoundPolicy,
    participant_registry_hash: Sha256,
    capacity_vector_hash: Sha256,
    problem_hash: Sha256,
    problem_binding_hash: Sha256,
    objective_specification_hash: Sha256,
    solver_result_hash: Sha256,
    solver_name: Identifier,
) -> Sha256:
    payload = {
        "origin": "SOLVER_PRODUCED",
        "participant_ids": participant_ids.to_json(),
        "values_mw": list(values_mw),
        "side": side.value,
        "scenario_id": scenario_id.value,
        "request_hash": request_hash.to_json() if request_hash else None,
        "capacity_hash": capacity_hash.to_json(),
        "objective_id": objective_id.to_json(),
        "bound_policy": bound_policy,
        "participant_registry_hash": participant_registry_hash.to_json(),
        "capacity_vector_hash": capacity_vector_hash.to_json(),
        "problem_hash": problem_hash.to_json(),
        "problem_binding_hash": problem_binding_hash.to_json(),
        "objective_specification_hash": objective_specification_hash.to_json(),
        "solver_result_hash": solver_result_hash.to_json(),
        "solver_name": solver_name.to_json(),
    }
    return Sha256(hashlib.sha256(canonical_bytes(payload)).hexdigest())


@dataclass(frozen=True, slots=True)
class DiagnosticSolverAllocationBinding:
    """A solver-produced allocation with complete diagnostic provenance."""

    allocation: AllocationVector
    bound_policy: AllocationBoundPolicy
    capacity_vector: ExplicitCapacityVector
    problem: ObjectiveProblemBinding
    objective_specification: ObjectiveSpecification
    solver_result: DiagnosticSolveResult
    # Keep the complete typed admission record.  Storing only a vector/hash
    # pair would let a direct constructor forge mutually-consistent fields
    # while pointing at a different admission contract hash.
    admitted_request: AdmittedRequestVector | None = None
    serialization_id = "diagnostic_solver_allocation_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.allocation, AllocationVector):
            raise ValidationError("allocation must be an AllocationVector")
        if self.bound_policy not in {"NONE", "ADMITTED_REQUEST"}:
            raise ValidationError("allocation bound policy is not registered")
        if not isinstance(self.capacity_vector, ExplicitCapacityVector):
            raise ValidationError("solver binding requires an ExplicitCapacityVector")
        if not isinstance(
            self.problem,
            (DiagnosticLinearProblem, DiagnosticMaxExportStage2ProblemBinding, DiagnosticFairnessQPBinding),
        ):
            raise ValidationError("solver binding requires a typed diagnostic LP or QP problem binding")
        if not isinstance(self.objective_specification, (MaxExportObjectiveSpecification, FairnessObjectiveSpecification)):
            raise ValidationError("solver binding requires a registered objective specification")
        if not isinstance(self.solver_result, DiagnosticSolveResult):
            raise ValidationError("solver binding requires a DiagnosticSolveResult")
        _validate_objective_problem_binding(self.problem, self.objective_specification)
        if isinstance(self.problem, DiagnosticFairnessQPBinding) and self.problem.request_hash != self.allocation.request_hash:
            raise ValidationError("fairness QP request provenance does not match solver allocation")
        if self.bound_policy == "ADMITTED_REQUEST":
            if not isinstance(self.admitted_request, AdmittedRequestVector):
                raise ValidationError("ADMITTED_REQUEST solver binding requires a typed admitted request")
            admitted = self.admitted_request
            if admitted.participant_ids != self.allocation.participant_ids:
                raise ValidationError("admitted request participant order does not match allocation")
            if admitted.source_side is not self.allocation.side:
                raise ValidationError("admitted request side does not match allocation")
            if admitted.participant_registry_hash != self.capacity_vector.participant_registry_hash:
                raise ValidationError("admitted request registry does not match capacity vector")
            if admitted.capacity_spec_hash != self.capacity_vector.capacity_spec_hash:
                raise ValidationError("admitted request capacity specification does not match capacity vector")
            if self.allocation.request_hash != admitted.request_hash:
                raise ValidationError("solver allocation does not reference the supplied admitted request")
            if any(
                value > request + ADMITTED_REQUEST_BOUND_TOLERANCE_MW
                for value, request in zip(
                    self.allocation.values_mw.values,
                    admitted.admitted_values_mw.values,
                )
            ):
                raise ValidationError("solver allocation exceeds the explicit admitted request bound")
            if isinstance(self.problem, DiagnosticFairnessQPBinding):
                if self.problem.request_hash != admitted.request_hash:
                    raise ValidationError("fairness QP request provenance does not match admitted request")
                expected_vector_hash = Sha256(canonical_hash({
                    "participant_ids": self.problem.problem.variable_ids.to_json(),
                    "values_mw": self.problem.request_mw.to_json(),
                }))
                if any(
                    value > request + ADMITTED_REQUEST_BOUND_TOLERANCE_MW
                    for value, request in zip(self.problem.request_mw.values, admitted.admitted_values_mw.values)
                ):
                    raise ValidationError("fairness QP request vector exceeds admitted request")
                if self.problem.request_vector_hash != expected_vector_hash:
                    raise ValidationError("fairness QP request vector hash differs from request vector")
        elif self.admitted_request is not None:
            raise ValidationError("admitted request is only valid with ADMITTED_REQUEST policy")
        if self.allocation.objective_id != self.objective_specification.objective_id:
            raise ValidationError("allocation objective does not match objective specification")
        if self.allocation.participant_ids != self.capacity_vector.participant_ids:
            raise ValidationError("solver allocation and capacity participant order do not match")
        if self.allocation.capacity_hash != self.capacity_vector.capacity_spec_hash:
            raise ValidationError("solver allocation capacity does not match capacity vector")
        actual_problem = _problem_from_binding(self.problem)
        if self.solver_result.problem_hash != actual_problem.problem_hash:
            raise ValidationError("solver result does not belong to the supplied problem")
        if self.solver_result.variable_ids != self.allocation.participant_ids:
            raise ValidationError("solver result variable order does not match allocation")
        if self.solver_result.normalized_status is not NormalizedSolverStatus.PASS or self.solver_result.solution is None:
            raise ValidationError("only successful diagnostic solver results can bind an allocation")
        if self.allocation.values_mw != self.solver_result.solution:
            raise ValidationError("solver allocation values do not match solver result solution")
        if any(
            value < -SOLVER_BOUND_TOLERANCE_MW or value > capacity + SOLVER_BOUND_TOLERANCE_MW
            for value, capacity in zip(
                self.solver_result.solution.values,
                self.capacity_vector.values_mw.values,
            )
        ):
            raise ValidationError("solver allocation violates explicit capacity bounds")
        if self.solver_result.result_hash != _solver_result_hash(self.solver_result):
            raise ValidationError("solver result hash does not match its contents")
        if not is_solver_emitted_result(self.solver_result):
            raise ValidationError("solver result was not emitted by a diagnostic solver")
        expected_hash = _allocation_hash(
            participant_ids=self.allocation.participant_ids,
            values_mw=self.allocation.values_mw.values,
            side=self.allocation.side,
            scenario_id=self.allocation.scenario_id,
            request_hash=self.allocation.request_hash,
            capacity_hash=self.allocation.capacity_hash,  # type: ignore[arg-type]
            objective_id=self.allocation.objective_id,  # type: ignore[arg-type]
            bound_policy=self.bound_policy,
            participant_registry_hash=self.capacity_vector.participant_registry_hash,
            capacity_vector_hash=self.capacity_vector.capacity_vector_hash,
            problem_hash=actual_problem.problem_hash,
            problem_binding_hash=_problem_binding_hash(self.problem),
            objective_specification_hash=self.objective_specification.specification_hash,
            solver_result_hash=self.solver_result.result_hash,
            solver_name=self.solver_result.solver_name,
        )
        if self.allocation.allocation_hash != expected_hash:
            raise ValidationError("solver allocation hash does not match complete provenance")

    @property
    def participant_ids(self) -> IdentifierVector:
        return self.allocation.participant_ids

    @property
    def values_mw(self):
        return self.allocation.values_mw

    @property
    def side(self) -> AllocationSide:
        return self.allocation.side

    @property
    def scenario_id(self) -> Identifier:
        """Scenario identity carried by the solver-produced allocation."""

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
    def participant_registry_hash(self) -> Sha256:
        return self.capacity_vector.participant_registry_hash

    @property
    def capacity_vector_hash(self) -> Sha256:
        return self.capacity_vector.capacity_vector_hash

    @property
    def objective_id(self) -> Identifier:
        """Registered objective identity exposed to typed profile bindings."""

        return self.objective_specification.objective_id

    @property
    def domain_hash(self) -> Sha256 | None:
        """Exact feasible-domain identity bound into the objective/problem.

        Candidate objective specifications may intentionally omit this identity;
        such allocations remain usable for isolated solver unit diagnostics but
        cannot be wrapped as a profile for proxy/AC comparison.
        """

        if isinstance(self.problem, DiagnosticFairnessQPBinding):
            return self.problem.domain_hash
        return self.objective_specification.feasible_domain_hash

    def to_json(self) -> dict[str, Any]:
        return {
            "allocation": self.allocation.to_json(),
            "bound_policy": self.bound_policy,
            "capacity_vector": self.capacity_vector.to_json(),
            "problem_hash": _problem_from_binding(self.problem).problem_hash.to_json(),
            "problem_binding_hash": _problem_binding_hash(self.problem).to_json(),
            "objective_specification_hash": self.objective_specification.specification_hash.to_json(),
            "domain_hash": self.domain_hash.to_json() if self.domain_hash else None,
            "solver_result_hash": self.solver_result.result_hash.to_json(),
            "solver_name": self.solver_result.solver_name.to_json(),
            "admitted_request": self.admitted_request.to_json() if self.admitted_request else None,
        }

    @property
    def admitted_request_values_mw(self) -> FloatVector | None:
        """Backward-compatible view derived from the typed admission record."""

        return self.admitted_request.admitted_values_mw if self.admitted_request else None

    @property
    def admitted_request_vector_hash(self) -> Sha256 | None:
        """Hash of the admitted numeric vector, derived without caller input."""

        if self.admitted_request is None:
            return None
        return Sha256(canonical_hash({
            "participant_ids": self.admitted_request.participant_ids.to_json(),
            "values_mw": self.admitted_request.admitted_values_mw.to_json(),
        }))


def bind_diagnostic_solver_allocation(
    *,
    problem: ObjectiveProblemBinding,
    objective_specification: ObjectiveSpecification,
    solver_result: DiagnosticSolveResult,
    participant_ids: IdentifierVector,
    side: AllocationSide,
    scenario_id: Identifier,
    capacity_vector: ExplicitCapacityVector,
    admitted_request: AdmittedRequestVector | None,
    bound_policy: AllocationBoundPolicy,
) -> DiagnosticSolverAllocationBinding:
    """Create a solver-produced binding after checking a successful solve."""

    if not isinstance(participant_ids, IdentifierVector):
        raise ValidationError("participant_ids must be an IdentifierVector")
    if solver_result.normalized_status is not NormalizedSolverStatus.PASS or solver_result.solution is None:
        raise ValidationError("cannot bind an unsuccessful diagnostic solver result")
    if not is_solver_emitted_result(solver_result):
        raise ValidationError("solver result was not emitted by a diagnostic solver")
    if solver_result.variable_ids != participant_ids:
        raise ValidationError("solver result variable order does not match participant IDs")
    _validate_objective_problem_binding(problem, objective_specification)
    if capacity_vector.participant_ids != participant_ids:
        raise ValidationError("capacity vector participant order does not match participant IDs")
    if any(
        value < -SOLVER_BOUND_TOLERANCE_MW or value > capacity + SOLVER_BOUND_TOLERANCE_MW
        for value, capacity in zip(solver_result.solution.values, capacity_vector.values_mw.values)
    ):
        raise ValidationError("solver allocation violates explicit capacity bounds")
    if bound_policy == "ADMITTED_REQUEST":
        if admitted_request is None:
            raise ValidationError("ADMITTED_REQUEST solver binding requires an admitted request")
        if admitted_request.source_side is not side:
            raise ValidationError("solver allocation side does not match admitted request")
        if admitted_request.participant_ids != participant_ids:
            raise ValidationError("admitted request participant order does not match solver result")
        if admitted_request.participant_registry_hash is None:
            raise ValidationError("solver binding requires request registry provenance")
        if admitted_request.participant_registry_hash != capacity_vector.participant_registry_hash:
            raise ValidationError("request and capacity registry provenance do not match")
        if admitted_request.capacity_spec_hash != capacity_vector.capacity_spec_hash:
            raise ValidationError("request and capacity specification provenance do not match")
        if any(
            value > request + ADMITTED_REQUEST_BOUND_TOLERANCE_MW
            for value, request in zip(solver_result.solution.values, admitted_request.admitted_values_mw.values)
        ):
            raise ValidationError("solver allocation exceeds the explicit admitted request bound")
        request_hash = admitted_request.request_hash
        if isinstance(problem, DiagnosticFairnessQPBinding):
            if problem.request_hash != admitted_request.request_hash:
                raise ValidationError("fairness QP admitted-request hash mismatch")
            if any(
                value > admitted + ADMITTED_REQUEST_BOUND_TOLERANCE_MW
                for value, admitted in zip(problem.request_mw.values, admitted_request.admitted_values_mw.values)
            ):
                raise ValidationError("fairness QP request vector exceeds admitted request")
            expected_vector_hash = Sha256(canonical_hash({
                "participant_ids": problem.problem.variable_ids.to_json(),
                "values_mw": problem.request_mw.to_json(),
            }))
            if problem.request_vector_hash != expected_vector_hash:
                raise ValidationError("fairness QP request-vector hash differs from request vector")
    else:
        request_hash = None
    if capacity_vector.capacity_spec_hash is None:
        raise ValidationError("solver binding requires capacity specification provenance")
    allocation_hash = _allocation_hash(
        participant_ids=participant_ids,
        values_mw=solver_result.solution.values,
        side=side,
        scenario_id=scenario_id,
        request_hash=request_hash,
        capacity_hash=capacity_vector.capacity_spec_hash,
        objective_id=objective_specification.objective_id,
        bound_policy=bound_policy,
        participant_registry_hash=capacity_vector.participant_registry_hash,
        capacity_vector_hash=capacity_vector.capacity_vector_hash,
        problem_hash=_problem_from_binding(problem).problem_hash,
        problem_binding_hash=_problem_binding_hash(problem),
        objective_specification_hash=objective_specification.specification_hash,
        solver_result_hash=solver_result.result_hash,
        solver_name=solver_result.solver_name,
    )
    allocation = AllocationVector(
        participant_ids=participant_ids,
        values_mw=solver_result.solution,
        vector_length=len(participant_ids.values),
        side=side,
        scenario_id=scenario_id,
        allocation_hash=allocation_hash,
        request_hash=request_hash,
        capacity_hash=capacity_vector.capacity_spec_hash,
        objective_id=objective_specification.objective_id,
    )
    return DiagnosticSolverAllocationBinding(
        allocation=allocation,
        bound_policy=bound_policy,
        capacity_vector=capacity_vector,
        problem=problem,
        objective_specification=objective_specification,
        solver_result=solver_result,
        admitted_request=admitted_request,
    )


__all__ = ["DiagnosticSolverAllocationBinding", "bind_diagnostic_solver_allocation"]
