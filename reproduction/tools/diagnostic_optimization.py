"""Generic diagnostic LP/QP kernel, isolated from the R4R business path.

This module is deliberately not imported by ``src/r4r``.  It accepts explicit
variable and constraint contracts, invokes numerical adapters only for matrix
and solver diagnostics, and returns records whose status is permanently
``DIAGNOSTIC_ONLY``.  It never creates a canonical scenario or paper evidence.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models import ProxyModel
from r4r.proxy_evaluator import ProxyEvaluationSpecification
from r4r.serialization import canonical_hash
from r4r.types import (
    FiniteFloat,
    FloatMatrix,
    FloatVector,
    Identifier,
    IdentifierVector,
    NormalizedSolverStatus,
    RawSolverStatus,
    Sha256,
)


_SENSES = {"LESS_EQUAL", "GREATER_EQUAL", "EQUAL"}
_OBJECTIVE_SENSES = {"MINIMIZE", "MAXIMIZE"}
_OBJECTIVE_FAMILIES = {"MAX_EXPORT", "FAIRNESS"}
_OBJECTIVE_STAGES = {"MAX_EXPORT_STAGE1", "MAX_EXPORT_STAGE2", "FAIRNESS_QP"}
# Solver-produced records are issued by the numerical adapter in this
# process.  Downstream provenance bindings use this registry in addition to
# content hashes so a caller cannot manufacture a new result and simply
# recompute its hash while claiming it came from a solver execution.
_EMITTED_RESULT_HASHES: set[str] = set()


def _validate_objective_metadata(
    objective_specification_hash: Sha256 | None,
    objective_family: str | None,
    objective_stage: str | None,
) -> None:
    """Validate the optional typed objective-to-problem provenance.

    Generic diagnostic problems remain usable without objective metadata.  A
    problem produced by a registered objective adapter, however, must carry
    all three fields so a later solver binding cannot merely attach an
    unrelated objective specification hash.
    """
    if objective_specification_hash is not None and not isinstance(objective_specification_hash, Sha256):
        raise ValidationError("objective_specification_hash must be Sha256 or null")
    if objective_family is not None and objective_family not in _OBJECTIVE_FAMILIES:
        raise ValidationError("objective_family is not registered")
    if objective_stage is not None and objective_stage not in _OBJECTIVE_STAGES:
        raise ValidationError("objective_stage is not registered")
    present = (
        objective_specification_hash is not None,
        objective_family is not None,
        objective_stage is not None,
    )
    if any(present) and not all(present):
        raise ValidationError("objective metadata must be complete or absent")
    if objective_family == "MAX_EXPORT" and objective_stage not in {"MAX_EXPORT_STAGE1", "MAX_EXPORT_STAGE2"}:
        raise ValidationError("MAX_EXPORT objective metadata has an invalid stage")
    if objective_family == "FAIRNESS" and objective_stage != "FAIRNESS_QP":
        raise ValidationError("FAIRNESS objective metadata has an invalid stage")


def build_diagnostic_lp_from_proxy_model(
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    *,
    variable_ids: IdentifierVector,
    objective: FloatVector,
    lower_bounds: FloatVector,
    upper_bounds: FloatVector,
    objective_sense: str,
    context_hash: Sha256,
    branch_sense: str,
    voltage_sense: str,
) -> "DiagnosticLinearProblem":
    """Translate explicit proxy rows into a diagnostic LP contract.

    Constraint senses are mandatory inputs.  This adapter deliberately does
    not infer how ``SIGNED_*`` or ``ABSOLUTE_MAGNITUDE`` proxy modes should be
    expanded, so an unresolved scientific sign convention cannot be hidden in
    a generic optimization helper.
    """

    if not isinstance(proxy_model, ProxyModel) or not isinstance(proxy_specification, ProxyEvaluationSpecification):
        raise ValidationError("proxy model and specification are required")
    if variable_ids != proxy_specification.participant_ids:
        raise ValidationError("diagnostic LP variable IDs must match proxy participant order")
    if branch_sense not in _SENSES or voltage_sense not in _SENSES:
        raise ValidationError("proxy diagnostic constraint senses are not registered")
    n = len(variable_ids.values)
    if len(proxy_model.branch_sensitivity_matrix.values) != len(proxy_specification.branch_constraint_ids.values):
        raise ValidationError("branch proxy rows do not match specification IDs")
    if len(proxy_model.voltage_sensitivity_matrix.values) != len(proxy_specification.voltage_constraint_ids.values):
        raise ValidationError("voltage proxy rows do not match specification IDs")
    if any(len(row) != n for row in (*proxy_model.branch_sensitivity_matrix.values, *proxy_model.voltage_sensitivity_matrix.values)):
        raise ValidationError("proxy matrix columns do not match variable IDs")
    rows = tuple(
        [
            DiagnosticConstraintRow(
                constraint_id=constraint_id,
                coefficients=FloatVector(coefficients),
                rhs=FiniteFloat(rhs),
                sense=branch_sense,
            )
            for constraint_id, coefficients, rhs in zip(
                proxy_specification.branch_constraint_ids.values,
                proxy_model.branch_sensitivity_matrix.values,
                proxy_model.branch_headroom_mw.values,
            )
        ]
        + [
            DiagnosticConstraintRow(
                constraint_id=constraint_id,
                coefficients=FloatVector(coefficients),
                rhs=FiniteFloat(rhs),
                sense=voltage_sense,
            )
            for constraint_id, coefficients, rhs in zip(
                proxy_specification.voltage_constraint_ids.values,
                proxy_model.voltage_sensitivity_matrix.values,
                proxy_model.voltage_headroom_pu.values,
            )
        ]
    )
    return DiagnosticLinearProblem(
        variable_ids=variable_ids,
        objective=objective,
        constraints=rows,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_sense=objective_sense,
        context_hash=context_hash,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticConstraintRow(ContractModel):
    constraint_id: Identifier
    coefficients: FloatVector
    rhs: FiniteFloat
    sense: str
    serialization_id = "diagnostic_constraint_row.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, Identifier):
            raise ValidationError("constraint_id must be Identifier")
        if not isinstance(self.coefficients, FloatVector) or not self.coefficients.values:
            raise ValidationError("constraint coefficients must be a non-empty FloatVector")
        if self.sense not in _SENSES:
            raise ValidationError("constraint sense is not registered")

    def to_json(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id.to_json(),
            "coefficients": self.coefficients.to_json(),
            "rhs": self.rhs.to_json(),
            "sense": self.sense,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticLinearProblem(ContractModel):
    variable_ids: IdentifierVector
    objective: FloatVector
    constraints: tuple[DiagnosticConstraintRow, ...]
    lower_bounds: FloatVector
    upper_bounds: FloatVector
    objective_sense: str
    context_hash: Sha256
    objective_specification_hash: Sha256 | None = None
    objective_family: str | None = None
    objective_stage: str | None = None
    serialization_id = "diagnostic_linear_problem.v1"

    def __post_init__(self) -> None:
        _validate_problem_fields(
            self.variable_ids,
            self.objective,
            self.constraints,
            self.lower_bounds,
            self.upper_bounds,
            self.objective_sense,
            self.context_hash,
        )
        _validate_objective_metadata(
            self.objective_specification_hash,
            self.objective_family,
            self.objective_stage,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "variable_ids": self.variable_ids.to_json(),
            "objective": self.objective.to_json(),
            "constraints": [row.to_json() for row in self.constraints],
            "lower_bounds": self.lower_bounds.to_json(),
            "upper_bounds": self.upper_bounds.to_json(),
            "objective_sense": self.objective_sense,
            "context_hash": self.context_hash.to_json(),
            "objective_specification_hash": (
                self.objective_specification_hash.to_json()
                if self.objective_specification_hash else None
            ),
            "objective_family": self.objective_family,
            "objective_stage": self.objective_stage,
        }

    @property
    def problem_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class DiagnosticQuadraticProblem(ContractModel):
    variable_ids: IdentifierVector
    quadratic_matrix: FloatMatrix
    linear_objective: FloatVector
    constraints: tuple[DiagnosticConstraintRow, ...]
    lower_bounds: FloatVector
    upper_bounds: FloatVector
    context_hash: Sha256
    objective_specification_hash: Sha256 | None = None
    objective_family: str | None = None
    objective_stage: str | None = None
    serialization_id = "diagnostic_quadratic_problem.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.variable_ids, IdentifierVector) or not self.variable_ids.values:
            raise ValidationError("variable_ids must be a non-empty IdentifierVector")
        n = len(self.variable_ids.values)
        if not isinstance(self.quadratic_matrix, FloatMatrix) or len(self.quadratic_matrix.values) != n or any(len(row) != n for row in self.quadratic_matrix.values):
            raise ValidationError("quadratic matrix must be square and match variable count")
        if any(abs(a - b) > 1e-12 for row_a, row_b in zip(self.quadratic_matrix.values, zip(*self.quadratic_matrix.values)) for a, b in zip(row_a, row_b)):
            raise ValidationError("quadratic matrix must be symmetric")
        _validate_problem_fields(
            self.variable_ids,
            self.linear_objective,
            self.constraints,
            self.lower_bounds,
            self.upper_bounds,
            "MINIMIZE",
            self.context_hash,
        )
        _validate_objective_metadata(
            self.objective_specification_hash,
            self.objective_family,
            self.objective_stage,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "variable_ids": self.variable_ids.to_json(),
            "quadratic_matrix": self.quadratic_matrix.to_json(),
            "linear_objective": self.linear_objective.to_json(),
            "constraints": [row.to_json() for row in self.constraints],
            "lower_bounds": self.lower_bounds.to_json(),
            "upper_bounds": self.upper_bounds.to_json(),
            "context_hash": self.context_hash.to_json(),
            "objective_specification_hash": (
                self.objective_specification_hash.to_json()
                if self.objective_specification_hash else None
            ),
            "objective_family": self.objective_family,
            "objective_stage": self.objective_stage,
        }

    @property
    def problem_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class DiagnosticSolveResult(ContractModel):
    problem_hash: Sha256
    solver_name: Identifier
    raw_status: RawSolverStatus
    normalized_status: NormalizedSolverStatus
    diagnostic_only: bool
    variable_ids: IdentifierVector
    solution: FloatVector | None
    objective_value: FiniteFloat | None
    constraint_residuals: FloatVector | None
    failure_reason: Identifier | None
    result_hash: Sha256
    serialization_id = "diagnostic_solve_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.problem_hash, Sha256) or not isinstance(self.result_hash, Sha256):
            raise ValidationError("problem and result hashes must be Sha256")
        if not isinstance(self.solver_name, Identifier):
            raise ValidationError("solver_name must be Identifier")
        if not isinstance(self.raw_status, RawSolverStatus) or not isinstance(self.normalized_status, NormalizedSolverStatus):
            raise ValidationError("solver statuses must be registered enums")
        if self.diagnostic_only is not True:
            raise ValidationError("diagnostic solver results cannot be promoted")
        if not isinstance(self.variable_ids, IdentifierVector) or not self.variable_ids.values:
            raise ValidationError("variable_ids must be a non-empty IdentifierVector")
        if self.solution is not None and len(self.solution.values) != len(self.variable_ids.values):
            raise ValidationError("solution shape must match variable IDs")
        if self.constraint_residuals is not None and not isinstance(self.constraint_residuals, FloatVector):
            raise ValidationError("constraint_residuals must be FloatVector or null")
        if self.failure_reason is not None and not isinstance(self.failure_reason, Identifier):
            raise ValidationError("failure_reason must be Identifier or null")

    def to_json(self) -> dict[str, Any]:
        return {
            "problem_hash": self.problem_hash.to_json(),
            "solver_name": self.solver_name.to_json(),
            "raw_status": self.raw_status.value,
            "normalized_status": self.normalized_status.value,
            "diagnostic_only": self.diagnostic_only,
            "variable_ids": self.variable_ids.to_json(),
            "solution": self.solution.to_json() if self.solution else None,
            "objective_value": self.objective_value.to_json() if self.objective_value else None,
            "constraint_residuals": self.constraint_residuals.to_json() if self.constraint_residuals else None,
            "failure_reason": self.failure_reason.to_json() if self.failure_reason else None,
            "result_hash": self.result_hash.to_json(),
        }


def _validate_problem_fields(variable_ids, objective, constraints, lower_bounds, upper_bounds, objective_sense, context_hash) -> None:
    if not isinstance(variable_ids, IdentifierVector) or not variable_ids.values:
        raise ValidationError("variable_ids must be a non-empty IdentifierVector")
    if not isinstance(objective, FloatVector):
        raise ValidationError("objective must be FloatVector")
    n = len(variable_ids.values)
    if len(objective.values) != n or len(lower_bounds.values) != n or len(upper_bounds.values) != n:
        raise ValidationError("objective and variable bounds must match variable count")
    if objective_sense not in _OBJECTIVE_SENSES:
        raise ValidationError("objective sense is not registered")
    if not isinstance(lower_bounds, FloatVector) or not isinstance(upper_bounds, FloatVector):
        raise ValidationError("variable bounds must be FloatVector values")
    if any(lower > upper for lower, upper in zip(lower_bounds.values, upper_bounds.values)):
        raise ValidationError("lower bound cannot exceed upper bound")
    if not isinstance(context_hash, Sha256):
        raise ValidationError("context_hash must be Sha256")
    if any(not isinstance(row, DiagnosticConstraintRow) for row in constraints):
        raise ValidationError("constraints must contain DiagnosticConstraintRow values")
    if any(len(row.coefficients.values) != n for row in constraints):
        raise ValidationError("constraint coefficients must match variable count")
    if len({row.constraint_id for row in constraints}) != len(constraints):
        raise ValidationError("constraint IDs must be unique")


def _constraint_residuals(rows: tuple[DiagnosticConstraintRow, ...], solution: tuple[float, ...]) -> FloatVector:
    residuals = []
    for row in rows:
        lhs = sum(coefficient * value for coefficient, value in zip(row.coefficients.values, solution))
        if row.sense == "LESS_EQUAL":
            residuals.append(row.rhs.value - lhs)
        elif row.sense == "GREATER_EQUAL":
            residuals.append(lhs - row.rhs.value)
        else:
            residuals.append(lhs - row.rhs.value)
    return FloatVector(residuals)


def _result(problem_hash, solver_name, raw_status, normalized_status, variable_ids, solution, objective_value, residuals, failure_reason) -> DiagnosticSolveResult:
    payload = {
        "problem_hash": problem_hash.to_json(),
        "solver_name": solver_name.to_json(),
        "raw_status": raw_status.value,
        "normalized_status": normalized_status.value,
        "variable_ids": variable_ids.to_json(),
        "solution": solution.to_json() if solution else None,
        "objective_value": objective_value.to_json() if objective_value else None,
        "constraint_residuals": residuals.to_json() if residuals else None,
        "failure_reason": failure_reason.to_json() if failure_reason else None,
    }
    result = DiagnosticSolveResult(
        problem_hash=problem_hash,
        solver_name=solver_name,
        raw_status=raw_status,
        normalized_status=normalized_status,
        diagnostic_only=True,
        variable_ids=variable_ids,
        solution=solution,
        objective_value=objective_value,
        constraint_residuals=residuals,
        failure_reason=failure_reason,
        result_hash=Sha256(canonical_hash(payload)),
    )
    _EMITTED_RESULT_HASHES.add(result.result_hash.value)
    return result


def is_solver_emitted_result(result: DiagnosticSolveResult) -> bool:
    """Return whether a result was emitted by a diagnostic solver adapter.

    The content hash proves integrity; the in-process issuance registry proves
    that the record originated from ``solve_diagnostic_lp``/``solve_diagnostic_qp``
    during this diagnostic execution rather than being newly manufactured by a
    caller.  This is intentionally diagnostic provenance, not manuscript
    evidence or a publication security mechanism.
    """
    if not isinstance(result, DiagnosticSolveResult):
        return False
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
    expected = Sha256(canonical_hash(payload))
    return result.result_hash == expected and result.result_hash.value in _EMITTED_RESULT_HASHES


def solve_diagnostic_lp(problem: DiagnosticLinearProblem) -> DiagnosticSolveResult:
    """Run a caller-specified LP through SciPy/HiGHS for numerical diagnostics."""

    try:
        from scipy.optimize import linprog
    except Exception:
        return _result(problem.problem_hash, Identifier("SCIPY_HIGHS"), RawSolverStatus.ERROR, NormalizedSolverStatus.UNAVAILABLE, problem.variable_ids, None, None, None, Identifier("DEPENDENCY_UNAVAILABLE"))
    c = list(problem.objective.values)
    if problem.objective_sense == "MAXIMIZE":
        c = [-value for value in c]
    aub: list[list[float]] = []
    bub: list[float] = []
    aeq: list[list[float]] = []
    beq: list[float] = []
    for row in problem.constraints:
        coefficients = list(row.coefficients.values)
        if row.sense == "LESS_EQUAL":
            aub.append(coefficients); bub.append(row.rhs.value)
        elif row.sense == "GREATER_EQUAL":
            aub.append([-value for value in coefficients]); bub.append(-row.rhs.value)
        else:
            aeq.append(coefficients); beq.append(row.rhs.value)
    try:
        solved = linprog(c, A_ub=aub or None, b_ub=bub or None, A_eq=aeq or None, b_eq=beq or None, bounds=list(zip(problem.lower_bounds.values, problem.upper_bounds.values)), method="highs")
    except Exception:
        return _result(problem.problem_hash, Identifier("SCIPY_HIGHS"), RawSolverStatus.ERROR, NormalizedSolverStatus.FAIL, problem.variable_ids, None, None, None, Identifier("SOLVER_ERROR"))
    raw = {0: RawSolverStatus.OPTIMAL, 2: RawSolverStatus.INFEASIBLE, 3: RawSolverStatus.UNBOUNDED}.get(int(solved.status), RawSolverStatus.ERROR)
    normalized = NormalizedSolverStatus.PASS if raw is RawSolverStatus.OPTIMAL else (NormalizedSolverStatus.INFEASIBLE if raw is RawSolverStatus.INFEASIBLE else NormalizedSolverStatus.FAIL)
    solution = FloatVector(solved.x) if solved.x is not None else None
    objective_value = FiniteFloat(sum(coefficient * value for coefficient, value in zip(problem.objective.values, solved.x))) if solved.x is not None else None
    residuals = _constraint_residuals(problem.constraints, solution.values) if solution is not None else None
    failure = None if raw is RawSolverStatus.OPTIMAL else Identifier({RawSolverStatus.INFEASIBLE: "LP_INFEASIBLE", RawSolverStatus.UNBOUNDED: "LP_UNBOUNDED"}.get(raw, "SOLVER_ERROR"))
    return _result(problem.problem_hash, Identifier("SCIPY_HIGHS"), raw, normalized, problem.variable_ids, solution, objective_value, residuals, failure)


def solve_diagnostic_qp(problem: DiagnosticQuadraticProblem) -> DiagnosticSolveResult:
    """Run a convex caller-specified QP through OSQP for numerical diagnostics."""

    try:
        import numpy as np
        import osqp
        from scipy import sparse
    except Exception:
        return _result(problem.problem_hash, Identifier("OSQP"), RawSolverStatus.ERROR, NormalizedSolverStatus.UNAVAILABLE, problem.variable_ids, None, None, None, Identifier("DEPENDENCY_UNAVAILABLE"))
    n = len(problem.variable_ids.values)
    rows = [list(row.coefficients.values) for row in problem.constraints]
    lower = []
    upper = []
    for row in problem.constraints:
        if row.sense == "LESS_EQUAL":
            lower.append(-np.inf); upper.append(row.rhs.value)
        elif row.sense == "GREATER_EQUAL":
            lower.append(row.rhs.value); upper.append(np.inf)
        else:
            lower.append(row.rhs.value); upper.append(row.rhs.value)
    rows.extend(np.eye(n).tolist())
    lower.extend(problem.lower_bounds.values); upper.extend(problem.upper_bounds.values)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.asarray(problem.quadratic_matrix.values)), q=np.asarray(problem.linear_objective.values), A=sparse.csc_matrix(np.asarray(rows)), l=np.asarray(lower), u=np.asarray(upper), verbose=False, polishing=True)
            solved = solver.solve()
    except Exception:
        return _result(problem.problem_hash, Identifier("OSQP"), RawSolverStatus.ERROR, NormalizedSolverStatus.FAIL, problem.variable_ids, None, None, None, Identifier("SOLVER_ERROR"))
    status_text = str(solved.info.status).lower()
    if status_text == "solved":
        raw, normalized = RawSolverStatus.OPTIMAL, NormalizedSolverStatus.PASS
    elif "infeasible" in status_text:
        raw, normalized = RawSolverStatus.INFEASIBLE, NormalizedSolverStatus.INFEASIBLE
    elif "unbounded" in status_text:
        raw, normalized = RawSolverStatus.UNBOUNDED, NormalizedSolverStatus.FAIL
    else:
        raw, normalized = RawSolverStatus.ERROR, NormalizedSolverStatus.FAIL
    solution = FloatVector(solved.x) if solved.x is not None and all(math.isfinite(float(x)) for x in solved.x) else None
    objective_value = None
    if solution is not None:
        x = solution.values
        quadratic = sum(problem.quadratic_matrix.values[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
        objective_value = FiniteFloat(0.5 * quadratic + sum(a * b for a, b in zip(problem.linear_objective.values, x)))
    residuals = _constraint_residuals(problem.constraints, solution.values) if solution is not None else None
    failure = None if raw is RawSolverStatus.OPTIMAL else Identifier("QP_INFEASIBLE" if raw is RawSolverStatus.INFEASIBLE else "SOLVER_ERROR")
    return _result(problem.problem_hash, Identifier("OSQP"), raw, normalized, problem.variable_ids, solution, objective_value, residuals, failure)
