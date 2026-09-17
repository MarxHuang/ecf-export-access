"""Diagnostic reference/reported allocation pairing for explicit LP inputs.

This orchestration layer is intentionally outside ``src/r4r``.  It solves two
caller-supplied diagnostic LPs, preserves both solver records, and computes
allocation-stage SG/OL/UD only when both sides produce aligned nonnegative
solutions.  It is not a canonical runner and cannot create manuscript
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.scenarios import AllocationVector
from r4r.pair_metrics import (
    AccessTransferMetricResult,
    PAPER_REDISTRIBUTION_DEFINITION_ID,
    compute_access_transfer_metrics,
)
from r4r.models import ProxyModel
from r4r.proxy_evaluator import ProxyBudget, ProxyEvaluationResult, ProxyEvaluationSpecification, evaluate_proxy
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, NormalizedSolverStatus, Sha256
from tools.diagnostic_optimization import DiagnosticLinearProblem, DiagnosticSolveResult, solve_diagnostic_lp


_PAIR_STATUSES = {"PAIRED_DIAGNOSTIC", "PAIR_UNAVAILABLE"}


@dataclass(frozen=True, slots=True)
class DiagnosticAllocationPairResult(ContractModel):
    """Two-side diagnostic solve records plus optional allocation metrics."""

    participant_ids: IdentifierVector
    reporter_id: Identifier
    reference_problem_hash: Sha256
    reported_problem_hash: Sha256
    reference_solver: DiagnosticSolveResult
    reported_solver: DiagnosticSolveResult
    reference_allocation: AllocationVector | None
    reported_allocation: AllocationVector | None
    metrics: AccessTransferMetricResult | None
    delivery_cap_mw: float
    status: str
    failure_reason: Identifier | None
    diagnostic_only: bool = True
    serialization_id = "diagnostic_allocation_pair.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.reporter_id not in self.participant_ids.values:
            raise ValidationError("reporter_id must be present in participant_ids")
        if not isinstance(self.reference_problem_hash, Sha256) or not isinstance(self.reported_problem_hash, Sha256):
            raise ValidationError("problem hashes must be Sha256")
        if not isinstance(self.reference_solver, DiagnosticSolveResult) or not isinstance(self.reported_solver, DiagnosticSolveResult):
            raise ValidationError("solver records must be DiagnosticSolveResult values")
        if self.reference_allocation is not None and not isinstance(self.reference_allocation, AllocationVector):
            raise ValidationError("reference_allocation must be AllocationVector or null")
        if self.reported_allocation is not None and not isinstance(self.reported_allocation, AllocationVector):
            raise ValidationError("reported_allocation must be AllocationVector or null")
        if (self.reference_allocation is None) != (self.reported_allocation is None):
            raise ValidationError("paired allocations must be present on both sides or neither")
        if self.metrics is not None and not isinstance(self.metrics, AccessTransferMetricResult):
            raise ValidationError("metrics must be AccessTransferMetricResult or null")
        if self.reference_allocation is None and self.metrics is not None:
            raise ValidationError("metrics require paired allocations")
        if isinstance(self.delivery_cap_mw, bool) or not isinstance(self.delivery_cap_mw, (int, float)) or not math.isfinite(float(self.delivery_cap_mw)) or self.delivery_cap_mw < 0:
            raise ValidationError("delivery_cap_mw must be a nonnegative finite scalar")
        if self.status not in _PAIR_STATUSES:
            raise ValidationError("pair status is not registered")
        if self.status == "PAIRED_DIAGNOSTIC" and (self.reference_allocation is None or self.metrics is None):
            raise ValidationError("paired diagnostic status requires allocations and metrics")
        if self.status == "PAIR_UNAVAILABLE" and self.failure_reason is None:
            raise ValidationError("unavailable pair requires a failure reason")
        if self.failure_reason is not None and not isinstance(self.failure_reason, Identifier):
            raise ValidationError("failure_reason must be Identifier or null")
        if self.diagnostic_only is not True:
            raise ValidationError("diagnostic pair results cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "reference_problem_hash": self.reference_problem_hash.to_json(),
            "reported_problem_hash": self.reported_problem_hash.to_json(),
            "reference_solver": self.reference_solver.to_json(),
            "reported_solver": self.reported_solver.to_json(),
            "reference_allocation": self.reference_allocation.to_json() if self.reference_allocation else None,
            "reported_allocation": self.reported_allocation.to_json() if self.reported_allocation else None,
            "metrics": self.metrics.to_json() if self.metrics else None,
            "delivery_cap_mw": float(self.delivery_cap_mw),
            "status": self.status,
            "failure_reason": self.failure_reason.to_json() if self.failure_reason else None,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def pair_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def _allocation_from_solution(
    *,
    problem: DiagnosticLinearProblem,
    solver: DiagnosticSolveResult,
    side: AllocationSide,
    scenario_id: Identifier,
) -> AllocationVector | None:
    if solver.normalized_status is not NormalizedSolverStatus.PASS or solver.solution is None:
        return None
    if any(value < 0.0 for value in solver.solution.values):
        return None
    values = FloatVector(solver.solution.values)
    allocation_hash = Sha256(canonical_hash({
        "participant_ids": problem.variable_ids.to_json(),
        "values_mw": values.to_json(),
        "side": side.value,
        "scenario_id": scenario_id.to_json(),
        "problem_hash": problem.problem_hash.to_json(),
        "solver_result_hash": solver.result_hash.to_json(),
    }))
    return AllocationVector(
        participant_ids=problem.variable_ids,
        values_mw=values,
        vector_length=len(values.values),
        side=side,
        scenario_id=scenario_id,
        allocation_hash=allocation_hash,
    )


def solve_diagnostic_allocation_pair(
    reference_problem: DiagnosticLinearProblem,
    reported_problem: DiagnosticLinearProblem,
    *,
    reporter_id: Identifier,
    reference_scenario_id: Identifier,
    reported_scenario_id: Identifier,
    delivery_cap_mw: float,
) -> DiagnosticAllocationPairResult:
    """Solve and pair two explicit diagnostic LPs without canonical promotion."""

    if reference_problem.variable_ids != reported_problem.variable_ids:
        raise ValidationError("reference and reported LP variable order must match")
    participant_ids = reference_problem.variable_ids
    if reporter_id not in participant_ids.values:
        raise ValidationError("reporter_id must be present in LP variable IDs")
    reference_solver = solve_diagnostic_lp(reference_problem)
    reported_solver = solve_diagnostic_lp(reported_problem)
    reference_allocation = _allocation_from_solution(
        problem=reference_problem,
        solver=reference_solver,
        side=AllocationSide.REFERENCE,
        scenario_id=reference_scenario_id,
    )
    reported_allocation = _allocation_from_solution(
        problem=reported_problem,
        solver=reported_solver,
        side=AllocationSide.REPORTED,
        scenario_id=reported_scenario_id,
    )
    if reference_allocation is None or reported_allocation is None:
        failure = Identifier("REFERENCE_SOLVE_FAILED" if reference_allocation is None else "REPORTED_SOLVE_FAILED")
        return DiagnosticAllocationPairResult(
            participant_ids=participant_ids,
            reporter_id=reporter_id,
            reference_problem_hash=reference_problem.problem_hash,
            reported_problem_hash=reported_problem.problem_hash,
            reference_solver=reference_solver,
            reported_solver=reported_solver,
            reference_allocation=None,
            reported_allocation=None,
            metrics=None,
            delivery_cap_mw=delivery_cap_mw,
            status="PAIR_UNAVAILABLE",
            failure_reason=failure,
        )
    metrics = compute_access_transfer_metrics(
        participant_ids,
        reference_allocation.values_mw.values,
        reported_allocation.values_mw.values,
        reporter_id,
        delivery_cap_mw,
        redistribution_definition_id=Identifier(PAPER_REDISTRIBUTION_DEFINITION_ID),
    )
    return DiagnosticAllocationPairResult(
        participant_ids=participant_ids,
        reporter_id=reporter_id,
        reference_problem_hash=reference_problem.problem_hash,
        reported_problem_hash=reported_problem.problem_hash,
        reference_solver=reference_solver,
        reported_solver=reported_solver,
        reference_allocation=reference_allocation,
        reported_allocation=reported_allocation,
        metrics=metrics,
        delivery_cap_mw=delivery_cap_mw,
        status="PAIRED_DIAGNOSTIC",
        failure_reason=None,
    )


def evaluate_diagnostic_proxy_pair(
    pair: DiagnosticAllocationPairResult,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    *,
    branch_budget: ProxyBudget | None = None,
    voltage_budget: ProxyBudget | None = None,
) -> tuple[ProxyEvaluationResult, ProxyEvaluationResult]:
    """Evaluate both paired allocations on the diagnostic proxy path.

    The returned ``overall_pass`` flags describe proxy feasibility only.  They
    do not constitute AC feasibility, a pair evidence gate, or manuscript
    eligibility.
    """

    if pair.status != "PAIRED_DIAGNOSTIC" or pair.reference_allocation is None or pair.reported_allocation is None:
        raise ValidationError("proxy pair evaluation requires a paired diagnostic allocation")
    if pair.participant_ids != proxy_specification.participant_ids:
        raise ValidationError("proxy pair participant order does not match specification")
    reference = evaluate_proxy(
        proxy_model,
        proxy_specification,
        pair.participant_ids,
        pair.reference_allocation.values_mw,
        branch_budget=branch_budget,
        voltage_budget=voltage_budget,
        formal_execution=False,
    )
    reported = evaluate_proxy(
        proxy_model,
        proxy_specification,
        pair.participant_ids,
        pair.reported_allocation.values_mw,
        branch_budget=branch_budget,
        voltage_budget=voltage_budget,
        formal_execution=False,
    )
    return reference, reported


__all__ = [
    "DiagnosticAllocationPairResult",
    "evaluate_diagnostic_proxy_pair",
    "solve_diagnostic_allocation_pair",
]
