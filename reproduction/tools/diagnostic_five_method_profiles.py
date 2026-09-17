"""Typed construction of the fixed five diagnostic allocation profiles.

The factory is deliberately separate from the AC runner: it materializes the
three algebraic rules and the two solver-backed objectives from one admitted
request and one explicit feasible domain.  No result is promoted or treated as
paper evidence.
"""
from __future__ import annotations

from typing import Sequence

from r4r.algebraic_allocation import (
    BoundedRequestVector,
    allocate_equal_kw_reduction,
    allocate_flat_level,
    allocate_weighted_proportional,
    bind_algebraic_allocation,
)
from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.allocation_pipeline import (
    AlgebraicAllocationBinding,
    DiagnosticProfileBinding,
    ExplicitCapacityVector,
)
from r4r.errors import ValidationError
from r4r.objective_contracts import FairnessObjectiveSpecification, MaxExportObjectiveSpecification
from r4r.request_contracts import AdmittedRequestVector
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, Sha256
from tools.diagnostic_objective_problems import (
    DiagnosticFairnessQPBinding,
    build_diagnostic_fairness_qp_problem,
    build_diagnostic_max_export_stage1_problem,
    build_diagnostic_max_export_stage2_problem,
)
from tools.diagnostic_optimization import solve_diagnostic_lp, solve_diagnostic_qp
from tools.diagnostic_solver_allocation import (
    DiagnosticSolverAllocationBinding,
    bind_diagnostic_solver_allocation,
)


def _reference_hash(participant_ids, values) -> Sha256:
    return Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": values.to_json()}))


def build_ieee141_five_method_profiles(
    *,
    domain: FeasibleAllocationDomain,
    request_values_mw: FloatVector,
    request_source_hash: Sha256,
    admitted_request: AdmittedRequestVector,
    capacity_vector: ExplicitCapacityVector,
    max_export_specification: MaxExportObjectiveSpecification,
    fairness_specification: FairnessObjectiveSpecification,
    context_hash: Sha256,
    scenario_id: Identifier,
    side: AllocationSide,
    tie_break_sense: str = "MINIMIZE",
) -> tuple[tuple[Identifier, DiagnosticProfileBinding], ...]:
    """Build all five profiles with concrete typed allocation provenance."""

    if not isinstance(domain, FeasibleAllocationDomain):
        raise ValidationError("domain must be FeasibleAllocationDomain")
    if not isinstance(request_values_mw, FloatVector) or not isinstance(request_source_hash, Sha256):
        raise ValidationError("factory requires a typed admitted request vector and request source hash")
    if not isinstance(admitted_request, AdmittedRequestVector):
        raise ValidationError("factory requires a typed admitted request")
    if request_values_mw != admitted_request.admitted_values_mw:
        raise ValidationError("request vector and admitted request vectors differ")
    if admitted_request.participant_ids != domain.participant_ids:
        raise ValidationError("admitted request participant order does not match domain")
    if admitted_request.participant_registry_hash != domain.participant_registry_hash:
        raise ValidationError("admitted request registry provenance does not match domain")
    if capacity_vector.participant_ids != domain.participant_ids:
        raise ValidationError("capacity vector participant order does not match domain")
    if capacity_vector.participant_registry_hash != domain.participant_registry_hash:
        raise ValidationError("capacity vector registry provenance does not match domain")
    if admitted_request.capacity_spec_hash != capacity_vector.capacity_spec_hash:
        raise ValidationError("admitted request and capacity vector use different capacity specifications")
    if not isinstance(max_export_specification, MaxExportObjectiveSpecification):
        raise ValidationError("max-export specification is required")
    if max_export_specification.feasible_domain_hash != domain.domain_hash:
        raise ValidationError("max-export specification is not bound to domain")
    if not isinstance(fairness_specification, FairnessObjectiveSpecification):
        raise ValidationError("fairness specification is required")
    if fairness_specification.feasible_domain_hash != domain.domain_hash:
        raise ValidationError("fairness specification is not bound to domain")
    if fairness_specification.participant_ids != domain.participant_ids:
        raise ValidationError("fairness participant order does not match domain")
    if not isinstance(context_hash, Sha256) or not isinstance(scenario_id, Identifier) or not isinstance(side, AllocationSide):
        raise ValidationError("factory provenance arguments have invalid types")

    # A network-feasible domain may tighten a raw admitted request by an
    # explicit capacity.  All five methods must consume that same effective
    # request vector; silently feeding the raw request to scalar rules would
    # either fail or create a result outside the declared domain.
    request = BoundedRequestVector(
        values_mw=FloatVector(
        min(value, upper)
        for value, upper in zip(
             request_values_mw.values,
            domain.upper_bounds_mw.values,
        )
        ),
        source_request_hash=request_source_hash,
    )
    source_hash = request_source_hash
    algebraic_results = (
        (Identifier("weighted_proportional"), Identifier("WEIGHTED_PROPORTIONAL"), allocate_weighted_proportional(domain, domain.participant_ids, request, source_request_hash=source_hash)),
        (Identifier("equal_kw_reduction"), Identifier("EQUAL_KW_REDUCTION"), allocate_equal_kw_reduction(domain, domain.participant_ids, request, source_request_hash=source_hash)),
        (Identifier("flat_level"), Identifier("FLAT_LEVEL"), allocate_flat_level(domain, domain.participant_ids, request, source_request_hash=source_hash)),
    )
    profiles: list[tuple[Identifier, DiagnosticProfileBinding]] = []
    for method_id, rule_id, result in algebraic_results:
        binding = bind_algebraic_allocation(
            result,
            side=side,
            scenario_id=scenario_id,
            participant_registry_hash=domain.participant_registry_hash,
        )
        profiles.append((
            method_id,
            DiagnosticProfileBinding(
                allocation=binding,
                method_id=method_id,
                scenario_id=scenario_id,
                rule_id=rule_id,
                objective_id=None,
                domain_hash=domain.domain_hash,
                request_hash=binding.request_hash,
                admitted_request_hash=admitted_request.request_hash,
            ),
        ))

    stage1_problem = build_diagnostic_max_export_stage1_problem(
        domain, max_export_specification, context_hash=context_hash,
    )
    stage1_result = solve_diagnostic_lp(stage1_problem)
    stage2_problem = build_diagnostic_max_export_stage2_problem(
        domain,
        max_export_specification,
        stage1_problem=stage1_problem,
        stage1_solver_result=stage1_result,
        context_hash=context_hash,
        tie_break_sense=tie_break_sense,
    )
    stage2_result = solve_diagnostic_lp(stage2_problem.problem)
    max_binding = bind_diagnostic_solver_allocation(
        problem=stage2_problem,
        objective_specification=max_export_specification,
        solver_result=stage2_result,
        participant_ids=domain.participant_ids,
        side=side,
        scenario_id=scenario_id,
        capacity_vector=capacity_vector,
        admitted_request=admitted_request,
        bound_policy="ADMITTED_REQUEST",
    )
    profiles.append((
        Identifier("max_export_lp"),
        DiagnosticProfileBinding(
            allocation=max_binding,
            method_id=Identifier("max_export_lp"),
            scenario_id=scenario_id,
            rule_id=None,
            objective_id=max_export_specification.objective_id,
            domain_hash=domain.domain_hash,
            request_hash=max_binding.request_hash,
            admitted_request_hash=admitted_request.request_hash,
        ),
    ))

    if fairness_specification.reference_mode == "ENDOGENOUS_MAX_EXPORT":
        if fairness_specification.max_export_specification_hash != max_export_specification.specification_hash:
            raise ValidationError("fairness endogenous reference does not bind this max-export objective")
    elif fairness_specification.reference_mode == "EXTERNAL_REFERENCE":
        expected_reference_hash = _reference_hash(domain.participant_ids, max_binding.values_mw)
        if fairness_specification.external_reference_allocation_hash != expected_reference_hash:
            raise ValidationError("fairness external reference does not bind max-export allocation")

    fairness_binding = build_diagnostic_fairness_qp_problem(
        domain,
        fairness_specification,
        request_mw=request.values_mw,
        reference_allocation_mw=max_binding.values_mw.values,
        context_hash=context_hash,
        request_hash=admitted_request.request_hash,
        request_source_hash=source_hash,
    )
    fairness_result = solve_diagnostic_qp(fairness_binding.problem)
    fairness_allocation = bind_diagnostic_solver_allocation(
        problem=fairness_binding,
        objective_specification=fairness_specification,
        solver_result=fairness_result,
        participant_ids=domain.participant_ids,
        side=side,
        scenario_id=scenario_id,
        capacity_vector=capacity_vector,
        admitted_request=admitted_request,
        bound_policy="ADMITTED_REQUEST",
    )
    profiles.append((
        Identifier("fairness_qp"),
        DiagnosticProfileBinding(
            allocation=fairness_allocation,
            method_id=Identifier("fairness_qp"),
            scenario_id=scenario_id,
            rule_id=None,
            objective_id=fairness_specification.objective_id,
            domain_hash=domain.domain_hash,
            request_hash=fairness_allocation.request_hash,
            admitted_request_hash=admitted_request.request_hash,
        ),
    ))
    return tuple(profiles)


__all__ = ["build_ieee141_five_method_profiles"]
