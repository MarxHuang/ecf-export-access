"""Build a typed, network-constrained allocation domain for diagnostics.

The builder translates an explicit proxy matrix into ``FeasibleAllocationDomain``
rows.  It never turns participant capacities into a feasible domain by itself,
and it never invokes an optimizer.  Absolute-magnitude rows are expanded into
two signed linear inequalities so the sign convention remains visible.
"""
from __future__ import annotations

from typing import Sequence

from r4r.allocation_domain import FeasibleAllocationDomain, LinearConstraintRow
from r4r.errors import ValidationError
from r4r.models import ProxyModel
from r4r.proxy_evaluator import ProxyBudget, ProxyEvaluationSpecification
from r4r.types import FloatVector, FiniteFloat, Identifier, IdentifierVector, Sha256


def _rows(
    *,
    matrix: Sequence[Sequence[float]],
    constraint_ids: IdentifierVector,
    budget: ProxyBudget,
    mode: str,
    unit: str,
    family: str,
    orientation_ids: Sequence[Identifier] | None = None,
) -> list[LinearConstraintRow]:
    if len(matrix) != len(constraint_ids.values) or len(matrix) != len(budget.tightened_budget.values):
        raise ValidationError(f"{family.lower()} matrix, IDs and budget rows must align")
    if orientation_ids is not None and tuple(orientation_ids) != constraint_ids.values:
        raise ValidationError(f"{family.lower()} constraint order does not match proxy orientation")
    rows: list[LinearConstraintRow] = []
    for index, (coefficients, constraint_id, rhs) in enumerate(
        zip(matrix, constraint_ids.values, budget.tightened_budget.values)
    ):
        if mode == "SIGNED_UPPER_BOUND":
            rows.append(LinearConstraintRow(
                constraint_id=constraint_id,
                family=family,
                coefficients=FloatVector(coefficients),
                rhs=FiniteFloat(rhs),
                sense="LESS_EQUAL",
                row_unit=unit,
                element_id=constraint_id,
                orientation="SIGNED_UPPER_BOUND",
                baseline_quantity="INCREMENTAL",
                limit_source="EXPLICIT_TIGHTENED_BUDGET",
            ))
        elif mode == "SIGNED_LOWER_BOUND":
            rows.append(LinearConstraintRow(
                constraint_id=constraint_id,
                family=family,
                coefficients=FloatVector(coefficients),
                rhs=FiniteFloat(rhs),
                sense="GREATER_EQUAL",
                row_unit=unit,
                element_id=constraint_id,
                orientation="SIGNED_LOWER_BOUND",
                baseline_quantity="INCREMENTAL",
                limit_source="EXPLICIT_TIGHTENED_BUDGET",
            ))
        elif mode == "ABSOLUTE_MAGNITUDE":
            rows.extend((
                LinearConstraintRow(
                    constraint_id=Identifier(f"{constraint_id.value}__POS"),
                    family=family,
                    coefficients=FloatVector(coefficients),
                    rhs=FiniteFloat(rhs),
                    sense="LESS_EQUAL",
                    row_unit=unit,
                    element_id=constraint_id,
                    orientation="ABSOLUTE_MAGNITUDE_POSITIVE",
                    baseline_quantity="INCREMENTAL",
                    limit_source="EXPLICIT_TIGHTENED_BUDGET",
                ),
                LinearConstraintRow(
                    constraint_id=Identifier(f"{constraint_id.value}__NEG"),
                    family=family,
                    coefficients=FloatVector(-value for value in coefficients),
                    rhs=FiniteFloat(rhs),
                    sense="LESS_EQUAL",
                    row_unit=unit,
                    element_id=constraint_id,
                    orientation="ABSOLUTE_MAGNITUDE_NEGATIVE",
                    baseline_quantity="INCREMENTAL",
                    limit_source="EXPLICIT_TIGHTENED_BUDGET",
                ),
            ))
        else:
            raise ValidationError(f"{family.lower()} constraint mode is not registered for domain translation")
    return rows


def build_diagnostic_feasible_allocation_domain(
    *,
    participant_ids: IdentifierVector,
    participant_registry_hash: Sha256,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    branch_budget: ProxyBudget,
    voltage_budget: ProxyBudget,
    lower_bounds_mw: Sequence[float],
    upper_bounds_mw: Sequence[float],
    upper_bound_source: str,
    feasibility_tolerance: float = 0.0,
    capacity_spec_hash: Sha256 | None = None,
    request_hash: Sha256 | None = None,
    scientific_status: str = "CANDIDATE",
    unresolved_fields: Sequence[str] = ("formal proxy/domain authorization open",),
) -> FeasibleAllocationDomain:
    """Translate explicit proxy rows into a diagnostic feasible domain."""
    if not isinstance(participant_ids, IdentifierVector) or not participant_ids.values:
        raise ValidationError("participant_ids must be a non-empty IdentifierVector")
    if not isinstance(participant_registry_hash, Sha256):
        raise ValidationError("participant_registry_hash must be Sha256")
    if not isinstance(proxy_model, ProxyModel) or not isinstance(proxy_specification, ProxyEvaluationSpecification):
        raise ValidationError("proxy_model and proxy_specification are required")
    if proxy_specification.participant_ids != participant_ids:
        raise ValidationError("proxy participant order does not match domain participant order")
    if proxy_specification.participant_registry_hash != participant_registry_hash:
        raise ValidationError("proxy and domain participant registry hashes differ")
    if tuple(proxy_model.branch_orientation) != proxy_specification.branch_constraint_ids.values:
        raise ValidationError("proxy branch orientation does not match specification constraint order")
    if proxy_specification.branch_budget_hash != branch_budget.budget_hash:
        raise ValidationError("branch budget hash does not match proxy specification")
    if proxy_specification.voltage_budget_hash != voltage_budget.budget_hash:
        raise ValidationError("voltage budget hash does not match proxy specification")
    if branch_budget.status != "VALID" or voltage_budget.status != "VALID":
        raise ValidationError("diagnostic domain requires valid tightened budgets")
    if branch_budget.unit != "MW" or voltage_budget.unit != "pu":
        raise ValidationError("proxy budget units do not match branch/voltage domains")
    bounds_lower = FloatVector(lower_bounds_mw)
    bounds_upper = FloatVector(upper_bounds_mw)
    n = len(participant_ids.values)
    if len(bounds_lower.values) != n or len(bounds_upper.values) != n:
        raise ValidationError("allocation bounds must match participant IDs")
    rows = _rows(
        matrix=proxy_model.branch_sensitivity_matrix.values,
        constraint_ids=proxy_specification.branch_constraint_ids,
        budget=branch_budget,
        mode=proxy_specification.branch_constraint_mode,
        unit="MW",
        family="BRANCH_INCREMENT",
        orientation_ids=proxy_model.branch_orientation,
    )
    rows.extend(_rows(
        matrix=proxy_model.voltage_sensitivity_matrix.values,
        constraint_ids=proxy_specification.voltage_constraint_ids,
        budget=voltage_budget,
        mode=proxy_specification.voltage_constraint_mode,
        unit="pu",
        family="VOLTAGE_UPPER" if proxy_specification.voltage_constraint_mode in {"SIGNED_UPPER_BOUND", "ABSOLUTE_MAGNITUDE"} else "VOLTAGE_LOWER",
    ))
    return FeasibleAllocationDomain(
        participant_ids=participant_ids,
        lower_bounds_mw=bounds_lower,
        upper_bounds_mw=bounds_upper,
        upper_bound_source=upper_bound_source,
        constraints=tuple(rows),
        participant_registry_hash=participant_registry_hash,
        capacity_spec_hash=capacity_spec_hash,
        request_hash=request_hash,
        proxy_spec_hash=proxy_specification.specification_hash,
        feasibility_tolerance=FiniteFloat(feasibility_tolerance),
        scientific_status=scientific_status,
        unresolved_fields=tuple(unresolved_fields),
    )


__all__ = ["build_diagnostic_feasible_allocation_domain"]
