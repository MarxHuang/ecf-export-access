"""Explicit linear proxy evaluation with a fail-closed scientific boundary.

The evaluator accepts sensitivity matrices and headroom vectors that are
already supplied by the caller.  It does not derive sensitivities, infer
branch orientation, choose a voltage sign convention, or clip budgets.  A
candidate specification can therefore be used for diagnostics while formal
execution remains blocked until the scientific definition is author-frozen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from r4r.errors import ContractError, UnfrozenScientificDefinitionError, ValidationError
from r4r.models import ProxyModel
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, FiniteFloat, Identifier, IdentifierVector, QAssumption, Sha256


_STATUSES = {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}
_MODEL_FAMILY = "EXPLICIT_LINEAR_SENSITIVITY"
_MODES = {"UNSPECIFIED", "SIGNED_UPPER_BOUND", "SIGNED_LOWER_BOUND", "ABSOLUTE_MAGNITUDE"}
_RESULT_STATUSES = {"VALID_FEASIBLE", "VALID_INFEASIBLE", "INVALID_BUDGET"}
_EVIDENCE_ROLES = {"DIAGNOSTIC_ONLY", "PRIMARY_CANDIDATE", "UNRESOLVED"}
_BOUND_SOURCES = {"CAPACITY", "ADMITTED_REQUEST", "MIN_CAPACITY_REQUEST", "EXPLICIT", "UNRESOLVED"}
_EVALUATION_FORMS = {"INCREMENTAL_BUDGET", "ABSOLUTE_LIMIT"}
_VOLTAGE_REPRESENTATIONS = {"V_MAGNITUDE", "V_SQUARED", "DELTA_V", "DELTA_V_SQUARED"}
_BRANCH_FLOW_QUANTITIES = {"FROM_END_ACTIVE_POWER", "TO_END_ACTIVE_POWER", "ABSOLUTE_ACTIVE_POWER"}
_PROXY_FAMILIES = {"BRANCH", "VOLTAGE"}
_FAMILY_STATUSES = {"PASS", "FAIL", "INVALID_BUDGET"}
_VALIDITY_STATUSES = {"VALID", "INVALID_BUDGET"}
_FEASIBILITY_STATUSES = {"FEASIBLE", "INFEASIBLE", "NOT_EVALUABLE"}
_NORMALIZATION_STATUSES = {"DEFINED", "UNDEFINED_ZERO_BUDGET", "INVALID_BUDGET"}


@dataclass(frozen=True, slots=True)
class ProxyConstraintRow(ContractModel):
    """Physical metadata for one branch or voltage proxy row."""

    constraint_id: Identifier
    family: str
    element_id: Identifier
    orientation: str
    baseline_quantity: str
    limit_source: str
    unit: str
    serialization_id = "proxy_constraint_row.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, Identifier) or not isinstance(self.element_id, Identifier):
            raise ValidationError("proxy constraint IDs must be Identifier values")
        if self.family not in _PROXY_FAMILIES:
            raise ValidationError("proxy constraint family is not registered")
        for name, value in (
            ("orientation", self.orientation),
            ("baseline_quantity", self.baseline_quantity),
            ("limit_source", self.limit_source),
            ("unit", self.unit),
        ):
            if not isinstance(value, str) or not value:
                raise ValidationError(f"proxy constraint {name} is required")
        expected_unit = "MW" if self.family == "BRANCH" else "pu"
        if self.unit != expected_unit:
            raise ValidationError(f"{self.family.lower()} proxy constraint unit must be {expected_unit}")

    def to_json(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id.to_json(),
            "family": self.family,
            "element_id": self.element_id.to_json(),
            "orientation": self.orientation,
            "baseline_quantity": self.baseline_quantity,
            "limit_source": self.limit_source,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class NormalizedViolationVector(ContractModel):
    """Violation divided by a positive budget, with zero-budget undefinedness explicit."""

    values: tuple[float | None, ...]
    statuses: tuple[str, ...]
    serialization_id = "proxy_normalized_violation.v1"

    def __post_init__(self) -> None:
        if len(self.values) != len(self.statuses):
            raise ValidationError("normalized violation values and statuses must align")
        for value, status in zip(self.values, self.statuses):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise ValidationError("normalized violation values must be finite or null")
            if status not in _NORMALIZATION_STATUSES:
                raise ValidationError("normalized violation status is not registered")
            if status == "DEFINED" and value is None:
                raise ValidationError("defined normalized violation cannot be null")
            if status != "DEFINED" and value is not None:
                raise ValidationError("undefined normalized violation must be null")

    def to_json(self) -> dict[str, Any]:
        return {"values": list(self.values), "statuses": list(self.statuses)}


@dataclass(frozen=True, slots=True)
class ProxyEvaluationSpecification(ContractModel):
    """Metadata required to interpret an explicit proxy matrix evaluation."""

    specification_id: Identifier
    model_family: Identifier
    scientific_status: str
    participant_ids: IdentifierVector
    branch_constraint_mode: str
    voltage_constraint_mode: str
    equation_ids: IdentifierVector
    participant_registry_id: Identifier
    network_id: Identifier
    network_hash: Sha256
    operating_point_id: Identifier
    operating_point_hash: Sha256
    load_scale: FiniteFloat
    q_assumption_id: QAssumption
    q_spec_hash: Sha256
    evidence_role: str
    variable_unit: str
    variable_semantics: str
    upper_bound_source: str
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_evaluation_form: str
    branch_flow_quantity: str
    voltage_representation: str
    participant_registry_hash: Sha256 | None = None
    branch_budget_hash: Sha256 | None = None
    voltage_budget_hash: Sha256 | None = None
    branch_constraint_rows: tuple[ProxyConstraintRow, ...] = ()
    voltage_constraint_rows: tuple[ProxyConstraintRow, ...] = ()
    branch_baseline_mw: FloatVector | None = None
    voltage_baseline_pu: FloatVector | None = None
    branch_tolerance_mw: FiniteFloat = FiniteFloat(0.0)
    voltage_tolerance_pu: FiniteFloat = FiniteFloat(0.0)
    serialization_id = "proxy_evaluation_spec.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.specification_id, Identifier):
            raise ValidationError("specification_id must be Identifier")
        if not isinstance(self.model_family, Identifier):
            raise ValidationError("model_family must be Identifier")
        if self.scientific_status not in _STATUSES:
            raise ValidationError("scientific_status is not registered")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.branch_constraint_mode not in _MODES or self.voltage_constraint_mode not in _MODES:
            raise ValidationError("proxy constraint modes are not registered")
        if not isinstance(self.equation_ids, IdentifierVector) or not self.equation_ids.values:
            raise ValidationError("equation_ids must be a non-empty IdentifierVector")
        if not isinstance(self.participant_registry_id, Identifier):
            raise ValidationError("participant_registry_id must be Identifier")
        if not isinstance(self.network_id, Identifier) or not isinstance(self.operating_point_id, Identifier):
            raise ValidationError("network_id and operating_point_id must be Identifier")
        for name, value in (
            ("network_hash", self.network_hash),
            ("operating_point_hash", self.operating_point_hash),
            ("q_spec_hash", self.q_spec_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if not isinstance(self.load_scale, FiniteFloat) or self.load_scale.value < 0.0:
            raise ValidationError("load_scale must be a nonnegative FiniteFloat")
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ValidationError("evidence_role is not registered")
        if not self.variable_unit or not self.variable_semantics:
            raise ValidationError("variable unit and semantics are required")
        if self.upper_bound_source not in _BOUND_SOURCES:
            raise ValidationError("upper_bound_source is not registered")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        if self.branch_evaluation_form not in _EVALUATION_FORMS:
            raise ValidationError("branch_evaluation_form is not registered")
        if self.branch_flow_quantity not in _BRANCH_FLOW_QUANTITIES:
            raise ValidationError("branch_flow_quantity is not registered")
        if self.voltage_representation not in _VOLTAGE_REPRESENTATIONS:
            raise ValidationError("voltage_representation is not registered")
        if self.participant_registry_hash is not None and not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be Sha256 or null")
        for name, value in (("branch_budget_hash", self.branch_budget_hash), ("voltage_budget_hash", self.voltage_budget_hash)):
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or null")
        for name, rows, ids, family in (
            ("branch_constraint_rows", self.branch_constraint_rows, self.branch_constraint_ids, "BRANCH"),
            ("voltage_constraint_rows", self.voltage_constraint_rows, self.voltage_constraint_ids, "VOLTAGE"),
        ):
            if any(not isinstance(row, ProxyConstraintRow) for row in rows):
                raise ValidationError(f"{name} must contain ProxyConstraintRow values")
            if rows and tuple(row.constraint_id for row in rows) != ids.values:
                raise ValidationError(f"{name} IDs must align with constraint IDs")
            if any(row.family != family for row in rows):
                raise ValidationError(f"{name} family does not match its constraint family")
        if self.branch_baseline_mw is not None and not isinstance(self.branch_baseline_mw, FloatVector):
            raise ValidationError("branch_baseline_mw must be FloatVector or null")
        if self.voltage_baseline_pu is not None and not isinstance(self.voltage_baseline_pu, FloatVector):
            raise ValidationError("voltage_baseline_pu must be FloatVector or null")
        if self.branch_tolerance_mw.value < 0.0 or self.voltage_tolerance_pu.value < 0.0:
            raise ValidationError("proxy tolerances must be nonnegative")

    def to_json(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id.to_json(),
            "model_family": self.model_family.to_json(),
            "scientific_status": self.scientific_status,
            "participant_ids": self.participant_ids.to_json(),
            "branch_constraint_mode": self.branch_constraint_mode,
            "voltage_constraint_mode": self.voltage_constraint_mode,
            "equation_ids": self.equation_ids.to_json(),
            "participant_registry_id": self.participant_registry_id.to_json(),
            "network_id": self.network_id.to_json(),
            "network_hash": self.network_hash.to_json(),
            "operating_point_id": self.operating_point_id.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "load_scale": self.load_scale.to_json(),
            "q_assumption_id": self.q_assumption_id.value,
            "q_spec_hash": self.q_spec_hash.to_json(),
            "evidence_role": self.evidence_role,
            "variable_unit": self.variable_unit,
            "variable_semantics": self.variable_semantics,
            "upper_bound_source": self.upper_bound_source,
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_evaluation_form": self.branch_evaluation_form,
            "branch_flow_quantity": self.branch_flow_quantity,
            "voltage_representation": self.voltage_representation,
            "participant_registry_hash": self.participant_registry_hash.to_json() if self.participant_registry_hash else None,
            "branch_budget_hash": self.branch_budget_hash.to_json() if self.branch_budget_hash else None,
            "voltage_budget_hash": self.voltage_budget_hash.to_json() if self.voltage_budget_hash else None,
            "branch_constraint_rows": [row.to_json() for row in self.branch_constraint_rows],
            "voltage_constraint_rows": [row.to_json() for row in self.voltage_constraint_rows],
            "branch_baseline_mw": self.branch_baseline_mw.to_json() if self.branch_baseline_mw else None,
            "voltage_baseline_pu": self.voltage_baseline_pu.to_json() if self.voltage_baseline_pu else None,
            "branch_tolerance_mw": self.branch_tolerance_mw.to_json(),
            "voltage_tolerance_pu": self.voltage_tolerance_pu.to_json(),
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class ProxyBudget(ContractModel):
    """Physical budget, calibration margin and tightened budget as one value."""

    physical_budget: FloatVector
    calibration_margin: FloatVector
    margin_multiplier: FiniteFloat
    tightened_budget: FloatVector
    unit: str
    margin_mode_id: Identifier | None = None
    serialization_id = "proxy_budget.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.physical_budget, FloatVector) or not isinstance(self.calibration_margin, FloatVector) or not isinstance(self.tightened_budget, FloatVector):
            raise ValidationError("proxy budget vectors must be FloatVector values")
        sizes = {len(self.physical_budget.values), len(self.calibration_margin.values), len(self.tightened_budget.values)}
        if len(sizes) != 1 or not self.physical_budget.values:
            raise ValidationError("proxy budget vectors must be non-empty and aligned")
        if any(value < 0.0 for value in self.physical_budget.values):
            raise ValidationError("physical budget must be nonnegative")
        if any(value < 0.0 for value in self.calibration_margin.values):
            raise ValidationError("calibration margin must be nonnegative")
        if self.margin_multiplier.value < 0.0:
            raise ValidationError("margin multiplier must be nonnegative")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValidationError("proxy budget unit is required")
        if self.margin_mode_id is not None and not isinstance(self.margin_mode_id, Identifier):
            raise ValidationError("margin_mode_id must be Identifier or null")
        expected = tuple(
            physical - self.margin_multiplier.value * margin
            for physical, margin in zip(self.physical_budget.values, self.calibration_margin.values)
        )
        if any(not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12) for actual, target in zip(self.tightened_budget.values, expected)):
            raise ValidationError("tightened budget must equal physical budget minus margin exactly")

    @property
    def status(self) -> str:
        return "INVALID_BUDGET" if any(value < 0.0 for value in self.tightened_budget.values) else "VALID"

    @property
    def budget_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def to_json(self) -> dict[str, Any]:
        return {
            "physical_budget": self.physical_budget.to_json(),
            "calibration_margin": self.calibration_margin.to_json(),
            "margin_multiplier": self.margin_multiplier.to_json(),
            "tightened_budget": self.tightened_budget.to_json(),
            "unit": self.unit,
            "margin_mode_id": self.margin_mode_id.to_json() if self.margin_mode_id else None,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ProxyEvaluationResult(ContractModel):
    """Raw proxy outputs and explicit feasibility status.

    ``branch_slack_mw`` and ``voltage_slack_pu`` are never clipped.  Negative
    values are retained so downstream diagnostics can distinguish a violation
    from an invalid negative budget.
    """

    specification_hash: Sha256
    allocation_hash: Sha256
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_increment_mw: FloatVector
    voltage_increment_pu: FloatVector
    branch_slack_mw: FloatVector
    voltage_slack_pu: FloatVector
    branch_violation_mw: FloatVector
    voltage_violation_pu: FloatVector
    max_branch_violation_mw: FiniteFloat
    max_voltage_violation_pu: FiniteFloat
    branch_pass: bool
    voltage_pass: bool
    overall_pass: bool
    status: str
    diagnostic_only: bool
    evidence_role: str
    failure_reasons: IdentifierVector
    branch_predicted_absolute_mw: FloatVector | None
    voltage_predicted_absolute_pu: FloatVector | None
    branch_normalized_violation: NormalizedViolationVector
    voltage_normalized_violation: NormalizedViolationVector
    offending_branch_constraint_ids: IdentifierVector
    offending_voltage_constraint_ids: IdentifierVector
    branch_family_status: str
    voltage_family_status: str
    validity_status: str
    feasibility_status: str
    serialization_id = "proxy_evaluation_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.specification_hash, Sha256) or not isinstance(self.allocation_hash, Sha256):
            raise ValidationError("specification and allocation hashes must be Sha256")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        for name in (
            "branch_increment_mw", "voltage_increment_pu", "branch_slack_mw", "voltage_slack_pu",
            "branch_violation_mw", "voltage_violation_pu",
        ):
            if not isinstance(getattr(self, name), FloatVector):
                raise ValidationError(f"{name} must be FloatVector")
        if not all(isinstance(value, bool) for value in (self.branch_pass, self.voltage_pass, self.overall_pass, self.diagnostic_only)):
            raise ValidationError("proxy pass and diagnostic flags must be booleans")
        if self.status not in _RESULT_STATUSES:
            raise ValidationError("proxy result status is not registered")
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ValidationError("evidence_role is not registered")
        if not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure_reasons must be IdentifierVector")
        for name, value in (("branch_predicted_absolute_mw", self.branch_predicted_absolute_mw), ("voltage_predicted_absolute_pu", self.voltage_predicted_absolute_pu)):
            if value is not None and not isinstance(value, FloatVector):
                raise ValidationError(f"{name} must be FloatVector or null")
        if not isinstance(self.branch_normalized_violation, NormalizedViolationVector) or not isinstance(self.voltage_normalized_violation, NormalizedViolationVector):
            raise ValidationError("normalized violations must be NormalizedViolationVector values")
        if not isinstance(self.offending_branch_constraint_ids, IdentifierVector) or not isinstance(self.offending_voltage_constraint_ids, IdentifierVector):
            raise ValidationError("offending constraint IDs must be IdentifierVector values")
        if self.branch_family_status not in _FAMILY_STATUSES or self.voltage_family_status not in _FAMILY_STATUSES:
            raise ValidationError("proxy family status is not registered")
        if self.validity_status not in _VALIDITY_STATUSES or self.feasibility_status not in _FEASIBILITY_STATUSES:
            raise ValidationError("proxy validity or feasibility status is not registered")

    def to_json(self) -> dict[str, Any]:
        return {
            "specification_hash": self.specification_hash.to_json(),
            "allocation_hash": self.allocation_hash.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_increment_mw": self.branch_increment_mw.to_json(),
            "voltage_increment_pu": self.voltage_increment_pu.to_json(),
            "branch_slack_mw": self.branch_slack_mw.to_json(),
            "voltage_slack_pu": self.voltage_slack_pu.to_json(),
            "branch_violation_mw": self.branch_violation_mw.to_json(),
            "voltage_violation_pu": self.voltage_violation_pu.to_json(),
            "max_branch_violation_mw": self.max_branch_violation_mw.to_json(),
            "max_voltage_violation_pu": self.max_voltage_violation_pu.to_json(),
            "branch_pass": self.branch_pass,
            "voltage_pass": self.voltage_pass,
            "overall_pass": self.overall_pass,
            "status": self.status,
            "diagnostic_only": self.diagnostic_only,
            "evidence_role": self.evidence_role,
            "failure_reasons": self.failure_reasons.to_json(),
            "branch_predicted_absolute_mw": self.branch_predicted_absolute_mw.to_json() if self.branch_predicted_absolute_mw else None,
            "voltage_predicted_absolute_pu": self.voltage_predicted_absolute_pu.to_json() if self.voltage_predicted_absolute_pu else None,
            "branch_normalized_violation": self.branch_normalized_violation.to_json(),
            "voltage_normalized_violation": self.voltage_normalized_violation.to_json(),
            "offending_branch_constraint_ids": self.offending_branch_constraint_ids.to_json(),
            "offending_voltage_constraint_ids": self.offending_voltage_constraint_ids.to_json(),
            "branch_family_status": self.branch_family_status,
            "voltage_family_status": self.voltage_family_status,
            "validity_status": self.validity_status,
            "feasibility_status": self.feasibility_status,
        }


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float], *, field: str) -> tuple[float, ...]:
    width = len(vector)
    if any(len(row) != width for row in matrix):
        raise ValidationError(f"{field} matrix width must match allocation length")
    return tuple(sum(float(value) * float(weight) for value, weight in zip(row, vector)) for row in matrix)


def _slack(increment: float, budget: float, mode: str) -> float:
    if mode == "SIGNED_UPPER_BOUND":
        return budget - increment
    if mode == "SIGNED_LOWER_BOUND":
        return increment - budget
    if mode == "ABSOLUTE_MAGNITUDE":
        return budget - abs(increment)
    raise UnfrozenScientificDefinitionError("proxy constraint mode is unspecified")


def _pass(increment: float, budget: float, mode: str, tolerance: float = 0.0) -> bool:
    return _slack(increment, budget, mode) >= -tolerance


def _normalized(violations: Sequence[float], budgets: Sequence[float]) -> NormalizedViolationVector:
    values: list[float | None] = []
    statuses: list[str] = []
    for violation, budget in zip(violations, budgets):
        if budget < 0.0:
            values.append(None)
            statuses.append("INVALID_BUDGET")
        elif budget == 0.0:
            values.append(None)
            statuses.append("UNDEFINED_ZERO_BUDGET")
        else:
            values.append(violation / budget)
            statuses.append("DEFINED")
    return NormalizedViolationVector(tuple(values), tuple(statuses))


def evaluate_proxy(
    model: ProxyModel,
    specification: ProxyEvaluationSpecification,
    allocation_participant_ids: IdentifierVector,
    allocation_mw: FloatVector,
    *,
    branch_budget: ProxyBudget | None = None,
    voltage_budget: ProxyBudget | None = None,
    formal_execution: bool = True,
) -> ProxyEvaluationResult:
    """Evaluate explicit branch/voltage proxy constraints.

    No participant reordering, matrix derivation, clipping, or sign inference
    is performed.  ``formal_execution=False`` is the diagnostic path for a
    candidate scientific definition.
    """

    if specification.model_family.value != _MODEL_FAMILY:
        raise ContractError(f"unsupported proxy model family {specification.model_family.value}")
    if formal_execution and specification.scientific_status != "AUTHOR_FROZEN":
        raise UnfrozenScientificDefinitionError(
            "formal proxy evaluation requires AUTHOR_FROZEN specification"
        )
    if specification.scientific_status not in _STATUSES:
        raise ValidationError("scientific_status is not registered")
    if allocation_participant_ids != specification.participant_ids:
        raise ValidationError("allocation participant order does not match proxy specification")
    if len(allocation_mw.values) != len(specification.participant_ids.values):
        raise ValidationError("allocation length does not match proxy participant registry")
    if any(value < 0.0 for value in allocation_mw.values):
        raise ValidationError("proxy allocation must be nonnegative")
    if specification.branch_constraint_mode == "UNSPECIFIED" or specification.voltage_constraint_mode == "UNSPECIFIED":
        raise UnfrozenScientificDefinitionError("proxy constraint mode is unspecified")
    if specification.branch_evaluation_form == "ABSOLUTE_LIMIT" and specification.branch_constraint_mode != "ABSOLUTE_MAGNITUDE":
        raise ValidationError("absolute branch evaluation requires ABSOLUTE_MAGNITUDE mode")
    if specification.branch_evaluation_form == "ABSOLUTE_LIMIT" and specification.branch_baseline_mw is None:
        raise ValidationError("absolute branch evaluation requires an explicit branch baseline")
    if specification.voltage_representation in {"V_MAGNITUDE", "V_SQUARED"} and specification.voltage_baseline_pu is None:
        raise ValidationError("absolute voltage representation requires an explicit voltage baseline")
    for name, budget, expected_hash, unit in (
        ("branch", branch_budget, specification.branch_budget_hash, "MW"),
        ("voltage", voltage_budget, specification.voltage_budget_hash, "pu"),
    ):
        if budget is not None:
            if budget.unit != unit:
                raise ValidationError(f"{name} budget unit must be {unit}")
            if expected_hash is None or expected_hash != budget.budget_hash:
                raise ValidationError(f"{name} budget hash does not match proxy specification")

    participant_count = len(specification.participant_ids.values)
    branch_matrix = model.branch_sensitivity_matrix.values
    voltage_matrix = model.voltage_sensitivity_matrix.values
    branch_headroom = branch_budget.tightened_budget.values if branch_budget is not None else model.branch_headroom_mw.values
    voltage_headroom = voltage_budget.tightened_budget.values if voltage_budget is not None else model.voltage_headroom_pu.values
    branch_increment = _matvec(branch_matrix, allocation_mw.values, field="branch")
    voltage_increment = _matvec(voltage_matrix, allocation_mw.values, field="voltage")
    if len(branch_increment) != len(branch_headroom):
        raise ValidationError("branch matrix rows must match branch headroom")
    if len(voltage_increment) != len(voltage_headroom):
        raise ValidationError("voltage matrix rows must match voltage headroom")
    if len(branch_matrix) == 0 or len(voltage_matrix) == 0:
        raise ValidationError("proxy matrices must contain at least one row")
    if len(branch_matrix[0]) != participant_count or len(voltage_matrix[0]) != participant_count:
        raise ValidationError("proxy matrix columns must match participant registry")
    if len(specification.branch_constraint_ids.values) != len(branch_matrix):
        raise ValidationError("branch constraint IDs must match matrix rows")
    if len(specification.voltage_constraint_ids.values) != len(voltage_matrix):
        raise ValidationError("voltage constraint IDs must match matrix rows")
    if specification.branch_baseline_mw is not None and len(specification.branch_baseline_mw.values) != len(branch_matrix):
        raise ValidationError("branch baseline must match branch matrix rows")
    if specification.voltage_baseline_pu is not None and len(specification.voltage_baseline_pu.values) != len(voltage_matrix):
        raise ValidationError("voltage baseline must match voltage matrix rows")

    branch_quantity = tuple(
        base + increment for base, increment in zip(specification.branch_baseline_mw.values, branch_increment)
    ) if specification.branch_evaluation_form == "ABSOLUTE_LIMIT" and specification.branch_baseline_mw is not None else branch_increment
    voltage_quantity = tuple(
        base + increment for base, increment in zip(specification.voltage_baseline_pu.values, voltage_increment)
    ) if specification.voltage_representation in {"V_MAGNITUDE", "V_SQUARED"} and specification.voltage_baseline_pu is not None else voltage_increment
    branch_slack = tuple(
        _slack(quantity, budget, specification.branch_constraint_mode)
        for quantity, budget in zip(branch_quantity, branch_headroom)
    )
    voltage_slack = tuple(
        _slack(quantity, budget, specification.voltage_constraint_mode)
        for quantity, budget in zip(voltage_quantity, voltage_headroom)
    )
    branch_violation = tuple(max(0.0, -value) for value in branch_slack)
    voltage_violation = tuple(max(0.0, -value) for value in voltage_slack)
    reasons: list[Identifier] = []
    if any(budget < 0.0 for budget in branch_headroom):
        reasons.append(Identifier("NEGATIVE_BRANCH_TIGHTENED_BUDGET"))
    if any(budget < 0.0 for budget in voltage_headroom):
        reasons.append(Identifier("NEGATIVE_VOLTAGE_TIGHTENED_BUDGET"))
    branch_invalid = any(budget < 0.0 for budget in branch_headroom)
    voltage_invalid = any(budget < 0.0 for budget in voltage_headroom)
    branch_pass = not branch_invalid and all(
        _pass(quantity, budget, specification.branch_constraint_mode, specification.branch_tolerance_mw.value)
        for quantity, budget in zip(branch_quantity, branch_headroom)
    )
    voltage_pass = not voltage_invalid and all(
        _pass(quantity, budget, specification.voltage_constraint_mode, specification.voltage_tolerance_pu.value)
        for quantity, budget in zip(voltage_quantity, voltage_headroom)
    )
    if not branch_pass or not voltage_pass:
        if not reasons:
            reasons.append(Identifier("PROXY_FAIL"))
    status = "INVALID_BUDGET" if branch_invalid or voltage_invalid else ("VALID_FEASIBLE" if branch_pass and voltage_pass else "VALID_INFEASIBLE")
    branch_family_status = "INVALID_BUDGET" if branch_invalid else ("PASS" if branch_pass else "FAIL")
    voltage_family_status = "INVALID_BUDGET" if voltage_invalid else ("PASS" if voltage_pass else "FAIL")
    validity_status = "INVALID_BUDGET" if branch_invalid or voltage_invalid else "VALID"
    feasibility_status = "NOT_EVALUABLE" if validity_status != "VALID" else ("FEASIBLE" if branch_pass and voltage_pass else "INFEASIBLE")
    branch_tolerance = specification.branch_tolerance_mw.value
    voltage_tolerance = specification.voltage_tolerance_pu.value
    offending_branch = [
        constraint_id for constraint_id, violation in zip(specification.branch_constraint_ids.values, branch_violation)
        if violation > branch_tolerance
    ]
    offending_voltage = [
        constraint_id for constraint_id, violation in zip(specification.voltage_constraint_ids.values, voltage_violation)
        if violation > voltage_tolerance
    ]
    branch_absolute = (
        FloatVector(base + increment for base, increment in zip(specification.branch_baseline_mw.values, branch_increment))
        if specification.branch_baseline_mw is not None else None
    )
    voltage_absolute = (
        FloatVector(base + increment for base, increment in zip(specification.voltage_baseline_pu.values, voltage_increment))
        if specification.voltage_baseline_pu is not None else None
    )
    allocation_hash = Sha256(
        canonical_hash(
            {
                "participant_ids": allocation_participant_ids.to_json(),
                "values_mw": allocation_mw.to_json(),
            }
        )
    )
    return ProxyEvaluationResult(
        specification_hash=specification.specification_hash,
        allocation_hash=allocation_hash,
        branch_constraint_ids=specification.branch_constraint_ids,
        voltage_constraint_ids=specification.voltage_constraint_ids,
        branch_increment_mw=FloatVector(branch_increment),
        voltage_increment_pu=FloatVector(voltage_increment),
        branch_slack_mw=FloatVector(branch_slack),
        voltage_slack_pu=FloatVector(voltage_slack),
        branch_violation_mw=FloatVector(branch_violation),
        voltage_violation_pu=FloatVector(voltage_violation),
        max_branch_violation_mw=FiniteFloat(max(branch_violation, default=0.0)),
        max_voltage_violation_pu=FiniteFloat(max(voltage_violation, default=0.0)),
        branch_pass=branch_pass,
        voltage_pass=voltage_pass,
        overall_pass=branch_pass and voltage_pass,
        status=status,
        diagnostic_only=specification.scientific_status != "AUTHOR_FROZEN",
        evidence_role=specification.evidence_role,
        failure_reasons=IdentifierVector(reasons),
        branch_predicted_absolute_mw=branch_absolute,
        voltage_predicted_absolute_pu=voltage_absolute,
        branch_normalized_violation=_normalized(branch_violation, branch_headroom),
        voltage_normalized_violation=_normalized(voltage_violation, voltage_headroom),
        offending_branch_constraint_ids=IdentifierVector(offending_branch),
        offending_voltage_constraint_ids=IdentifierVector(offending_voltage),
        branch_family_status=branch_family_status,
        voltage_family_status=voltage_family_status,
        validity_status=validity_status,
        feasibility_status=feasibility_status,
    )
