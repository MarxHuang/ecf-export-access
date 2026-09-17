"""Typed contracts for the next diagnostic reference/reported batch slice.

This module deliberately stops at the parameter and side-domain boundary.  A
network constraint core is shared by both sides, while each admitted request
gets its own allocation box and signed-row reduction certificate.  No default
physical limit or request value is inferred here.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

from r4r.allocation_pipeline import ExplicitCapacityVector
from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.errors import ValidationError
from r4r.models import OperatingPoint, ProxyModel, model_hash
from r4r.ac_solver import ACPowerFlowSolution
from r4r.network_parser import ParsedMatpowerCase
from r4r.participant_registry import ParticipantSelection
from r4r.proxy_evaluator import ProxyBudget, ProxyEvaluationSpecification
from r4r.request_contracts import AdmittedRequestVector, RawRequestVector, StrategicReportResult
from r4r.request_generation import RequestConstruction
from r4r.objective_contracts import FairnessObjectiveSpecification, MaxExportObjectiveSpecification
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, FiniteFloat, Identifier, IdentifierVector, Sha256, AllocationSide, QAssumption, MetricStatus
from r4r.metrics import normalized_jain, raw_jain
from r4r.models.evidence import JainMetricResult
from r4r.pair_metrics import (
    AccessTransferMetricResult,
    PAPER_REDISTRIBUTION_DEFINITION_ID,
    compute_access_transfer_metrics,
)
from tools.ac_network_screen_diagnostics import ACNetworkScreenLimits
from tools.ac_proxy_diagnostics import ACFiniteDifferenceProxyResult, OrientedVoltageProxyRows, reactive_spec_hash
from tools.diagnostic_feasible_domain import build_diagnostic_feasible_allocation_domain
from tools.diagnostic_signed_domain import SignedDomainReductionResult, reduce_signed_rows_for_fixed_regime
from tools.diagnostic_five_method_runner import DiagnosticMethodRunResult, DiagnosticParameterIdentity
from tools.diagnostic_five_method_profiles import build_ieee141_five_method_profiles
from tools.diagnostic_five_method_runner import DiagnosticFiveMethodBatchResult, run_ieee141_five_method_diagnostics
from tools.diagnostic_solver_allocation import DiagnosticSolverAllocationBinding
from tools.diagnostic_margin_modes import DiagnosticMarginModeBundle


@dataclass(frozen=True, slots=True)
class DiagnosticPhysicalLimits:
    """Explicit caller-supplied limits for one exploratory parameter set."""

    branch_budget_mw: FloatVector
    voltage_lower_limit_pu: FloatVector
    voltage_upper_limit_pu: FloatVector
    finite_difference_delta_mw: FiniteFloat
    feasibility_tolerance: FiniteFloat
    branch_tolerance_mw: FiniteFloat = FiniteFloat(0.0)
    voltage_tolerance_pu: FiniteFloat = FiniteFloat(0.0)
    # The manuscript's physical screen observes the oriented MATPOWER
    # sending-end active-power increment.  Keep the historical diagnostic
    # default for old fixtures, while allowing the candidate rebuild to bind
    # the explicit paper-aligned observable without changing a budget vector.
    branch_observable_mode: str = "BOTH_END_MAX_INCREMENT"
    status: str = "DIAGNOSTIC_PARAMETER_SET"
    serialization_id = "diagnostic_physical_limits.v1"

    def __post_init__(self) -> None:
        for name in (
            "branch_budget_mw",
            "voltage_lower_limit_pu",
            "voltage_upper_limit_pu",
        ):
            value = getattr(self, name)
            if not isinstance(value, FloatVector) or not value.values:
                raise ValidationError(f"{name} must be a non-empty FloatVector")
        if len(self.voltage_lower_limit_pu.values) != len(self.voltage_upper_limit_pu.values):
            raise ValidationError("voltage limits must have the same length")
        if any(value < 0.0 for value in self.branch_budget_mw.values):
            raise ValidationError("branch budgets must be nonnegative")
        if any(lower > upper for lower, upper in zip(self.voltage_lower_limit_pu.values, self.voltage_upper_limit_pu.values)):
            raise ValidationError("voltage lower limits cannot exceed upper limits")
        if self.finite_difference_delta_mw.value <= 0.0:
            raise ValidationError("finite-difference delta must be positive")
        if self.feasibility_tolerance.value < 0.0 or self.branch_tolerance_mw.value < 0.0 or self.voltage_tolerance_pu.value < 0.0:
            raise ValidationError("diagnostic tolerances must be nonnegative")
        if self.branch_observable_mode not in {
            "FROM_END_INCREMENT", "TO_END_INCREMENT", "BOTH_END_MAX_INCREMENT",
            "FROM_END_SIGNED_INCREMENT", "TO_END_SIGNED_INCREMENT",
        }:
            raise ValidationError("diagnostic branch observable mode is not registered")
        if self.status != "DIAGNOSTIC_PARAMETER_SET":
            raise ValidationError("physical limits cannot be promoted by this contract")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "branch_budget_mw": self.branch_budget_mw.to_json(),
            "voltage_lower_limit_pu": self.voltage_lower_limit_pu.to_json(),
            "voltage_upper_limit_pu": self.voltage_upper_limit_pu.to_json(),
            "finite_difference_delta_mw": self.finite_difference_delta_mw.to_json(),
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
            "branch_tolerance_mw": self.branch_tolerance_mw.to_json(),
            "voltage_tolerance_pu": self.voltage_tolerance_pu.to_json(),
            "branch_observable_mode": self.branch_observable_mode,
            "status": self.status,
        }

    @property
    def parameter_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def to_ac_limits(self) -> ACNetworkScreenLimits:
        return ACNetworkScreenLimits(
            branch_budget_mw=self.branch_budget_mw,
            voltage_lower_limit_pu=self.voltage_lower_limit_pu,
            voltage_upper_limit_pu=self.voltage_upper_limit_pu,
            branch_tolerance_mw=self.branch_tolerance_mw,
            voltage_tolerance_pu=self.voltage_tolerance_pu,
            branch_observable_mode=self.branch_observable_mode,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticNetworkConstraintCore:
    """Common network/proxy inputs shared by reference and reported sides."""

    participant_ids: IdentifierVector
    participant_registry_hash: Sha256
    proxy_model: ProxyModel
    proxy_specification: ProxyEvaluationSpecification
    branch_budget: ProxyBudget
    voltage_budget: ProxyBudget
    oriented_voltage_rows: OrientedVoltageProxyRows
    ac_limits: ACNetworkScreenLimits
    physical_limits: DiagnosticPhysicalLimits
    baseline_solution: ACPowerFlowSolution
    network_hash: Sha256
    operating_point_hash: Sha256
    q_spec_hash: Sha256
    proxy_matrix_hash: Sha256
    physical_parameter_hash: Sha256
    finite_difference_delta_mw: FiniteFloat
    finite_difference_result: ACFiniteDifferenceProxyResult
    feasibility_tolerance: FiniteFloat
    serialization_id = "diagnostic_network_constraint_core.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("core participant IDs must be non-empty")
        for name, value in (
            ("participant_registry_hash", self.participant_registry_hash),
            ("network_hash", self.network_hash),
            ("operating_point_hash", self.operating_point_hash),
            ("q_spec_hash", self.q_spec_hash),
            ("proxy_matrix_hash", self.proxy_matrix_hash),
            ("physical_parameter_hash", self.physical_parameter_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if not isinstance(self.proxy_model, ProxyModel) or not isinstance(self.proxy_specification, ProxyEvaluationSpecification):
            raise ValidationError("core requires typed proxy inputs")
        if not isinstance(self.branch_budget, ProxyBudget) or not isinstance(self.voltage_budget, ProxyBudget):
            raise ValidationError("core requires typed proxy budgets")
        if not isinstance(self.oriented_voltage_rows, OrientedVoltageProxyRows):
            raise ValidationError("core requires oriented voltage rows")
        if not isinstance(self.ac_limits, ACNetworkScreenLimits):
            raise ValidationError("core requires typed AC limits")
        if not isinstance(self.physical_limits, DiagnosticPhysicalLimits):
            raise ValidationError("core requires the typed physical-limit object")
        if not isinstance(self.baseline_solution, ACPowerFlowSolution) or not self.baseline_solution.solver_result.solve_success:
            raise ValidationError("core requires a converged typed baseline solution")
        if not isinstance(self.finite_difference_result, ACFiniteDifferenceProxyResult):
            raise ValidationError("core requires the typed finite-difference proxy result")
        if self.finite_difference_result.matrix_hash != self.proxy_matrix_hash:
            raise ValidationError("proxy matrix hash does not match finite-difference result")
        if self.finite_difference_result.delta_mw != self.finite_difference_delta_mw:
            raise ValidationError("finite-difference delta does not match proxy result")
        if self.finite_difference_result.participant_ids != self.participant_ids:
            raise ValidationError("finite-difference participant order does not match core")
        if self.finite_difference_result.network_hash != self.network_hash or self.finite_difference_result.operating_point_hash != self.operating_point_hash:
            raise ValidationError("finite-difference network/operating-point provenance differs")
        if self.finite_difference_result.participant_registry_hash != self.participant_registry_hash:
            raise ValidationError("finite-difference participant registry differs")
        if self.finite_difference_result.q_assumption_id is not self.proxy_specification.q_assumption_id:
            raise ValidationError("finite-difference Q assumption differs from proxy specification")
        if self.finite_difference_result.baseline_solution_hash != self.baseline_solution.solution_hash:
            raise ValidationError("finite-difference baseline differs from core baseline")
        if self.baseline_solution.bus_voltage_pu != self.oriented_voltage_rows.baseline_voltage_pu:
            raise ValidationError("oriented voltage rows do not use the core baseline voltages")
        if self.physical_limits.parameter_hash != self.physical_parameter_hash:
            raise ValidationError("physical parameter hash does not match typed physical limits")
        if self.physical_limits.finite_difference_delta_mw != self.finite_difference_delta_mw:
            raise ValidationError("physical limits delta does not match core delta")
        if self.physical_limits.feasibility_tolerance != self.feasibility_tolerance:
            raise ValidationError("physical limits tolerance does not match core tolerance")
        if self.physical_limits.to_ac_limits() != self.ac_limits:
            raise ValidationError("AC limits do not match typed physical limits")
        if self.finite_difference_result.branch_constraint_ids != self.proxy_specification.branch_constraint_ids:
            raise ValidationError("finite-difference branch row IDs do not match proxy rows")
        if self.finite_difference_result.voltage_constraint_ids != self.oriented_voltage_rows.bus_ids:
            raise ValidationError("finite-difference voltage row IDs do not match oriented bus order")
        if self.proxy_model.branch_sensitivity_matrix != self.finite_difference_result.branch_sensitivity_matrix:
            raise ValidationError("proxy branch matrix is not the finite-difference branch matrix")
        expected_oriented_voltage_rows = tuple(
            row
            for sensitivity in self.finite_difference_result.voltage_sensitivity_matrix.values
            for row in (sensitivity, tuple(-value for value in sensitivity))
        )
        if self.oriented_voltage_rows.sensitivity_matrix.values != expected_oriented_voltage_rows:
            raise ValidationError("oriented voltage rows are not derived from finite-difference voltage matrix")
        if self.proxy_model.voltage_sensitivity_matrix != self.oriented_voltage_rows.sensitivity_matrix:
            raise ValidationError("proxy voltage matrix is not the core oriented matrix")
        if self.proxy_specification.voltage_constraint_ids != self.oriented_voltage_rows.constraint_ids:
            raise ValidationError("proxy voltage row IDs do not match oriented rows")
        if self.proxy_model.branch_orientation != self.proxy_specification.branch_constraint_ids.values:
            raise ValidationError("proxy branch orientation does not match branch row order")
        if self.proxy_model.branch_headroom_mw != self.branch_budget.tightened_budget:
            raise ValidationError("proxy branch headroom does not match branch budget")
        if self.proxy_model.voltage_headroom_pu != self.oriented_voltage_rows.headroom_pu:
            raise ValidationError("proxy voltage headroom does not match oriented rows")
        if self.finite_difference_delta_mw.value <= 0.0:
            raise ValidationError("core finite-difference delta must be positive")
        if self.proxy_specification.participant_ids != self.participant_ids:
            raise ValidationError("proxy participant order does not match core")
        if self.proxy_specification.participant_registry_hash != self.participant_registry_hash:
            raise ValidationError("proxy participant registry does not match core")
        if self.proxy_specification.network_hash != self.network_hash:
            raise ValidationError("proxy network hash does not match core")
        if self.proxy_specification.operating_point_hash != self.operating_point_hash:
            raise ValidationError("proxy operating-point hash does not match core")
        if self.proxy_specification.q_spec_hash != self.q_spec_hash:
            raise ValidationError("proxy Q-spec hash does not match core")
        if self.proxy_specification.branch_budget_hash != self.branch_budget.budget_hash:
            raise ValidationError("branch budget does not match proxy specification")
        if self.proxy_specification.voltage_budget_hash != self.voltage_budget.budget_hash:
            raise ValidationError("voltage budget does not match proxy specification")
        if self.branch_budget.unit != "MW" or self.voltage_budget.unit != "pu":
            raise ValidationError("core budget units are not registered")
        if self.feasibility_tolerance.value < 0.0:
            raise ValidationError("core feasibility tolerance must be nonnegative")
        # The physical AC screen retains the physical branch budget.  The
        # proxy branch budget may be tightened by M1/M125 calibration margins;
        # these are distinct layers and must not be conflated.
        if self.ac_limits.branch_budget_mw != self.physical_limits.branch_budget_mw:
            raise ValidationError("AC branch limits and physical branch budget differ")
        if len(self.ac_limits.voltage_lower_limit_pu.values) != len(self.oriented_voltage_rows.bus_ids.values):
            raise ValidationError("AC voltage limits do not match oriented voltage rows")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "participant_ids": self.participant_ids.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "proxy_model_hash": Sha256(canonical_hash(self.proxy_model.to_json())).to_json(),
            "proxy_specification_hash": self.proxy_specification.specification_hash.to_json(),
            "branch_budget_hash": self.branch_budget.budget_hash.to_json(),
            "voltage_budget_hash": self.voltage_budget.budget_hash.to_json(),
            "oriented_voltage_rows_hash": self.oriented_voltage_rows.rows_hash.to_json(),
            "ac_limits_hash": self.ac_limits.limits_hash.to_json(),
            "physical_limits_hash": self.physical_limits.parameter_hash.to_json(),
            "baseline_solution_hash": self.baseline_solution.solution_hash.to_json(),
            "network_hash": self.network_hash.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "q_spec_hash": self.q_spec_hash.to_json(),
            "proxy_matrix_hash": self.proxy_matrix_hash.to_json(),
            "finite_difference_result_hash": self.finite_difference_result.matrix_hash.to_json(),
            "physical_parameter_hash": self.physical_parameter_hash.to_json(),
            "feasibility_tolerance": self.feasibility_tolerance.to_json(),
        }

    @property
    def core_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    @property
    def proxy_model_matrix_hash(self) -> Sha256:
        return Sha256(canonical_hash({
            "branch": self.proxy_model.branch_sensitivity_matrix.to_json(),
            "voltage": self.proxy_model.voltage_sensitivity_matrix.to_json(),
        }))

    def with_margin_mode(self, bundle: DiagnosticMarginModeBundle) -> "DiagnosticNetworkConstraintCore":
        """Derive a proxy core for one explicit M0/M1/M125 budget.

        Physical AC limits stay bound to ``physical_limits``.  Only the proxy
        headroom and proxy budget hashes change, so a calibrated mode cannot
        accidentally turn a proxy tightening into an AC limit change.
        Negative tightened budgets remain explicit invalid states and are not
        clipped or passed into an allocation domain.
        """
        if not isinstance(bundle, DiagnosticMarginModeBundle):
            raise ValidationError("margin mode core derivation requires a typed bundle")
        if bundle.budget_status != "VALID":
            raise ValidationError(f"margin mode {bundle.mode.value} has INVALID_BUDGET")
        if bundle.branch_physical_budget_mw != self.physical_limits.branch_budget_mw:
            raise ValidationError("margin mode branch physical budget differs from core")
        # ``oriented_voltage_rows.headroom_pu`` is proxy headroom and is
        # tightened on an M1/M125 derived core.  The only immutable physical
        # reference is the budget's physical vector; binding against rows
        # would allow a second margin subtraction.
        if bundle.voltage_physical_budget_pu != self.voltage_budget.physical_budget:
            raise ValidationError("margin mode voltage physical budget differs from core")
        oriented_rows = replace(
            self.oriented_voltage_rows,
            headroom_pu=bundle.voltage_budget.tightened_budget,
        )
        proxy_model = replace(
            self.proxy_model,
            branch_headroom_mw=bundle.branch_budget.tightened_budget,
            voltage_headroom_pu=bundle.voltage_budget.tightened_budget,
        )
        proxy_specification = replace(
            self.proxy_specification,
            branch_budget_hash=bundle.branch_budget.budget_hash,
            voltage_budget_hash=bundle.voltage_budget.budget_hash,
        )
        return replace(
            self,
            proxy_model=proxy_model,
            proxy_specification=proxy_specification,
            branch_budget=bundle.branch_budget,
            voltage_budget=bundle.voltage_budget,
            oriented_voltage_rows=oriented_rows,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticSideDomainBinding:
    """Non-splittable binding of one request/capacity box to the shared core."""

    core: DiagnosticNetworkConstraintCore
    admitted_request: AdmittedRequestVector
    capacity_vector: ExplicitCapacityVector
    side: AllocationSide
    expected_upper_bounds_mw: FloatVector
    reduction: SignedDomainReductionResult
    serialization_id = "diagnostic_side_domain_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.core, DiagnosticNetworkConstraintCore):
            raise ValidationError("side domain binding requires a typed core")
        if not isinstance(self.admitted_request, AdmittedRequestVector) or not isinstance(self.capacity_vector, ExplicitCapacityVector):
            raise ValidationError("side domain binding requires typed request and capacity")
        if not isinstance(self.side, AllocationSide) or self.admitted_request.source_side is not self.side:
            raise ValidationError("side domain binding request side does not match")
        if self.admitted_request.participant_ids != self.core.participant_ids or self.capacity_vector.participant_ids != self.core.participant_ids:
            raise ValidationError("side domain binding participant order does not match core")
        if self.admitted_request.participant_registry_hash != self.core.participant_registry_hash or self.capacity_vector.participant_registry_hash != self.core.participant_registry_hash:
            raise ValidationError("side domain binding participant registry does not match core")
        if self.admitted_request.capacity_spec_hash != self.capacity_vector.capacity_spec_hash:
            raise ValidationError("side domain binding capacity specification differs")
        if not isinstance(self.expected_upper_bounds_mw, FloatVector):
            raise ValidationError("side domain binding expected upper bounds must be FloatVector")
        expected = FloatVector(
            min(request, capacity)
            for request, capacity in zip(self.admitted_request.admitted_values_mw.values, self.capacity_vector.values_mw.values)
        )
        if self.expected_upper_bounds_mw != expected:
            raise ValidationError("side domain binding expected upper bounds are not min(request, capacity)")
        if not isinstance(self.reduction, SignedDomainReductionResult) or self.reduction.status != "REDUCED_MONOTONE":
            raise ValidationError("side domain binding requires a successful reduction")
        if self.reduction.full_domain is None or self.reduction.reduced_domain is None:
            raise ValidationError("side domain binding reduction lacks full/reduced domains")
        for name, domain in (("full", self.reduction.full_domain), ("reduced", self.reduction.reduced_domain)):
            if domain.participant_ids != self.core.participant_ids:
                raise ValidationError(f"{name} side domain participant order differs from core")
            if domain.request_hash != self.admitted_request.request_hash:
                raise ValidationError(f"{name} side domain request provenance differs")
            if domain.capacity_spec_hash != self.capacity_vector.capacity_spec_hash:
                raise ValidationError(f"{name} side domain capacity provenance differs")
            if domain.upper_bounds_mw != expected:
                raise ValidationError(f"{name} side domain upper bounds differ from min(request, capacity)")
            if domain.upper_bound_source != "MIN_CAPACITY_REQUEST":
                raise ValidationError(f"{name} side domain upper-bound source is not MIN_CAPACITY_REQUEST")
            if domain.proxy_spec_hash != self.core.proxy_specification.specification_hash:
                raise ValidationError(f"{name} side domain proxy provenance differs from core")
            if domain.feasibility_tolerance != self.core.feasibility_tolerance:
                raise ValidationError(f"{name} side domain tolerance differs from core")
        if self.reduction.full_domain.proxy_spec_hash != self.reduction.reduced_domain.proxy_spec_hash:
            raise ValidationError("side domain reduction changed proxy provenance")

    @property
    def core_hash(self) -> Sha256:
        return self.core.core_hash

    # Compatibility read-only views keep existing diagnostic callers from
    # unpacking the binding into independently replaceable pieces.
    @property
    def status(self) -> str:
        return self.reduction.status

    @property
    def full_domain(self) -> FeasibleAllocationDomain:
        return self.reduction.full_domain

    @property
    def reduced_domain(self) -> FeasibleAllocationDomain:
        return self.reduction.reduced_domain

    @property
    def certificates(self):
        return self.reduction.certificates

    @property
    def certificate_hash(self) -> Sha256:
        return self.reduction.certificate_hash

    @property
    def full_domain_hash(self) -> Sha256:
        return self.reduction.full_domain_hash

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "core_hash": self.core_hash.to_json(),
            "admitted_request_hash": self.admitted_request.request_hash.to_json(),
            "capacity_vector_hash": self.capacity_vector.capacity_vector_hash.to_json(),
            "side": self.side.value,
            "expected_upper_bounds_mw": self.expected_upper_bounds_mw.to_json(),
            "reduction": self.reduction.to_json(),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticSideRequestBinding:
    """Binds the request source instead of accepting a caller-supplied hash."""

    admitted_request: AdmittedRequestVector
    request_source_kind: Literal["REFERENCE_CONSTRUCTION", "STRATEGIC_REPORT"]
    request_construction: RequestConstruction | None = None
    strategic_report: StrategicReportResult | None = None
    # A reference construction has no request-contract object of its own, so
    # retain the exact raw reference request that it was built from.  For a
    # reported side the strategic report remains the sole source of truth and
    # this field must stay null.
    raw_request: RawRequestVector | None = None
    serialization_id = "diagnostic_side_request_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.admitted_request, AdmittedRequestVector):
            raise ValidationError("request binding requires a typed admitted request")
        if self.request_source_kind == "REFERENCE_CONSTRUCTION":
            if (
                not isinstance(self.request_construction, RequestConstruction)
                or not isinstance(self.raw_request, RawRequestVector)
                or self.strategic_report is not None
            ):
                raise ValidationError("reference request binding requires only RequestConstruction")
            if self.admitted_request.source_side is not AllocationSide.REFERENCE:
                raise ValidationError("reference request binding has wrong side")
            if self.raw_request.source_side is not AllocationSide.REFERENCE or self.raw_request.reporter_id is not None:
                raise ValidationError("reference raw request has wrong side or reporter")
            if self.request_construction.actual_request_mw != self.admitted_request.admitted_values_mw:
                raise ValidationError("reference construction does not match admitted request")
            if self.raw_request.participant_ids != self.admitted_request.participant_ids:
                raise ValidationError("reference raw request participant order differs")
            if self.raw_request.values_mw != self.admitted_request.raw_values_mw:
                raise ValidationError("reference raw request values differ from admitted request")
            if self.admitted_request.raw_request_hash != self.raw_request.request_hash:
                raise ValidationError("reference admitted request does not reference the bound raw request")
        elif self.request_source_kind == "STRATEGIC_REPORT":
            if not isinstance(self.strategic_report, StrategicReportResult):
                raise ValidationError("reported request binding requires StrategicReportResult")
            if self.request_construction is not None or self.raw_request is not None:
                raise ValidationError("reported request binding cannot carry an independent request source")
            if self.admitted_request.source_side is not AllocationSide.REPORTED:
                raise ValidationError("reported request binding has wrong side")
            if self.strategic_report.reported_admitted_request != self.admitted_request:
                raise ValidationError("reported request binding admitted request differs from strategic report")
        else:
            raise ValidationError("request source kind is not registered")

    @property
    def request_source_hash(self) -> Sha256:
        if self.request_source_kind == "REFERENCE_CONSTRUCTION":
            return self.request_construction.construction_hash
        return Sha256(model_hash(self.strategic_report))

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "request_source_kind": self.request_source_kind,
            "request_source_hash": self.request_source_hash.to_json(),
            "admitted_request_hash": self.admitted_request.request_hash.to_json(),
            "request_construction_hash": self.request_construction.construction_hash.to_json() if self.request_construction else None,
            "raw_request_hash": self.raw_request.request_hash.to_json() if self.raw_request else None,
            "strategic_report_hash": Sha256(model_hash(self.strategic_report)).to_json() if self.strategic_report else None,
        }

    @property
    def source_raw_request(self) -> RawRequestVector:
        """Return the raw request that is authoritative for this side."""
        if self.request_source_kind == "REFERENCE_CONSTRUCTION":
            # __post_init__ guarantees this is typed and present.
            return self.raw_request
        # __post_init__ guarantees the strategic report is typed and present.
        return self.strategic_report.reported_raw_request


def build_side_diagnostic_domain(
    core: DiagnosticNetworkConstraintCore,
    admitted_request: AdmittedRequestVector,
    capacity_vector: ExplicitCapacityVector,
    *,
    side: AllocationSide,
) -> DiagnosticSideDomainBinding:
    """Build and reduce one side's domain without copying the other side box."""
    if not isinstance(core, DiagnosticNetworkConstraintCore):
        raise ValidationError("core must be DiagnosticNetworkConstraintCore")
    if not isinstance(admitted_request, AdmittedRequestVector):
        raise ValidationError("side domain requires a typed admitted request")
    if not isinstance(capacity_vector, ExplicitCapacityVector):
        raise ValidationError("side domain requires a typed capacity vector")
    if not isinstance(side, AllocationSide) or admitted_request.source_side is not side:
        raise ValidationError("admitted request side does not match requested domain side")
    if admitted_request.participant_ids != core.participant_ids or capacity_vector.participant_ids != core.participant_ids:
        raise ValidationError("side domain participant order does not match core")
    if admitted_request.participant_registry_hash != core.participant_registry_hash or capacity_vector.participant_registry_hash != core.participant_registry_hash:
        raise ValidationError("side domain participant registry does not match core")
    if admitted_request.capacity_spec_hash != capacity_vector.capacity_spec_hash:
        raise ValidationError("side request and capacity specification differ")
    upper_bounds = [
        min(request, capacity)
        for request, capacity in zip(admitted_request.admitted_values_mw.values, capacity_vector.values_mw.values)
    ]
    full_domain = build_diagnostic_feasible_allocation_domain(
        participant_ids=core.participant_ids,
        participant_registry_hash=core.participant_registry_hash,
        proxy_model=core.proxy_model,
        proxy_specification=core.proxy_specification,
        branch_budget=core.branch_budget,
        voltage_budget=core.voltage_budget,
        lower_bounds_mw=[0.0] * len(core.participant_ids.values),
        upper_bounds_mw=upper_bounds,
        upper_bound_source="MIN_CAPACITY_REQUEST",
        feasibility_tolerance=core.feasibility_tolerance.value,
        capacity_spec_hash=capacity_vector.capacity_spec_hash,
        request_hash=admitted_request.request_hash,
        unresolved_fields=("diagnostic side-specific domain",),
    )
    reduction = reduce_signed_rows_for_fixed_regime(full_domain)
    return DiagnosticSideDomainBinding(
        core=core,
        admitted_request=admitted_request,
        capacity_vector=capacity_vector,
        side=side,
        expected_upper_bounds_mw=FloatVector(upper_bounds),
        reduction=reduction,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticSideBatchInputs:
    """All side-specific values required to run one five-method batch."""

    domain_binding: DiagnosticSideDomainBinding
    request_binding: DiagnosticSideRequestBinding
    max_export_specification: MaxExportObjectiveSpecification
    fairness_specification: FairnessObjectiveSpecification
    scenario_id: Identifier
    side: AllocationSide
    request_construction: RequestConstruction | None = None
    serialization_id = "diagnostic_side_batch_inputs.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.domain_binding, DiagnosticSideDomainBinding):
            raise ValidationError("side batch requires a non-splittable domain binding")
        if not isinstance(self.request_binding, DiagnosticSideRequestBinding):
            raise ValidationError("side batch requires a non-splittable request binding")
        if self.side is AllocationSide.REFERENCE:
            if not isinstance(self.request_construction, RequestConstruction):
                raise ValidationError("reference side requires its RequestConstruction source")
            if self.request_binding.request_source_kind != "REFERENCE_CONSTRUCTION":
                raise ValidationError("reference side request source kind is not construction")
            if self.request_binding.request_construction != self.request_construction:
                raise ValidationError("reference side construction is not the bound request source")
        elif self.request_construction is not None:
            raise ValidationError("reported side cannot carry an independent RequestConstruction")
        if not isinstance(self.max_export_specification, MaxExportObjectiveSpecification) or not isinstance(self.fairness_specification, FairnessObjectiveSpecification):
            raise ValidationError("side batch objective specifications are not typed")
        if not isinstance(self.scenario_id, Identifier) or not isinstance(self.side, AllocationSide):
            raise ValidationError("side batch scenario and side must be typed")
        if self.domain_binding.side is not self.side or self.request_binding.admitted_request.source_side is not self.side:
            raise ValidationError("side batch request side does not match side")
        if self.request_construction is not None and self.request_construction.actual_request_mw != self.request_binding.admitted_request.admitted_values_mw:
            raise ValidationError("side batch request construction does not match admitted request")
        if self.domain_binding.admitted_request != self.request_binding.admitted_request:
            raise ValidationError("side batch domain and request bindings do not join")
        if self.domain_binding.reduced_domain.participant_ids != self.request_binding.admitted_request.participant_ids:
            raise ValidationError("side batch domain and request participant order differs")
        if self.domain_binding.reduced_domain.request_hash != self.request_binding.admitted_request.request_hash:
            raise ValidationError("side batch domain and request provenance differs")
        side_domain = self.domain_binding.reduced_domain
        if self.max_export_specification.feasible_domain_hash != side_domain.domain_hash:
            raise ValidationError("max-export specification is not bound to side domain")
        if self.fairness_specification.feasible_domain_hash != side_domain.domain_hash:
            raise ValidationError("fairness specification is not bound to side domain")
        if self.max_export_specification.participant_ids != side_domain.participant_ids:
            raise ValidationError("max-export participant order does not match side domain")
        if self.fairness_specification.participant_ids != side_domain.participant_ids:
            raise ValidationError("fairness participant order does not match side domain")

    @property
    def reduction(self) -> DiagnosticSideDomainBinding:
        return self.domain_binding

    @property
    def admitted_request(self) -> AdmittedRequestVector:
        return self.request_binding.admitted_request

    @property
    def capacity_vector(self) -> ExplicitCapacityVector:
        return self.domain_binding.capacity_vector


def _max_export_policy_signature(specification: MaxExportObjectiveSpecification) -> Sha256:
    """Hash only the side-invariant max-export policy semantics.

    Objective IDs and feasible-domain hashes are intentionally excluded: they
    identify side-specific instances, whereas the pair must use one common
    two-stage policy.  The equation and deterministic tie-break fields remain
    in the signature because changing either changes the scientific contrast.
    """
    tie = specification.tie_break
    return Sha256(canonical_hash({
        "family": "MAX_EXPORT_TWO_STAGE",
        "participant_ids": specification.participant_ids.to_json(),
        "equation_ids": specification.equation_ids.to_json(),
        "primary_stage_name": specification.primary_stage_name,
        "secondary_stage_name": specification.secondary_stage_name,
        "preserve_primary_value": specification.preserve_primary_value,
        "tie_break": {
            "participant_ids": tie.participant_ids.to_json(),
            "weights": tie.weights.to_json() if tie.weights is not None else None,
            "tolerance_mw": tie.tolerance_mw.to_json(),
            "rule_id": tie.rule_id,
            "scientific_status": tie.scientific_status,
            "unresolved_fields": list(tie.unresolved_fields),
        },
        "scientific_status": specification.scientific_status,
        "unresolved_fields": list(specification.unresolved_fields),
    }))


def _fairness_policy_signature(specification: FairnessObjectiveSpecification) -> Sha256:
    """Hash the side-invariant fairness objective policy.

    Side-specific objective/domain and externally solved max-export allocation
    hashes are deliberately excluded; alpha, epsilon, reference mode, unit
    policy and equation registration are the policy semantics that must match.
    """
    return Sha256(canonical_hash({
        "family": "FAIRNESS_AWARE",
        "participant_ids": specification.participant_ids.to_json(),
        "equation_ids": specification.equation_ids.to_json(),
        "alpha_mw": specification.alpha_mw.to_json(),
        "epsilon_mw": specification.epsilon_mw.to_json(),
        "reference_mode": specification.reference_mode,
        "objective_unit_policy": specification.objective_unit_policy.value,
        "scientific_status": specification.scientific_status,
        "unresolved_fields": list(specification.unresolved_fields),
    }))


def _validate_batch_objective_bindings(
    batch: DiagnosticFiveMethodBatchResult,
    side_inputs: DiagnosticSideBatchInputs,
    *,
    side_label: str,
) -> None:
    """Bind declared side objectives to the actual solver-produced runs.

    Policy signatures establish that both sides intend the same method rule;
    complete specification hashes establish that the side input is the exact
    objective object used by its already executed max-export and fairness runs.
    """
    if len(batch.method_runs) != 5:
        raise ValidationError(f"{side_label} batch lacks the five required method runs")
    max_run = batch.method_runs[3]
    fairness_run = batch.method_runs[4]
    for run, expected_id, specification, label in (
        (max_run, "max_export_lp", side_inputs.max_export_specification, "max-export"),
        (fairness_run, "fairness_qp", side_inputs.fairness_specification, "fairness"),
    ):
        if run.method_id.value != expected_id:
            raise ValidationError(f"{side_label} actual method run is not the {label} objective")
        if not isinstance(run.profile.allocation, DiagnosticSolverAllocationBinding):
            raise ValidationError(f"{side_label} {label} method run lacks a solver allocation binding")
        actual_specification = run.profile.allocation.objective_specification
        if actual_specification.specification_hash != specification.specification_hash:
            raise ValidationError(f"{side_label} {label} objective specification differs from actual method run")
        if run.profile.objective_id != specification.objective_id:
            raise ValidationError(f"{side_label} {label} objective ID differs from actual method run")


@dataclass(frozen=True, slots=True)
class DiagnosticPairBatchInputs:
    """Complete typed input object for one reference/reported pair batch.

    The pair builder derives its parameter key from this object instead of
    accepting caller-supplied hashes for capacity, proxy, delta, or metrics.
    """

    core: DiagnosticNetworkConstraintCore
    reference_side: DiagnosticSideBatchInputs
    reported_side: DiagnosticSideBatchInputs
    physical_limits: DiagnosticPhysicalLimits
    finite_difference_result: ACFiniteDifferenceProxyResult
    reporter_id: Identifier
    delivery_cap_mw: FiniteFloat
    parameter_identity: DiagnosticParameterIdentity
    serialization_id = "diagnostic_pair_batch_inputs.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.core, DiagnosticNetworkConstraintCore):
            raise ValidationError("pair inputs require a typed common core")
        if not isinstance(self.reference_side, DiagnosticSideBatchInputs) or not isinstance(self.reported_side, DiagnosticSideBatchInputs):
            raise ValidationError("pair inputs require typed reference/reported side inputs")
        if self.reference_side.side is not AllocationSide.REFERENCE or self.reported_side.side is not AllocationSide.REPORTED:
            raise ValidationError("pair inputs require reference and reported sides")
        if self.reference_side.domain_binding.core_hash != self.core.core_hash or self.reported_side.domain_binding.core_hash != self.core.core_hash:
            raise ValidationError("pair side domains are not bound to the common core")
        if not isinstance(self.physical_limits, DiagnosticPhysicalLimits) or self.physical_limits.parameter_hash != self.core.physical_parameter_hash:
            raise ValidationError("pair physical limits do not match common core")
        if not isinstance(self.finite_difference_result, ACFiniteDifferenceProxyResult) or self.finite_difference_result != self.core.finite_difference_result:
            raise ValidationError("pair finite-difference result is not the core proxy result")
        if not isinstance(self.reporter_id, Identifier) or self.reporter_id not in self.core.participant_ids.values:
            raise ValidationError("pair reporter is not in the core participant set")
        if not isinstance(self.delivery_cap_mw, FiniteFloat) or self.delivery_cap_mw.value < 0.0:
            raise ValidationError("pair delivery cap must be a nonnegative FiniteFloat")
        if not isinstance(self.parameter_identity, DiagnosticParameterIdentity):
            raise ValidationError("pair inputs require typed parameter identity")
        left_capacity = self.reference_side.capacity_vector
        right_capacity = self.reported_side.capacity_vector
        if left_capacity != right_capacity:
            raise ValidationError("reference/reported pair must use the same typed capacity vector")
        reported_binding = self.reported_side.request_binding
        if reported_binding.request_source_kind != "STRATEGIC_REPORT" or not isinstance(reported_binding.strategic_report, StrategicReportResult):
            raise ValidationError("reported pair side must be bound to a StrategicReportResult")
        report = reported_binding.strategic_report
        if report.reporter_id != self.reporter_id:
            raise ValidationError("pair reporter does not match strategic report reporter")
        reference_raw = self.reference_side.request_binding.source_raw_request
        if report.reference_raw_request != reference_raw:
            raise ValidationError("strategic report reference raw request does not match reference side")
        reference_admitted = self.reference_side.request_binding.admitted_request
        if reference_admitted.raw_request_hash != reference_raw.request_hash or reference_admitted.raw_values_mw != reference_raw.values_mw:
            raise ValidationError("reference side admitted request is not bound to its raw request")
        if _max_export_policy_signature(self.reference_side.max_export_specification) != _max_export_policy_signature(self.reported_side.max_export_specification):
            raise ValidationError("reference/reported max-export objective policy differs")
        if _fairness_policy_signature(self.reference_side.fairness_specification) != _fairness_policy_signature(self.reported_side.fairness_specification):
            raise ValidationError("reference/reported fairness objective policy differs")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "core_hash": self.core.core_hash.to_json(),
            "reference_side": {
                "domain_hash": self.reference_side.domain_binding.reduction.reduced_domain.domain_hash.to_json(),
                "request_source_hash": self.reference_side.request_binding.request_source_hash.to_json(),
                # Persist the complete typed raw request, not only its digest.
                # A downstream verifier must be able to recompute the hash
                # from the registered RawRequestVector serialization and bind
                # the exact request to every method run.
                "raw_request": self.reference_side.request_binding.source_raw_request.to_json(),
                "max_export_specification": self.reference_side.max_export_specification.to_json(),
                "fairness_specification": self.reference_side.fairness_specification.to_json(),
            },
            "reported_side": {
                "domain_hash": self.reported_side.domain_binding.reduction.reduced_domain.domain_hash.to_json(),
                "request_source_hash": self.reported_side.request_binding.request_source_hash.to_json(),
                "raw_request": self.reported_side.request_binding.source_raw_request.to_json(),
                "max_export_specification": self.reported_side.max_export_specification.to_json(),
                "fairness_specification": self.reported_side.fairness_specification.to_json(),
            },
            "physical_limits": self.physical_limits.to_json(),
            "finite_difference_result": self.finite_difference_result.to_json(),
            "capacity_vector": self.reference_side.capacity_vector.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "delivery_cap_mw": self.delivery_cap_mw.to_json(),
            "parameter_identity": self.parameter_identity.to_json(),
            "max_export_policy_hash": _max_export_policy_signature(self.reference_side.max_export_specification).to_json(),
            "fairness_policy_hash": _fairness_policy_signature(self.reference_side.fairness_specification).to_json(),
        }

    @property
    def pair_inputs_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def run_diagnostic_side_batch(
    *,
    side_inputs: DiagnosticSideBatchInputs,
    parsed_case: ParsedMatpowerCase,
    selection: ParticipantSelection,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    parameter_identity: DiagnosticParameterIdentity,
) -> DiagnosticFiveMethodBatchResult:
    """Materialize exactly one side's five-method diagnostic batch."""
    if not isinstance(side_inputs, DiagnosticSideBatchInputs):
        raise ValidationError("side batch requires typed side inputs")
    if not isinstance(parameter_identity, DiagnosticParameterIdentity):
        raise ValidationError("side batch requires typed parameter identity")
    # The domain binding is the single source of the execution core.  Do not
    # accept a second caller-supplied core or baseline that could be spliced
    # into a domain built from a different finite-difference model.
    core = side_inputs.domain_binding.core
    if not isinstance(core, DiagnosticNetworkConstraintCore):
        raise ValidationError("side domain binding does not contain a typed core")
    if side_inputs.reduction.full_domain.proxy_spec_hash != core.proxy_specification.specification_hash:
        raise ValidationError("side reduction is not bound to the shared network core")
    domain = side_inputs.reduction.reduced_domain
    profiles = build_ieee141_five_method_profiles(
        domain=domain,
        request_values_mw=side_inputs.admitted_request.admitted_values_mw,
        request_source_hash=side_inputs.request_binding.request_source_hash,
        admitted_request=side_inputs.admitted_request,
        capacity_vector=side_inputs.capacity_vector,
        max_export_specification=side_inputs.max_export_specification,
        fairness_specification=side_inputs.fairness_specification,
        context_hash=core.proxy_matrix_hash,
        scenario_id=side_inputs.scenario_id,
        side=side_inputs.side,
    )
    return run_ieee141_five_method_diagnostics(
        parsed_case=parsed_case,
        selection=selection,
        operating_point=operating_point,
        reactive_specification=reactive_specification,
        scenario_id=side_inputs.scenario_id,
        batch_id=Identifier(f"{side_inputs.scenario_id.value}__{side_inputs.side.value}"),
        feasible_domain=domain,
        proxy_model=core.proxy_model,
        proxy_specification=core.proxy_specification,
        oriented_voltage_rows=core.oriented_voltage_rows,
        ac_limits=core.ac_limits,
        physical_branch_headroom_mw=core.physical_limits.branch_budget_mw,
        physical_voltage_headroom_pu=core.voltage_budget.physical_budget,
        proxy_branch_headroom_mw=core.branch_budget.tightened_budget,
        proxy_voltage_headroom_pu=core.voltage_budget.tightened_budget,
        profiles=profiles,
        shared_baseline_solution=core.baseline_solution,
        max_export_objective_id=side_inputs.max_export_specification.objective_id,
        fairness_objective_id=side_inputs.fairness_specification.objective_id,
        calibration_q=calibration_q,
        screening_q=screening_q,
        parameter_identity=parameter_identity,
        signed_domain_reduction=side_inputs.reduction.reduction,
        proxy_matrix_hash=core.proxy_matrix_hash,
        proxy_model_matrix_hash=core.proxy_model_matrix_hash,
        finite_difference_delta_mw=core.finite_difference_delta_mw,
        physical_parameter_hash=core.physical_parameter_hash,
        capacity_vector_hash=side_inputs.capacity_vector.capacity_vector_hash,
        request_source_hash=side_inputs.request_binding.request_source_hash,
    )


def _diagnostic_metric_spec_hash(
    reporter_id: Identifier,
    delivery_cap_mw: FiniteFloat,
    capacity_vector: ExplicitCapacityVector,
) -> Sha256:
    return Sha256(canonical_hash({
        "serialization_id": "diagnostic_metric_spec.v1",
        "reporter_id": reporter_id.to_json(),
        "delivery_cap_mw": delivery_cap_mw.to_json(),
        "capacity_vector": capacity_vector.to_json(),
        "jain_statuses": [status.value for status in MetricStatus],
        "normalized_jain_positive_capacity_only": True,
        "redistribution_definition_id": Identifier(PAPER_REDISTRIBUTION_DEFINITION_ID).to_json(),
        "epsilon": None,
        "allocation_clipping": False,
    }))


@dataclass(frozen=True, slots=True)
class DiagnosticMethodPairResult:
    """Pure allocation-stage pair metrics for one matched method."""

    method_id: Identifier
    reporter_id: Identifier
    reference_run: DiagnosticMethodRunResult
    reported_run: DiagnosticMethodRunResult
    delta_allocation_mw: FloatVector
    ol_nonreporters_mw: FiniteFloat
    sg_reporter_mw: FiniteFloat
    ud_reporter_mw: FiniteFloat
    ud_status: str
    reference_j_raw: JainMetricResult
    reported_j_raw: JainMetricResult
    reference_j_norm: JainMetricResult
    reported_j_norm: JainMetricResult
    access_transfer_metrics: AccessTransferMetricResult
    capacity_vector: ExplicitCapacityVector
    delivery_cap_mw: FiniteFloat
    capacity_vector_hash: Sha256
    metric_spec_hash: Sha256
    calibration_manifest_hash: Sha256 | None = None
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_method_pair_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, Identifier) or not isinstance(self.reporter_id, Identifier):
            raise ValidationError("pair method and reporter IDs must be typed")
        if not isinstance(self.reference_run, DiagnosticMethodRunResult) or not isinstance(self.reported_run, DiagnosticMethodRunResult):
            raise ValidationError("pair result requires atomic method runs")
        if self.reference_run.method_id != self.method_id or self.reported_run.method_id != self.method_id:
            raise ValidationError("pair method IDs do not align")
        if self.reference_run.profile.side is not AllocationSide.REFERENCE or self.reported_run.profile.side is not AllocationSide.REPORTED:
            raise ValidationError("pair sides must be reference and reported")
        if self.reference_run.profile.participant_ids != self.reported_run.profile.participant_ids:
            raise ValidationError("pair participant order does not align")
        if self.reporter_id not in self.reference_run.profile.participant_ids.values:
            raise ValidationError("pair reporter is not a participant")
        if len(self.delta_allocation_mw.values) != len(self.reference_run.profile.participant_ids.values):
            raise ValidationError("pair delta allocation does not align")
        if self.ud_status not in {"DEFINED", "UNRESOLVED"}:
            raise ValidationError("UD status is not registered")
        for name, value in (
            ("capacity_vector_hash", self.capacity_vector_hash),
            ("metric_spec_hash", self.metric_spec_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if self.calibration_manifest_hash is not None and not isinstance(self.calibration_manifest_hash, Sha256):
            raise ValidationError("calibration_manifest_hash must be Sha256 or null")
        if not isinstance(self.access_transfer_metrics, AccessTransferMetricResult):
            raise ValidationError("pair requires access-transfer metrics")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("method pair results cannot be promoted")
        if self.access_transfer_metrics.delta_mw != self.delta_allocation_mw:
            raise ValidationError("pair delta does not match access-transfer metrics")
        if self.access_transfer_metrics.sg_mw != self.sg_reporter_mw or self.access_transfer_metrics.ol_minus_reporter_mw != self.ol_nonreporters_mw:
            raise ValidationError("pair OL/SG values do not match access-transfer metrics")
        if self.access_transfer_metrics.ud_mw != self.ud_reporter_mw:
            raise ValidationError("pair UD value does not match access-transfer metrics")
        if not isinstance(self.capacity_vector, ExplicitCapacityVector) or not isinstance(self.delivery_cap_mw, FiniteFloat):
            raise ValidationError("pair result must retain the typed capacity and delivery cap used for metrics")
        if self.capacity_vector.participant_ids != self.reference_run.profile.participant_ids:
            raise ValidationError("pair capacity participant order does not match runs")
        if self.capacity_vector_hash != self.capacity_vector.capacity_vector_hash:
            raise ValidationError("pair capacity hash does not match the typed capacity vector")
        expected_transfer = compute_access_transfer_metrics(
            self.reference_run.profile.participant_ids,
            self.reference_run.profile.values_mw.values,
            self.reported_run.profile.values_mw.values,
            self.reporter_id,
            self.delivery_cap_mw.value,
            redistribution_definition_id=Identifier(PAPER_REDISTRIBUTION_DEFINITION_ID),
        )
        if expected_transfer != self.access_transfer_metrics:
            raise ValidationError("pair transfer metrics are not recomputed from the typed runs")
        if expected_transfer.delta_mw != self.delta_allocation_mw or expected_transfer.ol_minus_reporter_mw != self.ol_nonreporters_mw or expected_transfer.sg_mw != self.sg_reporter_mw or expected_transfer.ud_mw != self.ud_reporter_mw:
            raise ValidationError("pair transfer fields do not match recomputed metrics")
        expected_reference_raw = raw_jain(self.reference_run.profile.values_mw.values, Identifier("J_raw_reference"))
        expected_reported_raw = raw_jain(self.reported_run.profile.values_mw.values, Identifier("J_raw_reported"))
        expected_reference_norm = normalized_jain(self.reference_run.profile.values_mw.values, self.capacity_vector.values_mw.values, Identifier("J_norm_reference"))
        expected_reported_norm = normalized_jain(self.reported_run.profile.values_mw.values, self.capacity_vector.values_mw.values, Identifier("J_norm_reported"))
        if self.reference_j_raw.to_json() != expected_reference_raw.to_json() or self.reported_j_raw.to_json() != expected_reported_raw.to_json() or self.reference_j_norm.to_json() != expected_reference_norm.to_json() or self.reported_j_norm.to_json() != expected_reported_norm.to_json():
            raise ValidationError("pair Jain metrics are not recomputed from typed runs and capacity")
        expected_metric_hash = _diagnostic_metric_spec_hash(self.reporter_id, self.delivery_cap_mw, self.capacity_vector)
        if self.metric_spec_hash != expected_metric_hash:
            raise ValidationError("pair metric specification is not derived from typed inputs")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "method_id": self.method_id.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "reference_run_hash": self.reference_run.run_hash.to_json(),
            "reported_run_hash": self.reported_run.run_hash.to_json(),
            "reference_allocation_hash": self.reference_run.profile.allocation_hash.to_json(),
            "reported_allocation_hash": self.reported_run.profile.allocation_hash.to_json(),
            "delta_allocation_mw": self.delta_allocation_mw.to_json(),
            "ol_nonreporters_mw": self.ol_nonreporters_mw.to_json(),
            "sg_reporter_mw": self.sg_reporter_mw.to_json(),
            "ud_reporter_mw": self.ud_reporter_mw.to_json(),
            "ud_status": self.ud_status,
            "reference_j_raw": self.reference_j_raw.to_json(),
            "reported_j_raw": self.reported_j_raw.to_json(),
            "reference_j_norm": self.reference_j_norm.to_json(),
            "reported_j_norm": self.reported_j_norm.to_json(),
            "access_transfer_metrics": self.access_transfer_metrics.to_json(),
            "capacity_vector": self.capacity_vector.to_json(),
            "delivery_cap_mw": self.delivery_cap_mw.to_json(),
            "capacity_vector_hash": self.capacity_vector_hash.to_json(),
            "metric_spec_hash": self.metric_spec_hash.to_json(),
            "calibration_manifest_hash": self.calibration_manifest_hash.to_json() if self.calibration_manifest_hash else None,
            "status": self.status,
        }

    @property
    def pair_result_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def build_diagnostic_method_pair_result(
    reference_run: DiagnosticMethodRunResult,
    reported_run: DiagnosticMethodRunResult,
    *,
    reporter_id: Identifier,
    capacity_mw: ExplicitCapacityVector,
    delivery_cap_mw: FiniteFloat,
    calibration_manifest_hash: Sha256 | None = None,
) -> DiagnosticMethodPairResult:
    """Compute OL/SG/UD/Jain from the exact allocations used by both runs."""
    if not isinstance(capacity_mw, ExplicitCapacityVector) or not isinstance(delivery_cap_mw, FiniteFloat):
        raise ValidationError("pair metric capacity and delivery cap must be typed")
    if reference_run.method_id != reported_run.method_id:
        raise ValidationError("reference and reported method IDs must match")
    if reference_run.profile.participant_ids != reported_run.profile.participant_ids:
        raise ValidationError("reference and reported participant order must match")
    if reference_run.profile.side is not AllocationSide.REFERENCE or reported_run.profile.side is not AllocationSide.REPORTED:
        raise ValidationError("method pair requires reference/reported allocation sides")
    ids = reference_run.profile.participant_ids
    if capacity_mw.participant_ids != ids:
        raise ValidationError("capacity participant order does not match pair participants")
    if len(capacity_mw.values_mw.values) != len(ids.values):
        raise ValidationError("capacity vector must match pair participants")
    if reporter_id not in ids.values:
        raise ValidationError("reporter must be in pair participants")
    for name, left, right in (
        ("network", reference_run.proxy_audit.network_hash, reported_run.proxy_audit.network_hash),
        ("operating point", reference_run.proxy_audit.operating_point_hash, reported_run.proxy_audit.operating_point_hash),
        ("Q specification", reference_run.proxy_audit.q_spec_hash, reported_run.proxy_audit.q_spec_hash),
        ("proxy specification", reference_run.proxy_audit.proxy_specification_hash, reported_run.proxy_audit.proxy_specification_hash),
        ("baseline", reference_run.proxy_audit.baseline_solution_hash, reported_run.proxy_audit.baseline_solution_hash),
    ):
        if left != right:
            raise ValidationError(f"reference/reported {name} provenance differs")
    transfer = compute_access_transfer_metrics(
        ids,
        reference_run.profile.values_mw.values,
        reported_run.profile.values_mw.values,
        reporter_id,
        delivery_cap_mw.value,
        redistribution_definition_id=Identifier(PAPER_REDISTRIBUTION_DEFINITION_ID),
    )
    capacity_vector_hash = capacity_mw.capacity_vector_hash
    metric_spec_hash = _diagnostic_metric_spec_hash(reporter_id, delivery_cap_mw, capacity_mw)
    return DiagnosticMethodPairResult(
        method_id=reference_run.method_id,
        reporter_id=reporter_id,
        reference_run=reference_run,
        reported_run=reported_run,
        delta_allocation_mw=transfer.delta_mw,
        ol_nonreporters_mw=transfer.ol_minus_reporter_mw,
        sg_reporter_mw=transfer.sg_mw,
        ud_reporter_mw=transfer.ud_mw,
        ud_status="DEFINED",
        reference_j_raw=raw_jain(reference_run.profile.values_mw.values, Identifier("J_raw_reference")),
        reported_j_raw=raw_jain(reported_run.profile.values_mw.values, Identifier("J_raw_reported")),
        reference_j_norm=normalized_jain(reference_run.profile.values_mw.values, capacity_mw.values_mw.values, Identifier("J_norm_reference")),
        reported_j_norm=normalized_jain(reported_run.profile.values_mw.values, capacity_mw.values_mw.values, Identifier("J_norm_reported")),
        access_transfer_metrics=transfer,
        capacity_vector=capacity_mw,
        delivery_cap_mw=delivery_cap_mw,
        capacity_vector_hash=capacity_vector_hash,
        metric_spec_hash=metric_spec_hash,
        calibration_manifest_hash=calibration_manifest_hash,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticPairBatchParameterSet:
    """Typed key for one exploratory pair parameter set."""

    physical_parameter_hash: Sha256
    proxy_matrix_hash: Sha256
    finite_difference_delta_mw: FiniteFloat
    request_pair_hash: Sha256
    capacity_vector_hash: Sha256
    reporter_id: Identifier
    delivery_cap_mw: FiniteFloat
    reference_batch_hash: Sha256
    reported_batch_hash: Sha256
    reference_request_source_hash: Sha256
    reported_request_source_hash: Sha256
    reference_domain_hash: Sha256
    reported_domain_hash: Sha256
    pair_inputs_hash: Sha256
    serialization_id = "diagnostic_pair_batch_parameter_set.v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("physical_parameter_hash", self.physical_parameter_hash),
            ("proxy_matrix_hash", self.proxy_matrix_hash),
            ("request_pair_hash", self.request_pair_hash),
            ("capacity_vector_hash", self.capacity_vector_hash),
            ("reference_batch_hash", self.reference_batch_hash),
            ("reported_batch_hash", self.reported_batch_hash),
            ("reference_request_source_hash", self.reference_request_source_hash),
            ("reported_request_source_hash", self.reported_request_source_hash),
            ("reference_domain_hash", self.reference_domain_hash),
            ("reported_domain_hash", self.reported_domain_hash),
            ("pair_inputs_hash", self.pair_inputs_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if not isinstance(self.reporter_id, Identifier):
            raise ValidationError("reporter_id must be Identifier")
        if not isinstance(self.finite_difference_delta_mw, FiniteFloat) or not isinstance(self.delivery_cap_mw, FiniteFloat):
            raise ValidationError("parameter-set delta/cap must be FiniteFloat")
        if self.finite_difference_delta_mw.value <= 0.0 or self.delivery_cap_mw.value < 0.0:
            raise ValidationError("parameter-set delta/cap must be valid")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "physical_parameter_hash": self.physical_parameter_hash.to_json(),
            "proxy_matrix_hash": self.proxy_matrix_hash.to_json(),
            "finite_difference_delta_mw": self.finite_difference_delta_mw.to_json(),
            "request_pair_hash": self.request_pair_hash.to_json(),
            "capacity_vector_hash": self.capacity_vector_hash.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "delivery_cap_mw": self.delivery_cap_mw.to_json(),
            "reference_batch_hash": self.reference_batch_hash.to_json(),
            "reported_batch_hash": self.reported_batch_hash.to_json(),
            "reference_request_source_hash": self.reference_request_source_hash.to_json(),
            "reported_request_source_hash": self.reported_request_source_hash.to_json(),
            "reference_domain_hash": self.reference_domain_hash.to_json(),
            "reported_domain_hash": self.reported_domain_hash.to_json(),
            "pair_inputs_hash": self.pair_inputs_hash.to_json(),
        }

    @property
    def parameter_set_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class DiagnosticPairBatchResult:
    """Five method-pair joins for one explicit parameter set."""

    parameter_set: DiagnosticPairBatchParameterSet
    pair_inputs: DiagnosticPairBatchInputs
    reference_batch: DiagnosticFiveMethodBatchResult
    reported_batch: DiagnosticFiveMethodBatchResult
    reference_batch_hash: Sha256
    reported_batch_hash: Sha256
    method_pairs: tuple[DiagnosticMethodPairResult, ...]
    batch_job_id: Identifier | None = None
    # Populated only when this pair is explicitly joined to the calibration
    # manifest used by an evidence attachment.  A missing value remains valid
    # for ordinary diagnostic batches but cannot satisfy T3/T4 provenance.
    calibration_manifest_hash: Sha256 | None = None
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_pair_batch_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_set, DiagnosticPairBatchParameterSet):
            raise ValidationError("pair batch requires a typed parameter set")
        if not isinstance(self.pair_inputs, DiagnosticPairBatchInputs):
            raise ValidationError("pair batch requires the complete typed pair input object")
        if self.batch_job_id is not None and not isinstance(self.batch_job_id, Identifier):
            raise ValidationError("batch_job_id must be an Identifier or null")
        if self.calibration_manifest_hash is not None and not isinstance(self.calibration_manifest_hash, Sha256):
            raise ValidationError("calibration manifest hash must be Sha256 or null")
        if self.parameter_set.pair_inputs_hash != self.pair_inputs.pair_inputs_hash:
            raise ValidationError("pair parameter set is not bound to the typed pair inputs")
        if self.parameter_set.reporter_id != self.pair_inputs.reporter_id:
            raise ValidationError("pair parameter-set reporter does not match typed pair inputs")
        if self.parameter_set.delivery_cap_mw != self.pair_inputs.delivery_cap_mw:
            raise ValidationError("pair parameter-set delivery cap does not match typed pair inputs")
        if not isinstance(self.reference_batch, DiagnosticFiveMethodBatchResult) or not isinstance(self.reported_batch, DiagnosticFiveMethodBatchResult):
            raise ValidationError("pair batch requires typed reference/reported batches")
        _validate_batch_objective_bindings(
            self.reference_batch,
            self.pair_inputs.reference_side,
            side_label="reference",
        )
        _validate_batch_objective_bindings(
            self.reported_batch,
            self.pair_inputs.reported_side,
            side_label="reported",
        )
        for name, value in (
            ("reference_batch_hash", self.reference_batch_hash),
            ("reported_batch_hash", self.reported_batch_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if self.reference_batch.batch_hash != self.reference_batch_hash or self.reported_batch.batch_hash != self.reported_batch_hash:
            raise ValidationError("pair batch hashes do not match typed batches")
        if self.parameter_set.reference_batch_hash != self.reference_batch_hash or self.parameter_set.reported_batch_hash != self.reported_batch_hash:
            raise ValidationError("parameter set batch provenance does not match typed batches")
        if self.parameter_set.reference_domain_hash != self.reference_batch.feasible_domain_hash or self.parameter_set.reported_domain_hash != self.reported_batch.feasible_domain_hash:
            raise ValidationError("parameter set domain provenance does not match typed batches")
        if self.reference_batch.request_source_hash is None or self.reported_batch.request_source_hash is None:
            raise ValidationError("pair batches lack request-source provenance")
        if self.parameter_set.reference_request_source_hash != self.reference_batch.request_source_hash or self.parameter_set.reported_request_source_hash != self.reported_batch.request_source_hash:
            raise ValidationError("parameter set request-source provenance does not match typed batches")
        if len(self.method_pairs) != 5:
            raise ValidationError("pair batch must contain exactly five method pairs")
        if any(not isinstance(item, DiagnosticMethodPairResult) for item in self.method_pairs):
            raise ValidationError("pair batch entries must be typed method pairs")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("pair batches cannot be promoted")
        if len({item.method_id for item in self.method_pairs}) != 5:
            raise ValidationError("pair batch method IDs must be unique")
        if tuple(item.reference_run for item in self.method_pairs) != self.reference_batch.method_runs:
            raise ValidationError("pair reference runs are not the typed reference batch runs")
        if tuple(item.reported_run for item in self.method_pairs) != self.reported_batch.method_runs:
            raise ValidationError("pair reported runs are not the typed reported batch runs")
        if any(item.capacity_vector_hash != self.parameter_set.capacity_vector_hash for item in self.method_pairs):
            raise ValidationError("pair method capacity provenance differs from parameter set")
        if self.calibration_manifest_hash is not None and any(
            item.calibration_manifest_hash != self.calibration_manifest_hash
            for item in self.method_pairs
        ):
            raise ValidationError("pair method calibration manifest provenance differs from batch")
        if self.reference_batch.admitted_request_hash is None or self.reported_batch.admitted_request_hash is None:
            raise ValidationError("pair batches lack admitted-request provenance")
        expected_request_pair_hash = Sha256(canonical_hash({
            "reference_request_hash": self.reference_batch.admitted_request_hash.to_json(),
            "reported_request_hash": self.reported_batch.admitted_request_hash.to_json(),
        }))
        if self.parameter_set.request_pair_hash != expected_request_pair_hash:
            raise ValidationError("pair request hash does not match typed side batches")
        if self.parameter_set.physical_parameter_hash != self.pair_inputs.physical_limits.parameter_hash:
            raise ValidationError("pair physical parameter hash does not match typed pair inputs")
        if self.parameter_set.proxy_matrix_hash != self.pair_inputs.core.proxy_matrix_hash:
            raise ValidationError("pair proxy matrix hash does not match typed pair inputs")
        if self.parameter_set.finite_difference_delta_mw != self.pair_inputs.core.finite_difference_delta_mw:
            raise ValidationError("pair finite-difference delta does not match typed pair inputs")
        expected_capacity = self.pair_inputs.reference_side.capacity_vector.capacity_vector_hash
        if self.parameter_set.capacity_vector_hash != expected_capacity:
            raise ValidationError("pair capacity hash does not match typed pair inputs")
        if any(item.capacity_vector != self.pair_inputs.reference_side.capacity_vector for item in self.method_pairs):
            raise ValidationError("pair method capacity vectors do not match typed pair inputs")
        if any(item.reporter_id != self.pair_inputs.reporter_id or item.delivery_cap_mw != self.pair_inputs.delivery_cap_mw for item in self.method_pairs):
            raise ValidationError("pair method metrics use a different reporter or delivery cap")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "parameter_set": self.parameter_set.to_json(),
            "pair_inputs": self.pair_inputs.to_json(),
            # Retain the complete side batches in the persisted pair result.
            # The hashes above remain provenance fields, but are not a
            # substitute for the atomic runs, proxy audits, and AC screens
            # required to independently inspect a diagnostic grid.
            "reference_batch": self.reference_batch.to_json(),
            "reported_batch": self.reported_batch.to_json(),
            "reference_batch_hash": self.reference_batch_hash.to_json(),
            "reported_batch_hash": self.reported_batch_hash.to_json(),
            "method_pairs": [item.to_json() for item in self.method_pairs],
            "batch_job_id": self.batch_job_id.to_json() if self.batch_job_id else None,
            "calibration_manifest_hash": self.calibration_manifest_hash.to_json() if self.calibration_manifest_hash else None,
            "status": self.status,
        }

    @property
    def pair_batch_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    @property
    def parameter_set_hash(self) -> Sha256:
        return self.parameter_set.parameter_set_hash

    @property
    def request_pair_hash(self) -> Sha256:
        return self.parameter_set.request_pair_hash


def build_diagnostic_pair_batch(
    reference_batch: Any,
    reported_batch: Any,
    *,
    pair_inputs: DiagnosticPairBatchInputs,
    parameter_set_hash: Sha256 | None = None,
    proxy_matrix_hash: Sha256 | None = None,
    finite_difference_delta_mw: FiniteFloat | None = None,
    request_pair_hash: Sha256 | None = None,
    reporter_id: Identifier | None = None,
    capacity_mw: ExplicitCapacityVector | None = None,
    delivery_cap_mw: FiniteFloat | None = None,
    calibration_manifest_hash: Sha256 | None = None,
    job_id: Identifier | None = None,
) -> DiagnosticPairBatchResult:
    """Join two side batches using one complete, typed parameter object.

    The optional legacy arguments are accepted only as cross-checks; none is
    used as the source of a pair parameter or metric value.
    """
    # Local import avoids making the runner depend on this module while still
    # keeping the public input type explicit at runtime.
    from tools.diagnostic_five_method_runner import DiagnosticFiveMethodBatchResult

    if not isinstance(reference_batch, DiagnosticFiveMethodBatchResult) or not isinstance(reported_batch, DiagnosticFiveMethodBatchResult):
        raise ValidationError("pair batch requires two typed five-method batches")
    if not isinstance(pair_inputs, DiagnosticPairBatchInputs):
        raise ValidationError("pair batch requires complete typed pair inputs")
    if job_id is not None and not isinstance(job_id, Identifier):
        raise ValidationError("job_id must be an Identifier or null")
    if calibration_manifest_hash is not None and not isinstance(calibration_manifest_hash, Sha256):
        raise ValidationError("calibration_manifest_hash must be Sha256 or null")
    if reference_batch.feasible_domain_hash != pair_inputs.reference_side.domain_binding.reduced_domain.domain_hash or reported_batch.feasible_domain_hash != pair_inputs.reported_side.domain_binding.reduced_domain.domain_hash:
        raise ValidationError("pair batches do not match the typed side domains")
    if reference_batch.request_source_hash != pair_inputs.reference_side.request_binding.request_source_hash or reported_batch.request_source_hash != pair_inputs.reported_side.request_binding.request_source_hash:
        raise ValidationError("pair batches do not match the typed request sources")
    if reference_batch.method_ids != reported_batch.method_ids:
        raise ValidationError("reference and reported batch method order differs")
    _validate_batch_objective_bindings(
        reference_batch,
        pair_inputs.reference_side,
        side_label="reference",
    )
    _validate_batch_objective_bindings(
        reported_batch,
        pair_inputs.reported_side,
        side_label="reported",
    )
    if reference_batch.admitted_request_hash is None or reported_batch.admitted_request_hash is None:
        raise ValidationError("pair batches require admitted-request provenance")
    if reference_batch.request_source_hash is None or reported_batch.request_source_hash is None:
        raise ValidationError("pair batches require request-source provenance")
    expected_request_pair_hash = Sha256(canonical_hash({
        "reference_request_hash": reference_batch.admitted_request_hash.to_json(),
        "reported_request_hash": reported_batch.admitted_request_hash.to_json(),
    }))
    derived_request_pair_hash = expected_request_pair_hash
    if request_pair_hash is not None and request_pair_hash != derived_request_pair_hash:
        raise ValidationError("pair request hash does not match typed side batches")
    request_pair_hash = derived_request_pair_hash
    for name in (
        "network_hash",
        "operating_point_hash",
        "q_spec_hash",
        "proxy_specification_hash",
        "ac_limits_hash",
        "shared_baseline_solution_hash",
    ):
        if getattr(reference_batch, name) != getattr(reported_batch, name):
            raise ValidationError(f"reference and reported {name} differs")
    derived_parameter_hash = pair_inputs.physical_limits.parameter_hash
    derived_proxy_matrix_hash = pair_inputs.core.proxy_matrix_hash
    derived_delta = pair_inputs.core.finite_difference_delta_mw
    derived_capacity = pair_inputs.reference_side.capacity_vector
    derived_reporter = pair_inputs.reporter_id
    derived_delivery_cap = pair_inputs.delivery_cap_mw
    if parameter_set_hash is not None and parameter_set_hash != derived_parameter_hash:
        raise ValidationError("pair physical parameter hash does not match typed pair inputs")
    if proxy_matrix_hash is not None and proxy_matrix_hash != derived_proxy_matrix_hash:
        raise ValidationError("pair proxy matrix hash does not match typed pair inputs")
    if finite_difference_delta_mw is not None and finite_difference_delta_mw != derived_delta:
        raise ValidationError("pair finite-difference delta does not match typed pair inputs")
    if reporter_id is not None and reporter_id != derived_reporter:
        raise ValidationError("pair reporter does not match typed pair inputs")
    if capacity_mw is not None and capacity_mw != derived_capacity:
        raise ValidationError("pair capacity vector does not match typed pair inputs")
    if delivery_cap_mw is not None and delivery_cap_mw != derived_delivery_cap:
        raise ValidationError("pair delivery cap does not match typed pair inputs")
    parameter_set_hash = derived_parameter_hash
    proxy_matrix_hash = derived_proxy_matrix_hash
    finite_difference_delta_mw = derived_delta
    reporter_id = derived_reporter
    capacity_mw = derived_capacity
    delivery_cap_mw = derived_delivery_cap
    pair_ids = reference_batch.method_runs[0].profile.participant_ids
    if capacity_mw.participant_ids != pair_ids:
        raise ValidationError("pair capacity participant order does not match batches")
    for batch_name, batch in (("reference", reference_batch), ("reported", reported_batch)):
        if batch.proxy_matrix_hash is None or batch.physical_parameter_hash is None or batch.finite_difference_delta_mw is None:
            raise ValidationError(f"{batch_name} batch lacks key pair provenance")
        if batch.proxy_matrix_hash != proxy_matrix_hash:
            raise ValidationError(f"{batch_name} batch proxy matrix hash differs from pair parameter set")
        if batch.proxy_model_matrix_hash != pair_inputs.core.proxy_model_matrix_hash:
            raise ValidationError(f"{batch_name} batch proxy model matrix differs from typed pair inputs")
        if batch.physical_parameter_hash != parameter_set_hash:
            raise ValidationError(f"{batch_name} batch physical parameter hash differs from pair parameter set")
        if batch.finite_difference_delta_mw != finite_difference_delta_mw:
            raise ValidationError(f"{batch_name} batch finite-difference delta differs from pair parameter set")
        if batch.capacity_vector_hash != capacity_mw.capacity_vector_hash:
            raise ValidationError(f"{batch_name} batch capacity vector differs from pair capacity")
    capacity_vector_hash = capacity_mw.capacity_vector_hash
    parameter_set = DiagnosticPairBatchParameterSet(
        physical_parameter_hash=parameter_set_hash,
        proxy_matrix_hash=proxy_matrix_hash,
        finite_difference_delta_mw=finite_difference_delta_mw,
        request_pair_hash=request_pair_hash,
        capacity_vector_hash=capacity_vector_hash,
        reporter_id=reporter_id,
        delivery_cap_mw=delivery_cap_mw,
        reference_batch_hash=reference_batch.batch_hash,
        reported_batch_hash=reported_batch.batch_hash,
        reference_request_source_hash=reference_batch.request_source_hash,
        reported_request_source_hash=reported_batch.request_source_hash,
        reference_domain_hash=reference_batch.feasible_domain_hash,
        reported_domain_hash=reported_batch.feasible_domain_hash,
        pair_inputs_hash=pair_inputs.pair_inputs_hash,
    )
    pairs = tuple(
        build_diagnostic_method_pair_result(
            reference_run=reference_run,
            reported_run=reported_run,
            reporter_id=reporter_id,
            capacity_mw=capacity_mw,
            delivery_cap_mw=delivery_cap_mw,
            calibration_manifest_hash=calibration_manifest_hash,
        )
        for reference_run, reported_run in zip(reference_batch.method_runs, reported_batch.method_runs)
    )
    return DiagnosticPairBatchResult(
        parameter_set=parameter_set,
        pair_inputs=pair_inputs,
        reference_batch=reference_batch,
        reported_batch=reported_batch,
        reference_batch_hash=reference_batch.batch_hash,
        reported_batch_hash=reported_batch.batch_hash,
        method_pairs=pairs,
        batch_job_id=job_id,
        calibration_manifest_hash=calibration_manifest_hash,
    )


def run_diagnostic_reference_reported_batch(
    *,
    pair_inputs: DiagnosticPairBatchInputs,
    parsed_case: ParsedMatpowerCase,
    selection: ParticipantSelection,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    calibration_manifest_hash: Sha256 | None = None,
    job_id: Identifier | None = None,
) -> DiagnosticPairBatchResult:
    """Execute one complete reference/reported five-method diagnostic batch.

    The two side runs share the typed pair input object and the same network,
    operating point, Q assumptions, and AC/proxy core.  No caller-supplied
    hashes, allocations, or metrics are accepted here; those are derived by
    ``build_diagnostic_pair_batch`` from the typed side results.  This is the
    batch execution entry point for explicit IEEE141 parameter sets, while
    remaining diagnostic-only and separate from calibration/evidence code.
    """
    if not isinstance(pair_inputs, DiagnosticPairBatchInputs):
        raise ValidationError("diagnostic batch requires typed pair inputs")
    if not isinstance(parsed_case, ParsedMatpowerCase):
        raise ValidationError("diagnostic batch requires a parsed MATPOWER case")
    if not isinstance(selection, ParticipantSelection):
        raise ValidationError("diagnostic batch requires a typed participant selection")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("diagnostic batch requires a typed operating point")
    if not isinstance(reactive_specification, ReactiveInjectionSpecification):
        raise ValidationError("diagnostic batch requires a typed reactive specification")
    if calibration_manifest_hash is not None and not isinstance(calibration_manifest_hash, Sha256):
        raise ValidationError("diagnostic batch calibration manifest hash must be Sha256 or null")
    if job_id is not None and not isinstance(job_id, Identifier):
        raise ValidationError("job_id must be an Identifier or null")
    reference_batch = run_diagnostic_side_batch(
        side_inputs=pair_inputs.reference_side,
        parsed_case=parsed_case,
        selection=selection,
        operating_point=operating_point,
        reactive_specification=reactive_specification,
        calibration_q=calibration_q,
        screening_q=screening_q,
        parameter_identity=pair_inputs.parameter_identity,
    )
    reported_batch = run_diagnostic_side_batch(
        side_inputs=pair_inputs.reported_side,
        parsed_case=parsed_case,
        selection=selection,
        operating_point=operating_point,
        reactive_specification=reactive_specification,
        calibration_q=calibration_q,
        screening_q=screening_q,
        parameter_identity=pair_inputs.parameter_identity,
    )
    return build_diagnostic_pair_batch(
        reference_batch,
        reported_batch,
        pair_inputs=pair_inputs,
        calibration_manifest_hash=calibration_manifest_hash,
        job_id=job_id,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticReferenceReportedBatchJob:
    """One explicit parameter-set job for the diagnostic batch grid."""

    job_id: Identifier
    pair_inputs: DiagnosticPairBatchInputs
    parsed_case: ParsedMatpowerCase
    selection: ParticipantSelection
    operating_point: OperatingPoint
    reactive_specification: ReactiveInjectionSpecification
    calibration_q: QAssumption
    screening_q: QAssumption
    # Optional for ordinary allocation diagnostics; required by the typed
    # evidence adapter when this job is used as a T0--T5 source.
    calibration_manifest_hash: Sha256 | None = None
    serialization_id = "diagnostic_reference_reported_batch_job.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, Identifier):
            raise ValidationError("batch job ID must be an Identifier")
        if not isinstance(self.pair_inputs, DiagnosticPairBatchInputs):
            raise ValidationError("batch job requires typed pair inputs")
        if not isinstance(self.parsed_case, ParsedMatpowerCase):
            raise ValidationError("batch job requires a parsed MATPOWER case")
        if not isinstance(self.selection, ParticipantSelection):
            raise ValidationError("batch job requires typed participant selection")
        if not isinstance(self.operating_point, OperatingPoint):
            raise ValidationError("batch job requires a typed operating point")
        if not isinstance(self.reactive_specification, ReactiveInjectionSpecification):
            raise ValidationError("batch job requires a typed reactive specification")
        if not isinstance(self.calibration_q, QAssumption) or not isinstance(self.screening_q, QAssumption):
            raise ValidationError("batch job Q assumptions must be registered")
        if self.calibration_manifest_hash is not None and not isinstance(self.calibration_manifest_hash, Sha256):
            raise ValidationError("batch job calibration manifest hash must be Sha256 or null")
        core = self.pair_inputs.core
        if core.network_hash != self.parsed_case.source_hash:
            raise ValidationError("batch job core network does not match parsed case")
        if core.participant_ids != self.selection.participant_set.participant_ids:
            raise ValidationError("batch job participant order does not match selection")
        if core.participant_registry_hash != Sha256(self.selection.alignment_hash):
            raise ValidationError("batch job participant registry does not match selection")
        if core.operating_point_hash != Sha256(model_hash(self.operating_point)):
            raise ValidationError("batch job operating point does not match core")
        if core.q_spec_hash != reactive_spec_hash(self.reactive_specification):
            raise ValidationError("batch job reactive specification does not match core")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "job_id": self.job_id.to_json(),
            "pair_inputs_hash": self.pair_inputs.pair_inputs_hash.to_json(),
            "network_hash": self.pair_inputs.core.network_hash.to_json(),
            "participant_registry_hash": self.pair_inputs.core.participant_registry_hash.to_json(),
            "operating_point_hash": self.pair_inputs.core.operating_point_hash.to_json(),
            "q_spec_hash": self.pair_inputs.core.q_spec_hash.to_json(),
            "calibration_q": self.calibration_q.value,
            "screening_q": self.screening_q.value,
            "calibration_manifest_hash": (
                self.calibration_manifest_hash.to_json()
                if self.calibration_manifest_hash is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticReferenceReportedBatchGridResult:
    """Retained results for an explicit diagnostic parameter-set grid."""

    grid_id: Identifier
    jobs: tuple[DiagnosticReferenceReportedBatchJob, ...]
    batches: tuple[DiagnosticPairBatchResult, ...]
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_reference_reported_batch_grid_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.grid_id, Identifier):
            raise ValidationError("batch grid ID must be an Identifier")
        if not self.jobs or any(not isinstance(job, DiagnosticReferenceReportedBatchJob) for job in self.jobs):
            raise ValidationError("batch grid requires non-empty typed jobs")
        if len(self.jobs) != len(self.batches):
            raise ValidationError("batch grid jobs and results must align")
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValidationError("batch grid job IDs must be unique")
        if len({job.pair_inputs.pair_inputs_hash for job in self.jobs}) != len(self.jobs):
            raise ValidationError("batch grid jobs must have unique parameter inputs")
        if any(not isinstance(batch, DiagnosticPairBatchResult) for batch in self.batches):
            raise ValidationError("batch grid entries must be typed pair results")
        for job, batch in zip(self.jobs, self.batches):
            if batch.batch_job_id != job.job_id:
                raise ValidationError("batch grid job/result IDs do not align")
            if batch.pair_inputs != job.pair_inputs:
                raise ValidationError("batch grid job/result pair inputs do not align")
            if batch.reference_batch.network_hash != job.parsed_case.source_hash or batch.reported_batch.network_hash != job.parsed_case.source_hash:
                raise ValidationError("batch grid result network does not match its job")
            expected_operating_hash = Sha256(model_hash(job.operating_point))
            if batch.reference_batch.operating_point_hash != expected_operating_hash or batch.reported_batch.operating_point_hash != expected_operating_hash:
                raise ValidationError("batch grid result operating point does not match its job")
            expected_q_hash = reactive_spec_hash(job.reactive_specification)
            if batch.reference_batch.q_spec_hash != expected_q_hash or batch.reported_batch.q_spec_hash != expected_q_hash:
                raise ValidationError("batch grid result Q specification does not match its job")
            if batch.calibration_manifest_hash != job.calibration_manifest_hash:
                raise ValidationError("batch grid result calibration manifest does not match its job")
        if len({batch.pair_inputs.pair_inputs_hash for batch in self.batches}) != len(self.batches):
            raise ValidationError("batch grid cannot repeat one parameter input")
        first_core = self.batches[0].pair_inputs.core
        for batch in self.batches[1:]:
            core = batch.pair_inputs.core
            if core.network_hash != first_core.network_hash:
                raise ValidationError("batch grid mixes network inputs")
            if core.participant_ids != first_core.participant_ids or core.participant_registry_hash != first_core.participant_registry_hash:
                raise ValidationError("batch grid mixes participant registries or orders")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("batch grids cannot be promoted")

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "grid_id": self.grid_id.to_json(),
            "jobs": [job.to_json() for job in self.jobs],
            "batches": [batch.to_json() for batch in self.batches],
            "status": self.status,
        }

    @property
    def grid_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def _run_diagnostic_reference_reported_batch_job(
    job: DiagnosticReferenceReportedBatchJob,
) -> DiagnosticPairBatchResult:
    """Process-pool entry point for one typed reporter job."""

    try:
        return run_diagnostic_reference_reported_batch(
            pair_inputs=job.pair_inputs,
            parsed_case=job.parsed_case,
            selection=job.selection,
            operating_point=job.operating_point,
            reactive_specification=job.reactive_specification,
            calibration_q=job.calibration_q,
            screening_q=job.screening_q,
            calibration_manifest_hash=job.calibration_manifest_hash,
            job_id=job.job_id,
        )
    except Exception as exc:
        raise ValidationError(
            f"diagnostic batch job {job.job_id.value} failed: {exc}"
        ) from exc


def run_diagnostic_reference_reported_batch_grid(
    jobs: Sequence[DiagnosticReferenceReportedBatchJob],
    *,
    grid_id: Identifier,
    max_workers: int = 1,
) -> DiagnosticReferenceReportedBatchGridResult:
    """Run and retain every explicit parameter-set job in a diagnostic grid.

    ``max_workers`` is an execution-only control.  Results are collected in
    the original job order, so the typed payload and grid hash are identical
    to the sequential path.  The default remains one worker for compatibility
    and deterministic resource use; callers may opt into threaded numerical
    diagnostics when the solver stack releases the GIL.
    """
    job_tuple = tuple(jobs)
    if not job_tuple or any(not isinstance(job, DiagnosticReferenceReportedBatchJob) for job in job_tuple):
        raise ValidationError("diagnostic batch grid requires typed non-empty jobs")
    if len({job.job_id for job in job_tuple}) != len(job_tuple):
        raise ValidationError("diagnostic batch job IDs must be unique")

    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValidationError("max_workers must be a positive integer")

    if max_workers == 1:
        batch_results = [
            _run_diagnostic_reference_reported_batch_job(job)
            for job in job_tuple
        ]
    else:
        # Pair jobs contain immutable mapping views in the typed objective
        # specifications, which are intentionally not pickled across a
        # Windows process boundary.  Keep this orchestration layer threaded;
        # the independent AC finite-difference participant workers use the
        # process pool above where their task tuple is fully serializable.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(
                executor.map(_run_diagnostic_reference_reported_batch_job, job_tuple)
            )
    batches = tuple(batch_results)
    return DiagnosticReferenceReportedBatchGridResult(
        grid_id=grid_id,
        jobs=job_tuple,
        batches=batches,
    )


__all__ = [
    "DiagnosticPhysicalLimits",
    "DiagnosticNetworkConstraintCore",
    "DiagnosticSideDomainBinding",
    "DiagnosticSideRequestBinding",
    "DiagnosticSideBatchInputs",
    "DiagnosticPairBatchInputs",
    "build_side_diagnostic_domain",
    "DiagnosticMethodPairResult",
    "build_diagnostic_method_pair_result",
    "DiagnosticPairBatchResult",
    "DiagnosticPairBatchParameterSet",
    "build_diagnostic_pair_batch",
    "run_diagnostic_reference_reported_batch",
    "DiagnosticReferenceReportedBatchJob",
    "DiagnosticReferenceReportedBatchGridResult",
    "run_diagnostic_reference_reported_batch_grid",
]
