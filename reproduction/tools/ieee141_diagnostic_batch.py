"""Explicit IEEE141 diagnostic-batch factory and multi-job execution helpers.

The existing five-method runner already validates one fully materialized
reference/reported pair.  This module supplies the missing experiment-facing
layer: a caller provides the normalization vector, physical screen, request
construction constants, and every scenario parameter explicitly; the factory
then builds typed jobs that are executed by the existing pair-batch grid.

The factory is diagnostic-only.  It does not calibrate a proxy, promote AC
results, or read historical CSV/figure/solver outputs.  A grid is therefore a
reproducible numerical implementation result, not manuscript evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from r4r.ac_solver import solve_ac_power_flow
from r4r.allocation_pipeline import ExplicitCapacityVector
from r4r.errors import ValidationError
from r4r.injection_adapter import build_net_injections_from_operating_point_spec
from r4r.models import ProxyModel
from r4r.network_parser import ParsedMatpowerCase
from r4r.objective_contracts import (
    FairnessObjectiveSpecification,
    MaxExportObjectiveSpecification,
    TieBreakSpecification,
)
from r4r.operating_point import build_canonical_operating_point
from r4r.proxy_evaluator import ProxyBudget, ProxyEvaluationSpecification
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.request_contracts import RawRequestVector
from r4r.request_generation import RequestConstruction, construct_requests
from r4r.request_policies import IdentityAdmissionPolicy, ExplicitReferenceRequestPolicy, materialize_strategic_report
from r4r.serialization import canonical_dumps, canonical_hash
from r4r.types import AllocationSide, FiniteFloat, FloatVector, Identifier, IdentifierVector, ObjectiveUnitPolicy, QAssumption, Sha256
from tools.ac_network_screen_diagnostics import ACNetworkScreenLimits
from tools.ac_proxy_diagnostics import build_ac_finite_difference_proxy, build_oriented_voltage_proxy_rows, reactive_spec_hash
from tools.diagnostic_ieee141_context import IEEE141DiagnosticInputContext, build_ieee141_diagnostic_input_context
from tools.diagnostic_pair_batch import (
    DiagnosticNetworkConstraintCore,
    DiagnosticPairBatchInputs,
    DiagnosticPhysicalLimits,
    DiagnosticReferenceReportedBatchGridResult,
    DiagnosticReferenceReportedBatchJob,
    DiagnosticSideBatchInputs,
    DiagnosticSideRequestBinding,
    build_side_diagnostic_domain,
    run_diagnostic_reference_reported_batch_grid,
)
from tools.diagnostic_margin_modes import DiagnosticMarginModeBundle
from tools.diagnostic_five_method_runner import DiagnosticParameterIdentity


EXPECTED_PARTICIPANT_COUNT = 30
EXPECTED_BUS_COUNT = 141
EXPECTED_BRANCH_COUNT = 140


@dataclass(frozen=True, slots=True)
class IEEE141BatchRequestConfig:
    """Caller-supplied request construction and normalization values."""

    normalized_participant_load_mw: FloatVector
    intercept_mw: FiniteFloat
    load_factor: FiniteFloat
    clip_lower_mw: FiniteFloat
    clip_upper_mw: FiniteFloat
    serialization_id = "ieee141_batch_request_config.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.normalized_participant_load_mw, FloatVector) or not self.normalized_participant_load_mw.values:
            raise ValidationError("request config requires a non-empty normalized load vector")
        for name in ("intercept_mw", "load_factor", "clip_lower_mw", "clip_upper_mw"):
            if not isinstance(getattr(self, name), FiniteFloat):
                raise ValidationError(f"request config {name} must be FiniteFloat")
        if self.clip_lower_mw.value < 0.0 or self.clip_upper_mw.value < self.clip_lower_mw.value:
            raise ValidationError("request config clipping bounds must satisfy 0 <= lower <= upper")
        if self.load_factor.value < 0.0:
            raise ValidationError("request config load_factor must be nonnegative")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "normalized_participant_load_mw": self.normalized_participant_load_mw.to_json(),
            "intercept_mw": self.intercept_mw.to_json(),
            "load_factor": self.load_factor.to_json(),
            "clip_lower_mw": self.clip_lower_mw.to_json(),
            "clip_upper_mw": self.clip_upper_mw.to_json(),
        }

    @property
    def config_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class IEEE141BatchScenario:
    """One explicit unilateral reference/reported parameter-set job."""

    scenario_id: Identifier
    policy_id: Identifier
    reporter_id: Identifier
    report_delta_mw: FiniteFloat
    capacity_values_mw: FloatVector
    delivery_cap_mw: FiniteFloat
    stress_multiplier: FiniteFloat
    alpha_mw: FiniteFloat
    epsilon_mw: FiniteFloat
    tie_break_tolerance_mw: FiniteFloat
    tie_break_weights: FloatVector
    capacity_spec_id: Identifier
    request_spec_id: Identifier
    # When present, the reported request is generated as gamma times the
    # reporter's raw reference request.  ``report_delta_mw`` remains a
    # backwards-compatible explicit-delta field for legacy diagnostic slices.
    gamma: FiniteFloat | None = None
    # Parameter-design provenance is carried with the scenario so downstream
    # five-method batches cannot lose the capacity-model and headroom axes.
    capacity_model_id: Identifier | None = None
    capacity_parameter_id: Identifier | None = None
    capacity_parameter_value: FiniteFloat | None = None
    headroom_ratio: FiniteFloat | None = None
    serialization_id = "ieee141_batch_scenario.v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("scenario_id", self.scenario_id),
            ("policy_id", self.policy_id),
            ("reporter_id", self.reporter_id),
            ("capacity_spec_id", self.capacity_spec_id),
            ("request_spec_id", self.request_spec_id),
        ):
            if not isinstance(value, Identifier):
                raise ValidationError(f"scenario {name} must be Identifier")
        for name in (
            "report_delta_mw",
            "delivery_cap_mw",
            "stress_multiplier",
            "alpha_mw",
            "epsilon_mw",
            "tie_break_tolerance_mw",
        ):
            if not isinstance(getattr(self, name), FiniteFloat):
                raise ValidationError(f"scenario {name} must be FiniteFloat")
        if self.gamma is None and self.report_delta_mw.value <= 0.0:
            raise ValidationError("scenario report_delta_mw must be positive when gamma is absent")
        if self.gamma is not None:
            if not isinstance(self.gamma, FiniteFloat) or self.gamma.value < 1.0:
                raise ValidationError("scenario gamma must be a finite multiplier >= 1")
        metadata = (
            self.capacity_model_id,
            self.capacity_parameter_id,
            self.capacity_parameter_value,
            self.headroom_ratio,
        )
        if any(value is not None for value in metadata):
            if not isinstance(self.capacity_model_id, Identifier) or not isinstance(self.capacity_parameter_id, Identifier):
                raise ValidationError("scenario capacity provenance requires model and parameter IDs")
            if not isinstance(self.capacity_parameter_value, FiniteFloat) or not isinstance(self.headroom_ratio, FiniteFloat):
                raise ValidationError("scenario capacity provenance requires typed value and headroom")
            if self.capacity_parameter_value.value < 0.0:
                raise ValidationError("scenario capacity parameter must be nonnegative")
            if not 0.0 <= self.headroom_ratio.value <= 1.0:
                raise ValidationError("scenario headroom_ratio must be within [0,1]")
        if self.delivery_cap_mw.value < 0.0:
            raise ValidationError("scenario delivery_cap_mw must be nonnegative")
        if self.stress_multiplier.value <= 0.0:
            raise ValidationError("scenario stress_multiplier must be positive")
        if self.alpha_mw.value < 0.0 or self.epsilon_mw.value < 0.0 or self.tie_break_tolerance_mw.value < 0.0:
            raise ValidationError("scenario objective tolerances must be nonnegative")
        if not isinstance(self.capacity_values_mw, FloatVector) or not self.capacity_values_mw.values:
            raise ValidationError("scenario capacity_values_mw must be a non-empty FloatVector")
        if any(value < 0.0 for value in self.capacity_values_mw.values):
            raise ValidationError("scenario capacity values must be nonnegative")
        if not isinstance(self.tie_break_weights, FloatVector) or not self.tie_break_weights.values:
            raise ValidationError("scenario tie-break weights must be a non-empty FloatVector")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "scenario_id": self.scenario_id.to_json(),
            "policy_id": self.policy_id.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "report_delta_mw": self.report_delta_mw.to_json(),
            "capacity_values_mw": self.capacity_values_mw.to_json(),
            "delivery_cap_mw": self.delivery_cap_mw.to_json(),
            "stress_multiplier": self.stress_multiplier.to_json(),
            "alpha_mw": self.alpha_mw.to_json(),
            "epsilon_mw": self.epsilon_mw.to_json(),
            "tie_break_tolerance_mw": self.tie_break_tolerance_mw.to_json(),
            "tie_break_weights": self.tie_break_weights.to_json(),
            "capacity_spec_id": self.capacity_spec_id.to_json(),
            "request_spec_id": self.request_spec_id.to_json(),
            "gamma": self.gamma.to_json() if self.gamma is not None else None,
            "capacity_model_id": self.capacity_model_id.to_json() if self.capacity_model_id else None,
            "capacity_parameter_id": self.capacity_parameter_id.to_json() if self.capacity_parameter_id else None,
            "capacity_parameter_value": self.capacity_parameter_value.to_json() if self.capacity_parameter_value else None,
            "headroom_ratio": self.headroom_ratio.to_json() if self.headroom_ratio else None,
        }

    @property
    def scenario_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class IEEE141DiagnosticBatchFactory:
    """Shared IEEE141 core plus explicit builders for many diagnostic jobs."""

    context: IEEE141DiagnosticInputContext
    physical_limits: DiagnosticPhysicalLimits
    request_config: IEEE141BatchRequestConfig
    core: DiagnosticNetworkConstraintCore
    serialization_id = "ieee141_diagnostic_batch_factory.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.context, IEEE141DiagnosticInputContext):
            raise ValidationError("factory context must be IEEE141DiagnosticInputContext")
        if not isinstance(self.physical_limits, DiagnosticPhysicalLimits):
            raise ValidationError("factory physical_limits must be DiagnosticPhysicalLimits")
        if not isinstance(self.request_config, IEEE141BatchRequestConfig):
            raise ValidationError("factory request_config must be IEEE141BatchRequestConfig")
        if not isinstance(self.core, DiagnosticNetworkConstraintCore):
            raise ValidationError("factory core must be DiagnosticNetworkConstraintCore")
        if len(self.context.selection.participants) != EXPECTED_PARTICIPANT_COUNT:
            raise ValidationError("IEEE141 factory requires 30 participants")
        if len(self.context.parsed_case.network.buses) != EXPECTED_BUS_COUNT or len(self.context.parsed_case.network.branches) != EXPECTED_BRANCH_COUNT:
            raise ValidationError("IEEE141 factory requires 141 buses and 140 branches")
        if len(self.request_config.normalized_participant_load_mw.values) != EXPECTED_PARTICIPANT_COUNT:
            raise ValidationError("normalized load vector must contain 30 participant values")
        if self.context.reactive_specification.q_mode_id not in {QAssumption.Q0, QAssumption.Q95}:
            raise ValidationError("IEEE141 diagnostic batch factory requires Q0 or Q95")
        if self.core.network_hash != self.context.network_hash or self.core.participant_ids != self.context.participant_ids:
            raise ValidationError("factory core is not bound to the context")

    @classmethod
    def build(
        cls,
        case_path: str | Path,
        *,
        rating_mw_by_bus: Mapping[int, float],
        availability_mw_by_bus: Mapping[int, float],
        reactive_specification: ReactiveInjectionSpecification,
        physical_limits: DiagnosticPhysicalLimits,
        request_config: IEEE141BatchRequestConfig,
        finite_difference_workers: int = 1,
    ) -> "IEEE141DiagnosticBatchFactory":
        """Construct the shared baseline, AC-derived proxy and typed core once."""

        if not isinstance(reactive_specification, ReactiveInjectionSpecification):
            raise ValidationError("reactive_specification must be typed")
        if reactive_specification.q_mode_id not in {QAssumption.Q0, QAssumption.Q95}:
            raise ValidationError("IEEE141 diagnostic factory requires Q0 or Q95")
        context = build_ieee141_diagnostic_input_context(
            case_path,
            rating_mw_by_bus=rating_mw_by_bus,
            availability_mw_by_bus=availability_mw_by_bus,
            reactive_specification=reactive_specification,
        )
        if len(physical_limits.branch_budget_mw.values) != EXPECTED_BRANCH_COUNT:
            raise ValidationError("branch budget must contain 140 values")
        if len(physical_limits.voltage_lower_limit_pu.values) != EXPECTED_BUS_COUNT or len(physical_limits.voltage_upper_limit_pu.values) != EXPECTED_BUS_COUNT:
            raise ValidationError("voltage limits must contain 141 values")
        if len(request_config.normalized_participant_load_mw.values) != EXPECTED_PARTICIPANT_COUNT:
            raise ValidationError("normalized load vector must contain 30 values")

        ids = context.participant_ids
        branch_ids = IdentifierVector(
            Identifier(f"branch_{index:03d}")
            for index in range(1, EXPECTED_BRANCH_COUNT + 1)
        )
        bus_ids = IdentifierVector(Identifier(str(index)) for index in range(1, EXPECTED_BUS_COUNT + 1))
        q_hash = reactive_spec_hash(reactive_specification)
        base_spec = ProxyEvaluationSpecification(
            specification_id=Identifier("proxy-ieee141-full-diagnostic"),
            model_family=Identifier("EXPLICIT_LINEAR_SENSITIVITY"),
            scientific_status="CANDIDATE",
            participant_ids=ids,
            branch_constraint_mode="SIGNED_UPPER_BOUND",
            voltage_constraint_mode="SIGNED_UPPER_BOUND",
            equation_ids=IdentifierVector([Identifier("EQ003"), Identifier("EQ004")]),
            participant_registry_id=Identifier("participant-registry-ieee141"),
            network_id=Identifier("ieee141"),
            network_hash=context.network_hash,
            operating_point_id=Identifier("ieee141-scale-070"),
            operating_point_hash=context.operating_point_hash,
            load_scale=FiniteFloat(0.70),
            q_assumption_id=reactive_specification.q_mode_id,
            q_spec_hash=q_hash,
            evidence_role="DIAGNOSTIC_ONLY",
            variable_unit="MW",
            variable_semantics="EXPORT_ALLOCATION",
            upper_bound_source="EXPLICIT",
            branch_constraint_ids=branch_ids,
            voltage_constraint_ids=bus_ids,
            branch_evaluation_form="INCREMENTAL_BUDGET",
            branch_flow_quantity="FROM_END_ACTIVE_POWER",
            voltage_representation="DELTA_V",
            participant_registry_hash=context.participant_registry_hash,
        )
        finite = build_ac_finite_difference_proxy(
            context.parsed_case,
            context.selection,
            base_spec,
            reactive_specification,
            ids,
            operating_point=context.operating_point,
            delta_mw=physical_limits.finite_difference_delta_mw.value,
            branch_indices=range(EXPECTED_BRANCH_COUNT),
            voltage_indices=range(EXPECTED_BUS_COUNT),
            max_workers=finite_difference_workers,
        )
        zero_injection = build_net_injections_from_operating_point_spec(
            context.parsed_case,
            context.selection,
            [0.0] * EXPECTED_PARTICIPANT_COUNT,
            context.operating_point,
            reactive_specification,
            formal_execution=False,
        )
        baseline = solve_ac_power_flow(
            context.parsed_case,
            p_injection_mw=zero_injection.p_mw,
            q_injection_mvar=zero_injection.q_mvar,
        )
        if not baseline.solver_result.solve_success:
            raise ValidationError("IEEE141 diagnostic baseline did not converge")
        oriented = build_oriented_voltage_proxy_rows(
            bus_ids=bus_ids,
            baseline_voltage_pu=baseline.bus_voltage_pu,
            lower_limit_pu=physical_limits.voltage_lower_limit_pu,
            upper_limit_pu=physical_limits.voltage_upper_limit_pu,
            voltage_sensitivity_matrix=finite.voltage_sensitivity_matrix,
        )
        proxy_spec = replace(
            base_spec,
            voltage_constraint_ids=oriented.constraint_ids,
            branch_tolerance_mw=physical_limits.branch_tolerance_mw,
            voltage_tolerance_pu=physical_limits.voltage_tolerance_pu,
        )
        branch_budget = ProxyBudget(
            physical_budget=physical_limits.branch_budget_mw,
            calibration_margin=FloatVector([0.0] * EXPECTED_BRANCH_COUNT),
            margin_multiplier=FiniteFloat(0.0),
            tightened_budget=physical_limits.branch_budget_mw,
            unit="MW",
            margin_mode_id=Identifier("M0"),
        )
        voltage_budget = ProxyBudget(
            physical_budget=oriented.headroom_pu,
            calibration_margin=FloatVector([0.0] * len(oriented.headroom_pu.values)),
            margin_multiplier=FiniteFloat(0.0),
            tightened_budget=oriented.headroom_pu,
            unit="pu",
            margin_mode_id=Identifier("M0"),
        )
        proxy_spec = replace(
            proxy_spec,
            branch_budget_hash=branch_budget.budget_hash,
            voltage_budget_hash=voltage_budget.budget_hash,
        )
        proxy_model = ProxyModel(
            branch_sensitivity_matrix=finite.branch_sensitivity_matrix,
            voltage_sensitivity_matrix=oriented.sensitivity_matrix,
            branch_headroom_mw=physical_limits.branch_budget_mw,
            voltage_headroom_pu=oriented.headroom_pu,
            branch_orientation=branch_ids.values,
        )
        core = DiagnosticNetworkConstraintCore(
            participant_ids=ids,
            participant_registry_hash=context.participant_registry_hash,
            proxy_model=proxy_model,
            proxy_specification=proxy_spec,
            branch_budget=branch_budget,
            voltage_budget=voltage_budget,
            oriented_voltage_rows=oriented,
            ac_limits=physical_limits.to_ac_limits(),
            physical_limits=physical_limits,
            baseline_solution=baseline,
            network_hash=context.network_hash,
            operating_point_hash=context.operating_point_hash,
            q_spec_hash=q_hash,
            proxy_matrix_hash=finite.matrix_hash,
            physical_parameter_hash=physical_limits.parameter_hash,
            finite_difference_delta_mw=physical_limits.finite_difference_delta_mw,
            finite_difference_result=finite,
            feasibility_tolerance=physical_limits.feasibility_tolerance,
        )
        return cls(context=context, physical_limits=physical_limits, request_config=request_config, core=core)

    @property
    def factory_hash(self) -> Sha256:
        return Sha256(canonical_hash({
            "serialization_id": self.serialization_id,
            "context": self.context.to_json(),
            "physical_limits": self.physical_limits.to_json(),
            "request_config": self.request_config.to_json(),
            "core_hash": self.core.core_hash.to_json(),
        }))

    def for_margin_mode(self, bundle: DiagnosticMarginModeBundle) -> "IEEE141DiagnosticBatchFactory":
        """Return a diagnostic factory with one explicit proxy margin mode.

        The returned factory keeps the physical AC limits from this factory
        and changes only the proxy headroom.  A negative tightened budget is
        rejected by the typed core instead of being clipped.
        """
        if not isinstance(bundle, DiagnosticMarginModeBundle):
            raise ValidationError("for_margin_mode requires DiagnosticMarginModeBundle")
        return replace(self, core=self.core.with_margin_mode(bundle))

    def _capacity_hash(self, scenario: IEEE141BatchScenario) -> Sha256:
        return Sha256(canonical_hash({
            "capacity_spec_id": scenario.capacity_spec_id.to_json(),
            "participant_registry_hash": self.context.participant_registry_hash.to_json(),
            "values_mw": scenario.capacity_values_mw.to_json(),
        }))

    def _build_objective_specs(
        self,
        scenario: IEEE141BatchScenario,
        domain,
        *,
        side_token: str,
    ) -> tuple[MaxExportObjectiveSpecification, FairnessObjectiveSpecification]:
        ids = self.context.participant_ids
        tie = TieBreakSpecification(
            tie_break_id=Identifier(f"{scenario.policy_id.value}__tie_break"),
            participant_ids=ids,
            weights=scenario.tie_break_weights,
            tolerance_mw=scenario.tie_break_tolerance_mw,
            rule_id="INDEX_WEIGHTED",
            scientific_status="CANDIDATE",
            unresolved_fields=("diagnostic",),
        )
        max_spec = MaxExportObjectiveSpecification(
            objective_id=Identifier(f"{scenario.policy_id.value}__max_export__{side_token}"),
            participant_ids=ids,
            feasible_domain_hash=domain.domain_hash,
            equation_ids=IdentifierVector([Identifier("EQ021")]),
            tie_break=tie,
            primary_stage_name="MAX_TOTAL_EXPORT",
            secondary_stage_name="INDEX_WEIGHTED_TIE_BREAK",
            preserve_primary_value=True,
            scientific_status="CANDIDATE",
            unresolved_fields=("diagnostic",),
        )
        # The fairness reference is the endogenous result of this exact
        # max-export specification/domain pair.  Do not serialize the same
        # allocation as an ``EXTERNAL_REFERENCE``: doing so loses the proof
        # that the reference used the same feasible domain and tie policy.
        # The allocation remains available in the max-export solver result;
        # the fairness contract binds to the specification hash instead.
        fairness = FairnessObjectiveSpecification(
            objective_id=Identifier(f"{scenario.policy_id.value}__fairness__{side_token}"),
            participant_ids=ids,
            equation_ids=IdentifierVector([Identifier("EQ027")]),
            alpha_mw=scenario.alpha_mw,
            epsilon_mw=scenario.epsilon_mw,
            reference_mode="ENDOGENOUS_MAX_EXPORT",
            max_export_specification_hash=max_spec.specification_hash,
            external_reference_allocation_hash=None,
            objective_unit_policy=ObjectiveUnitPolicy.MIXED_COMPONENTS,
            scientific_status="CANDIDATE",
            unresolved_fields=("diagnostic",),
            feasible_domain_hash=domain.domain_hash,
        )
        return max_spec, fairness

    def _build_side(
        self,
        scenario: IEEE141BatchScenario,
        admitted_request,
        *,
        side: AllocationSide,
        request_source_kind: str,
        raw_request: RawRequestVector,
        request_construction: RequestConstruction | None,
        strategic_report=None,
    ) -> DiagnosticSideBatchInputs:
        capacity_hash = self._capacity_hash(scenario)
        # v2 typed deep binding: the capacity hash must identify the exact
        # reporter/Q/profile/source row that produced this vector.  The
        # source-grid hash itself is attached by the grid envelope later (it
        # would be circular to include it in the row payload).
        profile_hash = Sha256(canonical_hash({
            "operating_point_hash": self.context.operating_point_hash.to_json(),
            "q_spec_hash": reactive_spec_hash(self.context.reactive_specification),
            "scenario_id": scenario.scenario_id.to_json(),
            "capacity_spec_id": scenario.capacity_spec_id.to_json(),
        }))
        capacity = ExplicitCapacityVector(
            self.context.participant_ids,
            scenario.capacity_values_mw,
            capacity_hash,
            self.context.participant_registry_hash,
            reporter_id=scenario.reporter_id,
            q_mode=self.context.reactive_specification.q_mode_id,
            profile_hash=profile_hash,
            source_grid_row_id=scenario.scenario_id,
        )
        reduction = build_side_diagnostic_domain(
            self.core,
            admitted_request,
            capacity,
            side=side,
        )
        if reduction.status != "REDUCED_MONOTONE" or reduction.reduced_domain is None:
            raise ValidationError(f"{side.value} diagnostic domain reduction did not complete")
        max_spec, fairness_spec = self._build_objective_specs(
            scenario,
            reduction.reduced_domain,
            side_token=side.value,
        )
        return DiagnosticSideBatchInputs(
            domain_binding=reduction,
            request_binding=DiagnosticSideRequestBinding(
                admitted_request=admitted_request,
                request_source_kind=request_source_kind,
                request_construction=request_construction,
                raw_request=raw_request if request_source_kind == "REFERENCE_CONSTRUCTION" else None,
                strategic_report=strategic_report,
            ),
            max_export_specification=max_spec,
            fairness_specification=fairness_spec,
            scenario_id=Identifier(f"{scenario.scenario_id.value}__{side.value}"),
            side=side,
            request_construction=request_construction,
        )

    def build_job(
        self,
        scenario: IEEE141BatchScenario,
        *,
        calibration_manifest_hash: Sha256 | None = None,
    ) -> DiagnosticReferenceReportedBatchJob:
        """Materialize one typed job from explicit scenario parameters."""

        if not isinstance(scenario, IEEE141BatchScenario):
            raise ValidationError("factory scenario must be IEEE141BatchScenario")
        if calibration_manifest_hash is not None and not isinstance(calibration_manifest_hash, Sha256):
            raise ValidationError("calibration_manifest_hash must be Sha256 or null")
        ids = self.context.participant_ids
        if scenario.reporter_id not in ids.values:
            raise ValidationError("scenario reporter is not in the IEEE141 participant set")
        if len(scenario.capacity_values_mw.values) != EXPECTED_PARTICIPANT_COUNT:
            raise ValidationError("scenario capacity vector must contain 30 values")
        if len(scenario.tie_break_weights.values) != EXPECTED_PARTICIPANT_COUNT:
            raise ValidationError("scenario tie-break vector must contain 30 values")

        request = construct_requests(
            self.request_config.normalized_participant_load_mw.values,
            intercept=self.request_config.intercept_mw.value,
            load_factor=self.request_config.load_factor.value,
            clip_lower_mw=self.request_config.clip_lower_mw.value,
            clip_upper_mw=self.request_config.clip_upper_mw.value,
            stress_multiplier=scenario.stress_multiplier.value,
        )
        reference_raw = ExplicitReferenceRequestPolicy().build(
            ids,
            request.actual_request_mw.values,
            request_spec_id=scenario.request_spec_id,
            participant_registry_hash=self.context.participant_registry_hash,
        )
        capacity_hash = self._capacity_hash(scenario)
        reference_admitted = IdentityAdmissionPolicy().admit(
            reference_raw,
            capacity_spec_hash=capacity_hash,
        )
        reference_side = self._build_side(
            scenario,
            reference_admitted,
            side=AllocationSide.REFERENCE,
            request_source_kind="REFERENCE_CONSTRUCTION",
            raw_request=reference_raw,
            request_construction=request,
        )
        reporter_index = ids.values.index(scenario.reporter_id)
        reported_values = list(request.actual_request_mw.values)
        if scenario.gamma is not None:
            reported_values[reporter_index] = (
                scenario.gamma.value * request.actual_request_mw.values[reporter_index]
            )
            effective_report_delta_mw = (
                reported_values[reporter_index] - request.actual_request_mw.values[reporter_index]
            )
            if effective_report_delta_mw <= 0.0:
                raise ValidationError("gamma scenario must create a positive reporter deviation")
        else:
            effective_report_delta_mw = scenario.report_delta_mw.value
            reported_values[reporter_index] += effective_report_delta_mw
        reported_report = materialize_strategic_report(
            reference_raw,
            scenario.reporter_id,
            reported_values,
            admitted_values_mw=reported_values,
            admission_policy_id=Identifier("IDENTITY_ADMISSION"),
            capacity_spec_hash=capacity_hash,
            clip_mask=[False] * EXPECTED_PARTICIPANT_COUNT,
            clip_amount_mw=[0.0] * EXPECTED_PARTICIPANT_COUNT,
            effective_deviation_status="UNRESOLVED",
            report_parameters={
                "report_delta_mw": effective_report_delta_mw,
                "gamma": scenario.gamma.value if scenario.gamma is not None else None,
            },
        )
        reported_side = self._build_side(
            scenario,
            reported_report.reported_admitted_request,
            side=AllocationSide.REPORTED,
            request_source_kind="STRATEGIC_REPORT",
            raw_request=reported_report.reported_raw_request,
            request_construction=None,
            strategic_report=reported_report,
        )
        if (
            scenario.capacity_model_id is None
            or scenario.capacity_parameter_id is None
            or scenario.capacity_parameter_value is None
            or scenario.headroom_ratio is None
        ):
            raise ValidationError("scenario lacks complete parameter identity provenance")
        pair_inputs = DiagnosticPairBatchInputs(
            core=self.core,
            reference_side=reference_side,
            reported_side=reported_side,
            physical_limits=self.physical_limits,
            finite_difference_result=self.core.finite_difference_result,
            reporter_id=scenario.reporter_id,
            delivery_cap_mw=scenario.delivery_cap_mw,
            parameter_identity=DiagnosticParameterIdentity(
                parameter_case_id=scenario.scenario_id,
                scenario_design_hash=scenario.scenario_hash,
                gamma=scenario.gamma if scenario.gamma is not None else FiniteFloat(1.0),
                gamma_source="DECLARED_GAMMA" if scenario.gamma is not None else "LEGACY_DELTA",
                stress_multiplier=scenario.stress_multiplier,
                headroom_ratio=scenario.headroom_ratio,
                capacity_model_id=scenario.capacity_model_id,
                capacity_parameter_id=scenario.capacity_parameter_id,
                capacity_parameter_value=scenario.capacity_parameter_value,
            ),
        )
        return DiagnosticReferenceReportedBatchJob(
            job_id=Identifier(f"{scenario.scenario_id.value}__job"),
            pair_inputs=pair_inputs,
            parsed_case=self.context.parsed_case,
            selection=self.context.selection,
            operating_point=self.context.operating_point,
            reactive_specification=self.context.reactive_specification,
            calibration_q=self.context.reactive_specification.q_mode_id,
            screening_q=self.context.reactive_specification.q_mode_id,
            calibration_manifest_hash=calibration_manifest_hash,
        )

    def build_jobs(
        self,
        scenarios: Sequence[IEEE141BatchScenario],
        *,
        calibration_manifest_hash: Sha256 | None = None,
    ) -> tuple[DiagnosticReferenceReportedBatchJob, ...]:
        scenarios_tuple = tuple(scenarios)
        if not scenarios_tuple:
            raise ValidationError("diagnostic batch scenario grid cannot be empty")
        if len({scenario.scenario_id for scenario in scenarios_tuple}) != len(scenarios_tuple):
            raise ValidationError("diagnostic batch scenario IDs must be unique")
        if calibration_manifest_hash is not None and not isinstance(calibration_manifest_hash, Sha256):
            raise ValidationError("calibration_manifest_hash must be Sha256 or null")
        return tuple(
            self.build_job(
                scenario,
                calibration_manifest_hash=calibration_manifest_hash,
            )
            for scenario in scenarios_tuple
        )

    def run_grid(
        self,
        scenarios: Sequence[IEEE141BatchScenario],
        *,
        grid_id: Identifier,
        max_workers: int = 1,
        calibration_manifest_hash: Sha256 | None = None,
    ) -> DiagnosticReferenceReportedBatchGridResult:
        jobs = self.build_jobs(
            scenarios,
            calibration_manifest_hash=calibration_manifest_hash,
        )
        return run_diagnostic_reference_reported_batch_grid(
            jobs,
            grid_id=grid_id,
            max_workers=max_workers,
        )

    def source_signature(self, scenarios: Sequence[IEEE141BatchScenario]) -> Sha256:
        """Return the compact input signature for an M0 calibration source.

        The signature covers the factory's typed network/operating-point/
        participant/physical/request configuration and the ordered scenario
        hashes.  It is a provenance shortcut for later diagnostic runners;
        the full grid hash remains the authoritative result hash.
        """

        scenarios_tuple = tuple(scenarios)
        if not scenarios_tuple:
            raise ValidationError("source signature requires non-empty scenarios")
        if any(not isinstance(scenario, IEEE141BatchScenario) for scenario in scenarios_tuple):
            raise ValidationError("source signature scenarios must be typed")
        return Sha256(canonical_hash({
            "serialization_id": "ieee141_m0_source_signature.v1",
            "factory_hash": self.factory_hash.to_json(),
            "scenario_hashes": [scenario.scenario_hash.to_json() for scenario in scenarios_tuple],
        }))


def write_diagnostic_grid_result(
    result: DiagnosticReferenceReportedBatchGridResult,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one complete diagnostic grid without editing or post-processing values."""

    if not isinstance(result, DiagnosticReferenceReportedBatchGridResult):
        raise ValidationError("result must be DiagnosticReferenceReportedBatchGridResult")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "serialization_id": "ieee141_diagnostic_grid_output.v1",
        "grid": result.to_json(),
        "grid_hash": result.grid_hash.to_json(),
    }
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValidationError("grid metadata must be a mapping")
        payload.update(dict(metadata))
    destination.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


def build_unilateral_reporter_scan_scenarios(
    factory: IEEE141DiagnosticBatchFactory,
    *,
    report_delta_mw: FiniteFloat,
    capacity_limited_mw: FiniteFloat,
    delivery_cap_mw: FiniteFloat,
    stress_multiplier: FiniteFloat,
    alpha_mw: FiniteFloat,
    epsilon_mw: FiniteFloat,
    tie_break_tolerance_mw: FiniteFloat,
    gamma: FiniteFloat | None = None,
) -> tuple[IEEE141BatchScenario, ...]:
    """Build one explicit unilateral scenario for every registered reporter.

    The caller supplies all numerical values.  The helper only expands the
    fixed IEEE141 participant registry into a deterministic location scan;
    it does not choose a reporter subset or infer a capacity policy.
    """

    if not isinstance(factory, IEEE141DiagnosticBatchFactory):
        raise ValidationError("reporter scan requires IEEE141DiagnosticBatchFactory")
    if not isinstance(capacity_limited_mw, FiniteFloat) or capacity_limited_mw.value < 0.0:
        raise ValidationError("capacity_limited_mw must be a nonnegative FiniteFloat")
    if gamma is not None and (not isinstance(gamma, FiniteFloat) or gamma.value < 1.0):
        raise ValidationError("gamma must be a finite multiplier >= 1")
    ids = factory.context.participant_ids
    scenarios: list[IEEE141BatchScenario] = []
    for index, reporter_id in enumerate(ids.values):
        capacity = [1.0] * len(ids.values)
        capacity[(index + 1) % len(capacity)] = capacity_limited_mw.value
        scenarios.append(
            IEEE141BatchScenario(
                scenario_id=Identifier(f"ieee141-reporter-scan-{index + 1:02d}"),
                policy_id=Identifier(f"ieee141-reporter-policy-{index + 1:02d}"),
                reporter_id=reporter_id,
                report_delta_mw=report_delta_mw,
                capacity_values_mw=FloatVector(capacity),
                delivery_cap_mw=delivery_cap_mw,
                stress_multiplier=stress_multiplier,
                alpha_mw=alpha_mw,
                epsilon_mw=epsilon_mw,
                tie_break_tolerance_mw=tie_break_tolerance_mw,
                tie_break_weights=FloatVector(float(i) for i in range(1, len(ids.values) + 1)),
                capacity_spec_id=Identifier(f"ieee141-reporter-capacity-{index + 1:02d}"),
                request_spec_id=Identifier(f"ieee141-reporter-request-{index + 1:02d}"),
                gamma=gamma,
                capacity_model_id=Identifier("EXPLICIT_SCALAR_CAPACITY"),
                capacity_parameter_id=Identifier("capacity_limited_mw"),
                capacity_parameter_value=capacity_limited_mw,
                headroom_ratio=FiniteFloat(0.0),
            )
        )
    return tuple(scenarios)


__all__ = [
    "EXPECTED_PARTICIPANT_COUNT",
    "EXPECTED_BUS_COUNT",
    "EXPECTED_BRANCH_COUNT",
    "IEEE141BatchRequestConfig",
    "IEEE141BatchScenario",
    "IEEE141DiagnosticBatchFactory",
    "build_unilateral_reporter_scan_scenarios",
    "write_diagnostic_grid_result",
]
