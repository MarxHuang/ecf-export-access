"""Diagnostic-only five-method batch orchestration.

This module joins already-materialized algebraic/LP/QP profile bindings to the
common IEEE-141 proxy and full-network AC diagnostics.  It deliberately does
not calibrate, decide proxy/AC feasibility equivalence, or create manuscript
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from r4r.ac_solver import ACPowerFlowSolution, solve_ac_power_flow
from r4r.algebraic_allocation import AlgebraicAllocationBinding
from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.allocation_pipeline import DiagnosticProfileBindingProtocol
from r4r.errors import ValidationError
from r4r.injection_adapter import build_net_injections_from_operating_point_spec
from r4r.models import DiagnosticParameterIdentity, OperatingPoint, ProxyModel, model_hash
from r4r.network_parser import ParsedMatpowerCase
from r4r.objective_contracts import FairnessObjectiveSpecification
from r4r.participant_registry import ParticipantSelection
from r4r.proxy_evaluator import ProxyEvaluationSpecification
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.serialization import canonical_hash
from r4r.types import FiniteFloat, Identifier, IdentifierVector, QAssumption, Sha256
from tools.ac_network_screen_diagnostics import (
    ACNetworkScreenLimits,
    ACNetworkScreenResult,
    _screen_diagnostic_allocation_with_baseline,
)
from tools.ac_proxy_diagnostics import (
    ACProxyProfileDifferenceResult,
    OrientedVoltageProxyRows,
    audit_proxy_profiles_against_ac,
    reactive_spec_hash,
    validate_proxy_ac_screen_budget_alignment,
)
from tools.diagnostic_objective_problems import (
    DiagnosticFairnessQPBinding,
    DiagnosticMaxExportStage2ProblemBinding,
)
from tools.diagnostic_solver_allocation import DiagnosticSolverAllocationBinding
from tools.diagnostic_signed_domain import SignedDomainReductionResult
from tools.diagnostic_active_set import (
    DiagnosticAllocationActivityResult,
    build_diagnostic_allocation_activity,
)


EXPECTED_METHOD_IDS = frozenset({
    "weighted_proportional",
    "equal_kw_reduction",
    "flat_level",
    "max_export_lp",
    "fairness_qp",
})
EXPECTED_METHOD_ORDER = (
    "weighted_proportional",
    "equal_kw_reduction",
    "flat_level",
    "max_export_lp",
    "fairness_qp",
)


@dataclass(frozen=True, slots=True)
class DiagnosticMethodRunResult:
    """Atomic per-method join of allocation, proxy audit and AC screen."""

    method_id: Identifier
    profile: DiagnosticProfileBindingProtocol
    proxy_audit: ACProxyProfileDifferenceResult
    ac_screen: ACNetworkScreenResult
    status: str = "DIAGNOSTIC_ONLY"
    activity: DiagnosticAllocationActivityResult | None = None
    serialization_id = "diagnostic_method_run_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, Identifier):
            raise ValidationError("method run method_id must be Identifier")
        if not isinstance(self.profile, DiagnosticProfileBindingProtocol):
            raise ValidationError("method run requires a typed profile binding")
        if not isinstance(self.proxy_audit, ACProxyProfileDifferenceResult) or not isinstance(self.ac_screen, ACNetworkScreenResult):
            raise ValidationError("method run requires proxy and AC diagnostic results")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("method runs cannot be promoted")
        if self.profile.method_id != self.method_id or self.proxy_audit.method_id != self.method_id:
            raise ValidationError("method run method IDs do not align")
        for name, value in (
            ("allocation_hash", self.profile.allocation_hash),
            ("proxy allocation_hash", self.proxy_audit.allocation_hash),
            ("AC allocation_hash", self.ac_screen.allocation_hash),
        ):
            if value is None:
                raise ValidationError(f"{name} is required for an atomic method run")
        if self.profile.allocation_hash != self.proxy_audit.allocation_hash or self.profile.allocation_hash != self.ac_screen.allocation_hash:
            raise ValidationError("method run allocation provenance does not join")
        if self.proxy_audit.profile_solution_hash != self.ac_screen.solution_hash:
            raise ValidationError("method run proxy and AC solution provenance does not join")
        if self.proxy_audit.baseline_solution_hash != self.ac_screen.baseline_solution_hash:
            raise ValidationError("method run proxy and AC baseline provenance does not join")
        if self.proxy_audit.q_spec_hash != self.ac_screen.q_spec_hash:
            raise ValidationError("method run proxy and AC Q provenance does not join")
        if self.activity is not None:
            if not isinstance(self.activity, DiagnosticAllocationActivityResult):
                raise ValidationError("method run activity must be typed")
            if self.activity.source_allocation_hash != self.profile.allocation_hash:
                raise ValidationError("method run activity source allocation does not join profile")
            if self.activity.domain_hash != self.profile.domain_hash:
                raise ValidationError("method run activity domain does not join profile")
            if self.activity.participant_registry_hash != self.profile.participant_registry_hash:
                raise ValidationError("method run activity participant registry does not join profile")

    @property
    def run_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "method_id": self.method_id.to_json(),
            "profile": self.profile.to_json(),
            "proxy_audit": self.proxy_audit.to_json(),
            "ac_screen": self.ac_screen.to_json(),
            "status": self.status,
            "activity": self.activity.to_json() if self.activity is not None else None,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFiveMethodBatchResult:
    """Complete diagnostic output for one five-method scenario."""

    batch_id: Identifier
    scenario_id: Identifier
    method_ids: IdentifierVector
    proxy_audits: tuple[ACProxyProfileDifferenceResult, ...]
    ac_screens: tuple[tuple[Identifier, ACNetworkScreenResult], ...]
    shared_baseline_solution_hash: Sha256
    network_hash: Sha256
    operating_point_hash: Sha256
    q_spec_hash: Sha256
    feasible_domain_hash: Sha256
    proxy_specification_hash: Sha256
    ac_limits_hash: Sha256
    method_runs: tuple[DiagnosticMethodRunResult, ...]
    parameter_identity: DiagnosticParameterIdentity
    admitted_request_hash: Sha256 | None = None
    request_source_hash: Sha256 | None = None
    capacity_vector_hash: Sha256 | None = None
    # Key diagnostic provenance carried by each side batch.  These are kept
    # beside the atomic runs so a pair cannot substitute a matrix/delta or
    # physical parameter hash that was never used to produce the batch.
    proxy_matrix_hash: Sha256 | None = None
    proxy_model_matrix_hash: Sha256 | None = None
    finite_difference_delta_mw: FiniteFloat | None = None
    physical_parameter_hash: Sha256 | None = None
    signed_domain_reduction_certificate_hash: Sha256 | None = None
    full_screening_domain_hash: Sha256 | None = None
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_five_method_batch_result.v1"

    def __post_init__(self) -> None:
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("five-method batch results are diagnostic-only")
        if not isinstance(self.parameter_identity, DiagnosticParameterIdentity):
            raise ValidationError("five-method batch requires typed parameter identity")
        if len(self.method_ids.values) != 5 or len(set(self.method_ids.values)) != 5:
            raise ValidationError("five-method batch must contain five unique method IDs")
        if len(self.proxy_audits) != 5 or len(self.ac_screens) != 5:
            raise ValidationError("five-method batch output must contain five proxy and AC results")
        if any(result.status != "DIAGNOSTIC_ONLY" for result in self.proxy_audits):
            raise ValidationError("proxy audit result cannot be promoted")
        if any(result.status != "DIAGNOSTIC_ONLY" for _, result in self.ac_screens):
            raise ValidationError("AC screen result cannot be promoted")
        if tuple(result.method_id for result in self.proxy_audits) != self.method_ids.values:
            raise ValidationError("proxy audit method order does not match batch method order")
        if tuple(method_id for method_id, _ in self.ac_screens) != self.method_ids.values:
            raise ValidationError("AC screen method order does not match batch method order")
        if any(result.baseline_solution_hash != self.shared_baseline_solution_hash for result in self.proxy_audits):
            raise ValidationError("proxy audits do not share the batch baseline")
        if any(result.baseline_solution_hash != self.shared_baseline_solution_hash for _, result in self.ac_screens):
            raise ValidationError("AC screens do not share the batch baseline")
        if len(self.method_runs) != 5:
            raise ValidationError("five-method batch must contain five atomic method runs")
        if tuple(item.method_id for item in self.method_runs) != self.method_ids.values:
            raise ValidationError("atomic method runs do not match batch method order")
        for index, item in enumerate(self.method_runs):
            if item.proxy_audit != self.proxy_audits[index]:
                raise ValidationError("atomic method run proxy audit differs from batch audit")
            if item.ac_screen != self.ac_screens[index][1]:
                raise ValidationError("atomic method run AC screen differs from batch screen")
            if item.profile.domain_hash != self.feasible_domain_hash:
                raise ValidationError("atomic method profile domain differs from batch domain")
            if item.proxy_audit.network_hash != self.network_hash or item.proxy_audit.operating_point_hash != self.operating_point_hash or item.proxy_audit.q_spec_hash != self.q_spec_hash or item.proxy_audit.proxy_specification_hash != self.proxy_specification_hash:
                raise ValidationError("atomic method proxy provenance differs from batch provenance")
            if item.ac_screen.network_hash != self.network_hash or item.ac_screen.operating_point_hash != self.operating_point_hash or item.ac_screen.q_spec_hash != self.q_spec_hash or item.ac_screen.limits_hash != self.ac_limits_hash:
                raise ValidationError("atomic method AC provenance differs from batch provenance")
            if self.proxy_model_matrix_hash is not None and item.proxy_audit.proxy_matrix_hash != self.proxy_model_matrix_hash:
                raise ValidationError("atomic method proxy model matrix differs from batch provenance")
        if self.signed_domain_reduction_certificate_hash is not None and not isinstance(
            self.signed_domain_reduction_certificate_hash, Sha256
        ):
            raise ValidationError("signed-domain reduction certificate hash must be Sha256")
        if self.full_screening_domain_hash is not None and not isinstance(self.full_screening_domain_hash, Sha256):
            raise ValidationError("full screening-domain hash must be Sha256")
        if self.proxy_matrix_hash is not None and not isinstance(self.proxy_matrix_hash, Sha256):
            raise ValidationError("proxy matrix hash must be Sha256")
        if self.proxy_model_matrix_hash is not None and not isinstance(self.proxy_model_matrix_hash, Sha256):
            raise ValidationError("proxy model matrix hash must be Sha256")
        if self.admitted_request_hash is not None and not isinstance(self.admitted_request_hash, Sha256):
            raise ValidationError("admitted request hash must be Sha256")
        if self.request_source_hash is not None and not isinstance(self.request_source_hash, Sha256):
            raise ValidationError("request source hash must be Sha256")
        if self.capacity_vector_hash is not None and not isinstance(self.capacity_vector_hash, Sha256):
            raise ValidationError("capacity vector hash must be Sha256")
        if self.physical_parameter_hash is not None and not isinstance(self.physical_parameter_hash, Sha256):
            raise ValidationError("physical parameter hash must be Sha256")

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "batch_id": self.batch_id.to_json(),
            "scenario_id": self.scenario_id.to_json(),
            "method_ids": self.method_ids.to_json(),
            "proxy_audits": [result.to_json() for result in self.proxy_audits],
            "ac_screens": [
                {"method_id": method_id.to_json(), "result": result.to_json()}
                for method_id, result in self.ac_screens
            ],
            "shared_baseline_solution_hash": self.shared_baseline_solution_hash.to_json(),
            "network_hash": self.network_hash.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "q_spec_hash": self.q_spec_hash.to_json(),
            "feasible_domain_hash": self.feasible_domain_hash.to_json(),
            "proxy_specification_hash": self.proxy_specification_hash.to_json(),
            "ac_limits_hash": self.ac_limits_hash.to_json(),
            "method_runs": [item.to_json() for item in self.method_runs],
            "parameter_identity": self.parameter_identity.to_json(),
            "parameter_identity_hash": self.parameter_identity.identity_hash.to_json(),
            "admitted_request_hash": self.admitted_request_hash.to_json() if self.admitted_request_hash else None,
            "request_source_hash": self.request_source_hash.to_json() if self.request_source_hash else None,
            "capacity_vector_hash": self.capacity_vector_hash.to_json() if self.capacity_vector_hash else None,
            "proxy_matrix_hash": self.proxy_matrix_hash.to_json() if self.proxy_matrix_hash else None,
            "proxy_model_matrix_hash": self.proxy_model_matrix_hash.to_json() if self.proxy_model_matrix_hash else None,
            "finite_difference_delta_mw": (
                self.finite_difference_delta_mw.to_json()
                if self.finite_difference_delta_mw is not None else None
            ),
            "physical_parameter_hash": self.physical_parameter_hash.to_json() if self.physical_parameter_hash else None,
            "signed_domain_reduction_certificate_hash": (
                self.signed_domain_reduction_certificate_hash.to_json()
                if self.signed_domain_reduction_certificate_hash else None
            ),
            "full_screening_domain_hash": (
                self.full_screening_domain_hash.to_json()
                if self.full_screening_domain_hash else None
            ),
            "status": self.status,
        }

    @property
    def batch_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def run_ieee141_five_method_diagnostics(
    *,
    parsed_case: ParsedMatpowerCase,
    selection: ParticipantSelection,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    scenario_id: Identifier,
    batch_id: Identifier,
    feasible_domain: FeasibleAllocationDomain,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    oriented_voltage_rows: OrientedVoltageProxyRows,
    ac_limits: ACNetworkScreenLimits,
    profiles: Sequence[tuple[Identifier, DiagnosticProfileBindingProtocol]],
    shared_baseline_solution: ACPowerFlowSolution,
    max_export_objective_id: Identifier,
    fairness_objective_id: Identifier,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    parameter_identity: DiagnosticParameterIdentity,
    signed_domain_reduction: SignedDomainReductionResult | None = None,
    proxy_matrix_hash: Sha256 | None = None,
    proxy_model_matrix_hash: Sha256 | None = None,
    finite_difference_delta_mw: FiniteFloat | None = None,
    physical_parameter_hash: Sha256 | None = None,
    capacity_vector_hash: Sha256 | None = None,
    request_source_hash: Sha256,
    physical_branch_headroom_mw: FloatVector | None = None,
    physical_voltage_headroom_pu: FloatVector | None = None,
    proxy_branch_headroom_mw: FloatVector | None = None,
    proxy_voltage_headroom_pu: FloatVector | None = None,
) -> DiagnosticFiveMethodBatchResult:
    """Run residual and full-network AC diagnostics for the fixed five methods."""

    if not isinstance(scenario_id, Identifier) or not isinstance(batch_id, Identifier):
        raise ValidationError("batch and scenario IDs must be Identifier values")
    if not isinstance(parameter_identity, DiagnosticParameterIdentity):
        raise ValidationError("runner requires a typed parameter identity")
    if not isinstance(feasible_domain, FeasibleAllocationDomain):
        raise ValidationError("runner requires a typed feasible allocation domain")
    if feasible_domain.domain_hash is None:
        raise ValidationError("runner requires feasible-domain provenance")
    if signed_domain_reduction is not None:
        if not isinstance(signed_domain_reduction, SignedDomainReductionResult):
            raise ValidationError("runner reduction provenance must be a typed result")
        if signed_domain_reduction.status != "REDUCED_MONOTONE":
            raise ValidationError("runner cannot execute a domain without a successful reduction certificate")
        if signed_domain_reduction.reduced_domain is None:
            raise ValidationError("runner reduction provenance lacks its reduced domain")
        if signed_domain_reduction.reduced_domain.domain_hash != feasible_domain.domain_hash:
            raise ValidationError("runner domain does not match signed-domain reduction result")
        if signed_domain_reduction.full_domain.domain_hash != signed_domain_reduction.full_domain_hash:
            raise ValidationError("runner reduction source-domain hash is not self-consistent")
        full_ids = {row.constraint_id.value for row in signed_domain_reduction.full_domain.constraints}
        reduced_ids = {row.constraint_id.value for row in feasible_domain.constraints}
        removed_ids = {certificate.constraint_id.value for certificate in signed_domain_reduction.certificates}
        if reduced_ids & removed_ids or reduced_ids | removed_ids != full_ids:
            raise ValidationError("runner reduction certificate does not partition its full source domain")
        if any(certificate.status != "REDUNDANT" for certificate in signed_domain_reduction.certificates):
            raise ValidationError("runner reduction certificate contains a non-redundant removed row")
    elif "signed-row redundancy certificate retained" in feasible_domain.unresolved_fields:
        raise ValidationError("runner requires signed-domain reduction provenance for a reduced domain")
    if not isinstance(shared_baseline_solution, ACPowerFlowSolution) or not shared_baseline_solution.solver_result.solve_success:
        raise ValidationError("runner requires a converged shared baseline solution")
    if len(parsed_case.network.buses) != 141 or len(parsed_case.network.branches) != 140:
        raise ValidationError("IEEE141 runner requires 141 buses and 140 branches")
    if not isinstance(selection, ParticipantSelection):
        raise ValidationError("runner requires a typed participant selection")
    if len(selection.participants) != 30:
        raise ValidationError("IEEE141 runner requires exactly 30 participants")
    if len(profiles) != 5:
        raise ValidationError("runner requires exactly five method profiles")
    if not isinstance(proxy_specification, ProxyEvaluationSpecification):
        raise ValidationError("runner requires a typed proxy specification")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("runner requires a typed operating point")
    if not isinstance(reactive_specification, ReactiveInjectionSpecification):
        raise ValidationError("runner requires a typed reactive specification")
    if operating_point.load_scale.value != 0.70 or proxy_specification.load_scale.value != 0.70:
        raise ValidationError("IEEE141 runner load scale is frozen at 0.70")
    if reactive_specification.q_mode_id not in {QAssumption.Q0, QAssumption.Q95}:
        raise ValidationError("IEEE141 five-method runner requires Q0 or Q95")
    if not isinstance(max_export_objective_id, Identifier) or not isinstance(fairness_objective_id, Identifier):
        raise ValidationError("runner objective IDs must be Identifier values")
    if not isinstance(calibration_q, QAssumption) or not isinstance(screening_q, QAssumption):
        raise ValidationError("runner calibration and screening Q assumptions must be registered")
    if proxy_matrix_hash is not None and not isinstance(proxy_matrix_hash, Sha256):
        raise ValidationError("runner proxy matrix hash must be Sha256")
    if physical_parameter_hash is not None and not isinstance(physical_parameter_hash, Sha256):
        raise ValidationError("runner physical parameter hash must be Sha256")
    if capacity_vector_hash is not None and not isinstance(capacity_vector_hash, Sha256):
        raise ValidationError("runner capacity vector hash must be Sha256")
    if not isinstance(request_source_hash, Sha256):
        raise ValidationError("runner request source hash must be Sha256")
    if finite_difference_delta_mw is not None and not isinstance(finite_difference_delta_mw, FiniteFloat):
        raise ValidationError("runner finite-difference delta must be a typed finite value")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("IEEE141 runner operating point Q differs from reactive specification")
    if calibration_q not in {QAssumption.Q0, QAssumption.Q95}:
        raise ValidationError("IEEE141 runner calibration_q must be Q0 or Q95")
    if screening_q not in {QAssumption.Q0, QAssumption.Q95}:
        raise ValidationError("IEEE141 runner screening_q must be Q0 or Q95")
    if calibration_q is not screening_q:
        raise ValidationError("IEEE141 five-method runner currently requires matched-Q inputs")
    if screening_q is not reactive_specification.q_mode_id:
        raise ValidationError("IEEE141 runner screening_q differs from reactive specification")
    if proxy_specification.network_hash != parsed_case.source_hash:
        raise ValidationError("runner network provenance does not match parsed case")
    if proxy_specification.operating_point_hash != Sha256(model_hash(operating_point)):
        raise ValidationError("runner operating-point provenance does not match input")
    if proxy_specification.q_assumption_id is not reactive_specification.q_mode_id:
        raise ValidationError("runner proxy and reactive Q assumptions differ")
    if proxy_specification.q_spec_hash != reactive_spec_hash(reactive_specification):
        raise ValidationError("runner reactive-specification provenance does not match proxy specification")
    expected_registry_hash = Sha256(selection.alignment_hash)
    if proxy_specification.participant_registry_hash != expected_registry_hash:
        raise ValidationError("runner participant-registry provenance does not match selection")
    if feasible_domain.participant_registry_hash != expected_registry_hash:
        raise ValidationError("runner domain participant-registry provenance does not match selection")
    if feasible_domain.proxy_spec_hash != proxy_specification.specification_hash:
        raise ValidationError("runner domain/proxy specification provenance does not match")
    method_ids = tuple(profile.method_id for _, profile in profiles)
    if {method_id.value for method_id in method_ids} != EXPECTED_METHOD_IDS:
        raise ValidationError("runner method roster does not match the registered five methods")
    if tuple(method_id.value for method_id in method_ids) != EXPECTED_METHOD_ORDER:
        raise ValidationError("runner method order does not match the registered deterministic order")
    if len(set(method_ids)) != 5:
        raise ValidationError("runner method IDs must be unique")
    participant_ids = selection.participant_set.participant_ids
    if feasible_domain.participant_ids != participant_ids or proxy_specification.participant_ids != participant_ids:
        raise ValidationError("runner participant order does not match domain, selection, and proxy specification")

    expected_bindings: dict[str, tuple[str | None, Identifier | None]] = {
        "weighted_proportional": ("WEIGHTED_PROPORTIONAL", None),
        "equal_kw_reduction": ("EQUAL_KW_REDUCTION", None),
        "flat_level": ("FLAT_LEVEL", None),
        "max_export_lp": (None, max_export_objective_id),
        "fairness_qp": (None, fairness_objective_id),
    }
    request_hashes: set[Sha256 | None] = set()
    admitted_request_hashes: set[Sha256 | None] = set()
    solver_request_hashes: set[Sha256 | None] = set()
    algebraic_source_hashes: set[Sha256 | None] = set()
    algebraic_request_vector_hashes: set[Sha256 | None] = set()
    fairness_request_vector_hash: Sha256 | None = None
    fairness_source_hash: Sha256 | None = None
    allocation_profiles: list[tuple[Identifier, DiagnosticProfileBindingProtocol]] = []
    for profile_id, profile in profiles:
        if not isinstance(profile_id, Identifier) or not isinstance(profile, DiagnosticProfileBindingProtocol):
            raise ValidationError("runner profiles must be typed diagnostic bindings")
        if profile.scenario_id != scenario_id or profile.domain_hash != feasible_domain.domain_hash:
            raise ValidationError("runner profile scenario/domain join failed")
        if profile.participant_ids != participant_ids:
            raise ValidationError("runner profile participant order does not match selection")
        request_hashes.add(profile.request_hash)
        admitted_request_hashes.add(profile.admitted_request_hash)
        expected_rule, expected_objective = expected_bindings[profile.method_id.value]
        actual_rule = profile.rule_id.value if profile.rule_id is not None else None
        if actual_rule != expected_rule or profile.objective_id != expected_objective:
            raise ValidationError(f"runner method binding mismatch for {profile.method_id.value}")
        if profile.method_id.value in {"weighted_proportional", "equal_kw_reduction", "flat_level"}:
            if not isinstance(profile.allocation, AlgebraicAllocationBinding):
                raise ValidationError(
                    f"runner algebraic method {profile.method_id.value} is not backed by AlgebraicAllocationBinding"
                )
            if profile.allocation.rule_id.value != expected_rule:
                raise ValidationError(f"runner algebraic rule provenance mismatch for {profile.method_id.value}")
            algebraic_request_vector_hashes.add(profile.allocation.request_hash)
            algebraic_source_hashes.add(profile.allocation.source_request_hash)
        else:
            if not isinstance(profile.allocation, DiagnosticSolverAllocationBinding):
                raise ValidationError(
                    f"runner solver method {profile.method_id.value} is not backed by DiagnosticSolverAllocationBinding"
                )
            if profile.allocation.objective_id != expected_objective:
                raise ValidationError(f"runner solver objective provenance mismatch for {profile.method_id.value}")
            solver_request_hashes.add(profile.allocation.request_hash)
            if profile.method_id.value == "max_export_lp" and not isinstance(
                profile.allocation.problem, DiagnosticMaxExportStage2ProblemBinding
            ):
                raise ValidationError("max-export method lacks typed stage-two problem provenance")
            if profile.method_id.value == "fairness_qp":
                fairness_spec = profile.allocation.objective_specification
                if not isinstance(fairness_spec, FairnessObjectiveSpecification):
                    raise ValidationError("fairness method is not bound to a fairness objective specification")
                if fairness_spec.feasible_domain_hash != feasible_domain.domain_hash:
                    raise ValidationError("runner fairness objective is not bound to the shared feasible domain")
                if not isinstance(profile.allocation.problem, DiagnosticFairnessQPBinding):
                    raise ValidationError("runner fairness QP lacks request provenance")
                if profile.allocation.problem.request_hash != profile.request_hash or profile.request_hash is None:
                    raise ValidationError("runner fairness QP request does not match admitted request provenance")
                fairness_request_vector_hash = profile.allocation.problem.request_vector_hash
                fairness_source_hash = profile.allocation.problem.request_source_hash
        allocation_profiles.append((profile_id, profile))
    if None in request_hashes or len(algebraic_source_hashes) != 1 or None in algebraic_source_hashes:
        raise ValidationError("runner algebraic profiles require one non-null request-construction provenance")
    if len(algebraic_request_vector_hashes) != 1 or None in algebraic_request_vector_hashes:
        raise ValidationError("runner algebraic profiles must share one request vector")
    common_vector_hash = next(iter(algebraic_request_vector_hashes))
    if fairness_request_vector_hash != common_vector_hash:
        raise ValidationError("runner fairness and algebraic profiles use different request vectors")
    common_construction_hash = next(iter(algebraic_source_hashes))
    if request_source_hash != common_construction_hash:
        raise ValidationError("runner request source does not match five-method profile source")
    if fairness_source_hash != common_construction_hash:
        raise ValidationError("runner fairness and algebraic profiles use different request sources")
    if len(solver_request_hashes) != 1 or None in solver_request_hashes:
        raise ValidationError("runner solver profiles must share one non-null admitted request provenance")
    common_request_hash = next(iter(solver_request_hashes))
    if admitted_request_hashes != {common_request_hash}:
        raise ValidationError("runner profiles do not share one admitted-request provenance")
    # In the unmitigated path the admitted vector is also the effective domain
    # request.  A strict static mitigation transform can further tighten the
    # participant upper bounds, so the transformed domain legitimately binds
    # the algebraic effective-request hash while LP/QP provenance continues to
    # retain the original admitted-request hash.  Require the domain to match
    # one of these two explicitly observed hashes; never accept a caller hash.
    effective_request_hash = next(iter(algebraic_request_vector_hashes))
    if feasible_domain.request_hash not in {common_request_hash, effective_request_hash}:
        raise ValidationError("runner domain and profiles do not share request provenance")

    zero = [0.0] * len(selection.participants)
    baseline_injection = build_net_injections_from_operating_point_spec(
        parsed_case, selection, zero, operating_point, reactive_specification,
        formal_execution=False,
    )
    replayed_baseline = solve_ac_power_flow(
        parsed_case,
        p_injection_mw=baseline_injection.p_mw,
        q_injection_mvar=baseline_injection.q_mvar,
    )
    if not replayed_baseline.solver_result.solve_success:
        raise ValidationError("zero-allocation baseline replay did not converge")
    if replayed_baseline.solution_hash != shared_baseline_solution.solution_hash:
        raise ValidationError("shared baseline is not the current scenario zero-allocation baseline")

    branch_indices = tuple(range(len(parsed_case.network.branches)))
    voltage_indices = tuple(index for index in range(len(parsed_case.network.buses)) for _ in ("UPPER", "LOWER"))
    validate_proxy_ac_screen_budget_alignment(
        parsed_case,
        proxy_model=proxy_model,
        proxy_specification=proxy_specification,
        ac_limits=ac_limits,
        baseline_solution=shared_baseline_solution,
        oriented_voltage_rows=oriented_voltage_rows,
        expected_baseline_solution_hash=shared_baseline_solution.solution_hash,
        branch_indices=branch_indices,
        physical_branch_headroom_mw=physical_branch_headroom_mw,
        physical_voltage_headroom_pu=physical_voltage_headroom_pu,
        proxy_branch_headroom_mw=proxy_branch_headroom_mw,
        proxy_voltage_headroom_pu=proxy_voltage_headroom_pu,
    )
    proxy_audits = audit_proxy_profiles_against_ac(
        parsed_case,
        selection,
        proxy_model,
        proxy_specification,
        reactive_specification,
        participant_ids,
        allocation_profiles,
        operating_point=operating_point,
        branch_indices=branch_indices,
        voltage_indices=voltage_indices,
        oriented_voltage_rows=oriented_voltage_rows,
        shared_baseline_solution=shared_baseline_solution,
    )
    if any(result.baseline_solution_hash != shared_baseline_solution.solution_hash for result in proxy_audits):
        raise ValidationError("proxy audit baseline differs from runner baseline")
    ac_screens = tuple(
        (profile.method_id, _screen_diagnostic_allocation_with_baseline(
            parsed_case,
            selection,
            profile.allocation,
            operating_point,
            reactive_specification,
            calibration_q=calibration_q,
            screening_q=screening_q,
            limits=ac_limits,
            baseline_solution=shared_baseline_solution,
        ))
        for _, profile in allocation_profiles
    )
    for audit, (method_id, screen) in zip(proxy_audits, ac_screens):
        if audit.method_id != method_id:
            raise ValidationError("proxy audit and AC screen method order differs")
        if audit.allocation_hash != screen.allocation_hash:
            raise ValidationError("proxy audit and AC screen allocation differs")
        if audit.profile_solution_hash != screen.solution_hash:
            raise ValidationError("proxy audit and AC screen solved different AC points")
        if audit.q_spec_hash != screen.q_spec_hash:
            raise ValidationError("proxy audit and AC screen Q provenance differs")
    method_runs = tuple(
        DiagnosticMethodRunResult(
            method_id=profile.method_id,
            profile=profile,
            proxy_audit=audit,
            ac_screen=screen,
            activity=build_diagnostic_allocation_activity(
                feasible_domain,
                profile.participant_ids,
                profile.values_mw,
                source_allocation_hash=profile.allocation_hash,
            ),
        )
        for (_, profile), audit, (_, screen) in zip(allocation_profiles, proxy_audits, ac_screens)
    )
    return DiagnosticFiveMethodBatchResult(
        batch_id=batch_id,
        scenario_id=scenario_id,
        method_ids=IdentifierVector(method_ids),
        proxy_audits=proxy_audits,
        ac_screens=ac_screens,
        shared_baseline_solution_hash=shared_baseline_solution.solution_hash,
        network_hash=parsed_case.source_hash,
        operating_point_hash=Sha256(model_hash(operating_point)),
        q_spec_hash=proxy_specification.q_spec_hash,
        feasible_domain_hash=feasible_domain.domain_hash,
        proxy_specification_hash=proxy_specification.specification_hash,
        ac_limits_hash=ac_limits.limits_hash,
        method_runs=method_runs,
        parameter_identity=parameter_identity,
        admitted_request_hash=common_request_hash,
        request_source_hash=request_source_hash,
        capacity_vector_hash=capacity_vector_hash,
        proxy_matrix_hash=proxy_matrix_hash,
        proxy_model_matrix_hash=proxy_model_matrix_hash,
        finite_difference_delta_mw=finite_difference_delta_mw,
        physical_parameter_hash=physical_parameter_hash,
        signed_domain_reduction_certificate_hash=(
            signed_domain_reduction.certificate_hash if signed_domain_reduction else None
        ),
        full_screening_domain_hash=(
            signed_domain_reduction.full_domain_hash if signed_domain_reduction else None
        ),
    )


__all__ = [
    "EXPECTED_METHOD_IDS",
    "EXPECTED_METHOD_ORDER",
    "DiagnosticMethodRunResult",
    "DiagnosticParameterIdentity",
    "DiagnosticFiveMethodBatchResult",
    "run_ieee141_five_method_diagnostics",
]
