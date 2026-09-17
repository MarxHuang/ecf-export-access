"""Diagnostic AC finite-difference checks for an explicit proxy matrix.

This module compares a caller-supplied proxy prediction with two independently
solved AC operating points.  It is intentionally diagnostic-only: it does not
calibrate sensitivities, promote evidence, or select a canonical Q95 sign.
"""
from __future__ import annotations

from dataclasses import dataclass
import concurrent.futures
import math
from typing import Any, Sequence

from r4r.ac_solver import ACPowerFlowSolution, solve_ac_power_flow
from r4r.allocation_pipeline import (
    AllocationBindingProtocol,
    DiagnosticProfileBindingProtocol,
)
from r4r.errors import ValidationError
from r4r.injection_adapter import build_net_injections_from_operating_point_spec
from r4r.models import OperatingPoint, ProxyModel, model_hash
from r4r.models.base import ContractModel
from r4r.network_parser import ParsedMatpowerCase
from r4r.participant_registry import ParticipantSelection
from r4r.proxy_evaluator import ProxyEvaluationSpecification, evaluate_proxy
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatMatrix, FloatVector, FiniteFloat, Identifier, IdentifierVector, QAssumption, Sha256


@dataclass(frozen=True, slots=True)
class ACProxyDifferenceResult(ContractModel):
    """One participant finite-difference comparison; never primary evidence."""

    participant_id: Identifier
    delta_mw: FiniteFloat
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_proxy_increment_mw: FloatVector
    branch_ac_increment_mw: FloatVector
    branch_error_mw: FloatVector
    voltage_proxy_increment_pu: FloatVector
    voltage_ac_increment_pu: FloatVector
    voltage_error_pu: FloatVector
    max_branch_error_mw: FiniteFloat
    max_voltage_error_pu: FiniteFloat
    baseline_solution_hash: Sha256
    perturbed_solution_hash: Sha256
    q_assumption_id: QAssumption
    status: str = "DIAGNOSTIC_ONLY"
    operating_point_hash: Sha256 | None = None
    serialization_id = "ac_proxy_difference.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, Identifier):
            raise ValidationError("participant_id must be Identifier")
        if self.delta_mw.value <= 0.0:
            raise ValidationError("delta_mw must be positive")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        for name in (
            "branch_proxy_increment_mw", "branch_ac_increment_mw", "branch_error_mw",
            "voltage_proxy_increment_pu", "voltage_ac_increment_pu", "voltage_error_pu",
        ):
            if not isinstance(getattr(self, name), FloatVector):
                raise ValidationError(f"{name} must be FloatVector")
        if len(self.branch_constraint_ids.values) != len(self.branch_error_mw.values):
            raise ValidationError("branch result rows must align with branch constraint IDs")
        if len(self.voltage_constraint_ids.values) != len(self.voltage_error_pu.values):
            raise ValidationError("voltage result rows must align with voltage constraint IDs")
        if not isinstance(self.baseline_solution_hash, Sha256) or not isinstance(self.perturbed_solution_hash, Sha256):
            raise ValidationError("solution hashes must be Sha256")
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if self.operating_point_hash is not None and not isinstance(self.operating_point_hash, Sha256):
            raise ValidationError("operating_point_hash must be Sha256 when supplied")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("AC proxy comparison cannot be promoted by this interface")

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id.to_json(),
            "delta_mw": self.delta_mw.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_proxy_increment_mw": self.branch_proxy_increment_mw.to_json(),
            "branch_ac_increment_mw": self.branch_ac_increment_mw.to_json(),
            "branch_error_mw": self.branch_error_mw.to_json(),
            "voltage_proxy_increment_pu": self.voltage_proxy_increment_pu.to_json(),
            "voltage_ac_increment_pu": self.voltage_ac_increment_pu.to_json(),
            "voltage_error_pu": self.voltage_error_pu.to_json(),
            "max_branch_error_mw": self.max_branch_error_mw.to_json(),
            "max_voltage_error_pu": self.max_voltage_error_pu.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json(),
            "perturbed_solution_hash": self.perturbed_solution_hash.to_json(),
            "q_assumption_id": self.q_assumption_id.value,
            "status": self.status,
            "operating_point_hash": self.operating_point_hash.to_json() if self.operating_point_hash else None,
        }


@dataclass(frozen=True, slots=True)
class ACProxyGroupDifferenceResult(ContractModel):
    """One simultaneous multi-participant finite-difference comparison.

    This is deliberately distinct from :class:`ACProxyDifferenceResult`:
    the perturbation is applied to a declared participant group in one AC
    solve, so the result cannot be mistaken for a sum of independent
    single-bus derivatives.  It is a diagnostic profile primitive only.
    """

    profile_id: Identifier
    perturbed_participant_ids: IdentifierVector
    allocation_mw: FloatVector
    delta_mw: FiniteFloat
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_proxy_increment_mw: FloatVector
    branch_ac_increment_mw: FloatVector
    branch_error_mw: FloatVector
    voltage_proxy_increment_pu: FloatVector
    voltage_ac_increment_pu: FloatVector
    voltage_error_pu: FloatVector
    max_branch_error_mw: FiniteFloat
    max_voltage_error_pu: FiniteFloat
    baseline_solution_hash: Sha256
    perturbed_solution_hash: Sha256
    q_assumption_id: QAssumption
    operating_point_hash: Sha256
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "ac_proxy_group_difference.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, Identifier):
            raise ValidationError("profile_id must be Identifier")
        if not isinstance(self.perturbed_participant_ids, IdentifierVector) or not self.perturbed_participant_ids.values:
            raise ValidationError("multi-participant profile must declare participants")
        if len(set(self.perturbed_participant_ids.values)) != len(self.perturbed_participant_ids.values):
            raise ValidationError("multi-participant profile participants must be unique")
        if not isinstance(self.allocation_mw, FloatVector) or not self.allocation_mw.values:
            raise ValidationError("allocation_mw must be a non-empty FloatVector")
        if self.delta_mw.value <= 0.0:
            raise ValidationError("delta_mw must be positive")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        for name in (
            "branch_proxy_increment_mw", "branch_ac_increment_mw", "branch_error_mw",
            "voltage_proxy_increment_pu", "voltage_ac_increment_pu", "voltage_error_pu",
        ):
            if not isinstance(getattr(self, name), FloatVector):
                raise ValidationError(f"{name} must be FloatVector")
        if len(self.branch_constraint_ids.values) != len(self.branch_error_mw.values):
            raise ValidationError("branch result rows must align with branch constraint IDs")
        if len(self.voltage_constraint_ids.values) != len(self.voltage_error_pu.values):
            raise ValidationError("voltage result rows must align with voltage constraint IDs")
        if not isinstance(self.baseline_solution_hash, Sha256) or not isinstance(self.perturbed_solution_hash, Sha256):
            raise ValidationError("solution hashes must be Sha256")
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if not isinstance(self.operating_point_hash, Sha256):
            raise ValidationError("operating_point_hash must be Sha256")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("AC proxy group comparison cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "profile_id": self.profile_id.to_json(),
            "perturbed_participant_ids": self.perturbed_participant_ids.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "delta_mw": self.delta_mw.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_proxy_increment_mw": self.branch_proxy_increment_mw.to_json(),
            "branch_ac_increment_mw": self.branch_ac_increment_mw.to_json(),
            "branch_error_mw": self.branch_error_mw.to_json(),
            "voltage_proxy_increment_pu": self.voltage_proxy_increment_pu.to_json(),
            "voltage_ac_increment_pu": self.voltage_ac_increment_pu.to_json(),
            "voltage_error_pu": self.voltage_error_pu.to_json(),
            "max_branch_error_mw": self.max_branch_error_mw.to_json(),
            "max_voltage_error_pu": self.max_voltage_error_pu.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json(),
            "perturbed_solution_hash": self.perturbed_solution_hash.to_json(),
            "q_assumption_id": self.q_assumption_id.value,
            "operating_point_hash": self.operating_point_hash.to_json(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ACProxyProfileDifferenceResult(ContractModel):
    """One arbitrary allocation-profile AC/proxy comparison.

    Unlike :class:`ACProxyDifferenceResult`, this record is not restricted to
    a one-participant finite-difference perturbation.  It compares a complete
    typed allocation profile against an AC solve over every supplied branch and
    bus row.  It remains diagnostic-only and never changes proxy feasibility or
    evidence gates.
    """

    profile_id: Identifier
    allocation_mw: FloatVector
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_proxy_increment_mw: FloatVector
    branch_ac_increment_mw: FloatVector
    branch_error_mw: FloatVector
    voltage_proxy_increment_pu: FloatVector
    voltage_ac_increment_pu: FloatVector
    voltage_error_pu: FloatVector
    max_branch_error_mw: FiniteFloat
    max_voltage_error_pu: FiniteFloat
    baseline_solution_hash: Sha256
    profile_solution_hash: Sha256
    q_assumption_id: QAssumption
    allocation_side: AllocationSide
    allocation_hash: Sha256
    participant_registry_hash: Sha256
    network_hash: Sha256
    q_spec_hash: Sha256
    proxy_specification_hash: Sha256
    branch_flow_quantity: str
    voltage_representation: str
    operating_point_hash: Sha256
    method_id: Identifier
    status: str = "DIAGNOSTIC_ONLY"
    proxy_model_hash: Sha256 | None = None
    proxy_matrix_hash: Sha256 | None = None
    oriented_voltage_rows_hash: Sha256 | None = None
    constraint_row_mapping_hash: Sha256 | None = None
    profile_scenario_id: Identifier | None = None
    allocation_rule_id: Identifier | None = None
    allocation_objective_id: Identifier | None = None
    feasible_domain_hash: Sha256 | None = None
    request_hash: Sha256 | None = None
    serialization_id = "ac_proxy_profile_difference.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, Identifier):
            raise ValidationError("profile_id must be Identifier")
        if not isinstance(self.allocation_mw, FloatVector) or not self.allocation_mw.values:
            raise ValidationError("allocation_mw must be a non-empty FloatVector")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        for name in (
            "branch_proxy_increment_mw", "branch_ac_increment_mw", "branch_error_mw",
            "voltage_proxy_increment_pu", "voltage_ac_increment_pu", "voltage_error_pu",
        ):
            if not isinstance(getattr(self, name), FloatVector):
                raise ValidationError(f"{name} must be FloatVector")
        if len(self.branch_constraint_ids.values) == 0 or len(self.voltage_constraint_ids.values) == 0:
            raise ValidationError("profile comparison must contain branch and voltage rows")
        if len(self.branch_constraint_ids.values) != len(self.branch_proxy_increment_mw.values) or len(self.branch_constraint_ids.values) != len(self.branch_ac_increment_mw.values) or len(self.branch_constraint_ids.values) != len(self.branch_error_mw.values):
            raise ValidationError("all branch profile vectors must align with branch IDs")
        if len(self.voltage_constraint_ids.values) != len(self.voltage_proxy_increment_pu.values) or len(self.voltage_constraint_ids.values) != len(self.voltage_ac_increment_pu.values) or len(self.voltage_constraint_ids.values) != len(self.voltage_error_pu.values):
            raise ValidationError("all voltage profile vectors must align with voltage IDs")
        if not isinstance(self.method_id, Identifier):
            raise ValidationError("method_id must be an Identifier")
        for name, value in (
            ("baseline_solution_hash", self.baseline_solution_hash),
            ("profile_solution_hash", self.profile_solution_hash),
            ("allocation_hash", self.allocation_hash),
            ("participant_registry_hash", self.participant_registry_hash),
            ("network_hash", self.network_hash),
            ("q_spec_hash", self.q_spec_hash),
            ("proxy_specification_hash", self.proxy_specification_hash),
            ("operating_point_hash", self.operating_point_hash),
            ("proxy_model_hash", self.proxy_model_hash),
            ("proxy_matrix_hash", self.proxy_matrix_hash),
            ("constraint_row_mapping_hash", self.constraint_row_mapping_hash),
            ("feasible_domain_hash", self.feasible_domain_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if self.request_hash is not None and not isinstance(self.request_hash, Sha256):
            raise ValidationError("request_hash must be Sha256 or null")
        if self.oriented_voltage_rows_hash is not None and not isinstance(self.oriented_voltage_rows_hash, Sha256):
            raise ValidationError("oriented_voltage_rows_hash must be Sha256 or null")
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if not isinstance(self.allocation_side, AllocationSide):
            raise ValidationError("allocation_side must be registered")
        for name, value in (
            ("profile_scenario_id", self.profile_scenario_id),
            ("allocation_rule_id", self.allocation_rule_id),
            ("allocation_objective_id", self.allocation_objective_id),
        ):
            if value is None and name == "profile_scenario_id":
                raise ValidationError("profile_scenario_id is required")
            if value is not None and not isinstance(value, Identifier):
                raise ValidationError(f"{name} must be Identifier or null")
        if self.feasible_domain_hash is None:
            raise ValidationError("feasible_domain_hash is required")
        if (self.allocation_rule_id is None) == (self.allocation_objective_id is None):
            raise ValidationError("result must bind exactly one allocation rule or objective")
        if any(item.value.endswith("__UPPER") or item.value.endswith("__LOWER") for item in self.voltage_constraint_ids.values):
            if self.oriented_voltage_rows_hash is None:
                raise ValidationError("oriented voltage results require oriented_voltage_rows_hash")
        if not isinstance(self.branch_flow_quantity, str) or not self.branch_flow_quantity:
            raise ValidationError("branch_flow_quantity is required")
        if not isinstance(self.voltage_representation, str) or not self.voltage_representation:
            raise ValidationError("voltage_representation is required")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("AC proxy profile comparisons cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_proxy_increment_mw": self.branch_proxy_increment_mw.to_json(),
            "branch_ac_increment_mw": self.branch_ac_increment_mw.to_json(),
            "branch_error_mw": self.branch_error_mw.to_json(),
            "voltage_proxy_increment_pu": self.voltage_proxy_increment_pu.to_json(),
            "voltage_ac_increment_pu": self.voltage_ac_increment_pu.to_json(),
            "voltage_error_pu": self.voltage_error_pu.to_json(),
            "max_branch_error_mw": self.max_branch_error_mw.to_json(),
            "max_voltage_error_pu": self.max_voltage_error_pu.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json(),
            "profile_solution_hash": self.profile_solution_hash.to_json(),
            "q_assumption_id": self.q_assumption_id.value,
            "allocation_side": self.allocation_side.value,
            "allocation_hash": self.allocation_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "network_hash": self.network_hash.to_json(),
            "q_spec_hash": self.q_spec_hash.to_json(),
            "proxy_specification_hash": self.proxy_specification_hash.to_json(),
            "branch_flow_quantity": self.branch_flow_quantity,
            "voltage_representation": self.voltage_representation,
            "operating_point_hash": self.operating_point_hash.to_json(),
            "method_id": self.method_id.to_json(),
            "status": self.status,
            "proxy_model_hash": self.proxy_model_hash.to_json() if self.proxy_model_hash else None,
            "proxy_matrix_hash": self.proxy_matrix_hash.to_json() if self.proxy_matrix_hash else None,
            "oriented_voltage_rows_hash": self.oriented_voltage_rows_hash.to_json() if self.oriented_voltage_rows_hash else None,
            "constraint_row_mapping_hash": self.constraint_row_mapping_hash.to_json() if self.constraint_row_mapping_hash else None,
            "profile_scenario_id": self.profile_scenario_id.to_json() if self.profile_scenario_id else None,
            "allocation_rule_id": self.allocation_rule_id.to_json() if self.allocation_rule_id else None,
            "allocation_objective_id": self.allocation_objective_id.to_json() if self.allocation_objective_id else None,
            "feasible_domain_hash": self.feasible_domain_hash.to_json() if self.feasible_domain_hash else None,
            "request_hash": self.request_hash.to_json() if self.request_hash else None,
        }


@dataclass(frozen=True, slots=True)
class ACProxyAuditSummary(ContractModel):
    """Deterministic diagnostic aggregation; no calibration threshold applied."""

    q_assumption_id: QAssumption
    participant_ids: IdentifierVector
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    profile_count: int
    branch_error_by_participant_mw: FloatVector
    voltage_error_by_participant_pu: FloatVector
    max_branch_prediction_error_mw: FiniteFloat
    max_voltage_prediction_error_pu: FiniteFloat
    diagnostic_payload_hash: Sha256
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "ac_proxy_audit_summary.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        if not isinstance(self.profile_count, int) or isinstance(self.profile_count, bool) or self.profile_count <= 0:
            raise ValidationError("profile_count must be a positive integer")
        if len(self.branch_error_by_participant_mw.values) != self.profile_count or len(self.voltage_error_by_participant_pu.values) != self.profile_count:
            raise ValidationError("profile error vectors must match profile_count")
        if not isinstance(self.diagnostic_payload_hash, Sha256):
            raise ValidationError("diagnostic_payload_hash must be Sha256")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("AC proxy audit summaries are diagnostic-only")

    def to_json(self) -> dict[str, Any]:
        return {
            "q_assumption_id": self.q_assumption_id.value,
            "participant_ids": self.participant_ids.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "profile_count": self.profile_count,
            "branch_error_by_participant_mw": self.branch_error_by_participant_mw.to_json(),
            "voltage_error_by_participant_pu": self.voltage_error_by_participant_pu.to_json(),
            "max_branch_prediction_error_mw": self.max_branch_prediction_error_mw.to_json(),
            "max_voltage_prediction_error_pu": self.max_voltage_prediction_error_pu.to_json(),
            "diagnostic_payload_hash": self.diagnostic_payload_hash.to_json(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ACFiniteDifferenceProxyResult(ContractModel):
    """AC-derived sensitivity matrices retained as diagnostic candidates."""

    participant_ids: IdentifierVector
    branch_constraint_ids: IdentifierVector
    voltage_constraint_ids: IdentifierVector
    branch_sensitivity_matrix: FloatMatrix
    voltage_sensitivity_matrix: FloatMatrix
    delta_mw: FiniteFloat
    baseline_solution_hash: Sha256
    perturbed_solution_hashes: tuple[Sha256, ...]
    network_hash: Sha256
    operating_point_hash: Sha256
    participant_registry_hash: Sha256
    q_assumption_id: QAssumption
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "ac_finite_difference_proxy.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.branch_constraint_ids, IdentifierVector) or not isinstance(self.voltage_constraint_ids, IdentifierVector):
            raise ValidationError("constraint IDs must be IdentifierVector values")
        participant_count = len(self.participant_ids.values)
        for name, matrix, row_count in (
            ("branch_sensitivity_matrix", self.branch_sensitivity_matrix, len(self.branch_constraint_ids.values)),
            ("voltage_sensitivity_matrix", self.voltage_sensitivity_matrix, len(self.voltage_constraint_ids.values)),
        ):
            if not isinstance(matrix, FloatMatrix):
                raise ValidationError(f"{name} must be FloatMatrix")
            if len(matrix.values) != row_count or any(len(row) != participant_count for row in matrix.values):
                raise ValidationError(f"{name} must have constraint-by-participant shape")
        if self.delta_mw.value <= 0.0:
            raise ValidationError("delta_mw must be positive")
        for name, value in (
            ("baseline_solution_hash", self.baseline_solution_hash),
            ("network_hash", self.network_hash),
            ("operating_point_hash", self.operating_point_hash),
            ("participant_registry_hash", self.participant_registry_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if len(self.perturbed_solution_hashes) != participant_count or any(
            not isinstance(value, Sha256) for value in self.perturbed_solution_hashes
        ):
            raise ValidationError("perturbed solution hashes must align with participants")
        if not isinstance(self.q_assumption_id, QAssumption):
            raise ValidationError("q_assumption_id must be registered")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("AC-derived proxy matrices cannot be promoted by this interface")

    @property
    def matrix_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "branch_constraint_ids": self.branch_constraint_ids.to_json(),
            "voltage_constraint_ids": self.voltage_constraint_ids.to_json(),
            "branch_sensitivity_matrix": self.branch_sensitivity_matrix.to_json(),
            "voltage_sensitivity_matrix": self.voltage_sensitivity_matrix.to_json(),
            "delta_mw": self.delta_mw.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json(),
            "perturbed_solution_hashes": [value.to_json() for value in self.perturbed_solution_hashes],
            "network_hash": self.network_hash.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "q_assumption_id": self.q_assumption_id.value,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class OrientedVoltageProxyRows(ContractModel):
    """Two-sided voltage-headroom rows derived without clipping.

    A ``DELTA_V`` proxy row with one upper bound cannot represent both sides of
    an AC magnitude interval.  This diagnostic value makes the required
    transformation explicit: for every bus it emits an upper row ``+dV`` with
    headroom ``Vmax - V0`` and a lower row ``-dV`` with headroom ``V0 - Vmin``.
    Negative headroom is retained and is intentionally left for the proxy
    budget validity gate; this helper never silently repairs an infeasible
    baseline.
    """

    bus_ids: IdentifierVector
    constraint_ids: IdentifierVector
    row_bus_ids: tuple[Identifier, ...]
    row_orientations: tuple[str, ...]
    sensitivity_matrix: FloatMatrix
    baseline_voltage_pu: FloatVector
    headroom_pu: FloatVector
    serialization_id = "oriented_voltage_proxy_rows.v1"

    def __post_init__(self) -> None:
        n = len(self.bus_ids.values)
        if n == 0:
            raise ValidationError("at least one bus is required for oriented voltage rows")
        if len(self.constraint_ids.values) != 2 * n or len(self.row_bus_ids) != 2 * n:
            raise ValidationError("oriented voltage rows must contain upper and lower rows per bus")
        if any(not isinstance(item, Identifier) for item in self.row_bus_ids):
            raise ValidationError("oriented voltage row buses must be Identifier values")
        if len(self.row_orientations) != 2 * n or any(item not in {"UPPER", "LOWER"} for item in self.row_orientations):
            raise ValidationError("oriented voltage row orientations are not registered")
        if len(self.sensitivity_matrix.values) != 2 * n:
            raise ValidationError("oriented voltage sensitivity rows must align with row IDs")
        participant_count = len(self.sensitivity_matrix.values[0]) if self.sensitivity_matrix.values else 0
        if participant_count == 0 or any(len(row) != participant_count for row in self.sensitivity_matrix.values):
            raise ValidationError("oriented voltage sensitivity matrix must be rectangular and non-empty")
        if len(self.baseline_voltage_pu.values) != n or len(self.headroom_pu.values) != 2 * n:
            raise ValidationError("oriented voltage baseline/headroom vectors have invalid shape")
        expected_bus_rows = tuple(bus_id for bus_id in self.bus_ids.values for _ in (0, 1))
        if self.row_bus_ids != expected_bus_rows:
            raise ValidationError("oriented voltage row buses must preserve bus order")
        expected_orientations = tuple(orientation for _ in self.bus_ids.values for orientation in ("UPPER", "LOWER"))
        if self.row_orientations != expected_orientations:
            raise ValidationError("oriented voltage rows must use UPPER then LOWER order per bus")

    @property
    def rows_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def to_json(self) -> dict[str, Any]:
        return {
            "bus_ids": self.bus_ids.to_json(),
            "constraint_ids": self.constraint_ids.to_json(),
            "row_bus_ids": [item.to_json() for item in self.row_bus_ids],
            "row_orientations": list(self.row_orientations),
            "sensitivity_matrix": self.sensitivity_matrix.to_json(),
            "baseline_voltage_pu": self.baseline_voltage_pu.to_json(),
            "headroom_pu": self.headroom_pu.to_json(),
        }


def build_oriented_voltage_proxy_rows(
    *,
    bus_ids: IdentifierVector,
    baseline_voltage_pu: FloatVector,
    lower_limit_pu: FloatVector,
    upper_limit_pu: FloatVector,
    voltage_sensitivity_matrix: FloatMatrix,
) -> OrientedVoltageProxyRows:
    """Build explicit upper/lower ``DELTA_V`` rows from an AC baseline.

    The transformation is algebraic and diagnostic-only.  It does not choose
    scientific voltage limits, infer a calibration margin, or clip negative
    headroom.  The resulting rows can be passed to a signed-upper-bound proxy
    evaluator with ``2 * n_bus`` rows.
    """

    n = len(bus_ids.values)
    if n == 0:
        raise ValidationError("bus_ids must not be empty")
    if len(baseline_voltage_pu.values) != n or len(lower_limit_pu.values) != n or len(upper_limit_pu.values) != n:
        raise ValidationError("voltage baseline and limits must align with bus IDs")
    if len(voltage_sensitivity_matrix.values) != n:
        raise ValidationError("voltage sensitivity rows must align with bus IDs")
    if any(lower > upper for lower, upper in zip(lower_limit_pu.values, upper_limit_pu.values)):
        raise ValidationError("voltage lower limits must not exceed upper limits")
    if any(not row for row in voltage_sensitivity_matrix.values):
        raise ValidationError("voltage sensitivity rows must be non-empty")
    width = len(voltage_sensitivity_matrix.values[0])
    if any(len(row) != width for row in voltage_sensitivity_matrix.values):
        raise ValidationError("voltage sensitivity matrix must be rectangular")
    row_ids: list[Identifier] = []
    row_bus_ids: list[Identifier] = []
    orientations: list[str] = []
    rows: list[tuple[float, ...]] = []
    headroom: list[float] = []
    for bus_id, baseline, lower, upper, sensitivity in zip(
        bus_ids.values,
        baseline_voltage_pu.values,
        lower_limit_pu.values,
        upper_limit_pu.values,
        voltage_sensitivity_matrix.values,
    ):
        row_bus_ids.extend((bus_id, bus_id))
        row_ids.extend((Identifier(f"{bus_id.value}__UPPER"), Identifier(f"{bus_id.value}__LOWER")))
        orientations.extend(("UPPER", "LOWER"))
        rows.extend((tuple(sensitivity), tuple(-value for value in sensitivity)))
        headroom.extend((upper - baseline, baseline - lower))
    return OrientedVoltageProxyRows(
        bus_ids=bus_ids,
        constraint_ids=IdentifierVector(row_ids),
        row_bus_ids=tuple(row_bus_ids),
        row_orientations=tuple(orientations),
        sensitivity_matrix=FloatMatrix(rows),
        baseline_voltage_pu=baseline_voltage_pu,
        headroom_pu=FloatVector(headroom),
    )


def _row_identifier_aliases(parsed: ParsedMatpowerCase, *, family: str, index: int) -> set[Identifier]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValidationError(f"{family} row index must be an integer")
    if family == "BRANCH":
        if index < 0 or index >= len(parsed.network.branches):
            raise ValidationError("branch row index is outside the parsed network")
        branch_id = parsed.network.branches[index].branch_id
        return {Identifier(str(branch_id)), Identifier(f"branch_{branch_id:03d}")}
    if family == "VOLTAGE":
        if index < 0 or index >= len(parsed.network.buses):
            raise ValidationError("voltage row index is outside the parsed network")
        bus_id = parsed.network.buses[index].bus_id
        return {Identifier(str(bus_id)), Identifier(f"bus_{bus_id}"), Identifier(f"bus_{bus_id:03d}")}
    raise ValidationError("constraint row family is not registered")


def validate_ac_constraint_row_mapping(
    parsed: ParsedMatpowerCase,
    *,
    branch_constraint_ids: IdentifierVector,
    voltage_constraint_ids: IdentifierVector,
    branch_indices: Sequence[int],
    voltage_indices: Sequence[int],
) -> None:
    """Fail closed when proxy row IDs do not identify parsed AC rows.

    Shape and uniqueness checks alone permit a caller to permute rows and
    relabel them.  This validator ties every supplied ID to the parsed branch
    or bus at the exact AC vector index.  A small, explicit compatibility set
    is accepted for legacy serializations (``2``, ``bus_2`` and ``bus_002``),
    but permutation or unknown IDs are rejected.
    """

    if len(branch_constraint_ids.values) != len(branch_indices) or len(voltage_constraint_ids.values) != len(voltage_indices):
        raise ValidationError("constraint IDs and AC row indices must have equal length")
    if len(set(branch_indices)) != len(branch_indices) or len(set(voltage_indices)) != len(voltage_indices):
        raise ValidationError("AC row indices must be unique")
    for row_id, index in zip(branch_constraint_ids.values, branch_indices):
        if row_id not in _row_identifier_aliases(parsed, family="BRANCH", index=index):
            expected = sorted(item.value for item in _row_identifier_aliases(parsed, family="BRANCH", index=index))
            raise ValidationError(f"branch constraint ID {row_id.value!r} does not map to parsed branch row {index}; expected one of {expected}")
    for row_id, index in zip(voltage_constraint_ids.values, voltage_indices):
        if row_id not in _row_identifier_aliases(parsed, family="VOLTAGE", index=index):
            expected = sorted(item.value for item in _row_identifier_aliases(parsed, family="VOLTAGE", index=index))
            raise ValidationError(f"voltage constraint ID {row_id.value!r} does not map to parsed bus row {index}; expected one of {expected}")


def validate_oriented_ac_voltage_row_mapping(
    parsed: ParsedMatpowerCase,
    *,
    rows: OrientedVoltageProxyRows,
    voltage_indices: Sequence[int],
) -> None:
    """Validate the duplicated, signed AC rows used by an oriented voltage proxy."""

    if not isinstance(rows, OrientedVoltageProxyRows):
        raise ValidationError("oriented voltage rows must use the registered contract")
    if len(voltage_indices) != len(rows.constraint_ids.values):
        raise ValidationError("oriented voltage indices must align with row IDs")
    for position, (row_id, bus_id, orientation, index) in enumerate(
        zip(rows.constraint_ids.values, rows.row_bus_ids, rows.row_orientations, voltage_indices)
    ):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(parsed.network.buses):
            raise ValidationError(f"oriented voltage row index {index!r} is outside the parsed network")
        parsed_bus_id = parsed.network.buses[index].bus_id
        if bus_id.value != str(parsed_bus_id):
            raise ValidationError(f"oriented voltage row {position} does not map to parsed bus row {index}")
        expected_id = Identifier(f"{bus_id.value}__{orientation}")
        if row_id != expected_id:
            raise ValidationError(f"oriented voltage row ID {row_id.value!r} is not canonical; expected {expected_id.value!r}")
    for offset in range(0, len(voltage_indices), 2):
        if tuple(rows.row_orientations[offset:offset + 2]) != ("UPPER", "LOWER"):
            raise ValidationError("oriented voltage rows must use UPPER then LOWER order")
        if voltage_indices[offset] != voltage_indices[offset + 1]:
            raise ValidationError("oriented voltage UPPER/LOWER rows must reference the same bus")
    if len(set(voltage_indices)) * 2 != len(voltage_indices):
        raise ValidationError("each oriented voltage bus must occur exactly twice")


def validate_proxy_ac_screen_budget_alignment(
    parsed: ParsedMatpowerCase,
    *,
    proxy_model: ProxyModel,
    ac_limits: Any,
    baseline_solution: ACPowerFlowSolution,
    oriented_voltage_rows: OrientedVoltageProxyRows | None = None,
    expected_baseline_solution_hash: Sha256 | None = None,
    proxy_specification: ProxyEvaluationSpecification | None = None,
    branch_indices: Sequence[int] | None = None,
    physical_branch_headroom_mw: FloatVector | None = None,
    physical_voltage_headroom_pu: FloatVector | None = None,
    proxy_branch_headroom_mw: FloatVector | None = None,
    proxy_voltage_headroom_pu: FloatVector | None = None,
) -> None:
    """Validate immutable physical limits and the active proxy budget.

    This validator intentionally accepts the screen-limit contract by its
    structural fields to avoid importing the full-network screen module (that
    module imports this diagnostic module for Q-spec hashing).  Physical AC
    limits are immutable, while M1/M125 may carry a separately tightened
    proxy headroom.  Callers may provide both vectors explicitly; omitting
    them preserves the M0 behavior where proxy and physical budgets coincide.
    """

    if not isinstance(proxy_model, ProxyModel):
        raise ValidationError("proxy_model must be a ProxyModel")
    if not isinstance(baseline_solution, ACPowerFlowSolution):
        raise ValidationError("baseline_solution must be an AC power-flow solution")
    if not baseline_solution.solver_result.solve_success:
        raise ValidationError("budget-alignment baseline must be a converged AC solution")
    if expected_baseline_solution_hash is not None:
        if not isinstance(expected_baseline_solution_hash, Sha256):
            raise ValidationError("expected baseline solution hash must be SHA-256")
        if baseline_solution.solution_hash != expected_baseline_solution_hash:
            raise ValidationError("budget-alignment baseline hash does not match shared AC baseline")
    n_branch = len(parsed.network.branches)
    n_bus = len(parsed.network.buses)
    for name in ("branch_budget_mw", "voltage_lower_limit_pu", "voltage_upper_limit_pu"):
        if not isinstance(getattr(ac_limits, name, None), FloatVector):
            raise ValidationError(f"AC limits must provide {name}")
    if len(ac_limits.branch_budget_mw.values) != n_branch:
        raise ValidationError("AC branch budget must cover the parsed network")
    if (
        len(ac_limits.voltage_lower_limit_pu.values) != n_bus
        or len(ac_limits.voltage_upper_limit_pu.values) != n_bus
        or len(baseline_solution.bus_voltage_pu.values) != n_bus
    ):
        raise ValidationError("AC voltage limits and baseline must cover the parsed network")
    if len(proxy_model.branch_headroom_mw.values) != n_branch:
        raise ValidationError("proxy branch headroom must cover the parsed network")
    for name, value in (
        ("physical_branch_headroom_mw", physical_branch_headroom_mw),
        ("physical_voltage_headroom_pu", physical_voltage_headroom_pu),
        ("proxy_branch_headroom_mw", proxy_branch_headroom_mw),
        ("proxy_voltage_headroom_pu", proxy_voltage_headroom_pu),
    ):
        if value is not None and not isinstance(value, FloatVector):
            raise ValidationError(f"{name} must be a FloatVector when supplied")
    if branch_indices is None:
        branch_indices = tuple(range(n_branch))
    else:
        branch_indices = tuple(branch_indices)
    if len(branch_indices) != n_branch or len(set(branch_indices)) != n_branch:
        raise ValidationError("branch indices must cover the parsed network exactly")
    if any(index < 0 or index >= n_branch for index in branch_indices):
        raise ValidationError("branch index is outside the parsed network")
    if proxy_specification is not None:
        if not isinstance(proxy_specification, ProxyEvaluationSpecification):
            raise ValidationError("proxy_specification must be a registered proxy specification")
        if len(proxy_specification.branch_constraint_ids.values) != len(branch_indices):
            raise ValidationError("proxy branch rows and branch indices have different shapes")
        for row_id, index in zip(proxy_specification.branch_constraint_ids.values, branch_indices):
            if row_id not in _row_identifier_aliases(parsed, family="BRANCH", index=index):
                raise ValidationError(f"proxy branch row {row_id.value!r} does not map to parsed branch row {index}")
    expected_physical_branch_budget = FloatVector(ac_limits.branch_budget_mw.values[index] for index in branch_indices)
    if physical_branch_headroom_mw is None:
        physical_branch_headroom_mw = expected_physical_branch_budget
    elif len(physical_branch_headroom_mw.values) != n_branch:
        raise ValidationError("physical branch headroom must cover the parsed network")
    if tuple(physical_branch_headroom_mw.values[index] for index in branch_indices) != expected_physical_branch_budget.values:
        raise ValidationError("physical branch headroom does not match AC branch budget")
    if proxy_branch_headroom_mw is None:
        proxy_branch_headroom_mw = physical_branch_headroom_mw
    if len(proxy_branch_headroom_mw.values) != n_branch:
        raise ValidationError("proxy branch headroom must cover the parsed network")
    if proxy_model.branch_headroom_mw != proxy_branch_headroom_mw:
        raise ValidationError("proxy branch headroom does not match the active proxy budget")
    bus_index = {str(bus.bus_id): index for index, bus in enumerate(parsed.network.buses)}
    expected_physical_voltage_headroom: tuple[float, ...] = tuple(
        value
        for index in range(n_bus)
        for value in (
            ac_limits.voltage_upper_limit_pu.values[index] - baseline_solution.bus_voltage_pu.values[index],
            baseline_solution.bus_voltage_pu.values[index] - ac_limits.voltage_lower_limit_pu.values[index],
        )
    )
    if physical_voltage_headroom_pu is None:
        physical_voltage_headroom_pu = FloatVector(expected_physical_voltage_headroom)
    elif len(physical_voltage_headroom_pu.values) != 2 * n_bus:
        raise ValidationError("physical voltage headroom must contain upper/lower rows for every bus")
    if any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-10)
        for actual, target in zip(physical_voltage_headroom_pu.values, expected_physical_voltage_headroom)
    ):
        raise ValidationError("physical voltage headroom does not match AC voltage limits")
    if proxy_voltage_headroom_pu is None:
        proxy_voltage_headroom_pu = physical_voltage_headroom_pu
    if len(proxy_voltage_headroom_pu.values) != 2 * n_bus:
        raise ValidationError("proxy voltage headroom must contain upper/lower rows for every bus")
    if oriented_voltage_rows is None:
        if len(proxy_model.voltage_headroom_pu.values) != n_bus:
            # Legacy non-oriented models have one upper-headroom row per bus;
            # retain the old shape check while still requiring the explicit
            # active proxy vector when supplied.
            raise ValidationError("proxy voltage headroom and AC voltage limits have different shapes")
        expected_proxy_upper = tuple(proxy_voltage_headroom_pu.values[2 * index] for index in range(n_bus))
        if any(not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-10) for actual, target in zip(proxy_model.voltage_headroom_pu.values, expected_proxy_upper)):
            raise ValidationError("proxy voltage headroom does not match the active proxy budget")
        return
    if len(oriented_voltage_rows.bus_ids.values) != n_bus:
        raise ValidationError("oriented voltage rows must cover the parsed network")
    if len(oriented_voltage_rows.baseline_voltage_pu.values) != n_bus:
        raise ValidationError("oriented voltage baseline must cover the parsed network")
    if proxy_model.voltage_sensitivity_matrix != oriented_voltage_rows.sensitivity_matrix:
        raise ValidationError("proxy voltage sensitivity matrix does not match oriented rows")
    if proxy_model.voltage_headroom_pu != oriented_voltage_rows.headroom_pu:
        raise ValidationError("proxy voltage headroom does not match oriented rows")
    expected_baseline: list[float] = []
    expected_headroom: list[float] = []
    for bus_id in oriented_voltage_rows.bus_ids.values:
        index = bus_index.get(bus_id.value)
        if index is None:
            raise ValidationError(f"oriented row bus {bus_id.value!r} is absent from parsed network")
        baseline = baseline_solution.bus_voltage_pu.values[index]
        expected_baseline.append(baseline)
        expected_headroom.extend((
            ac_limits.voltage_upper_limit_pu.values[index] - baseline,
            baseline - ac_limits.voltage_lower_limit_pu.values[index],
        ))
    if any(not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-10) for actual, target in zip(oriented_voltage_rows.baseline_voltage_pu.values, expected_baseline)):
        raise ValidationError("oriented row baseline does not match shared AC baseline")
    # The rows are the active proxy budget.  Their immutable physical source
    # was checked above; M1/M125 rows are allowed to be tighter than that
    # source, but must still equal the explicitly supplied proxy vector.
    if any(not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-10) for actual, target in zip(physical_voltage_headroom_pu.values, expected_headroom)):
        raise ValidationError("physical voltage headroom does not match AC voltage limits")
    if oriented_voltage_rows.headroom_pu != proxy_voltage_headroom_pu:
        raise ValidationError("oriented row headroom does not match the active proxy budget")


def bind_ac_proxy_result_to_model(
    result: ACFiniteDifferenceProxyResult,
    *,
    branch_headroom_mw: FloatVector,
    voltage_headroom_pu: FloatVector,
    branch_orientation: Sequence[Identifier],
) -> ProxyModel:
    """Bind an AC-derived matrix to an explicit diagnostic ``ProxyModel``.

    Budgets and branch orientations are caller-supplied physical metadata;
    this function does not infer, tighten, clip, or calibrate them.  The
    returned model remains suitable only for ``formal_execution=False`` proxy
    evaluation until the scientific contract authorizes the later rounds.
    """

    if not isinstance(result, ACFiniteDifferenceProxyResult):
        raise ValidationError("result must be an ACFiniteDifferenceProxyResult")
    if not isinstance(branch_headroom_mw, FloatVector) or not isinstance(voltage_headroom_pu, FloatVector):
        raise ValidationError("proxy headroom inputs must be FloatVector values")
    orientation = tuple(branch_orientation)
    if any(not isinstance(item, Identifier) for item in orientation):
        raise ValidationError("branch orientation must use registered identifiers")
    if orientation != result.branch_constraint_ids.values:
        raise ValidationError("branch orientation must match AC matrix row IDs")
    if len(branch_headroom_mw.values) != len(result.branch_constraint_ids.values):
        raise ValidationError("branch headroom must match AC matrix rows")
    if len(voltage_headroom_pu.values) != len(result.voltage_constraint_ids.values):
        raise ValidationError("voltage headroom must match AC matrix rows")
    participant_count = len(result.participant_ids.values)
    if any(len(row) != participant_count for row in result.branch_sensitivity_matrix.values):
        raise ValidationError("branch AC matrix columns must match participants")
    if any(len(row) != participant_count for row in result.voltage_sensitivity_matrix.values):
        raise ValidationError("voltage AC matrix columns must match participants")
    return ProxyModel(
        branch_sensitivity_matrix=result.branch_sensitivity_matrix,
        voltage_sensitivity_matrix=result.voltage_sensitivity_matrix,
        branch_headroom_mw=branch_headroom_mw,
        voltage_headroom_pu=voltage_headroom_pu,
        branch_orientation=orientation,
    )


def summarize_ac_proxy_differences(results: Sequence[ACProxyDifferenceResult]) -> ACProxyAuditSummary:
    """Aggregate explicit finite-difference rows without judging calibration."""

    rows = tuple(results)
    if not rows:
        raise ValidationError("at least one AC proxy difference result is required")
    first = rows[0]
    for row in rows:
        if row.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("only diagnostic rows may be summarized")
        if row.q_assumption_id is not first.q_assumption_id:
            raise ValidationError("Q assumptions must match across audit rows")
        if row.branch_constraint_ids != first.branch_constraint_ids or row.voltage_constraint_ids != first.voltage_constraint_ids:
            raise ValidationError("constraint row IDs must match across audit rows")
    participant_ids = IdentifierVector(row.participant_id for row in rows)
    branch_errors = FloatVector(row.max_branch_error_mw.value for row in rows)
    voltage_errors = FloatVector(row.max_voltage_error_pu.value for row in rows)
    payload = [row.to_json() for row in rows]
    return ACProxyAuditSummary(
        q_assumption_id=first.q_assumption_id,
        participant_ids=participant_ids,
        branch_constraint_ids=first.branch_constraint_ids,
        voltage_constraint_ids=first.voltage_constraint_ids,
        profile_count=len(rows),
        branch_error_by_participant_mw=branch_errors,
        voltage_error_by_participant_pu=voltage_errors,
        max_branch_prediction_error_mw=FiniteFloat(max(branch_errors.values)),
        max_voltage_prediction_error_pu=FiniteFloat(max(voltage_errors.values)),
        diagnostic_payload_hash=Sha256(canonical_hash(payload)),
    )


def reactive_spec_hash(specification: ReactiveInjectionSpecification) -> Sha256:
    """Return a stable hash for the complete reactive-injection specification."""

    return Sha256(canonical_hash({
        "q_mode_id": specification.q_mode_id.value,
        "power_factor": specification.power_factor,
        "magnitude_rule_id": specification.magnitude_rule_id.to_json(),
        "q_sign_convention": specification.q_sign_convention,
        "positive_p_definition": specification.positive_p_definition,
        "positive_q_definition": specification.positive_q_definition,
        "capability_limit_policy": specification.capability_limit_policy.to_json(),
        "scientific_status": specification.scientific_status,
    }))


def _selected(values: Sequence[float], indices: Sequence[int], *, field: str) -> FloatVector:
    result = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(values):
            raise ValidationError(f"{field} index is outside the AC solution")
        result.append(values[index])
    return FloatVector(result)


def _branch_quantity(solution: ACPowerFlowSolution, quantity: str) -> FloatVector:
    if quantity == "FROM_END_ACTIVE_POWER":
        return solution.branch_p_from_mw
    if quantity == "TO_END_ACTIVE_POWER":
        return solution.branch_p_to_mw
    if quantity == "ABSOLUTE_ACTIVE_POWER":
        return FloatVector(abs(value) for value in solution.branch_p_from_mw.values)
    raise ValidationError("branch flow quantity is not registered")


def _voltage_quantity(solution: ACPowerFlowSolution, representation: str) -> FloatVector:
    if representation in {"V_MAGNITUDE", "DELTA_V"}:
        return solution.bus_voltage_pu
    if representation in {"V_SQUARED", "DELTA_V_SQUARED"}:
        return FloatVector(value * value for value in solution.bus_voltage_pu.values)
    raise ValidationError("voltage representation is not registered")


def _difference(post: FloatVector, base: FloatVector) -> FloatVector:
    if len(post.values) != len(base.values):
        raise ValidationError("AC base and post vectors are misaligned")
    return FloatVector(post_value - base_value for post_value, base_value in zip(post.values, base.values))


def _require_success(solution: ACPowerFlowSolution, side: str) -> None:
    if not solution.solver_result.solve_success:
        raise ValidationError(f"{side} AC solve did not succeed")


def _run_single_participant_finite_difference_task(
    task: tuple[Any, ...],
) -> ACProxyDifferenceResult:
    """Execute one independent AC perturbation in a process worker.

    The task is deliberately a plain tuple of typed, serializable values so
    Windows ``spawn`` workers can be used without closing over a local
    function.  Execution parallelism is not part of any scientific payload or
    hash; all input identities and result ordering remain caller-controlled.
    """

    (
        parsed,
        selection,
        proxy_model,
        proxy_specification,
        reactive_specification,
        participant_ids,
        participant_index,
        participant_id,
        operating_point,
        delta,
        branch_indices,
        voltage_indices,
        base_solution_hash,
        base_branch,
        base_voltage,
        oriented_voltage,
        oriented_bus_indices,
        operating_point_hash,
    ) = task
    allocation = [0.0] * len(selection.participants)
    allocation[participant_index[participant_id]] = delta.value
    injection = build_net_injections_from_operating_point_spec(
        parsed,
        selection,
        allocation,
        operating_point,
        reactive_specification,
        formal_execution=False,
    )
    post_solution = solve_ac_power_flow(
        parsed,
        p_injection_mw=injection.p_mw,
        q_injection_mvar=injection.q_mvar,
    )
    _require_success(post_solution, f"participant {participant_id.value}")
    post_branch = _branch_quantity(post_solution, proxy_specification.branch_flow_quantity)
    post_voltage = _voltage_quantity(post_solution, proxy_specification.voltage_representation)
    ac_branch = _selected(
        _difference(post_branch, base_branch).values,
        branch_indices,
        field="branch",
    )
    raw_voltage_difference = _difference(post_voltage, base_voltage)
    if oriented_voltage:
        ac_voltage = FloatVector(
            raw_voltage_difference.values[index] * sign
            for index, sign in oriented_bus_indices
        )
    else:
        ac_voltage = _selected(
            raw_voltage_difference.values,
            voltage_indices,
            field="voltage",
        )
    proxy = evaluate_proxy(
        proxy_model,
        proxy_specification,
        participant_ids,
        FloatVector(allocation),
        formal_execution=False,
    )
    proxy_branch = proxy.branch_increment_mw
    proxy_voltage = proxy.voltage_increment_pu
    branch_error = FloatVector(
        ac - predicted
        for ac, predicted in zip(ac_branch.values, proxy_branch.values)
    )
    voltage_error = FloatVector(
        ac - predicted
        for ac, predicted in zip(ac_voltage.values, proxy_voltage.values)
    )
    return ACProxyDifferenceResult(
        participant_id=participant_id,
        delta_mw=delta,
        branch_constraint_ids=proxy_specification.branch_constraint_ids,
        voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
        branch_proxy_increment_mw=proxy_branch,
        branch_ac_increment_mw=ac_branch,
        branch_error_mw=branch_error,
        voltage_proxy_increment_pu=proxy_voltage,
        voltage_ac_increment_pu=ac_voltage,
        voltage_error_pu=voltage_error,
        max_branch_error_mw=FiniteFloat(
            max((abs(value) for value in branch_error.values), default=0.0)
        ),
        max_voltage_error_pu=FiniteFloat(
            max((abs(value) for value in voltage_error.values), default=0.0)
        ),
        baseline_solution_hash=base_solution_hash,
        perturbed_solution_hash=post_solution.solution_hash,
        q_assumption_id=reactive_specification.q_mode_id,
        operating_point_hash=operating_point_hash,
    )


def _run_finite_difference_column_task(task: tuple[Any, ...]) -> tuple[Sha256, FloatVector, FloatVector]:
    """Solve one participant perturbation for the parallel proxy builder."""

    (
        parsed,
        selection,
        proxy_specification,
        reactive_specification,
        participant_index,
        participant_id,
        operating_point,
        delta,
        branch_indices,
        voltage_indices,
        base_branch,
        base_voltage,
    ) = task
    allocation = [0.0] * len(selection.participants)
    allocation[participant_index[participant_id]] = delta.value
    injection = build_net_injections_from_operating_point_spec(
        parsed, selection, allocation, operating_point, reactive_specification,
        formal_execution=False,
    )
    post_solution = solve_ac_power_flow(
        parsed, p_injection_mw=injection.p_mw, q_injection_mvar=injection.q_mvar,
    )
    _require_success(post_solution, f"participant {participant_id.value}")
    post_branch = _branch_quantity(post_solution, proxy_specification.branch_flow_quantity)
    post_voltage = _voltage_quantity(post_solution, proxy_specification.voltage_representation)
    branch_column = _selected(_difference(post_branch, base_branch).values, branch_indices, field="branch")
    voltage_column = _selected(_difference(post_voltage, base_voltage).values, voltage_indices, field="voltage")
    return post_solution.solution_hash, branch_column, voltage_column


def build_ac_finite_difference_proxy(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    proxy_specification: ProxyEvaluationSpecification,
    reactive_specification: ReactiveInjectionSpecification,
    participant_ids: IdentifierVector,
    *,
    operating_point: OperatingPoint,
    delta_mw: float,
    branch_indices: Sequence[int],
    voltage_indices: Sequence[int],
    max_workers: int = 1,
) -> ACFiniteDifferenceProxyResult:
    """Generate AC finite-difference sensitivity matrices for diagnostics.

    Every participant perturbation is an independent AC solve around the
    explicit operating point.  The result is a candidate matrix for later
    calibration/proxy work; it is permanently diagnostic-only and cannot be
    used as a formal objective or evidence artifact by this interface.
    """

    delta = FiniteFloat(delta_mw)
    if delta.value <= 0.0:
        raise ValidationError("delta_mw must be positive")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValidationError("max_workers must be a positive integer")
    if participant_ids != proxy_specification.participant_ids:
        raise ValidationError("finite-difference participant IDs must match proxy specification")
    if reactive_specification.q_mode_id is not proxy_specification.q_assumption_id:
        raise ValidationError("reactive and proxy Q assumptions differ")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("operating_point must be an OperatingPoint")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if operating_point.load_scale != proxy_specification.load_scale:
        raise ValidationError("operating-point load scale does not match proxy specification")
    operating_point_hash = Sha256(model_hash(operating_point))
    if operating_point_hash != proxy_specification.operating_point_hash:
        raise ValidationError("operating-point hash does not match proxy specification")
    if proxy_specification.network_hash != parsed.source_hash:
        raise ValidationError("proxy specification network hash does not match parsed case")
    if proxy_specification.participant_registry_hash is not None and proxy_specification.participant_registry_hash != Sha256(selection.alignment_hash):
        raise ValidationError("proxy specification participant registry hash does not match selection")
    if proxy_specification.network_hash != parsed.source_hash:
        raise ValidationError("proxy specification network hash does not match parsed case")
    participant_registry_hash = Sha256(selection.alignment_hash)
    if proxy_specification.participant_registry_hash is not None and proxy_specification.participant_registry_hash != participant_registry_hash:
        raise ValidationError("proxy specification participant registry hash does not match selection")
    if reactive_spec_hash(reactive_specification) != proxy_specification.q_spec_hash:
        raise ValidationError("reactive specification hash does not match proxy specification")
    if len(branch_indices) != len(proxy_specification.branch_constraint_ids.values) or len(voltage_indices) != len(proxy_specification.voltage_constraint_ids.values):
        raise ValidationError("explicit AC row indices must match proxy constraint IDs")
    validate_ac_constraint_row_mapping(
        parsed,
        branch_constraint_ids=proxy_specification.branch_constraint_ids,
        voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
        branch_indices=branch_indices,
        voltage_indices=voltage_indices,
    )
    participant_index = {participant.participant_id: index for index, participant in enumerate(selection.participants)}
    if tuple(participant_ids.values) != tuple(participant_index):
        raise ValidationError("participant IDs must preserve selection order")

    zero = [0.0] * len(selection.participants)
    base_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, zero, operating_point, reactive_specification,
        formal_execution=False,
    )
    base_solution = solve_ac_power_flow(
        parsed, p_injection_mw=base_injection.p_mw, q_injection_mvar=base_injection.q_mvar,
    )
    _require_success(base_solution, "baseline")
    base_branch = _branch_quantity(base_solution, proxy_specification.branch_flow_quantity)
    base_voltage = _voltage_quantity(base_solution, proxy_specification.voltage_representation)
    tasks = tuple(
        (
            parsed,
            selection,
            proxy_specification,
            reactive_specification,
            participant_index,
            participant_id,
            operating_point,
            delta,
            tuple(branch_indices),
            tuple(voltage_indices),
            base_branch,
            base_voltage,
        )
        for participant_id in participant_ids.values
    )
    if max_workers == 1:
        columns = [_run_finite_difference_column_task(task) for task in tasks]
    else:
        # Independent Python-level AC solves use process workers.  ``map``
        # retains participant order so worker count never changes scientific
        # serialization or hashes.
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            columns = list(executor.map(_run_finite_difference_column_task, tasks))
    perturbed_hashes = [item[0] for item in columns]
    branch_columns = [item[1] for item in columns]
    voltage_columns = [item[2] for item in columns]

    branch_matrix = FloatMatrix(
        [
            [column.values[row] / delta.value for column in branch_columns]
            for row in range(len(proxy_specification.branch_constraint_ids.values))
        ]
    )
    voltage_matrix = FloatMatrix(
        [
            [column.values[row] / delta.value for column in voltage_columns]
            for row in range(len(proxy_specification.voltage_constraint_ids.values))
        ]
    )
    return ACFiniteDifferenceProxyResult(
        participant_ids=participant_ids,
        branch_constraint_ids=proxy_specification.branch_constraint_ids,
        voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
        branch_sensitivity_matrix=branch_matrix,
        voltage_sensitivity_matrix=voltage_matrix,
        delta_mw=delta,
        baseline_solution_hash=base_solution.solution_hash,
        perturbed_solution_hashes=tuple(perturbed_hashes),
        network_hash=proxy_specification.network_hash,
        operating_point_hash=operating_point_hash,
        participant_registry_hash=participant_registry_hash,
        q_assumption_id=reactive_specification.q_mode_id,
    )


def audit_single_participant_finite_difference(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    reactive_specification: ReactiveInjectionSpecification,
    participant_ids: IdentifierVector,
    *,
    operating_point: OperatingPoint,
    delta_mw: float,
    branch_indices: Sequence[int],
    voltage_indices: Sequence[int],
    max_workers: int = 1,
    target_participant_ids: IdentifierVector | None = None,
    baseline_solution: ACPowerFlowSolution | None = None,
) -> tuple[ACProxyDifferenceResult, ...]:
    """Compare explicit proxy increments with AC increments for selected IDs.

    The participant IDs and AC row indices are explicit inputs.  This function
    does not infer a full participant set, choose Q95 signs, calibrate margins,
    or change evidence eligibility.
    """

    delta = FiniteFloat(delta_mw)
    if delta.value <= 0.0:
        raise ValidationError("delta_mw must be positive")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValidationError("max_workers must be a positive integer")
    if participant_ids != proxy_specification.participant_ids:
        raise ValidationError("finite-difference participant IDs must match proxy specification")
    if target_participant_ids is None:
        target_participant_ids = participant_ids
    if not isinstance(target_participant_ids, IdentifierVector) or not target_participant_ids.values:
        raise ValidationError("target_participant_ids must be a non-empty IdentifierVector")
    if len(set(target_participant_ids.values)) != len(target_participant_ids.values):
        raise ValidationError("target_participant_ids must be unique")
    if any(item not in participant_ids.values for item in target_participant_ids.values):
        raise ValidationError("target_participant_ids contains an unknown participant")
    if baseline_solution is not None and not isinstance(baseline_solution, ACPowerFlowSolution):
        raise ValidationError("baseline_solution must be ACPowerFlowSolution or null")
    if reactive_specification.q_mode_id is not proxy_specification.q_assumption_id:
        raise ValidationError("reactive and proxy Q assumptions differ")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("operating_point must be an OperatingPoint")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if operating_point.load_scale != proxy_specification.load_scale:
        raise ValidationError("operating-point load scale does not match proxy specification")
    operating_point_hash = Sha256(model_hash(operating_point))
    if operating_point_hash != proxy_specification.operating_point_hash:
        raise ValidationError("operating-point hash does not match proxy specification")
    if reactive_spec_hash(reactive_specification) != proxy_specification.q_spec_hash:
        raise ValidationError("reactive specification hash does not match proxy specification")
    if len(branch_indices) != len(proxy_model.branch_sensitivity_matrix.values) or len(voltage_indices) != len(proxy_model.voltage_sensitivity_matrix.values):
        raise ValidationError("explicit AC row indices must match proxy matrix rows")
    oriented_voltage = any("__" in row_id.value for row_id in proxy_specification.voltage_constraint_ids.values)
    if oriented_voltage:
        if len(voltage_indices) != 2 * len(parsed.network.buses):
            raise ValidationError("oriented AC voltage rows must contain upper/lower rows for every bus")
        bus_index = {str(bus.bus_id): index for index, bus in enumerate(parsed.network.buses)}
        seen_orientations: set[tuple[str, str]] = set()
        oriented_bus_indices: list[tuple[int, int]] = []
        for row_id in proxy_specification.voltage_constraint_ids.values:
            parts = row_id.value.rsplit("__", 1)
            if len(parts) != 2 or parts[1] not in {"UPPER", "LOWER"} or parts[0] not in bus_index:
                raise ValidationError(f"oriented voltage constraint ID {row_id.value!r} is not canonical")
            key = (parts[0], parts[1])
            if key in seen_orientations:
                raise ValidationError("oriented voltage rows cannot repeat a bus/orientation")
            seen_orientations.add(key)
            oriented_bus_indices.append((bus_index[parts[0]], 1 if parts[1] == "UPPER" else -1))
        expected_keys = {(str(bus.bus_id), orientation) for bus in parsed.network.buses for orientation in ("UPPER", "LOWER")}
        if seen_orientations != expected_keys:
            raise ValidationError("oriented voltage rows must cover each parsed bus exactly twice")
    else:
        validate_ac_constraint_row_mapping(
            parsed,
            branch_constraint_ids=proxy_specification.branch_constraint_ids,
            voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
            branch_indices=branch_indices,
            voltage_indices=voltage_indices,
        )
        oriented_bus_indices = []
    participant_index = {participant.participant_id: index for index, participant in enumerate(selection.participants)}
    if tuple(participant_ids.values) != tuple(participant_index):
        raise ValidationError("participant IDs must preserve selection order")
    zero = [0.0] * len(selection.participants)
    base_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, zero, operating_point, reactive_specification,
        formal_execution=False,
    )
    base_solution = baseline_solution or solve_ac_power_flow(
        parsed,
        p_injection_mw=base_injection.p_mw,
        q_injection_mvar=base_injection.q_mvar,
    )
    _require_success(base_solution, "baseline")
    base_branch = _branch_quantity(base_solution, proxy_specification.branch_flow_quantity)
    base_voltage = _voltage_quantity(base_solution, proxy_specification.voltage_representation)
    if max_workers == 1:
        results = [
            _run_single_participant_finite_difference_task(task)
            for task in (
                (
                    parsed,
                    selection,
                    proxy_model,
                    proxy_specification,
                    reactive_specification,
                    participant_ids,
                    participant_index,
                    participant_id,
                    operating_point,
                    delta,
                    tuple(branch_indices),
                    tuple(voltage_indices),
                    base_solution.solution_hash,
                    base_branch,
                    base_voltage,
                    oriented_voltage,
                    tuple(oriented_bus_indices),
                    operating_point_hash,
                )
                for participant_id in target_participant_ids.values
            )
        ]
    else:
        # A process pool is intentional here: the bundled AC implementation is
        # Python-level numerical code and does not reliably release the GIL.
        # ``executor.map`` preserves participant order, while workers use
        # independent processes/cores.  The worker count is execution-only and
        # never enters a scientific hash.
        tasks = tuple(
            (
                parsed,
                selection,
                proxy_model,
                proxy_specification,
                reactive_specification,
                participant_ids,
                participant_index,
                participant_id,
                operating_point,
                delta,
                tuple(branch_indices),
                tuple(voltage_indices),
                base_solution.solution_hash,
                base_branch,
                base_voltage,
                oriented_voltage,
                tuple(oriented_bus_indices),
                operating_point_hash,
            )
            for participant_id in target_participant_ids.values
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_run_single_participant_finite_difference_task, tasks))
    return tuple(results)


def audit_multi_participant_finite_difference(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    reactive_specification: ReactiveInjectionSpecification,
    participant_ids: IdentifierVector,
    *,
    profile_id: Identifier,
    perturbed_participant_ids: IdentifierVector,
    operating_point: OperatingPoint,
    delta_mw: float,
    branch_indices: Sequence[int],
    voltage_indices: Sequence[int],
    baseline_solution: ACPowerFlowSolution | None = None,
) -> ACProxyGroupDifferenceResult:
    """Compare one simultaneous multi-participant perturbation with AC.

    The caller supplies the participant group and profile identity.  The
    function solves the common zero-allocation baseline once and a single AC
    point with ``delta_mw`` applied to every declared group member.  No
    clipping, calibration, thresholding, or evidence promotion occurs here.
    """

    delta = FiniteFloat(delta_mw)
    if delta.value <= 0.0:
        raise ValidationError("delta_mw must be positive")
    if not isinstance(profile_id, Identifier):
        raise ValidationError("profile_id must be Identifier")
    if not isinstance(perturbed_participant_ids, IdentifierVector) or not perturbed_participant_ids.values:
        raise ValidationError("multi-participant profile must declare participants")
    if len(set(perturbed_participant_ids.values)) != len(perturbed_participant_ids.values):
        raise ValidationError("multi-participant profile participants must be unique")
    if participant_ids != proxy_specification.participant_ids:
        raise ValidationError("finite-difference participant IDs must match proxy specification")
    if baseline_solution is not None and not isinstance(baseline_solution, ACPowerFlowSolution):
        raise ValidationError("baseline_solution must be ACPowerFlowSolution or null")
    if any(item not in participant_ids.values for item in perturbed_participant_ids.values):
        raise ValidationError("multi-participant profile contains an unknown participant")
    if reactive_specification.q_mode_id is not proxy_specification.q_assumption_id:
        raise ValidationError("reactive and proxy Q assumptions differ")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("operating_point must be an OperatingPoint")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if operating_point.load_scale != proxy_specification.load_scale:
        raise ValidationError("operating-point load scale does not match proxy specification")
    operating_point_hash = Sha256(model_hash(operating_point))
    if operating_point_hash != proxy_specification.operating_point_hash:
        raise ValidationError("operating-point hash does not match proxy specification")
    if proxy_specification.network_hash != parsed.source_hash:
        raise ValidationError("proxy specification network hash does not match parsed case")
    participant_registry_hash = Sha256(selection.alignment_hash)
    if proxy_specification.participant_registry_hash is not None and proxy_specification.participant_registry_hash != participant_registry_hash:
        raise ValidationError("proxy specification participant registry hash does not match selection")
    if reactive_spec_hash(reactive_specification) != proxy_specification.q_spec_hash:
        raise ValidationError("reactive specification hash does not match proxy specification")
    if len(branch_indices) != len(proxy_model.branch_sensitivity_matrix.values) or len(voltage_indices) != len(proxy_model.voltage_sensitivity_matrix.values):
        raise ValidationError("explicit AC row indices must match proxy matrix rows")
    oriented_voltage = any("__" in row_id.value for row_id in proxy_specification.voltage_constraint_ids.values)
    if oriented_voltage:
        if len(voltage_indices) != 2 * len(parsed.network.buses):
            raise ValidationError("oriented AC voltage rows must contain upper/lower rows for every bus")
        bus_index = {str(bus.bus_id): index for index, bus in enumerate(parsed.network.buses)}
        oriented_bus_indices: list[tuple[int, int]] = []
        seen_orientations: set[tuple[str, str]] = set()
        for row_id in proxy_specification.voltage_constraint_ids.values:
            parts = row_id.value.rsplit("__", 1)
            if len(parts) != 2 or parts[1] not in {"UPPER", "LOWER"} or parts[0] not in bus_index:
                raise ValidationError(f"oriented voltage constraint ID {row_id.value!r} is not canonical")
            key = (parts[0], parts[1])
            if key in seen_orientations:
                raise ValidationError("oriented voltage rows cannot repeat a bus/orientation")
            seen_orientations.add(key)
            oriented_bus_indices.append((bus_index[parts[0]], 1 if parts[1] == "UPPER" else -1))
        expected_keys = {(str(bus.bus_id), orientation) for bus in parsed.network.buses for orientation in ("UPPER", "LOWER")}
        if seen_orientations != expected_keys:
            raise ValidationError("oriented voltage rows must cover each parsed bus exactly twice")
    else:
        validate_ac_constraint_row_mapping(
            parsed,
            branch_constraint_ids=proxy_specification.branch_constraint_ids,
            voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
            branch_indices=branch_indices,
            voltage_indices=voltage_indices,
        )
        oriented_bus_indices = []
    participant_index = {participant.participant_id: index for index, participant in enumerate(selection.participants)}
    if tuple(participant_ids.values) != tuple(participant_index):
        raise ValidationError("participant IDs must preserve selection order")
    zero = [0.0] * len(selection.participants)
    base_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, zero, operating_point, reactive_specification,
        formal_execution=False,
    )
    base_solution = baseline_solution or solve_ac_power_flow(
        parsed,
        p_injection_mw=base_injection.p_mw,
        q_injection_mvar=base_injection.q_mvar,
    )
    _require_success(base_solution, "baseline")
    allocation = [0.0] * len(selection.participants)
    for participant_id in perturbed_participant_ids.values:
        allocation[participant_index[participant_id]] = delta.value
    injection = build_net_injections_from_operating_point_spec(
        parsed, selection, allocation, operating_point, reactive_specification,
        formal_execution=False,
    )
    post_solution = solve_ac_power_flow(
        parsed, p_injection_mw=injection.p_mw, q_injection_mvar=injection.q_mvar,
    )
    _require_success(post_solution, f"profile {profile_id.value}")
    base_branch = _branch_quantity(base_solution, proxy_specification.branch_flow_quantity)
    post_branch = _branch_quantity(post_solution, proxy_specification.branch_flow_quantity)
    ac_branch = _selected(_difference(post_branch, base_branch).values, branch_indices, field="branch")
    base_voltage = _voltage_quantity(base_solution, proxy_specification.voltage_representation)
    post_voltage = _voltage_quantity(post_solution, proxy_specification.voltage_representation)
    raw_voltage_difference = _difference(post_voltage, base_voltage)
    if oriented_voltage:
        ac_voltage = FloatVector(raw_voltage_difference.values[index] * sign for index, sign in oriented_bus_indices)
    else:
        ac_voltage = _selected(raw_voltage_difference.values, voltage_indices, field="voltage")
    allocation_vector = FloatVector(allocation)
    proxy = evaluate_proxy(proxy_model, proxy_specification, participant_ids, allocation_vector, formal_execution=False)
    branch_error = FloatVector(ac - predicted for ac, predicted in zip(ac_branch.values, proxy.branch_increment_mw.values))
    voltage_error = FloatVector(ac - predicted for ac, predicted in zip(ac_voltage.values, proxy.voltage_increment_pu.values))
    return ACProxyGroupDifferenceResult(
        profile_id=profile_id,
        perturbed_participant_ids=perturbed_participant_ids,
        allocation_mw=allocation_vector,
        delta_mw=delta,
        branch_constraint_ids=proxy_specification.branch_constraint_ids,
        voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
        branch_proxy_increment_mw=proxy.branch_increment_mw,
        branch_ac_increment_mw=ac_branch,
        branch_error_mw=branch_error,
        voltage_proxy_increment_pu=proxy.voltage_increment_pu,
        voltage_ac_increment_pu=ac_voltage,
        voltage_error_pu=voltage_error,
        max_branch_error_mw=FiniteFloat(max((abs(value) for value in branch_error.values), default=0.0)),
        max_voltage_error_pu=FiniteFloat(max((abs(value) for value in voltage_error.values), default=0.0)),
        baseline_solution_hash=base_solution.solution_hash,
        perturbed_solution_hash=post_solution.solution_hash,
        q_assumption_id=reactive_specification.q_mode_id,
        operating_point_hash=operating_point_hash,
    )


def audit_proxy_profiles_against_ac(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    proxy_model: ProxyModel,
    proxy_specification: ProxyEvaluationSpecification,
    reactive_specification: ReactiveInjectionSpecification,
    participant_ids: IdentifierVector,
    profiles: Sequence[tuple[Identifier, DiagnosticProfileBindingProtocol]],
    *,
    operating_point: OperatingPoint,
    branch_indices: Sequence[int],
    voltage_indices: Sequence[int],
    oriented_voltage_rows: OrientedVoltageProxyRows | None = None,
    shared_baseline_solution: ACPowerFlowSolution | None = None,
) -> tuple[ACProxyProfileDifferenceResult, ...]:
    """Compare complete allocation profiles with AC over all selected rows.

    ``branch_indices`` and ``voltage_indices`` must cover the rows represented
    by the supplied proxy model.  The function solves one common zero-
    allocation baseline and one AC point per profile, then compares the exact
    proxy increments with the corresponding AC increments.  Negative errors
    are retained; no clipping, thresholding, calibration or evidence promotion
    occurs here.
    """

    if not profiles:
        raise ValidationError("at least one allocation profile is required")
    if participant_ids != proxy_specification.participant_ids:
        raise ValidationError("profile participant IDs must match proxy specification")
    if reactive_specification.q_mode_id is not proxy_specification.q_assumption_id:
        raise ValidationError("reactive and proxy Q assumptions differ")
    if not isinstance(operating_point, OperatingPoint):
        raise ValidationError("operating_point must be an OperatingPoint")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if operating_point.load_scale != proxy_specification.load_scale:
        raise ValidationError("operating-point load scale does not match proxy specification")
    operating_point_hash = Sha256(model_hash(operating_point))
    if operating_point_hash != proxy_specification.operating_point_hash:
        raise ValidationError("operating-point hash does not match proxy specification")
    if proxy_specification.network_hash != parsed.source_hash:
        raise ValidationError("proxy specification network hash does not match parsed case")
    participant_registry_hash = Sha256(selection.alignment_hash)
    if proxy_specification.participant_registry_hash is not None and proxy_specification.participant_registry_hash != participant_registry_hash:
        raise ValidationError("proxy specification participant registry hash does not match selection")
    if reactive_spec_hash(reactive_specification) != proxy_specification.q_spec_hash:
        raise ValidationError("reactive specification hash does not match proxy specification")
    if len(branch_indices) != len(proxy_model.branch_sensitivity_matrix.values):
        raise ValidationError("branch indices must match proxy model rows")
    if len(voltage_indices) != len(proxy_model.voltage_sensitivity_matrix.values):
        raise ValidationError("voltage indices must match proxy model rows")
    if oriented_voltage_rows is None:
        validate_ac_constraint_row_mapping(
            parsed,
            branch_constraint_ids=proxy_specification.branch_constraint_ids,
            voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
            branch_indices=branch_indices,
            voltage_indices=voltage_indices,
        )
    else:
        validate_ac_constraint_row_mapping(
            parsed,
            branch_constraint_ids=proxy_specification.branch_constraint_ids,
            voltage_constraint_ids=IdentifierVector(
                Identifier(str(parsed.network.buses[index].bus_id)) for index in voltage_indices[::2]
            ),
            branch_indices=branch_indices,
            voltage_indices=voltage_indices[::2],
        )
        validate_oriented_ac_voltage_row_mapping(parsed, rows=oriented_voltage_rows, voltage_indices=voltage_indices)
    if oriented_voltage_rows is not None:
        if proxy_specification.voltage_representation != "DELTA_V":
            raise ValidationError("oriented voltage audit currently requires DELTA_V representation")
        if proxy_specification.voltage_constraint_ids != oriented_voltage_rows.constraint_ids:
            raise ValidationError("proxy voltage row IDs do not match oriented voltage rows")
        if len(proxy_model.voltage_sensitivity_matrix.values) != len(oriented_voltage_rows.constraint_ids.values):
            raise ValidationError("proxy voltage matrix does not match oriented voltage rows")
        if proxy_model.voltage_sensitivity_matrix != oriented_voltage_rows.sensitivity_matrix:
            raise ValidationError("proxy voltage sensitivity matrix does not match oriented voltage rows")
        if proxy_model.voltage_headroom_pu != oriented_voltage_rows.headroom_pu:
            raise ValidationError("proxy voltage headroom does not match oriented voltage rows")
    participant_index = {participant.participant_id: index for index, participant in enumerate(selection.participants)}
    if tuple(participant_ids.values) != tuple(participant_index):
        raise ValidationError("participant IDs must preserve selection order")
    profile_ids: set[Identifier] = set()
    for profile_id, profile_binding in profiles:
        if not isinstance(profile_id, Identifier):
            raise ValidationError("profile IDs must be Identifier values")
        if profile_id in profile_ids:
            raise ValidationError("profile IDs must be unique")
        profile_ids.add(profile_id)
        if not isinstance(profile_binding, DiagnosticProfileBindingProtocol):
            raise ValidationError("profiles must carry typed method/provenance bindings")
        allocation_binding = profile_binding.allocation
        if profile_binding.method_id.value == "":
            raise ValidationError("profile method_id must not be empty")
        if profile_binding.scenario_id != allocation_binding.scenario_id:
            raise ValidationError("profile scenario does not match allocation")
        if allocation_binding.participant_ids != participant_ids:
            raise ValidationError("profile allocation participant order must match participants")
        if allocation_binding.participant_registry_hash is None:
            raise ValidationError("typed profile allocation requires participant registry provenance")
        if allocation_binding.participant_registry_hash != participant_registry_hash:
            raise ValidationError("profile allocation registry hash does not match selection")
        if len(allocation_binding.values_mw.values) != len(participant_ids.values):
            raise ValidationError("profile allocation shape must match participants")
        if any(value < 0.0 for value in allocation_binding.values_mw.values):
            raise ValidationError("profile allocations must be nonnegative")
    if shared_baseline_solution is None:
        zero = [0.0] * len(selection.participants)
        base_injection = build_net_injections_from_operating_point_spec(
            parsed, selection, zero, operating_point, reactive_specification, formal_execution=False,
        )
        base_solution = solve_ac_power_flow(parsed, p_injection_mw=base_injection.p_mw, q_injection_mvar=base_injection.q_mvar)
    else:
        base_solution = shared_baseline_solution
    _require_success(base_solution, "baseline")
    if oriented_voltage_rows is not None:
        baseline_indices = tuple(voltage_indices[::2])
        expected_baseline = tuple(base_solution.bus_voltage_pu.values[index] for index in baseline_indices)
        if len(expected_baseline) != len(oriented_voltage_rows.baseline_voltage_pu.values) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10)
            for actual, expected in zip(oriented_voltage_rows.baseline_voltage_pu.values, expected_baseline)
        ):
            raise ValidationError("oriented voltage baseline does not match the shared AC baseline solution")
    base_branch = _branch_quantity(base_solution, proxy_specification.branch_flow_quantity)
    base_voltage = _voltage_quantity(base_solution, proxy_specification.voltage_representation)
    results: list[ACProxyProfileDifferenceResult] = []
    for profile_id, profile_binding in profiles:
        allocation_binding = profile_binding.allocation
        allocation = allocation_binding.values_mw
        injection = build_net_injections_from_operating_point_spec(
            parsed, selection, allocation.values, operating_point, reactive_specification, formal_execution=False,
            allocation_binding=allocation_binding,
        )
        profile_solution = solve_ac_power_flow(parsed, p_injection_mw=injection.p_mw, q_injection_mvar=injection.q_mvar)
        _require_success(profile_solution, f"profile {profile_id.value}")
        ac_branch = _selected(_difference(_branch_quantity(profile_solution, proxy_specification.branch_flow_quantity), base_branch).values, branch_indices, field="branch")
        voltage_difference = _difference(
            _voltage_quantity(profile_solution, proxy_specification.voltage_representation),
            base_voltage,
        )
        if oriented_voltage_rows is None:
            ac_voltage = _selected(voltage_difference.values, voltage_indices, field="voltage")
        else:
            ac_voltage = FloatVector(
                voltage_difference.values[index] * (1.0 if orientation == "UPPER" else -1.0)
                for index, orientation in zip(
                    voltage_indices,
                    oriented_voltage_rows.row_orientations,
                )
            )
        proxy = evaluate_proxy(proxy_model, proxy_specification, participant_ids, allocation, formal_execution=False)
        branch_error = FloatVector(ac - predicted for ac, predicted in zip(ac_branch.values, proxy.branch_increment_mw.values))
        voltage_error = FloatVector(ac - predicted for ac, predicted in zip(ac_voltage.values, proxy.voltage_increment_pu.values))
        results.append(ACProxyProfileDifferenceResult(
            profile_id=profile_id,
            allocation_mw=allocation,
            branch_constraint_ids=proxy_specification.branch_constraint_ids,
            voltage_constraint_ids=proxy_specification.voltage_constraint_ids,
            branch_proxy_increment_mw=proxy.branch_increment_mw,
            branch_ac_increment_mw=ac_branch,
            branch_error_mw=branch_error,
            voltage_proxy_increment_pu=proxy.voltage_increment_pu,
            voltage_ac_increment_pu=ac_voltage,
            voltage_error_pu=voltage_error,
            max_branch_error_mw=FiniteFloat(max((abs(value) for value in branch_error.values), default=0.0)),
            max_voltage_error_pu=FiniteFloat(max((abs(value) for value in voltage_error.values), default=0.0)),
            baseline_solution_hash=base_solution.solution_hash,
            profile_solution_hash=profile_solution.solution_hash,
            q_assumption_id=reactive_specification.q_mode_id,
            allocation_side=allocation_binding.side,
            allocation_hash=allocation_binding.allocation_hash,
            participant_registry_hash=participant_registry_hash,
            network_hash=parsed.source_hash,
            q_spec_hash=proxy_specification.q_spec_hash,
            proxy_specification_hash=proxy_specification.specification_hash,
            branch_flow_quantity=proxy_specification.branch_flow_quantity,
            voltage_representation=proxy_specification.voltage_representation,
            operating_point_hash=operating_point_hash,
            method_id=profile_binding.method_id,
            proxy_model_hash=Sha256(model_hash(proxy_model)),
            proxy_matrix_hash=Sha256(canonical_hash({
                "branch": proxy_model.branch_sensitivity_matrix.to_json(),
                "voltage": proxy_model.voltage_sensitivity_matrix.to_json(),
            })),
            oriented_voltage_rows_hash=oriented_voltage_rows.rows_hash if oriented_voltage_rows is not None else None,
            constraint_row_mapping_hash=Sha256(canonical_hash({
                "branch_indices": list(branch_indices),
                "voltage_indices": list(voltage_indices),
                "branch_ids": proxy_specification.branch_constraint_ids.to_json(),
                "voltage_ids": proxy_specification.voltage_constraint_ids.to_json(),
                "voltage_orientations": list(oriented_voltage_rows.row_orientations) if oriented_voltage_rows is not None else None,
            })),
            profile_scenario_id=profile_binding.scenario_id,
            allocation_rule_id=profile_binding.rule_id,
            allocation_objective_id=profile_binding.objective_id,
            feasible_domain_hash=profile_binding.domain_hash,
            request_hash=profile_binding.request_hash,
        ))
    return tuple(results)
