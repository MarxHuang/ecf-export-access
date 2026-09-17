"""Typed R5 downstream stage and side-binding interface.

This module contains no LP/QP or AC implementation.  It makes the boundary
between request admission, mitigation and a future allocation rule explicit:

    raw -> admitted -> mitigated-admitted -> effective bounded -> domain

Every side is bound independently.  A caller cannot swap a reference domain,
silently use an admitted request as a mitigated request, or omit the
mitigation specification that produced a transformed domain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from r4r.algebraic_allocation import BoundedRequestVector
from r4r.allocation_domain import FeasibleAllocationDomain
from r4r.effective_deviation import TypedStrategicScenario
from r4r.errors import ValidationError
from r4r.mitigation import (
    DomainMitigationTransformationResult,
    MitigatedRequestVector,
    MitigationModeSpecification,
)
from r4r.models.base import ContractModel
from r4r.request_contracts import AdmittedRequestVector
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, Identifier, IdentifierVector, MitigationMode, Sha256


_STAGES = {"UNRESOLVED", "CANDIDATE", "READY_DIAGNOSTIC"}


@dataclass(frozen=True, slots=True)
class SideFeasibleDomainBinding(ContractModel):
    """Side-specific domain with effective-request provenance."""

    side: AllocationSide
    scenario_id: Identifier
    participant_ids: IdentifierVector
    input_effective_request_hash: Sha256
    capacity_spec_hash: Sha256
    mitigation_specification_hash: Sha256
    base_domain_hash: Sha256
    transformed_domain_hash: Sha256
    feasible_domain: FeasibleAllocationDomain
    transformation: DomainMitigationTransformationResult
    status: str = "DIAGNOSTIC_ONLY"
    diagnostic_only: bool = True
    serialization_id = "side_feasible_domain_binding.v2"

    def __post_init__(self) -> None:
        if not isinstance(self.side, AllocationSide):
            raise ValidationError("domain side must be registered")
        if not isinstance(self.scenario_id, Identifier):
            raise ValidationError("domain scenario_id must be Identifier")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("domain participant_ids must be non-empty")
        for name, value in (
            ("input_effective_request_hash", self.input_effective_request_hash),
            ("capacity_spec_hash", self.capacity_spec_hash),
            ("mitigation_specification_hash", self.mitigation_specification_hash),
            ("base_domain_hash", self.base_domain_hash),
            ("transformed_domain_hash", self.transformed_domain_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be SHA-256")
        if not isinstance(self.feasible_domain, FeasibleAllocationDomain):
            raise ValidationError("feasible_domain must be typed")
        if not isinstance(self.transformation, DomainMitigationTransformationResult):
            raise ValidationError("transformation must be a typed domain mitigation result")
        if self.transformation.side is not self.side:
            raise ValidationError("domain transformation side does not match binding")
        if self.transformation.base_domain_hash != self.base_domain_hash:
            raise ValidationError("base domain hash is not bound to transformation")
        if self.transformation.mitigation_specification_hash != self.mitigation_specification_hash:
            raise ValidationError("mitigation specification is not bound to transformation")
        if self.transformation.capacity_spec_hash != self.capacity_spec_hash:
            raise ValidationError("capacity specification is not bound to transformation")
        if self.transformation.transformed_domain != self.feasible_domain:
            raise ValidationError("feasible domain is not the typed transformation output")
        if self.feasible_domain.participant_ids != self.participant_ids:
            raise ValidationError("domain participant order is not bound")
        if self.feasible_domain.capacity_spec_hash != self.capacity_spec_hash:
            raise ValidationError("domain capacity provenance is not bound")
        if self.feasible_domain.request_hash != self.input_effective_request_hash:
            raise ValidationError("domain request hash must bind the effective request")
        if self.transformed_domain_hash != self.feasible_domain.domain_hash:
            raise ValidationError("transformed_domain_hash does not match feasible domain")
        if self.transformation.effective_request_hash != self.input_effective_request_hash:
            raise ValidationError("effective request hash is not bound to transformation")
        if self.status != "DIAGNOSTIC_ONLY" or self.diagnostic_only is not True:
            raise ValidationError("side domain bindings remain diagnostic-only")

    @property
    def binding_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def finalize_side_domain_binding(
    transformation: DomainMitigationTransformationResult,
    effective_request: BoundedRequestVector,
    *,
    side: AllocationSide,
    scenario_id: Identifier,
    mitigation_specification: MitigationModeSpecification,
) -> tuple[BoundedRequestVector, FeasibleAllocationDomain, SideFeasibleDomainBinding]:
    """Close a typed domain transformation/effective-request join.

    A bare domain and caller-provided provenance hashes are deliberately not
    accepted.  The typed transformation must already prove the rows, bounds,
    base-domain identity and mitigation specification.
    """

    if not isinstance(transformation, DomainMitigationTransformationResult) or not isinstance(effective_request, BoundedRequestVector):
        raise ValidationError("transformation and effective_request must be typed")
    if not isinstance(side, AllocationSide) or not isinstance(scenario_id, Identifier):
        raise ValidationError("side and scenario_id must be typed")
    if not isinstance(mitigation_specification, MitigationModeSpecification):
        raise ValidationError("mitigation_specification must be typed")
    if transformation.side is not side:
        raise ValidationError("transformation side does not match requested side")
    if transformation.mitigation_specification_hash != mitigation_specification.specification_hash:
        raise ValidationError("transformation does not match mitigation specification")
    if transformation.effective_request_hash != effective_request.request_hash:
        raise ValidationError("effective request does not match typed transformation")
    closed_domain = transformation.transformed_domain
    if closed_domain.upper_bounds_mw != effective_request.values_mw:
        raise ValidationError("closed domain upper bounds must equal effective request")
    closed_effective = replace(effective_request, domain_hash=closed_domain.domain_hash)
    binding = SideFeasibleDomainBinding(
        side=side,
        scenario_id=scenario_id,
        participant_ids=closed_domain.participant_ids,
        input_effective_request_hash=closed_effective.request_hash,
        capacity_spec_hash=closed_domain.capacity_spec_hash,
        mitigation_specification_hash=mitigation_specification.specification_hash,
        base_domain_hash=transformation.base_domain_hash,
        transformed_domain_hash=closed_domain.domain_hash,
        feasible_domain=closed_domain,
        transformation=transformation,
    )
    return closed_effective, closed_domain, binding


@dataclass(frozen=True, slots=True)
class TypedScenarioExecutionBundle(ContractModel):
    """The only typed diagnostic entry point for future R7/R8 allocation."""

    typed_strategic_scenario: TypedStrategicScenario
    reference_admitted_request: AdmittedRequestVector
    reported_admitted_request: AdmittedRequestVector
    reference_mitigated_request: MitigatedRequestVector
    reported_mitigated_request: MitigatedRequestVector
    reference_effective_request: BoundedRequestVector
    reported_effective_request: BoundedRequestVector
    reference_feasible_domain: FeasibleAllocationDomain
    reported_feasible_domain: FeasibleAllocationDomain
    reference_domain_binding: SideFeasibleDomainBinding
    reported_domain_binding: SideFeasibleDomainBinding
    mitigation_specification: MitigationModeSpecification
    mitigation_mode: MitigationMode
    allocation_stage_status: str
    diagnostic_only: bool = True
    serialization_id = "typed_scenario_execution_bundle.v2"

    def __post_init__(self) -> None:
        scenario = self.typed_strategic_scenario
        if not isinstance(scenario, TypedStrategicScenario):
            raise ValidationError("typed_strategic_scenario is required")
        if not isinstance(self.mitigation_specification, MitigationModeSpecification):
            raise ValidationError("mitigation_specification is required")
        if self.mitigation_specification.mode is not self.mitigation_mode:
            raise ValidationError("mitigation mode does not match mitigation specification")
        mode_hash = self.mitigation_specification.specification_hash
        admitted_pairs = (
            (self.reference_admitted_request, scenario.reference_raw_request, AllocationSide.REFERENCE),
            (self.reported_admitted_request, scenario.reported_raw_request, AllocationSide.REPORTED),
        )
        for admitted, raw, side in admitted_pairs:
            if not isinstance(admitted, AdmittedRequestVector):
                raise ValidationError("admitted requests must be typed")
            if admitted.source_side is not side or admitted.participant_ids != scenario.participant_ids:
                raise ValidationError("admitted request side/order is not bound to scenario")
            if admitted.raw_request_hash != raw.request_hash or admitted.raw_values_mw != raw.values_mw:
                raise ValidationError("admitted request must bind its scenario raw request")
            if admitted.capacity_spec_hash != scenario.capacity_spec_hash:
                raise ValidationError("admitted capacity provenance does not match scenario")
            expected_admitted_hash = (
                scenario.reference_admitted_request_hash
                if side is AllocationSide.REFERENCE
                else scenario.reported_admitted_request_hash
            )
            if admitted.request_hash != expected_admitted_hash:
                raise ValidationError("admitted request does not match scenario source-bundle admission hash")
            if admitted.participant_registry_hash not in {None, scenario.reference_raw_request.participant_registry_hash}:
                raise ValidationError("admitted participant provenance does not match scenario")

        transformed_pairs = (
            (self.reference_mitigated_request, self.reference_admitted_request, AllocationSide.REFERENCE),
            (self.reported_mitigated_request, self.reported_admitted_request, AllocationSide.REPORTED),
        )
        for transformed, admitted, side in transformed_pairs:
            if not isinstance(transformed, MitigatedRequestVector):
                raise ValidationError("mitigated requests must be typed")
            raw = scenario.reference_raw_request if side is AllocationSide.REFERENCE else scenario.reported_raw_request
            if transformed.raw_request_hash != raw.request_hash or transformed.raw_values_mw != raw.values_mw:
                raise ValidationError("mitigated request must bind scenario raw request")
            if transformed.mode is not self.mitigation_mode:
                raise ValidationError("mitigation mode does not match both transformed requests")
            if transformed.mode_specification_hash != mode_hash:
                raise ValidationError("mitigation specification hash is not bound to transformed request")
            if transformed.admitted_request_hash != admitted.request_hash:
                raise ValidationError("mitigated request must bind the admitted request hash")
            if transformed.admitted_values_mw != admitted.admitted_values_mw:
                raise ValidationError("mitigated request must retain admitted values")

        if self.reference_mitigated_request.mode_specification_hash != self.reported_mitigated_request.mode_specification_hash:
            raise ValidationError("reference and reported mitigation specification hashes must match")

        domain_bindings = (
            (self.reference_domain_binding, self.reference_feasible_domain, self.reference_effective_request, AllocationSide.REFERENCE),
            (self.reported_domain_binding, self.reported_feasible_domain, self.reported_effective_request, AllocationSide.REPORTED),
        )
        for binding, domain, effective, side in domain_bindings:
            if binding.feasible_domain.domain_hash != domain.domain_hash:
                raise ValidationError("side domain binding does not match bundle domain")
            if binding.side is not side or binding.scenario_id != scenario.scenario_id:
                raise ValidationError("side domain binding is not aligned with scenario")
            if binding.participant_ids != scenario.participant_ids:
                raise ValidationError("side domain participant order is not bound")
            if binding.mitigation_specification_hash != mode_hash:
                raise ValidationError("side domain mitigation specification is not bound")
            if binding.input_effective_request_hash != effective.request_hash:
                raise ValidationError("side domain must bind the effective request")
            if binding.capacity_spec_hash != scenario.capacity_binding.capacity_spec_hash:
                raise ValidationError("side domain capacity identity does not match scenario capacity binding")
            if binding.transformation.capacity_binding != scenario.capacity_binding:
                raise ValidationError("side transformation capacity binding does not match scenario capacity")
            if binding.transformation.effective_request_hash != effective.request_hash:
                raise ValidationError("side transformation effective hash does not match effective request")
            if binding.transformation.admitted_request_hash != (
                self.reference_admitted_request.request_hash
                if side is AllocationSide.REFERENCE
                else self.reported_admitted_request.request_hash
            ):
                raise ValidationError("side transformation admitted hash does not match execution admission")
            if binding.transformation.mitigated_request_hash != (
                self.reference_mitigated_request.filtered_request_hash
                if side is AllocationSide.REFERENCE
                else self.reported_mitigated_request.filtered_request_hash
            ):
                raise ValidationError("side transformation mitigated hash does not match execution mitigation")
            if binding.feasible_domain.upper_bounds_mw != effective.values_mw:
                raise ValidationError("side domain upper bounds must equal the effective request")

        effective_pairs = (
            (self.reference_effective_request, self.reference_mitigated_request, self.reference_admitted_request, self.reference_feasible_domain),
            (self.reported_effective_request, self.reported_mitigated_request, self.reported_admitted_request, self.reported_feasible_domain),
        )
        capacity = scenario.capacity_binding.values_mw
        for effective, transformed, admitted, domain in effective_pairs:
            if not isinstance(effective, BoundedRequestVector) or effective.participant_ids != scenario.participant_ids:
                raise ValidationError("effective requests must be typed and aligned")
            if effective.source_request_hash != transformed.raw_request_hash:
                raise ValidationError("effective request source hash does not bind raw request")
            if effective.filtered_request_hash != transformed.filtered_request_hash:
                raise ValidationError("effective request filtered hash does not bind mitigation")
            if effective.admitted_request_hash != admitted.request_hash:
                raise ValidationError("effective request admitted hash does not bind admission")
            if effective.capacity_spec_hash != scenario.capacity_spec_hash or effective.domain_hash != domain.domain_hash:
                raise ValidationError("effective request capacity/domain provenance is not bound")
            expected_upper = tuple(
                min(admitted_value, filtered_value, capacity_value, domain_value)
                for admitted_value, filtered_value, capacity_value, domain_value in zip(
                    admitted.admitted_values_mw.values,
                    transformed.filtered_values_mw.values,
                    capacity.values,
                    domain.upper_bounds_mw.values,
                )
            )
            if effective.values_mw.values != expected_upper:
                raise ValidationError("effective request must equal the explicit admitted/mitigated/capacity/domain minimum")

        if self.allocation_stage_status not in _STAGES:
            raise ValidationError("allocation_stage_status is not registered")
        if self.allocation_stage_status == "READY_DIAGNOSTIC":
            evaluation = scenario.deviation_evaluation
            if (
                not evaluation.eq006_consistent
                or not evaluation.reporter_equation_consistent
                or not evaluation.nonreporter_identity_consistent
                or evaluation.gamma_domain_status != "VALID"
            ):
                raise ValidationError("READY_DIAGNOSTIC requires valid EQ006 and gamma evidence")
            if evaluation.classification in {"UNRESOLVED", "INVALID_INPUT", "BOUNDARY_AMBIGUOUS"}:
                raise ValidationError("READY_DIAGNOSTIC requires a resolved deviation classification")
        if self.diagnostic_only is not True:
            raise ValidationError("execution bundles remain diagnostic-only")


__all__ = ["SideFeasibleDomainBinding", "TypedScenarioExecutionBundle", "finalize_side_domain_binding"]
