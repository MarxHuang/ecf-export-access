"""Explicit mitigation transformations for diagnostic allocation runs.

The manuscript separates request filtering from network-margin reservation.
This module implements that interface without solving an allocation problem or
promoting evidence.  A transformed request and a tightened feasible domain
retain their independent hashes, so a caller cannot silently replace either
input with a clipped capacity box.

The static case-study configuration supplies a strategic-reporter score of
``0.70`` and margin ratios ``0.05`` and ``0.10``.  Those values are explicit
arguments to :func:`build_manuscript_mitigation_specs`; callers must still
provide the configuration/source hash.  All outputs are diagnostic-only while
the active contract keeps EQ071--EQ075 open for the formal R15 close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from r4r.allocation_domain import FeasibleAllocationDomain, LinearConstraintRow
from r4r.allocation_pipeline import ExplicitCapacityVector
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.request_contracts import AdmittedRequestVector, RawRequestVector
from r4r.request_policies import IdentityAdmissionPolicy
from r4r.serialization import canonical_hash
from r4r.types import (
    AllocationSide,
    FloatVector,
    Identifier,
    IdentifierVector,
    MitigationMode,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


_MODES = tuple(
    MitigationMode(item)
    for item in ("None", "CF", "MR_005", "MR_010", "C_005", "C_010")
)
_NETWORK_FAMILIES = {"BRANCH_INCREMENT", "VOLTAGE_UPPER", "VOLTAGE_LOWER"}


@dataclass(frozen=True, slots=True)
class MitigationModeSpecification(ContractModel):
    """One explicit static mitigation mode configuration."""

    mode: MitigationMode
    strategic_score: FiniteFloat
    margin_ratio: FiniteFloat
    source_config_hash: Sha256
    score_scope: str = "STRATEGIC_REPORTER_ONLY"
    diagnostic_only: bool = True
    serialization_id = "mitigation_mode_specification.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MitigationMode):
            raise ValidationError("mitigation mode must be registered")
        if not isinstance(self.strategic_score, FiniteFloat) or not 0.0 <= self.strategic_score.value <= 1.0:
            raise ValidationError("strategic_score must lie in [0, 1]")
        if not isinstance(self.margin_ratio, FiniteFloat) or not 0.0 <= self.margin_ratio.value < 1.0:
            raise ValidationError("margin_ratio must lie in [0, 1)")
        if not isinstance(self.source_config_hash, Sha256):
            raise ValidationError("source_config_hash must be SHA-256")
        if self.score_scope != "STRATEGIC_REPORTER_ONLY":
            raise ValidationError("only strategic-reporter static filtering is registered")
        if self.diagnostic_only is not True:
            raise ValidationError("mitigation mode specifications remain diagnostic-only")
        expected_margin = {
            MitigationMode.NONE: 0.0,
            MitigationMode.CF: 0.0,
            MitigationMode.MR_005: 0.05,
            MitigationMode.MR_010: 0.10,
            MitigationMode.C_005: 0.05,
            MitigationMode.C_010: 0.10,
        }[self.mode]
        if not math.isclose(self.margin_ratio.value, expected_margin, rel_tol=0.0, abs_tol=1e-12):
            raise ValidationError("mode margin ratio does not match the registered mode")

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "mode": self.mode.value,
            "strategic_score": self.strategic_score.to_json(),
            "margin_ratio": self.margin_ratio.to_json(),
            "source_config_hash": self.source_config_hash.to_json(),
            "score_scope": self.score_scope,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def specification_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def build_manuscript_mitigation_specs(
    source_config_hash: Sha256,
    *,
    strategic_score: float = 0.70,
    margin_005: float = 0.05,
    margin_010: float = 0.10,
) -> tuple[MitigationModeSpecification, ...]:
    """Build the ordered six-mode static configuration.

    The values are not inferred from a result.  A caller should hash the
    current manuscript configuration and pass that digest here.
    """

    if not isinstance(source_config_hash, Sha256):
        raise ValidationError("source_config_hash must be SHA-256")
    score = FiniteFloat(strategic_score)
    k005 = FiniteFloat(margin_005)
    k010 = FiniteFloat(margin_010)
    return tuple(
        MitigationModeSpecification(
            mode=mode,
            strategic_score=score,
            margin_ratio=(FiniteFloat(0.0) if mode in {MitigationMode.NONE, MitigationMode.CF} else k005 if mode in {MitigationMode.MR_005, MitigationMode.C_005} else k010),
            source_config_hash=source_config_hash,
        )
        for mode in _MODES
    )


def build_static_score_vector(
    participant_ids: IdentifierVector,
    *,
    side: AllocationSide,
    reporter_id: Identifier | None,
    specification: MitigationModeSpecification,
) -> FloatVector:
    """Return the static score vector for one reference/reported side.

    CF/C modes filter only the strategic reporter; passive participants and the
    honest reference side retain score one.  None/MR modes are identity score
    vectors.  The scope is explicit instead of being inferred from a side.
    """

    if not isinstance(participant_ids, IdentifierVector) or not participant_ids.values:
        raise ValidationError("participant_ids must be non-empty")
    if not isinstance(side, AllocationSide) or not isinstance(specification, MitigationModeSpecification):
        raise ValidationError("side and specification must be typed")
    if side is AllocationSide.REPORTED:
        if not isinstance(reporter_id, Identifier) or reporter_id not in participant_ids.values:
            raise ValidationError("reported mitigation requires a registered reporter")
    elif reporter_id is not None:
        raise ValidationError("reference mitigation cannot carry a reporter")
    values = [1.0] * len(participant_ids.values)
    if side is AllocationSide.REPORTED and specification.mode in {MitigationMode.CF, MitigationMode.C_005, MitigationMode.C_010}:
        values[participant_ids.values.index(reporter_id)] = specification.strategic_score.value
    return FloatVector(values)


@dataclass(frozen=True, slots=True)
class MitigatedRequestVector(ContractModel):
    """Admitted request plus an explicit mitigation transformation.

    ``admitted_request_hash`` and ``admitted_values_mw`` are the R5 stage
    boundary.  The optional values keep the older low-level diagnostic helper
    readable, but every R5 execution bundle requires them.  Thus the typed
    downstream path is ``raw -> admitted -> mitigated -> effective`` rather
    than silently equating admission with mitigation.
    """

    raw_request_hash: Sha256
    raw_values_mw: FloatVector
    score: FloatVector
    filtered_values_mw: FloatVector
    mode: MitigationMode
    mode_specification_hash: Sha256
    admitted_request_hash: Sha256 | None = None
    admitted_values_mw: FloatVector | None = None
    diagnostic_only: bool = True
    serialization_id = "mitigated_request_vector.v2"

    def __post_init__(self) -> None:
        for name, value in (("raw_request_hash", self.raw_request_hash), ("mode_specification_hash", self.mode_specification_hash)):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be SHA-256")
        if not isinstance(self.raw_values_mw, FloatVector) or not isinstance(self.score, FloatVector) or not isinstance(self.filtered_values_mw, FloatVector):
            raise ValidationError("mitigated request vectors must be FloatVector values")
        if len(self.raw_values_mw.values) != len(self.score.values) or len(self.raw_values_mw.values) != len(self.filtered_values_mw.values):
            raise ValidationError("mitigated request vectors must align")
        if (self.admitted_request_hash is None) != (self.admitted_values_mw is None):
            raise ValidationError("admitted request hash and values must be supplied together")
        if self.admitted_request_hash is not None:
            if not isinstance(self.admitted_request_hash, Sha256):
                raise ValidationError("admitted_request_hash must be SHA-256")
            if not isinstance(self.admitted_values_mw, FloatVector) or len(self.admitted_values_mw.values) != len(self.raw_values_mw.values):
                raise ValidationError("admitted_values_mw must align with raw request")
            if any(value < 0.0 for value in self.admitted_values_mw.values):
                raise ValidationError("admitted request values must be nonnegative")
        if any(value < 0.0 for value in self.raw_values_mw.values):
            raise ValidationError("raw requests must be nonnegative")
        if any(value < 0.0 or value > 1.0 for value in self.score.values):
            raise ValidationError("delivery scores must lie in [0, 1]")
        admitted_values = self.admitted_values_mw.values if self.admitted_values_mw is not None else self.raw_values_mw.values
        expected = FloatVector(raw * score for raw, score in zip(admitted_values, self.score.values))
        if self.filtered_values_mw != expected:
            raise ValidationError("filtered request must equal score times raw request")
        if not isinstance(self.mode, MitigationMode):
            raise ValidationError("mitigation mode must be registered")
        if self.diagnostic_only is not True:
            raise ValidationError("mitigated requests remain diagnostic-only")

    @property
    def filtered_request_hash(self) -> Sha256:
        return Sha256(canonical_hash({
            "serialization_id": self.serialization_id,
            "participant_request": self.filtered_values_mw.to_json(),
            "mode": self.mode.value,
            "mode_specification_hash": self.mode_specification_hash.to_json(),
        }))

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "raw_request_hash": self.raw_request_hash.to_json(),
            "raw_values_mw": self.raw_values_mw.to_json(),
            "score": self.score.to_json(),
            "filtered_values_mw": self.filtered_values_mw.to_json(),
            "filtered_request_hash": self.filtered_request_hash.to_json(),
            "mode": self.mode.value,
            "mode_specification_hash": self.mode_specification_hash.to_json(),
            "admitted_request_hash": self.admitted_request_hash.to_json() if self.admitted_request_hash else None,
            "admitted_values_mw": self.admitted_values_mw.to_json() if self.admitted_values_mw else None,
            "diagnostic_only": self.diagnostic_only,
        }


def _request_hash(participant_ids: IdentifierVector, values_mw: FloatVector) -> Sha256:
    """Hash the effective participant-ordered request vector."""

    return Sha256(canonical_hash({
        "participant_ids": participant_ids.to_json(),
        "values_mw": values_mw.to_json(),
    }))


@dataclass(frozen=True, slots=True)
class DomainMitigationTransformationResult(ContractModel):
    """Typed proof that one feasible domain came from one mitigation transform.

    The result is deliberately separate from :class:`FeasibleAllocationDomain`
    so a caller cannot pass a bare domain together with caller-chosen hashes to
    the execution bundle.  ``effective_request_hash`` is computed from the
    transformed upper bounds, which are the explicit
    ``min(base-domain upper, admitted, mitigated, capacity)`` entrance to the
    future allocation stage.
    """

    side: AllocationSide
    base_domain_hash: Sha256
    base_domain: FeasibleAllocationDomain
    transformed_domain: FeasibleAllocationDomain
    admitted_request: AdmittedRequestVector
    mitigated_request: MitigatedRequestVector
    capacity_binding: ExplicitCapacityVector
    mitigation_specification: MitigationModeSpecification
    mitigation_specification_hash: Sha256
    capacity_spec_hash: Sha256
    admitted_request_hash: Sha256
    mitigated_request_hash: Sha256
    effective_request_hash: Sha256
    status: str = "DIAGNOSTIC_ONLY"
    diagnostic_only: bool = True
    serialization_id = "domain_mitigation_transformation_result.v2"

    def __post_init__(self) -> None:
        if not isinstance(self.side, AllocationSide):
            raise ValidationError("domain transformation side must be registered")
        for name, value in (
            ("base_domain_hash", self.base_domain_hash),
            ("mitigation_specification_hash", self.mitigation_specification_hash),
            ("capacity_spec_hash", self.capacity_spec_hash),
            ("admitted_request_hash", self.admitted_request_hash),
            ("mitigated_request_hash", self.mitigated_request_hash),
            ("effective_request_hash", self.effective_request_hash),
        ):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be SHA-256")
        if not isinstance(self.transformed_domain, FeasibleAllocationDomain):
            raise ValidationError("transformed_domain must be FeasibleAllocationDomain")
        if not isinstance(self.base_domain, FeasibleAllocationDomain):
            raise ValidationError("base_domain must be FeasibleAllocationDomain")
        if self.base_domain.domain_hash != self.base_domain_hash:
            raise ValidationError("base_domain_hash does not match base_domain")
        if self.base_domain.participant_ids != self.transformed_domain.participant_ids:
            raise ValidationError("base and transformed domains must share participant order")
        if not isinstance(self.admitted_request, AdmittedRequestVector):
            raise ValidationError("admitted_request must be typed")
        if not isinstance(self.mitigated_request, MitigatedRequestVector):
            raise ValidationError("mitigated_request must be typed")
        if not isinstance(self.capacity_binding, ExplicitCapacityVector):
            raise ValidationError("capacity_binding must be typed")
        if not isinstance(self.mitigation_specification, MitigationModeSpecification):
            raise ValidationError("mitigation_specification must be typed")
        if self.admitted_request_hash != self.admitted_request.request_hash:
            raise ValidationError("admitted_request_hash does not match typed admitted request")
        if self.mitigated_request_hash != self.mitigated_request.filtered_request_hash:
            raise ValidationError("mitigated_request_hash does not match typed mitigated request")
        if self.mitigation_specification_hash != self.mitigation_specification.specification_hash:
            raise ValidationError("mitigation_specification_hash does not match typed specification")
        if self.capacity_spec_hash != self.capacity_binding.capacity_spec_hash:
            raise ValidationError("capacity_spec_hash does not match typed capacity binding")
        if self.admitted_request.source_side is not self.side:
            raise ValidationError("admitted request side does not match transformation")
        expected_reporter = (
            self.admitted_request.reporter_id
            if self.side is AllocationSide.REPORTED
            else None
        )
        if self.side is AllocationSide.REPORTED:
            if expected_reporter is None:
                raise ValidationError("reported admitted request must carry its reporter")
        elif self.admitted_request.reporter_id is not None:
            raise ValidationError("reference admitted request cannot carry a reporter")
        if self.mitigated_request.mode is not self.mitigation_specification.mode:
            raise ValidationError("mitigated request mode does not match mitigation specification")
        expected_score = build_static_score_vector(
            self.admitted_request.participant_ids,
            side=self.side,
            reporter_id=expected_reporter,
            specification=self.mitigation_specification,
        )
        if self.mitigated_request.score != expected_score:
            raise ValidationError("mitigated request score does not match registered side/mode semantics")
        if self.admitted_request.participant_ids != self.transformed_domain.participant_ids:
            raise ValidationError("admitted request participant order does not match transformation")
        if self.capacity_binding.participant_ids != self.admitted_request.participant_ids:
            raise ValidationError("capacity binding participant order does not match admitted request")
        if self.capacity_binding.participant_registry_hash != self.admitted_request.participant_registry_hash:
            raise ValidationError("capacity binding registry does not match admitted request")
        if self.base_domain.participant_registry_hash != self.admitted_request.participant_registry_hash:
            raise ValidationError("base domain registry does not match admitted request")
        if self.capacity_binding.capacity_spec_hash != self.admitted_request.capacity_spec_hash:
            raise ValidationError("capacity binding specification does not match admitted request")
        if self.mitigated_request.admitted_request_hash != self.admitted_request.request_hash:
            raise ValidationError("mitigated request does not bind admitted request")
        if self.mitigated_request.admitted_values_mw != self.admitted_request.admitted_values_mw:
            raise ValidationError("mitigated request admitted values do not match admitted request")
        if self.mitigated_request.mode_specification_hash != self.mitigation_specification.specification_hash:
            raise ValidationError("mitigated request does not bind mitigation specification")
        expected_upper = FloatVector(
            min(base_upper, admitted, mitigated, capacity)
            for base_upper, admitted, mitigated, capacity in zip(
                self.base_domain.upper_bounds_mw.values,
                self.admitted_request.admitted_values_mw.values,
                self.mitigated_request.filtered_values_mw.values,
                self.capacity_binding.values_mw.values,
            )
        )
        if self.transformed_domain.upper_bounds_mw != expected_upper:
            raise ValidationError("transformed domain upper bounds do not match typed request/capacity inputs")
        if _request_hash(self.transformed_domain.participant_ids, expected_upper) != self.effective_request_hash:
            raise ValidationError("effective_request_hash does not match typed transformed upper bounds")
        if self.mitigated_request.filtered_request_hash != self.mitigated_request_hash:
            raise ValidationError("mitigated_request_hash does not match typed mitigated request")
        expected_capacity = self.transformed_domain.capacity_spec_hash
        if expected_capacity != self.capacity_binding.capacity_spec_hash:
            raise ValidationError("transformed domain capacity identity does not match typed capacity")
        ratio = self.mitigation_specification.margin_ratio.value
        expected_rows: list[LinearConstraintRow] = []
        for row in self.base_domain.constraints:
            if row.family in {"PARTICIPANT_LOWER_BOUND", "PARTICIPANT_UPPER_BOUND"}:
                expected_rows.append(row)
            elif row.family in _NETWORK_FAMILIES:
                if row.sense == "EQUAL" or row.sense not in {"LESS_EQUAL", "GREATER_EQUAL"} or row.rhs.value < 0.0:
                    raise ValidationError("base domain contains a row not transformable by the mitigation specification")
                expected_rows.append(replace(
                    row,
                    rhs=FiniteFloat((1.0 - ratio) * row.rhs.value),
                    limit_source="MITIGATION_INWARD_TIGHTENED" if ratio > 0.0 else row.limit_source,
                ))
            elif ratio > 0.0:
                raise ValidationError(f"network-margin semantics are not registered for family {row.family}")
            else:
                expected_rows.append(row)
        if self.transformed_domain.constraints != tuple(expected_rows):
            raise ValidationError("transformed network rows are not generated by the typed mitigation specification")
        marker = f"mitigation_mode={self.mitigation_specification.mode.value};specification_hash={self.mitigation_specification.specification_hash.value}"
        expected_unresolved = list(self.base_domain.unresolved_fields)
        if marker not in expected_unresolved:
            expected_unresolved.append(marker)
        if self.transformed_domain.unresolved_fields != tuple(expected_unresolved):
            raise ValidationError("transformed domain unresolved fields contain an unauthorized change")
        expected_transformed_domain = replace(
            self.base_domain,
            upper_bounds_mw=expected_upper,
            constraints=tuple(expected_rows),
            request_hash=self.effective_request_hash,
            capacity_spec_hash=self.capacity_binding.capacity_spec_hash,
            unresolved_fields=tuple(expected_unresolved),
        )
        if self.transformed_domain != expected_transformed_domain:
            raise ValidationError(
                "transformed domain changes fields outside the registered mitigation transform"
            )
        protected_domain_fields = (
            "participant_ids",
            "lower_bounds_mw",
            "upper_bound_source",
            "participant_registry_hash",
            "proxy_spec_hash",
            "feasibility_tolerance",
            "scientific_status",
        )
        for field_name in protected_domain_fields:
            if getattr(self.transformed_domain, field_name) != getattr(self.base_domain, field_name):
                raise ValidationError(f"transformed domain changed protected field {field_name}")
        if self.transformed_domain.capacity_spec_hash != self.capacity_spec_hash:
            raise ValidationError("transformed domain capacity provenance is not bound")
        if self.transformed_domain.request_hash != self.effective_request_hash:
            raise ValidationError("transformed domain must bind effective request hash")
        if self.transformed_domain.upper_bounds_mw is None:
            raise ValidationError("transformed domain must expose participant upper bounds")
        if self.status != "DIAGNOSTIC_ONLY" or self.diagnostic_only is not True:
            raise ValidationError("domain transformations remain diagnostic-only")

    @property
    def transformed_domain_hash(self) -> Sha256:
        return self.transformed_domain.domain_hash


def transform_request(
    raw_request: RawRequestVector,
    score: FloatVector,
    specification: MitigationModeSpecification,
) -> MitigatedRequestVector:
    """Apply only the registered multiplicative filtering equation."""

    if not isinstance(raw_request, RawRequestVector):
        raise ValidationError("raw_request must be a RawRequestVector")
    if not isinstance(score, FloatVector) or len(score.values) != len(raw_request.values_mw.values):
        raise ValidationError("score must align with raw request")
    if not isinstance(specification, MitigationModeSpecification):
        raise ValidationError("specification must be typed")
    return MitigatedRequestVector(
        raw_request_hash=raw_request.request_hash,
        raw_values_mw=raw_request.values_mw,
        score=score,
        filtered_values_mw=FloatVector(raw * factor for raw, factor in zip(raw_request.values_mw.values, score.values)),
        mode=specification.mode,
        mode_specification_hash=specification.specification_hash,
    )


def transform_admitted_request(
    admitted_request: AdmittedRequestVector,
    score: FloatVector,
    specification: MitigationModeSpecification,
) -> MitigatedRequestVector:
    """Apply mitigation to an already materialized admitted request.

    This is the strict R5 stage boundary.  The admission object is retained
    verbatim and its hash is carried into the mitigation record; no capacity
    or admission rule is reconstructed by the mitigation transform.
    """

    if not isinstance(admitted_request, AdmittedRequestVector):
        raise ValidationError("admitted_request must be an AdmittedRequestVector")
    if not isinstance(score, FloatVector) or len(score.values) != len(admitted_request.admitted_values_mw.values):
        raise ValidationError("score must align with admitted request")
    if not isinstance(specification, MitigationModeSpecification):
        raise ValidationError("specification must be typed")
    return MitigatedRequestVector(
        raw_request_hash=admitted_request.raw_request_hash,
        raw_values_mw=admitted_request.raw_values_mw,
        score=score,
        filtered_values_mw=FloatVector(
            admitted * factor
            for admitted, factor in zip(admitted_request.admitted_values_mw.values, score.values)
        ),
        mode=specification.mode,
        mode_specification_hash=specification.specification_hash,
        admitted_request_hash=admitted_request.request_hash,
        admitted_values_mw=admitted_request.admitted_values_mw,
    )


def _materialize_mitigated_admitted_request_legacy_diagnostic(
    raw_request: RawRequestVector,
    transformed: MitigatedRequestVector,
    *,
    capacity_spec_hash: Sha256,
) -> AdmittedRequestVector:
    """Compatibility-only helper retained for historical diagnostics.

    The leading underscore is intentional: the R5 contract does not expose a
    public stage-inversion API that turns a post-mitigation vector back into an
    ``AdmittedRequestVector``.  New callers must enter through an already
    materialized admission and :func:`apply_mitigation_mode_strict`.
    """

    if not isinstance(raw_request, RawRequestVector) or not isinstance(transformed, MitigatedRequestVector):
        raise ValidationError("raw and transformed request inputs must be typed")
    if transformed.raw_request_hash != raw_request.request_hash or transformed.raw_values_mw != raw_request.values_mw:
        raise ValidationError("transformed request does not reference raw request")
    if not isinstance(capacity_spec_hash, Sha256):
        raise ValidationError("capacity_spec_hash must be SHA-256")
    amount = FloatVector(raw - filtered for raw, filtered in zip(raw_request.values_mw.values, transformed.filtered_values_mw.values))
    mask = tuple(value > 0.0 for value in amount.values)
    return AdmittedRequestVector(
        participant_ids=raw_request.participant_ids,
        raw_values_mw=raw_request.values_mw,
        admitted_values_mw=transformed.filtered_values_mw,
        raw_request_hash=raw_request.request_hash,
        admission_policy_id=Identifier(f"MITIGATION_{transformed.mode.value}"),
        clip_mask=mask,
        clip_amount_mw=amount,
        capacity_spec_hash=capacity_spec_hash,
        source_side=raw_request.source_side,
        reporter_id=raw_request.reporter_id,
        admission_spec_hash=transformed.mode_specification_hash,
        admission_parameters={
            "mode": transformed.mode.value,
            "score": transformed.score.to_json(),
        },
        admission_semantics="GENERAL_TRANSFORM",
        participant_registry_hash=raw_request.participant_registry_hash,
    )


def materialize_effective_mitigated_request(
    transformation: DomainMitigationTransformationResult,
):
    """Materialize the request actually supplied to an allocation rule.

    The materializer accepts exactly one object-level transformation proof.
    Admission, mitigation, capacity and domain values are read from that
    proof, so a caller cannot repeat the stage join with an inconsistent raw
    capacity vector or bare domain.  The return type is imported lazily to
    avoid the intentional dependency cycle between mitigation and allocation.
    """

    from r4r.algebraic_allocation import BoundedRequestVector

    if not isinstance(transformation, DomainMitigationTransformationResult):
        raise ValidationError("effective mitigation request requires a typed domain transformation")
    transformed = transformation.mitigated_request
    admitted_request = transformation.admitted_request
    capacity_binding = transformation.capacity_binding
    transformed_domain = transformation.transformed_domain
    capacity_values_mw = capacity_binding.values_mw
    if transformed.raw_request_hash != admitted_request.raw_request_hash:
        raise ValidationError("transformed and admitted request raw hashes do not match")
    if transformed.admitted_request_hash != admitted_request.request_hash:
        raise ValidationError("transformed request must bind the admitted request hash")
    if transformed.admitted_values_mw != admitted_request.admitted_values_mw:
        raise ValidationError("transformed request must retain admitted request values")
    if transformed_domain.participant_ids != admitted_request.participant_ids:
        raise ValidationError("effective request participant order does not match transformed domain")
    if len(capacity_values_mw.values) != len(admitted_request.participant_ids.values):
        raise ValidationError("effective request capacity vector does not align with participants")
    if any(value < 0.0 for value in capacity_values_mw.values):
        raise ValidationError("effective request capacity values must be nonnegative")
    effective_input = transformed.filtered_values_mw
    expected_upper = FloatVector(
        min(admitted, mitigated, capacity, domain_upper)
        for admitted, mitigated, capacity, domain_upper in zip(
            admitted_request.admitted_values_mw.values,
            effective_input.values,
            capacity_values_mw.values,
            transformed_domain.upper_bounds_mw.values,
        )
    )
    if transformed_domain.upper_bounds_mw != expected_upper:
        raise ValidationError("transformed domain upper bounds are not the admitted/mitigated/capacity/domain minimum")
    effective = expected_upper
    bound_amount = FloatVector(
        filtered - effective
        for filtered, effective in zip(transformed.filtered_values_mw.values, effective.values)
    )
    return BoundedRequestVector(
        values_mw=effective,
        source_request_hash=transformed.raw_request_hash,
        participant_ids=admitted_request.participant_ids,
        filtered_request_hash=transformed.filtered_request_hash,
        admitted_request_hash=admitted_request.request_hash,
        capacity_spec_hash=admitted_request.capacity_spec_hash,
        domain_hash=transformed_domain.domain_hash,
        bound_mask=tuple(value > 1e-12 for value in bound_amount.values),
        bound_amount_mw=bound_amount,
    )


def tighten_network_domain(
    domain: FeasibleAllocationDomain,
    *,
    capacity_values_mw: FloatVector,
    admitted_request: AdmittedRequestVector,
    transformed_request: MitigatedRequestVector,
    side: AllocationSide,
    specification: MitigationModeSpecification,
) -> FeasibleAllocationDomain:
    """Apply inward network-margin reservation and return the closed domain.

    Participant capacity bounds are not margin-tightened.  Network rows are
    scaled toward the zero incremental baseline; equality rows and unknown
    network families fail closed instead of receiving an invented transform.
    ``transformed_request`` is mandatory so a caller cannot accidentally build
    a domain from the pre-mitigation admitted vector.
    """

    return tighten_network_domain_result(
        domain,
        capacity_binding=ExplicitCapacityVector(
            participant_ids=domain.participant_ids,
            values_mw=capacity_values_mw,
            capacity_spec_hash=admitted_request.capacity_spec_hash,
            participant_registry_hash=admitted_request.participant_registry_hash
            or domain.participant_registry_hash,
        ),
        admitted_request=admitted_request,
        transformed_request=transformed_request,
        side=side,
        specification=specification,
    ).transformed_domain


def tighten_network_domain_result(
    domain: FeasibleAllocationDomain,
    *,
    capacity_binding: ExplicitCapacityVector,
    admitted_request: AdmittedRequestVector,
    transformed_request: MitigatedRequestVector,
    side: AllocationSide,
    specification: MitigationModeSpecification,
    base_domain_hash: Sha256 | None = None,
) -> DomainMitigationTransformationResult:
    """Return the typed mitigation/domain transformation proof."""

    if not isinstance(domain, FeasibleAllocationDomain) or not isinstance(capacity_binding, ExplicitCapacityVector):
        raise ValidationError("domain and capacity binding must be typed")
    if not isinstance(admitted_request, AdmittedRequestVector) or not isinstance(transformed_request, MitigatedRequestVector):
        raise ValidationError("admitted and transformed requests must be typed")
    if not isinstance(side, AllocationSide) or not isinstance(specification, MitigationModeSpecification):
        raise ValidationError("side and specification must be typed")
    if domain.participant_ids != admitted_request.participant_ids or capacity_binding.participant_ids != domain.participant_ids:
        raise ValidationError("mitigated domain participant order does not match request/capacity")
    if capacity_binding.capacity_spec_hash != admitted_request.capacity_spec_hash:
        raise ValidationError("capacity binding does not match admitted request")
    if capacity_binding.participant_registry_hash != admitted_request.participant_registry_hash:
        raise ValidationError("capacity binding registry does not match admitted request")
    if transformed_request.admitted_request_hash != admitted_request.request_hash:
        raise ValidationError("transformed request does not bind admitted request")
    if transformed_request.admitted_values_mw != admitted_request.admitted_values_mw:
        raise ValidationError("transformed request admitted values do not match admission")
    if transformed_request.mode_specification_hash != specification.specification_hash:
        raise ValidationError("transformed request does not bind mitigation specification")
    if transformed_request.raw_request_hash != admitted_request.raw_request_hash:
        raise ValidationError("transformed request raw provenance does not match admission")
    if base_domain_hash is None:
        base_domain_hash = domain.domain_hash
    if not isinstance(base_domain_hash, Sha256) or base_domain_hash != domain.domain_hash:
        raise ValidationError("base_domain_hash must equal the supplied base domain hash")
    capacity_spec_hash = capacity_binding.capacity_spec_hash
    if domain.capacity_spec_hash is not None and domain.capacity_spec_hash != capacity_spec_hash:
        raise ValidationError("domain capacity provenance does not match admitted request")
    upper = FloatVector(
        min(base_upper, admitted, mitigated, capacity)
        for base_upper, admitted, mitigated, capacity in zip(
            domain.upper_bounds_mw.values,
            admitted_request.admitted_values_mw.values,
            transformed_request.filtered_values_mw.values,
            capacity_binding.values_mw.values,
        )
    )
    ratio = specification.margin_ratio.value
    rows: list[LinearConstraintRow] = []
    for row in domain.constraints:
        if row.family in {"PARTICIPANT_LOWER_BOUND", "PARTICIPANT_UPPER_BOUND"}:
            rows.append(row)
            continue
        if row.family not in _NETWORK_FAMILIES:
            if ratio > 0.0:
                raise ValidationError(f"network-margin semantics are not registered for family {row.family}")
            rows.append(row)
            continue
        if row.sense == "EQUAL":
            raise ValidationError("network-margin reservation cannot transform equality rows")
        if row.sense not in {"LESS_EQUAL", "GREATER_EQUAL"} or row.rhs.value < 0.0:
            raise ValidationError("network-margin reservation requires nonnegative signed bound rows")
        rows.append(replace(
            row,
            rhs=FiniteFloat((1.0 - ratio) * row.rhs.value),
            limit_source="MITIGATION_INWARD_TIGHTENED" if ratio > 0.0 else row.limit_source,
        ))
    unresolved = list(domain.unresolved_fields)
    marker = f"mitigation_mode={specification.mode.value};specification_hash={specification.specification_hash.value}"
    if marker not in unresolved:
        unresolved.append(marker)
    effective_hash = _request_hash(domain.participant_ids, upper)
    transformed_domain = replace(
        domain,
        upper_bounds_mw=upper,
        constraints=tuple(rows),
        request_hash=effective_hash,
        capacity_spec_hash=capacity_spec_hash,
        unresolved_fields=tuple(unresolved),
    )
    return DomainMitigationTransformationResult(
        side=side,
        base_domain_hash=base_domain_hash,
        base_domain=domain,
        transformed_domain=transformed_domain,
        admitted_request=admitted_request,
        mitigated_request=transformed_request,
        capacity_binding=capacity_binding,
        mitigation_specification=specification,
        mitigation_specification_hash=specification.specification_hash,
        capacity_spec_hash=capacity_spec_hash,
        admitted_request_hash=admitted_request.request_hash,
        mitigated_request_hash=transformed_request.filtered_request_hash,
        effective_request_hash=effective_hash,
    )


def apply_mitigation_mode_legacy_diagnostic(
    domain: FeasibleAllocationDomain,
    raw_request: RawRequestVector,
    *,
    capacity_values_mw: FloatVector,
    side: AllocationSide,
    reporter_id: Identifier | None,
    specification: MitigationModeSpecification,
) -> tuple[MitigatedRequestVector, AdmittedRequestVector, FeasibleAllocationDomain]:
    """Compatibility-only raw-request helper; never use for the strict R5 path."""

    capacity_hash = domain.capacity_spec_hash
    if capacity_hash is None:
        capacity_hash = Sha256(canonical_hash({"capacity_values_mw": capacity_values_mw.to_json()}))
    capacity_binding = ExplicitCapacityVector(
        participant_ids=raw_request.participant_ids,
        values_mw=capacity_values_mw,
        capacity_spec_hash=capacity_hash,
        participant_registry_hash=raw_request.participant_registry_hash,
    )
    if raw_request.source_side is not side:
        raise ValidationError("raw request side does not match mitigation side")
    admitted = IdentityAdmissionPolicy().admit(
        raw_request,
        capacity_spec_hash=capacity_hash,
        admission_spec_hash=capacity_hash,
        admission_parameters={"policy": "IDENTITY_ADMISSION"},
        admission_semantics="CLIPPING_ONLY",
    )
    transformed, admitted, transformation = apply_mitigation_mode_strict(
        domain,
        admitted,
        capacity_binding=capacity_binding,
        side=side,
        reporter_id=reporter_id,
        specification=specification,
    )
    return transformed, admitted, transformation.transformed_domain


def apply_mitigation_mode_strict(
    domain: FeasibleAllocationDomain,
    admitted_request: AdmittedRequestVector,
    *,
    capacity_binding: ExplicitCapacityVector,
    side: AllocationSide,
    reporter_id: Identifier | None,
    specification: MitigationModeSpecification,
) -> tuple[MitigatedRequestVector, AdmittedRequestVector, DomainMitigationTransformationResult]:
    """Strict diagnostic entry point from an existing admitted request.

    Admission is intentionally outside this function.  The strict R5 path
    cannot reconstruct identity admission from a raw request and thereby lose
    a scenario bundle's clipping or other registered admission semantics.
    """

    if not isinstance(admitted_request, AdmittedRequestVector):
        raise ValidationError("strict mitigation requires an existing AdmittedRequestVector")
    if not isinstance(capacity_binding, ExplicitCapacityVector):
        raise ValidationError("strict mitigation requires an ExplicitCapacityVector")
    if admitted_request.source_side is not side:
        raise ValidationError("admitted request side does not match mitigation side")
    if admitted_request.admission_semantics == "GENERAL_TRANSFORM":
        raise ValidationError("strict mitigation cannot consume a mitigation-derived admission")
    if admitted_request.admission_policy_id.value.startswith("MITIGATION_"):
        raise ValidationError("strict mitigation cannot reapply mitigation to a transformed admission")
    if side is AllocationSide.REPORTED:
        if admitted_request.reporter_id is None or reporter_id != admitted_request.reporter_id:
            raise ValidationError("reported mitigation reporter does not match admitted request reporter")
    elif reporter_id is not None or admitted_request.reporter_id is not None:
        raise ValidationError("reference mitigation cannot carry a reporter")
    if admitted_request.participant_ids != capacity_binding.participant_ids:
        raise ValidationError("admitted request and capacity binding participant order do not match")
    if admitted_request.capacity_spec_hash != capacity_binding.capacity_spec_hash:
        raise ValidationError("admitted request and capacity binding specification do not match")
    if admitted_request.participant_registry_hash != capacity_binding.participant_registry_hash:
        raise ValidationError("admitted request and capacity binding registry do not match")
    scores = build_static_score_vector(
        admitted_request.participant_ids,
        side=side,
        reporter_id=reporter_id,
        specification=specification,
    )
    transformed = transform_admitted_request(admitted_request, scores, specification)
    transformation = tighten_network_domain_result(
        domain,
        capacity_binding=capacity_binding,
        admitted_request=admitted_request,
        transformed_request=transformed,
        side=side,
        specification=specification,
    )
    return transformed, admitted_request, transformation


apply_mitigation_to_admitted_request_strict = apply_mitigation_mode_strict


__all__ = [
    "MitigationModeSpecification",
    "MitigatedRequestVector",
    "DomainMitigationTransformationResult",
    "apply_mitigation_mode_strict",
    "apply_mitigation_to_admitted_request_strict",
    "build_manuscript_mitigation_specs",
    "build_static_score_vector",
    "materialize_effective_mitigated_request",
    "tighten_network_domain",
    "tighten_network_domain_result",
    "transform_admitted_request",
    "transform_request",
]
