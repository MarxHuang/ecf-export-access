"""Explicit request/report/admission policies for the R5 interface slice."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from r4r.errors import ReferenceError, ValidationError
from r4r.request_contracts import AdmittedRequestVector, FrozenParameterMap, RawRequestVector, StrategicReportResult
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, Sha256


class ExplicitReferenceRequestPolicy:
    """Materialize a caller-provided reference request without transformation."""

    def build(
        self,
        participant_ids: IdentifierVector,
        values_mw: Sequence[float],
        *,
        request_spec_id: Identifier,
        participant_registry_hash: Sha256,
    ) -> RawRequestVector:
        values = FloatVector(values_mw)
        if len(values.values) != len(participant_ids.values):
            raise ValidationError("reference request must align with participant IDs")
        return RawRequestVector(
            participant_ids,
            values,
            request_spec_id,
            participant_registry_hash,
            AllocationSide.REFERENCE,
            None,
        )


class ExplicitStrategicReportPolicy:
    """Apply an explicit unilateral report transform without clipping."""

    def transform(
        self,
        reference: RawRequestVector,
        reporter_id: Identifier,
        reported_values_mw: Sequence[float],
    ) -> RawRequestVector:
        if reference.source_side is not AllocationSide.REFERENCE:
            raise ValidationError("strategic transform requires a reference raw request")
        if reporter_id not in reference.participant_ids.values:
            raise ReferenceError("reporter is absent from the participant registry")
        reported = FloatVector(reported_values_mw)
        if len(reported.values) != len(reference.values_mw.values):
            raise ValidationError("reported request must align with the reference vector")
        reporter_index = reference.participant_ids.values.index(reporter_id)
        for index, (before, after) in enumerate(zip(reference.values_mw.values, reported.values)):
            if index != reporter_index and after != before:
                raise ValidationError("single-reporter policy changed a non-reporter element")
        return RawRequestVector(
            reference.participant_ids,
            reported,
            reference.request_spec_id,
            reference.participant_registry_hash,
            AllocationSide.REPORTED,
            reporter_id,
        )


class ExplicitAdmissionPolicy:
    """Record an explicit admitted vector and clipping audit without inferring a rule."""

    def admit(
        self,
        raw: RawRequestVector,
        admitted_values_mw: Sequence[float],
        *,
        admission_policy_id: Identifier,
        capacity_spec_hash: Sha256,
        clip_mask: Sequence[bool],
        clip_amount_mw: Sequence[float],
        admission_spec_hash: Sha256 | None = None,
        admission_parameters: Mapping[str, Any] | None = None,
        admission_semantics: str = "UNRESOLVED",
    ) -> AdmittedRequestVector:
        admitted = FloatVector(admitted_values_mw)
        amount = FloatVector(clip_amount_mw)
        if len(admitted.values) != len(raw.values_mw.values):
            raise ValidationError("admitted request must align with raw request")
        if len(clip_mask) != len(admitted.values) or any(not isinstance(value, bool) for value in clip_mask):
            raise ValidationError("clip_mask must align and contain booleans")
        if len(amount.values) != len(admitted.values):
            raise ValidationError("clip amount must align with admitted request")
        return AdmittedRequestVector(
            raw.participant_ids,
            raw.values_mw,
            admitted,
            raw.request_hash,
            admission_policy_id,
            tuple(clip_mask),
            amount,
            capacity_spec_hash,
            raw.source_side,
            raw.reporter_id,
            admission_spec_hash,
            FrozenParameterMap(admission_parameters) if admission_parameters is not None else None,
            admission_semantics,
            raw.participant_registry_hash,
        )


class IdentityAdmissionPolicy:
    """Diagnostic identity admission; it does not impose a capacity bound."""

    def admit(
        self,
        raw: RawRequestVector,
        *,
        capacity_spec_hash: Sha256,
        admission_spec_hash: Sha256 | None = None,
        admission_parameters: Mapping[str, Any] | None = None,
        admission_semantics: str = "UNRESOLVED",
    ) -> AdmittedRequestVector:
        return ExplicitAdmissionPolicy().admit(
            raw,
            raw.values_mw.values,
            admission_policy_id=Identifier("IDENTITY_ADMISSION"),
            capacity_spec_hash=capacity_spec_hash,
            clip_mask=[False] * len(raw.values_mw.values),
            clip_amount_mw=[0.0] * len(raw.values_mw.values),
            admission_spec_hash=admission_spec_hash,
            admission_parameters=admission_parameters,
            admission_semantics=admission_semantics,
        )


def materialize_strategic_report(
    reference: RawRequestVector,
    reporter_id: Identifier,
    reported_values_mw: Sequence[float],
    *,
    admitted_values_mw: Sequence[float],
    admission_policy_id: Identifier,
    capacity_spec_hash: Sha256,
    clip_mask: Sequence[bool],
    clip_amount_mw: Sequence[float],
    effective_deviation_status: str = "UNRESOLVED",
    report_parameters: Mapping[str, Any] | None = None,
    admission_spec_hash: Sha256 | None = None,
    admission_parameters: Mapping[str, Any] | None = None,
    admission_semantics: str = "UNRESOLVED",
) -> StrategicReportResult:
    """Materialize all report layers without promoting a deviation claim.

    The raw difference set is derived mechanically.  ``effective_deviation``
    remains an explicit caller input because EQ006/PAR062 are evaluated by the
    separately bound diagnostic evaluator; this helper must not silently turn
    a changed report into a formal strategic-deviation or manuscript claim.
    """
    reported = ExplicitStrategicReportPolicy().transform(
        reference, reporter_id, reported_values_mw
    )
    admitted = ExplicitAdmissionPolicy().admit(
        reported,
        admitted_values_mw,
        admission_policy_id=admission_policy_id,
        capacity_spec_hash=capacity_spec_hash,
        clip_mask=clip_mask,
        clip_amount_mw=clip_amount_mw,
        admission_spec_hash=admission_spec_hash,
        admission_parameters=admission_parameters,
        admission_semantics=admission_semantics,
    )
    changed = IdentifierVector(
        participant_id
        for participant_id, before, after in zip(
            reference.participant_ids.values,
            reference.values_mw.values,
            reported.values_mw.values,
        )
        if after != before
    )
    return StrategicReportResult(
        reference_raw_request=reference,
        reported_raw_request=reported,
        reported_admitted_request=admitted,
        reporter_id=reporter_id,
        changed_participant_ids=changed,
        effective_deviation_status=effective_deviation_status,
        report_parameters=report_parameters,
    )


__all__ = [
    "ExplicitAdmissionPolicy",
    "ExplicitReferenceRequestPolicy",
    "ExplicitStrategicReportPolicy",
    "IdentityAdmissionPolicy",
    "materialize_strategic_report",
]
