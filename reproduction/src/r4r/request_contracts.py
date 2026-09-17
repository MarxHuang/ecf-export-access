"""Raw, admitted and strategic-report request contracts.

These objects intentionally separate a reported value from any later
admission/clipping policy. No capacity rule or effective-deviation threshold
is inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any
import math

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, Sha256


def _freeze_parameter(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_parameter(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_parameter(item) for item in value))
    return value


@dataclass(frozen=True, slots=True)
class FrozenParameterMap(Mapping[str, Any]):
    """Immutable, canonicalizable report/admission parameter mapping."""

    _values: MappingProxyType

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise ValidationError("parameters must be a mapping")
        if any(not isinstance(key, str) for key in values):
            raise ValidationError("parameter keys must be strings")
        frozen = {key: _freeze_parameter(value) for key, value in values.items()}
        try:
            from r4r.serialization import canonical_bytes
            canonical_bytes(frozen)
        except Exception as exc:
            raise ValidationError("parameters must be canonicalizable") from exc
        object.__setattr__(self, "_values", MappingProxyType(frozen))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._values) == dict(other)
        return NotImplemented

    def to_json(self) -> dict[str, Any]:
        return dict(self._values)


def _validate_side_reporter(side: AllocationSide, reporter_id: Identifier | None) -> None:
    if not isinstance(side, AllocationSide):
        raise ValidationError("request side must be reference or reported")
    if side is AllocationSide.REFERENCE and reporter_id is not None:
        raise ValidationError("reference request cannot carry a reporter ID")
    if side is AllocationSide.REPORTED and reporter_id is None:
        raise ValidationError("reported request requires a reporter ID")
    if reporter_id is not None and not isinstance(reporter_id, Identifier):
        raise ValidationError("reporter_id must be an Identifier")


_ADMISSION_SEMANTICS = {"CLIPPING_ONLY", "GENERAL_TRANSFORM", "UNRESOLVED"}


@dataclass(frozen=True, slots=True)
class RawRequestVector(ContractModel):
    """The request before any admission or clipping policy is applied."""

    participant_ids: IdentifierVector
    values_mw: FloatVector
    request_spec_id: Identifier
    participant_registry_hash: Sha256
    source_side: AllocationSide
    reporter_id: Identifier | None
    serialization_id = "raw_request_vector.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector):
            raise ValidationError("participant_ids must be an IdentifierVector")
        if not isinstance(self.values_mw, FloatVector):
            raise ValidationError("values_mw must be a finite FloatVector")
        if len(self.participant_ids.values) != len(self.values_mw.values):
            raise ValidationError("raw request vectors must align")
        if not isinstance(self.request_spec_id, Identifier):
            raise ValidationError("request_spec_id must be an Identifier")
        if not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be SHA-256")
        _validate_side_reporter(self.source_side, self.reporter_id)

    @property
    def request_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class AdmittedRequestVector(ContractModel):
    """Request after an explicitly named admission policy."""

    participant_ids: IdentifierVector
    raw_values_mw: FloatVector
    admitted_values_mw: FloatVector
    raw_request_hash: Sha256
    admission_policy_id: Identifier
    clip_mask: tuple[bool, ...]
    clip_amount_mw: FloatVector
    capacity_spec_hash: Sha256
    source_side: AllocationSide
    reporter_id: Identifier | None
    admission_spec_hash: Sha256 | None = None
    admission_parameters: FrozenParameterMap | None = None
    admission_semantics: str = "UNRESOLVED"
    participant_registry_hash: Sha256 | None = None
    serialization_id = "admitted_request_vector.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector):
            raise ValidationError("participant_ids must be an IdentifierVector")
        if not isinstance(self.raw_values_mw, FloatVector) or not isinstance(self.admitted_values_mw, FloatVector):
            raise ValidationError("raw and admitted values must be finite FloatVectors")
        n = len(self.participant_ids.values)
        if len(self.raw_values_mw.values) != n or len(self.admitted_values_mw.values) != n:
            raise ValidationError("raw/admitted request vectors must align")
        if any(value < 0 for value in self.admitted_values_mw.values):
            raise ValidationError("admitted request values must be nonnegative")
        if len(self.clip_mask) != n or any(not isinstance(value, bool) for value in self.clip_mask):
            raise ValidationError("clip_mask must align and contain booleans")
        if len(self.clip_amount_mw.values) != n or any(value < 0 for value in self.clip_amount_mw.values):
            raise ValidationError("clip_amount_mw must be aligned and nonnegative")
        # A clipping audit is an explicit record, not a hint.  When a mask
        # says that an element was clipped, the recorded amount must be the
        # exact raw-to-admitted reduction; unmarked elements must carry zero
        # clipping.  This keeps raw strategic reports and admitted requests
        # independently reproducible without imposing an admission policy.
        for raw, admitted, clipped, amount in zip(
            self.raw_values_mw.values,
            self.admitted_values_mw.values,
            self.clip_mask,
            self.clip_amount_mw.values,
        ):
            expected = raw - admitted
            if not clipped:
                if (
                    not math.isclose(raw, admitted, rel_tol=0.0, abs_tol=1e-12)
                    and self.admission_semantics != "GENERAL_TRANSFORM"
                ):
                    raise ValidationError("non-clipping request changes require GENERAL_TRANSFORM semantics")
                if not math.isclose(amount, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    raise ValidationError("unclipped request elements must have zero clip amount")
            else:
                if expected <= 0.0 or not math.isclose(amount, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValidationError("clip amount must equal the raw-to-admitted reduction")
        if self.admission_semantics == "CLIPPING_ONLY" and any(
            admitted > raw
            for raw, admitted in zip(self.raw_values_mw.values, self.admitted_values_mw.values)
        ):
            raise ValidationError("CLIPPING_ONLY admission cannot increase request values")
        if not isinstance(self.raw_request_hash, Sha256) or not isinstance(self.capacity_spec_hash, Sha256):
            raise ValidationError("request and capacity hashes must be SHA-256")
        if self.admission_spec_hash is not None and not isinstance(self.admission_spec_hash, Sha256):
            raise ValidationError("admission_spec_hash must be SHA-256 or null")
        if not isinstance(self.admission_policy_id, Identifier):
            raise ValidationError("admission_policy_id must be an Identifier")
        if self.admission_semantics not in _ADMISSION_SEMANTICS:
            raise ValidationError("admission_semantics is not registered")
        if self.admission_semantics != "UNRESOLVED" and self.admission_spec_hash is None:
            raise ValidationError("resolved admission semantics require an admission specification hash")
        if self.participant_registry_hash is not None and not isinstance(self.participant_registry_hash, Sha256):
            raise ValidationError("participant_registry_hash must be SHA-256 or null")
        if self.admission_parameters is not None:
            if isinstance(self.admission_parameters, FrozenParameterMap):
                pass
            elif isinstance(self.admission_parameters, Mapping):
                object.__setattr__(self, "admission_parameters", FrozenParameterMap(self.admission_parameters))
            else:
                raise ValidationError("admission_parameters must be a mapping or null")
        _validate_side_reporter(self.source_side, self.reporter_id)

    @property
    def request_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class StrategicReportResult(ContractModel):
    """Reference/report pair with raw and admitted deviations kept separate."""

    reference_raw_request: RawRequestVector
    reported_raw_request: RawRequestVector
    reported_admitted_request: AdmittedRequestVector
    reporter_id: Identifier
    changed_participant_ids: IdentifierVector
    effective_deviation_status: str
    report_parameters: FrozenParameterMap | None = None
    serialization_id = "strategic_report_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.reference_raw_request, RawRequestVector) or not isinstance(self.reported_raw_request, RawRequestVector):
            raise ValidationError("strategic result requires raw reference and reported requests")
        if not isinstance(self.reported_admitted_request, AdmittedRequestVector):
            raise ValidationError("strategic result requires admitted reported request")
        if self.reference_raw_request.source_side is not AllocationSide.REFERENCE:
            raise ValidationError("reference_raw_request has wrong side")
        if self.reported_raw_request.source_side is not AllocationSide.REPORTED:
            raise ValidationError("reported_raw_request has wrong side")
        if self.reported_admitted_request.source_side is not AllocationSide.REPORTED:
            raise ValidationError("reported_admitted_request has wrong side")
        if self.reporter_id != self.reported_raw_request.reporter_id or self.reporter_id != self.reported_admitted_request.reporter_id:
            raise ValidationError("reporter IDs must align across report layers")
        if self.reference_raw_request.participant_ids != self.reported_raw_request.participant_ids:
            raise ValidationError("reference and reported raw vectors must align")
        if self.reported_raw_request.participant_ids != self.reported_admitted_request.participant_ids:
            raise ValidationError("raw and admitted reported vectors must align")
        if self.reported_admitted_request.raw_request_hash != self.reported_raw_request.request_hash:
            raise ValidationError("admitted request must reference the reported raw request hash")
        if self.reported_admitted_request.raw_values_mw != self.reported_raw_request.values_mw:
            raise ValidationError("admitted request must preserve the reported raw request vector")
        if (
            self.reported_admitted_request.participant_registry_hash is not None
            and self.reported_admitted_request.participant_registry_hash
            != self.reported_raw_request.participant_registry_hash
        ):
            raise ValidationError("admitted request must preserve the participant registry hash")
        if self.reference_raw_request.participant_registry_hash != self.reported_raw_request.participant_registry_hash:
            raise ValidationError("reference and reported requests must share participant registry hash")
        if self.reference_raw_request.request_spec_id != self.reported_raw_request.request_spec_id:
            raise ValidationError("reference and reported requests must share request specification")
        if not isinstance(self.changed_participant_ids, IdentifierVector):
            raise ValidationError("changed_participant_ids must be an IdentifierVector")
        if any(item not in self.reported_raw_request.participant_ids.values for item in self.changed_participant_ids.values):
            raise ValidationError("changed participant is absent from the report registry")
        expected_changed = tuple(
            participant_id
            for participant_id, reference, reported in zip(
                self.reference_raw_request.participant_ids.values,
                self.reference_raw_request.values_mw.values,
                self.reported_raw_request.values_mw.values,
            )
            if reported != reference
        )
        if tuple(self.changed_participant_ids.values) != expected_changed:
            raise ValidationError("changed_participant_ids must equal the raw reference/report difference set")
        if self.effective_deviation_status not in {"UNRESOLVED", "EFFECTIVE", "NOT_EFFECTIVE", "NOT_APPLICABLE"}:
            raise ValidationError("effective_deviation_status is not registered")
        if self.report_parameters is not None:
            if isinstance(self.report_parameters, FrozenParameterMap):
                pass
            elif isinstance(self.report_parameters, Mapping):
                object.__setattr__(self, "report_parameters", FrozenParameterMap(self.report_parameters))
            else:
                raise ValidationError("report_parameters must be a mapping or null")

    @property
    def raw_deviation_mw(self) -> FloatVector:
        return FloatVector(
            reported - reference
            for reference, reported in zip(
                self.reference_raw_request.values_mw.values,
                self.reported_raw_request.values_mw.values,
            )
        )

    @property
    def admitted_deviation_mw(self) -> FloatVector:
        return FloatVector(
            admitted - reference
            for reference, admitted in zip(
                self.reference_raw_request.values_mw.values,
                self.reported_admitted_request.admitted_values_mw.values,
            )
        )
