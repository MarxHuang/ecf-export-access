"""Calibration strata/profile/result records; no calibration execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier, require_reference
from r4r.contracts.registry_loader import require_registered_failure_reasons
from r4r.types import (
    CalibrationProfileKind,
    FloatVector,
    GateStatus,
    Identifier,
    IdentifierVector,
    ObjectReference,
    ObjectReferenceVector,
    QAssumption,
    Sha256,
    SplitKind,
)
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class CalibrationStratum(ContractModel):
    capacity_model: Identifier
    parameter_id: Identifier
    zeta: FiniteFloat
    calibration_q_assumption: ObjectReference
    serialization_id = "calibration_stratum.v1"

    def __post_init__(self) -> None:
        require_identifier(self.capacity_model, "capacity_model")
        require_identifier(self.parameter_id, "parameter_id")
        require_reference(self.calibration_q_assumption, "calibration_q_assumption", "ENT009")


@dataclass(frozen=True, slots=True)
class CalibrationProfile(ContractModel):
    profile_id: Identifier
    allocation_mw: FloatVector
    split: SplitKind
    profile_kind: CalibrationProfileKind | None = None
    perturbed_participant_ids: IdentifierVector = field(default_factory=lambda: IdentifierVector([]))
    perturbation_amplitudes_mw: FloatVector = field(default_factory=lambda: FloatVector([]))
    source_scenario_id: Identifier | None = None
    allocation_source: str = "UNSPECIFIED"
    serialization_id = "calibration_profile.v1"

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "profile_id")
        if not isinstance(self.split, SplitKind):
            raise ValidationError("calibration split must be train, holdout or realized audit")
        if self.profile_kind is not None and not isinstance(self.profile_kind, CalibrationProfileKind):
            raise ValidationError("calibration profile_kind must be a registered CalibrationProfileKind")
        if not isinstance(self.perturbed_participant_ids, IdentifierVector):
            raise ValidationError("perturbed_participant_ids must be IdentifierVector")
        if not isinstance(self.perturbation_amplitudes_mw, FloatVector):
            raise ValidationError("perturbation_amplitudes_mw must be FloatVector")
        if any(value <= 0.0 for value in self.perturbation_amplitudes_mw.values):
            raise ValidationError("perturbation amplitudes must be positive")
        if self.source_scenario_id is not None:
            require_identifier(self.source_scenario_id, "source_scenario_id")
        if not isinstance(self.allocation_source, str) or not self.allocation_source:
            raise ValidationError("allocation_source must be a non-empty string")
        if self.profile_kind in {
            CalibrationProfileKind.SINGLE_BUS,
            CalibrationProfileKind.MULTI_BUS,
            CalibrationProfileKind.COMMON_PROXY,
        }:
            if not self.perturbed_participant_ids.values:
                raise ValidationError("finite-amplitude profiles must declare perturbed participants")
            if tuple(self.perturbation_amplitudes_mw.values) != (1e-4, 1e-3):
                raise ValidationError("finite-amplitude profiles must declare [1e-4, 1e-3] MW")
            if self.source_scenario_id is None:
                raise ValidationError("finite-amplitude profiles must declare source_scenario_id")

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "profile_id": self.profile_id.to_json(),
            "allocation_mw": self.allocation_mw.to_json(),
            "split": self.split.value,
            "profile_kind": self.profile_kind.value if self.profile_kind is not None else None,
            "perturbed_participant_ids": self.perturbed_participant_ids.to_json(),
            "perturbation_amplitudes_mw": self.perturbation_amplitudes_mw.to_json(),
            "source_scenario_id": self.source_scenario_id.to_json() if self.source_scenario_id is not None else None,
            "allocation_source": self.allocation_source,
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult(ContractModel):
    FAILURE_GATE_IDS: ClassVar[tuple[str, ...]] = ("G008", "G009", "G010")
    stratum_id: ObjectReference
    q_assumption: QAssumption
    train_profile_count: int
    train_profile_hash: Sha256
    holdout_count: int
    holdout_hash: Sha256
    branch_margin_vector_mw: FloatVector
    voltage_margin_vector_pu: FloatVector
    exceedance_records: IdentifierVector
    realized_audit_records: IdentifierVector
    status: GateStatus
    failure_reasons: IdentifierVector = field(default_factory=lambda: IdentifierVector([]))
    serialization_id = "calibration_result.v1"

    def __post_init__(self) -> None:
        require_reference(self.stratum_id, "stratum_id", "ENT017")
        if not isinstance(self.q_assumption, QAssumption):
            raise ValidationError("q_assumption must be registered")
        if self.train_profile_count < 0 or self.holdout_count < 0:
            raise ValidationError("calibration profile counts must be nonnegative")
        if not isinstance(self.status, GateStatus):
            raise ValidationError("calibration status must be registered")
        try:
            require_registered_failure_reasons(self.failure_reasons, gate_ids=self.FAILURE_GATE_IDS, field="CalibrationResult.failure_reasons")
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
