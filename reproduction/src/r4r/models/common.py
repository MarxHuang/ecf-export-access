"""Common atomic entities represented without numerical execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from r4r.errors import ValidationError
from r4r.contracts.registry_loader import gate_failure_reason_ids, registered_failure_reason_ids
from r4r.models.base import ContractModel
from r4r.types import (
    DiagnosticOrPrimary,
    FiniteFloat,
    FloatVector,
    GateStatus,
    Identifier,
    IdentifierVector,
    MarginMode as MarginModeId,
    MitigationMode as MitigationModeId,
    ObjectReference,
    QAssumption as QAssumptionId,
    Sha256,
)


@dataclass(frozen=True, slots=True)
class OperatingPoint(ContractModel):
    load_scale: FiniteFloat
    q_assumption: QAssumptionId
    p_load: FloatVector
    q_load: FloatVector
    serialization_id = "operating_point.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.q_assumption, QAssumptionId):
            raise ValidationError("q_assumption must be a registered Q token")
        if len(self.p_load.values) != len(self.q_load.values):
            raise ValidationError("p_load and q_load shapes must match")


@dataclass(frozen=True, slots=True)
class QAssumption(ContractModel):
    q_id: QAssumptionId
    value: Any
    serialization_id = "q_assumption.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.q_id, QAssumptionId):
            raise ValidationError("q_id must be a registered Q token")


@dataclass(frozen=True, slots=True)
class MarginMode(ContractModel):
    mode_id: MarginModeId
    multiplier: FiniteFloat | None
    serialization_id = "margin_mode.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, MarginModeId):
            raise ValidationError("mode_id must be M0, M1 or M125")
        if self.multiplier is not None and not isinstance(self.multiplier, FiniteFloat):
            raise ValidationError("multiplier must be finite or explicit null")


@dataclass(frozen=True, slots=True)
class MitigationMode(ContractModel):
    mode_id: MitigationModeId
    counterfactual_id: MitigationModeId
    diagnostic_or_primary: DiagnosticOrPrimary
    serialization_id = "mitigation_mode.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, MitigationModeId) or not isinstance(self.counterfactual_id, MitigationModeId):
            raise ValidationError("mitigation IDs must be registered six-mode tokens")
        if not isinstance(self.diagnostic_or_primary, DiagnosticOrPrimary):
            raise ValidationError("diagnostic_or_primary is required")


@dataclass(frozen=True, slots=True)
class GateResult(ContractModel):
    gate_id: Identifier
    passed: bool
    failure_reasons: IdentifierVector
    serialization_id = "gate_result.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, Identifier) or not isinstance(self.passed, bool):
            raise ValidationError("gate_id and passed have strict types")
        if not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure_reasons must be an IdentifierVector")
        gate_id = self.gate_id.value
        allowed_by_gate = gate_failure_reason_ids()
        if gate_id not in allowed_by_gate:
            raise ValidationError(f"gate_id is not a registered gate: {gate_id}")
        reason_ids = tuple(reason.value for reason in self.failure_reasons.values)
        unknown = set(reason_ids) - registered_failure_reason_ids()
        if unknown:
            raise ValidationError(f"unregistered failure reasons: {sorted(unknown)}")
        disallowed = set(reason_ids) - set(allowed_by_gate[gate_id])
        if disallowed:
            raise ValidationError(
                f"failure reasons are not allowed for {gate_id}: {sorted(disallowed)}"
            )
        if self.passed and reason_ids:
            raise ValidationError("passed gates must not carry failure reasons")
        if not self.passed and not reason_ids:
            raise ValidationError("failed gates must carry at least one failure reason")


@dataclass(frozen=True, slots=True)
class ProvenanceFields(ContractModel):
    source_hash: Sha256
    overlay_hash: Sha256
    merge_tool_hash: Sha256
    serialization_id = "provenance.v1"
