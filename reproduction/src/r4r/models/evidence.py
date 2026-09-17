"""Evidence and mitigation records; promotion remains a separate gate."""
from __future__ import annotations

from dataclasses import dataclass

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier
from r4r.types import DiagnosticOrPrimary, GateStatus, Identifier, IdentifierVector, MetricStatus
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class EvidenceDecision(ContractModel):
    status: GateStatus
    required_gates: IdentifierVector
    diagnostic_only: bool
    serialization_id = "evidence_decision.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, GateStatus) or not isinstance(self.diagnostic_only, bool):
            raise ValidationError("evidence decision status and diagnostic flag are explicit")


@dataclass(frozen=True, slots=True)
class JainMetricResult(ContractModel):
    metric_id: Identifier
    value: FiniteFloat | None
    status: MetricStatus
    participant_count: int
    serialization_id = "jain_metric_result.v1"

    def __post_init__(self) -> None:
        require_identifier(self.metric_id, "metric_id")
        if not isinstance(self.status, MetricStatus) or self.participant_count < 0:
            raise ValidationError("Jain metric status/count invalid")
        value_bearing_statuses = {MetricStatus.VALID, MetricStatus.DEFINED}
        if self.status in value_bearing_statuses and self.value is None:
            raise ValidationError("valid Jain metric requires a value")
        if self.status not in value_bearing_statuses and self.value is not None:
            raise ValidationError("undefined Jain metric must not silently carry a value")


@dataclass(frozen=True, slots=True)
class MitigationResult(ContractModel):
    mode_id: Identifier
    transformation_id: Identifier
    rerun_status: GateStatus
    serialization_id = "mitigation.v1"

    def __post_init__(self) -> None:
        require_identifier(self.mode_id, "mode_id")
        require_identifier(self.transformation_id, "transformation_id")
        if not isinstance(self.rerun_status, GateStatus):
            raise ValidationError("mitigation rerun status must be registered")
