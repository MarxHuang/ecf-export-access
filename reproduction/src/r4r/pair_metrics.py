"""Allocation-stage access-transfer metrics.

The functions here consume an explicit reference/reported pair and do not
solve, clip, rebalance, or infer a strategic motive.  SG, OL and UD follow the
registered equations.  Redistribution is intentionally a separately named
definition: without an explicit definition ID it remains ``UNRESOLVED`` rather
than silently being equated with OL.  The manuscript Fig. 6 caption defines
the registered diagnostic as ``R = 1/2 sum_i |Delta x_i|``; this is independent
of OL and is used by higher-level pair builders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, Identifier, IdentifierVector, Sha256
from r4r.types.scalars import FiniteFloat


PAPER_REDISTRIBUTION_DEFINITION_ID = "REDISTRIBUTION_HALF_L1_DELTA"
_REDISTRIBUTION_DEFINITIONS = {PAPER_REDISTRIBUTION_DEFINITION_ID}
_REDISTRIBUTION_STATUSES = {"DEFINED", "UNRESOLVED"}


def _finite_nonnegative(values: Sequence[float], field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{field} must be a numeric sequence")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{field} must be a numeric sequence") from exc
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValidationError(f"{field}[{index}] must be a finite real scalar")
        if float(value) < 0.0:
            raise ValidationError(f"{field}[{index}] must be nonnegative")
    return tuple(float(value) for value in result)


@dataclass(frozen=True, slots=True)
class AccessTransferMetricResult(ContractModel):
    participant_ids: IdentifierVector
    reporter_id: Identifier
    reference_hash: Sha256
    reported_hash: Sha256
    delta_mw: FloatVector
    sg_mw: FiniteFloat
    ol_minus_reporter_mw: FiniteFloat
    ud_mw: FiniteFloat
    delivery_cap_mw: FiniteFloat
    redistribution_mw: FiniteFloat | None
    redistribution_definition_id: Identifier | None
    redistribution_status: str
    status: str
    serialization_id = "access_transfer_metrics.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.reporter_id not in self.participant_ids.values:
            raise ValidationError("reporter_id must be present in participant_ids")
        for name, value in (("reference_hash", self.reference_hash), ("reported_hash", self.reported_hash)):
            if not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if len(self.delta_mw.values) != len(self.participant_ids.values):
            raise ValidationError("delta_mw must match participant IDs")
        for name in ("sg_mw", "ol_minus_reporter_mw", "ud_mw", "delivery_cap_mw"):
            if getattr(self, name).value < 0.0:
                raise ValidationError(f"{name} must be nonnegative")
        if self.redistribution_mw is not None and self.redistribution_mw.value < 0.0:
            raise ValidationError("redistribution_mw must be nonnegative")
        if self.redistribution_definition_id is not None and not isinstance(self.redistribution_definition_id, Identifier):
            raise ValidationError("redistribution_definition_id must be Identifier or null")
        if self.redistribution_status not in _REDISTRIBUTION_STATUSES:
            raise ValidationError("redistribution_status is not registered")
        if self.redistribution_status == "DEFINED" and self.redistribution_mw is None:
            raise ValidationError("defined redistribution requires a value")
        if self.redistribution_status == "UNRESOLVED" and self.redistribution_mw is not None:
            raise ValidationError("unresolved redistribution must not carry a value")
        if self.status != "DEFINED":
            raise ValidationError("access-transfer metric status is not registered")

    def to_json(self) -> dict[str, Any]:
        return {
            "participant_ids": self.participant_ids.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "reference_hash": self.reference_hash.to_json(),
            "reported_hash": self.reported_hash.to_json(),
            "delta_mw": self.delta_mw.to_json(),
            "sg_mw": self.sg_mw.to_json(),
            "ol_minus_reporter_mw": self.ol_minus_reporter_mw.to_json(),
            "ud_mw": self.ud_mw.to_json(),
            "delivery_cap_mw": self.delivery_cap_mw.to_json(),
            "redistribution_mw": self.redistribution_mw.to_json() if self.redistribution_mw else None,
            "redistribution_definition_id": self.redistribution_definition_id.to_json() if self.redistribution_definition_id else None,
            "redistribution_status": self.redistribution_status,
            "status": self.status,
        }

    @property
    def metric_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class CoalitionAccessTransferMetricResult(ContractModel):
    """Allocation-transfer metrics for a simultaneous reporting coalition.

    Positive and negative changes are retained on both sides of the coalition
    boundary.  This prevents an aggregate coalition gain from concealing a
    loss borne by one of its own members and prevents a net noncoalition value
    from concealing offsetting transfers among honest participants.
    """

    participant_ids: IdentifierVector
    coalition_ids: IdentifierVector
    reference_hash: Sha256
    reported_hash: Sha256
    delta_mw: FloatVector
    coalition_gain_mw: FiniteFloat
    coalition_loss_mw: FiniteFloat
    coalition_net_change_mw: FiniteFloat
    noncoalition_gain_mw: FiniteFloat
    noncoalition_loss_mw: FiniteFloat
    noncoalition_net_change_mw: FiniteFloat
    coalition_undeliverable_mw: FiniteFloat
    coalition_delivery_caps_mw: FloatVector
    redistribution_mw: FiniteFloat
    status: str
    serialization_id = "coalition_access_transfer_metrics.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.coalition_ids, IdentifierVector) or len(self.coalition_ids.values) < 2:
            raise ValidationError("coalition_ids must contain at least two participants")
        if any(item not in self.participant_ids.values for item in self.coalition_ids.values):
            raise ValidationError("coalition participant is absent from participant_ids")
        if len(self.delta_mw.values) != len(self.participant_ids.values):
            raise ValidationError("delta_mw must match participant IDs")
        if len(self.coalition_delivery_caps_mw.values) != len(self.coalition_ids.values):
            raise ValidationError("coalition delivery caps must match coalition IDs")
        if any(value < 0.0 for value in self.coalition_delivery_caps_mw.values):
            raise ValidationError("coalition delivery caps must be nonnegative")
        for name in (
            "coalition_gain_mw", "coalition_loss_mw", "noncoalition_gain_mw",
            "noncoalition_loss_mw", "coalition_undeliverable_mw", "redistribution_mw",
        ):
            if getattr(self, name).value < 0.0:
                raise ValidationError(f"{name} must be nonnegative")
        if self.status != "DEFINED":
            raise ValidationError("coalition access-transfer metric status is not registered")

    @property
    def metric_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def compute_access_transfer_metrics(
    participant_ids: IdentifierVector,
    reference_mw: Sequence[float],
    reported_mw: Sequence[float],
    reporter_id: Identifier,
    delivery_cap_mw: float,
    *,
    redistribution_definition_id: Identifier | None = None,
) -> AccessTransferMetricResult:
    """Compute SG, OL and UD for one explicit reference/reported pair."""

    if not isinstance(participant_ids, IdentifierVector) or not participant_ids.values:
        raise ValidationError("participant_ids must be a non-empty IdentifierVector")
    if reporter_id not in participant_ids.values:
        raise ValidationError("reporter_id must be present in participant_ids")
    reference = _finite_nonnegative(reference_mw, "reference_mw")
    reported = _finite_nonnegative(reported_mw, "reported_mw")
    if len(reference) != len(participant_ids.values) or len(reported) != len(participant_ids.values):
        raise ValidationError("reference and reported vectors must match participant IDs")
    if isinstance(delivery_cap_mw, bool) or not isinstance(delivery_cap_mw, (int, float)) or not math.isfinite(float(delivery_cap_mw)) or delivery_cap_mw < 0.0:
        raise ValidationError("delivery_cap_mw must be a nonnegative finite scalar")
    if redistribution_definition_id is not None:
        if not isinstance(redistribution_definition_id, Identifier) or redistribution_definition_id.value not in _REDISTRIBUTION_DEFINITIONS:
            raise ValidationError("redistribution definition is not registered")
    delta = tuple(after - before for before, after in zip(reference, reported))
    reporter_index = participant_ids.values.index(reporter_id)
    sg = max(delta[reporter_index], 0.0)
    ol = sum(max(-change, 0.0) for index, change in enumerate(delta) if index != reporter_index)
    ud = max(reported[reporter_index] - float(delivery_cap_mw), 0.0)
    reference_hash = Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": list(reference)}))
    reported_hash = Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": list(reported)}))
    defined = redistribution_definition_id is not None
    redistribution = 0.5 * sum(abs(change) for change in delta) if defined else None
    return AccessTransferMetricResult(
        participant_ids=participant_ids,
        reporter_id=reporter_id,
        reference_hash=reference_hash,
        reported_hash=reported_hash,
        delta_mw=FloatVector(delta),
        sg_mw=FiniteFloat(sg),
        ol_minus_reporter_mw=FiniteFloat(ol),
        ud_mw=FiniteFloat(ud),
        delivery_cap_mw=FiniteFloat(delivery_cap_mw),
        redistribution_mw=FiniteFloat(redistribution) if redistribution is not None else None,
        redistribution_definition_id=redistribution_definition_id,
        redistribution_status="DEFINED" if defined else "UNRESOLVED",
        status="DEFINED",
    )


def compute_coalition_access_transfer_metrics(
    participant_ids: IdentifierVector,
    reference_mw: Sequence[float],
    reported_mw: Sequence[float],
    coalition_ids: IdentifierVector,
    coalition_delivery_caps_mw: Sequence[float],
) -> CoalitionAccessTransferMetricResult:
    """Compute simultaneous multi-reporter transfer metrics for one pair."""

    if not isinstance(participant_ids, IdentifierVector) or not participant_ids.values:
        raise ValidationError("participant_ids must be a non-empty IdentifierVector")
    if not isinstance(coalition_ids, IdentifierVector) or len(coalition_ids.values) < 2:
        raise ValidationError("coalition_ids must contain at least two participants")
    if any(item not in participant_ids.values for item in coalition_ids.values):
        raise ValidationError("coalition participant is absent from participant_ids")
    reference = _finite_nonnegative(reference_mw, "reference_mw")
    reported = _finite_nonnegative(reported_mw, "reported_mw")
    delivery_caps = _finite_nonnegative(coalition_delivery_caps_mw, "coalition_delivery_caps_mw")
    if len(reference) != len(participant_ids.values) or len(reported) != len(participant_ids.values):
        raise ValidationError("reference and reported vectors must match participant IDs")
    if len(delivery_caps) != len(coalition_ids.values):
        raise ValidationError("coalition delivery caps must match coalition IDs")
    coalition_index = {participant_ids.values.index(item) for item in coalition_ids.values}
    delta = tuple(after - before for before, after in zip(reference, reported))
    coalition_gain = math.fsum(max(delta[index], 0.0) for index in coalition_index)
    coalition_loss = math.fsum(max(-delta[index], 0.0) for index in coalition_index)
    noncoalition_gain = math.fsum(max(change, 0.0) for index, change in enumerate(delta) if index not in coalition_index)
    noncoalition_loss = math.fsum(max(-change, 0.0) for index, change in enumerate(delta) if index not in coalition_index)
    coalition_undeliverable = math.fsum(
        max(reported[participant_ids.values.index(participant_id)] - cap, 0.0)
        for participant_id, cap in zip(coalition_ids.values, delivery_caps)
    )
    reference_hash = Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": list(reference)}))
    reported_hash = Sha256(canonical_hash({"participant_ids": participant_ids.to_json(), "values_mw": list(reported)}))
    return CoalitionAccessTransferMetricResult(
        participant_ids=participant_ids,
        coalition_ids=coalition_ids,
        reference_hash=reference_hash,
        reported_hash=reported_hash,
        delta_mw=FloatVector(delta),
        coalition_gain_mw=FiniteFloat(coalition_gain),
        coalition_loss_mw=FiniteFloat(coalition_loss),
        coalition_net_change_mw=FiniteFloat(math.fsum(delta[index] for index in coalition_index)),
        noncoalition_gain_mw=FiniteFloat(noncoalition_gain),
        noncoalition_loss_mw=FiniteFloat(noncoalition_loss),
        noncoalition_net_change_mw=FiniteFloat(math.fsum(change for index, change in enumerate(delta) if index not in coalition_index)),
        coalition_undeliverable_mw=FiniteFloat(coalition_undeliverable),
        coalition_delivery_caps_mw=FloatVector(delivery_caps),
        redistribution_mw=FiniteFloat(0.5 * math.fsum(abs(change) for change in delta)),
        status="DEFINED",
    )


__all__ = [
    "AccessTransferMetricResult",
    "CoalitionAccessTransferMetricResult",
    "PAPER_REDISTRIBUTION_DEFINITION_ID",
    "compute_access_transfer_metrics",
    "compute_coalition_access_transfer_metrics",
]
