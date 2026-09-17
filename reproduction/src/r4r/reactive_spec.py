"""Parameterised Q0/Q95 specification with explicit-sign fail-closed gates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from r4r.errors import UnfrozenScientificDefinitionError, ValidationError
from r4r.types import FloatVector, Identifier, QAssumption


_STATUSES = {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}
_SIGNS = {"POSITIVE_BUS_INJECTION", "NEGATIVE_BUS_INJECTION", "UNSPECIFIED"}


@dataclass(frozen=True, slots=True)
class ReactiveInjectionSpecification:
    q_mode_id: QAssumption
    power_factor: float | None
    magnitude_rule_id: Identifier
    q_sign_convention: str
    positive_p_definition: str
    positive_q_definition: str
    capability_limit_policy: Identifier
    scientific_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.q_mode_id, QAssumption):
            raise ValidationError("q_mode_id must be a registered Q token")
        if not isinstance(self.magnitude_rule_id, Identifier) or not isinstance(self.capability_limit_policy, Identifier):
            raise ValidationError("Q rule and capability policy IDs must be identifiers")
        if self.q_sign_convention not in _SIGNS:
            raise ValidationError("q_sign_convention is not registered")
        if self.scientific_status not in _STATUSES:
            raise ValidationError("scientific_status is not registered")
        if not self.positive_p_definition or not self.positive_q_definition:
            raise ValidationError("positive P/Q definitions are required")
        if self.q_mode_id is QAssumption.Q95:
            if self.power_factor is None or not math.isfinite(self.power_factor) or not 0 < self.power_factor <= 1:
                raise ValidationError("Q95 power_factor must satisfy 0 < pf <= 1")
        elif self.power_factor is not None and (not math.isfinite(self.power_factor) or not 0 < self.power_factor <= 1):
            raise ValidationError("power_factor must satisfy 0 < pf <= 1 when supplied")


def q95_magnitude(allocation_mw: Sequence[float], power_factor: float) -> FloatVector:
    """Return |ΔQ| for fixed power factor; no sign or clipping is selected."""
    if not math.isfinite(power_factor) or not 0 < power_factor <= 1:
        raise ValidationError("power_factor must satisfy 0 < pf <= 1")
    values = FloatVector(allocation_mw)
    if any(value < 0 for value in values.values):
        raise ValidationError("export allocation must be nonnegative")
    factor = math.tan(math.acos(power_factor))
    return FloatVector(abs(value) * factor for value in values.values)


def build_participant_q(
    allocation_mw: Sequence[float],
    specification: ReactiveInjectionSpecification,
    *,
    formal_execution: bool = True,
) -> FloatVector:
    """Build participant-Q only when the sign is explicit.

    Candidate specifications may be used for diagnostic tests by setting
    ``formal_execution=False``. Unresolved or candidate definitions cannot
    enter formal matched-Q/primary execution.
    """
    if formal_execution and specification.scientific_status != "AUTHOR_FROZEN":
        raise UnfrozenScientificDefinitionError(
            "formal reactive injection requires AUTHOR_FROZEN specification"
        )
    if specification.q_mode_id is QAssumption.Q0:
        if specification.q_sign_convention != "UNSPECIFIED":
            raise ValidationError("Q0 participant-Q specification must use UNSPECIFIED sign")
        return FloatVector([0.0] * len(allocation_mw))
    magnitude = q95_magnitude(allocation_mw, specification.power_factor)  # type: ignore[arg-type]
    if specification.q_sign_convention == "UNSPECIFIED":
        raise UnfrozenScientificDefinitionError("Q95 sign convention is unresolved")
    sign = 1.0 if specification.q_sign_convention == "POSITIVE_BUS_INJECTION" else -1.0
    return FloatVector(sign * value for value in magnitude.values)
