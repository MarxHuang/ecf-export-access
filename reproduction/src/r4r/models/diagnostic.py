"""Typed diagnostic provenance objects shared by batch runners and registries."""
from __future__ import annotations

from dataclasses import dataclass

from r4r.errors import ValidationError
from r4r.serialization import canonical_hash
from r4r.types import FiniteFloat, Identifier, Sha256


@dataclass(frozen=True, slots=True)
class DiagnosticParameterIdentity:
    """Identity of the declared parameter case that produced one batch."""

    parameter_case_id: Identifier
    scenario_design_hash: Sha256
    gamma: FiniteFloat
    stress_multiplier: FiniteFloat
    headroom_ratio: FiniteFloat
    capacity_model_id: Identifier
    capacity_parameter_id: Identifier
    capacity_parameter_value: FiniteFloat
    gamma_source: str = "DECLARED_GAMMA"
    serialization_id = "diagnostic_parameter_identity.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_case_id, Identifier):
            raise ValidationError("parameter_case_id must be Identifier")
        if not isinstance(self.scenario_design_hash, Sha256):
            raise ValidationError("scenario_design_hash must be Sha256")
        if not isinstance(self.capacity_model_id, Identifier) or not isinstance(self.capacity_parameter_id, Identifier):
            raise ValidationError("capacity model and parameter IDs must be Identifier")
        for name, value in (
            ("gamma", self.gamma),
            ("stress_multiplier", self.stress_multiplier),
            ("headroom_ratio", self.headroom_ratio),
            ("capacity_parameter_value", self.capacity_parameter_value),
        ):
            if not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be FiniteFloat")
        if self.gamma.value < 1.0:
            raise ValidationError("parameter identity gamma must be >= 1")
        if self.gamma_source not in {"DECLARED_GAMMA", "LEGACY_DELTA"}:
            raise ValidationError("parameter identity gamma_source is not registered")
        if self.stress_multiplier.value <= 0.0:
            raise ValidationError("parameter identity stress_multiplier must be positive")
        if not 0.0 <= self.headroom_ratio.value <= 1.0:
            raise ValidationError("parameter identity headroom_ratio must be within [0,1]")
        if self.capacity_parameter_value.value < 0.0:
            raise ValidationError("parameter identity capacity parameter must be nonnegative")

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "parameter_case_id": self.parameter_case_id.to_json(),
            "scenario_design_hash": self.scenario_design_hash.to_json(),
            "gamma": self.gamma.to_json(),
            "gamma_source": self.gamma_source,
            "stress_multiplier": self.stress_multiplier.to_json(),
            "headroom_ratio": self.headroom_ratio.to_json(),
            "capacity_model_id": self.capacity_model_id.to_json(),
            "capacity_parameter_id": self.capacity_parameter_id.to_json(),
            "capacity_parameter_value": self.capacity_parameter_value.to_json(),
        }

    @property
    def identity_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

