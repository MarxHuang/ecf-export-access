"""Finite scalar values without silent clipping."""
from __future__ import annotations

import math
from dataclasses import dataclass

from r4r.errors import ValidationError


@dataclass(frozen=True, slots=True)
class FiniteFloat:
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            raise ValidationError("value must be a finite real scalar")
        object.__setattr__(self, "value", float(self.value))

    def to_json(self) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class NonNegativeFloat(FiniteFloat):
    def __post_init__(self) -> None:
        FiniteFloat.__post_init__(self)
        if self.value < 0:
            raise ValidationError("value must be nonnegative")
