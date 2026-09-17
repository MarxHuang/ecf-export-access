"""Immutable finite vectors and matrices with explicit shape checks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from r4r.errors import ValidationError
from .identifiers import Identifier, ObjectReference


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError("collection members must be finite real scalars")
    return float(value)


@dataclass(frozen=True, slots=True)
class FloatVector:
    values: tuple[float, ...]

    def __init__(self, values: Iterable[float]) -> None:
        object.__setattr__(self, "values", tuple(_finite(x) for x in values))

    def to_json(self) -> list[float]:
        return list(self.values)


@dataclass(frozen=True, slots=True)
class FloatMatrix:
    values: tuple[tuple[float, ...], ...]

    def __init__(self, values: Iterable[Iterable[float]]) -> None:
        rows = tuple(tuple(_finite(x) for x in row) for row in values)
        width = len(rows[0]) if rows else 0
        if any(len(row) != width for row in rows):
            raise ValidationError("matrix rows must be rectangular")
        object.__setattr__(self, "values", rows)

    def to_json(self) -> list[list[float]]:
        return [list(row) for row in self.values]


@dataclass(frozen=True, slots=True)
class IdentifierVector:
    values: tuple[Identifier, ...]

    def __init__(self, values: Iterable[Identifier]) -> None:
        result = tuple(values)
        if any(not isinstance(item, Identifier) for item in result):
            raise ValidationError("identifier vector members must be Identifier values")
        if len(set(result)) != len(result):
            raise ValidationError("identifier vector members must be unique")
        object.__setattr__(self, "values", result)

    def to_json(self) -> list[str]:
        return [item.to_json() for item in self.values]


@dataclass(frozen=True, slots=True)
class ObjectReferenceVector:
    values: tuple[ObjectReference, ...]

    def __init__(self, values: Iterable[ObjectReference]) -> None:
        result = tuple(values)
        if any(not isinstance(item, ObjectReference) for item in result):
            raise ValidationError("object-reference vector members must be ObjectReference values")
        object.__setattr__(self, "values", result)

    def to_json(self) -> list[dict[str, str]]:
        return [item.to_json() for item in self.values]
