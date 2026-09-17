"""Identifier and reference values; hashes and references are not units."""
from __future__ import annotations

import re
from dataclasses import dataclass

from r4r.errors import ReferenceError, ValidationError

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ENTITY = re.compile(r"^ENT[0-9]{3}$")


@dataclass(frozen=True, slots=True)
class Identifier:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value or "\x00" in self.value or "/" in self.value or "\\" in self.value:
            raise ValidationError("identifier must be a non-empty package-relative token")

    def __str__(self) -> str:
        return self.value

    def to_json(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Sha256:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _SHA256.fullmatch(self.value):
            raise ValidationError("sha256 must be exactly 64 hexadecimal characters")

    def __str__(self) -> str:
        return self.value.upper()

    def to_json(self) -> str:
        return str(self)


@dataclass(frozen=True, slots=True)
class ObjectReference:
    entity_id: str
    object_id: Identifier

    def __post_init__(self) -> None:
        if not _ENTITY.fullmatch(self.entity_id):
            raise ReferenceError("entity_id must match registered ENTxxx form")
        if not isinstance(self.object_id, Identifier):
            raise ReferenceError("object_id must be an Identifier")

    def to_json(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "object_id": self.object_id.to_json()}
