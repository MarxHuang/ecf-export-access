"""Small structural checks shared by composite contract objects."""
from __future__ import annotations

from collections.abc import Mapping

from r4r.errors import ReferenceError, ValidationError
from r4r.types import Identifier, ObjectReference


def require_identifier(value: object, field: str) -> None:
    if not isinstance(value, Identifier):
        raise ValidationError(f"{field} must be Identifier")


def require_reference(value: object, field: str, entity_id: str | None = None) -> None:
    if not isinstance(value, ObjectReference):
        raise ReferenceError(f"{field} must be ObjectReference")
    if entity_id is not None and value.entity_id != entity_id:
        raise ReferenceError(f"{field} must reference {entity_id}")


def require_mapping(value: object, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be a mapping")
