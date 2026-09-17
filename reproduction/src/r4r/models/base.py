"""Common immutable model protocol and stable object hashing."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, ClassVar

from r4r.errors import ValidationError
from r4r.serialization import canonical_hash, encode


class ContractModel:
    serialization_id: ClassVar[str]

    def to_json(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise ValidationError("contract models must be dataclasses")
        return {field.name: encode(getattr(self, field.name)) for field in fields(self)}

    def object_hash(self) -> str:
        return model_hash(self)


def validate_model(value: Any) -> None:
    if not isinstance(value, ContractModel) or not is_dataclass(value):
        raise ValidationError("value is not a registered contract model")
    value.to_json()


def model_hash(value: ContractModel) -> str:
    validate_model(value)
    return canonical_hash({"serialization_id": value.serialization_id, "fields": value.to_json()})
