"""Structured errors for typed scientific-contract values."""


class ContractError(ValueError):
    """Base error for invalid contract values."""


class ValidationError(ContractError):
    """A value violates a registered shape, unit or domain rule."""


class ReferenceError(ContractError):
    """An object reference is malformed or points to the wrong entity type."""


class SerializationError(ContractError):
    """A value cannot be represented by canonical-json-v1."""


class UnfrozenScientificDefinitionError(ContractError):
    """A formal execution requires a scientific definition that is unresolved."""
