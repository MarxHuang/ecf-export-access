"""Network-shaped data entities; no parser and no power-flow calculation."""
from __future__ import annotations

from dataclasses import dataclass

from r4r.errors import ReferenceError, ValidationError
from r4r.models.base import ContractModel
from r4r.types import FiniteFloat, FloatMatrix, FloatVector, Identifier, IdentifierVector, NonNegativeFloat, ObjectReference


@dataclass(frozen=True, slots=True)
class Bus(ContractModel):
    bus_id: int
    base_kv: FiniteFloat
    serialization_id = "bus.v1"

    def __post_init__(self) -> None:
        if isinstance(self.bus_id, bool) or not isinstance(self.bus_id, int) or self.bus_id <= 0:
            raise ValidationError("bus_id must be a positive integer")
        if self.base_kv.value <= 0:
            raise ValidationError("base_kv must be positive")


@dataclass(frozen=True, slots=True)
class Branch(ContractModel):
    branch_id: int
    from_bus: ObjectReference
    to_bus: ObjectReference
    r: FiniteFloat
    x: FiniteFloat
    serialization_id = "branch.v1"

    def __post_init__(self) -> None:
        if isinstance(self.branch_id, bool) or not isinstance(self.branch_id, int) or self.branch_id <= 0:
            raise ValidationError("branch_id must be a positive integer")
        for name, ref in (("from_bus", self.from_bus), ("to_bus", self.to_bus)):
            if not isinstance(ref, ObjectReference) or ref.entity_id != "ENT001":
                raise ReferenceError(f"{name} must reference ENT001 Bus")
        if self.from_bus == self.to_bus:
            raise ValidationError("branch endpoints must be distinct")


@dataclass(frozen=True, slots=True)
class NetworkModel(ContractModel):
    buses: tuple[Bus, ...]
    branches: tuple[Branch, ...]
    base_mva: NonNegativeFloat
    serialization_id = "network.v1"

    def __post_init__(self) -> None:
        if not self.buses:
            raise ValidationError("network must contain at least one bus")
        if len({bus.bus_id for bus in self.buses}) != len(self.buses):
            raise ValidationError("duplicate bus IDs are forbidden")
        if len({branch.branch_id for branch in self.branches}) != len(self.branches):
            raise ValidationError("duplicate branch IDs are forbidden")
        bus_ids = {str(bus.bus_id) for bus in self.buses}
        for branch in self.branches:
            if branch.from_bus.object_id.value not in bus_ids or branch.to_bus.object_id.value not in bus_ids:
                raise ReferenceError("branch references a bus absent from the network")
        if self.base_mva.value <= 0:
            raise ValidationError("base_mva must be positive")


@dataclass(frozen=True, slots=True)
class ProxyModel(ContractModel):
    branch_sensitivity_matrix: FloatMatrix
    voltage_sensitivity_matrix: FloatMatrix
    branch_headroom_mw: FloatVector
    voltage_headroom_pu: FloatVector
    branch_orientation: tuple[Identifier, ...]
    serialization_id = "proxy_model.v1"

    def __post_init__(self) -> None:
        if len(self.branch_sensitivity_matrix.values) != len(self.branch_headroom_mw.values):
            raise ValidationError("branch sensitivity/headroom row alignment mismatch")
        if len(self.voltage_sensitivity_matrix.values) != len(self.voltage_headroom_pu.values):
            raise ValidationError("voltage sensitivity/headroom row alignment mismatch")
        if len(self.branch_orientation) != len(self.branch_headroom_mw.values):
            raise ValidationError("branch orientation/headroom alignment mismatch")
        if any(not isinstance(item, Identifier) for item in self.branch_orientation):
            raise ValidationError("branch orientation must use registered identifiers")
