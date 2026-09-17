"""AC-screen and solver cross-check result interfaces.

The primary AC numerical implementation lives in :mod:`r4r.ac_solver`; this
module remains the typed screening/cross-check interface and does not perform
screening or evidence promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier
from r4r.contracts.registry_loader import require_registered_failure_reasons
from r4r.types import (
    FloatVector,
    GateStatus,
    Identifier,
    IdentifierVector,
    QAssumption,
    ScreenSide,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class ACScreenResult(ContractModel):
    FAILURE_GATE_IDS: ClassVar[tuple[str, ...]] = ("G013",)
    side: ScreenSide
    calibration_q: QAssumption
    screening_q: QAssumption
    ac_solver_id: Identifier
    ac_solver_version: str
    converged: bool
    voltage_pass: bool
    branch_pass: bool
    side_pass: bool
    vmin_pu: FiniteFloat
    vmax_pu: FiniteFloat
    offending_bus_id: Identifier | None
    offending_voltage_pu: FiniteFloat | None
    voltage_limit_pu: FiniteFloat | None
    branch_from_bus_id: Identifier | None
    branch_to_bus_id: Identifier | None
    offending_branch_id: Identifier | None
    branch_orientation: Identifier | None
    failure_category: str | None
    baseline_flow_mw: FiniteFloat | None
    post_flow_mw: FiniteFloat | None
    incremental_flow_mw: FiniteFloat | None
    branch_budget_mw: FiniteFloat | None
    branch_violation_mw: FiniteFloat | None
    voltage_lower_budget_pu: FiniteFloat | None
    voltage_upper_budget_pu: FiniteFloat | None
    voltage_violation_pu: FiniteFloat | None
    q_profile_hash: Sha256
    failure_reasons: IdentifierVector = field(default_factory=lambda: IdentifierVector([]))
    serialization_id = "ac_screen.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.side, ScreenSide) or not isinstance(self.calibration_q, QAssumption) or not isinstance(self.screening_q, QAssumption):
            raise ValidationError("AC side and Q fields must use registered enums")
        require_identifier(self.ac_solver_id, "ac_solver_id")
        if not isinstance(self.ac_solver_version, str) or not self.ac_solver_version:
            raise ValidationError("ac_solver_version must be non-empty")
        if not all(isinstance(value, bool) for value in (self.converged, self.voltage_pass, self.branch_pass, self.side_pass)):
            raise ValidationError("AC pass flags must be explicit booleans")
        if self.failure_category is not None and self.failure_category not in {
            "AC_NOT_CONVERGED", "VOLTAGE_VIOLATION", "BRANCH_THERMAL_VIOLATION",
            "POWER_BALANCE_FAILURE", "RATE_POLICY_FAILURE", "SCREEN_INPUT_INVALID",
        }:
            raise ValidationError("failure_category is not registered")
        try:
            require_registered_failure_reasons(self.failure_reasons, gate_ids=self.FAILURE_GATE_IDS, field="ACScreenResult.failure_reasons")
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ACSolverCrosscheckResult(ContractModel):
    primary_solver_hash: Sha256
    independent_solver_hash: Sha256
    status: GateStatus
    # These context fields are optional for generic interface tests, but an
    # evidence-bound AC cross-check must populate all of them.  They prevent
    # accepting arbitrary members of a pair-wide solution-hash set.
    network_hash: Sha256 | None = None
    operating_point_hash: Sha256 | None = None
    participant_registry_hash: Sha256 | None = None
    q_spec_hash: Sha256 | None = None
    side: ScreenSide | None = None
    allocation_hash: Sha256 | None = None
    primary_input_identity_hash: Sha256 | None = None
    independent_input_identity_hash: Sha256 | None = None
    serialization_id = "ac_solver_crosscheck.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, GateStatus):
            raise ValidationError("status must be registered GateStatus")
        for name in (
            "primary_solver_hash", "independent_solver_hash",
            "network_hash", "operating_point_hash", "participant_registry_hash",
            "q_spec_hash", "allocation_hash", "primary_input_identity_hash",
            "independent_input_identity_hash",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or null")
        if self.side is not None and not isinstance(self.side, ScreenSide):
            raise ValidationError("AC cross-check side must be ScreenSide or null")
        context = (
            self.network_hash, self.operating_point_hash,
            self.participant_registry_hash, self.q_spec_hash,
            self.side, self.allocation_hash,
        )
        if any(value is not None for value in context) and not all(value is not None for value in context):
            raise ValidationError("AC cross-check context must be complete or null")
        inputs = (self.primary_input_identity_hash, self.independent_input_identity_hash)
        if any(value is not None for value in inputs) and not all(value is not None for value in inputs):
            raise ValidationError("AC cross-check input identities must be paired or null")
        if all(value is not None for value in inputs) and inputs[0] != inputs[1]:
            raise ValidationError("AC cross-check primary and independent inputs differ")

    @property
    def exact_context_bound(self) -> bool:
        return all(
            value is not None
            for value in (
                self.network_hash, self.operating_point_hash,
                self.participant_registry_hash, self.q_spec_hash,
                self.side, self.allocation_hash,
                self.primary_input_identity_hash,
                self.independent_input_identity_hash,
            )
        )

    def to_json(self) -> dict[str, object]:
        return {
            "serialization_id": self.serialization_id,
            "primary_solver_hash": self.primary_solver_hash.to_json(),
            "independent_solver_hash": self.independent_solver_hash.to_json(),
            "status": self.status.value,
            "network_hash": self.network_hash.to_json() if self.network_hash else None,
            "operating_point_hash": self.operating_point_hash.to_json() if self.operating_point_hash else None,
            "participant_registry_hash": self.participant_registry_hash.to_json() if self.participant_registry_hash else None,
            "q_spec_hash": self.q_spec_hash.to_json() if self.q_spec_hash else None,
            "side": self.side.value if self.side else None,
            "allocation_hash": self.allocation_hash.to_json() if self.allocation_hash else None,
            "primary_input_identity_hash": self.primary_input_identity_hash.to_json() if self.primary_input_identity_hash else None,
            "independent_input_identity_hash": self.independent_input_identity_hash.to_json() if self.independent_input_identity_hash else None,
        }
