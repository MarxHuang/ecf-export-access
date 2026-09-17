"""Explicit diagnostic realization of the frozen M0/M1/M125 margin modes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.proxy_evaluator import ProxyBudget
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, FiniteFloat, Identifier, MarginMode, Sha256


_MODE_MULTIPLIERS = {
    MarginMode.M0: 0.0,
    MarginMode.M1: 1.0,
    MarginMode.M125: 1.25,
}


@dataclass(frozen=True, slots=True)
class DiagnosticMarginModeBundle(ContractModel):
    """One mode's branch and voltage budgets, retained without promotion."""

    mode: MarginMode
    multiplier: FiniteFloat
    branch_physical_budget_mw: FloatVector
    branch_calibration_margin_mw: FloatVector
    voltage_physical_budget_pu: FloatVector
    voltage_calibration_margin_pu: FloatVector
    branch_budget: ProxyBudget
    voltage_budget: ProxyBudget
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_margin_mode_bundle.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MarginMode):
            raise ValidationError("mode must be M0, M1 or M125")
        if not isinstance(self.multiplier, FiniteFloat):
            raise ValidationError("multiplier must be FiniteFloat")
        expected_multiplier = _MODE_MULTIPLIERS[self.mode]
        if self.multiplier.value != expected_multiplier:
            raise ValidationError("margin multiplier does not match registered mode")
        for name, value in (
            ("branch_physical_budget_mw", self.branch_physical_budget_mw),
            ("branch_calibration_margin_mw", self.branch_calibration_margin_mw),
            ("voltage_physical_budget_pu", self.voltage_physical_budget_pu),
            ("voltage_calibration_margin_pu", self.voltage_calibration_margin_pu),
        ):
            if not isinstance(value, FloatVector) or not value.values:
                raise ValidationError(f"{name} must be a non-empty FloatVector")
        if len(self.branch_physical_budget_mw.values) != len(self.branch_calibration_margin_mw.values):
            raise ValidationError("branch physical budget and margin must align")
        if len(self.voltage_physical_budget_pu.values) != len(self.voltage_calibration_margin_pu.values):
            raise ValidationError("voltage physical budget and margin must align")
        if any(value < 0.0 for value in (
            *self.branch_physical_budget_mw.values,
            *self.branch_calibration_margin_mw.values,
            *self.voltage_physical_budget_pu.values,
            *self.voltage_calibration_margin_pu.values,
        )):
            raise ValidationError("physical budgets and calibration margins must be nonnegative")
        if not isinstance(self.branch_budget, ProxyBudget) or not isinstance(self.voltage_budget, ProxyBudget):
            raise ValidationError("branch and voltage budgets must be ProxyBudget values")
        if self.branch_budget.margin_mode_id != Identifier(self.mode.value) or self.voltage_budget.margin_mode_id != Identifier(self.mode.value):
            raise ValidationError("ProxyBudget margin mode IDs must match the bundle mode")
        expected_branch = tuple(
            physical - expected_multiplier * margin
            for physical, margin in zip(self.branch_physical_budget_mw.values, self.branch_calibration_margin_mw.values)
        )
        expected_voltage = tuple(
            physical - expected_multiplier * margin
            for physical, margin in zip(self.voltage_physical_budget_pu.values, self.voltage_calibration_margin_pu.values)
        )
        if self.branch_budget.tightened_budget.values != expected_branch:
            raise ValidationError("branch margin must be subtracted exactly once")
        if self.voltage_budget.tightened_budget.values != expected_voltage:
            raise ValidationError("voltage margin must be subtracted exactly once")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("margin mode bundles cannot be promoted")

    @property
    def budget_status(self) -> str:
        return "VALID" if self.branch_budget.status == "VALID" and self.voltage_budget.status == "VALID" else "INVALID_BUDGET"

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "multiplier": self.multiplier.to_json(),
            "branch_physical_budget_mw": self.branch_physical_budget_mw.to_json(),
            "branch_calibration_margin_mw": self.branch_calibration_margin_mw.to_json(),
            "voltage_physical_budget_pu": self.voltage_physical_budget_pu.to_json(),
            "voltage_calibration_margin_pu": self.voltage_calibration_margin_pu.to_json(),
            "branch_budget": self.branch_budget.to_json(),
            "voltage_budget": self.voltage_budget.to_json(),
            "budget_status": self.budget_status,
            "status": self.status,
        }

    @property
    def bundle_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


@dataclass(frozen=True, slots=True)
class DiagnosticMarginModeSet(ContractModel):
    """Complete M0/M1/M125 diagnostic set with shared physical inputs."""

    bundles: tuple[DiagnosticMarginModeBundle, ...]
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "diagnostic_margin_mode_set.v1"

    def __post_init__(self) -> None:
        if any(not isinstance(bundle, DiagnosticMarginModeBundle) for bundle in self.bundles):
            raise ValidationError("margin mode set must contain DiagnosticMarginModeBundle values")
        if tuple(bundle.mode for bundle in self.bundles) != (MarginMode.M0, MarginMode.M1, MarginMode.M125):
            raise ValidationError("margin mode set must be ordered M0, M1, M125")
        if any(bundle.status != "DIAGNOSTIC_ONLY" for bundle in self.bundles):
            raise ValidationError("margin mode set contains promotable data")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("margin mode sets cannot be promoted")
        first = self.bundles[0]
        for bundle in self.bundles[1:]:
            if bundle.branch_physical_budget_mw != first.branch_physical_budget_mw or bundle.voltage_physical_budget_pu != first.voltage_physical_budget_pu:
                raise ValidationError("margin modes must share physical budgets")
            if bundle.branch_calibration_margin_mw != first.branch_calibration_margin_mw or bundle.voltage_calibration_margin_pu != first.voltage_calibration_margin_pu:
                raise ValidationError("margin modes must share calibration margins")

    def for_mode(self, mode: MarginMode) -> DiagnosticMarginModeBundle:
        if not isinstance(mode, MarginMode):
            raise ValidationError("mode must be a registered MarginMode")
        return self.bundles[(MarginMode.M0, MarginMode.M1, MarginMode.M125).index(mode)]

    def to_json(self) -> dict[str, Any]:
        return {"bundles": [bundle.to_json() for bundle in self.bundles], "status": self.status}

    @property
    def set_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def build_diagnostic_margin_mode_bundle(
    mode: MarginMode,
    *,
    branch_physical_budget_mw: Sequence[float],
    branch_calibration_margin_mw: Sequence[float],
    voltage_physical_budget_pu: Sequence[float],
    voltage_calibration_margin_pu: Sequence[float],
) -> DiagnosticMarginModeBundle:
    """Materialize one explicit frozen margin mode without clipping."""

    if not isinstance(mode, MarginMode):
        raise ValidationError("mode must be a registered MarginMode")
    branch_physical = FloatVector(branch_physical_budget_mw)
    branch_margin = FloatVector(branch_calibration_margin_mw)
    voltage_physical = FloatVector(voltage_physical_budget_pu)
    voltage_margin = FloatVector(voltage_calibration_margin_pu)
    multiplier = _MODE_MULTIPLIERS[mode]
    branch_tightened = FloatVector(
        physical - multiplier * margin
        for physical, margin in zip(branch_physical.values, branch_margin.values)
    )
    voltage_tightened = FloatVector(
        physical - multiplier * margin
        for physical, margin in zip(voltage_physical.values, voltage_margin.values)
    )
    return DiagnosticMarginModeBundle(
        mode=mode,
        multiplier=FiniteFloat(multiplier),
        branch_physical_budget_mw=branch_physical,
        branch_calibration_margin_mw=branch_margin,
        voltage_physical_budget_pu=voltage_physical,
        voltage_calibration_margin_pu=voltage_margin,
        branch_budget=ProxyBudget(
            physical_budget=branch_physical,
            calibration_margin=branch_margin,
            margin_multiplier=FiniteFloat(multiplier),
            tightened_budget=branch_tightened,
            unit="MW",
            margin_mode_id=Identifier(mode.value),
        ),
        voltage_budget=ProxyBudget(
            physical_budget=voltage_physical,
            calibration_margin=voltage_margin,
            margin_multiplier=FiniteFloat(multiplier),
            tightened_budget=voltage_tightened,
            unit="pu",
            margin_mode_id=Identifier(mode.value),
        ),
    )


def build_diagnostic_margin_mode_set(
    *,
    branch_physical_budget_mw: Sequence[float],
    branch_calibration_margin_mw: Sequence[float],
    voltage_physical_budget_pu: Sequence[float],
    voltage_calibration_margin_pu: Sequence[float],
) -> DiagnosticMarginModeSet:
    """Build the complete ordered M0/M1/M125 diagnostic set."""

    args = {
        "branch_physical_budget_mw": branch_physical_budget_mw,
        "branch_calibration_margin_mw": branch_calibration_margin_mw,
        "voltage_physical_budget_pu": voltage_physical_budget_pu,
        "voltage_calibration_margin_pu": voltage_calibration_margin_pu,
    }
    return DiagnosticMarginModeSet(tuple(
        build_diagnostic_margin_mode_bundle(mode, **args)
        for mode in (MarginMode.M0, MarginMode.M1, MarginMode.M125)
    ))


__all__ = [
    "DiagnosticMarginModeBundle",
    "DiagnosticMarginModeSet",
    "build_diagnostic_margin_mode_bundle",
    "build_diagnostic_margin_mode_set",
]
