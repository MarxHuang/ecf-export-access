"""Full-network AC screening for diagnostic allocation profiles.

The legacy screen in :mod:`tools.ac_screen_diagnostics` intentionally keeps a
single representative branch and scalar voltage limits.  This module adds the
next diagnostic slice without changing that interface: every parsed branch is
checked at both MATPOWER ends and every parsed bus is checked against an
explicit lower/upper limit.  Limits are caller-supplied scenario inputs; the
MATPOWER ``RATE_A=0`` values in case141 are not silently interpreted as native
thermal ratings.

No result from this module is physical or manuscript evidence.  In particular,
the returned pair cannot satisfy T4/T5 or promote a proxy allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from r4r.ac_solver import ACPowerFlowSolution, solve_ac_power_flow
from r4r.allocation_pipeline import AllocationBindingProtocol, RuleAllocationBindingProtocol
from r4r.errors import ValidationError
from r4r.injection_adapter import build_net_injections_from_operating_point_spec
from r4r.models import OperatingPoint, model_hash
from r4r.models.base import ContractModel
from r4r.network_parser import ParsedMatpowerCase
from r4r.participant_registry import ParticipantSelection
from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.serialization import canonical_hash
from r4r.types import AllocationSide, FloatVector, Identifier, IdentifierVector, QAssumption, ScreenSide, Sha256
from r4r.types.scalars import FiniteFloat
from tools.diagnostic_allocation_pair import DiagnosticAllocationPairResult
from tools.ac_proxy_diagnostics import reactive_spec_hash


AC_PHYSICAL_LIMIT_FAILURE_NAMESPACE = "AC_PHYSICAL_LIMIT"


def _positive_finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ACNetworkScreenLimits(ContractModel):
    """Explicit full-network limits used by one diagnostic screen."""

    branch_budget_mw: FloatVector
    voltage_lower_limit_pu: FloatVector
    voltage_upper_limit_pu: FloatVector
    branch_tolerance_mw: FiniteFloat = FiniteFloat(0.0)
    voltage_tolerance_pu: FiniteFloat = FiniteFloat(0.0)
    branch_observable_mode: str = "BOTH_END_MAX_INCREMENT"
    branch_limit_source: Identifier = Identifier("SCENARIO_INCREMENTAL_ACTIVE_MW")
    voltage_limit_source: Identifier = Identifier("SCENARIO_BUS_VOLTAGE_BOUNDS")
    serialization_id = "ac_network_screen_limits.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.branch_budget_mw, FloatVector) or not self.branch_budget_mw.values:
            raise ValidationError("branch_budget_mw must be a non-empty FloatVector")
        if not isinstance(self.voltage_lower_limit_pu, FloatVector) or not self.voltage_lower_limit_pu.values:
            raise ValidationError("voltage_lower_limit_pu must be a non-empty FloatVector")
        if not isinstance(self.voltage_upper_limit_pu, FloatVector):
            raise ValidationError("voltage_upper_limit_pu must be a FloatVector")
        if len(self.voltage_lower_limit_pu.values) != len(self.voltage_upper_limit_pu.values):
            raise ValidationError("voltage lower and upper limits must align")
        if any(value < 0.0 for value in self.branch_budget_mw.values):
            raise ValidationError("branch budgets must be nonnegative")
        if any(lower > upper for lower, upper in zip(self.voltage_lower_limit_pu.values, self.voltage_upper_limit_pu.values)):
            raise ValidationError("each voltage lower limit must not exceed its upper limit")
        if self.branch_tolerance_mw.value < 0.0 or self.voltage_tolerance_pu.value < 0.0:
            raise ValidationError("screen tolerances must be nonnegative")
        if self.branch_observable_mode not in {
            "FROM_END_INCREMENT", "TO_END_INCREMENT", "BOTH_END_MAX_INCREMENT",
            "FROM_END_SIGNED_INCREMENT", "TO_END_SIGNED_INCREMENT",
        }:
            raise ValidationError("branch_observable_mode is not registered")
        if not isinstance(self.branch_limit_source, Identifier) or not isinstance(self.voltage_limit_source, Identifier):
            raise ValidationError("limit sources must be Identifier values")

    def to_json(self) -> dict[str, Any]:
        return {
            "branch_budget_mw": self.branch_budget_mw.to_json(),
            "voltage_lower_limit_pu": self.voltage_lower_limit_pu.to_json(),
            "voltage_upper_limit_pu": self.voltage_upper_limit_pu.to_json(),
            "branch_tolerance_mw": self.branch_tolerance_mw.to_json(),
            "voltage_tolerance_pu": self.voltage_tolerance_pu.to_json(),
            "branch_observable_mode": self.branch_observable_mode,
            "branch_limit_source": self.branch_limit_source.to_json(),
            "voltage_limit_source": self.voltage_limit_source.to_json(),
        }

    @property
    def limits_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class ACBranchScreenDiagnostics(ContractModel):
    """Per-branch, both-end active-flow diagnostics."""

    branch_ids: IdentifierVector
    from_bus_ids: tuple[Identifier, ...]
    to_bus_ids: tuple[Identifier, ...]
    baseline_from_flow_mw: FloatVector
    post_from_flow_mw: FloatVector
    from_increment_mw: FloatVector
    baseline_to_flow_mw: FloatVector
    post_to_flow_mw: FloatVector
    to_increment_mw: FloatVector
    max_abs_increment_mw: FloatVector
    screened_increment_mw: FloatVector
    branch_observable_mode: str
    budget_mw: FloatVector
    violation_mw: FloatVector
    pass_flags: tuple[bool, ...]
    offending_branch_ids: IdentifierVector
    serialization_id = "ac_branch_screen_diagnostics.v1"

    def __post_init__(self) -> None:
        n = len(self.branch_ids.values)
        if n == 0:
            raise ValidationError("branch diagnostics must not be empty")
        if len(self.from_bus_ids) != n or len(self.to_bus_ids) != n:
            raise ValidationError("branch endpoint IDs must align with branch IDs")
        if any(not isinstance(item, Identifier) for item in self.from_bus_ids + self.to_bus_ids):
            raise ValidationError("branch endpoint IDs must be Identifier values")
        for name in (
            "baseline_from_flow_mw", "post_from_flow_mw", "from_increment_mw",
            "baseline_to_flow_mw", "post_to_flow_mw", "to_increment_mw",
            "max_abs_increment_mw", "budget_mw", "violation_mw",
        ):
            value = getattr(self, name)
            if not isinstance(value, FloatVector) or len(value.values) != n:
                raise ValidationError(f"{name} must align with branch IDs")
        if not isinstance(self.screened_increment_mw, FloatVector) or len(self.screened_increment_mw.values) != n:
            raise ValidationError("screened_increment_mw must align with branch IDs")
        if self.branch_observable_mode not in {
            "FROM_END_INCREMENT", "TO_END_INCREMENT", "BOTH_END_MAX_INCREMENT",
            "FROM_END_SIGNED_INCREMENT", "TO_END_SIGNED_INCREMENT",
        }:
            raise ValidationError("branch_observable_mode is not registered")
        if len(self.pass_flags) != n or any(not isinstance(value, bool) for value in self.pass_flags):
            raise ValidationError("branch pass flags must align with branch IDs")
        if any(item not in self.branch_ids.values for item in self.offending_branch_ids.values):
            raise ValidationError("offending branch IDs must belong to branch IDs")

    def to_json(self) -> dict[str, Any]:
        return {
            "branch_ids": self.branch_ids.to_json(),
            "from_bus_ids": [item.to_json() for item in self.from_bus_ids],
            "to_bus_ids": [item.to_json() for item in self.to_bus_ids],
            "baseline_from_flow_mw": self.baseline_from_flow_mw.to_json(),
            "post_from_flow_mw": self.post_from_flow_mw.to_json(),
            "from_increment_mw": self.from_increment_mw.to_json(),
            "baseline_to_flow_mw": self.baseline_to_flow_mw.to_json(),
            "post_to_flow_mw": self.post_to_flow_mw.to_json(),
            "to_increment_mw": self.to_increment_mw.to_json(),
            "max_abs_increment_mw": self.max_abs_increment_mw.to_json(),
            "screened_increment_mw": self.screened_increment_mw.to_json(),
            "branch_observable_mode": self.branch_observable_mode,
            "budget_mw": self.budget_mw.to_json(),
            "violation_mw": self.violation_mw.to_json(),
            "pass_flags": list(self.pass_flags),
            "offending_branch_ids": self.offending_branch_ids.to_json(),
        }


@dataclass(frozen=True, slots=True)
class ACBusVoltageScreenDiagnostics(ContractModel):
    """Per-bus voltage diagnostics with separate lower/upper slack."""

    bus_ids: IdentifierVector
    voltage_pu: FloatVector
    lower_limit_pu: FloatVector
    upper_limit_pu: FloatVector
    lower_slack_pu: FloatVector
    upper_slack_pu: FloatVector
    violation_pu: FloatVector
    pass_flags: tuple[bool, ...]
    offending_bus_ids: IdentifierVector
    serialization_id = "ac_bus_voltage_screen_diagnostics.v1"

    def __post_init__(self) -> None:
        n = len(self.bus_ids.values)
        if n == 0:
            raise ValidationError("bus diagnostics must not be empty")
        for name in (
            "voltage_pu", "lower_limit_pu", "upper_limit_pu", "lower_slack_pu",
            "upper_slack_pu", "violation_pu",
        ):
            value = getattr(self, name)
            if not isinstance(value, FloatVector) or len(value.values) != n:
                raise ValidationError(f"{name} must align with bus IDs")
        if len(self.pass_flags) != n or any(not isinstance(value, bool) for value in self.pass_flags):
            raise ValidationError("bus pass flags must align with bus IDs")
        if any(item not in self.bus_ids.values for item in self.offending_bus_ids.values):
            raise ValidationError("offending bus IDs must belong to bus IDs")

    def to_json(self) -> dict[str, Any]:
        return {
            "bus_ids": self.bus_ids.to_json(),
            "voltage_pu": self.voltage_pu.to_json(),
            "lower_limit_pu": self.lower_limit_pu.to_json(),
            "upper_limit_pu": self.upper_limit_pu.to_json(),
            "lower_slack_pu": self.lower_slack_pu.to_json(),
            "upper_slack_pu": self.upper_slack_pu.to_json(),
            "violation_pu": self.violation_pu.to_json(),
            "pass_flags": list(self.pass_flags),
            "offending_bus_ids": self.offending_bus_ids.to_json(),
        }


@dataclass(frozen=True, slots=True)
class ACNetworkScreenResult(ContractModel):
    """One side of a full-network AC screen, always diagnostic-only."""

    side: ScreenSide
    calibration_q: QAssumption
    screening_q: QAssumption
    ac_solver_id: Identifier
    ac_solver_version: str
    converged: bool
    voltage_pass: bool
    branch_pass: bool
    side_pass: bool
    branch: ACBranchScreenDiagnostics
    voltage: ACBusVoltageScreenDiagnostics
    q_profile_hash: Sha256
    failure_reasons: IdentifierVector
    baseline_solution_hash: Sha256 | None = None
    solution_hash: Sha256 | None = None
    allocation_hash: Sha256 | None = None
    participant_registry_hash: Sha256 | None = None
    network_hash: Sha256 | None = None
    operating_point_hash: Sha256 | None = None
    q_spec_hash: Sha256 | None = None
    limits_hash: Sha256 | None = None
    quantities_valid: bool = True
    status: str = "DIAGNOSTIC_ONLY"
    failure_namespace: str = AC_PHYSICAL_LIMIT_FAILURE_NAMESPACE
    serialization_id = "ac_network_screen.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.side, ScreenSide) or not isinstance(self.calibration_q, QAssumption) or not isinstance(self.screening_q, QAssumption):
            raise ValidationError("screen side and Q fields must use registered enums")
        if not isinstance(self.ac_solver_id, Identifier) or not self.ac_solver_version:
            raise ValidationError("AC solver identity is required")
        if not all(isinstance(value, bool) for value in (self.converged, self.voltage_pass, self.branch_pass, self.side_pass)):
            raise ValidationError("AC screen flags must be booleans")
        if not isinstance(self.quantities_valid, bool):
            raise ValidationError("quantities_valid must be a boolean")
        if self.quantities_valid != self.converged:
            raise ValidationError("quantities_valid must equal convergence status")
        if self.side_pass != (self.converged and self.voltage_pass and self.branch_pass):
            raise ValidationError("side_pass must equal convergence and family pass flags")
        if self.branch_pass != all(self.branch.pass_flags):
            raise ValidationError("branch_pass must equal the conjunction of branch pass flags")
        if self.voltage_pass != all(self.voltage.pass_flags):
            raise ValidationError("voltage_pass must equal the conjunction of voltage pass flags")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("full-network AC screens cannot be promoted")
        if self.failure_namespace != AC_PHYSICAL_LIMIT_FAILURE_NAMESPACE:
            raise ValidationError("AC screen failure namespace must identify physical limits")
        if not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure_reasons must be IdentifierVector")
        for name in (
            "baseline_solution_hash", "solution_hash", "allocation_hash",
            "participant_registry_hash", "network_hash", "operating_point_hash",
            "q_spec_hash", "limits_hash",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Sha256):
                raise ValidationError(f"{name} must be Sha256 or null")

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "calibration_q": self.calibration_q.value,
            "screening_q": self.screening_q.value,
            "ac_solver_id": self.ac_solver_id.to_json(),
            "ac_solver_version": self.ac_solver_version,
            "converged": self.converged,
            "voltage_pass": self.voltage_pass,
            "branch_pass": self.branch_pass,
            "side_pass": self.side_pass,
            "branch": self.branch.to_json(),
            "voltage": self.voltage.to_json(),
            "q_profile_hash": self.q_profile_hash.to_json(),
            "failure_reasons": self.failure_reasons.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json() if self.baseline_solution_hash else None,
            "solution_hash": self.solution_hash.to_json() if self.solution_hash else None,
            "allocation_hash": self.allocation_hash.to_json() if self.allocation_hash else None,
            "participant_registry_hash": self.participant_registry_hash.to_json() if self.participant_registry_hash else None,
            "network_hash": self.network_hash.to_json() if self.network_hash else None,
            "operating_point_hash": self.operating_point_hash.to_json() if self.operating_point_hash else None,
            "q_spec_hash": self.q_spec_hash.to_json() if self.q_spec_hash else None,
            "limits_hash": self.limits_hash.to_json() if self.limits_hash else None,
            "quantities_valid": self.quantities_valid,
            "status": self.status,
            "failure_namespace": self.failure_namespace,
        }

    @property
    def screen_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


@dataclass(frozen=True, slots=True)
class ACNetworkScreenPairResult(ContractModel):
    """Reference/reported full-network pair with shared baseline provenance."""

    reference: ACNetworkScreenResult
    reported: ACNetworkScreenResult
    baseline_solution_hash: Sha256
    reference_solution_hash: Sha256
    reported_solution_hash: Sha256
    pair_hash: Sha256
    participant_registry_hash: Sha256
    pair_input_hash: Sha256
    network_hash: Sha256
    operating_point_hash: Sha256
    q_spec_hash: Sha256
    limits_hash: Sha256
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "ac_network_screen_pair.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ACNetworkScreenResult) or not isinstance(self.reported, ACNetworkScreenResult):
            raise ValidationError("reference and reported screens must be full-network results")
        for name in (
            "baseline_solution_hash", "reference_solution_hash", "reported_solution_hash",
            "pair_hash", "participant_registry_hash", "pair_input_hash", "network_hash",
            "operating_point_hash", "q_spec_hash", "limits_hash",
        ):
            if not isinstance(getattr(self, name), Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("full-network AC pairs cannot be promoted")
        expected = (
            (self.reference, ScreenSide.REFERENCE, self.reference_solution_hash),
            (self.reported, ScreenSide.REPORTED, self.reported_solution_hash),
        )
        for screen, expected_side, expected_solution_hash in expected:
            if screen.side is not expected_side:
                raise ValidationError("nested screen side does not match pair role")
            if screen.baseline_solution_hash != self.baseline_solution_hash:
                raise ValidationError("nested screen baseline hash does not match pair")
            if screen.solution_hash != expected_solution_hash:
                raise ValidationError("nested screen solution hash does not match pair")
            for name, pair_value in (
                ("participant_registry_hash", self.participant_registry_hash),
                ("network_hash", self.network_hash),
                ("operating_point_hash", self.operating_point_hash),
                ("q_spec_hash", self.q_spec_hash),
                ("limits_hash", self.limits_hash),
            ):
                if getattr(screen, name) != pair_value:
                    raise ValidationError(f"nested screen {name} does not match pair")
        if self.reference.calibration_q is not self.reported.calibration_q or self.reference.screening_q is not self.reported.screening_q:
            raise ValidationError("reference and reported Q assumptions must be aligned")

    def to_json(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_json(),
            "reported": self.reported.to_json(),
            "baseline_solution_hash": self.baseline_solution_hash.to_json(),
            "reference_solution_hash": self.reference_solution_hash.to_json(),
            "reported_solution_hash": self.reported_solution_hash.to_json(),
            "pair_hash": self.pair_hash.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "pair_input_hash": self.pair_input_hash.to_json(),
            "network_hash": self.network_hash.to_json(),
            "operating_point_hash": self.operating_point_hash.to_json(),
            "q_spec_hash": self.q_spec_hash.to_json(),
            "limits_hash": self.limits_hash.to_json(),
            "status": self.status,
        }

    @property
    def screen_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def _branch_id(branch_id: int) -> Identifier:
    return Identifier(f"branch_{branch_id:03d}")


def validate_proxy_ac_screen_observables(
    *,
    branch_flow_quantity: str,
    branch_constraint_mode: str,
    branch_observable_mode: str,
    voltage_representation: str,
    voltage_constraint_mode: str,
    voltage_rows_oriented: bool,
) -> None:
    """Fail closed unless proxy and AC screen expose the same predicates.

    This helper deliberately does not select the paper's canonical physical
    definition.  It only verifies that a caller's explicit choices are
    algebraically compatible.  The legacy absolute/both-end screen therefore
    cannot be silently compared to a signed one-row proxy.
    """

    expected_branch_mode = {
        "FROM_END_ACTIVE_POWER": "FROM_END_SIGNED_INCREMENT",
        "TO_END_ACTIVE_POWER": "TO_END_SIGNED_INCREMENT",
    }.get(branch_flow_quantity)
    if expected_branch_mode is None:
        raise ValidationError("ABSOLUTE_ACTIVE_POWER has no signed incremental AC-screen equivalent")
    if branch_constraint_mode != "SIGNED_UPPER_BOUND":
        raise ValidationError("like-for-like signed branch comparison requires SIGNED_UPPER_BOUND")
    if branch_observable_mode != expected_branch_mode:
        raise ValidationError("proxy branch quantity and AC branch observable mode are not aligned")
    if voltage_representation not in {"DELTA_V", "DELTA_V_SQUARED", "V_MAGNITUDE", "V_SQUARED"}:
        raise ValidationError("voltage representation is not registered")
    if voltage_constraint_mode != "SIGNED_UPPER_BOUND":
        raise ValidationError("like-for-like voltage comparison requires SIGNED_UPPER_BOUND")
    if not voltage_rows_oriented:
        raise ValidationError("absolute AC voltage bounds require oriented upper/lower proxy rows")
    if voltage_representation != "DELTA_V":
        raise ValidationError("oriented voltage rows currently require DELTA_V representation")


def _screen_side(
    *,
    side: ScreenSide,
    solution: ACPowerFlowSolution,
    baseline_solution: ACPowerFlowSolution,
    parsed: ParsedMatpowerCase,
    injection_q: FloatVector,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
    baseline_solution_hash: Sha256 | None = None,
    allocation_hash: Sha256 | None = None,
    participant_registry_hash: Sha256 | None = None,
) -> ACNetworkScreenResult:
    branch_count = len(parsed.network.branches)
    bus_count = len(parsed.network.buses)
    if len(limits.branch_budget_mw.values) != branch_count:
        raise ValidationError("branch limit vector must match parsed branch count")
    if len(limits.voltage_lower_limit_pu.values) != bus_count:
        raise ValidationError("voltage limit vectors must match parsed bus count")
    converged = solution.solver_result.solve_success
    if converged:
        baseline_from = baseline_solution.branch_p_from_mw.values
        baseline_to = baseline_solution.branch_p_to_mw.values
        post_from = solution.branch_p_from_mw.values
        post_to = solution.branch_p_to_mw.values
        voltages = solution.bus_voltage_pu.values
        if len(baseline_from) != branch_count or len(baseline_to) != branch_count or len(post_from) != branch_count or len(post_to) != branch_count:
            raise ValidationError("converged AC branch vectors must match parsed branch count")
        if len(voltages) != bus_count:
            raise ValidationError("converged AC voltage vector must match parsed bus count")
    else:
        # A failed AC solve is represented as a shaped, fail-closed diagnostic.
        # Zero placeholders are never interpreted as physical quantities because
        # all family/pass flags are forced false and AC_NONCONVERGENCE is recorded.
        baseline_from = tuple(0.0 for _ in range(branch_count))
        baseline_to = tuple(0.0 for _ in range(branch_count))
        post_from = tuple(0.0 for _ in range(branch_count))
        post_to = tuple(0.0 for _ in range(branch_count))
        voltages = tuple(0.0 for _ in range(bus_count))
    from_increment = tuple(post - base for post, base in zip(post_from, baseline_from))
    to_increment = tuple(post - base for post, base in zip(post_to, baseline_to))
    max_abs_increment = tuple(max(abs(a), abs(b)) for a, b in zip(from_increment, to_increment))
    if limits.branch_observable_mode == "FROM_END_INCREMENT":
        screened_increment = tuple(abs(value) for value in from_increment)
    elif limits.branch_observable_mode == "TO_END_INCREMENT":
        screened_increment = tuple(abs(value) for value in to_increment)
    elif limits.branch_observable_mode == "FROM_END_SIGNED_INCREMENT":
        screened_increment = from_increment
    elif limits.branch_observable_mode == "TO_END_SIGNED_INCREMENT":
        screened_increment = to_increment
    else:
        screened_increment = max_abs_increment
    branch_violation = tuple(max(0.0, value - budget) for value, budget in zip(screened_increment, limits.branch_budget_mw.values))
    branch_flags = tuple(converged and violation <= limits.branch_tolerance_mw.value for violation in branch_violation)
    lower_slack = tuple(value - lower for value, lower in zip(voltages, limits.voltage_lower_limit_pu.values))
    upper_slack = tuple(upper - value for value, upper in zip(voltages, limits.voltage_upper_limit_pu.values))
    voltage_violation = tuple(max(0.0, -lower, -upper) for lower, upper in zip(lower_slack, upper_slack))
    voltage_flags = tuple(converged and violation <= limits.voltage_tolerance_pu.value for violation in voltage_violation)
    branch_ids = IdentifierVector(_branch_id(branch.branch_id) for branch in parsed.network.branches)
    from_bus_ids = tuple(Identifier(str(branch.from_bus.object_id.value)) for branch in parsed.network.branches)
    to_bus_ids = tuple(Identifier(str(branch.to_bus.object_id.value)) for branch in parsed.network.branches)
    bus_ids = IdentifierVector(Identifier(str(bus.bus_id)) for bus in parsed.network.buses)
    offending_branches = IdentifierVector(branch_id for branch_id, flag in zip(branch_ids.values, branch_flags) if not flag)
    offending_buses = IdentifierVector(bus_id for bus_id, flag in zip(bus_ids.values, voltage_flags) if not flag)
    reasons: list[Identifier] = []
    prefix = "REFERENCE" if side is ScreenSide.REFERENCE else "REPORTED"
    if not converged:
        reasons.append(Identifier(f"{prefix}_AC_NONCONVERGENCE"))
    if converged and not all(voltage_flags):
        reasons.append(Identifier(f"{prefix}_VOLTAGE_FAIL"))
    if converged and not all(branch_flags):
        reasons.append(Identifier(f"{prefix}_BRANCH_FAIL"))
    branch = ACBranchScreenDiagnostics(
        branch_ids=branch_ids,
        from_bus_ids=from_bus_ids,
        to_bus_ids=to_bus_ids,
        baseline_from_flow_mw=FloatVector(baseline_from),
        post_from_flow_mw=FloatVector(post_from),
        from_increment_mw=FloatVector(from_increment),
        baseline_to_flow_mw=FloatVector(baseline_to),
        post_to_flow_mw=FloatVector(post_to),
        to_increment_mw=FloatVector(to_increment),
        max_abs_increment_mw=FloatVector(max_abs_increment),
        screened_increment_mw=FloatVector(screened_increment),
        branch_observable_mode=limits.branch_observable_mode,
        budget_mw=limits.branch_budget_mw,
        violation_mw=FloatVector(branch_violation),
        pass_flags=branch_flags,
        offending_branch_ids=offending_branches,
    )
    voltage = ACBusVoltageScreenDiagnostics(
        bus_ids=bus_ids,
        voltage_pu=FloatVector(voltages),
        lower_limit_pu=limits.voltage_lower_limit_pu,
        upper_limit_pu=limits.voltage_upper_limit_pu,
        lower_slack_pu=FloatVector(lower_slack),
        upper_slack_pu=FloatVector(upper_slack),
        violation_pu=FloatVector(voltage_violation),
        pass_flags=voltage_flags,
        offending_bus_ids=offending_buses,
    )
    return ACNetworkScreenResult(
        side=side,
        calibration_q=calibration_q,
        screening_q=screening_q,
        ac_solver_id=solution.solver_result.solver_name,
        ac_solver_version=solution.solver_result.solver_version,
        converged=converged,
        voltage_pass=all(voltage_flags),
        branch_pass=all(branch_flags),
        side_pass=converged and all(voltage_flags) and all(branch_flags),
        branch=branch,
        voltage=voltage,
        q_profile_hash=Sha256(canonical_hash({
            "q_assumption": reactive_specification.q_mode_id.value,
            "q_injection_mvar": injection_q.to_json(),
            "operating_point_hash": model_hash(operating_point),
        })),
        failure_reasons=IdentifierVector(reasons),
        baseline_solution_hash=baseline_solution_hash,
        solution_hash=solution.solution_hash,
        allocation_hash=allocation_hash,
        participant_registry_hash=participant_registry_hash,
        network_hash=parsed.source_hash,
        operating_point_hash=Sha256(model_hash(operating_point)),
        q_spec_hash=reactive_spec_hash(reactive_specification),
        limits_hash=limits.limits_hash,
        quantities_valid=converged,
    )


def _validate_common_inputs(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
) -> None:
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if screening_q is not reactive_specification.q_mode_id:
        raise ValidationError("screening_q must match actual reactive injection specification")
    if calibration_q not in {QAssumption.Q0, QAssumption.Q95} or screening_q not in {QAssumption.Q0, QAssumption.Q95}:
        raise ValidationError("AC screen Q assumptions must be Q0 or Q95")
    if calibration_q is not screening_q:
        raise ValidationError("matched-Q full-network screen requires calibration_q == screening_q")
    if len(limits.branch_budget_mw.values) != len(parsed.network.branches):
        raise ValidationError("branch limits do not match parsed network")
    if len(limits.voltage_lower_limit_pu.values) != len(parsed.network.buses):
        raise ValidationError("voltage limits do not match parsed network")
    if len(selection.participants) == 0:
        raise ValidationError("participant selection must not be empty")


def solve_diagnostic_ac_baseline(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
) -> ACPowerFlowSolution:
    """Solve the shared zero-allocation AC baseline once for a scenario."""

    zero = [0.0] * len(selection.participants)
    baseline_injection = build_net_injections_from_operating_point_spec(
        parsed,
        selection,
        zero,
        operating_point,
        reactive_specification,
        formal_execution=False,
    )
    baseline_solution = solve_ac_power_flow(
        parsed,
        p_injection_mw=baseline_injection.p_mw,
        q_injection_mvar=baseline_injection.q_mvar,
    )
    if not baseline_solution.solver_result.solve_success:
        raise ValidationError("baseline AC solve did not converge")
    return baseline_solution


def screen_diagnostic_ac_network_pair(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    pair: DiagnosticAllocationPairResult,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
) -> ACNetworkScreenPairResult:
    """Run a matched-Q full-network screen for a typed allocation pair."""

    if pair.status != "PAIRED_DIAGNOSTIC" or pair.reference_allocation is None or pair.reported_allocation is None:
        raise ValidationError("full-network AC screening requires a paired diagnostic allocation")
    if pair.participant_ids != selection.participant_set.participant_ids:
        raise ValidationError("allocation pair IDs must match participant selection")
    _validate_common_inputs(parsed, selection, operating_point, reactive_specification, calibration_q, screening_q, limits)
    zero = [0.0] * len(selection.participants)
    baseline_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, zero, operating_point, reactive_specification, formal_execution=False,
    )
    baseline_solution = solve_ac_power_flow(parsed, p_injection_mw=baseline_injection.p_mw, q_injection_mvar=baseline_injection.q_mvar)
    if not baseline_solution.solver_result.solve_success:
        raise ValidationError("baseline AC solve did not converge")
    reference_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, pair.reference_allocation.values_mw.values, operating_point, reactive_specification,
        formal_execution=False, allocation_hash=pair.reference_allocation.allocation_hash,
    )
    reported_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, pair.reported_allocation.values_mw.values, operating_point, reactive_specification,
        formal_execution=False, allocation_hash=pair.reported_allocation.allocation_hash,
    )
    reference_solution = solve_ac_power_flow(parsed, p_injection_mw=reference_injection.p_mw, q_injection_mvar=reference_injection.q_mvar)
    reported_solution = solve_ac_power_flow(parsed, p_injection_mw=reported_injection.p_mw, q_injection_mvar=reported_injection.q_mvar)
    reference = _screen_side(
        side=ScreenSide.REFERENCE, solution=reference_solution, baseline_solution=baseline_solution,
        parsed=parsed, injection_q=reference_injection.q_mvar, operating_point=operating_point,
        reactive_specification=reactive_specification, calibration_q=calibration_q, screening_q=screening_q,
        limits=limits, baseline_solution_hash=baseline_solution.solution_hash,
        allocation_hash=pair.reference_allocation.allocation_hash,
        participant_registry_hash=Sha256(selection.alignment_hash),
    )
    reported = _screen_side(
        side=ScreenSide.REPORTED, solution=reported_solution, baseline_solution=baseline_solution,
        parsed=parsed, injection_q=reported_injection.q_mvar, operating_point=operating_point,
        reactive_specification=reactive_specification, calibration_q=calibration_q, screening_q=screening_q,
        limits=limits, baseline_solution_hash=baseline_solution.solution_hash,
        allocation_hash=pair.reported_allocation.allocation_hash,
        participant_registry_hash=Sha256(selection.alignment_hash),
    )
    pair_input_hash = Sha256(canonical_hash({
        "pair_hash": pair.pair_hash.to_json(),
        "limits_hash": limits.limits_hash.to_json(),
        "operating_point_hash": model_hash(operating_point),
        "q_spec_hash": reactive_spec_hash(reactive_specification).to_json(),
    }))
    payload = {
        "reference": reference.to_json(),
        "reported": reported.to_json(),
        "baseline_solution_hash": baseline_solution.solution_hash.to_json(),
        "reference_solution_hash": reference_solution.solution_hash.to_json(),
        "reported_solution_hash": reported_solution.solution_hash.to_json(),
        "pair_input_hash": pair_input_hash.to_json(),
    }
    pair_hash = Sha256(canonical_hash(payload))
    return ACNetworkScreenPairResult(
        reference=reference,
        reported=reported,
        baseline_solution_hash=baseline_solution.solution_hash,
        reference_solution_hash=reference_solution.solution_hash,
        reported_solution_hash=reported_solution.solution_hash,
        pair_hash=pair_hash,
        participant_registry_hash=Sha256(selection.alignment_hash),
        pair_input_hash=pair_input_hash,
        network_hash=parsed.source_hash,
        operating_point_hash=Sha256(model_hash(operating_point)),
        q_spec_hash=reactive_spec_hash(reactive_specification),
        limits_hash=limits.limits_hash,
    )


def screen_diagnostic_ac_network_binding_pair(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    reference_allocation: AllocationBindingProtocol,
    reported_allocation: AllocationBindingProtocol,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
    baseline_solution: ACPowerFlowSolution | None = None,
) -> ACNetworkScreenPairResult:
    """Screen an explicit typed binding pair against one AC baseline.

    This is the downstream bridge for algebraic or other typed allocation
    bindings.  It deliberately does not accept raw vectors and it never
    constructs a proxy pair or promotes the result: both sides are screened
    at the same matched Q assumption and the returned pair remains
    ``DIAGNOSTIC_ONLY``.
    """

    for name, allocation, expected_side in (
        ("reference_allocation", reference_allocation, AllocationSide.REFERENCE),
        ("reported_allocation", reported_allocation, AllocationSide.REPORTED),
    ):
        if not isinstance(allocation, AllocationBindingProtocol):
            raise ValidationError(f"{name} must implement AllocationBindingProtocol")
        if allocation.side is not expected_side:
            raise ValidationError(f"{name} has the wrong allocation side")
        if allocation.participant_ids != selection.participant_set.participant_ids:
            raise ValidationError(f"{name} participant order does not match selection")
    if reference_allocation.scenario_id != reported_allocation.scenario_id:
        raise ValidationError("reference and reported allocations must share scenario_id")
    if isinstance(reference_allocation, RuleAllocationBindingProtocol) or isinstance(reported_allocation, RuleAllocationBindingProtocol):
        if not isinstance(reference_allocation, RuleAllocationBindingProtocol) or not isinstance(reported_allocation, RuleAllocationBindingProtocol):
            raise ValidationError("reference and reported allocations must use the same rule binding type")
        if reference_allocation.rule_id != reported_allocation.rule_id:
            raise ValidationError("reference and reported allocations must share rule_id")
    if reference_allocation.participant_registry_hash != reported_allocation.participant_registry_hash:
        raise ValidationError("reference and reported allocations must share participant registry provenance")
    _validate_common_inputs(
        parsed,
        selection,
        operating_point,
        reactive_specification,
        calibration_q,
        screening_q,
        limits,
    )
    if baseline_solution is None:
        baseline_solution = solve_diagnostic_ac_baseline(
            parsed,
            selection,
            operating_point,
            reactive_specification,
        )
    elif not isinstance(baseline_solution, ACPowerFlowSolution) or not baseline_solution.solver_result.solve_success:
        raise ValidationError("provided baseline AC solution must be converged")
    reference = _screen_diagnostic_allocation_with_baseline(
        parsed,
        selection,
        reference_allocation,
        operating_point,
        reactive_specification,
        calibration_q=calibration_q,
        screening_q=screening_q,
        limits=limits,
        baseline_solution=baseline_solution,
    )
    reported = _screen_diagnostic_allocation_with_baseline(
        parsed,
        selection,
        reported_allocation,
        operating_point,
        reactive_specification,
        calibration_q=calibration_q,
        screening_q=screening_q,
        limits=limits,
        baseline_solution=baseline_solution,
    )
    if reference.solution_hash is None or reported.solution_hash is None:
        raise ValidationError("AC screen pair requires typed reference/reported solution hashes")
    participant_registry_hash = Sha256(selection.alignment_hash)
    q_spec_hash = reactive_spec_hash(reactive_specification)
    operating_point_hash = Sha256(model_hash(operating_point))
    pair_input_hash = Sha256(canonical_hash({
        "reference_allocation_hash": reference_allocation.allocation_hash.to_json(),
        "reported_allocation_hash": reported_allocation.allocation_hash.to_json(),
        "reference_request_hash": reference_allocation.request_hash.to_json() if reference_allocation.request_hash else None,
        "reported_request_hash": reported_allocation.request_hash.to_json() if reported_allocation.request_hash else None,
        "limits_hash": limits.limits_hash.to_json(),
        "operating_point_hash": operating_point_hash.to_json(),
        "q_spec_hash": q_spec_hash.to_json(),
        "calibration_q": calibration_q.value,
        "screening_q": screening_q.value,
    }))
    pair_payload = {
        "reference": reference.to_json(),
        "reported": reported.to_json(),
        "baseline_solution_hash": baseline_solution.solution_hash.to_json(),
        "reference_solution_hash": reference.solution_hash.to_json(),
        "reported_solution_hash": reported.solution_hash.to_json(),
        "pair_input_hash": pair_input_hash.to_json(),
    }
    pair_hash = Sha256(canonical_hash(pair_payload))
    return ACNetworkScreenPairResult(
        reference=reference,
        reported=reported,
        baseline_solution_hash=baseline_solution.solution_hash,
        reference_solution_hash=reference.solution_hash,
        reported_solution_hash=reported.solution_hash,
        pair_hash=pair_hash,
        participant_registry_hash=participant_registry_hash,
        pair_input_hash=pair_input_hash,
        network_hash=parsed.source_hash,
        operating_point_hash=operating_point_hash,
        q_spec_hash=q_spec_hash,
        limits_hash=limits.limits_hash,
    )


def screen_diagnostic_ac_network_allocation(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation: AllocationBindingProtocol,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
) -> ACNetworkScreenResult:
    """Screen one typed allocation against all branch and bus limits."""

    if allocation.participant_ids != selection.participant_set.participant_ids:
        raise ValidationError("allocation binding IDs must match participant selection")
    _validate_common_inputs(parsed, selection, operating_point, reactive_specification, calibration_q, screening_q, limits)
    zero = [0.0] * len(selection.participants)
    baseline_injection = build_net_injections_from_operating_point_spec(parsed, selection, zero, operating_point, reactive_specification, formal_execution=False)
    baseline_solution = solve_ac_power_flow(parsed, p_injection_mw=baseline_injection.p_mw, q_injection_mvar=baseline_injection.q_mvar)
    if not baseline_solution.solver_result.solve_success:
        raise ValidationError("baseline AC solve did not converge")
    return _screen_diagnostic_allocation_with_baseline(
        parsed, selection, allocation, operating_point, reactive_specification,
        calibration_q=calibration_q, screening_q=screening_q, limits=limits,
        baseline_solution=baseline_solution,
    )


def _screen_diagnostic_allocation_with_baseline(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation: AllocationBindingProtocol,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
    baseline_solution: ACPowerFlowSolution,
) -> ACNetworkScreenResult:
    """Screen one binding using an already solved common baseline."""
    if not isinstance(allocation, AllocationBindingProtocol):
        raise ValidationError("allocation must implement AllocationBindingProtocol")
    if allocation.participant_ids != selection.participant_set.participant_ids:
        raise ValidationError("allocation binding IDs must match participant selection")
    _validate_common_inputs(parsed, selection, operating_point, reactive_specification, calibration_q, screening_q, limits)
    injection = build_net_injections_from_operating_point_spec(
        parsed, selection, allocation.values_mw.values, operating_point, reactive_specification,
        formal_execution=False, allocation_binding=allocation,
    )
    solution = solve_ac_power_flow(parsed, p_injection_mw=injection.p_mw, q_injection_mvar=injection.q_mvar)
    return _screen_side(
        side=ScreenSide(allocation.side.value), solution=solution, baseline_solution=baseline_solution,
        parsed=parsed, injection_q=injection.q_mvar, operating_point=operating_point,
        reactive_specification=reactive_specification, calibration_q=calibration_q, screening_q=screening_q,
        limits=limits, baseline_solution_hash=baseline_solution.solution_hash,
        allocation_hash=allocation.allocation_hash,
        participant_registry_hash=Sha256(selection.alignment_hash),
    )


def screen_diagnostic_cross_q_allocation_with_baseline(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocation: AllocationBindingProtocol,
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
    baseline_solution: ACPowerFlowSolution,
) -> ACNetworkScreenResult:
    """Screen one allocation under a distinct explicit screening Q.

    The ordinary helper intentionally requires matched-Q inputs.  This
    sibling keeps that fail-closed behavior while providing the separate
    cross-Q diagnostic path required by the scientific contract.
    """
    if not isinstance(allocation, AllocationBindingProtocol):
        raise ValidationError("allocation must implement AllocationBindingProtocol")
    if allocation.participant_ids != selection.participant_set.participant_ids:
        raise ValidationError("allocation binding IDs must match participant selection")
    if not isinstance(baseline_solution, ACPowerFlowSolution) or not baseline_solution.solver_result.solve_success:
        raise ValidationError("cross-Q screen requires a converged baseline solution")
    if operating_point.q_assumption is not reactive_specification.q_mode_id:
        raise ValidationError("operating-point and reactive Q assumptions differ")
    if calibration_q not in {QAssumption.Q0, QAssumption.Q95} or screening_q not in {QAssumption.Q0, QAssumption.Q95}:
        raise ValidationError("cross-Q screen Q assumptions must be Q0 or Q95")
    if calibration_q is screening_q:
        raise ValidationError("cross-Q screen requires distinct calibration and screening Q")
    if screening_q is not reactive_specification.q_mode_id:
        raise ValidationError("screening_q must match actual reactive injection specification")
    if len(limits.branch_budget_mw.values) != len(parsed.network.branches):
        raise ValidationError("branch limits do not match parsed network")
    if len(limits.voltage_lower_limit_pu.values) != len(parsed.network.buses) or len(limits.voltage_upper_limit_pu.values) != len(parsed.network.buses):
        raise ValidationError("voltage limits do not match parsed network")
    injection = build_net_injections_from_operating_point_spec(
        parsed,
        selection,
        allocation.values_mw.values,
        operating_point,
        reactive_specification,
        formal_execution=False,
        allocation_binding=allocation,
    )
    solution = solve_ac_power_flow(
        parsed,
        p_injection_mw=injection.p_mw,
        q_injection_mvar=injection.q_mvar,
    )
    return _screen_side(
        side=ScreenSide(allocation.side.value),
        solution=solution,
        baseline_solution=baseline_solution,
        parsed=parsed,
        injection_q=injection.q_mvar,
        operating_point=operating_point,
        reactive_specification=reactive_specification,
        calibration_q=calibration_q,
        screening_q=screening_q,
        limits=limits,
        baseline_solution_hash=baseline_solution.solution_hash,
        allocation_hash=allocation.allocation_hash,
        participant_registry_hash=Sha256(selection.alignment_hash),
    )


def screen_diagnostic_ac_network_allocations(
    parsed: ParsedMatpowerCase,
    selection: ParticipantSelection,
    allocations: Sequence[AllocationBindingProtocol],
    operating_point: OperatingPoint,
    reactive_specification: ReactiveInjectionSpecification,
    *,
    calibration_q: QAssumption,
    screening_q: QAssumption,
    limits: ACNetworkScreenLimits,
) -> tuple[ACNetworkScreenResult, ...]:
    """Screen multiple typed allocations against one shared AC baseline."""
    if not allocations:
        raise ValidationError("at least one allocation binding is required")
    if any(not isinstance(item, AllocationBindingProtocol) for item in allocations):
        raise ValidationError("all allocations must implement AllocationBindingProtocol")
    hashes = [item.allocation_hash for item in allocations]
    if len(set(hashes)) != len(hashes):
        raise ValidationError("allocation bindings must have unique allocation hashes")
    _validate_common_inputs(parsed, selection, operating_point, reactive_specification, calibration_q, screening_q, limits)
    zero = [0.0] * len(selection.participants)
    baseline_injection = build_net_injections_from_operating_point_spec(
        parsed, selection, zero, operating_point, reactive_specification, formal_execution=False,
    )
    baseline_solution = solve_ac_power_flow(parsed, p_injection_mw=baseline_injection.p_mw, q_injection_mvar=baseline_injection.q_mvar)
    if not baseline_solution.solver_result.solve_success:
        raise ValidationError("baseline AC solve did not converge")
    return tuple(
        _screen_diagnostic_allocation_with_baseline(
            parsed, selection, allocation, operating_point, reactive_specification,
            calibration_q=calibration_q, screening_q=screening_q, limits=limits,
            baseline_solution=baseline_solution,
        )
        for allocation in allocations
    )
