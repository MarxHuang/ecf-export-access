"""Network-feasible proportional export allocation for Australian replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .australian_opendss import AustralianDSSScreenResult, AustralianOpenDSSModel


@dataclass(frozen=True, slots=True)
class ProportionalAllocationResult:
    allocation_kw: np.ndarray
    request_scale: float
    status: str
    power_flow_count: int
    load_only_screen: AustralianDSSScreenResult
    authorized_export_screen: AustralianDSSScreenResult

    @property
    def pair_pass(self) -> bool:
        return (
            self.status == "OPTIMAL_AC_PREFIX"
            and self.load_only_screen.physical_pair_pass
            and self.authorized_export_screen.physical_pair_pass
        )


def allocate_proportional_ac_prefix(
    model: AustralianOpenDSSModel,
    household_load_kw: Sequence[float],
    request_kw: Sequence[float],
    participant_mask: Sequence[bool],
    *,
    scalar_tolerance: float = 1.0e-5,
    max_bisection_iterations: int = 24,
    authorization_voltage_max_pu: float | None = None,
) -> ProportionalAllocationResult:
    """Maximize a common request scale on the feasible prefix from zero.

    Both the no-PV load endpoint and the full authorized-export endpoint are
    screened.  Realised PV is deliberately absent from this allocation step.
    If the full request fails, bisection returns the largest tested scale in
    the connected feasible prefix containing zero.
    """

    load = np.asarray(household_load_kw, dtype=float)
    request = np.asarray(request_kw, dtype=float)
    participant = np.asarray(participant_mask, dtype=bool)
    if not (load.shape == request.shape == participant.shape) or load.ndim != 1:
        raise ValueError("allocation vectors must be aligned and one-dimensional")
    if np.any(~np.isfinite(load)) or np.any(load < 0.0):
        raise ValueError("household load must be finite and nonnegative")
    if np.any(~np.isfinite(request)) or np.any(request < 0.0):
        raise ValueError("request must be finite and nonnegative")
    if np.any(request[~participant] > 1.0e-12):
        raise ValueError("nonparticipants cannot submit an export request")
    if not math.isfinite(scalar_tolerance) or not 0.0 < scalar_tolerance < 1.0:
        raise ValueError("scalar tolerance must lie in (0,1)")
    if max_bisection_iterations < 1:
        raise ValueError("at least one bisection iteration is required")
    authorization_limit = (
        model.voltage_max_pu
        if authorization_voltage_max_pu is None
        else float(authorization_voltage_max_pu)
    )
    if not model.voltage_min_pu < authorization_limit <= model.voltage_max_pu:
        raise ValueError(
            "authorization voltage limit must exceed the physical lower limit "
            "and not exceed the physical upper limit"
        )

    def authorization_pass(screen: AustralianDSSScreenResult) -> bool:
        return (
            screen.converged
            and screen.customer_undervoltage_count == 0
            and screen.customer_voltage_max_pu <= authorization_limit + 1.0e-9
            and screen.transformer_overload_count == 0
        )

    zero = np.zeros_like(request)
    load_only, _, _ = model.solve(load, zero, zero, participant)
    power_flow_count = 1
    zero_export = model.solve_authorized_export(load, zero, participant)
    power_flow_count += 1
    if not load_only.physical_pair_pass or not authorization_pass(zero_export):
        return ProportionalAllocationResult(
            allocation_kw=zero,
            request_scale=0.0,
            status="INVALID_BASE_STATE",
            power_flow_count=power_flow_count,
            load_only_screen=load_only,
            authorized_export_screen=zero_export,
        )
    if float(np.sum(request)) == 0.0:
        return ProportionalAllocationResult(
            allocation_kw=zero,
            request_scale=1.0,
            status="OPTIMAL_AC_PREFIX",
            power_flow_count=power_flow_count,
            load_only_screen=load_only,
            authorized_export_screen=zero_export,
        )

    full = model.solve_authorized_export(load, request, participant)
    power_flow_count += 1
    if authorization_pass(full):
        return ProportionalAllocationResult(
            allocation_kw=request.copy(),
            request_scale=1.0,
            status="OPTIMAL_AC_PREFIX",
            power_flow_count=power_flow_count,
            load_only_screen=load_only,
            authorized_export_screen=full,
        )

    lower = 0.0
    upper = 1.0
    lower_screen = zero_export
    for _ in range(max_bisection_iterations):
        if upper - lower <= scalar_tolerance:
            break
        midpoint = 0.5 * (lower + upper)
        candidate = midpoint * request
        screen = model.solve_authorized_export(load, candidate, participant)
        power_flow_count += 1
        if authorization_pass(screen):
            lower = midpoint
            lower_screen = screen
        else:
            upper = midpoint
    allocation = lower * request
    return ProportionalAllocationResult(
        allocation_kw=allocation,
        request_scale=lower,
        status="OPTIMAL_AC_PREFIX",
        power_flow_count=power_flow_count,
        load_only_screen=load_only,
        authorized_export_screen=lower_screen,
    )


__all__ = ["ProportionalAllocationResult", "allocate_proportional_ac_prefix"]
