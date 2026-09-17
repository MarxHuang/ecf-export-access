"""Counterfactual, telemetry-assisted feedback for scarce export access.

The functions in this module are deliberately independent of a particular
allocation solver.  A runner supplies the observed and replayed allocations;
this module decides whether the observed idle authorization displaced usable
access and updates the next-interval participant ceiling.  The deployable
interface distinguishes metered delivery from post-event available-export
telemetry; simulation truth enters only through an explicitly registered ideal
telemetry assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID = "A_TEL_EXACT_V1"


def ideal_verified_available_export_telemetry(
    realized_capability_mw: Sequence[float],
) -> list[float]:
    """Return the registered exact post-event available-export telemetry.

    This identity map is intentionally explicit.  It records the controlled
    experiment's assumption ``a_hat_tel(t) = a(t)`` while keeping replay code
    dependent on a telemetry interface rather than on latent simulation truth.
    Non-finite or negative inputs are rejected instead of silently clipped.
    """

    telemetry = [float(value) for value in realized_capability_mw]
    if any(not math.isfinite(value) or value < 0.0 for value in telemetry):
        raise ValueError(
            "verified available-export telemetry must be finite and nonnegative"
        )
    return telemetry


@dataclass(frozen=True)
class ReplayAssessment:
    """Auditable result of one same-realization allocation replay."""

    idle_indices: tuple[int, ...]
    total_idle_award_mw: float
    outsider_allocation_gain_mw: float
    outsider_delivery_gain_mw: float
    total_delivery_gain_mw: float
    allocation_redistribution_mw: float
    gate_triggered: bool
    gate_reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "idle_indices": list(self.idle_indices),
            "total_idle_award_mw": self.total_idle_award_mw,
            "outsider_allocation_gain_mw": self.outsider_allocation_gain_mw,
            "outsider_delivery_gain_mw": self.outsider_delivery_gain_mw,
            "total_delivery_gain_mw": self.total_delivery_gain_mw,
            "allocation_redistribution_mw": self.allocation_redistribution_mw,
            "gate_triggered": self.gate_triggered,
            "gate_reason": self.gate_reason,
        }


def metered_delivery(
    allocation_mw: Sequence[float], available_export_mw: Sequence[float],
) -> list[float]:
    """Return non-discretionary delivery for an available-export vector."""

    if len(allocation_mw) != len(available_export_mw):
        raise ValueError(
            "allocation and available-export vectors must have equal length"
        )
    return [
        min(max(float(award), 0.0), max(float(available_export), 0.0))
        for award, available_export in zip(allocation_mw, available_export_mw)
    ]


def fulfillment_ratio(
    allocation_mw: Sequence[float], delivery_mw: Sequence[float], *, tolerance_mw: float,
) -> list[float]:
    """Return delivery-to-award ratios used only by the legacy ablation."""

    if len(allocation_mw) != len(delivery_mw):
        raise ValueError("allocation and delivery vectors must have equal length")
    return [
        1.0
        if float(award) <= float(tolerance_mw)
        else min(1.0, max(0.0, float(delivery) / float(award)))
        for award, delivery in zip(allocation_mw, delivery_mw)
    ]


def fulfillment_replay_request(
    effective_request_mw: Sequence[float],
    allocation_mw: Sequence[float],
    delivery_mw: Sequence[float],
    *,
    tolerance_mw: float,
) -> tuple[list[float], tuple[int, ...]]:
    """Cap idle-award participants at metered delivery for one replay.

    The replay removes no request from a participant that fully used its award.
    It is a post-event counterfactual used only to decide the next transition;
    it does not rewrite the completed allocation.
    """

    if not (
        len(effective_request_mw) == len(allocation_mw) == len(delivery_mw)
    ):
        raise ValueError("request, allocation, and delivery vectors must have equal length")
    idle = tuple(
        index
        for index, (award, delivered) in enumerate(zip(allocation_mw, delivery_mw))
        if float(award) - float(delivered) > float(tolerance_mw)
    )
    replay = [float(value) for value in effective_request_mw]
    for index in idle:
        replay[index] = min(replay[index], max(float(delivery_mw[index]), 0.0))
    return replay, idle


def assess_replay(
    *,
    allocation_mw: Sequence[float],
    delivery_mw: Sequence[float],
    replay_allocation_mw: Sequence[float],
    replay_delivery_mw: Sequence[float],
    idle_indices: Sequence[int],
    replay_valid: bool,
    tolerance_mw: float,
) -> ReplayAssessment:
    """Test whether an idle award displaced usable access elsewhere."""

    if not (
        len(allocation_mw)
        == len(delivery_mw)
        == len(replay_allocation_mw)
        == len(replay_delivery_mw)
    ):
        raise ValueError("observed and replay vectors must have equal length")
    idle = tuple(sorted({int(index) for index in idle_indices}))
    if any(index < 0 or index >= len(allocation_mw) for index in idle):
        raise ValueError("idle index lies outside the participant vector")
    idle_set = set(idle)
    outsiders = [index for index in range(len(allocation_mw)) if index not in idle_set]
    total_idle = math.fsum(
        max(float(allocation_mw[index]) - float(delivery_mw[index]), 0.0)
        for index in idle
    )
    outsider_allocation_gain = math.fsum(
        max(float(replay_allocation_mw[index]) - float(allocation_mw[index]), 0.0)
        for index in outsiders
    )
    outsider_delivery_gain = math.fsum(
        max(float(replay_delivery_mw[index]) - float(delivery_mw[index]), 0.0)
        for index in outsiders
    )
    total_delivery_gain = math.fsum(replay_delivery_mw) - math.fsum(delivery_mw)
    redistribution = 0.5 * math.fsum(
        abs(float(replay) - float(observed))
        for replay, observed in zip(replay_allocation_mw, allocation_mw)
    )

    if not idle:
        reason = "NO_IDLE_AWARD"
    elif not replay_valid:
        reason = "REPLAY_INVALID"
    elif outsider_allocation_gain <= float(tolerance_mw):
        reason = "NO_OUTSIDER_ACCESS_RELEASE"
    elif outsider_delivery_gain <= float(tolerance_mw):
        reason = "NO_OUTSIDER_DELIVERY_RECOVERY"
    elif total_delivery_gain <= float(tolerance_mw):
        reason = "NO_NET_DELIVERY_RECOVERY"
    else:
        reason = "RECOVERABLE_ALLOCATION_EXTERNALITY"
    return ReplayAssessment(
        idle_indices=idle,
        total_idle_award_mw=total_idle,
        outsider_allocation_gain_mw=outsider_allocation_gain,
        outsider_delivery_gain_mw=outsider_delivery_gain,
        total_delivery_gain_mw=total_delivery_gain,
        allocation_redistribution_mw=redistribution,
        gate_triggered=reason == "RECOVERABLE_ALLOCATION_EXTERNALITY",
        gate_reason=reason,
    )


def update_access_ceiling(
    *,
    ceiling_before_mw: Sequence[float],
    registered_capacity_mw: Sequence[float],
    delivery_mw: Sequence[float],
    idle_indices: Sequence[int],
    gate_triggered: bool,
    beta: float,
) -> tuple[list[float], list[float]]:
    """Update an MW-valued admissible ceiling with explicit recovery.

    When the replay gate is open, an under-delivering participant's metered
    delivery is an observed capability value under ``y=min(x,a)``.  Otherwise
    the registered capacity is the recovery target.  The latter prevents a
    weather-only mismatch with no displaced usable access from causing a new
    restriction.
    """

    if not (
        len(ceiling_before_mw)
        == len(registered_capacity_mw)
        == len(delivery_mw)
    ):
        raise ValueError("ceiling, capacity, and delivery vectors must have equal length")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    idle = {int(index) for index in idle_indices}
    if any(index < 0 or index >= len(ceiling_before_mw) for index in idle):
        raise ValueError("idle index lies outside the participant vector")
    targets = [float(value) for value in registered_capacity_mw]
    if gate_triggered:
        for index in idle:
            targets[index] = min(
                max(float(delivery_mw[index]), 0.0),
                float(registered_capacity_mw[index]),
            )
    updated = [
        min(
            max(
                (1.0 - float(beta)) * float(previous)
                + float(beta) * float(target),
                0.0,
            ),
            float(capacity),
        )
        for previous, target, capacity in zip(
            ceiling_before_mw, targets, registered_capacity_mw,
        )
    ]
    return updated, targets


def apply_access_ceiling(
    request_mw: Sequence[float],
    registered_capacity_mw: Sequence[float],
    ceiling_mw: Sequence[float],
) -> list[float]:
    """Apply capacity and metered ceilings without modifying the feasible set."""

    if not (
        len(request_mw) == len(registered_capacity_mw) == len(ceiling_mw)
    ):
        raise ValueError("request, capacity, and ceiling vectors must have equal length")
    return [
        min(
            max(float(request), 0.0),
            max(float(capacity), 0.0),
            max(float(ceiling), 0.0),
        )
        for request, capacity, ceiling in zip(
            request_mw, registered_capacity_mw, ceiling_mw,
        )
    ]


__all__ = [
    "EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID",
    "ReplayAssessment",
    "apply_access_ceiling",
    "assess_replay",
    "fulfillment_ratio",
    "fulfillment_replay_request",
    "ideal_verified_available_export_telemetry",
    "metered_delivery",
    "update_access_ceiling",
]
