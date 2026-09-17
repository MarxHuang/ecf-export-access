from __future__ import annotations

import pytest

from r4r.externality_feedback import (
    EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID,
    apply_access_ceiling,
    assess_replay,
    fulfillment_replay_request,
    ideal_verified_available_export_telemetry,
    metered_delivery,
    update_access_ceiling,
)


def test_exact_verified_telemetry_is_an_explicit_identity_interface() -> None:
    truth = [0.0, 2.5, 4.0]
    telemetry = ideal_verified_available_export_telemetry(truth)
    assert EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID == "A_TEL_EXACT_V1"
    assert telemetry == truth
    assert telemetry is not truth
    assert metered_delivery([1.0, 3.0, 5.0], telemetry) == pytest.approx(
        [0.0, 2.5, 4.0]
    )


@pytest.mark.parametrize("invalid", [[-1.0], [float("nan")], [float("inf")]])
def test_exact_verified_telemetry_rejects_invalid_device_values(
    invalid: list[float],
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ideal_verified_available_export_telemetry(invalid)


def test_replay_changes_only_idle_award_requests() -> None:
    replay, idle = fulfillment_replay_request(
        [8.0, 4.0, 3.0],
        [6.0, 4.0, 2.0],
        [4.0, 4.0, 2.0],
        tolerance_mw=1.0e-9,
    )
    assert idle == (0,)
    assert replay == pytest.approx([4.0, 4.0, 3.0])


def test_replay_gate_requires_access_release_and_delivery_recovery() -> None:
    assessment = assess_replay(
        allocation_mw=[6.0, 4.0],
        delivery_mw=[4.0, 4.0],
        replay_allocation_mw=[4.0, 6.0],
        replay_delivery_mw=[4.0, 6.0],
        idle_indices=[0],
        replay_valid=True,
        tolerance_mw=1.0e-9,
    )
    assert assessment.gate_triggered
    assert assessment.gate_reason == "RECOVERABLE_ALLOCATION_EXTERNALITY"
    assert assessment.outsider_allocation_gain_mw == pytest.approx(2.0)
    assert assessment.total_delivery_gain_mw == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("replay_allocation", "replay_delivery", "reason"),
    [
        ([6.0, 4.0], [4.0, 4.0], "NO_OUTSIDER_ACCESS_RELEASE"),
        ([4.0, 6.0], [4.0, 4.0], "NO_OUTSIDER_DELIVERY_RECOVERY"),
    ],
)
def test_mismatch_without_usable_release_does_not_open_gate(
    replay_allocation: list[float], replay_delivery: list[float], reason: str,
) -> None:
    assessment = assess_replay(
        allocation_mw=[6.0, 4.0],
        delivery_mw=[4.0, 4.0],
        replay_allocation_mw=replay_allocation,
        replay_delivery_mw=replay_delivery,
        idle_indices=[0],
        replay_valid=True,
        tolerance_mw=1.0e-9,
    )
    assert not assessment.gate_triggered
    assert assessment.gate_reason == reason


def test_feeder_wide_shortfall_has_no_outsider_and_does_not_open_gate() -> None:
    assessment = assess_replay(
        allocation_mw=[6.0, 4.0],
        delivery_mw=[3.0, 2.0],
        replay_allocation_mw=[3.0, 2.0],
        replay_delivery_mw=[3.0, 2.0],
        idle_indices=[0, 1],
        replay_valid=True,
        tolerance_mw=1.0e-9,
    )
    assert not assessment.gate_triggered
    assert assessment.gate_reason == "NO_OUTSIDER_ACCESS_RELEASE"


def test_ceiling_penalizes_only_gated_idle_awards_and_otherwise_recovers() -> None:
    updated, target = update_access_ceiling(
        ceiling_before_mw=[10.0, 6.0],
        registered_capacity_mw=[10.0, 10.0],
        delivery_mw=[4.0, 6.0],
        idle_indices=[0],
        gate_triggered=True,
        beta=0.5,
    )
    assert target == pytest.approx([4.0, 10.0])
    assert updated == pytest.approx([7.0, 8.0])

    recovered, target = update_access_ceiling(
        ceiling_before_mw=updated,
        registered_capacity_mw=[10.0, 10.0],
        delivery_mw=[2.0, 2.0],
        idle_indices=[0, 1],
        gate_triggered=False,
        beta=0.5,
    )
    assert target == pytest.approx([10.0, 10.0])
    assert recovered == pytest.approx([8.5, 9.0])


def test_access_ceiling_is_nonbypassable() -> None:
    assert apply_access_ceiling(
        [100.0, 5.0], [10.0, 10.0], [7.0, 8.0],
    ) == pytest.approx([7.0, 5.0])
