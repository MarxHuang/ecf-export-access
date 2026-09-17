from __future__ import annotations

import numpy as np
import pytest

from r4r.australian_data import AustralianDataError
from r4r.australian_policy import (
    deterministic_dropout_mask,
    fit_persistence_upper_residual,
    matched_uniform_request,
    post_event_availability_signal,
    persistence_request,
)


def test_training_residual_uses_only_consecutive_complete_days() -> None:
    available = np.array(
        [
            [[1.0, 1.0]],
            [[3.0, 5.0]],
            [[100.0, 7.0]],
            [[5.0, 9.0]],
        ]
    )
    complete = np.array([[True, True], [True, True], [False, True], [True, True]])
    fitted = fit_persistence_upper_residual(available, complete, quantile=0.5)
    assert fitted.shape == (1, 2)
    assert fitted[0, 0] == pytest.approx(2.0)
    assert fitted[0, 1] == pytest.approx(2.0)


def test_persistence_request_adds_frozen_upper_residual_and_caps_capacity() -> None:
    request = persistence_request(
        np.array([[1.0, 3.0], [2.0, 4.0]]),
        [2.5, 5.0],
        upper_residual_kw=np.array([[2.0, 1.0], [0.0, 3.0]]),
    )
    np.testing.assert_allclose(request, [[2.5, 4.0], [2.0, 5.0]])


def test_matched_uniform_control_has_exact_total_reduction() -> None:
    raw = np.array([4.0, 2.0, 0.0])
    selective = np.array([1.0, 2.0, 0.0])
    matched, factor = matched_uniform_request(raw, selective)
    assert factor == pytest.approx(0.5)
    assert matched == pytest.approx([2.0, 1.0, 0.0])
    assert np.sum(raw - matched) == pytest.approx(np.sum(raw - selective))


def test_matched_uniform_rejects_selective_request_above_raw() -> None:
    with pytest.raises(AustralianDataError, match="cannot exceed"):
        matched_uniform_request([1.0], [1.1])


def test_post_event_signal_modes_keep_metered_delivery_as_a_floor() -> None:
    available = np.array([1.0, 4.0, 2.0])
    delivered = np.array([0.8, 1.0, 2.0])
    np.testing.assert_allclose(
        post_event_availability_signal(available, delivered, mode="EXACT"),
        available,
    )
    np.testing.assert_allclose(
        post_event_availability_signal(
            available, delivered, mode="MULTIPLICATIVE", scale=0.5
        ),
        [0.8, 2.0, 2.0],
    )
    np.testing.assert_allclose(
        post_event_availability_signal(
            available,
            delivered,
            mode="LAG_ONE_INTERVAL",
            previous_available_mw=[0.2, 3.0, 1.0],
        ),
        [0.8, 3.0, 2.0],
    )
    np.testing.assert_allclose(
        post_event_availability_signal(
            available,
            delivered,
            mode="DETERMINISTIC_DROPOUT",
            dropout_mask=[True, False, True],
        ),
        [0.8, 4.0, 2.0],
    )


def test_lagged_signal_uses_delivery_only_at_daily_initialization() -> None:
    signal = post_event_availability_signal(
        [2.0, 3.0], [0.5, 1.0], mode="LAG_ONE_INTERVAL"
    )
    np.testing.assert_allclose(signal, [0.5, 1.0])


def test_dropout_mask_is_schedule_independent_and_bounded() -> None:
    first = deterministic_dropout_mask(
        1000,
        0.10,
        seed="AU_TELEMETRY_DROPOUT_V1",
        date="2012-07-02",
        interval_index=7,
    )
    second = deterministic_dropout_mask(
        1000,
        0.10,
        seed="AU_TELEMETRY_DROPOUT_V1",
        date="2012-07-02",
        interval_index=7,
    )
    assert first.dtype == np.bool_
    assert np.array_equal(first, second)
    assert 60 <= int(first.sum()) <= 140
