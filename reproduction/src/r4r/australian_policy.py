"""Pure policy utilities for the Australian time-series replay."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Sequence

import numpy as np

from .australian_data import AustralianDataError


def fit_persistence_upper_residual(
    available_export_kw: np.ndarray,
    complete_customer_day: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    """Fit a customer/interval upper residual on consecutive training days.

    The output has shape ``[interval, customer]`` and is computed only from
    the training array supplied by the caller.  A residual is current-day
    available export minus previous-day same-interval export.
    """

    available = np.asarray(available_export_kw, dtype=float)
    complete = np.asarray(complete_customer_day, dtype=bool)
    if available.ndim != 3:
        raise AustralianDataError("available export must have [day, interval, customer] shape")
    if complete.shape != (available.shape[0], available.shape[2]):
        raise AustralianDataError("completeness mask does not align with available export")
    if available.shape[0] < 2:
        raise AustralianDataError("at least two training days are required")
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise AustralianDataError("residual quantile must lie strictly between zero and one")
    residual = available[1:] - available[:-1]
    valid = complete[1:] & complete[:-1]
    residual = np.where(valid[:, None, :], residual, np.nan)
    with np.errstate(invalid="ignore"):
        fitted = np.nanquantile(residual, quantile, axis=0)
    if np.any(~np.isfinite(fitted)):
        raise AustralianDataError("training data do not identify every residual coefficient")
    return np.maximum(fitted, 0.0)


def persistence_request(
    previous_day_available_kw: np.ndarray,
    registered_capacity_kw: Sequence[float],
    *,
    upper_residual_kw: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a previous-day same-interval request without future leakage."""

    previous = np.asarray(previous_day_available_kw, dtype=float)
    capacity = np.asarray(registered_capacity_kw, dtype=float)
    if previous.ndim != 2 or previous.shape[1] != capacity.size:
        raise AustralianDataError("persistence input must have [interval, customer] shape")
    if np.any(~np.isfinite(previous)) or np.any(previous < 0.0):
        raise AustralianDataError("previous-day availability must be finite and nonnegative")
    if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0.0):
        raise AustralianDataError("registered capacity must be finite and positive")
    if upper_residual_kw is None:
        residual = np.zeros_like(previous)
    else:
        residual = np.asarray(upper_residual_kw, dtype=float)
        if residual.shape != previous.shape:
            raise AustralianDataError("upper residual does not align with persistence input")
        if np.any(~np.isfinite(residual)) or np.any(residual < 0.0):
            raise AustralianDataError("upper residual must be finite and nonnegative")
    return np.minimum(previous + residual, capacity[None, :])


def matched_uniform_request(
    capacity_bounded_request_kw: Sequence[float],
    selective_request_kw: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Remove the selective policy's total request by a common factor."""

    raw = np.asarray(capacity_bounded_request_kw, dtype=float)
    selective = np.asarray(selective_request_kw, dtype=float)
    if raw.ndim != 1 or selective.shape != raw.shape:
        raise AustralianDataError("matched-control vectors must align")
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise AustralianDataError("raw request must be finite and nonnegative")
    if np.any(~np.isfinite(selective)) or np.any(selective < -1.0e-12):
        raise AustralianDataError("selective request must be finite and nonnegative")
    if np.any(selective > raw + 1.0e-9):
        raise AustralianDataError("selective request cannot exceed raw request")
    raw_total = float(np.sum(raw))
    target_total = float(np.sum(selective))
    if raw_total == 0.0:
        return np.zeros_like(raw), 1.0
    factor = min(max(target_total / raw_total, 0.0), 1.0)
    matched = factor * raw
    return matched, factor


def deterministic_dropout_mask(
    count: int,
    fraction: float,
    *,
    seed: str,
    date: str,
    interval_index: int,
) -> np.ndarray:
    """Return a reproducible participant-order dropout mask.

    A SHA-256 digest of the frozen seed and interval identity initializes a
    PCG64 stream. The fixed connection order then supplies one draw per
    connection, so worker scheduling cannot change the selected dropouts.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise AustralianDataError("dropout count must be a nonnegative integer")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise AustralianDataError("dropout fraction must lie in [0, 1]")
    if not seed or not date:
        raise AustralianDataError("dropout seed and date must be nonempty")
    if (
        isinstance(interval_index, bool)
        or not isinstance(interval_index, int)
        or interval_index < 0
    ):
        raise AustralianDataError("interval index must be a nonnegative integer")
    digest = sha256(f"{seed}|{date}|{interval_index}".encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    return rng.random(count) < fraction


def post_event_availability_signal(
    available_mw: Sequence[float],
    delivered_mw: Sequence[float],
    *,
    mode: str,
    scale: float = 1.0,
    previous_available_mw: Sequence[float] | None = None,
    dropout_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Construct a post-event availability signal for a sensitivity run.

    Every mode is floored at metered delivery. Completed delivery is observed,
    so an inferred availability smaller than that amount is inconsistent.
    ``LAG_ONE_INTERVAL`` uses delivery alone when no prior interval is supplied,
    matching the daily state reset in the external replay.
    """

    available = np.asarray(available_mw, dtype=float)
    delivered = np.asarray(delivered_mw, dtype=float)
    if available.ndim != 1 or delivered.shape != available.shape:
        raise AustralianDataError("availability and delivery signals must align")
    if np.any(~np.isfinite(available)) or np.any(available < 0.0):
        raise AustralianDataError("available export must be finite and nonnegative")
    if np.any(~np.isfinite(delivered)) or np.any(delivered < 0.0):
        raise AustralianDataError("metered delivery must be finite and nonnegative")
    if np.any(delivered > available + 1.0e-9):
        raise AustralianDataError("metered delivery cannot exceed actual availability")

    normalized_mode = str(mode).upper()
    if normalized_mode == "EXACT":
        candidate = available
    elif normalized_mode == "MULTIPLICATIVE":
        if not math.isfinite(scale) or scale < 0.0:
            raise AustralianDataError("telemetry scale must be finite and nonnegative")
        candidate = scale * available
    elif normalized_mode == "LAG_ONE_INTERVAL":
        if previous_available_mw is None:
            candidate = delivered
        else:
            candidate = np.asarray(previous_available_mw, dtype=float)
            if candidate.shape != available.shape:
                raise AustralianDataError("lagged availability signal must align")
            if np.any(~np.isfinite(candidate)) or np.any(candidate < 0.0):
                raise AustralianDataError(
                    "lagged availability must be finite and nonnegative"
                )
    elif normalized_mode == "DETERMINISTIC_DROPOUT":
        if dropout_mask is None:
            raise AustralianDataError("dropout mode requires a dropout mask")
        mask = np.asarray(dropout_mask, dtype=bool)
        if mask.shape != available.shape:
            raise AustralianDataError("dropout mask must align with availability")
        candidate = np.where(mask, delivered, available)
    else:
        raise AustralianDataError(f"unsupported telemetry mode: {mode}")
    return np.maximum(delivered, candidate)


__all__ = [
    "deterministic_dropout_mask",
    "fit_persistence_upper_residual",
    "matched_uniform_request",
    "post_event_availability_signal",
    "persistence_request",
]
