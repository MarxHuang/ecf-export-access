"""Deterministic allocation metrics from the frozen scientific contract.

This module is deliberately independent of optimization and AC execution.  It
implements only the two Jain indicators whose mathematical definitions are
already frozen in the active contract.  Inputs are validated, never clipped,
and never regularized with an epsilon.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from r4r.errors import ValidationError
from r4r.models.evidence import JainMetricResult
from r4r.types import Identifier, MetricStatus
from r4r.types.scalars import FiniteFloat


def _finite_vector(values: Sequence[float], field: str) -> tuple[float, ...]:
    """Return a finite numeric vector without changing any value."""
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{field} must be a numeric sequence")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{field} must be a numeric sequence") from exc
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{field}[{index}] must be a finite real scalar")
        if not math.isfinite(float(value)):
            raise ValidationError(f"{field}[{index}] must be a finite real scalar")
    return tuple(float(value) for value in result)


def raw_jain(allocation_mw: Sequence[float], metric_id: Identifier | None = None) -> JainMetricResult:
    """Compute ``(sum(x)**2) / (n * sum(x**2))``.

    An empty vector and an all-zero vector have explicit undefined statuses.
    Signed values are retained exactly; this function does not infer or apply a
    non-negativity policy that belongs to the allocation-domain module.
    """
    values = _finite_vector(allocation_mw, "allocation_mw")
    identifier = metric_id or Identifier("J_raw")
    count = len(values)
    if count == 0:
        return JainMetricResult(identifier, None, MetricStatus.UNDEFINED_EMPTY_SET, 0)
    if all(value == 0.0 for value in values):
        return JainMetricResult(identifier, None, MetricStatus.UNDEFINED_ALL_ZERO, count)
    numerator = math.fsum(values) ** 2
    denominator = count * math.fsum(value * value for value in values)
    return JainMetricResult(identifier, FiniteFloat(numerator / denominator), MetricStatus.VALID, count)


def normalized_jain(
    allocation_mw: Sequence[float],
    capacity_mw: Sequence[float],
    metric_id: Identifier | None = None,
) -> JainMetricResult:
    """Compute Jain fairness over participants with strictly positive capacity.

    ``C_i=0`` is excluded from both the numerator and denominator, whereas any
    negative capacity returns ``INVALID_NEGATIVE_CAPACITY``.  An empty input
    set and a nonempty set with no positive capacity have distinct statuses.
    If all retained normalized allocations are zero, the result is
    ``UNDEFINED_ALL_ZERO``.  Length mismatches and non-finite values are hard
    validation errors rather than silent padding, truncation, or clipping.
    """
    values = _finite_vector(allocation_mw, "allocation_mw")
    capacities = _finite_vector(capacity_mw, "capacity_mw")
    if len(values) != len(capacities):
        raise ValidationError("allocation_mw and capacity_mw must have equal length")
    identifier = metric_id or Identifier("J_norm")
    if not values:
        return JainMetricResult(identifier, None, MetricStatus.UNDEFINED_EMPTY_SET, 0)
    if any(capacity < 0.0 for capacity in capacities):
        return JainMetricResult(
            identifier,
            None,
            MetricStatus.INVALID_NEGATIVE_CAPACITY,
            len(capacities),
        )
    normalized = tuple(value / capacity for value, capacity in zip(values, capacities) if capacity > 0.0)
    count = len(normalized)
    if count == 0:
        return JainMetricResult(identifier, None, MetricStatus.UNDEFINED_NO_POSITIVE_CAPACITY, 0)
    if all(value == 0.0 for value in normalized):
        return JainMetricResult(identifier, None, MetricStatus.UNDEFINED_ALL_ZERO, count)
    numerator = math.fsum(normalized) ** 2
    denominator = count * math.fsum(value * value for value in normalized)
    return JainMetricResult(identifier, FiniteFloat(numerator / denominator), MetricStatus.VALID, count)


__all__ = ["normalized_jain", "raw_jain"]
