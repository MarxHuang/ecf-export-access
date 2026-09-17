"""Canonical load operating-point construction for diagnostic AC inputs.

The network parser exposes native MATPOWER load vectors.  The scientific
contract fixes the case-study operating point at a 0.70 load scale, so callers
must use this explicit constructor instead of silently reusing native loads.
This module only materializes the typed input; it performs no AC solve and no
participant-Q or Q95 sign selection.
"""
from __future__ import annotations

from r4r.errors import ValidationError
from r4r.models import OperatingPoint
from r4r.network_parser import ParsedMatpowerCase
from r4r.types import FloatVector, FiniteFloat, QAssumption


CANONICAL_LOAD_SCALE = 0.70


def build_canonical_operating_point(
    parsed: ParsedMatpowerCase,
    *,
    q_assumption: QAssumption,
    load_scale: float = CANONICAL_LOAD_SCALE,
) -> OperatingPoint:
    """Scale parsed P/Q demand once and bind the explicit Q assumption."""

    if not isinstance(q_assumption, QAssumption):
        raise ValidationError("q_assumption must be a registered Q token")
    if load_scale != CANONICAL_LOAD_SCALE:
        raise ValidationError("canonical operating-point load scale is fixed at 0.70")
    bus_count = len(parsed.network.buses)
    if len(parsed.p_load_mw.values) != bus_count or len(parsed.q_load_mvar.values) != bus_count:
        raise ValidationError("parsed load vectors do not match network bus count")
    scale = FiniteFloat(load_scale)
    return OperatingPoint(
        load_scale=scale,
        q_assumption=q_assumption,
        p_load=FloatVector(value * scale.value for value in parsed.p_load_mw.values),
        q_load=FloatVector(value * scale.value for value in parsed.q_load_mvar.values),
    )


__all__ = ["CANONICAL_LOAD_SCALE", "build_canonical_operating_point"]
