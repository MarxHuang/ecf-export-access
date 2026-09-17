"""Reference/report request construction for the IEEE 141 domain slice.

The manuscript specifies the base request as ``0.08 + 0.10 * normalized
bus-load``, clipped to ``0.05--0.20 MW`` *before* stress scaling.  This module
implements that sequence while requiring the caller to provide the already
normalized load vector.  The normalization policy itself is intentionally not
reconstructed from historical code: the paper source and the archived R4R
notes disagree on that detail, so silently choosing one here would change the
scientific experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from r4r.errors import ValidationError
from r4r.models.scenarios import RequestScenario
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, Identifier, RequestKind, Sha256
from r4r.types.scalars import FiniteFloat


@dataclass(frozen=True, slots=True)
class RequestConstruction:
    """All stages of one request construction, including explicit clipping."""

    normalized_bus_load: FloatVector
    unclipped_base_mw: FloatVector
    clipped_base_mw: FloatVector
    actual_request_mw: FloatVector
    clipped_indices: tuple[int, ...]
    stress_multiplier: FiniteFloat
    clip_lower_mw: FiniteFloat
    clip_upper_mw: FiniteFloat
    intercept_mw: FiniteFloat = FiniteFloat(0.08)
    load_factor: FiniteFloat = FiniteFloat(0.10)

    def __post_init__(self) -> None:
        lengths = {
            len(self.normalized_bus_load.values),
            len(self.unclipped_base_mw.values),
            len(self.clipped_base_mw.values),
            len(self.actual_request_mw.values),
        }
        if len(lengths) != 1:
            raise ValidationError("request construction vectors must have equal length")
        if tuple(sorted(set(self.clipped_indices))) != self.clipped_indices:
            raise ValidationError("clipped_indices must be sorted and unique")
        if any(index < 0 or index >= len(self.unclipped_base_mw.values) for index in self.clipped_indices):
            raise ValidationError("clipped_indices contains an invalid vector index")
        for name, value in (
            ("stress_multiplier", self.stress_multiplier),
            ("clip_lower_mw", self.clip_lower_mw),
            ("clip_upper_mw", self.clip_upper_mw),
            ("intercept_mw", self.intercept_mw),
            ("load_factor", self.load_factor),
        ):
            if not isinstance(value, FiniteFloat):
                raise ValidationError(f"{name} must be a FiniteFloat")
        if self.clip_lower_mw.value < 0.0 or self.clip_upper_mw.value < self.clip_lower_mw.value:
            raise ValidationError("request clipping bounds must satisfy 0 <= lower <= upper")
        if self.stress_multiplier.value <= 0.0:
            raise ValidationError("stress_multiplier must be strictly positive")
        expected_unclipped = FloatVector(
            self.intercept_mw.value + self.load_factor.value * load
            for load in self.normalized_bus_load.values
        )
        if expected_unclipped != self.unclipped_base_mw:
            raise ValidationError("unclipped_base_mw does not match intercept/load_factor formula")
        expected_clipped = FloatVector(
            min(max(value, self.clip_lower_mw.value), self.clip_upper_mw.value)
            for value in self.unclipped_base_mw.values
        )
        if expected_clipped != self.clipped_base_mw:
            raise ValidationError("clipped_base_mw does not match explicit clipping bounds")
        expected_indices = tuple(
            index
            for index, (raw, clipped) in enumerate(
                zip(self.unclipped_base_mw.values, self.clipped_base_mw.values)
            )
            if raw != clipped
        )
        if expected_indices != self.clipped_indices:
            raise ValidationError("clipped_indices does not match explicit clipping")
        expected_actual = FloatVector(
            self.stress_multiplier.value * value
            for value in self.clipped_base_mw.values
        )
        if expected_actual != self.actual_request_mw:
            raise ValidationError("actual_request_mw does not match stress multiplier")

    def to_scenario(
        self,
        scenario_id: Identifier,
        request_kind: RequestKind,
        reporter_id: Identifier | None = None,
    ) -> RequestScenario:
        """Bind the actual vector to the registered request entity.

        The returned scenario is the value consumed by later allocation code;
        this object must be retained alongside it whenever clipping provenance
        is needed for an audit.
        """
        return RequestScenario(scenario_id, request_kind, self.actual_request_mw, reporter_id)

    def to_json(self) -> dict[str, object]:
        """Serialize every construction stage, including explicit clipping."""
        return {
            "normalized_bus_load": self.normalized_bus_load.to_json(),
            "unclipped_base_mw": self.unclipped_base_mw.to_json(),
            "clipped_base_mw": self.clipped_base_mw.to_json(),
            "actual_request_mw": self.actual_request_mw.to_json(),
            "clipped_indices": list(self.clipped_indices),
            "stress_multiplier": self.stress_multiplier.to_json(),
            "clip_lower_mw": self.clip_lower_mw.to_json(),
            "clip_upper_mw": self.clip_upper_mw.to_json(),
            "intercept_mw": self.intercept_mw.to_json(),
            "load_factor": self.load_factor.to_json(),
        }

    @property
    def construction_hash(self) -> Sha256:
        """Hash every request-construction stage used by downstream rules."""
        return Sha256(canonical_hash({
            "serialization_id": "request_construction.v1",
            "fields": self.to_json(),
        }))


def construct_requests(
    normalized_bus_load: Iterable[float],
    *,
    intercept: float = 0.08,
    load_factor: float = 0.10,
    clip_lower_mw: float = 0.05,
    clip_upper_mw: float = 0.20,
    stress_multiplier: float = 1.0,
) -> RequestConstruction:
    """Construct base and stressed requests from normalized bus loads.

    Clipping is explicit in the returned ``clipped_indices`` and is applied
    before the stress multiplier, matching the manuscript's case-study text.
    No historical rank normalization, padding, truncation, or implicit
    clipping is performed.
    """
    loads = FloatVector(normalized_bus_load)
    intercept_value = FiniteFloat(intercept)
    factor_value = FiniteFloat(load_factor)
    lower_value = FiniteFloat(clip_lower_mw)
    upper_value = FiniteFloat(clip_upper_mw)
    stress_value = FiniteFloat(stress_multiplier)
    if lower_value.value < 0 or upper_value.value < lower_value.value:
        raise ValidationError("request clipping bounds must satisfy 0 <= lower <= upper")
    if stress_value.value <= 0:
        raise ValidationError("stress_multiplier must be strictly positive")

    unclipped = FloatVector(intercept_value.value + factor_value.value * load for load in loads.values)
    clipped_values: list[float] = []
    clipped_indices: list[int] = []
    for index, value in enumerate(unclipped.values):
        clipped = min(max(value, lower_value.value), upper_value.value)
        clipped_values.append(clipped)
        if clipped != value:
            clipped_indices.append(index)
    clipped_base = FloatVector(clipped_values)
    actual = FloatVector(stress_value.value * value for value in clipped_base.values)
    return RequestConstruction(
        normalized_bus_load=loads,
        unclipped_base_mw=unclipped,
        clipped_base_mw=clipped_base,
        actual_request_mw=actual,
        clipped_indices=tuple(clipped_indices),
        stress_multiplier=stress_value,
        clip_lower_mw=lower_value,
        clip_upper_mw=upper_value,
        intercept_mw=intercept_value,
        load_factor=factor_value,
    )


__all__ = ["RequestConstruction", "construct_requests"]
