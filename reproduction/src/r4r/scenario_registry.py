"""Deterministic, solve-free scenario registry for later runner rounds."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Sequence

from r4r.errors import ValidationError
from r4r.serialization.canonical_json import canonical_bytes
from r4r.types import Identifier, QAssumption


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    """One registered experiment combination; no solver result is attached."""

    scenario_id: Identifier
    capacity_model: Identifier
    network_model: Identifier
    objective: Identifier
    reporter_id: Identifier
    stress_multiplier: float
    q_assumption: QAssumption
    participant_registry_hash: str | None = None
    scientific_status: str = "UNRESOLVED"
    unresolved_fields: tuple[str, ...] = ()
    executable: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("scenario_id", self.scenario_id),
            ("capacity_model", self.capacity_model),
            ("network_model", self.network_model),
            ("objective", self.objective),
            ("reporter_id", self.reporter_id),
        ):
            if not isinstance(value, Identifier):
                raise ValidationError(f"{name} must be an Identifier")
        if not isinstance(self.q_assumption, QAssumption):
            raise ValidationError("q_assumption must be a registered Q token")
        if self.stress_multiplier <= 0:
            raise ValidationError("stress_multiplier must be strictly positive")
        if self.participant_registry_hash is not None:
            if len(self.participant_registry_hash) != 64 or any(c not in "0123456789abcdef" for c in self.participant_registry_hash):
                raise ValidationError("participant_registry_hash must be a lowercase SHA-256 digest")
        if self.scientific_status not in {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}:
            raise ValidationError("scientific_status is not registered")
        if self.executable != (self.scientific_status == "AUTHOR_FROZEN"):
            raise ValidationError("executable must agree with scientific_status")


@dataclass(frozen=True, slots=True)
class ScenarioRegistry:
    """Stable ordered cases and a hash of the complete registry payload."""

    cases: tuple[ScenarioCase, ...]
    registry_hash: str

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValidationError("scenario registry must not be empty")
        ids = [case.scenario_id.value for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValidationError("scenario IDs must be unique")
        if len(self.registry_hash) != 64 or any(c not in "0123456789abcdef" for c in self.registry_hash):
            raise ValidationError("registry_hash must be a lowercase SHA-256 digest")


def build_scenario_registry(
    *,
    capacity_models: Sequence[Identifier],
    network_models: Sequence[Identifier],
    objectives: Sequence[Identifier],
    reporter_ids: Sequence[Identifier],
    stress_multipliers: Sequence[float],
    q_assumptions: Sequence[QAssumption],
    participant_registry_hash: str | None = None,
    participant_scientific_status: str = "UNRESOLVED",
    network_scientific_status: str = "UNRESOLVED",
    objective_scientific_status: str = "UNRESOLVED",
    capacity_scientific_status: str = "UNRESOLVED",
    request_scientific_status: str = "UNRESOLVED",
    q_scientific_status: str = "UNRESOLVED",
) -> ScenarioRegistry:
    """Generate the Cartesian registry in caller-declared dimension order."""
    dimensions = {
        "capacity_models": tuple(capacity_models),
        "network_models": tuple(network_models),
        "objectives": tuple(objectives),
        "reporter_ids": tuple(reporter_ids),
        "stress_multipliers": tuple(float(value) for value in stress_multipliers),
        "q_assumptions": tuple(q_assumptions),
    }
    if any(not values for values in dimensions.values()):
        raise ValidationError("all scenario registry dimensions must be non-empty")
    for name in ("capacity_models", "network_models", "objectives", "reporter_ids"):
        if any(not isinstance(value, Identifier) for value in dimensions[name]):
            raise ValidationError(f"{name} must contain only Identifiers")
    if any(not isinstance(value, QAssumption) for value in dimensions["q_assumptions"]):
        raise ValidationError("q_assumptions must contain registered Q tokens")
    if any(value <= 0 for value in dimensions["stress_multipliers"]):
        raise ValidationError("stress multipliers must be strictly positive")
    statuses = {
        "participant_registry": participant_scientific_status,
        "network_model": network_scientific_status,
        "objective_definition": objective_scientific_status,
        "capacity_specification": capacity_scientific_status,
        "request_specification": request_scientific_status,
        "q_specification": q_scientific_status,
    }
    if any(value not in {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"} for value in statuses.values()):
        raise ValidationError("scientific status is not registered")
    unresolved_fields = tuple(name for name, value in statuses.items() if value != "AUTHOR_FROZEN")
    overall_status = "AUTHOR_FROZEN" if not unresolved_fields else (
        "UNRESOLVED" if any(value == "UNRESOLVED" for value in statuses.values()) else "CANDIDATE"
    )
    registry_token = participant_registry_hash[:12] if participant_registry_hash else "registry_unresolved"
    cases: list[ScenarioCase] = []
    for index, (capacity, network, objective, reporter, stress, q_assumption) in enumerate(
        product(
            dimensions["capacity_models"],
            dimensions["network_models"],
            dimensions["objectives"],
            dimensions["reporter_ids"],
            dimensions["stress_multipliers"],
            dimensions["q_assumptions"],
        ),
        1,
    ):
        scenario_id = Identifier(
            f"sc{index:06d}_{registry_token}_{capacity.value}_{network.value}_{objective.value}_"
            f"{reporter.value}_s{stress:g}_{q_assumption.value}"
        )
        cases.append(
            ScenarioCase(
                scenario_id=scenario_id,
                capacity_model=capacity,
                network_model=network,
                objective=objective,
                reporter_id=reporter,
                stress_multiplier=stress,
                q_assumption=q_assumption,
                participant_registry_hash=participant_registry_hash,
                scientific_status=overall_status,
                unresolved_fields=unresolved_fields,
                executable=(overall_status == "AUTHOR_FROZEN"),
            )
        )
    payload = [
        {
            "scenario_id": case.scenario_id.value,
            "capacity_model": case.capacity_model.value,
            "network_model": case.network_model.value,
            "objective": case.objective.value,
            "reporter_id": case.reporter_id.value,
            "stress_multiplier": case.stress_multiplier,
            "q_assumption": case.q_assumption.value,
            "participant_registry_hash": case.participant_registry_hash,
            "scientific_status": case.scientific_status,
            "unresolved_fields": list(case.unresolved_fields),
            "executable": case.executable,
        }
        for case in cases
    ]
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return ScenarioRegistry(tuple(cases), digest)
