"""Typed design contract for simultaneous multi-reporter scenarios."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import FiniteFloat, FloatVector, Identifier, IdentifierVector, Sha256


SPATIAL_PATTERNS = {"LOCAL", "DISPERSED"}
SELECTION_METHOD_ID = "TOPOLOGY_ONLY_NEAREST_CENTER_AND_FARTHEST_POINT_V1"


def _finite_nonnegative_vector(value: FloatVector, field: str) -> None:
    if not isinstance(value, FloatVector):
        raise ValidationError(f"{field} must be a FloatVector")
    if any(item < 0.0 for item in value.values):
        raise ValidationError(f"{field} must be nonnegative")


@dataclass(frozen=True, slots=True)
class MultiReporterScenario(ContractModel):
    """A frozen request pair whose changed coordinates are one coalition."""

    scenario_id: Identifier
    participant_ids: IdentifierVector
    participant_bus_ids: tuple[int, ...]
    coalition_ids: IdentifierVector
    coalition_bus_ids: tuple[int, ...]
    spatial_pattern: str
    report_multiplier: FiniteFloat
    reference_request_mw: FloatVector
    reported_request_mw: FloatVector
    capacity_mw: FloatVector
    source_scenario_id: Identifier
    source_envelope_hash: Sha256
    topology_hash: Sha256
    selection_method_id: Identifier
    mean_pairwise_distance_edges: FiniteFloat
    minimum_pairwise_distance_edges: int
    maximum_pairwise_distance_edges: int
    serialization_id = "simultaneous_multi_reporter_scenario.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, Identifier) or not isinstance(self.source_scenario_id, Identifier):
            raise ValidationError("scenario identifiers must be Identifier values")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if len(self.participant_bus_ids) != len(self.participant_ids.values):
            raise ValidationError("participant bus mapping must align with participant IDs")
        if any(isinstance(bus, bool) or not isinstance(bus, int) or bus <= 0 for bus in self.participant_bus_ids):
            raise ValidationError("participant bus IDs must be positive integers")
        if len(set(self.participant_bus_ids)) != len(self.participant_bus_ids):
            raise ValidationError("participant bus IDs must be unique")
        if not isinstance(self.coalition_ids, IdentifierVector) or len(self.coalition_ids.values) < 2:
            raise ValidationError("coalition must contain at least two participants")
        if any(item not in self.participant_ids.values for item in self.coalition_ids.values):
            raise ValidationError("coalition participant is absent from participant registry")
        expected_buses = tuple(self.participant_bus_ids[self.participant_ids.values.index(item)] for item in self.coalition_ids.values)
        if self.coalition_bus_ids != expected_buses:
            raise ValidationError("coalition bus IDs must follow the frozen participant mapping")
        if self.spatial_pattern not in SPATIAL_PATTERNS:
            raise ValidationError("spatial pattern is not registered")
        if not isinstance(self.report_multiplier, FiniteFloat) or self.report_multiplier.value <= 1.0:
            raise ValidationError("report multiplier must be finite and greater than one")
        for name in ("reference_request_mw", "reported_request_mw", "capacity_mw"):
            _finite_nonnegative_vector(getattr(self, name), name)
            if len(getattr(self, name).values) != len(self.participant_ids.values):
                raise ValidationError(f"{name} must align with participant IDs")
        if any(value <= 0.0 for value in self.capacity_mw.values):
            raise ValidationError("capacity values must be positive")
        coalition = set(self.coalition_ids.values)
        changed: list[Identifier] = []
        for participant_id, reference, reported in zip(
            self.participant_ids.values, self.reference_request_mw.values, self.reported_request_mw.values,
        ):
            if not math.isclose(reference, reported, rel_tol=0.0, abs_tol=1.0e-12):
                changed.append(participant_id)
            if participant_id in coalition:
                expected = self.report_multiplier.value * reference
                if not math.isclose(reported, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
                    raise ValidationError("each coalition coordinate must use the frozen report multiplier")
            elif not math.isclose(reference, reported, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValidationError("noncoalition request coordinates must remain unchanged")
        if tuple(changed) != tuple(self.coalition_ids.values):
            raise ValidationError("changed coordinates must equal coalition IDs in participant order")
        if not isinstance(self.source_envelope_hash, Sha256) or not isinstance(self.topology_hash, Sha256):
            raise ValidationError("source and topology hashes must be SHA-256")
        if not isinstance(self.selection_method_id, Identifier) or self.selection_method_id.value != SELECTION_METHOD_ID:
            raise ValidationError("selection method is not the frozen topology-only method")
        if not isinstance(self.mean_pairwise_distance_edges, FiniteFloat) or self.mean_pairwise_distance_edges.value <= 0.0:
            raise ValidationError("mean pairwise distance must be positive")
        if (
            isinstance(self.minimum_pairwise_distance_edges, bool)
            or isinstance(self.maximum_pairwise_distance_edges, bool)
            or not isinstance(self.minimum_pairwise_distance_edges, int)
            or not isinstance(self.maximum_pairwise_distance_edges, int)
            or self.minimum_pairwise_distance_edges <= 0
            or self.maximum_pairwise_distance_edges < self.minimum_pairwise_distance_edges
        ):
            raise ValidationError("pairwise topology distance bounds are invalid")

    @property
    def scenario_hash(self) -> Sha256:
        return Sha256(canonical_hash({"serialization_id": self.serialization_id, "fields": self.to_json()}))


def topology_distances(
    branch_rows: Sequence[Mapping[str, Any]], participant_bus_ids: Sequence[int],
) -> dict[tuple[int, int], int]:
    """Return participant shortest-path distances on a connected radial graph."""

    adjacency: dict[int, set[int]] = {}
    edge_count = 0
    for row in branch_rows:
        try:
            left = int(row["from_bus_id"])
            right = int(row["to_bus_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("topology branch row is invalid") from exc
        if left <= 0 or right <= 0 or left == right:
            raise ValidationError("topology branch endpoints are invalid")
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        edge_count += 1
    buses = tuple(int(bus) for bus in participant_bus_ids)
    if len(set(buses)) != len(buses) or any(bus not in adjacency for bus in buses):
        raise ValidationError("participant buses must be unique members of the topology")
    if edge_count != len(adjacency) - 1:
        raise ValidationError("multi-reporter topology selection requires a radial tree")
    result: dict[tuple[int, int], int] = {}
    for source in buses:
        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            node, distance = queue.popleft()
            if node in buses:
                result[(source, node)] = distance
            for neighbor in sorted(adjacency[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        if len(visited) != len(adjacency):
            raise ValidationError("topology must be connected")
    return result


def _distance_stats(indices: Sequence[int], buses: Sequence[int], distances: Mapping[tuple[int, int], int]) -> dict[str, float | int]:
    pairwise = [distances[(buses[left], buses[right])] for pos, left in enumerate(indices) for right in indices[pos + 1:]]
    if not pairwise:
        raise ValidationError("coalition distance statistics require at least two members")
    return {
        "mean": math.fsum(pairwise) / len(pairwise),
        "minimum": min(pairwise),
        "maximum": max(pairwise),
    }


def select_topology_coalition(
    participant_ids: Sequence[str], participant_bus_ids: Sequence[int], branch_rows: Sequence[Mapping[str, Any]],
    coalition_size: int, spatial_pattern: str,
) -> tuple[list[int], dict[str, float | int]]:
    """Select a coalition without reading any allocation or AC result.

    LOCAL minimizes mean pairwise tree distance among each participant and its
    nearest neighbors.  DISPERSED uses deterministic farthest-point sampling,
    initialized by the most distant participant pair.
    """

    ids = tuple(str(item) for item in participant_ids)
    buses = tuple(int(item) for item in participant_bus_ids)
    if len(ids) != len(buses) or len(set(ids)) != len(ids):
        raise ValidationError("participant IDs and bus IDs must form a unique aligned mapping")
    if isinstance(coalition_size, bool) or not isinstance(coalition_size, int) or coalition_size < 2 or coalition_size > len(ids):
        raise ValidationError("coalition size must be between two and participant count")
    if spatial_pattern not in SPATIAL_PATTERNS:
        raise ValidationError("spatial pattern is not registered")
    distances = topology_distances(branch_rows, buses)
    if spatial_pattern == "LOCAL":
        candidates: list[tuple[tuple[float, int, tuple[str, ...]], list[int], dict[str, float | int]]] = []
        for center in range(len(ids)):
            nearest = sorted(range(len(ids)), key=lambda index: (distances[(buses[center], buses[index])], ids[index]))[:coalition_size]
            chosen = sorted(nearest)
            stats = _distance_stats(chosen, buses, distances)
            key = (float(stats["mean"]), int(stats["maximum"]), tuple(ids[index] for index in chosen))
            candidates.append((key, chosen, stats))
        _, selected, stats = min(candidates, key=lambda item: item[0])
        return selected, stats
    farthest_pairs = []
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            farthest_pairs.append((distances[(buses[left], buses[right])], ids[left], ids[right], left, right))
    maximum = max(item[0] for item in farthest_pairs)
    _, _, _, left, right = min((item for item in farthest_pairs if item[0] == maximum), key=lambda item: (item[1], item[2]))
    selected = [left, right]
    while len(selected) < coalition_size:
        remaining = [index for index in range(len(ids)) if index not in selected]
        ranked = sorted(
            remaining,
            key=lambda index: (
                -min(distances[(buses[index], buses[chosen])] for chosen in selected),
                -sum(distances[(buses[index], buses[chosen])] for chosen in selected),
                ids[index],
            ),
        )
        selected.append(ranked[0])
    selected.sort()
    return selected, _distance_stats(selected, buses, distances)


__all__ = [
    "MultiReporterScenario", "SELECTION_METHOD_ID", "SPATIAL_PATTERNS",
    "select_topology_coalition", "topology_distances",
]
