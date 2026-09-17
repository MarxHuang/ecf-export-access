"""Explicit participant selection and alignment for the IEEE 141 case.

This module deliberately does not infer the scientific participant list,
capacity model, or Q semantics. Callers must provide the selected bus order
and capacities explicitly; the function only validates and materializes the
typed participant objects needed by later rounds.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from r4r.errors import ReferenceError, ValidationError
from r4r.models import Participant, ParticipantSet
from r4r.serialization.canonical_json import canonical_bytes
from r4r.types import Identifier, IdentifierVector, NonNegativeFloat, ObjectReference


@dataclass(frozen=True, slots=True)
class ParticipantSelection:
    """Validated participant objects plus their stable bus alignment."""

    participants: tuple[Participant, ...]
    participant_set: ParticipantSet
    reporter_ids: IdentifierVector
    bus_to_participant: tuple[tuple[int, Identifier], ...]
    alignment_hash: str
    scientific_status: str = "UNRESOLVED"
    declared_complete: str = "UNKNOWN"
    ordering_rule: str = "EXPLICIT_INPUT_ORDER"
    co_location_policy: str = "UNRESOLVED"
    slack_bus_policy: str = "EXCLUDE_SLACK"
    network_bus_ids: tuple[int, ...] = ()
    slack_bus_id: int | None = None
    network_hash: str = ""

    def __post_init__(self) -> None:
        if not self.participants:
            raise ValidationError("participant selection must not be empty")
        if len(self.participants) != len(self.participant_set.participant_ids.values):
            raise ValidationError("participant and participant-set lengths must match")
        if len(self.bus_to_participant) != len(self.participants):
            raise ValidationError("bus alignment length must match participant count")
        expected = tuple((int(p.bus_id.object_id.value), p.participant_id) for p in self.participants)
        if self.bus_to_participant != expected:
            raise ValidationError("bus alignment must preserve participant order")
        if len(self.alignment_hash) != 64 or any(c not in "0123456789abcdef" for c in self.alignment_hash):
            raise ValidationError("alignment_hash must be a lowercase SHA-256 digest")
        if self.scientific_status not in {"UNRESOLVED", "CANDIDATE", "AUTHOR_FROZEN"}:
            raise ValidationError("scientific_status is not registered")
        if self.declared_complete not in {"UNKNOWN", "INCOMPLETE", "COMPLETE"}:
            raise ValidationError("declared_complete is not registered")
        if self.co_location_policy not in {"ONE_PARTICIPANT_PER_BUS", "COLOCATION_ALLOWED", "UNRESOLVED"}:
            raise ValidationError("co-location policy is not registered")
        if self.slack_bus_policy not in {"EXCLUDE_SLACK", "ALLOW_SLACK", "UNRESOLVED"}:
            raise ValidationError("slack-bus policy is not registered")
        if not self.network_bus_ids or len(set(self.network_bus_ids)) != len(self.network_bus_ids):
            raise ValidationError("network_bus_ids must be a non-empty unique tuple")
        if self.slack_bus_id not in self.network_bus_ids:
            raise ReferenceError("slack_bus_id must reference network_bus_ids")
        if len(self.network_hash) != 64 or any(c not in "0123456789abcdef" for c in self.network_hash):
            raise ValidationError("network_hash must be a lowercase SHA-256 digest")
        if not self.ordering_rule:
            raise ValidationError("ordering and co-location policies are required")


def select_deterministic_nonzero_load_buses(
    *,
    network_bus_ids: Sequence[int],
    slack_bus_id: int,
    load_mw_by_bus: Mapping[int, float],
    branch_endpoints: Sequence[tuple[int, int]],
    participant_count: int,
) -> tuple[int, ...]:
    """Select the frozen deterministic eligible downstream bus set.

    Eligibility is explicit: a bus must be a non-slack bus reachable from the
    reference bus in the supplied network graph and have strictly positive
    load.  The stable bus-ID ordering is applied before taking the requested
    count.  No capacity, request, or reactive-power value is inferred here.
    """
    network = tuple(network_bus_ids)
    if not network or len(set(network)) != len(network):
        raise ValidationError("network bus IDs must be unique and non-empty")
    if isinstance(participant_count, bool) or not isinstance(participant_count, int) or participant_count <= 0:
        raise ValidationError("participant_count must be a positive integer")
    if isinstance(slack_bus_id, bool) or not isinstance(slack_bus_id, int) or slack_bus_id not in set(network):
        raise ReferenceError("slack bus must reference the network")
    if any(isinstance(bus, bool) or not isinstance(bus, int) or bus <= 0 for bus in network):
        raise ValidationError("network bus IDs must be positive integers")
    if any(bus not in load_mw_by_bus for bus in network):
        missing = [bus for bus in network if bus not in load_mw_by_bus]
        raise ReferenceError(f"missing load values for network buses: {missing}")
    for bus in network:
        value = load_mw_by_bus[bus]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValidationError(f"load at bus {bus} must be a finite nonnegative scalar")
    adjacency: dict[int, set[int]] = {bus: set() for bus in network}
    network_set = set(network)
    for edge in branch_endpoints:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValidationError("branch_endpoints must contain (from_bus, to_bus) pairs")
        from_bus, to_bus = edge
        if (
            isinstance(from_bus, bool) or isinstance(to_bus, bool)
            or not isinstance(from_bus, int) or not isinstance(to_bus, int)
            or from_bus not in network_set or to_bus not in network_set
            or from_bus == to_bus
        ):
            raise ReferenceError("branch endpoint must reference two distinct network buses")
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)
    reachable: set[int] = {slack_bus_id}
    frontier = [slack_bus_id]
    while frontier:
        current = frontier.pop()
        for child in adjacency[current]:
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    eligible = tuple(
        bus for bus in sorted(network)
        if bus != slack_bus_id
        and bus in reachable
        and float(load_mw_by_bus[bus]) > 0.0
    )
    if len(eligible) < participant_count:
        raise ValidationError(
            f"eligible downstream nonzero-load buses ({len(eligible)}) are fewer than participant_count ({participant_count})"
        )
    return eligible[:participant_count]


def build_participant_selection(
    selected_bus_ids: Sequence[int],
    *,
    network_bus_ids: Sequence[int],
    slack_bus_id: int,
    load_mw_by_bus: Mapping[int, float],
    rating_mw_by_bus: Mapping[int, float],
    availability_mw_by_bus: Mapping[int, float] | None = None,
    rating_mw_by_participant: Sequence[float] | None = None,
    availability_mw_by_participant: Sequence[float] | None = None,
    reporter_bus_ids: Sequence[int] = (),
    participant_count: int | None = None,
    scientific_status: str = "UNRESOLVED",
    declared_complete: str = "UNKNOWN",
    ordering_rule: str = "EXPLICIT_INPUT_ORDER",
    co_location_policy: str = "UNRESOLVED",
    slack_bus_policy: str = "EXCLUDE_SLACK",
) -> ParticipantSelection:
    """Materialize a participant registry from explicit scientific inputs.

    ``selected_bus_ids`` is authoritative and order-sensitive. No ranking,
    padding, truncation, or fallback selection is performed here.
    """
    selected = tuple(selected_bus_ids)
    network = tuple(network_bus_ids)
    if not selected:
        raise ValidationError("selected_bus_ids must not be empty")
    if len(set(network)) != len(network):
        raise ValidationError("network bus IDs must be unique")
    all_ids = (*network, slack_bus_id, *selected)
    if any(isinstance(bus, bool) or not isinstance(bus, int) or bus <= 0 for bus in all_ids):
        raise ValidationError("bus IDs must be positive integers")
    if co_location_policy not in {"ONE_PARTICIPANT_PER_BUS", "COLOCATION_ALLOWED", "UNRESOLVED"}:
        raise ValidationError("co-location policy is not registered")
    if slack_bus_policy not in {"EXCLUDE_SLACK", "ALLOW_SLACK", "UNRESOLVED"}:
        raise ValidationError("slack-bus policy is not registered")
    if len(set(selected)) != len(selected) and co_location_policy != "COLOCATION_ALLOWED":
        raise ValidationError("selected participant buses must be unique")
    if (rating_mw_by_participant is None) != (availability_mw_by_participant is None):
        raise ValidationError(
            "participant-level rating and availability must be supplied together"
        )
    if rating_mw_by_participant is not None:
        if len(rating_mw_by_participant) != len(selected) or len(availability_mw_by_participant or ()) != len(selected):
            raise ValidationError("participant-level capacities must match selected participant order")
    if len(set(selected)) != len(selected) and rating_mw_by_participant is None:
        raise ValidationError(
            "co-located participants require explicit participant-level capacities"
        )
    if participant_count is not None and len(selected) != participant_count:
        raise ValidationError("selected participant count does not match participant_count")
    network_set = set(network)
    if slack_bus_id not in network_set:
        raise ReferenceError("slack bus must reference a network bus")
    if slack_bus_id in selected and slack_bus_policy != "ALLOW_SLACK":
        raise ValidationError("slack bus cannot be a participant")
    missing = [bus for bus in selected if bus not in network_set]
    if missing:
        raise ReferenceError(f"participant buses absent from network: {missing}")
    reporters = tuple(reporter_bus_ids)
    if len(set(reporters)) != len(reporters):
        raise ValidationError("reporter bus IDs must be unique")
    if any(bus not in selected for bus in reporters):
        raise ReferenceError("reporter bus must be in the participant selection")

    availability = rating_mw_by_bus if availability_mw_by_bus is None else availability_mw_by_bus
    participants: list[Participant] = []
    for index, bus_id in enumerate(selected, 1):
        if bus_id not in load_mw_by_bus:
            raise ReferenceError(f"missing load for participant bus {bus_id}")
        if (
            float(load_mw_by_bus[bus_id]) <= 0
            and not (bus_id == slack_bus_id and slack_bus_policy == "ALLOW_SLACK")
        ):
            raise ValidationError(f"participant bus {bus_id} must have positive load")
        if rating_mw_by_participant is None:
            if bus_id not in rating_mw_by_bus or bus_id not in availability:
                raise ReferenceError(f"missing capacity for participant bus {bus_id}")
            rating_value = rating_mw_by_bus[bus_id]
            availability_value = availability[bus_id]
        else:
            rating_value = rating_mw_by_participant[index - 1]
            availability_value = (availability_mw_by_participant or ())[index - 1]
        rating = NonNegativeFloat(rating_value)
        available = NonNegativeFloat(availability_value)
        if available.value > rating.value:
            raise ValidationError(f"availability exceeds rating at participant bus {bus_id}")
        participants.append(
            Participant(
                participant_id=Identifier(f"p{index:03d}"),
                bus_id=ObjectReference("ENT001", Identifier(str(bus_id))),
                rating_mw=rating,
                availability_mw=available,
            )
        )

    ids = IdentifierVector(p.participant_id for p in participants)
    reporter_id_set = IdentifierVector(
        participants[selected.index(bus_id)].participant_id for bus_id in reporters
    )
    alignment = tuple((p.bus_id.object_id.value, p.participant_id.value) for p in participants)
    # The registry hash is provenance for the complete participant definition,
    # not just bus/order alignment. Ratings and availabilities affect every
    # downstream capacity/request binding and must therefore be covered.
    payload = {
        "network_bus_ids": list(network),
        "slack_bus_id": slack_bus_id,
        "participants": [participant.to_json() for participant in participants],
        "reporters": reporter_id_set.to_json(),
        "ordering_rule": ordering_rule,
        "co_location_policy": co_location_policy,
        "slack_bus_policy": slack_bus_policy,
        "scientific_status": scientific_status,
        "declared_complete": declared_complete,
    }
    network_digest = hashlib.sha256(
        canonical_bytes({"network_bus_ids": list(network), "slack_bus_id": slack_bus_id})
    ).hexdigest()
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return ParticipantSelection(
        participants=tuple(participants),
        participant_set=ParticipantSet(ids, ids),
        reporter_ids=reporter_id_set,
        bus_to_participant=tuple((int(bus), Identifier(pid)) for bus, pid in alignment),
        alignment_hash=digest,
        scientific_status=scientific_status,
        declared_complete=declared_complete,
        ordering_rule=ordering_rule,
        co_location_policy=co_location_policy,
        slack_bus_policy=slack_bus_policy,
        network_bus_ids=network,
        slack_bus_id=slack_bus_id,
        network_hash=network_digest,
    )
