"""Optional engineering-MVA overlay contracts.

This module is deliberately independent from the manuscript-comparable
scenario-defined active-MW screen.  It does not read or mutate MATPOWER
``RATE_A/B/C`` fields, infer ampacity from ``R/X`` or historical flows, or
promote an overlay to physical evidence.  It only provides a typed,
provenance-bound diagnostic lane for the case in which independent engineering
rating data are supplied later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import decode_mapping
from r4r.types import (
    Identifier,
    IdentifierVector,
    MvaScreenStatus,
    PhaseModel,
    RatingSourceType,
    RatingStatus,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


_SQRT3 = math.sqrt(3.0)
_RATING_FIELDS = (
    "branch_id",
    "from_bus_id",
    "to_bus_id",
    "voltage_ll_kv",
    "continuous_ampacity_a",
    "circuit_count",
    "derating_factor",
    "phase_model",
    "rate_a_mva",
    "rate_b_mva",
    "rate_c_mva",
    "source_record_id",
    "mapping_method",
    "status",
)


def _require_identifier(value: object, field: str) -> Identifier:
    if not isinstance(value, Identifier):
        raise ValidationError(f"{field} must be an Identifier")
    return value


def _require_positive(value: FiniteFloat | None, field: str) -> None:
    if value is None or value.value <= 0:
        raise ValidationError(f"{field} must be a positive finite value")


def _optional_positive(value: FiniteFloat | None, field: str) -> None:
    if value is not None and value.value <= 0:
        raise ValidationError(f"{field} must be positive when supplied")


@dataclass(frozen=True, slots=True)
class EngineeringBranchRating(ContractModel):
    """One independently sourced continuous rating for one mapped branch.

    ``rate_b_mva`` and ``rate_c_mva`` are intentionally unavailable in this
    contract.  A supplied continuous rating must be reproducible from the
    explicit voltage/ampacity/circuit/derating fields and a source record.
    """

    branch_id: Identifier
    from_bus_id: Identifier
    to_bus_id: Identifier
    voltage_ll_kv: FiniteFloat | None
    continuous_ampacity_a: FiniteFloat | None
    circuit_count: int | None
    derating_factor: FiniteFloat | None
    phase_model: PhaseModel
    rate_a_mva: FiniteFloat | None
    rate_b_mva: None
    rate_c_mva: None
    source_record_id: Identifier | None
    mapping_method: Identifier
    status: RatingStatus
    serialization_id = "engineering_branch_rating.v1"

    def __post_init__(self) -> None:
        _require_identifier(self.branch_id, "branch_id")
        _require_identifier(self.from_bus_id, "from_bus_id")
        _require_identifier(self.to_bus_id, "to_bus_id")
        if self.from_bus_id == self.to_bus_id:
            raise ValidationError("branch endpoints must be distinct")
        if not isinstance(self.phase_model, PhaseModel):
            raise ValidationError("phase_model must be an explicit PhaseModel")
        if self.rate_b_mva is not None or self.rate_c_mva is not None:
            raise ValidationError("RATE_B/RATE_C remain unavailable in this overlay")
        if not isinstance(self.mapping_method, Identifier):
            raise ValidationError("mapping_method must be an Identifier")
        if not isinstance(self.status, RatingStatus):
            raise ValidationError("status must be a RatingStatus")
        if self.circuit_count is not None:
            if isinstance(self.circuit_count, bool) or not isinstance(self.circuit_count, int) or self.circuit_count <= 0:
                raise ValidationError("circuit_count must be a positive integer when supplied")
        _optional_positive(self.voltage_ll_kv, "voltage_ll_kv")
        _optional_positive(self.continuous_ampacity_a, "continuous_ampacity_a")
        if self.derating_factor is not None and not (0 < self.derating_factor.value <= 1):
            raise ValidationError("derating_factor must be in (0, 1]")
        if self.source_record_id is not None and not isinstance(self.source_record_id, Identifier):
            raise ValidationError("source_record_id must be an Identifier or null")

        engineering_inputs = (
            self.voltage_ll_kv,
            self.continuous_ampacity_a,
            self.circuit_count,
            self.derating_factor,
            self.source_record_id,
        )
        complete = all(value is not None for value in engineering_inputs)
        if self.status is RatingStatus.AVAILABLE:
            if not complete or self.phase_model is not PhaseModel.THREE_PHASE_BALANCED:
                raise ValidationError("AVAILABLE rating requires complete balanced three-phase engineering inputs")
            _require_positive(self.rate_a_mva, "rate_a_mva")
            expected = _mva_from_ampacity(
                self.voltage_ll_kv.value,
                self.continuous_ampacity_a.value,
                self.circuit_count,
                self.derating_factor.value,
            )
            if not math.isclose(self.rate_a_mva.value, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValidationError("rate_a_mva must equal sqrt(3)*V_LL*I*circuit_count*derating/1000")
        elif self.status is RatingStatus.UNAVAILABLE_MISSING_ENGINEERING_RATING:
            if self.rate_a_mva is not None:
                raise ValidationError("missing engineering rating cannot carry rate_a_mva")
        else:  # defensive if the enum grows without updating this model
            raise ValidationError("unsupported rating status")

    @classmethod
    def from_ampacity(
        cls,
        *,
        branch_id: Identifier,
        from_bus_id: Identifier,
        to_bus_id: Identifier,
        voltage_ll_kv: FiniteFloat,
        continuous_ampacity_a: FiniteFloat,
        circuit_count: int,
        derating_factor: FiniteFloat,
        source_record_id: Identifier,
        mapping_method: Identifier,
        phase_model: PhaseModel = PhaseModel.THREE_PHASE_BALANCED,
    ) -> "EngineeringBranchRating":
        if phase_model is not PhaseModel.THREE_PHASE_BALANCED:
            raise ValidationError("sqrt(3) construction requires explicit balanced three-phase model")
        rate = _mva_from_ampacity(
            voltage_ll_kv.value,
            continuous_ampacity_a.value,
            circuit_count,
            derating_factor.value,
        )
        return cls(
            branch_id=branch_id,
            from_bus_id=from_bus_id,
            to_bus_id=to_bus_id,
            voltage_ll_kv=voltage_ll_kv,
            continuous_ampacity_a=continuous_ampacity_a,
            circuit_count=circuit_count,
            derating_factor=derating_factor,
            phase_model=phase_model,
            rate_a_mva=FiniteFloat(rate),
            rate_b_mva=None,
            rate_c_mva=None,
            source_record_id=source_record_id,
            mapping_method=mapping_method,
            status=RatingStatus.AVAILABLE,
        )


@dataclass(frozen=True, slots=True)
class EngineeringRatingOverlay(ContractModel):
    """A raw-case-preserving collection of independently sourced ratings."""

    overlay_id: Identifier
    network_hash: Sha256
    raw_case_hash: Sha256
    native_case_mutated: bool
    native_rating_used: bool
    rating_source_type: RatingSourceType
    source_references: IdentifierVector
    engineering_assumptions: tuple[str, ...]
    monitored_branch_ids: tuple[Identifier, ...]
    branch_ratings: tuple[EngineeringBranchRating, ...]
    diagnostic_only: bool
    primary_evidence_eligible: bool
    serialization_id = "engineering_rating_overlay.v1"

    def __post_init__(self) -> None:
        _require_identifier(self.overlay_id, "overlay_id")
        if not isinstance(self.network_hash, Sha256) or not isinstance(self.raw_case_hash, Sha256):
            raise ValidationError("overlay network and raw case hashes must be Sha256")
        if self.native_case_mutated is not False or self.native_rating_used is not False:
            raise ValidationError("engineering overlay cannot mutate or masquerade as native ratings")
        if not isinstance(self.rating_source_type, RatingSourceType):
            raise ValidationError("rating_source_type must be registered")
        if not isinstance(self.source_references, IdentifierVector) or not self.source_references.values:
            raise ValidationError("overlay requires at least one source reference")
        if not isinstance(self.engineering_assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.engineering_assumptions
        ):
            raise ValidationError("engineering_assumptions must be a non-empty tuple of strings")
        if not isinstance(self.monitored_branch_ids, tuple) or not self.monitored_branch_ids:
            raise ValidationError("monitored_branch_ids must be a non-empty tuple")
        if any(not isinstance(item, Identifier) for item in self.monitored_branch_ids):
            raise ValidationError("monitored_branch_ids must contain Identifier values")
        if len(set(self.monitored_branch_ids)) != len(self.monitored_branch_ids):
            raise ValidationError("monitored_branch_ids must be unique")
        if not isinstance(self.branch_ratings, tuple):
            raise ValidationError("branch_ratings must be a tuple")
        if any(not isinstance(item, EngineeringBranchRating) for item in self.branch_ratings):
            raise ValidationError("branch_ratings must contain EngineeringBranchRating values")
        branch_ids = [item.branch_id for item in self.branch_ratings]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValidationError("duplicate branch ratings are forbidden")
        if set(branch_ids) != set(self.monitored_branch_ids):
            raise ValidationError("every monitored branch must have exactly one rating row, including explicit unavailable rows")
        if self.diagnostic_only is not True or self.primary_evidence_eligible is not False:
            raise ValidationError("engineering overlay is diagnostic-only and cannot be primary evidence")

    def rating_for(self, branch_id: Identifier) -> EngineeringBranchRating | None:
        for rating in self.branch_ratings:
            if rating.branch_id == branch_id:
                return rating
        return None


@dataclass(frozen=True, slots=True)
class MvaTwoEndScreenResult(ContractModel):
    """Two-end apparent-power diagnostic for one AC branch state."""

    branch_id: Identifier
    p_from_mw: FiniteFloat
    q_from_mvar: FiniteFloat
    p_to_mw: FiniteFloat
    q_to_mvar: FiniteFloat
    s_from_mva: FiniteFloat
    s_to_mva: FiniteFloat
    s_max_mva: FiniteFloat
    rate_a_mva: FiniteFloat | None
    margin_mva: FiniteFloat | None
    from_is_limiting: bool
    to_is_limiting: bool
    rating_status: RatingStatus
    screen_status: MvaScreenStatus
    screen_pass: bool | None
    serialization_id = "mva_two_end_screen_result.v1"

    def __post_init__(self) -> None:
        _require_identifier(self.branch_id, "branch_id")
        for name in (
            "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar",
            "s_from_mva", "s_to_mva", "s_max_mva",
        ):
            if not isinstance(getattr(self, name), FiniteFloat):
                raise ValidationError(f"{name} must be a finite scalar")
        expected_from = math.hypot(self.p_from_mw.value, self.q_from_mvar.value)
        expected_to = math.hypot(self.p_to_mw.value, self.q_to_mvar.value)
        expected_max = max(expected_from, expected_to)
        if not math.isclose(self.s_from_mva.value, expected_from, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError("s_from_mva must be sqrt(p_from_mw^2 + q_from_mvar^2)")
        if not math.isclose(self.s_to_mva.value, expected_to, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError("s_to_mva must be sqrt(p_to_mw^2 + q_to_mvar^2)")
        if not math.isclose(self.s_max_mva.value, expected_max, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError("s_max_mva must equal max(s_from_mva, s_to_mva)")
        if self.from_is_limiting != (expected_from >= expected_to):
            raise ValidationError("from_is_limiting does not match the two-end maximum")
        if self.to_is_limiting != (expected_to >= expected_from):
            raise ValidationError("to_is_limiting does not match the two-end maximum")
        if not isinstance(self.rating_status, RatingStatus) or not isinstance(self.screen_status, MvaScreenStatus):
            raise ValidationError("rating and screen statuses must be registered enums")
        if self.rating_status is RatingStatus.AVAILABLE:
            _require_positive(self.rate_a_mva, "rate_a_mva")
            if self.margin_mva is None or not math.isclose(
                self.margin_mva.value, self.rate_a_mva.value - self.s_max_mva.value, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValidationError("margin_mva must equal rate_a_mva - s_max_mva")
            if self.screen_status not in (MvaScreenStatus.PASS, MvaScreenStatus.FAIL):
                raise ValidationError("available rating must have PASS or FAIL screen status")
            if self.screen_pass is not (self.screen_status is MvaScreenStatus.PASS):
                raise ValidationError("screen_pass does not match screen_status")
        else:
            if self.rate_a_mva is not None or self.margin_mva is not None:
                raise ValidationError("missing rating cannot carry rate or margin")
            if self.screen_status is not MvaScreenStatus.NOT_EVALUATED_MISSING_ENGINEERING_RATING or self.screen_pass is not None:
                raise ValidationError("missing rating must be NOT_EVALUATED and have null screen_pass")


@dataclass(frozen=True, slots=True)
class MvaTwoEndScreenAggregate(ContractModel):
    """Aggregate that keeps missing ratings in the denominator."""

    evaluated_branch_count: int
    unavailable_rating_count: int
    failing_branch_count: int
    first_failing_branch_id: Identifier | None
    maximum_violation_mva: FiniteFloat | None
    screen_status: MvaScreenStatus
    screen_pass: bool | None
    serialization_id = "mva_two_end_screen_aggregate.v1"

    def __post_init__(self) -> None:
        for name in ("evaluated_branch_count", "unavailable_rating_count", "failing_branch_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a nonnegative integer")
        if self.unavailable_rating_count > self.evaluated_branch_count or self.failing_branch_count > self.evaluated_branch_count:
            raise ValidationError("aggregate counts are inconsistent")
        if self.first_failing_branch_id is not None and not isinstance(self.first_failing_branch_id, Identifier):
            raise ValidationError("first_failing_branch_id must be an Identifier or null")
        if self.maximum_violation_mva is not None and self.maximum_violation_mva.value < 0:
            raise ValidationError("maximum_violation_mva must be nonnegative")
        if not isinstance(self.screen_status, MvaScreenStatus):
            raise ValidationError("screen_status must be registered")
        if self.unavailable_rating_count:
            if self.screen_status is not MvaScreenStatus.NOT_EVALUATED_MISSING_ENGINEERING_RATING or self.screen_pass is not None:
                raise ValidationError("missing ratings make the aggregate NOT_EVALUATED")
        elif self.evaluated_branch_count == 0:
            if self.screen_status is not MvaScreenStatus.NOT_EVALUATED_EMPTY or self.screen_pass is not None:
                raise ValidationError("empty aggregate must be NOT_EVALUATED_EMPTY")
        else:
            expected = self.failing_branch_count == 0
            if self.screen_status not in (MvaScreenStatus.PASS, MvaScreenStatus.FAIL) or self.screen_pass is not expected:
                raise ValidationError("aggregate pass state does not match branch failures")


def _mva_from_ampacity(voltage_ll_kv: float, ampacity_a: float, circuit_count: int, derating_factor: float) -> float:
    if voltage_ll_kv <= 0 or ampacity_a <= 0 or circuit_count <= 0 or not (0 < derating_factor <= 1):
        raise ValidationError("engineering inputs must be positive and derating must be in (0, 1]")
    return _SQRT3 * voltage_ll_kv * ampacity_a * circuit_count * derating_factor / 1000.0


def evaluate_mva_two_end(
    *,
    branch_id: Identifier,
    p_from_mw: FiniteFloat,
    q_from_mvar: FiniteFloat,
    p_to_mw: FiniteFloat,
    q_to_mvar: FiniteFloat,
    rating: EngineeringBranchRating | None,
    tolerance_mva: FiniteFloat = FiniteFloat(1e-9),
) -> MvaTwoEndScreenResult:
    """Evaluate ``max(|S_from|, |S_to|)`` without silently dropping missing ratings."""
    if tolerance_mva.value < 0:
        raise ValidationError("tolerance_mva must be nonnegative")
    s_from = math.hypot(p_from_mw.value, q_from_mvar.value)
    s_to = math.hypot(p_to_mw.value, q_to_mvar.value)
    s_max = max(s_from, s_to)
    from_limiting = s_from >= s_to
    to_limiting = s_to >= s_from
    common = dict(
        branch_id=branch_id,
        p_from_mw=p_from_mw,
        q_from_mvar=q_from_mvar,
        p_to_mw=p_to_mw,
        q_to_mvar=q_to_mvar,
        s_from_mva=FiniteFloat(s_from),
        s_to_mva=FiniteFloat(s_to),
        s_max_mva=FiniteFloat(s_max),
        from_is_limiting=from_limiting,
        to_is_limiting=to_limiting,
    )
    if rating is None or rating.status is not RatingStatus.AVAILABLE:
        return MvaTwoEndScreenResult(
            **common,
            rate_a_mva=None,
            margin_mva=None,
            rating_status=RatingStatus.UNAVAILABLE_MISSING_ENGINEERING_RATING,
            screen_status=MvaScreenStatus.NOT_EVALUATED_MISSING_ENGINEERING_RATING,
            screen_pass=None,
        )
    margin = rating.rate_a_mva.value - s_max  # type: ignore[union-attr]
    passed = margin >= -tolerance_mva.value
    return MvaTwoEndScreenResult(
        **common,
        rate_a_mva=rating.rate_a_mva,
        margin_mva=FiniteFloat(margin),
        rating_status=RatingStatus.AVAILABLE,
        screen_status=MvaScreenStatus.PASS if passed else MvaScreenStatus.FAIL,
        screen_pass=passed,
    )


def aggregate_mva_results(results: Iterable[MvaTwoEndScreenResult]) -> MvaTwoEndScreenAggregate:
    rows = tuple(results)
    missing = tuple(row for row in rows if row.rating_status is RatingStatus.UNAVAILABLE_MISSING_ENGINEERING_RATING)
    failing = tuple(row for row in rows if row.screen_status is MvaScreenStatus.FAIL)
    violations = tuple(-row.margin_mva.value for row in failing if row.margin_mva is not None and row.margin_mva.value < 0)
    maximum = FiniteFloat(max(violations)) if violations else None
    if missing:
        status = MvaScreenStatus.NOT_EVALUATED_MISSING_ENGINEERING_RATING
        screen_pass: bool | None = None
    elif not rows:
        status = MvaScreenStatus.NOT_EVALUATED_EMPTY
        screen_pass = None
    else:
        status = MvaScreenStatus.PASS if not failing else MvaScreenStatus.FAIL
        screen_pass = not failing
    return MvaTwoEndScreenAggregate(
        evaluated_branch_count=len(rows),
        unavailable_rating_count=len(missing),
        failing_branch_count=len(failing),
        first_failing_branch_id=failing[0].branch_id if failing else None,
        maximum_violation_mva=maximum,
        screen_status=status,
        screen_pass=screen_pass,
    )


def load_engineering_rating_overlay(
    payload: Mapping[str, Any],
    *,
    expected_network_hash: Sha256 | None = None,
    expected_raw_case_hash: Sha256 | None = None,
) -> EngineeringRatingOverlay:
    """Strictly load an overlay object; unknown fields and hash mismatches fail closed."""
    allowed_overlay = {
        "overlay_id", "network_hash", "raw_case_hash", "native_case_mutated", "native_rating_used",
        "rating_source_type", "source_references", "engineering_assumptions", "monitored_branch_ids", "branch_ratings",
        "diagnostic_only", "primary_evidence_eligible",
    }
    data = decode_mapping(payload, required=allowed_overlay, allowed=allowed_overlay)
    if expected_network_hash is not None and data["network_hash"] != str(expected_network_hash):
        raise ValidationError("overlay network_hash does not match expected network")
    if expected_raw_case_hash is not None and data["raw_case_hash"] != str(expected_raw_case_hash):
        raise ValidationError("overlay raw_case_hash does not match expected raw case")

    def ident(value: object, field: str) -> Identifier:
        if not isinstance(value, str):
            raise ValidationError(f"{field} must be a string identifier")
        return Identifier(value)

    def finite_or_none(value: object, field: str) -> FiniteFloat | None:
        return None if value is None else FiniteFloat(value)  # type: ignore[arg-type]

    def enum_value(enum_type: Any, value: object, field: str) -> Any:
        try:
            return enum_type(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field} is not a registered enum value") from exc

    if not isinstance(data["source_references"], (list, tuple)):
        raise ValidationError("source_references must be an array")
    if not isinstance(data["engineering_assumptions"], (list, tuple)):
        raise ValidationError("engineering_assumptions must be an array")
    if not isinstance(data["monitored_branch_ids"], (list, tuple)):
        raise ValidationError("monitored_branch_ids must be an array")
    if not isinstance(data["branch_ratings"], (list, tuple)):
        raise ValidationError("branch_ratings must be an array")
    if any(not isinstance(item, str) for item in data["engineering_assumptions"]):
        raise ValidationError("engineering_assumptions must contain strings")

    rows: list[EngineeringBranchRating] = []
    for raw in data["branch_ratings"]:
        if not isinstance(raw, Mapping):
            raise ValidationError("branch_ratings must contain objects")
        item = decode_mapping(raw, required=_RATING_FIELDS, allowed=_RATING_FIELDS)
        if item["rate_b_mva"] is not None or item["rate_c_mva"] is not None:
            raise ValidationError("RATE_B/RATE_C must remain null in the candidate overlay")
        rows.append(
            EngineeringBranchRating(
                branch_id=ident(item["branch_id"], "branch_id"),
                from_bus_id=ident(item["from_bus_id"], "from_bus_id"),
                to_bus_id=ident(item["to_bus_id"], "to_bus_id"),
                voltage_ll_kv=finite_or_none(item["voltage_ll_kv"], "voltage_ll_kv"),
                continuous_ampacity_a=finite_or_none(item["continuous_ampacity_a"], "continuous_ampacity_a"),
                circuit_count=item["circuit_count"],
                derating_factor=finite_or_none(item["derating_factor"], "derating_factor"),
                phase_model=enum_value(PhaseModel, item["phase_model"], "phase_model"),
                rate_a_mva=finite_or_none(item["rate_a_mva"], "rate_a_mva"),
                rate_b_mva=None,
                rate_c_mva=None,
                source_record_id=None if item["source_record_id"] is None else ident(item["source_record_id"], "source_record_id"),
                mapping_method=ident(item["mapping_method"], "mapping_method"),
                status=enum_value(RatingStatus, item["status"], "status"),
            )
        )
    return EngineeringRatingOverlay(
        overlay_id=ident(data["overlay_id"], "overlay_id"),
        network_hash=Sha256(data["network_hash"]),
        raw_case_hash=Sha256(data["raw_case_hash"]),
        native_case_mutated=data["native_case_mutated"],
        native_rating_used=data["native_rating_used"],
        rating_source_type=enum_value(RatingSourceType, data["rating_source_type"], "rating_source_type"),
        source_references=IdentifierVector(ident(item, "source_reference") for item in data["source_references"]),
        engineering_assumptions=tuple(data["engineering_assumptions"]),
        monitored_branch_ids=tuple(ident(item, "monitored_branch_id") for item in data["monitored_branch_ids"]),
        branch_ratings=tuple(rows),
        diagnostic_only=data["diagnostic_only"],
        primary_evidence_eligible=data["primary_evidence_eligible"],
    )


__all__ = [
    "EngineeringBranchRating",
    "EngineeringRatingOverlay",
    "MvaTwoEndScreenResult",
    "MvaTwoEndScreenAggregate",
    "aggregate_mva_results",
    "evaluate_mva_two_end",
    "load_engineering_rating_overlay",
]
