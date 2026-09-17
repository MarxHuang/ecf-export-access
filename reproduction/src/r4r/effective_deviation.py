"""Parameterized diagnostic implementation of the R5 deviation record.

The report equation is explicitly bound to EQ006 and the deviation metric
equation to EQ066.  The author-approved PAR062 policy is available through an
explicit builder; candidate policies remain explicit at call time.  The public
API retains signed, positive and absolute vectors without clipping and always
returns diagnostic-only data.
The private legacy EQ066 helper below is not exported and cannot be used as a
formal R5 scenario interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Sequence

from r4r.errors import ValidationError
from r4r.allocation_pipeline import ExplicitCapacityVector
from r4r.models.base import ContractModel
from r4r.request_contracts import AdmittedRequestVector, FrozenParameterMap, RawRequestVector, StrategicReportResult
from r4r.request_generation import RequestConstruction
from r4r.request_scenario import DiagnosticRequestScenarioBundle
from r4r.serialization import canonical_hash
from r4r.types import FloatVector, FiniteFloat, Identifier, IdentifierVector, QAssumption, Sha256, Unit


def _finite_vector(values: Sequence[float], *, field: str, nonnegative: bool = False) -> FloatVector:
    try:
        vector = FloatVector(values)
    except Exception as exc:
        raise ValidationError(f"{field} must be a finite numeric vector") from exc
    if nonnegative and any(value < 0.0 for value in vector.values):
        raise ValidationError(f"{field} must be nonnegative")
    return vector


@dataclass(frozen=True, slots=True)
class _LegacyEq066Evaluation(ContractModel):
    """Private pre-R5 EQ066 record retained only for migration diagnostics."""

    scenario_id: Identifier
    participant_ids: IdentifierVector
    participant_registry_hash: Sha256
    reporter_id: Identifier
    request_spec_id: Identifier
    capacity_spec_hash: Sha256
    reference_request_mw: FloatVector
    reported_request_mw: FloatVector
    actual_request_mw: FloatVector
    capacity_reference_mw: FloatVector
    signed_deviation_mw: FloatVector
    absolute_deviation_mw: FloatVector
    capacity_relative_deviation: FloatVector
    reporter_signed_deviation_mw: FiniteFloat
    reporter_absolute_deviation_mw: FiniteFloat
    reporter_relative_deviation: FiniteFloat
    threshold_mw: FiniteFloat
    effective_deviation_status: str
    equation_id: Identifier = Identifier("EQ066")
    threshold_parameter_id: Identifier = Identifier("PAR062")
    status: str = "DIAGNOSTIC_ONLY"
    serialization_id = "effective_deviation_evaluation.v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("scenario_id", self.scenario_id),
            ("reporter_id", self.reporter_id),
            ("request_spec_id", self.request_spec_id),
            ("equation_id", self.equation_id),
            ("threshold_parameter_id", self.threshold_parameter_id),
        ):
            if not isinstance(value, Identifier):
                raise ValidationError(f"{name} must be Identifier")
        if self.equation_id != Identifier("EQ066"):
            raise ValidationError("effective deviation must be bound to EQ066")
        if self.threshold_parameter_id != Identifier("PAR062"):
            raise ValidationError("effective deviation threshold must be bound to PAR062")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if not isinstance(self.participant_registry_hash, Sha256) or not isinstance(self.capacity_spec_hash, Sha256):
            raise ValidationError("participant and capacity provenance must be Sha256")
        if self.reporter_id not in self.participant_ids.values:
            raise ValidationError("reporter_id must be present in participant_ids")
        vectors = (
            self.reference_request_mw,
            self.reported_request_mw,
            self.actual_request_mw,
            self.capacity_reference_mw,
            self.signed_deviation_mw,
            self.absolute_deviation_mw,
            self.capacity_relative_deviation,
        )
        if any(not isinstance(vector, FloatVector) for vector in vectors):
            raise ValidationError("effective deviation vectors must be FloatVector values")
        n = len(self.participant_ids.values)
        if any(len(vector.values) != n for vector in vectors):
            raise ValidationError("effective deviation vectors must match participant order")
        if any(value < 0.0 for value in (*self.reference_request_mw.values, *self.reported_request_mw.values, *self.actual_request_mw.values, *self.capacity_reference_mw.values)):
            raise ValidationError("request and capacity vectors must be nonnegative")
        if any(value <= 0.0 for value in self.capacity_reference_mw.values):
            raise ValidationError("capacity_reference_mw must be strictly positive for relative deviation")
        expected_signed = FloatVector(
            reported - reference
            for reference, reported in zip(self.reference_request_mw.values, self.reported_request_mw.values)
        )
        expected_absolute = FloatVector(abs(value) for value in expected_signed.values)
        expected_relative = FloatVector(
            absolute / capacity
            for absolute, capacity in zip(expected_absolute.values, self.capacity_reference_mw.values)
        )
        if self.signed_deviation_mw != expected_signed:
            raise ValidationError("signed_deviation_mw does not match reported-reference")
        if self.absolute_deviation_mw != expected_absolute:
            raise ValidationError("absolute_deviation_mw does not match signed deviation")
        if self.capacity_relative_deviation != expected_relative:
            raise ValidationError("capacity_relative_deviation does not match capacity normalization")
        reporter_index = self.participant_ids.values.index(self.reporter_id)
        if self.reporter_signed_deviation_mw.value != expected_signed.values[reporter_index]:
            raise ValidationError("reporter signed deviation is not bound to the vector")
        if self.reporter_absolute_deviation_mw.value != expected_absolute.values[reporter_index]:
            raise ValidationError("reporter absolute deviation is not bound to the vector")
        if self.reporter_relative_deviation.value != expected_relative.values[reporter_index]:
            raise ValidationError("reporter relative deviation is not bound to the vector")
        if self.threshold_mw.value < 0.0:
            raise ValidationError("threshold_mw must be nonnegative")
        if self.effective_deviation_status not in {"EFFECTIVE", "NOT_EFFECTIVE"}:
            raise ValidationError("effective_deviation_status is not registered")
        expected_status = (
            "EFFECTIVE"
            if self.reporter_absolute_deviation_mw.value > self.threshold_mw.value
            else "NOT_EFFECTIVE"
        )
        if self.effective_deviation_status != expected_status:
            raise ValidationError("effective_deviation_status does not match explicit threshold")
        if self.status != "DIAGNOSTIC_ONLY":
            raise ValidationError("effective deviation evaluations cannot be promoted")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "scenario_id": self.scenario_id.to_json(),
            "participant_ids": self.participant_ids.to_json(),
            "participant_registry_hash": self.participant_registry_hash.to_json(),
            "reporter_id": self.reporter_id.to_json(),
            "request_spec_id": self.request_spec_id.to_json(),
            "capacity_spec_hash": self.capacity_spec_hash.to_json(),
            "reference_request_mw": self.reference_request_mw.to_json(),
            "reported_request_mw": self.reported_request_mw.to_json(),
            "actual_request_mw": self.actual_request_mw.to_json(),
            "capacity_reference_mw": self.capacity_reference_mw.to_json(),
            "signed_deviation_mw": self.signed_deviation_mw.to_json(),
            "absolute_deviation_mw": self.absolute_deviation_mw.to_json(),
            "capacity_relative_deviation": self.capacity_relative_deviation.to_json(),
            "reporter_signed_deviation_mw": self.reporter_signed_deviation_mw.to_json(),
            "reporter_absolute_deviation_mw": self.reporter_absolute_deviation_mw.to_json(),
            "reporter_relative_deviation": self.reporter_relative_deviation.to_json(),
            "threshold_mw": self.threshold_mw.to_json(),
            "effective_deviation_status": self.effective_deviation_status,
            "equation_id": self.equation_id.to_json(),
            "threshold_parameter_id": self.threshold_parameter_id.to_json(),
            "status": self.status,
        }

    @property
    def evaluation_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def _evaluate_legacy_eq066_diagnostic(
    report: StrategicReportResult,
    *,
    scenario_id: Identifier,
    actual_request_mw: Sequence[float],
    capacity_reference_mw: Sequence[float],
    capacity_spec_hash: Sha256,
    threshold_mw: float,
) -> _LegacyEq066Evaluation:
    """Evaluate the private legacy EQ066 diagnostic record.

    No default threshold is permitted.  The strategic-report contract still
    enforces a unilateral reporter; this function rejects any non-reporter
    change instead of silently aggregating it.
    """

    if not isinstance(report, StrategicReportResult):
        raise ValidationError("report must be StrategicReportResult")
    if not isinstance(scenario_id, Identifier):
        raise ValidationError("scenario_id must be Identifier")
    if not isinstance(capacity_spec_hash, Sha256):
        raise ValidationError("capacity_spec_hash must be Sha256")
    if isinstance(threshold_mw, bool) or not isinstance(threshold_mw, (int, float)) or not math.isfinite(float(threshold_mw)):
        raise ValidationError("threshold_mw must be an explicit finite scalar")
    threshold = FiniteFloat(float(threshold_mw))
    if threshold.value < 0.0:
        raise ValidationError("threshold_mw must be nonnegative")
    participant_ids = report.reference_raw_request.participant_ids
    reference = report.reference_raw_request.values_mw
    reported = report.reported_raw_request.values_mw
    actual = _finite_vector(actual_request_mw, field="actual_request_mw", nonnegative=True)
    capacity = _finite_vector(capacity_reference_mw, field="capacity_reference_mw", nonnegative=True)
    n = len(participant_ids.values)
    if len(actual.values) != n or len(capacity.values) != n:
        raise ValidationError("actual and capacity vectors must match participant order")
    reporter_index = participant_ids.values.index(report.reporter_id)
    if any(
        index != reporter_index and reported_value != reference_value
        for index, (reference_value, reported_value) in enumerate(zip(reference.values_mw.values, reported.values_mw.values))
    ):
        raise ValidationError("effective deviation requires a unilateral reporter change")
    if any(value <= 0.0 for value in capacity.values):
        raise ValidationError("capacity_reference_mw must be strictly positive")
    signed = FloatVector(reported_value - reference_value for reference_value, reported_value in zip(reference.values_mw.values, reported.values_mw.values))
    absolute = FloatVector(abs(value) for value in signed.values)
    relative = FloatVector(absolute_value / cap for absolute_value, cap in zip(absolute.values, capacity.values))
    reporter_signed = FiniteFloat(signed.values[reporter_index])
    reporter_absolute = FiniteFloat(absolute.values[reporter_index])
    reporter_relative = FiniteFloat(relative.values[reporter_index])
    return _LegacyEq066Evaluation(
        scenario_id=scenario_id,
        participant_ids=participant_ids,
        participant_registry_hash=report.reference_raw_request.participant_registry_hash,
        reporter_id=report.reporter_id,
        request_spec_id=report.reference_raw_request.request_spec_id,
        capacity_spec_hash=capacity_spec_hash,
        reference_request_mw=reference.values_mw,
        reported_request_mw=reported.values_mw,
        actual_request_mw=actual,
        capacity_reference_mw=capacity,
        signed_deviation_mw=signed,
        absolute_deviation_mw=absolute,
        capacity_relative_deviation=relative,
        reporter_signed_deviation_mw=reporter_signed,
        reporter_absolute_deviation_mw=reporter_absolute,
        reporter_relative_deviation=reporter_relative,
        threshold_mw=threshold,
        effective_deviation_status=("EFFECTIVE" if reporter_absolute.value > threshold.value else "NOT_EFFECTIVE"),
    )


_METRIC_STAGES = {"RAW_REPORT", "EFFECTIVE_REQUEST"}
_DIRECTIONS = {"SIGNED", "POSITIVE_ONLY", "ABSOLUTE"}
_NORMALIZATION_BASES = {"NONE", "REFERENCE_RAW_REQUEST", "CAPACITY"}
_ZERO_REFERENCE_POLICIES = {
    "ZERO_IF_ZERO",
    "INFINITE_POSITIVE",
    "EXPLICIT_SPECIAL_CASE",
    "EPSILON_REGULARIZED",
}
_SCIENTIFIC_STATUSES = {"CANDIDATE", "AUTHOR_APPROVED"}
_GAMMA_DOMAINS = {
    "UNRESOLVED",
    "OVER_REPORT_ONLY",
    "UNDER_REPORT_ONLY",
    "BIDIRECTIONAL_NONNEGATIVE",
}
_COMPARISON_OPERATORS = {"GT", "GE"}
_THRESHOLD_RULES = {
    "FIXED_SCALAR",
    "MAX_ABSOLUTE_AND_CAPACITY_RELATIVE",
}
_BOUNDARY_POLICIES = {
    "EXACT_OPERATOR",
    "AMBIGUITY_BAND",
    "CONSERVATIVE_BELOW",
    "CONSERVATIVE_ABOVE",
}
_SCOPES = {"REPORTER_ONLY", "PER_PARTICIPANT", "SCENARIO_ANY"}
_Q_SCOPES = {"Q_INVARIANT", "Q_SPECIFIC"}
_RELATIVE_STATUSES = {
    "NOT_APPLICABLE",
    "DEFINED",
    "ZERO_REFERENCE_ZERO_REPORT",
    "INFINITE_POSITIVE_DEVIATION",
    "UNDEFINED_ZERO_REFERENCE",
}
_GAMMA_DOMAIN_STATUSES = {"VALID", "INVALID", "UNRESOLVED"}
_CLASSIFICATIONS = {
    "HONEST_OR_WITHIN_THRESHOLD",
    "STRATEGIC_DEVIATION",
    "UNRESOLVED",
    "INVALID_INPUT",
    "BOUNDARY_AMBIGUOUS",
}
_SCENARIO_STATUSES = {
    "DIAGNOSTIC_ONLY",
    "UNRESOLVED",
    "INVALID",
}


def _require_identifier(value: object, field: str) -> Identifier:
    if not isinstance(value, Identifier):
        raise ValidationError(f"{field} must be Identifier")
    return value


def _require_text(value: object, field: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValidationError(f"{field} must be one of {sorted(allowed)}")
    return value


def _as_finite_vector(values: Sequence[float] | FloatVector, *, field: str) -> FloatVector:
    if isinstance(values, FloatVector):
        return values
    try:
        return FloatVector(values)
    except Exception as exc:
        raise ValidationError(f"{field} must be a finite numeric vector") from exc


@dataclass(frozen=True, slots=True)
class ReportingEquationSpecification(ContractModel):
    """Independent EQ006 reporting transform specification.

    EQ006 describes how the reported vector is generated; EQ066 describes how
    the resulting deviation is measured.  Keeping this object separate makes
    the gamma domain and numeric tolerance auditable without treating them as
    metric inputs.
    """

    equation_id: Identifier
    gamma_domain: str
    numeric_tolerance_mw: FiniteFloat
    scope: str
    scientific_status: str
    serialization_id = "reporting_equation_specification.v1"

    def __post_init__(self) -> None:
        _require_identifier(self.equation_id, "equation_id")
        if self.equation_id != Identifier("EQ006"):
            raise ValidationError("reporting equation specification must bind EQ006")
        _require_text(self.gamma_domain, "gamma_domain", _GAMMA_DOMAINS)
        if not isinstance(self.numeric_tolerance_mw, FiniteFloat) or self.numeric_tolerance_mw.value < 0.0:
            raise ValidationError("numeric_tolerance_mw must be a nonnegative FiniteFloat")
        if self.scope != "UNILATERAL_REPORTER":
            raise ValidationError("EQ006 scope must be UNILATERAL_REPORTER")
        _require_text(self.scientific_status, "scientific_status", _SCIENTIFIC_STATUSES)


@dataclass(frozen=True, slots=True)
class EffectiveDeviationMetricSpecification(ContractModel):
    """Explicit definition of the metric used for R5 diagnostic classification.

    The metric keeps every semantic choice explicit.  The author-approved
    threshold is supplied by a separate policy object so the metric never
    embeds a hidden epsilon, comparison, or capacity rule.
    """

    metric_id: Identifier
    equation_id: Identifier
    evaluation_stage: str
    directionality: str
    normalization_basis: str
    zero_reference_policy: str
    epsilon_mw_optional: FiniteFloat | None
    unit: Unit
    scientific_status: str
    serialization_id = "effective_deviation_metric_specification.v1"

    def __post_init__(self) -> None:
        _require_identifier(self.metric_id, "metric_id")
        _require_identifier(self.equation_id, "equation_id")
        if self.equation_id != Identifier("EQ066"):
            raise ValidationError("metric specification must bind the registered EQ066 deviation equation")
        _require_text(self.evaluation_stage, "evaluation_stage", _METRIC_STAGES)
        _require_text(self.directionality, "directionality", _DIRECTIONS)
        _require_text(self.normalization_basis, "normalization_basis", _NORMALIZATION_BASES)
        _require_text(self.zero_reference_policy, "zero_reference_policy", _ZERO_REFERENCE_POLICIES)
        if self.epsilon_mw_optional is not None:
            if not isinstance(self.epsilon_mw_optional, FiniteFloat) or self.epsilon_mw_optional.value <= 0.0:
                raise ValidationError("epsilon_mw_optional must be an explicit positive FiniteFloat")
        if not isinstance(self.unit, Unit):
            raise ValidationError("unit must be a registered physical Unit")
        if self.normalization_basis == "NONE" and self.unit is not Unit.MW:
            raise ValidationError("unnormalized deviation must use MW")
        if self.normalization_basis != "NONE" and self.unit is not Unit.PU:
            raise ValidationError("normalized deviation must use pu")
        if self.zero_reference_policy == "EPSILON_REGULARIZED" and self.epsilon_mw_optional is None:
            raise ValidationError("EPSILON_REGULARIZED requires an explicit epsilon")
        if self.zero_reference_policy != "EPSILON_REGULARIZED" and self.epsilon_mw_optional is not None:
            raise ValidationError("epsilon is only valid with EPSILON_REGULARIZED")
        _require_text(self.scientific_status, "scientific_status", _SCIENTIFIC_STATUSES)


@dataclass(frozen=True, slots=True)
class StrategicDeviationThresholdPolicy(ContractModel):
    """Explicit PAR062 policy; threshold and comparison semantics are required."""

    policy_id: Identifier
    metric_id: Identifier
    threshold_value: FiniteFloat
    threshold_unit: Unit
    comparison_operator: str
    comparison_tolerance: FiniteFloat
    boundary_policy: str
    scope: str
    q_scope: str
    applicable_q_modes: tuple[QAssumption, ...]
    scientific_status: str
    threshold_rule: str = "FIXED_SCALAR"
    capacity_relative_fraction: FiniteFloat | None = None
    serialization_id = "strategic_deviation_threshold_policy.v2"

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        _require_identifier(self.metric_id, "metric_id")
        if not isinstance(self.threshold_value, FiniteFloat) or self.threshold_value.value < 0.0:
            raise ValidationError("threshold_value must be a nonnegative FiniteFloat")
        if not isinstance(self.threshold_unit, Unit):
            raise ValidationError("threshold_unit must be a registered physical Unit")
        _require_text(self.comparison_operator, "comparison_operator", _COMPARISON_OPERATORS)
        _require_text(self.threshold_rule, "threshold_rule", _THRESHOLD_RULES)
        if self.threshold_rule == "MAX_ABSOLUTE_AND_CAPACITY_RELATIVE":
            if self.threshold_unit is not Unit.MW:
                raise ValidationError("composite threshold rule requires an MW absolute floor")
            if (
                not isinstance(self.capacity_relative_fraction, FiniteFloat)
                or self.capacity_relative_fraction.value <= 0.0
            ):
                raise ValidationError(
                    "composite threshold rule requires a positive capacity-relative fraction"
                )
        elif self.capacity_relative_fraction is not None:
            raise ValidationError(
                "capacity_relative_fraction is only valid for the composite threshold rule"
            )
        if not isinstance(self.comparison_tolerance, FiniteFloat) or self.comparison_tolerance.value < 0.0:
            raise ValidationError("comparison_tolerance must be a nonnegative FiniteFloat")
        _require_text(self.boundary_policy, "boundary_policy", _BOUNDARY_POLICIES)
        _require_text(self.scope, "scope", _SCOPES)
        _require_text(self.q_scope, "q_scope", _Q_SCOPES)
        if not isinstance(self.applicable_q_modes, tuple) or any(not isinstance(item, QAssumption) for item in self.applicable_q_modes):
            raise ValidationError("applicable_q_modes must be a tuple of registered Q assumptions")
        if len(set(self.applicable_q_modes)) != len(self.applicable_q_modes):
            raise ValidationError("applicable_q_modes must be unique")
        allowed_q = {QAssumption.Q0, QAssumption.Q95}
        if any(item not in allowed_q for item in self.applicable_q_modes):
            raise ValidationError("applicable_q_modes must use Q0/Q95 only")
        if self.q_scope == "Q_SPECIFIC" and len(self.applicable_q_modes) != 1:
            raise ValidationError("Q_SPECIFIC policy must name exactly one applicable Q")
        if self.q_scope == "Q_INVARIANT" and self.applicable_q_modes not in {(), (QAssumption.Q0, QAssumption.Q95), (QAssumption.Q95, QAssumption.Q0)}:
            raise ValidationError("Q_INVARIANT policy must apply to both Q modes or explicitly remain global")
        _require_text(self.scientific_status, "scientific_status", _SCIENTIFIC_STATUSES)


def build_author_approved_effective_deviation_metric(
    metric_id: Identifier,
) -> EffectiveDeviationMetricSpecification:
    """Build the unique author-approved EQ066 classification metric.

    Classification uses the absolute MW deviation.  Signed and
    capacity-relative vectors remain separate diagnostic outputs and are never
    overloaded into this scalar metric field.
    """

    return EffectiveDeviationMetricSpecification(
        metric_id=metric_id,
        equation_id=Identifier("EQ066"),
        evaluation_stage="RAW_REPORT",
        directionality="ABSOLUTE",
        normalization_basis="NONE",
        zero_reference_policy="EXPLICIT_SPECIAL_CASE",
        epsilon_mw_optional=None,
        unit=Unit.MW,
        scientific_status="AUTHOR_APPROVED",
    )


def build_author_approved_strategic_deviation_policy(
    metric_id: Identifier,
    policy_id: Identifier,
    *,
    q_scope: str = "Q_INVARIANT",
    applicable_q_modes: tuple[QAssumption, ...] = (),
) -> StrategicDeviationThresholdPolicy:
    """Build the manuscript-frozen EQ066/PAR062 threshold policy.

    The Round-4R pre-registration cited by the manuscript review fixes the
    effective-deviation rule at ``max(1e-4 MW, 0.01 C_i)``.  The manuscript
    itself uses ``gamma >= 1`` to define a unilateral over-report and reports
    SG/OL/UD at the allocation stage; this EQ066 threshold is an auxiliary
    request-classification boundary, not a manuscript result metric.  The
    absolute floor and the capacity-relative fraction remain separate typed
    fields so their units and provenance cannot be confused.  The resulting
    policy is still diagnostic-only until the downstream evidence gates are
    satisfied.
    """

    # This is the canonical constructor for the author-approved PAR062
    # overlay, not a general policy factory.  Keep the arguments for source
    # compatibility, but reject attempts to manufacture an author-approved
    # policy for a Q-specific slice.  Such alternatives remain candidate
    # diagnostic policies and must be constructed explicitly.
    if q_scope != "Q_INVARIANT" or applicable_q_modes != ():
        raise ValidationError(
            "author-approved PAR062 policy is Q_INVARIANT with no Q-specific modes"
        )

    return StrategicDeviationThresholdPolicy(
        policy_id=policy_id,
        metric_id=metric_id,
        threshold_value=FiniteFloat(1e-4),
        threshold_unit=Unit.MW,
        comparison_operator="GT",
        comparison_tolerance=FiniteFloat(0.0),
        boundary_policy="EXACT_OPERATOR",
        scope="REPORTER_ONLY",
        q_scope=q_scope,
        applicable_q_modes=applicable_q_modes,
        scientific_status="AUTHOR_APPROVED",
        threshold_rule="MAX_ABSOLUTE_AND_CAPACITY_RELATIVE",
        capacity_relative_fraction=FiniteFloat(0.01),
    )


def _is_author_approved_metric_specification(
    metric_specification: EffectiveDeviationMetricSpecification,
) -> bool:
    """Return whether a metric is exactly the frozen EQ066 metric.

    ``scientific_status`` is metadata supplied by a caller and is therefore
    not sufficient evidence of approval.  Equality with the canonical
    builder closes the directionality, stage, normalization, zero policy,
    epsilon, unit, and equation choices in one auditable predicate.
    """

    if metric_specification.scientific_status != "AUTHOR_APPROVED":
        return False
    return metric_specification == build_author_approved_effective_deviation_metric(
        metric_specification.metric_id
    )


def _is_author_approved_threshold_policy(
    threshold_policy: StrategicDeviationThresholdPolicy,
    metric_id: Identifier,
) -> bool:
    """Return whether a policy is exactly the frozen PAR062 overlay."""

    if threshold_policy.scientific_status != "AUTHOR_APPROVED":
        return False
    try:
        canonical_policy = build_author_approved_strategic_deviation_policy(
            metric_id,
            threshold_policy.policy_id,
        )
    except ValidationError:
        return False
    return threshold_policy == canonical_policy


def _is_author_approved_reporting_specification(
    reporting_specification: ReportingEquationSpecification,
) -> bool:
    """Return whether EQ006 has the approved unilateral over-report domain."""

    return (
        reporting_specification.scientific_status == "AUTHOR_APPROVED"
        and reporting_specification.equation_id == Identifier("EQ006")
        and reporting_specification.gamma_domain == "OVER_REPORT_ONLY"
        and reporting_specification.scope == "UNILATERAL_REPORTER"
    )


@dataclass(frozen=True, slots=True)
class StrategicDeviationEvaluation(ContractModel):
    """Typed, fail-closed result of one explicit effective-deviation evaluation."""

    reference_raw_request: RawRequestVector
    reported_raw_request: RawRequestVector
    reporter_id: Identifier
    gamma: FiniteFloat
    eq006_consistent: bool
    reporter_equation_consistent: bool
    nonreporter_identity_consistent: bool
    signed_deviation_mw: FloatVector
    positive_deviation_mw: FloatVector
    absolute_deviation_mw: FloatVector
    relative_deviation: FiniteFloat | None
    relative_deviation_status: str
    metric_id: Identifier
    report_equation_id: Identifier
    deviation_equation_id: Identifier
    reporting_specification_hash: Sha256
    metric_specification_hash: Sha256
    threshold_policy_id: Identifier
    threshold_policy_hash: Sha256
    evaluation_stage: str
    directionality: str
    normalization_basis: str
    zero_reference_policy: str
    scope: str
    q_scope: str
    applicable_q_modes: tuple[QAssumption, ...]
    boundary_policy: str
    gamma_domain: str
    gamma_domain_status: str
    equation_tolerance_mw: FiniteFloat
    effective_deviation_value: FiniteFloat | None
    threshold_value: FiniteFloat
    threshold_unit: Unit
    comparison_operator: str
    comparison_tolerance: FiniteFloat
    threshold_exceeded: bool | None
    classification: str
    scientific_status: str
    failure_reasons: IdentifierVector
    diagnostic_only: bool
    capacity_reference_mw: FloatVector | None = None
    capacity_spec_hash: Sha256 | None = None
    capacity_binding: ExplicitCapacityVector | None = None
    capacity_relative_deviation: FloatVector | None = None
    reporter_capacity_relative_deviation: FiniteFloat | None = None
    q_mode: QAssumption | None = None
    serialization_id = "strategic_deviation_evaluation.v3"

    def __post_init__(self) -> None:
        if not isinstance(self.reference_raw_request, RawRequestVector) or not isinstance(self.reported_raw_request, RawRequestVector):
            raise ValidationError("evaluation requires typed raw reference and reported requests")
        if self.reference_raw_request.source_side.value != "reference" or self.reported_raw_request.source_side.value != "reported":
            raise ValidationError("raw request sides are not aligned")
        if self.reference_raw_request.participant_ids != self.reported_raw_request.participant_ids:
            raise ValidationError("raw request participant order must match")
        if self.reference_raw_request.participant_registry_hash != self.reported_raw_request.participant_registry_hash:
            raise ValidationError("raw request participant registry hashes must match")
        if self.reported_raw_request.reporter_id != self.reporter_id:
            raise ValidationError("reporter ID must match the reported raw request")
        _require_identifier(self.reporter_id, "reporter_id")
        if self.reporter_id not in self.reference_raw_request.participant_ids.values:
            raise ValidationError("reporter_id must be present in participant IDs")
        if not isinstance(self.gamma, FiniteFloat):
            raise ValidationError("gamma must be finite")
        if not isinstance(self.eq006_consistent, bool):
            raise ValidationError("eq006_consistent must be boolean")
        if not isinstance(self.reporter_equation_consistent, bool):
            raise ValidationError("reporter_equation_consistent must be boolean")
        if not isinstance(self.nonreporter_identity_consistent, bool):
            raise ValidationError("nonreporter_identity_consistent must be boolean")
        if self.eq006_consistent != self.reporter_equation_consistent:
            raise ValidationError("eq006_consistent must mirror reporter_equation_consistent")
        n = len(self.reference_raw_request.participant_ids.values)
        vectors = (self.signed_deviation_mw, self.positive_deviation_mw, self.absolute_deviation_mw)
        if any(not isinstance(vector, FloatVector) or len(vector.values) != n for vector in vectors):
            raise ValidationError("deviation vectors must be aligned FloatVectors")
        expected_signed = FloatVector(
            reported - reference
            for reference, reported in zip(
                self.reference_raw_request.values_mw.values,
                self.reported_raw_request.values_mw.values,
            )
        )
        if self.signed_deviation_mw != expected_signed:
            raise ValidationError("signed deviation is not bound to raw requests")
        if self.positive_deviation_mw != FloatVector(max(value, 0.0) for value in expected_signed.values):
            raise ValidationError("positive deviation is not bound to signed deviation")
        if self.absolute_deviation_mw != FloatVector(abs(value) for value in expected_signed.values):
            raise ValidationError("absolute deviation is not bound to signed deviation")
        if self.capacity_reference_mw is not None:
            if not isinstance(self.capacity_reference_mw, FloatVector) or len(self.capacity_reference_mw.values) != n:
                raise ValidationError("capacity_reference_mw must align with participant IDs")
            if any(value <= 0.0 for value in self.capacity_reference_mw.values):
                raise ValidationError("capacity_reference_mw must be strictly positive")
            if self.capacity_spec_hash is None or not isinstance(self.capacity_spec_hash, Sha256):
                raise ValidationError("capacity_spec_hash is required with capacity_reference_mw")
            if self.capacity_binding is not None:
                if not isinstance(self.capacity_binding, ExplicitCapacityVector):
                    raise ValidationError("capacity_binding must be ExplicitCapacityVector")
                if self.capacity_binding.participant_ids != self.reference_raw_request.participant_ids:
                    raise ValidationError("capacity_binding participant order must match requests")
                if self.capacity_binding.participant_registry_hash != self.reference_raw_request.participant_registry_hash:
                    raise ValidationError("capacity_binding participant registry must match requests")
                if self.capacity_binding.values_mw != self.capacity_reference_mw:
                    raise ValidationError("capacity_binding values must match capacity_reference_mw")
                if self.capacity_binding.capacity_spec_hash != self.capacity_spec_hash:
                    raise ValidationError("capacity_binding hash must match capacity_spec_hash")
            expected_capacity_relative = FloatVector(
                absolute / capacity
                for absolute, capacity in zip(
                    self.absolute_deviation_mw.values,
                    self.capacity_reference_mw.values,
                )
            )
            if self.capacity_relative_deviation != expected_capacity_relative:
                raise ValidationError("capacity_relative_deviation is not bound to absolute deviation and capacity")
            reporter_index = self.reference_raw_request.participant_ids.values.index(self.reporter_id)
            if self.reporter_capacity_relative_deviation != FiniteFloat(expected_capacity_relative.values[reporter_index]):
                raise ValidationError("reporter_capacity_relative_deviation is not bound to capacity-relative vector")
        elif self.capacity_binding is not None:
            raise ValidationError("capacity_binding requires capacity_reference_mw")
        elif self.capacity_relative_deviation is not None or self.reporter_capacity_relative_deviation is not None:
            raise ValidationError("capacity-relative deviation requires explicit capacity_reference_mw")
        elif self.capacity_spec_hash is not None:
            raise ValidationError("capacity_spec_hash requires capacity_reference_mw")
        if self.relative_deviation_status not in _RELATIVE_STATUSES:
            raise ValidationError("relative_deviation_status is not registered")
        if self.relative_deviation is not None and not isinstance(self.relative_deviation, FiniteFloat):
            raise ValidationError("relative_deviation must be finite or null")
        _require_identifier(self.metric_id, "metric_id")
        _require_identifier(self.report_equation_id, "report_equation_id")
        _require_identifier(self.deviation_equation_id, "deviation_equation_id")
        if self.report_equation_id != Identifier("EQ006"):
            raise ValidationError("report_equation_id must bind the registered EQ006 report equation")
        if self.deviation_equation_id != Identifier("EQ066"):
            raise ValidationError("deviation_equation_id must bind the registered EQ066 deviation equation")
        if not isinstance(self.reporting_specification_hash, Sha256):
            raise ValidationError("reporting_specification_hash must be Sha256")
        if not isinstance(self.metric_specification_hash, Sha256):
            raise ValidationError("metric_specification_hash must be Sha256")
        _require_identifier(self.threshold_policy_id, "threshold_policy_id")
        if not isinstance(self.threshold_policy_hash, Sha256):
            raise ValidationError("threshold_policy_hash must be Sha256")
        _require_text(self.evaluation_stage, "evaluation_stage", _METRIC_STAGES)
        _require_text(self.directionality, "directionality", _DIRECTIONS)
        _require_text(self.normalization_basis, "normalization_basis", _NORMALIZATION_BASES)
        _require_text(self.zero_reference_policy, "zero_reference_policy", _ZERO_REFERENCE_POLICIES)
        _require_text(self.scope, "scope", _SCOPES)
        _require_text(self.q_scope, "q_scope", _Q_SCOPES)
        _require_text(self.boundary_policy, "boundary_policy", _BOUNDARY_POLICIES)
        _require_text(self.gamma_domain, "gamma_domain", _GAMMA_DOMAINS)
        _require_text(self.gamma_domain_status, "gamma_domain_status", _GAMMA_DOMAIN_STATUSES)
        if not isinstance(self.equation_tolerance_mw, FiniteFloat) or self.equation_tolerance_mw.value < 0.0:
            raise ValidationError("equation_tolerance_mw must be nonnegative")
        if not isinstance(self.effective_deviation_value, (FiniteFloat, type(None))):
            raise ValidationError("effective_deviation_value must be finite or null")
        if not isinstance(self.threshold_value, FiniteFloat) or self.threshold_value.value < 0.0:
            raise ValidationError("threshold_value must be nonnegative")
        if not isinstance(self.threshold_unit, Unit):
            raise ValidationError("threshold_unit must be a registered Unit")
        _require_text(self.comparison_operator, "comparison_operator", _COMPARISON_OPERATORS)
        if not isinstance(self.comparison_tolerance, FiniteFloat) or self.comparison_tolerance.value < 0.0:
            raise ValidationError("comparison_tolerance must be nonnegative")
        if not isinstance(self.applicable_q_modes, tuple) or any(not isinstance(item, QAssumption) for item in self.applicable_q_modes):
            raise ValidationError("evaluation applicable_q_modes must be typed")
        if len(set(self.applicable_q_modes)) != len(self.applicable_q_modes) or any(item not in {QAssumption.Q0, QAssumption.Q95} for item in self.applicable_q_modes):
            raise ValidationError("evaluation applicable_q_modes must contain unique Q0/Q95 values")
        _require_text(self.classification, "classification", _CLASSIFICATIONS)
        _require_text(self.scientific_status, "scientific_status", _SCIENTIFIC_STATUSES)
        if not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure_reasons must be an IdentifierVector")
        if not isinstance(self.diagnostic_only, bool):
            raise ValidationError("diagnostic_only must be boolean")
        if not self.diagnostic_only:
            raise ValidationError("R5 evaluator results are always diagnostic-only; external closure is required")
        if self.q_mode is not None and not isinstance(self.q_mode, QAssumption):
            raise ValidationError("q_mode must be a registered Q assumption or null")
        if self.q_scope == "Q_SPECIFIC" and self.q_mode not in self.applicable_q_modes:
            raise ValidationError("Q_SPECIFIC evaluation Q is outside the policy applicability set")
        if self.applicable_q_modes and self.q_mode is not None and self.q_mode not in self.applicable_q_modes:
            raise ValidationError("evaluation Q is outside the policy applicability set")
        if self.classification == "STRATEGIC_DEVIATION" and self.threshold_exceeded is not True:
            raise ValidationError("strategic classification requires threshold_exceeded=True")
        if self.classification == "HONEST_OR_WITHIN_THRESHOLD" and self.threshold_exceeded is not False:
            raise ValidationError("honest classification requires threshold_exceeded=False")
        if self.classification == "BOUNDARY_AMBIGUOUS" and self.threshold_exceeded is not None:
            raise ValidationError("boundary classification must leave threshold_exceeded unresolved")
        if not self.eq006_consistent and self.classification != "UNRESOLVED":
            raise ValidationError("EQ006 mismatch must fail closed as UNRESOLVED")
        if self.gamma_domain_status != "VALID" and self.classification != "UNRESOLVED":
            raise ValidationError("unresolved or invalid gamma domain must fail closed as UNRESOLVED")

    @property
    def reporter_index(self) -> int:
        return self.reference_raw_request.participant_ids.values.index(self.reporter_id)

    @property
    def reporter_signed_deviation_mw(self) -> FiniteFloat:
        return FiniteFloat(self.signed_deviation_mw.values[self.reporter_index])

    @property
    def reporter_absolute_deviation_mw(self) -> FiniteFloat:
        return FiniteFloat(self.absolute_deviation_mw.values[self.reporter_index])

    @property
    def reporter_positive_deviation_mw(self) -> FiniteFloat:
        return FiniteFloat(self.positive_deviation_mw.values[self.reporter_index])


@dataclass(frozen=True, slots=True)
class TypedStrategicScenario(ContractModel):
    """Scenario object that binds R5 classification to its raw request pair."""

    scenario_id: Identifier
    scenario_family: Identifier
    participant_ids: IdentifierVector
    reporter_id: Identifier
    reference_raw_request: RawRequestVector
    reported_raw_request: RawRequestVector
    gamma: FiniteFloat
    reporting_equation_specification: ReportingEquationSpecification
    deviation_metric_specification: EffectiveDeviationMetricSpecification
    threshold_policy: StrategicDeviationThresholdPolicy
    deviation_evaluation: StrategicDeviationEvaluation
    capacity_vector: FloatVector
    capacity_binding: ExplicitCapacityVector
    classification: str
    scenario_status: str
    failure_reasons: IdentifierVector
    diagnostic_only: bool
    request_configuration: FrozenParameterMap
    reference_admitted_request_hash: Sha256
    reported_admitted_request_hash: Sha256
    q_mode: QAssumption | None = None
    capacity_spec_hash: Sha256 | None = None
    serialization_id = "typed_strategic_scenario.v2"

    def __post_init__(self) -> None:
        _require_identifier(self.scenario_id, "scenario_id")
        _require_identifier(self.scenario_family, "scenario_family")
        _require_identifier(self.reporter_id, "reporter_id")
        if not isinstance(self.participant_ids, IdentifierVector) or not self.participant_ids.values:
            raise ValidationError("participant_ids must be a non-empty IdentifierVector")
        if self.participant_ids != self.reference_raw_request.participant_ids or self.participant_ids != self.reported_raw_request.participant_ids:
            raise ValidationError("scenario participant IDs must match both raw requests")
        if self.reporter_id != self.deviation_evaluation.reporter_id:
            raise ValidationError("scenario reporter is not bound to deviation evaluation")
        if self.reference_raw_request != self.deviation_evaluation.reference_raw_request or self.reported_raw_request != self.deviation_evaluation.reported_raw_request:
            raise ValidationError("scenario raw requests are not bound to deviation evaluation")
        if not isinstance(self.gamma, FiniteFloat) or self.gamma != self.deviation_evaluation.gamma:
            raise ValidationError("scenario gamma is not bound to deviation evaluation")
        if self.deviation_metric_specification.metric_id != self.deviation_evaluation.metric_id:
            raise ValidationError("scenario metric specification is not bound to evaluation")
        if not isinstance(self.reporting_equation_specification, ReportingEquationSpecification):
            raise ValidationError("scenario requires an independent EQ006 reporting specification")
        if _contract_hash(self.reporting_equation_specification) != self.deviation_evaluation.reporting_specification_hash:
            raise ValidationError("scenario reporting specification hash is not bound to evaluation")
        if self.reporting_equation_specification.gamma_domain != self.deviation_evaluation.gamma_domain:
            raise ValidationError("scenario reporting gamma domain is not bound to evaluation")
        if self.reporting_equation_specification.numeric_tolerance_mw != self.deviation_evaluation.equation_tolerance_mw:
            raise ValidationError("scenario reporting tolerance is not bound to evaluation")
        if _contract_hash(self.deviation_metric_specification) != self.deviation_evaluation.metric_specification_hash:
            raise ValidationError("scenario metric specification hash is not bound to evaluation")
        if self.threshold_policy.metric_id != self.deviation_metric_specification.metric_id:
            raise ValidationError("threshold policy metric does not match metric specification")
        if _contract_hash(self.threshold_policy) != self.deviation_evaluation.threshold_policy_hash:
            raise ValidationError("scenario threshold policy hash is not bound to evaluation")
        if self.threshold_policy.policy_id != self.deviation_evaluation.threshold_policy_id:
            raise ValidationError("scenario threshold policy ID is not bound to evaluation")
        if self.deviation_evaluation.applicable_q_modes != self.threshold_policy.applicable_q_modes:
            raise ValidationError("scenario Q applicability is not bound to threshold policy")
        if not isinstance(self.capacity_vector, FloatVector) or len(self.capacity_vector.values) != len(self.participant_ids.values):
            raise ValidationError("capacity_vector must align with participant IDs")
        if any(value < 0.0 for value in self.capacity_vector.values):
            raise ValidationError("capacity_vector must be nonnegative")
        if not isinstance(self.capacity_binding, ExplicitCapacityVector):
            raise ValidationError("typed strategic scenarios require an ExplicitCapacityVector binding")
        if self.capacity_binding.participant_ids != self.participant_ids:
            raise ValidationError("capacity binding participant order is not bound to scenario")
        if self.capacity_binding.values_mw != self.capacity_vector:
            raise ValidationError("capacity binding values are not bound to scenario")
        if self.capacity_spec_hash is not None and not isinstance(self.capacity_spec_hash, Sha256):
            raise ValidationError("capacity_spec_hash must be Sha256 or null")
        if self.capacity_spec_hash != self.capacity_binding.capacity_spec_hash:
            raise ValidationError("capacity spec hash is not bound to typed capacity object")
        if self.capacity_binding.participant_registry_hash != self.reference_raw_request.participant_registry_hash:
            raise ValidationError("capacity binding participant provenance does not match scenario")
        if self.deviation_evaluation.capacity_spec_hash != self.capacity_spec_hash:
            raise ValidationError("scenario capacity provenance is not bound to evaluation")
        if self.deviation_evaluation.capacity_binding is not None and self.deviation_evaluation.capacity_binding != self.capacity_binding:
            raise ValidationError("scenario capacity binding is not bound to evaluation")
        if self.deviation_evaluation.capacity_reference_mw is not None and self.deviation_evaluation.capacity_reference_mw != self.capacity_vector:
            raise ValidationError("scenario capacity vector is not bound to evaluation")
        if self.deviation_evaluation.capacity_relative_deviation is not None:
            expected_capacity_relative = FloatVector(
                absolute / capacity
                for absolute, capacity in zip(
                    self.deviation_evaluation.absolute_deviation_mw.values,
                    self.capacity_vector.values,
                )
            )
            if self.deviation_evaluation.capacity_relative_deviation != expected_capacity_relative:
                raise ValidationError("scenario capacity-relative deviation is not bound to capacity")
        _require_text(self.classification, "classification", _CLASSIFICATIONS)
        _require_text(self.scenario_status, "scenario_status", _SCENARIO_STATUSES)
        if not isinstance(self.failure_reasons, IdentifierVector):
            raise ValidationError("failure_reasons must be an IdentifierVector")
        if not isinstance(self.reference_admitted_request_hash, Sha256) or not isinstance(self.reported_admitted_request_hash, Sha256):
            raise ValidationError("scenario admitted request hashes must be SHA-256")
        if self.classification != self.deviation_evaluation.classification:
            raise ValidationError("scenario classification is not bound to deviation evaluation")
        if self.failure_reasons != self.deviation_evaluation.failure_reasons:
            raise ValidationError("scenario failure reasons are not bound to deviation evaluation")
        if self.q_mode != self.deviation_evaluation.q_mode:
            raise ValidationError("scenario q_mode is not bound to deviation evaluation")
        expected_status = "UNRESOLVED" if self.classification == "UNRESOLVED" else "DIAGNOSTIC_ONLY"
        if self.scenario_status != expected_status:
            raise ValidationError("scenario_status does not follow classification")
        if not isinstance(self.diagnostic_only, bool):
            raise ValidationError("diagnostic_only must be boolean")
        if self.deviation_evaluation.diagnostic_only != self.diagnostic_only:
            raise ValidationError("scenario diagnostic flag is not bound to evaluation")
        if self.q_mode is not None and not isinstance(self.q_mode, QAssumption):
            raise ValidationError("q_mode must be a registered Q assumption or null")
        if isinstance(self.request_configuration, Mapping) and not isinstance(self.request_configuration, FrozenParameterMap):
            object.__setattr__(self, "request_configuration", FrozenParameterMap(self.request_configuration))
        elif not isinstance(self.request_configuration, FrozenParameterMap) or not len(self.request_configuration):
            raise ValidationError("request_configuration must be a non-empty explicit mapping")
        if not self.diagnostic_only:
            raise ValidationError("formal scenario eligibility requires an external closure gate")


@dataclass(frozen=True, slots=True)
class TypedStrategicScenarioRoster(ContractModel):
    """Ordered candidate roster for the 30 reporter scenarios.

    This object freezes roster identity and shared policy provenance while
    remaining diagnostic-only.  It deliberately has no path to formal
    eligibility; a separate closure gate must be implemented and passed.
    """

    roster_id: Identifier
    scenario_family: Identifier
    ordered_scenario_ids: IdentifierVector
    ordered_reporter_ids: IdentifierVector
    scenarios: tuple[TypedStrategicScenario, ...]
    expected_scenario_count: int
    reporting_specification_hash: Sha256
    metric_specification_hash: Sha256
    threshold_policy_hash: Sha256
    gamma_domain: str
    equation_numeric_tolerance_mw: FiniteFloat
    request_configuration_hash: Sha256
    capacity_policy_hash: Sha256
    capacity_bindings: tuple[ExplicitCapacityVector, ...]
    participant_ids: IdentifierVector
    behavior_q_policy: str
    status: str
    diagnostic_only: bool
    serialization_id = "typed_strategic_scenario_roster.v2"

    def __post_init__(self) -> None:
        _require_identifier(self.roster_id, "roster_id")
        _require_identifier(self.scenario_family, "scenario_family")
        if not isinstance(self.ordered_scenario_ids, IdentifierVector) or not isinstance(self.ordered_reporter_ids, IdentifierVector):
            raise ValidationError("roster IDs must be typed IdentifierVectors")
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValidationError("roster scenarios must be a non-empty tuple")
        if any(not isinstance(scenario, TypedStrategicScenario) for scenario in self.scenarios):
            raise ValidationError("roster scenarios must be TypedStrategicScenario values")
        if isinstance(self.expected_scenario_count, bool) or not isinstance(self.expected_scenario_count, int) or self.expected_scenario_count <= 0:
            raise ValidationError("expected_scenario_count must be a positive integer")
        if not isinstance(self.capacity_policy_hash, Sha256):
            raise ValidationError("capacity_policy_hash must be SHA-256")
        if len(self.scenarios) != self.expected_scenario_count:
            raise ValidationError("roster scenario count does not match expected_scenario_count")
        if len(self.ordered_scenario_ids.values) != len(self.scenarios) or len(self.ordered_reporter_ids.values) != len(self.scenarios):
            raise ValidationError("ordered roster IDs must align with scenarios")
        if len(set(self.ordered_scenario_ids.values)) != len(self.scenarios):
            raise ValidationError("scenario IDs must be unique")
        if len(set(self.ordered_reporter_ids.values)) != len(self.scenarios):
            raise ValidationError("reporter IDs must be unique")
        if tuple(scenario.scenario_id for scenario in self.scenarios) != self.ordered_scenario_ids.values:
            raise ValidationError("ordered_scenario_ids must preserve scenario order")
        if tuple(scenario.reporter_id for scenario in self.scenarios) != self.ordered_reporter_ids.values:
            raise ValidationError("ordered_reporter_ids must preserve scenario order")
        if any(scenario.scenario_family != self.scenario_family for scenario in self.scenarios):
            raise ValidationError("roster scenario family is not uniform")
        if any(scenario.participant_ids != self.participant_ids for scenario in self.scenarios):
            raise ValidationError("roster participant order is not uniform")
        if not isinstance(self.capacity_bindings, tuple):
            raise ValidationError("capacity_bindings must be a tuple")
        if len(self.capacity_bindings) != len(self.scenarios):
            raise ValidationError("capacity_bindings must contain one typed binding per scenario")
        if any(not isinstance(binding, ExplicitCapacityVector) for binding in self.capacity_bindings):
            raise ValidationError("capacity_bindings must contain ExplicitCapacityVector values")
        if any(
            _contract_hash(scenario.reporting_equation_specification) != self.reporting_specification_hash
            or _contract_hash(scenario.deviation_metric_specification) != self.metric_specification_hash
            or _contract_hash(scenario.threshold_policy) != self.threshold_policy_hash
            or scenario.deviation_evaluation.gamma_domain != self.gamma_domain
            or scenario.deviation_evaluation.equation_tolerance_mw != self.equation_numeric_tolerance_mw
            or Sha256(canonical_hash(scenario.request_configuration.to_json())) != self.request_configuration_hash
            or binding != scenario.capacity_binding
            for scenario, binding in zip(self.scenarios, self.capacity_bindings)
        ):
            raise ValidationError("roster policy, gamma, request and per-scenario capacity provenance is not uniform")
        expected_capacity_policy_hash = Sha256(canonical_hash({
            "policy": "PER_SCENARIO_TYPED_CAPACITY",
            "bindings": [binding.capacity_vector_hash.to_json() for binding in self.capacity_bindings],
        }))
        if self.capacity_policy_hash != expected_capacity_policy_hash:
            raise ValidationError("capacity_policy_hash does not match ordered capacity bindings")
        _require_text(self.behavior_q_policy, "behavior_q_policy", {"Q_INVARIANT", "Q_SPECIFIC"})
        q_modes = tuple(scenario.q_mode for scenario in self.scenarios)
        if self.behavior_q_policy == "Q_INVARIANT" and any(mode is not None for mode in q_modes):
            raise ValidationError("Q_INVARIANT roster scenarios must leave behavior q_mode unset")
        if self.behavior_q_policy == "Q_SPECIFIC":
            if any(mode is None for mode in q_modes) or len(set(q_modes)) != 1:
                raise ValidationError("Q_SPECIFIC roster scenarios must share one explicit Q mode")
        if self.status != "CANDIDATE_DIAGNOSTIC":
            raise ValidationError("formal roster closure requires a separate external gate")
        if self.expected_scenario_count != 30:
            raise ValidationError("the frozen reporter roster must contain exactly 30 scenarios")
        if not isinstance(self.diagnostic_only, bool) or not self.diagnostic_only:
            raise ValidationError("candidate roster is diagnostic-only")

    @property
    def roster_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))


def build_typed_strategic_scenario_roster(
    scenarios: Sequence[TypedStrategicScenario],
    *,
    roster_id: Identifier,
    scenario_family: Identifier,
    expected_scenario_count: int,
    behavior_q_policy: str,
) -> TypedStrategicScenarioRoster:
    """Materialize an explicit candidate roster; no implicit sorting or padding."""

    typed = tuple(scenarios)
    if not typed:
        raise ValidationError("scenarios must not be empty")
    first = typed[0]
    if first.capacity_binding.capacity_spec_hash is None:
        raise ValidationError("candidate roster requires capacity provenance")
    capacity_bindings = tuple(scenario.capacity_binding for scenario in typed)
    capacity_policy_hash = Sha256(canonical_hash({
        "policy": "PER_SCENARIO_TYPED_CAPACITY",
        "bindings": [binding.capacity_vector_hash.to_json() for binding in capacity_bindings],
    }))
    return TypedStrategicScenarioRoster(
        roster_id=roster_id,
        scenario_family=scenario_family,
        ordered_scenario_ids=IdentifierVector(scenario.scenario_id for scenario in typed),
        ordered_reporter_ids=IdentifierVector(scenario.reporter_id for scenario in typed),
        scenarios=typed,
        expected_scenario_count=expected_scenario_count,
        reporting_specification_hash=_contract_hash(first.reporting_equation_specification),
        metric_specification_hash=_contract_hash(first.deviation_metric_specification),
        threshold_policy_hash=_contract_hash(first.threshold_policy),
        gamma_domain=first.deviation_evaluation.gamma_domain,
        equation_numeric_tolerance_mw=first.deviation_evaluation.equation_tolerance_mw,
        request_configuration_hash=Sha256(canonical_hash(first.request_configuration.to_json())),
        capacity_policy_hash=capacity_policy_hash,
        capacity_bindings=capacity_bindings,
        participant_ids=first.participant_ids,
        behavior_q_policy=behavior_q_policy,
        status="CANDIDATE_DIAGNOSTIC",
        diagnostic_only=True,
    )


def _metric_numerator(directionality: str, signed: FloatVector) -> FloatVector:
    if directionality == "SIGNED":
        return signed
    if directionality == "POSITIVE_ONLY":
        return FloatVector(max(value, 0.0) for value in signed.values)
    return FloatVector(abs(value) for value in signed.values)


def _normalized_scalar(
    numerator: float,
    denominator: float,
    *,
    zero_reference_policy: str,
    epsilon_mw_optional: FiniteFloat | None,
) -> tuple[FiniteFloat | None, str]:
    if denominator != 0.0:
        return FiniteFloat(numerator / denominator), "DEFINED"
    if numerator == 0.0:
        if zero_reference_policy == "EPSILON_REGULARIZED":
            epsilon = epsilon_mw_optional.value if epsilon_mw_optional is not None else None
            if epsilon is None:
                raise ValidationError("regularized normalization requires explicit epsilon")
            return FiniteFloat(0.0), "DEFINED"
        if zero_reference_policy == "ZERO_IF_ZERO":
            return FiniteFloat(0.0), "ZERO_REFERENCE_ZERO_REPORT"
        return None, "UNDEFINED_ZERO_REFERENCE"
    if zero_reference_policy == "EPSILON_REGULARIZED":
        epsilon = epsilon_mw_optional.value if epsilon_mw_optional is not None else None
        if epsilon is None:
            raise ValidationError("regularized normalization requires explicit epsilon")
        return FiniteFloat(numerator / epsilon), "DEFINED"
    if zero_reference_policy == "INFINITE_POSITIVE" and numerator > 0.0:
        return None, "INFINITE_POSITIVE_DEVIATION"
    return None, "UNDEFINED_ZERO_REFERENCE"


def _contract_hash(value: ContractModel) -> Sha256:
    return Sha256(canonical_hash({"serialization_id": value.serialization_id, "fields": value.to_json()}))


def evaluate_effective_deviation(
    reference_request: RawRequestVector,
    reported_request: RawRequestVector,
    reporter_id: Identifier,
    gamma: FiniteFloat,
    metric_specification: EffectiveDeviationMetricSpecification,
    threshold_policy: StrategicDeviationThresholdPolicy,
    *,
    capacity_reference_mw: Sequence[float] | FloatVector | None = None,
    capacity_spec_hash: Sha256 | None = None,
    capacity_binding: ExplicitCapacityVector | None = None,
    q_mode: QAssumption | None = None,
    reporting_specification: ReportingEquationSpecification,
) -> StrategicDeviationEvaluation:
    """Evaluate R5 with every scientific choice supplied explicitly.

    There is intentionally no default threshold, comparison operator,
    normalization, epsilon, Q scope, or reporter scope.  Equation and roster
    mismatches produce an unresolved diagnostic result rather than an inferred
    correction.
    """

    if not isinstance(reference_request, RawRequestVector) or not isinstance(reported_request, RawRequestVector):
        raise ValidationError("reference_request and reported_request must be RawRequestVector")
    if not isinstance(reporter_id, Identifier):
        raise ValidationError("reporter_id must be Identifier")
    if not isinstance(gamma, FiniteFloat):
        raise ValidationError("gamma must be FiniteFloat")
    if not isinstance(metric_specification, EffectiveDeviationMetricSpecification):
        raise ValidationError("metric_specification must be typed")
    if not isinstance(threshold_policy, StrategicDeviationThresholdPolicy):
        raise ValidationError("threshold_policy must be typed and explicit")
    if not isinstance(reporting_specification, ReportingEquationSpecification):
        raise ValidationError("reporting_specification is mandatory and must bind EQ006")
    if threshold_policy.metric_id != metric_specification.metric_id:
        raise ValidationError("threshold policy metric must match metric specification")
    if metric_specification.evaluation_stage != "RAW_REPORT":
        raise ValidationError("this evaluator accepts RAW_REPORT stage only")
    if threshold_policy.threshold_unit is not metric_specification.unit:
        raise ValidationError("threshold unit must match metric unit")
    if capacity_binding is not None:
        if not isinstance(capacity_binding, ExplicitCapacityVector):
            raise ValidationError("capacity_binding must be an ExplicitCapacityVector")
        if capacity_binding.participant_ids != reference_request.participant_ids:
            raise ValidationError("capacity_binding participant order must match requests")
        if capacity_binding.participant_registry_hash != reference_request.participant_registry_hash:
            raise ValidationError("capacity_binding participant registry must match requests")
        if capacity_reference_mw is None:
            capacity_reference_mw = capacity_binding.values_mw
        elif _as_finite_vector(capacity_reference_mw, field="capacity_reference_mw") != capacity_binding.values_mw:
            raise ValidationError("capacity_reference_mw must match capacity_binding values")
        if capacity_spec_hash is None:
            capacity_spec_hash = capacity_binding.capacity_spec_hash
        elif capacity_spec_hash != capacity_binding.capacity_spec_hash:
            raise ValidationError("capacity_spec_hash must match capacity_binding")
    if threshold_policy.threshold_rule == "MAX_ABSOLUTE_AND_CAPACITY_RELATIVE":
        if metric_specification.normalization_basis != "NONE" or metric_specification.unit is not Unit.MW:
            raise ValidationError(
                "composite strategic-deviation threshold requires an absolute MW metric"
            )
        if capacity_reference_mw is None:
            raise ValidationError(
                "composite strategic-deviation threshold requires capacity_reference_mw"
            )
        if capacity_binding is None:
            raise ValidationError(
                "author-approved composite threshold requires capacity_binding"
            )
    if threshold_policy.scope != "REPORTER_ONLY":
        raise ValidationError("unilateral evaluator requires REPORTER_ONLY scope")
    if threshold_policy.q_scope == "Q_SPECIFIC" and q_mode is None:
        raise ValidationError("Q_SPECIFIC threshold policy requires q_mode")
    if q_mode is not None and not isinstance(q_mode, QAssumption):
        raise ValidationError("q_mode must be a registered Q assumption")
    if threshold_policy.applicable_q_modes and q_mode not in threshold_policy.applicable_q_modes:
        raise ValidationError("q_mode is outside the threshold policy applicability set")
    if reference_request.source_side.value != "reference" or reported_request.source_side.value != "reported":
        raise ValidationError("raw request sides must be reference and reported")
    if reference_request.participant_ids != reported_request.participant_ids:
        raise ValidationError("reference and reported participant order must match")
    if reference_request.participant_registry_hash != reported_request.participant_registry_hash:
        raise ValidationError("reference and reported participant registry hashes must match")
    if reported_request.reporter_id != reporter_id:
        raise ValidationError("reported request reporter does not match reporter_id")
    if reporter_id not in reference_request.participant_ids.values:
        raise ValidationError("reporter_id is absent from participant registry")
    n = len(reference_request.participant_ids.values)
    if capacity_reference_mw is not None:
        capacity = _as_finite_vector(capacity_reference_mw, field="capacity_reference_mw")
        if len(capacity.values) != n:
            raise ValidationError("capacity_reference_mw must align with participant IDs")
        if any(value <= 0.0 for value in capacity.values):
            raise ValidationError("capacity_reference_mw must be strictly positive")
        if capacity_spec_hash is None or not isinstance(capacity_spec_hash, Sha256):
            raise ValidationError("capacity_spec_hash is required when capacity is supplied")
    else:
        capacity = None
        if capacity_spec_hash is not None:
            raise ValidationError("capacity_spec_hash requires capacity_reference_mw")
    if metric_specification.normalization_basis == "CAPACITY" and capacity is None:
        raise ValidationError("CAPACITY normalization requires explicit capacity_reference_mw")

    reference = reference_request.values_mw.values
    reported = reported_request.values_mw.values
    signed = FloatVector(after - before for before, after in zip(reference, reported))
    positive = FloatVector(max(value, 0.0) for value in signed.values)
    absolute = FloatVector(abs(value) for value in signed.values)
    reporter_index = reference_request.participant_ids.values.index(reporter_id)
    capacity_relative_vector = (
        FloatVector(
            absolute_value / capacity_value
            for absolute_value, capacity_value in zip(absolute.values, capacity.values)
        )
        if capacity is not None
        else None
    )
    reporter_capacity_relative = (
        FiniteFloat(capacity_relative_vector.values[reporter_index])
        if capacity_relative_vector is not None
        else None
    )

    failures: list[Identifier] = []
    expected_reporter = gamma.value * reference[reporter_index]
    equation_tolerance = reporting_specification.numeric_tolerance_mw.value
    equation_consistent = math.isclose(
        reported[reporter_index], expected_reporter, rel_tol=0.0, abs_tol=equation_tolerance
    )
    nonreporter_identity = all(
        math.isclose(before, after, rel_tol=0.0, abs_tol=equation_tolerance)
        for index, (before, after) in enumerate(zip(reference, reported))
        if index != reporter_index
    )
    if not equation_consistent:
        failures.append(Identifier("REQUEST_VECTOR_MISMATCH"))
    if not nonreporter_identity and not any(reason.value == "REQUEST_VECTOR_MISMATCH" for reason in failures):
        failures.append(Identifier("REQUEST_VECTOR_MISMATCH"))
    if all(value == 0.0 for value in signed.values):
        failures.append(Identifier("NO_REPORTER_CHANGE"))

    if reporting_specification.gamma_domain == "UNRESOLVED":
        gamma_domain_status = "UNRESOLVED"
    elif reporting_specification.gamma_domain == "OVER_REPORT_ONLY":
        gamma_domain_status = "VALID" if gamma.value >= 1.0 else "INVALID"
    elif reporting_specification.gamma_domain == "UNDER_REPORT_ONLY":
        gamma_domain_status = "VALID" if 0.0 <= gamma.value <= 1.0 else "INVALID"
    else:
        gamma_domain_status = "VALID" if gamma.value >= 0.0 else "INVALID"
    if gamma_domain_status == "INVALID":
        failures.append(Identifier("TRANSFORMATION_INVALID"))

    numerator = _metric_numerator(metric_specification.directionality, signed)
    if metric_specification.normalization_basis == "NONE":
        metric_value = FiniteFloat(numerator.values[reporter_index])
        relative_value = None
        relative_status = "NOT_APPLICABLE"
    elif metric_specification.normalization_basis == "REFERENCE_RAW_REQUEST":
        metric_value, relative_status = _normalized_scalar(
            numerator.values[reporter_index],
            reference[reporter_index],
            zero_reference_policy=metric_specification.zero_reference_policy,
            epsilon_mw_optional=metric_specification.epsilon_mw_optional,
        )
        relative_value = metric_value
    else:
        assert capacity is not None
        metric_value = FiniteFloat(numerator.values[reporter_index] / capacity.values[reporter_index])
        relative_value, relative_status = _normalized_scalar(
            numerator.values[reporter_index],
            reference[reporter_index],
            zero_reference_policy=metric_specification.zero_reference_policy,
            epsilon_mw_optional=metric_specification.epsilon_mw_optional,
        )

    effective_threshold_value = threshold_policy.threshold_value
    if threshold_policy.threshold_rule == "MAX_ABSOLUTE_AND_CAPACITY_RELATIVE":
        assert capacity is not None
        assert threshold_policy.capacity_relative_fraction is not None
        effective_threshold_value = FiniteFloat(
            max(
                threshold_policy.threshold_value.value,
                threshold_policy.capacity_relative_fraction.value
                * capacity.values[reporter_index],
            )
        )

    threshold_exceeded: bool | None
    if metric_value is None and relative_status == "INFINITE_POSITIVE_DEVIATION":
        threshold_exceeded = True
    elif metric_value is None:
        threshold_exceeded = None
    else:
        delta = metric_value.value - effective_threshold_value.value
        tolerance = threshold_policy.comparison_tolerance.value
        if threshold_policy.boundary_policy == "AMBIGUITY_BAND" and abs(delta) <= tolerance and tolerance > 0.0:
            threshold_exceeded = None
        elif threshold_policy.boundary_policy == "CONSERVATIVE_BELOW" and abs(delta) <= tolerance:
            threshold_exceeded = False
        elif threshold_policy.boundary_policy == "CONSERVATIVE_ABOVE" and abs(delta) <= tolerance:
            threshold_exceeded = True
        elif threshold_policy.comparison_operator == "GT":
            threshold_exceeded = metric_value.value > effective_threshold_value.value
        else:
            threshold_exceeded = metric_value.value >= effective_threshold_value.value

    if not equation_consistent or not nonreporter_identity or gamma_domain_status != "VALID":
        classification = "UNRESOLVED"
    elif threshold_exceeded is None:
        classification = "BOUNDARY_AMBIGUOUS" if metric_value is not None else "UNRESOLVED"
    elif threshold_exceeded:
        classification = "STRATEGIC_DEVIATION"
    else:
        classification = "HONEST_OR_WITHIN_THRESHOLD"
        if not any(reason.value == "NO_REPORTER_CHANGE" for reason in failures):
            failures.append(Identifier("EFFECTIVE_DEVIATION_BELOW_THRESHOLD"))

    # An approval label on any input is caller metadata, not evidence.  The
    # evaluator grants AUTHOR_APPROVED only when all three supplied objects
    # exactly match the frozen author overlay.  In particular this prevents a
    # hand-built POSITIVE_ONLY/SIGNED metric, a fixed-scalar or Q-specific
    # policy, or an UNDER_REPORT_ONLY EQ006 specification from being promoted
    # merely by setting ``scientific_status``.
    author_approved_semantics = (
        _is_author_approved_metric_specification(metric_specification)
        and _is_author_approved_threshold_policy(threshold_policy, metric_specification.metric_id)
        and _is_author_approved_reporting_specification(reporting_specification)
    )
    scientific_status = (
        "AUTHOR_APPROVED"
        if author_approved_semantics and capacity_binding is not None
        else "CANDIDATE"
    )
    # Policy approval is only a definition-level signal.  Formal scenario
    # eligibility requires a separate closure gate and never comes from this
    # evaluator alone.
    diagnostic_only = True
    return StrategicDeviationEvaluation(
        reference_raw_request=reference_request,
        reported_raw_request=reported_request,
        reporter_id=reporter_id,
        gamma=gamma,
        eq006_consistent=equation_consistent,
        reporter_equation_consistent=equation_consistent,
        nonreporter_identity_consistent=nonreporter_identity,
        signed_deviation_mw=signed,
        positive_deviation_mw=positive,
        absolute_deviation_mw=absolute,
        relative_deviation=relative_value,
        relative_deviation_status=relative_status,
        metric_id=metric_specification.metric_id,
        report_equation_id=Identifier("EQ006"),
        deviation_equation_id=metric_specification.equation_id,
        reporting_specification_hash=_contract_hash(reporting_specification),
        metric_specification_hash=_contract_hash(metric_specification),
        threshold_policy_id=threshold_policy.policy_id,
        threshold_policy_hash=_contract_hash(threshold_policy),
        evaluation_stage=metric_specification.evaluation_stage,
        directionality=metric_specification.directionality,
        normalization_basis=metric_specification.normalization_basis,
        zero_reference_policy=metric_specification.zero_reference_policy,
        scope=threshold_policy.scope,
        q_scope=threshold_policy.q_scope,
        applicable_q_modes=threshold_policy.applicable_q_modes,
        boundary_policy=threshold_policy.boundary_policy,
        gamma_domain=reporting_specification.gamma_domain,
        gamma_domain_status=gamma_domain_status,
        equation_tolerance_mw=reporting_specification.numeric_tolerance_mw,
        effective_deviation_value=metric_value,
        threshold_value=effective_threshold_value,
        threshold_unit=threshold_policy.threshold_unit,
        comparison_operator=threshold_policy.comparison_operator,
        comparison_tolerance=threshold_policy.comparison_tolerance,
        threshold_exceeded=threshold_exceeded if equation_consistent and nonreporter_identity and gamma_domain_status == "VALID" else None,
        classification=classification,
        scientific_status=scientific_status,
        failure_reasons=IdentifierVector(failures),
        diagnostic_only=diagnostic_only,
        capacity_reference_mw=capacity,
        capacity_spec_hash=capacity_spec_hash,
        capacity_binding=capacity_binding,
        capacity_relative_deviation=capacity_relative_vector,
        reporter_capacity_relative_deviation=reporter_capacity_relative,
        q_mode=q_mode,
    )


def _strict_bundle_fields(
    bundle: DiagnosticRequestScenarioBundle,
) -> tuple[
    RawRequestVector,
    RawRequestVector,
    Identifier,
    RequestConstruction,
    AdmittedRequestVector,
    AdmittedRequestVector,
]:
    """Validate the structural surface of DiagnosticRequestScenarioBundle.

    The bundle contract lives in ``r4r`` so this boundary uses a real
    ``isinstance`` check rather than a class-name/attribute duck type.
    """

    if not isinstance(bundle, DiagnosticRequestScenarioBundle):
        raise ValidationError("strict scenario builder requires DiagnosticRequestScenarioBundle")
    required = (
        "participant_ids",
        "participant_registry_hash",
        "request_construction",
        "reference_raw_request",
        "reference_admitted_request",
        "reported_raw_request",
        "reported_admitted_request",
        "strategic_report",
    )
    if any(not hasattr(bundle, name) for name in required):
        raise ValidationError("diagnostic scenario bundle is missing a required typed layer")
    construction = getattr(bundle, "request_construction")
    reference = getattr(bundle, "reference_raw_request")
    reported = getattr(bundle, "reported_raw_request")
    reference_admitted = getattr(bundle, "reference_admitted_request")
    reported_admitted = getattr(bundle, "reported_admitted_request")
    strategic_report = getattr(bundle, "strategic_report")
    if not isinstance(construction, RequestConstruction):
        raise ValidationError("bundle request_construction must be RequestConstruction")
    if not isinstance(reference, RawRequestVector) or not isinstance(reported, RawRequestVector):
        raise ValidationError("bundle raw requests must be typed")
    if not isinstance(reference_admitted, AdmittedRequestVector) or not isinstance(reported_admitted, AdmittedRequestVector):
        raise ValidationError("bundle admitted requests must be typed")
    if not isinstance(strategic_report, StrategicReportResult):
        raise ValidationError("bundle strategic_report must be typed")
    if reference.values_mw != construction.actual_request_mw:
        raise ValidationError("bundle reference raw request is not generated by RequestConstruction")
    if reference_admitted.raw_request_hash != reference.request_hash or reference_admitted.raw_values_mw != reference.values_mw:
        raise ValidationError("bundle reference admission does not bind reference raw request")
    if reported_admitted.raw_request_hash != reported.request_hash or reported_admitted.raw_values_mw != reported.values_mw:
        raise ValidationError("bundle reported admission does not bind reported raw request")
    if strategic_report.reference_raw_request != reference or strategic_report.reported_raw_request != reported:
        raise ValidationError("bundle strategic report is not bound to raw request pair")
    if getattr(bundle, "status", None) != "DIAGNOSTIC_ONLY":
        raise ValidationError("request scenario bundle must remain diagnostic-only")
    return (
        reference,
        reported,
        strategic_report.reporter_id,
        construction,
        reference_admitted,
        reported_admitted,
    )


def build_typed_strategic_scenario(
    *,
    source_bundle: object,
    scenario_id: Identifier,
    scenario_family: Identifier,
    gamma: FiniteFloat,
    metric_specification: EffectiveDeviationMetricSpecification,
    threshold_policy: StrategicDeviationThresholdPolicy,
    capacity_binding: ExplicitCapacityVector,
    request_configuration: Mapping[str, Any] | FrozenParameterMap,
    reporting_specification: ReportingEquationSpecification,
    q_mode: QAssumption | None = None,
) -> TypedStrategicScenario:
    """Create a typed scenario only from the complete diagnostic bundle."""

    if not isinstance(source_bundle, DiagnosticRequestScenarioBundle):
        raise ValidationError("source_bundle must be DiagnosticRequestScenarioBundle")
    (
        reference_request,
        reported_request,
        reporter_id,
        construction,
        reference_admitted,
        reported_admitted,
    ) = _strict_bundle_fields(source_bundle)
    if not isinstance(capacity_binding, ExplicitCapacityVector):
        raise ValidationError("typed strategic scenarios require an ExplicitCapacityVector binding")
    if capacity_binding.participant_ids != reference_request.participant_ids:
        raise ValidationError("capacity binding participant order does not match the request bundle")
    if capacity_binding.participant_registry_hash != reference_request.participant_registry_hash:
        raise ValidationError("capacity binding participant registry does not match request bundle")
    if reference_admitted.capacity_spec_hash != capacity_binding.capacity_spec_hash:
        raise ValidationError("reference admission capacity identity does not match typed capacity")
    if reported_admitted.capacity_spec_hash != capacity_binding.capacity_spec_hash:
        raise ValidationError("reported admission capacity identity does not match typed capacity")
    if not isinstance(reporting_specification, ReportingEquationSpecification):
        raise ValidationError("reporting_specification is mandatory and must bind EQ006")
    configuration = request_configuration
    if configuration is None:
        raise ValidationError("request_configuration is required")
    if not isinstance(configuration, FrozenParameterMap):
        configuration = FrozenParameterMap(configuration)
    required_configuration = {
        "intercept_mw",
        "load_factor",
        "clip_lower_mw",
        "clip_upper_mw",
        "stress_multiplier",
    }
    if set(configuration) != required_configuration:
        raise ValidationError(
            "typed strategic scenarios require the complete explicit RequestConstruction configuration"
        )
    expected_configuration = {
        "intercept_mw": construction.intercept_mw.value,
        "load_factor": construction.load_factor.value,
        "clip_lower_mw": construction.clip_lower_mw.value,
        "clip_upper_mw": construction.clip_upper_mw.value,
        "stress_multiplier": construction.stress_multiplier.value,
    }
    for key in required_configuration:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValidationError(f"request_configuration[{key}] must be a finite scalar")
        if not math.isclose(float(value), expected_configuration[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValidationError(f"request_configuration[{key}] does not match RequestConstruction")
    evaluation = evaluate_effective_deviation(
        reference_request,
        reported_request,
        reporter_id,
        gamma,
        metric_specification,
        threshold_policy,
        capacity_reference_mw=capacity_binding.values_mw,
        capacity_spec_hash=capacity_binding.capacity_spec_hash,
        capacity_binding=capacity_binding,
        q_mode=q_mode,
        reporting_specification=reporting_specification,
    )
    scenario_status = (
        "UNRESOLVED"
        if evaluation.classification == "UNRESOLVED"
        else "DIAGNOSTIC_ONLY"
    )
    return TypedStrategicScenario(
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        participant_ids=reference_request.participant_ids,
        reporter_id=reporter_id,
        reference_raw_request=reference_request,
        reported_raw_request=reported_request,
        gamma=gamma,
        reporting_equation_specification=reporting_specification,
        deviation_metric_specification=metric_specification,
        threshold_policy=threshold_policy,
        deviation_evaluation=evaluation,
        capacity_vector=capacity_binding.values_mw,
        capacity_binding=capacity_binding,
        reference_admitted_request_hash=reference_admitted.request_hash,
        reported_admitted_request_hash=reported_admitted.request_hash,
        classification=evaluation.classification,
        scenario_status=scenario_status,
        failure_reasons=evaluation.failure_reasons,
        diagnostic_only=evaluation.diagnostic_only,
        request_configuration=configuration,
        q_mode=q_mode,
        capacity_spec_hash=capacity_binding.capacity_spec_hash,
    )


__all__ = [
    "EffectiveDeviationMetricSpecification",
    "ReportingEquationSpecification",
    "StrategicDeviationEvaluation",
    "StrategicDeviationThresholdPolicy",
    "TypedStrategicScenario",
    "TypedStrategicScenarioRoster",
    "build_author_approved_effective_deviation_metric",
    "build_author_approved_strategic_deviation_policy",
    "build_typed_strategic_scenario",
    "build_typed_strategic_scenario_roster",
    "evaluate_effective_deviation",
]
