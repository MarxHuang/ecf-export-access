"""Stable registered enum tokens. Values are serialized exactly as specified."""
from enum import StrEnum


class QAssumptionId(StrEnum):
    Q0 = "Q0"
    Q95 = "Q95"
    MATCHED_Q = "matched-Q"
    CROSS_Q = "cross-Q"


class MarginModeId(StrEnum):
    M0 = "M0"
    M1 = "M1"
    M125 = "M125"


class MitigationModeId(StrEnum):
    NONE = "None"
    CF = "CF"
    MR_005 = "MR_005"
    MR_010 = "MR_010"
    C_005 = "C_005"
    C_010 = "C_010"


class RequestKind(StrEnum):
    REFERENCE = "reference_request"
    STRATEGIC_REPORT = "strategic_report"


class DiagnosticOrPrimary(StrEnum):
    PRIMARY = "primary"
    DIAGNOSTIC = "diagnostic"


class EvidenceTier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"
    BLOCKED = "BLOCKED"


class RawSolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


class NormalizedSolverStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INFEASIBLE = "INFEASIBLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class AllocationSide(StrEnum):
    REFERENCE = "reference"
    REPORTED = "reported"


class PairStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ALIGNED = "NOT_ALIGNED"


class ObjectiveUnitPolicy(StrEnum):
    COMMON_MW = "common_MW"
    MIXED_COMPONENTS = "mixed_components"
    UNDEFINED = "UNDEFINED"


class ScreenSide(StrEnum):
    REFERENCE = "reference"
    REPORTED = "reported"


class SplitKind(StrEnum):
    TRAIN = "train"
    HOLDOUT = "independent_holdout"
    REALIZED_ALLOCATION_AUDIT = "realized_allocation_audit"


class CalibrationProfileKind(StrEnum):
    """Registered scientific role of a calibration profile."""

    SINGLE_BUS = "single_bus"
    MULTI_BUS = "multi_bus"
    REALIZED_ALLOCATION = "realized_allocation"
    COMMON_PROXY = "common_proxy"


class MetricStatus(StrEnum):
    VALID = "VALID"
    # Retained only for deserializing historical diagnostic artifacts.  New
    # active-contract calculations emit VALID.
    DEFINED = "DEFINED"
    UNDEFINED_ALL_ZERO = "UNDEFINED_ALL_ZERO"
    # Retained only for deserializing historical diagnostic artifacts.  New
    # active-contract calculations distinguish the two empty-domain cases.
    UNDEFINED_EMPTY = "UNDEFINED_EMPTY"
    UNDEFINED_EMPTY_SET = "UNDEFINED_EMPTY_SET"
    UNDEFINED_NO_POSITIVE_CAPACITY = "UNDEFINED_NO_POSITIVE_CAPACITY"
    INVALID_NEGATIVE_CAPACITY = "INVALID_NEGATIVE_CAPACITY"


class RatingSourceType(StrEnum):
    """Provenance class for an optional engineering rating overlay."""

    INDEPENDENT_ENGINEERING = "independent_engineering"
    AUTHOR_APPROVED_ENGINEERING = "author_approved_engineering"


class PhaseModel(StrEnum):
    """Phase assumptions are explicit; they are never inferred from a MATPOWER case."""

    THREE_PHASE_BALANCED = "three_phase_balanced"
    SINGLE_PHASE = "single_phase"
    UNKNOWN = "unknown"


class RatingStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE_MISSING_ENGINEERING_RATING = "UNAVAILABLE_MISSING_ENGINEERING_RATING"


class MvaScreenStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED_MISSING_ENGINEERING_RATING = "NOT_EVALUATED_MISSING_ENGINEERING_RATING"
    NOT_EVALUATED_EMPTY = "NOT_EVALUATED_EMPTY"


class SyntheticBranchClass(StrEnum):
    """Topology-only classes used by the candidate synthetic-rate policy."""

    ROOT_FEEDER = "ROOT_FEEDER"
    TRUNK = "TRUNK"
    LATERAL = "LATERAL"
    TRANSFORMER = "TRANSFORMER"


class SyntheticRatePolicyStatus(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    FROZEN = "FROZEN"
    SUPERSEDED = "SUPERSEDED"


# Backward-compatible type-level names. Entity model classes use the shorter
# names without confusing enum values with entity instances.
QAssumption = QAssumptionId
MarginMode = MarginModeId
MitigationMode = MitigationModeId
