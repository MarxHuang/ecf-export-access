"""Immutable Round 1 domain objects; no numerical business implementation."""

from .base import ContractModel, model_hash, validate_model
from .common import GateResult, MarginMode, MitigationMode, OperatingPoint, ProvenanceFields, QAssumption
from .network import Branch, Bus, NetworkModel, ProxyModel
from .participants import Participant, ParticipantSet
from .optimization import ObjectiveDefinition, OptimizationProblem, SolverResult
from .scenarios import AllocationPairResult, AllocationVector, CapacityScenario, RequestScenario, ScenarioDefinition
from .ac import ACScreenResult, ACSolverCrosscheckResult
from .calibration import CalibrationProfile, CalibrationResult, CalibrationStratum
from .evidence import EvidenceDecision, JainMetricResult, MitigationResult
from .provenance import RunMetadata
from .diagnostic import DiagnosticParameterIdentity

__all__ = [
    "ACScreenResult", "ACSolverCrosscheckResult", "AllocationPairResult", "AllocationVector", "Branch", "Bus", "CalibrationProfile", "CalibrationResult", "CalibrationStratum", "CapacityScenario", "ContractModel", "EvidenceDecision", "GateResult", "JainMetricResult", "MarginMode",
    "MitigationMode", "NetworkModel", "OperatingPoint", "Participant",
    "MitigationResult", "NetworkModel", "ParticipantSet", "ProvenanceFields", "ProxyModel", "QAssumption", "RequestScenario", "RunMetadata", "ScenarioDefinition", "ObjectiveDefinition", "OptimizationProblem", "SolverResult", "model_hash", "validate_model",
    "DiagnosticParameterIdentity",
]
