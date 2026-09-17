"""Frozen, scenario-defined two-ended MVA screen support.

The raw MATPOWER RATE columns are unavailable for IEEE-141.  This module
therefore models a *synthetic scenario policy* as a separate, immutable input.
It is deliberately not an engineering ampacity model: each rate is derived
from a development-only anchor and a pre-registered class headroom rule, then
the frozen policy is used without reading evaluation or mitigation results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from r4r.ac_solver import ACPowerFlowSolution
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.serialization import canonical_hash
from r4r.types import (
    FiniteFloat,
    Identifier,
    IdentifierVector,
    NonNegativeFloat,
    QAssumption,
    Sha256,
    SyntheticBranchClass,
    SyntheticRatePolicyStatus,
)


@dataclass(frozen=True, slots=True)
class SyntheticRateBranch(ContractModel):
    """One topology-labelled synthetic continuous MVA limit."""

    branch_id: Identifier
    from_bus_id: Identifier
    to_bus_id: Identifier
    branch_class: SyntheticBranchClass
    anchor_mva: NonNegativeFloat
    minimum_mva: NonNegativeFloat
    headroom_ratio: NonNegativeFloat
    rate_a_mva: NonNegativeFloat
    serialization_id = "synthetic_rate_branch.v1"

    def __post_init__(self) -> None:
        for name in ("branch_id", "from_bus_id", "to_bus_id"):
            if not isinstance(getattr(self, name), Identifier):
                raise ValidationError(f"{name} must be Identifier")
        if self.from_bus_id == self.to_bus_id:
            raise ValidationError("synthetic branch endpoints must be distinct")
        if not isinstance(self.branch_class, SyntheticBranchClass):
            raise ValidationError("branch_class must be a registered topology class")
        for name in ("anchor_mva", "minimum_mva", "headroom_ratio", "rate_a_mva"):
            if not isinstance(getattr(self, name), NonNegativeFloat):
                raise ValidationError(f"{name} must be NonNegativeFloat")
        expected = max(
            self.minimum_mva.value,
            (1.0 + self.headroom_ratio.value) * self.anchor_mva.value,
        )
        if not math.isclose(self.rate_a_mva.value, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError(
                "rate_a_mva must equal max(minimum_mva, (1+headroom_ratio)*anchor_mva)"
            )

    @classmethod
    def from_anchor(
        cls,
        *,
        branch_id: Identifier,
        from_bus_id: Identifier,
        to_bus_id: Identifier,
        branch_class: SyntheticBranchClass,
        anchor_mva: float,
        minimum_mva: float,
        headroom_ratio: float,
    ) -> "SyntheticRateBranch":
        anchor = NonNegativeFloat(anchor_mva)
        minimum = NonNegativeFloat(minimum_mva)
        headroom = NonNegativeFloat(headroom_ratio)
        return cls(
            branch_id=branch_id,
            from_bus_id=from_bus_id,
            to_bus_id=to_bus_id,
            branch_class=branch_class,
            anchor_mva=anchor,
            minimum_mva=minimum,
            headroom_ratio=headroom,
            rate_a_mva=NonNegativeFloat(max(minimum.value, (1.0 + headroom.value) * anchor.value)),
        )


@dataclass(frozen=True, slots=True)
class SyntheticRatePolicy(ContractModel):
    """Development/evaluation-separated synthetic MVA rating policy."""

    policy_id: Identifier
    version: Identifier
    network_hash: Sha256
    raw_case_hash: Sha256
    q_modes: tuple[QAssumption, ...]
    development_scenario_ids: IdentifierVector
    evaluation_scenario_ids: IdentifierVector
    branch_ratings: tuple[SyntheticRateBranch, ...]
    voltage_lower_limit_pu: FiniteFloat
    voltage_upper_limit_pu: FiniteFloat
    branch_tolerance_mva: NonNegativeFloat
    voltage_tolerance_pu: NonNegativeFloat
    development_input_hash: Sha256
    selection_rule: Identifier
    status: SyntheticRatePolicyStatus
    diagnostic_only: bool = True
    primary_evidence_eligible: bool = False
    serialization_id = "synthetic_rate_policy.v1"

    def __post_init__(self) -> None:
        for name in ("policy_id", "version", "selection_rule"):
            if not isinstance(getattr(self, name), Identifier):
                raise ValidationError(f"{name} must be Identifier")
        for name in ("network_hash", "raw_case_hash", "development_input_hash"):
            if not isinstance(getattr(self, name), Sha256):
                raise ValidationError(f"{name} must be Sha256")
        if not isinstance(self.q_modes, tuple) or tuple(self.q_modes) != (QAssumption.Q0, QAssumption.Q95):
            raise ValidationError("synthetic policy must bind Q0 and Q95 in that order")
        if not isinstance(self.development_scenario_ids, IdentifierVector) or not self.development_scenario_ids.values:
            raise ValidationError("development scenario IDs must be non-empty")
        if not isinstance(self.evaluation_scenario_ids, IdentifierVector) or not self.evaluation_scenario_ids.values:
            raise ValidationError("evaluation scenario IDs must be non-empty")
        if set(self.development_scenario_ids.values) & set(self.evaluation_scenario_ids.values):
            raise ValidationError("development/evaluation scenario sets must be disjoint")
        if not isinstance(self.branch_ratings, tuple) or not self.branch_ratings:
            raise ValidationError("branch_ratings must be non-empty")
        if any(not isinstance(item, SyntheticRateBranch) for item in self.branch_ratings):
            raise ValidationError("branch_ratings must contain SyntheticRateBranch values")
        branch_ids = [item.branch_id for item in self.branch_ratings]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValidationError("branch IDs must be unique")
        if self.voltage_lower_limit_pu.value >= self.voltage_upper_limit_pu.value:
            raise ValidationError("voltage lower bound must be below upper bound")
        if not isinstance(self.status, SyntheticRatePolicyStatus):
            raise ValidationError("status must be a registered synthetic policy status")
        if self.diagnostic_only is not True or self.primary_evidence_eligible is not False:
            raise ValidationError("synthetic rate policy is diagnostic-only until promotion gates are implemented")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "policy_id": self.policy_id.to_json(),
            "version": self.version.to_json(),
            "network_hash": self.network_hash.to_json(),
            "raw_case_hash": self.raw_case_hash.to_json(),
            "q_modes": [item.value for item in self.q_modes],
            "development_scenario_ids": self.development_scenario_ids.to_json(),
            "evaluation_scenario_ids": self.evaluation_scenario_ids.to_json(),
            "branch_ratings": [item.to_json() for item in self.branch_ratings],
            "voltage_lower_limit_pu": self.voltage_lower_limit_pu.to_json(),
            "voltage_upper_limit_pu": self.voltage_upper_limit_pu.to_json(),
            "branch_tolerance_mva": self.branch_tolerance_mva.to_json(),
            "voltage_tolerance_pu": self.voltage_tolerance_pu.to_json(),
            "development_input_hash": self.development_input_hash.to_json(),
            "selection_rule": self.selection_rule.to_json(),
            "status": self.status.value,
            "diagnostic_only": self.diagnostic_only,
            "primary_evidence_eligible": self.primary_evidence_eligible,
        }

    @property
    def policy_hash(self) -> Sha256:
        return Sha256(canonical_hash(self.to_json()))

    def freeze(self) -> "SyntheticRatePolicy":
        """Return a new policy version marked frozen; no fields are mutated."""

        return replace(self, status=SyntheticRatePolicyStatus.FROZEN)

    def assert_evaluation_scenario(self, scenario_id: Identifier) -> None:
        if scenario_id not in self.evaluation_scenario_ids.values:
            raise ValidationError("scenario is not in the frozen evaluation split")


def two_end_apparent_power_mva(solution: ACPowerFlowSolution) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return exact two-ended apparent-power magnitudes from an AC solution."""

    if not solution.solver_result.solve_success:
        raise ValidationError("cannot compute MVA observables from a failed AC solve")
    if not (
        len(solution.branch_p_from_mw.values)
        == len(solution.branch_q_from_mvar.values)
        == len(solution.branch_p_to_mw.values)
        == len(solution.branch_q_to_mvar.values)
    ):
        raise ValidationError("AC branch flow vectors have inconsistent shapes")
    from_values = tuple(
        math.hypot(p, q)
        for p, q in zip(solution.branch_p_from_mw.values, solution.branch_q_from_mvar.values)
    )
    to_values = tuple(
        math.hypot(p, q)
        for p, q in zip(solution.branch_p_to_mw.values, solution.branch_q_to_mvar.values)
    )
    return from_values, to_values


def screen_solution_against_policy(
    solution: ACPowerFlowSolution,
    policy: SyntheticRatePolicy,
) -> dict[str, Any]:
    """Evaluate every policy branch and every bus without clipping or sign loss."""

    if policy.status is not SyntheticRatePolicyStatus.FROZEN:
        raise ValidationError("only a frozen synthetic policy may screen evaluation rows")
    if not solution.solver_result.solve_success:
        return {
            "status": "NOT_EVALUATED_AC_NONCONVERGENCE",
            "screen_pass": False,
            "quantities_valid": False,
            "solution_hash": solution.solution_hash.to_json(),
        }
    from_values, to_values = two_end_apparent_power_mva(solution)
    if len(from_values) != len(policy.branch_ratings):
        raise ValidationError("policy branch count does not match AC branch count")
    branch_rows: list[dict[str, Any]] = []
    for rating, s_from, s_to in zip(policy.branch_ratings, from_values, to_values):
        s_max = max(s_from, s_to)
        margin = rating.rate_a_mva.value - s_max
        branch_rows.append({
            "branch_id": rating.branch_id.to_json(),
            "s_from_mva": s_from,
            "s_to_mva": s_to,
            "s_max_mva": s_max,
            "rate_a_mva": rating.rate_a_mva.value,
            "margin_mva": margin,
            "from_is_limiting": s_from >= s_to,
            "to_is_limiting": s_to >= s_from,
            "pass": margin >= -policy.branch_tolerance_mva.value,
        })
    voltage_rows = [
        {
            "bus_index": index + 1,
            "voltage_pu": value,
            "lower_limit_pu": policy.voltage_lower_limit_pu.value,
            "upper_limit_pu": policy.voltage_upper_limit_pu.value,
            "pass": (
                value >= policy.voltage_lower_limit_pu.value - policy.voltage_tolerance_pu.value
                and value <= policy.voltage_upper_limit_pu.value + policy.voltage_tolerance_pu.value
            ),
        }
        for index, value in enumerate(solution.bus_voltage_pu.values)
    ]
    branch_pass = all(row["pass"] for row in branch_rows)
    voltage_pass = all(row["pass"] for row in voltage_rows)
    return {
        "status": "PASS" if branch_pass and voltage_pass else "FAIL",
        "screen_pass": branch_pass and voltage_pass,
        "quantities_valid": True,
        "solution_hash": solution.solution_hash.to_json(),
        "branch_pass": branch_pass,
        "voltage_pass": voltage_pass,
        "branch_rows": branch_rows,
        "voltage_rows": voltage_rows,
        "maximum_violation_mva": max(
            (max(0.0, -row["margin_mva"]) for row in branch_rows),
            default=0.0,
        ),
        "offending_branch_ids": [row["branch_id"] for row in branch_rows if not row["pass"]],
        "offending_bus_indices": [row["bus_index"] for row in voltage_rows if not row["pass"]],
    }


def load_synthetic_rate_policy(payload: Mapping[str, Any]) -> SyntheticRatePolicy:
    """Strictly load a candidate/frozen policy and verify its content hash."""

    if not isinstance(payload, Mapping):
        raise ValidationError("synthetic rate policy payload must be a mapping")
    required = {
        "serialization_id", "policy_id", "version", "network_hash", "raw_case_hash",
        "q_modes", "development_scenario_ids", "evaluation_scenario_ids", "branch_ratings",
        "voltage_lower_limit_pu", "voltage_upper_limit_pu", "branch_tolerance_mva",
        "voltage_tolerance_pu", "development_input_hash", "selection_rule", "status",
        "diagnostic_only", "primary_evidence_eligible",
    }
    missing = required - set(payload)
    if missing:
        raise ValidationError(f"synthetic rate policy missing fields: {sorted(missing)}")
    if payload["serialization_id"] != "synthetic_rate_policy.v1":
        raise ValidationError("unsupported synthetic rate policy serialization")

    def ident(value: Any, field: str) -> Identifier:
        if not isinstance(value, str):
            raise ValidationError(f"{field} must be a string identifier")
        return Identifier(value)

    def sha(value: Any, field: str) -> Sha256:
        if not isinstance(value, str):
            raise ValidationError(f"{field} must be a SHA-256 string")
        return Sha256(value)

    def id_vector(value: Any, field: str) -> IdentifierVector:
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be an array")
        return IdentifierVector(ident(item, field) for item in value)

    rows: list[SyntheticRateBranch] = []
    raw_rows = payload["branch_ratings"]
    if not isinstance(raw_rows, list):
        raise ValidationError("branch_ratings must be an array")
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValidationError("branch_ratings must contain objects")
        rows.append(SyntheticRateBranch(
            branch_id=ident(raw["branch_id"], "branch_id"),
            from_bus_id=ident(raw["from_bus_id"], "from_bus_id"),
            to_bus_id=ident(raw["to_bus_id"], "to_bus_id"),
            branch_class=SyntheticBranchClass(raw["branch_class"]),
            anchor_mva=NonNegativeFloat(raw["anchor_mva"]),
            minimum_mva=NonNegativeFloat(raw["minimum_mva"]),
            headroom_ratio=NonNegativeFloat(raw["headroom_ratio"]),
            rate_a_mva=NonNegativeFloat(raw["rate_a_mva"]),
        ))
    q_modes = payload["q_modes"]
    if not isinstance(q_modes, list):
        raise ValidationError("q_modes must be an array")
    policy = SyntheticRatePolicy(
        policy_id=ident(payload["policy_id"], "policy_id"),
        version=ident(payload["version"], "version"),
        network_hash=sha(payload["network_hash"], "network_hash"),
        raw_case_hash=sha(payload["raw_case_hash"], "raw_case_hash"),
        q_modes=tuple(QAssumption(item) for item in q_modes),
        development_scenario_ids=id_vector(payload["development_scenario_ids"], "development_scenario_ids"),
        evaluation_scenario_ids=id_vector(payload["evaluation_scenario_ids"], "evaluation_scenario_ids"),
        branch_ratings=tuple(rows),
        voltage_lower_limit_pu=FiniteFloat(payload["voltage_lower_limit_pu"]),
        voltage_upper_limit_pu=FiniteFloat(payload["voltage_upper_limit_pu"]),
        branch_tolerance_mva=NonNegativeFloat(payload["branch_tolerance_mva"]),
        voltage_tolerance_pu=NonNegativeFloat(payload["voltage_tolerance_pu"]),
        development_input_hash=sha(payload["development_input_hash"], "development_input_hash"),
        selection_rule=ident(payload["selection_rule"], "selection_rule"),
        status=SyntheticRatePolicyStatus(payload["status"]),
        diagnostic_only=payload["diagnostic_only"],
        primary_evidence_eligible=payload["primary_evidence_eligible"],
    )
    declared_hash = payload.get("policy_hash")
    if declared_hash is not None and declared_hash != policy.policy_hash.to_json():
        raise ValidationError("synthetic rate policy hash mismatch")
    return policy


__all__ = [
    "SyntheticRateBranch",
    "SyntheticRatePolicy",
    "load_synthetic_rate_policy",
    "screen_solution_against_policy",
    "two_end_apparent_power_mva",
]
