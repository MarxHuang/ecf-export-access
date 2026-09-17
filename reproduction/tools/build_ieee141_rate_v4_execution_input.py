"""Build the fresh, pre-result IEEE-141 RATE-V4 design input.

This is an intentionally small scenario-design builder.  It materializes the
input consumed by :mod:`tools.ieee141_rate_v4_design_ac_runner` entirely from
the raw pinned case, the frozen 0.70 operating point, and the paper's Model-L
capacity multipliers.  It does not read any historical output, RATE, solver
result, mitigation result, or evaluation result.

The strategic vector is a *pre-registered development stress vector*, not a
claim about the final allocation mechanism.  The RATE anchor profiles use the
L1/L2 capacities for the DEC015 reference solve; the stress vector deliberately
uses a separately declared L5 reported-capacity envelope (half of that envelope
for ordinary participants and the full envelope for the named reporter).  This
keeps the physical stress test out of the anchor QCP while making the
development quality gate capable of detecting non-trivial export congestion.
The stress vector is excluded from the subsequent source, calibration,
evaluation, and mitigation roles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from r4r.network_parser import ParsedMatpowerCase, parse_matpower_case
from r4r.operating_point import CANONICAL_LOAD_SCALE
from r4r.serialization import canonical_dumps, canonical_hash
from tools.diagnostic_ieee141_context import IEEE141_CASE_SHA256
from tools.ieee141_rate_v4_design_ac_runner import (
    PARTICIPANT_BUS_IDS,
    PARTICIPANT_IDS,
    _READ_ASSERTION_FIELDS,
    build_deterministic_rate_design_root_roles,
)
from tools.ieee141_rate_v4_groundwork import validate_rate_design_protocol


SERIALIZATION_ID = "case141_rate_v4_design_execution_input.v1"
MANIFEST_SERIALIZATION_ID = "case141_rate_v4_design_execution_input_manifest.v1"
RATE_DESIGN_MODEL_L_MULTIPLIERS = (1.0, 2.0)
REPORTER_IDS = ("p001", "p006", "p011", "p016", "p021", "p026", "p030", "p002", "p003")
MODEL_L_MULTIPLIERS = RATE_DESIGN_MODEL_L_MULTIPLIERS
ROOT_COUNT = len(MODEL_L_MULTIPLIERS) * len(REPORTER_IDS)
RATE_DESIGN_ROOT_COUNT = 6


def _fresh_read_assertions() -> dict[str, bool]:
    return {field: False for field in _READ_ASSERTION_FIELDS}


def _model_l_capacity(parsed: ParsedMatpowerCase, multiplier: float) -> list[float]:
    bus_ids = [int(bus.bus_id) for bus in parsed.network.buses]
    raw_load = dict(zip(bus_ids, parsed.p_load_mw.values))
    return [float(multiplier) * CANONICAL_LOAD_SCALE * float(raw_load[bus]) for bus in PARTICIPANT_BUS_IDS]


def _root_descriptors() -> list[dict[str, Any]]:
    return [
        {
            "root_id": f"ratev4_l{multiplier:g}_{reporter_id}",
            "capacity_model": "MODEL_L",
            "capacity_multiplier": multiplier,
            "reporter_id": reporter_id,
        }
        for multiplier in MODEL_L_MULTIPLIERS
        for reporter_id in REPORTER_IDS
    ]


def rate_v4_q95_specification() -> dict[str, Any]:
    """Return the frozen Q95 operating assumption used by all RATE-V4 runs."""

    return {
        "power_factor": 0.95,
        "q_sign_convention": "NEGATIVE_BUS_INJECTION",
        "magnitude_rule": "FIXED_POWER_FACTOR_PARTICIPANT_REACTIVE_INJECTION",
    }


def rate_v4_anchor_execution() -> dict[str, Any]:
    """Return the frozen primary/independent AC execution configuration."""

    return {
        "finite_difference_delta_mw": 0.001,
        "anchor_branch_margin_mva": 1.0e-5,
        # Baseline Q0 voltage is only about 5.45e-4 pu above Vmin; this
        # inward margin is deliberately below that measured source slack.
        "anchor_voltage_margin_pu": 1.0e-4,
        "primary_ac_tolerance": 1.0e-8,
        "primary_ac_max_iterations": 100,
        "independent_ac_tolerance": 1.0e-8,
        "independent_ac_max_iterations": 100,
        "crosscheck_tolerances": {
            "voltage_pu": 1.0e-5,
            "angle_deg": 1.0e-3,
            "branch_p_mw": 1.0e-4,
            "branch_q_mvar": 1.0e-4,
            "bus_p_mismatch_mw": 1.0e-5,
            "bus_q_mismatch_mvar": 1.0e-5,
            "slack_p_mw": 1.0e-4,
            "slack_q_mvar": 1.0e-4,
            "active_loss_mw": 1.0e-4,
            "reactive_loss_mvar": 1.0e-4,
        },
        "quality_mva_tolerance": 1.0e-7,
        "quality_voltage_tolerance_pu": 1.0e-7,
        # DEC015 tau_H is a scientific hierarchy tolerance, so the QCP
        # solver must be tighter than its quarter-gap certificate rather
        # than implicitly relaxing that contract through barrier error.
        "dec015_solver_feasibility_tolerance": 1.0e-9,
        "dec015_solver_optimality_tolerance": 1.0e-9,
        "dec015_solver_threads": 8,
    }


def build_rate_v4_execution_input(
    *, parsed: ParsedMatpowerCase, protocol: Mapping[str, Any], rate_design_root_count: int = RATE_DESIGN_ROOT_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return runtime input plus an auditable, non-runtime construction manifest."""

    protocol_hash = validate_rate_design_protocol(protocol)
    descriptors = _root_descriptors()
    root_ids = [item["root_id"] for item in descriptors]
    if len(root_ids) != ROOT_COUNT or len(set(root_ids)) != ROOT_COUNT:
        raise ValueError("RATE_V4_ROOT_ROSTER_CONSTRUCTION_INVALID")
    split = build_deterministic_rate_design_root_roles(
        protocol=protocol, root_profile_ids=root_ids, rate_design_root_count=rate_design_root_count,
    )
    design_roots = set(split["root_role_ids"]["RATE_DESIGN"])
    profile_rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if descriptor["root_id"] not in design_roots:
            continue
        capacity = _model_l_capacity(parsed, float(descriptor["capacity_multiplier"]))
        reporter_index = PARTICIPANT_IDS.index(descriptor["reporter_id"])
        strategic_capacity = _model_l_capacity(parsed, 5.0)
        strategic = [0.5 * value for value in strategic_capacity]
        strategic[reporter_index] = strategic_capacity[reporter_index]
        if (any(value <= 0.0 for value in capacity)
                or any(value <= 0.0 for value in strategic_capacity)
                or any(value < 0.0 or value > limit for value, limit in zip(strategic, strategic_capacity))):
            raise ValueError("RATE_V4_PRE_REGISTERED_DEVELOPMENT_VECTOR_INVALID")
        profile_rows.append({
            "root_id": descriptor["root_id"],
            "capacity_mw": capacity,
            "honest_request_mw": list(capacity),
            "strategic_capacity_mw": strategic_capacity,
            "strategic_unmitigated_allocation_mw": strategic,
            "strategic_allocation_provenance": "PRE_REGISTERED_UNMITIGATED_STRATEGIC_DEVELOPMENT_ALLOCATION",
        })
    if {row["root_id"] for row in profile_rows} != design_roots:
        raise ValueError("RATE_V4_RATE_DESIGN_PROFILE_COVERAGE_INVALID")
    input_payload: dict[str, Any] = {
        "serialization_id": SERIALIZATION_ID,
        "status": "FRESH_RATE_DESIGN_EXECUTION_INPUT",
        "root_profile_ids": root_ids,
        "rate_design_root_count": int(rate_design_root_count),
        "rate_design_profiles": sorted(profile_rows, key=lambda row: row["root_id"]),
        "participant_mapping": {
            "participant_ids": list(PARTICIPANT_IDS),
            "participant_bus_ids": list(PARTICIPANT_BUS_IDS),
            "mapping_policy": "CASE141_FROZEN_TYPED_PARTICIPANT_MAPPING_V10",
        },
        "q95_specification": rate_v4_q95_specification(),
        "anchor_execution": rate_v4_anchor_execution(),
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
    }
    manifest = {
        "serialization_id": MANIFEST_SERIALIZATION_ID,
        "status": "PRE_RESULT_RATE_DESIGN_INPUT_CONSTRUCTION_ONLY",
        "raw_case_hash": parsed.source_hash.to_json(),
        "protocol_hash": protocol_hash,
        "load_scale": CANONICAL_LOAD_SCALE,
        "capacity_construction": {
            "model": "MODEL_L",
            "formula": "C_i = c_L * P_load_i_at_0_70",
            "multipliers": list(MODEL_L_MULTIPLIERS),
        },
        "strategic_stress_construction": {
            "reported_capacity_model": "MODEL_L",
            "reported_capacity_multiplier": 5.0,
            "formula": "C_reported_i = 5 * P_load_i_at_0_70; x_i=0.5*C_reported_i for i!=reporter; x_reporter=C_reported_reporter",
            "role": "PRE_REGISTERED_DEVELOPMENT_QUALITY_ONLY",
            "true_anchor_capacity_is_not_replaced": True,
            "not_a_final_allocation_rule": True,
        },
        "root_roster": descriptors,
        "root_split": split,
        "execution_input_hash": canonical_hash(input_payload),
        "input_read_assertions": _fresh_read_assertions(),
        "historical_output_reused": False,
        "primary_evidence_eligible": False,
        "manuscript_evidence_eligible": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return input_payload, manifest


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--rate-design-root-count", type=int, default=RATE_DESIGN_ROOT_COUNT)
    parser.add_argument("--expected-case-sha256", default=IEEE141_CASE_SHA256)
    args = parser.parse_args(argv)
    parsed = parse_matpower_case(args.case, expected_sha256=args.expected_case_sha256)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    input_payload, manifest = build_rate_v4_execution_input(
        parsed=parsed, protocol=protocol, rate_design_root_count=args.rate_design_root_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_dumps(input_payload) + "\n", encoding="utf-8")
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
    print(canonical_dumps({
        "status": manifest["status"], "execution_input_hash": manifest["execution_input_hash"],
        "manifest_hash": manifest["manifest_hash"], "rate_design_root_count": len(input_payload["rate_design_profiles"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_rate_v4_execution_input", "main", "rate_v4_anchor_execution",
    "rate_v4_q95_specification",
]
