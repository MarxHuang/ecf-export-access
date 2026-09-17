"""Audit the V16 exact verified-telemetry assumption without rerunning solves.

The historical V16 result used the realized capability vector when evaluating
same-realization replay delivery.  The publication interface now names the
post-event input explicitly as verified available-export telemetry and freezes
the controlled-experiment assumption ``a_hat_tel(t) = a(t)``.  This audit reads
the immutable result, reconstructs every observed/replay delivery and replay
gate through that telemetry interface, and proves that the relabeling changes no
allocation, AC result, gate decision, or reported number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from r4r.externality_feedback import (
    EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID,
    assess_replay,
    fulfillment_replay_request,
    ideal_verified_available_export_telemetry,
    metered_delivery,
)
from r4r.serialization import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT / "outputs" / "externality_conditioned_feedback_v16" / "V16_RESULTS.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "externality_conditioned_feedback_v16"
    / "V16_VERIFIED_TELEMETRY_ASSUMPTION_AUDIT.json"
)
EXTERNALITY_CONTROL_ROLE = "COUNTERFACTUAL_EXTERNALITY_CONDITIONED_FEEDBACK"
AUTHORIZATION_TOLERANCE_MW = 1.0e-9
REPLAY_TOLERANCE_MW = 2.0e-10
IDENTITY_TOLERANCE_MW = 2.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _maximum_abs_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )


def _steps(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for trajectory in result["trajectory_summaries"]:
        yield from trajectory["steps"]


def audit(result_path: Path, output_path: Path) -> dict[str, Any]:
    source_sha_before = _sha256(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))

    interval_count = 0
    telemetry_record_count = 0
    ecf_interval_count = 0
    replay_attempt_count = 0
    replay_gate_open_count = 0
    invalid_telemetry_count = 0
    gate_mismatch_count = 0
    gate_reason_mismatch_count = 0
    replay_metric_mismatch_count = 0
    max_telemetry_truth_error = 0.0
    max_observed_delivery_error = 0.0
    max_replay_request_error = 0.0
    max_replay_delivery_error = 0.0
    max_replay_metric_error = 0.0

    for step in _steps(result):
        interval_count += 1
        capability = [float(value) for value in step["capability_mw"]]
        try:
            telemetry = ideal_verified_available_export_telemetry(capability)
        except ValueError:
            invalid_telemetry_count += 1
            continue
        telemetry_record_count += 1
        max_telemetry_truth_error = max(
            max_telemetry_truth_error,
            _maximum_abs_error(telemetry, capability),
        )

        observed_delivery = metered_delivery(step["allocation_mw"], capability)
        telemetry_delivery = metered_delivery(
            step["allocation_mw"], telemetry
        )
        max_observed_delivery_error = max(
            max_observed_delivery_error,
            _maximum_abs_error(observed_delivery, step["delivery_mw"]),
            _maximum_abs_error(telemetry_delivery, step["delivery_mw"]),
        )

        if step["control_role"] != EXTERNALITY_CONTROL_ROLE:
            continue
        ecf_interval_count += 1
        reconstructed_request, idle_indices = fulfillment_replay_request(
            step["effective_request_mw"],
            step["allocation_mw"],
            step["delivery_mw"],
            tolerance_mw=AUTHORIZATION_TOLERANCE_MW,
        )
        max_replay_request_error = max(
            max_replay_request_error,
            _maximum_abs_error(reconstructed_request, step["replay_request_mw"]),
        )
        if tuple(int(value) for value in step["replay_idle_indices"]) != idle_indices:
            max_replay_request_error = math.inf

        replay_attempted = bool(step["replay_attempted"])
        replay_attempt_count += int(replay_attempted)
        replay_gate_open_count += int(bool(step["replay_gate_triggered"]))
        reconstructed_replay_delivery = metered_delivery(
            step["replay_allocation_mw"], telemetry
        )
        max_replay_delivery_error = max(
            max_replay_delivery_error,
            _maximum_abs_error(
                reconstructed_replay_delivery, step["replay_delivery_mw"]
            ),
        )

        replay_valid = (
            bool(step["replay_solver_valid"])
            and bool(step["replay_ac_numerical_valid"])
            and bool(step["replay_physical_screen_pass"])
            if replay_attempted
            else True
        )
        assessment = assess_replay(
            allocation_mw=step["allocation_mw"],
            delivery_mw=step["delivery_mw"],
            replay_allocation_mw=step["replay_allocation_mw"],
            replay_delivery_mw=reconstructed_replay_delivery,
            idle_indices=idle_indices,
            replay_valid=replay_valid,
            tolerance_mw=REPLAY_TOLERANCE_MW,
        )
        if assessment.gate_triggered != bool(step["replay_gate_triggered"]):
            gate_mismatch_count += 1
        if assessment.gate_reason != step["replay_gate_reason"]:
            gate_reason_mismatch_count += 1

        metric_pairs = (
            (assessment.total_idle_award_mw, step["replay_total_idle_award_mw"]),
            (
                assessment.outsider_allocation_gain_mw,
                step["replay_outsider_allocation_gain_mw"],
            ),
            (
                assessment.outsider_delivery_gain_mw,
                step["replay_outsider_delivery_gain_mw"],
            ),
            (
                assessment.total_delivery_gain_mw,
                step["replay_total_delivery_gain_mw"],
            ),
            (
                assessment.allocation_redistribution_mw,
                step["replay_allocation_redistribution_mw"],
            ),
        )
        metric_error = max(
            (abs(float(left) - float(right)) for left, right in metric_pairs),
            default=0.0,
        )
        max_replay_metric_error = max(max_replay_metric_error, metric_error)
        replay_metric_mismatch_count += int(metric_error > IDENTITY_TOLERANCE_MW)

    source_sha_after = _sha256(result_path)
    expected_interval_count = int(result["interval_record_count"])
    expected_replay_attempt_count = int(result["replay_attempt_count"])
    expected_gate_open_count = int(result["replay_gate_open_count"])
    gate_results = {
        "SOURCE_RESULT_TREE_UNCHANGED": source_sha_before == source_sha_after,
        "INTERVAL_RECORD_COMPLETENESS": interval_count == expected_interval_count,
        "TELEMETRY_RECORD_COMPLETENESS": telemetry_record_count == interval_count,
        "TELEMETRY_VALUES_FINITE_NONNEGATIVE": invalid_telemetry_count == 0,
        "EXACT_TELEMETRY_TRUTH_BINDING": max_telemetry_truth_error == 0.0,
        "OBSERVED_DELIVERY_IDENTITY": (
            max_observed_delivery_error <= IDENTITY_TOLERANCE_MW
        ),
        "REPLAY_ATTEMPT_COMPLETENESS": (
            replay_attempt_count == expected_replay_attempt_count
        ),
        "REPLAY_REQUEST_IDENTITY": (
            max_replay_request_error <= IDENTITY_TOLERANCE_MW
        ),
        "REPLAY_DELIVERY_IDENTITY": (
            max_replay_delivery_error <= IDENTITY_TOLERANCE_MW
        ),
        "REPLAY_METRIC_IDENTITY": (
            replay_metric_mismatch_count == 0
            and max_replay_metric_error <= IDENTITY_TOLERANCE_MW
        ),
        "REPLAY_GATE_IDENTITY": (
            gate_mismatch_count == 0
            and replay_gate_open_count == expected_gate_open_count
        ),
        "REPLAY_GATE_REASON_IDENTITY": gate_reason_mismatch_count == 0,
        "NO_SOLVER_OR_AC_REEXECUTION": True,
    }
    payload: dict[str, Any] = {
        "serialization_id": "r4r.v16_verified_telemetry_assumption_audit.v1",
        "status": (
            "V16_VERIFIED_TELEMETRY_ASSUMPTION_AUDIT_COMPLETE"
            if all(gate_results.values())
            else "V16_VERIFIED_TELEMETRY_ASSUMPTION_AUDIT_FAILED"
        ),
        "scientific_role": "EXACT_TELEMETRY_INTERFACE_EQUIVALENCE_AUDIT_NO_NEW_SOLVES",
        "assumption_id": EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID,
        "assumption": (
            "For every participant and completed interval, verified post-event "
            "available-export telemetry equals realized simulation capability "
            "componentwise: a_hat_i^tel(t)=a_i(t)."
        ),
        "information_boundary": {
            "feedback_receives": [
                "submitted request",
                "registered capacity",
                "completed allocation",
                "metered delivery",
                "verified post-event available-export telemetry",
                "frozen allocation and network models",
            ],
            "feedback_does_not_receive": [
                "private signal s_i(t)",
                "conditional distribution F_i(.|s_i(t))",
                "pre-allocation realization of available export",
            ],
        },
        "source_binding": {
            "v16_design_hash": result["design_hash"],
            "v16_result_hash": result["result_hash"],
            "v16_result_file_sha256": source_sha_before,
        },
        "counts": {
            "interval_records": interval_count,
            "telemetry_records": telemetry_record_count,
            "ecf_interval_records": ecf_interval_count,
            "replay_attempts": replay_attempt_count,
            "replay_gates_open": replay_gate_open_count,
        },
        "maximum_absolute_errors_mw": {
            "telemetry_minus_simulation_truth": max_telemetry_truth_error,
            "observed_delivery_reconstruction": max_observed_delivery_error,
            "replay_request_reconstruction": max_replay_request_error,
            "replay_delivery_reconstruction": max_replay_delivery_error,
            "replay_metric_reconstruction": max_replay_metric_error,
        },
        "mismatch_counts": {
            "invalid_telemetry_records": invalid_telemetry_count,
            "replay_metric_records": replay_metric_mismatch_count,
            "replay_gate_decisions": gate_mismatch_count,
            "replay_gate_reasons": gate_reason_mismatch_count,
        },
        "gate_results": gate_results,
        "scope_limits": [
            "The audit validates the registered ideal verified-telemetry case only.",
            "It does not establish field estimator accuracy, communication latency, dropout resilience, or privacy governance.",
            "Those deployment effects are outside the numerical scope of the present paper.",
        ],
        "audit_hash": None,
    }
    payload["audit_hash"] = canonical_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# V16 verified-telemetry assumption audit",
                "",
                f"- Status: `{payload['status']}`",
                f"- Assumption: `{EXACT_VERIFIED_TELEMETRY_ASSUMPTION_ID}` with `a_hat_tel(t)=a(t)` componentwise.",
                f"- Interval records checked: `{interval_count}`.",
                f"- Replay attempts checked: `{replay_attempt_count}`; gates reproduced: `{replay_gate_open_count}` open.",
                f"- Maximum observed-delivery reconstruction error: `{max_observed_delivery_error:.3e}` MW.",
                f"- Maximum replay-delivery reconstruction error: `{max_replay_delivery_error:.3e}` MW.",
                "- Solver and AC calculations reexecuted: `False`.",
                "- Scope: exact verified telemetry only; error, delay, dropout, and field estimator validation remain outside this paper.",
                f"- Audit hash: `{payload['audit_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = audit(args.result.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "interval_records": payload["counts"]["interval_records"],
                "replay_attempts": payload["counts"]["replay_attempts"],
                "audit_hash": payload["audit_hash"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
