"""Run the explicit Q0/Q95 M0 IEEE141 unilateral reporter diagnostic scan.

This is a numerical implementation entry point, not the manuscript runner.
All generated records remain diagnostic-only.  The scenario values below are
the explicit first scan requested for the clean-room rebuild; changing them
requires changing the invocation/configuration and regenerating the output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from r4r.reactive_spec import ReactiveInjectionSpecification
from r4r.types import FiniteFloat, FloatVector, Identifier, QAssumption
from tools.diagnostic_ieee141_context import build_ieee141_diagnostic_input_context
from tools.diagnostic_pair_batch import DiagnosticPhysicalLimits
from tools.ieee141_diagnostic_batch import (
    IEEE141BatchRequestConfig,
    IEEE141DiagnosticBatchFactory,
    build_unilateral_reporter_scan_scenarios,
    write_diagnostic_grid_result,
)
from r4r.serialization import canonical_dumps


PARTICIPANT_BUSES = (
    8, 9, 12, 13, 17, 20, 21, 23, 26, 27, 29, 32, 34, 35, 36,
    37, 39, 41, 44, 48, 49, 51, 52, 53, 56, 58, 59, 61, 62, 64,
)


def reporter_scan_grid_id(q_mode: str, start: int, stop: int) -> Identifier:
    """Return a Q-bound grid identifier for a reporter-scan interval.

    The previous implementation hard-coded ``q0`` in this identifier, which
    allowed a Q95 payload to carry a Q0-looking grid id.  Grid ids are part of
    the typed input join, so the label must be derived from the requested Q
    mode rather than from a caller's output filename.
    """

    q_label = str(q_mode).lower()
    if q_label not in {"q0", "q95"}:
        raise ValueError("q_mode must be Q0 or Q95")
    if start < 0 or stop <= start:
        raise ValueError("reporter-scan interval must satisfy 0 <= start < stop")
    return Identifier(f"ieee141-{q_label}-m0-reporter-scan-{start + 1:02d}-{stop:02d}")


def _q0() -> ReactiveInjectionSpecification:
    return ReactiveInjectionSpecification(
        q_mode_id=QAssumption.Q0,
        power_factor=None,
        magnitude_rule_id=Identifier("q0_zero_participant_q"),
        q_sign_convention="UNSPECIFIED",
        positive_p_definition="bus_net_injection_positive",
        positive_q_definition="bus_net_injection_positive",
        capability_limit_policy=Identifier("NO_CAPABILITY_CLIP"),
        scientific_status="CANDIDATE",
    )


def _q95() -> ReactiveInjectionSpecification:
    return ReactiveInjectionSpecification(
        q_mode_id=QAssumption.Q95,
        power_factor=0.95,
        magnitude_rule_id=Identifier("q95_fixed_power_factor_magnitude"),
        q_sign_convention="NEGATIVE_BUS_INJECTION",
        positive_p_definition="bus_net_injection_positive",
        positive_q_definition="bus_net_injection_positive",
        capability_limit_policy=Identifier("NO_CAPABILITY_CLIP"),
        scientific_status="CANDIDATE",
    )


def build_ieee141_m0_factory(
    case_path: Path,
    *,
    q_mode: QAssumption = QAssumption.Q0,
    physical_limits: DiagnosticPhysicalLimits | None = None,
    finite_difference_workers: int = 1,
) -> IEEE141DiagnosticBatchFactory:
    """Build explicit Q0 or Q95 M0 inputs from the pinned case."""

    if q_mode is QAssumption.Q0:
        q_spec = _q0()
    elif q_mode is QAssumption.Q95:
        q_spec = _q95()
    else:
        raise ValueError("q_mode must be Q0 or Q95")
    context = build_ieee141_diagnostic_input_context(
        case_path,
        rating_mw_by_bus={bus: 1.0 for bus in PARTICIPANT_BUSES},
        availability_mw_by_bus={bus: 1.0 for bus in PARTICIPANT_BUSES},
        reactive_specification=q_spec,
    )
    loads_by_bus = dict(zip(
        (bus.bus_id for bus in context.parsed_case.network.buses),
        context.operating_point.p_load.values,
    ))
    participant_loads = tuple(
        loads_by_bus[int(participant.bus_id.object_id.value)]
        for participant in context.selection.participants
    )
    normalization_scale = max(participant_loads)
    request_config = IEEE141BatchRequestConfig(
        normalized_participant_load_mw=FloatVector(value / normalization_scale for value in participant_loads),
        intercept_mw=FiniteFloat(0.08),
        load_factor=FiniteFloat(0.10),
        clip_lower_mw=FiniteFloat(0.05),
        clip_upper_mw=FiniteFloat(0.20),
    )
    if physical_limits is None:
        physical_limits = DiagnosticPhysicalLimits(
            branch_budget_mw=FloatVector([100.0] * 140),
            voltage_lower_limit_pu=FloatVector([0.0] * 141),
            voltage_upper_limit_pu=FloatVector([2.0] * 141),
            finite_difference_delta_mw=FiniteFloat(0.01),
            feasibility_tolerance=FiniteFloat(1e-9),
        )
    return IEEE141DiagnosticBatchFactory.build(
        case_path,
        rating_mw_by_bus={bus: 1.0 for bus in PARTICIPANT_BUSES},
        availability_mw_by_bus={bus: 1.0 for bus in PARTICIPANT_BUSES},
        reactive_specification=q_spec,
        physical_limits=physical_limits,
        request_config=request_config,
        finite_difference_workers=finite_difference_workers,
    )


def build_q0_m0_factory(case_path: Path) -> IEEE141DiagnosticBatchFactory:
    """Backward-compatible Q0/M0 factory entry point."""

    return build_ieee141_m0_factory(case_path, q_mode=QAssumption.Q0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--q-mode", choices=("Q0", "Q95"), default="Q0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Execution-only worker count; results retain deterministic job order.",
    )
    args = parser.parse_args()

    factory = build_ieee141_m0_factory(args.case, q_mode=QAssumption(args.q_mode))
    scenarios = build_unilateral_reporter_scan_scenarios(
        factory,
        report_delta_mw=FiniteFloat(0.05),
        capacity_limited_mw=FiniteFloat(0.05),
        delivery_cap_mw=FiniteFloat(1.0),
        stress_multiplier=FiniteFloat(5.0),
        alpha_mw=FiniteFloat(0.20),
        epsilon_mw=FiniteFloat(1e-6),
        tie_break_tolerance_mw=FiniteFloat(1e-9),
    )
    if args.start < 0 or args.start >= len(scenarios):
        raise SystemExit(f"--start must be in [0,{len(scenarios) - 1}]")
    stop = len(scenarios) if args.stop is None else args.stop
    if stop <= args.start or stop > len(scenarios):
        raise SystemExit(f"--stop must be in ({args.start},{len(scenarios)}]")
    selected_scenarios = scenarios[args.start:stop]
    result = factory.run_grid(
        selected_scenarios,
        grid_id=reporter_scan_grid_id(args.q_mode, args.start, stop),
        max_workers=args.workers,
    )
    write_diagnostic_grid_result(result, args.output)
    summary = {
        "serialization_id": "ieee141_reporter_scan_summary.v1",
        "grid_id": result.grid_id.to_json(),
        "grid_hash": result.grid_hash.to_json(),
        "factory_hash": factory.factory_hash.to_json(),
        "batch_count": result.batch_count,
        "execution_workers": args.workers,
        "job_ids": [job.job_id.to_json() for job in result.jobs],
        "reference_ac_pass_count": sum(
            all(screen.side_pass for _, screen in batch.reference_batch.ac_screens)
            for batch in result.batches
        ),
        "reported_ac_pass_count": sum(
            all(screen.side_pass for _, screen in batch.reported_batch.ac_screens)
            for batch in result.batches
        ),
        "diagnostic_only": True,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(canonical_dumps(summary) + "\n", encoding="utf-8")
    print(canonical_dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
