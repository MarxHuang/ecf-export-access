"""Independent radial backward/forward-sweep AC power-flow solver.

This module is the Round 4 implementation slice.  It deliberately does not
call :mod:`r4r.ac_solver` or reuse its Newton/Jacobian implementation.  The
two implementations share only the registered ``ACPowerFlowSolution`` output
schema so that their numerical records can be compared without changing the
scientific contract.
"""
from __future__ import annotations

import cmath
import hashlib
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from r4r.errors import ValidationError
from r4r.models.ac import ACSolverCrosscheckResult
from r4r.models.optimization import SolverResult
from r4r.network_parser import ParsedMatpowerCase
from r4r.serialization import canonical_hash
from r4r.types import (
    FloatVector,
    GateStatus,
    Identifier,
    IdentifierVector,
    NormalizedSolverStatus,
    ObjectiveUnitPolicy,
    RawSolverStatus,
    Sha256,
    ScreenSide,
)
from r4r.types.scalars import FiniteFloat


class IndependentACPowerFlowError(ValidationError):
    """The independent radial input or sweep is not admissible."""


@dataclass(frozen=True, slots=True)
class RadialEdge:
    """Rooted view of one parsed branch; parser order is retained separately."""

    branch_index: int
    parent_index: int
    child_index: int
    impedance_pu: complex
    shunt_admittance_pu: complex


@dataclass(frozen=True, slots=True)
class IndependentACPowerFlowSolution:
    """Common AC output shape produced by the independent implementation."""

    solver_result: SolverResult
    bus_voltage_pu: FloatVector
    bus_angle_deg: FloatVector
    branch_p_from_mw: FloatVector
    branch_q_from_mvar: FloatVector
    branch_p_to_mw: FloatVector
    branch_q_to_mvar: FloatVector
    reference_bus_id: Identifier
    network_input_hash: Sha256
    p_injection_hash: Sha256
    q_injection_hash: Sha256
    solver_config_hash: Sha256
    solver_environment_hash: Sha256
    solution_hash: Sha256
    bus_p_mismatch_mw: FloatVector
    bus_q_mismatch_mvar: FloatVector
    slack_p_mw: FiniteFloat
    slack_q_mvar: FiniteFloat
    total_active_loss_mw: FiniteFloat
    total_reactive_loss_mvar: FiniteFloat
    global_p_balance_residual_mw: FiniteFloat
    global_q_balance_residual_mvar: FiniteFloat

    def to_json(self) -> dict[str, object]:
        return {
            "solver_result": self.solver_result.to_json(),
            "bus_voltage_pu": self.bus_voltage_pu.to_json(),
            "bus_angle_deg": self.bus_angle_deg.to_json(),
            "branch_p_from_mw": self.branch_p_from_mw.to_json(),
            "branch_q_from_mvar": self.branch_q_from_mvar.to_json(),
            "branch_p_to_mw": self.branch_p_to_mw.to_json(),
            "branch_q_to_mvar": self.branch_q_to_mvar.to_json(),
            "reference_bus_id": self.reference_bus_id.to_json(),
            "network_input_hash": self.network_input_hash.to_json(),
            "p_injection_hash": self.p_injection_hash.to_json(),
            "q_injection_hash": self.q_injection_hash.to_json(),
            "solver_config_hash": self.solver_config_hash.to_json(),
            "solver_environment_hash": self.solver_environment_hash.to_json(),
            "solution_hash": self.solution_hash.to_json(),
            "bus_p_mismatch_mw": self.bus_p_mismatch_mw.to_json(),
            "bus_q_mismatch_mvar": self.bus_q_mismatch_mvar.to_json(),
            "slack_p_mw": self.slack_p_mw.to_json(),
            "slack_q_mvar": self.slack_q_mvar.to_json(),
            "total_active_loss_mw": self.total_active_loss_mw.to_json(),
            "total_reactive_loss_mvar": self.total_reactive_loss_mvar.to_json(),
            "global_p_balance_residual_mw": self.global_p_balance_residual_mw.to_json(),
            "global_q_balance_residual_mvar": self.global_q_balance_residual_mvar.to_json(),
        }


def _network_hash(parsed: ParsedMatpowerCase) -> Sha256:
    """Reproduce the registered network-input hash without importing Newton."""

    return Sha256(canonical_hash({
        "source_hash": parsed.source_hash.to_json(),
        "base_mva": parsed.network.base_mva.to_json(),
        "bus_ids": [bus.bus_id for bus in parsed.network.buses],
        "bus_types": parsed.bus_types,
        "branch_status": parsed.branch_status,
        "branches": [
            {"id": branch.branch_id, "from": branch.from_bus.to_json(), "to": branch.to_bus.to_json(),
             "r_pu": branch.r.to_json(), "x_pu": branch.x.to_json()}
            for branch in parsed.network.branches
        ],
        "load_p_mw": parsed.p_load_mw.to_json(),
        "load_q_mvar": parsed.q_load_mvar.to_json(),
        "generation_p_mw": parsed.p_generation_mw.to_json(),
        "generation_q_mvar": parsed.q_generation_mvar.to_json(),
        "branch_b_pu": parsed.branch_b_pu.to_json(),
        "initial_vm_pu": parsed.bus_vm_pu.to_json(),
        "initial_va_deg": parsed.bus_va_deg.to_json(),
    }))


def _root_tree(parsed: ParsedMatpowerCase, reference_index: int) -> tuple[list[RadialEdge], list[int], dict[int, list[int]]]:
    """Root the parser's undirected tree at the unique reference bus."""

    bus_index = {bus.bus_id: index for index, bus in enumerate(parsed.network.buses)}
    adjacency: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(bus_index))}
    for branch_index, branch in enumerate(parsed.network.branches):
        left = bus_index[int(branch.from_bus.object_id.value)]
        right = bus_index[int(branch.to_bus.object_id.value)]
        adjacency[left].append((right, branch_index))
        adjacency[right].append((left, branch_index))

    parent: dict[int, int] = {reference_index: -1}
    edge_for_child: dict[int, int] = {}
    order = [reference_index]
    for current in order:
        for neighbour, branch_index in sorted(adjacency[current], key=lambda item: item[0]):
            if neighbour in parent:
                continue
            parent[neighbour] = current
            edge_for_child[neighbour] = branch_index
            order.append(neighbour)
    if len(parent) != len(bus_index) or len(edge_for_child) != len(bus_index) - 1:
        raise IndependentACPowerFlowError("AC_RADIAL_TOPOLOGY_INVALID")

    children: dict[int, list[int]] = {index: [] for index in range(len(bus_index))}
    edges: list[RadialEdge] = []
    for child in order[1:]:
        parent_index = parent[child]
        branch_index = edge_for_child[child]
        branch = parsed.network.branches[branch_index]
        z = complex(branch.r.value, branch.x.value)
        if abs(z) == 0.0:
            raise IndependentACPowerFlowError("AC_ZERO_BRANCH_IMPEDANCE")
        edge = RadialEdge(
            branch_index=branch_index,
            parent_index=parent_index,
            child_index=child,
            impedance_pu=z,
            shunt_admittance_pu=1j * parsed.branch_b_pu.values[branch_index] / 2.0,
        )
        edges.append(edge)
        children[parent_index].append(branch_index)
    return edges, order, children


def _branch_outputs(
    parsed: ParsedMatpowerCase,
    voltage: list[complex],
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Compute stored F_BUS->T_BUS flows and bus injections from pi branches."""

    bus_index = {bus.bus_id: index for index, bus in enumerate(parsed.network.buses)}
    p_from: list[float] = []
    q_from: list[float] = []
    p_to: list[float] = []
    q_to: list[float] = []
    p_calc = [0.0] * len(voltage)
    q_calc = [0.0] * len(voltage)
    base = parsed.network.base_mva.value
    for branch_index, branch in enumerate(parsed.network.branches):
        left = bus_index[int(branch.from_bus.object_id.value)]
        right = bus_index[int(branch.to_bus.object_id.value)]
        z = complex(branch.r.value, branch.x.value)
        y = 1.0 / z
        y_shunt = 1j * parsed.branch_b_pu.values[branch_index] / 2.0
        i_from = (voltage[left] - voltage[right]) * y + y_shunt * voltage[left]
        i_to = (voltage[right] - voltage[left]) * y + y_shunt * voltage[right]
        s_from = voltage[left] * i_from.conjugate() * base
        s_to = voltage[right] * i_to.conjugate() * base
        p_from.append(s_from.real)
        q_from.append(s_from.imag)
        p_to.append(s_to.real)
        q_to.append(s_to.imag)
        p_calc[left] += s_from.real / base
        q_calc[left] += s_from.imag / base
        p_calc[right] += s_to.real / base
        q_calc[right] += s_to.imag / base
    return p_from, q_from, p_to, q_to, p_calc, q_calc


def _solution_hash(result: SolverResult, voltage: list[complex], flows: tuple[list[float], ...], reference_bus_id: int, evidence: dict[str, object]) -> Sha256:
    p_from, q_from, p_to, q_to = flows
    solver_json = result.to_json()
    solver_json["solution_hash"] = None
    # Nullable hierarchy metadata was added to solver_result.v1 after the
    # existing AC artifacts were generated.  Omit absent metadata from the
    # numerical solution hash; populated hierarchy evidence remains hash-bound.
    for field in (
        "stage1_primal_lower_mw", "stage1_dual_upper_mw", "tau_h_mw",
        "stage2_primary_floor_mw", "stage2_normalized_decision_enclosure",
        "stage2_canonical_objective_mw", "stage2_objective_gradient_mw_inverse",
        "stage2_strong_convexity_modulus_mw_inverse", "stage2_weight_policy",
        "stage2_eta_dimensionless", "stage2_s_t_mw",
        "hierarchy_attestation", "fairness_reference_attestation",
    ):
        if solver_json.get(field) is None:
            solver_json.pop(field, None)
    return Sha256(canonical_hash({
        "solver_result": solver_json,
        "bus_voltage_pu": [abs(value) for value in voltage],
        "bus_angle_deg": [math.degrees(cmath.phase(value)) for value in voltage],
        "branch_p_from_mw": p_from,
        "branch_q_from_mvar": q_from,
        "branch_p_to_mw": p_to,
        "branch_q_to_mvar": q_to,
        "reference_bus_id": str(reference_bus_id),
        "network_input_hash": result.network_input_hash.to_json() if result.network_input_hash is not None else None,
        "p_injection_hash": result.p_injection_hash.to_json() if result.p_injection_hash is not None else None,
        "q_injection_hash": result.q_injection_hash.to_json() if result.q_injection_hash is not None else None,
        "solver_config_hash": result.solver_config_hash.to_json() if result.solver_config_hash is not None else None,
        "solver_environment_hash": result.solver_environment_hash.to_json() if result.solver_environment_hash is not None else None,
        **evidence,
    }))


def solve_independent_radial_ac(
    parsed: ParsedMatpowerCase,
    *,
    p_injection_mw: FloatVector | None = None,
    q_injection_mvar: FloatVector | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    solver_id: Identifier = Identifier("ac_bfs_radial"),
    solver_version: str = "r4.0.0",
) -> IndependentACPowerFlowSolution:
    """Solve one radial PQ case by independent backward/forward sweeps.

    Net injections are generation minus demand.  The backward sweep carries
    constant-power load currents and receiving-end pi shunts toward the root;
    the forward sweep solves each series drop with the receiving shunt retained
    analytically.  No Newton Jacobian, allocation, clipping, or proxy result is
    used.
    """

    n = len(parsed.network.buses)
    m = len(parsed.network.branches)
    if len(parsed.p_load_mw.values) != n or len(parsed.q_load_mvar.values) != n:
        raise IndependentACPowerFlowError("AC_INVALID_INPUT_SHAPE")
    if len(parsed.branch_b_pu.values) != m:
        raise IndependentACPowerFlowError("AC_INVALID_BRANCH_SHAPE")
    if not math.isfinite(tolerance) or tolerance <= 0 or isinstance(max_iterations, bool) or max_iterations <= 0:
        raise IndependentACPowerFlowError("AC_INVALID_NUMERICAL_PARAMETER")
    p = list((p_injection_mw or FloatVector(
        a - b for a, b in zip(parsed.p_generation_mw.values, parsed.p_load_mw.values)
    )).values)
    q = list((q_injection_mvar or FloatVector(
        a - b for a, b in zip(parsed.q_generation_mvar.values, parsed.q_load_mvar.values)
    )).values)
    if len(p) != n or len(q) != n:
        raise IndependentACPowerFlowError("AC_INVALID_INPUT_SHAPE")

    reference = [index for index, bus_type in enumerate(parsed.bus_types) if bus_type == 3]
    if len(reference) != 1:
        raise IndependentACPowerFlowError("AC_REFERENCE_BUS_INVALID")
    if any(bus_type != 1 and bus_type != 3 for bus_type in parsed.bus_types):
        raise IndependentACPowerFlowError("AC_BFS_REQUIRES_PQ_OR_REFERENCE")
    ref_i = reference[0]
    edges, order, children = _root_tree(parsed, ref_i)
    edge_by_branch = {edge.branch_index: edge for edge in edges}
    if len(edge_by_branch) != m:
        raise IndependentACPowerFlowError("AC_RADIAL_TOPOLOGY_INVALID")

    network_hash = _network_hash(parsed)
    p_hash = Sha256(canonical_hash({"unit": "MW", "values": p}))
    q_hash = Sha256(canonical_hash({"unit": "MVAr", "values": q}))
    config_hash = Sha256(canonical_hash({
        "solver_id": solver_id.to_json(), "solver_version": solver_version,
        "tolerance": tolerance, "max_iterations": max_iterations,
        "equation_set": "radial_backward_forward_sweep_v1",
        "topology_root": parsed.network.buses[ref_i].bus_id,
        "branch_current_convention": "stored_pi_branch_F_BUS_to_T_BUS",
    }))
    environment_hash = Sha256(canonical_hash({
        "implementation": "r4r.independent_ac_solver",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))

    voltage = [cmath.rect(vm, math.radians(angle)) for vm, angle in zip(parsed.bus_vm_pu.values, parsed.bus_va_deg.values)]
    if any(abs(value) == 0.0 or not math.isfinite(value.real) or not math.isfinite(value.imag) for value in voltage):
        raise IndependentACPowerFlowError("AC_INVALID_INITIAL_VOLTAGE")
    base = parsed.network.base_mva.value
    converged = False
    iteration = 0
    failure = ""
    last_mismatch: list[float] = []
    try:
        for iteration in range(1, max_iterations + 1):
            load_current = []
            for p_value, q_value, bus_voltage in zip(p, q, voltage):
                if abs(bus_voltage) < 1e-12:
                    raise IndependentACPowerFlowError("AC_INVALID_VOLTAGE_ITERATE")
                net_injection_pu = complex(p_value, q_value) / base
                load_current.append(-net_injection_pu.conjugate() / bus_voltage.conjugate())

            receiving_current: dict[int, complex] = {}
            sending_current: dict[int, complex] = {}
            for bus in reversed(order[1:]):
                branch_indexes = children[bus]
                downstream = sum((sending_current[index] for index in branch_indexes), 0j)
                edge = next(edge for edge in edges if edge.child_index == bus)
                receiving = load_current[bus] + downstream
                receiving_current[edge.branch_index] = receiving
                sending_current[edge.branch_index] = (
                    receiving + edge.shunt_admittance_pu * voltage[bus]
                    + edge.shunt_admittance_pu * voltage[edge.parent_index]
                )

            updated = voltage[:]
            updated[ref_i] = voltage[ref_i]
            for parent in order:
                for branch_index in children[parent]:
                    edge = edge_by_branch[branch_index]
                    denominator = 1.0 + edge.impedance_pu * edge.shunt_admittance_pu
                    if abs(denominator) < 1e-14:
                        raise IndependentACPowerFlowError("AC_BFS_SINGULAR_DROP")
                    updated[edge.child_index] = (
                        updated[edge.parent_index] - edge.impedance_pu * receiving_current[branch_index]
                    ) / denominator
                    if abs(updated[edge.child_index]) < 1e-12 or not math.isfinite(updated[edge.child_index].real) or not math.isfinite(updated[edge.child_index].imag):
                        raise IndependentACPowerFlowError("AC_INVALID_VOLTAGE_ITERATE")
            voltage = updated
            _, _, _, _, p_calc, q_calc = _branch_outputs(parsed, voltage)
            angle_vars = [index for index in range(n) if index != ref_i]
            mismatch = [(p[index] / base) - p_calc[index] for index in angle_vars]
            mismatch.extend((q[index] / base) - q_calc[index] for index in range(n) if parsed.bus_types[index] == 1)
            last_mismatch = mismatch
            if max((abs(value) for value in mismatch), default=0.0) < tolerance:
                converged = True
                break
    except IndependentACPowerFlowError as exc:
        failure = str(exc)

    p_from, q_from, p_to, q_to, p_calc, q_calc = _branch_outputs(parsed, voltage)
    angle_vars = [index for index in range(n) if index != ref_i]
    last_mismatch = [(p[index] / base) - p_calc[index] for index in angle_vars]
    last_mismatch.extend((q[index] / base) - q_calc[index] for index in range(n) if parsed.bus_types[index] == 1)
    residual_norm = max((abs(value) for value in last_mismatch), default=0.0)
    if not converged and not failure:
        failure = "AC_BFS_NONCONVERGENCE"
    reasons = IdentifierVector([] if converged else [Identifier(failure)])
    solver_result = SolverResult(
        solver_id,
        solver_version,
        RawSolverStatus.FEASIBLE if converged else RawSolverStatus.ERROR,
        NormalizedSolverStatus.PASS if converged else NormalizedSolverStatus.FAIL,
        solver_domain="AC",
        evidence_scope="NUMERICAL_IMPLEMENTATION_ONLY",
        objective_unit_policy=ObjectiveUnitPolicy.UNDEFINED,
        primal_residual=FloatVector(last_mismatch),
        iteration_count=iteration,
        solve_attempted=True,
        solve_success=converged,
        evidence_valid=converged,
        failure_reasons=reasons,
        residual_norm=FiniteFloat(residual_norm),
        tolerance_reference=Identifier("ac_convergence_tolerance"),
        raw_primal_residual=FloatVector(last_mismatch),
        scaled_primal_residual=FloatVector(value / tolerance for value in last_mismatch),
        network_input_hash=network_hash,
        p_injection_hash=p_hash,
        q_injection_hash=q_hash,
        solver_config_hash=config_hash,
        solver_environment_hash=environment_hash,
    )
    bus_p_mismatch = [0.0 if index == ref_i else (p[index] / base - p_calc[index]) * base for index in range(n)]
    bus_q_mismatch = [0.0 if parsed.bus_types[index] != 1 else (q[index] / base - q_calc[index]) * base for index in range(n)]
    slack_p = p_calc[ref_i] * base
    slack_q = q_calc[ref_i] * base
    total_active_loss = sum(p_from) + sum(p_to)
    total_reactive_loss = sum(q_from) + sum(q_to)
    global_p_balance = slack_p + sum(p[index] for index in range(n) if index != ref_i) - total_active_loss
    global_q_balance = slack_q + sum(q[index] for index in range(n) if index != ref_i) - total_reactive_loss
    evidence = {
        "bus_p_mismatch_mw": bus_p_mismatch,
        "bus_q_mismatch_mvar": bus_q_mismatch,
        "slack_p_mw": slack_p,
        "slack_q_mvar": slack_q,
        "total_active_loss_mw": total_active_loss,
        "total_reactive_loss_mvar": total_reactive_loss,
        "global_p_balance_residual_mw": global_p_balance,
        "global_q_balance_residual_mvar": global_q_balance,
    }
    solution_hash = _solution_hash(solver_result, voltage, (p_from, q_from, p_to, q_to), parsed.network.buses[ref_i].bus_id, evidence)
    solver_result = SolverResult(**{
        **{field: getattr(solver_result, field) for field in solver_result.__dataclass_fields__},
        "solution_hash": solution_hash,
    })
    return IndependentACPowerFlowSolution(
        solver_result,
        FloatVector(abs(value) for value in voltage),
        FloatVector(math.degrees(cmath.phase(value)) for value in voltage),
        FloatVector(p_from), FloatVector(q_from), FloatVector(p_to), FloatVector(q_to),
        Identifier(str(parsed.network.buses[ref_i].bus_id)),
        network_hash, p_hash, q_hash, config_hash, environment_hash, solution_hash,
        FloatVector(bus_p_mismatch), FloatVector(bus_q_mismatch),
        FiniteFloat(slack_p), FiniteFloat(slack_q),
        FiniteFloat(total_active_loss), FiniteFloat(total_reactive_loss),
        FiniteFloat(global_p_balance), FiniteFloat(global_q_balance),
    )


def crosscheck_ac_solutions(
    primary: object,
    independent: IndependentACPowerFlowSolution,
    *,
    tolerance_profile: Mapping[str, float] | None = None,
    voltage_tolerance_pu: float = 1e-6,
    angle_tolerance_deg: float = 1e-4,
    branch_p_tolerance_mw: float = 1e-4,
    branch_q_tolerance_mvar: float = 1e-4,
    p_mismatch_tolerance_mw: float = 1e-5,
    q_mismatch_tolerance_mvar: float = 1e-5,
    slack_p_tolerance_mw: float = 1e-4,
    slack_q_tolerance_mvar: float = 1e-4,
    active_loss_tolerance_mw: float = 1e-4,
    reactive_loss_tolerance_mvar: float = 1e-4,
    network_hash: Sha256 | None = None,
    operating_point_hash: Sha256 | None = None,
    participant_registry_hash: Sha256 | None = None,
    q_spec_hash: Sha256 | None = None,
    side: ScreenSide | None = None,
    allocation_hash: Sha256 | None = None,
) -> tuple[ACSolverCrosscheckResult, dict[str, float]]:
    """Compare two solved records without promoting a physical AC pair."""

    profile = {
        "voltage_pu": voltage_tolerance_pu,
        "angle_deg": angle_tolerance_deg,
        "branch_p_mw": branch_p_tolerance_mw,
        "branch_q_mvar": branch_q_tolerance_mvar,
        "bus_p_mismatch_mw": p_mismatch_tolerance_mw,
        "bus_q_mismatch_mvar": q_mismatch_tolerance_mvar,
        "slack_p_mw": slack_p_tolerance_mw,
        "slack_q_mvar": slack_q_tolerance_mvar,
        "active_loss_mw": active_loss_tolerance_mw,
        "reactive_loss_mvar": reactive_loss_tolerance_mvar,
    }
    if tolerance_profile is not None:
        profile = {str(key): float(value) for key, value in tolerance_profile.items()}
        expected_keys = {
            "voltage_pu", "angle_deg", "branch_p_mw", "branch_q_mvar",
            "bus_p_mismatch_mw", "bus_q_mismatch_mvar", "slack_p_mw",
            "slack_q_mvar", "active_loss_mw", "reactive_loss_mvar",
        }
        if set(profile) != expected_keys or any(value < 0 for value in profile.values()):
            raise IndependentACPowerFlowError("AC_CROSSCHECK_TOLERANCE_PROFILE_INVALID")

    if not hasattr(primary, "solver_result"):
        raise IndependentACPowerFlowError("AC_CROSSCHECK_PRIMARY_RESULT_INVALID")
    primary_result = primary.solver_result
    independent_result = independent.solver_result
    shape_match = all(
        len(getattr(primary, field).values) == len(getattr(independent, field).values)
        for field in (
            "bus_voltage_pu", "bus_angle_deg", "branch_p_from_mw", "branch_q_from_mvar",
            "branch_p_to_mw", "branch_q_to_mvar", "bus_p_mismatch_mw", "bus_q_mismatch_mvar",
        )
    )
    input_hash_match = (
        primary.network_input_hash == independent.network_input_hash
        and primary.p_injection_hash == independent.p_injection_hash
        and primary.q_injection_hash == independent.q_injection_hash
        and primary.reference_bus_id == independent.reference_bus_id
    )
    branch_p_pairs = (
        (primary.branch_p_from_mw.values, independent.branch_p_from_mw.values),
        (primary.branch_p_to_mw.values, independent.branch_p_to_mw.values),
    )
    branch_q_pairs = (
        (primary.branch_q_from_mvar.values, independent.branch_q_from_mvar.values),
        (primary.branch_q_to_mvar.values, independent.branch_q_to_mvar.values),
    )
    metrics = {
        "max_voltage_difference_pu": max((abs(a - b) for a, b in zip(primary.bus_voltage_pu.values, independent.bus_voltage_pu.values)), default=0.0),
        "max_angle_difference_deg": max((abs(a - b) for a, b in zip(primary.bus_angle_deg.values, independent.bus_angle_deg.values)), default=0.0),
        "max_branch_p_difference_mw": max((abs(a - b) for left, right in branch_p_pairs for a, b in zip(left, right)), default=0.0),
        "max_branch_q_difference_mvar": max((abs(a - b) for left, right in branch_q_pairs for a, b in zip(left, right)), default=0.0),
        "max_bus_p_mismatch_difference_mw": max((abs(a - b) for a, b in zip(primary.bus_p_mismatch_mw.values, independent.bus_p_mismatch_mw.values)), default=0.0),
        "max_bus_q_mismatch_difference_mvar": max((abs(a - b) for a, b in zip(primary.bus_q_mismatch_mvar.values, independent.bus_q_mismatch_mvar.values)), default=0.0),
        "slack_p_difference_mw": abs(primary.slack_p_mw.value - independent.slack_p_mw.value),
        "slack_q_difference_mvar": abs(primary.slack_q_mvar.value - independent.slack_q_mvar.value),
        "active_loss_difference_mw": abs(primary.total_active_loss_mw.value - independent.total_active_loss_mw.value),
        "reactive_loss_difference_mvar": abs(primary.total_reactive_loss_mvar.value - independent.total_reactive_loss_mvar.value),
        "shape_match": 1.0 if shape_match else 0.0,
        "input_hash_match": 1.0 if input_hash_match else 0.0,
    }
    metrics["tolerance_profile_hash"] = canonical_hash(profile)
    passed = bool(primary_result.solve_success and independent_result.solve_success)
    passed = passed and shape_match and input_hash_match
    passed = passed and metrics["max_voltage_difference_pu"] <= profile["voltage_pu"]
    passed = passed and metrics["max_angle_difference_deg"] <= profile["angle_deg"]
    passed = passed and metrics["max_branch_p_difference_mw"] <= profile["branch_p_mw"]
    passed = passed and metrics["max_branch_q_difference_mvar"] <= profile["branch_q_mvar"]
    passed = passed and metrics["max_bus_p_mismatch_difference_mw"] <= profile["bus_p_mismatch_mw"]
    passed = passed and metrics["max_bus_q_mismatch_difference_mvar"] <= profile["bus_q_mismatch_mvar"]
    passed = passed and metrics["slack_p_difference_mw"] <= profile["slack_p_mw"]
    passed = passed and metrics["slack_q_difference_mvar"] <= profile["slack_q_mvar"]
    passed = passed and metrics["active_loss_difference_mw"] <= profile["active_loss_mw"]
    passed = passed and metrics["reactive_loss_difference_mvar"] <= profile["reactive_loss_mvar"]
    status = GateStatus.PASS if passed else GateStatus.FAIL
    for name, value in (
        ("network_hash", network_hash),
        ("operating_point_hash", operating_point_hash),
        ("participant_registry_hash", participant_registry_hash),
        ("q_spec_hash", q_spec_hash),
        ("allocation_hash", allocation_hash),
    ):
        if value is not None and not isinstance(value, Sha256):
            raise IndependentACPowerFlowError(f"AC_CROSSCHECK_{name.upper()}_INVALID")
    if side is not None and not isinstance(side, ScreenSide):
        raise IndependentACPowerFlowError("AC_CROSSCHECK_SIDE_INVALID")
    context_values = (network_hash, operating_point_hash, participant_registry_hash, q_spec_hash, side, allocation_hash)
    if any(value is not None for value in context_values) and not all(value is not None for value in context_values):
        raise IndependentACPowerFlowError("AC_CROSSCHECK_CONTEXT_INCOMPLETE")

    def input_identity(solution: object) -> Sha256:
        return Sha256(canonical_hash({
            "network_input_hash": getattr(solution, "network_input_hash", None).to_json(),
            "p_injection_hash": getattr(solution, "p_injection_hash", None).to_json(),
            "q_injection_hash": getattr(solution, "q_injection_hash", None).to_json(),
            "reference_bus_id": getattr(solution, "reference_bus_id", None).to_json(),
        }))

    primary_input_identity_hash = input_identity(primary)
    independent_input_identity_hash = input_identity(independent)
    if passed and primary_input_identity_hash != independent_input_identity_hash:
        # Keep the result fail-closed even if a future caller changes the
        # numerical comparison without updating the input-identity check.
        status = GateStatus.FAIL
    return ACSolverCrosscheckResult(
        primary_result.solution_hash,
        independent_result.solution_hash,
        status,
        network_hash=network_hash,
        operating_point_hash=operating_point_hash,
        participant_registry_hash=participant_registry_hash,
        q_spec_hash=q_spec_hash,
        side=side,
        allocation_hash=allocation_hash,
        primary_input_identity_hash=primary_input_identity_hash,
        independent_input_identity_hash=independent_input_identity_hash,
    ), metrics
