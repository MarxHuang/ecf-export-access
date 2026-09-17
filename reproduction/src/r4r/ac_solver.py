"""Deterministic polar Newton--Raphson AC power-flow slice.

This module is deliberately limited to a single MATPOWER-derived operating
point.  It does not allocate requests, optimize, calibrate, screen, or create
manuscript evidence.  The solver consumes only the statically parsed source
fields and returns a numerical solver record plus explicit operating-point
vectors for the later AC screening round.
"""
from __future__ import annotations

import cmath
import math
import platform
import sys
from dataclasses import dataclass

from r4r.errors import ValidationError
from r4r.models.optimization import SolverResult
from r4r.network_parser import ParsedMatpowerCase
from r4r.serialization import canonical_hash
from r4r.types import (
    FloatVector,
    Identifier,
    IdentifierVector,
    NormalizedSolverStatus,
    ObjectiveUnitPolicy,
    RawSolverStatus,
    Sha256,
)
from r4r.types.scalars import FiniteFloat


class ACPowerFlowError(ValidationError):
    """The AC input or Newton system is not numerically admissible."""


@dataclass(frozen=True, slots=True)
class ACPowerFlowSolution:
    """Typed numerical output; promotion to AC evidence is a later gate."""

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


def _linear_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    if n == 0:
        return []
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        scale = abs(a[pivot][col])
        if scale < 1e-14:
            raise ACPowerFlowError("AC_JACOBIAN_SINGULAR")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


def _canonical_solution_hash(
    result: SolverResult,
    bus_voltage_pu: list[float],
    bus_angle_deg: list[float],
    branch_p_from_mw: list[float],
    branch_q_from_mvar: list[float],
    branch_p_to_mw: list[float],
    branch_q_to_mvar: list[float],
    reference_bus_id: str,
    bus_p_mismatch_mw: list[float],
    bus_q_mismatch_mvar: list[float],
    slack_p_mw: float,
    slack_q_mvar: float,
    total_active_loss_mw: float,
    total_reactive_loss_mvar: float,
    global_p_balance_residual_mw: float,
    global_q_balance_residual_mvar: float,
) -> Sha256:
    """Hash the exact public solution payload, excluding only its hash field."""

    solver_json = result.to_json()
    solver_json["solution_hash"] = None
    # Keep the solver-result hash stable when v1 adds nullable hierarchy
    # evidence fields.  Those fields are separately attested when populated;
    # an absent field must not rewrite historical numerical solution hashes.
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
        "bus_voltage_pu": list(bus_voltage_pu),
        "bus_angle_deg": list(bus_angle_deg),
        "branch_p_from_mw": list(branch_p_from_mw),
        "branch_q_from_mvar": list(branch_q_from_mvar),
        "branch_p_to_mw": list(branch_p_to_mw),
        "branch_q_to_mvar": list(branch_q_to_mvar),
        "reference_bus_id": reference_bus_id,
        "network_input_hash": result.network_input_hash.to_json() if result.network_input_hash is not None else None,
        "p_injection_hash": result.p_injection_hash.to_json() if result.p_injection_hash is not None else None,
        "q_injection_hash": result.q_injection_hash.to_json() if result.q_injection_hash is not None else None,
        "solver_config_hash": result.solver_config_hash.to_json() if result.solver_config_hash is not None else None,
        "solver_environment_hash": result.solver_environment_hash.to_json() if result.solver_environment_hash is not None else None,
        "bus_p_mismatch_mw": list(bus_p_mismatch_mw),
        "bus_q_mismatch_mvar": list(bus_q_mismatch_mvar),
        "slack_p_mw": slack_p_mw,
        "slack_q_mvar": slack_q_mvar,
        "total_active_loss_mw": total_active_loss_mw,
        "total_reactive_loss_mvar": total_reactive_loss_mvar,
        "global_p_balance_residual_mw": global_p_balance_residual_mw,
        "global_q_balance_residual_mvar": global_q_balance_residual_mvar,
    }))


def _failure_solution(
    parsed: ParsedMatpowerCase,
    solver_result: SolverResult,
    vm: list[float],
    va: list[float],
    *,
    hashes: tuple[Sha256, Sha256, Sha256, Sha256, Sha256],
    ref_index: int,
) -> ACPowerFlowSolution:
    network_hash, p_hash, q_hash, config_hash, environment_hash = hashes
    zeros = FloatVector([0.0] * len(parsed.network.branches))
    bus_zeros = FloatVector([0.0] * len(parsed.network.buses))
    solution_hash = _canonical_solution_hash(
        solver_result,
        vm,
        [math.degrees(value) for value in va],
        zeros.values,
        zeros.values,
        zeros.values,
        zeros.values,
        str(parsed.network.buses[ref_index].bus_id),
        bus_zeros.values,
        bus_zeros.values,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    solver_result = SolverResult(**{
        **{field: getattr(solver_result, field) for field in solver_result.__dataclass_fields__},
        "network_input_hash": network_hash,
        "p_injection_hash": p_hash,
        "q_injection_hash": q_hash,
        "solver_config_hash": config_hash,
        "solver_environment_hash": environment_hash,
        "solution_hash": solution_hash,
    })
    return ACPowerFlowSolution(
        solver_result,
        FloatVector(vm),
        FloatVector(math.degrees(value) for value in va),
        zeros,
        zeros,
        zeros,
        zeros,
        Identifier(str(parsed.network.buses[ref_index].bus_id)),
        network_hash, p_hash, q_hash, config_hash, environment_hash, solution_hash,
        bus_zeros, bus_zeros, FiniteFloat(0.0), FiniteFloat(0.0),
        FiniteFloat(0.0), FiniteFloat(0.0), FiniteFloat(0.0), FiniteFloat(0.0),
    )


def solve_ac_power_flow(
    parsed: ParsedMatpowerCase,
    *,
    p_injection_mw: FloatVector | None = None,
    q_injection_mvar: FloatVector | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    solver_id: Identifier = Identifier("ac_newton_raphson"),
    solver_version: str = "r3.0.0",
) -> ACPowerFlowSolution:
    """Solve one parsed case using the registered polar AC equations.

    ``p_injection_mw`` and ``q_injection_mvar`` are net injections (generation
    minus demand).  Omitting them uses the source generator minus converted
    load vectors.  No clipping, load scaling, or feasibility substitution is
    performed here.
    """
    n = len(parsed.network.buses)
    m = len(parsed.network.branches)
    if len(parsed.p_load_mw.values) != n or len(parsed.q_load_mvar.values) != n:
        raise ACPowerFlowError("AC_INVALID_INPUT_SHAPE")
    if len(parsed.p_generation_mw.values) != n or len(parsed.q_generation_mvar.values) != n:
        raise ACPowerFlowError("AC_INVALID_INPUT_SHAPE")
    if len(parsed.branch_b_pu.values) != m:
        raise ACPowerFlowError("AC_INVALID_BRANCH_SHAPE")
    if not math.isfinite(tolerance) or tolerance <= 0 or isinstance(max_iterations, bool) or max_iterations <= 0:
        raise ACPowerFlowError("AC_INVALID_NUMERICAL_PARAMETER")
    p = list((p_injection_mw or FloatVector(
        a - b for a, b in zip(parsed.p_generation_mw.values, parsed.p_load_mw.values)
    )).values)
    q = list((q_injection_mvar or FloatVector(
        a - b for a, b in zip(parsed.q_generation_mvar.values, parsed.q_load_mvar.values)
    )).values)
    if len(p) != n or len(q) != n:
        raise ACPowerFlowError("AC_INVALID_INPUT_SHAPE")

    network_hash = Sha256(canonical_hash({
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
    p_hash = Sha256(canonical_hash({"unit": "MW", "values": p}))
    q_hash = Sha256(canonical_hash({"unit": "MVAr", "values": q}))
    config_hash = Sha256(canonical_hash({
        "solver_id": solver_id.to_json(), "solver_version": solver_version,
        "tolerance": tolerance, "max_iterations": max_iterations,
        "equation_set": "polar_ac_newton_raphson_v1",
        "angle_variable_order": "all_non_reference_bus_order",
        "voltage_variable_order": "pq_bus_order",
    }))
    environment_hash = Sha256(canonical_hash({
        "implementation": "r4r.ac_solver",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))

    ref = [i for i, bus_type in enumerate(parsed.bus_types) if bus_type == 3]
    if len(ref) != 1:
        raise ACPowerFlowError("AC_REFERENCE_BUS_INVALID")
    if any(bus_type == 4 for bus_type in parsed.bus_types):
        raise ACPowerFlowError("AC_UNSUPPORTED_BUS_TYPE")
    ref_i = ref[0]
    pv = [i for i, bus_type in enumerate(parsed.bus_types) if bus_type == 2]
    pq = [i for i, bus_type in enumerate(parsed.bus_types) if bus_type == 1]
    angle_vars = [i for i in range(n) if i != ref_i]
    voltage_vars = pq
    vm = list(parsed.bus_vm_pu.values)
    va = [math.radians(value) for value in parsed.bus_va_deg.values]
    if any(value <= 0 for value in vm):
        raise ACPowerFlowError("AC_INVALID_INITIAL_VOLTAGE")

    ybus = [[0j for _ in range(n)] for _ in range(n)]
    branch_admittance: list[complex] = []
    bus_index = {bus.bus_id: i for i, bus in enumerate(parsed.network.buses)}
    for branch, b_shunt in zip(parsed.network.branches, parsed.branch_b_pu.values):
        i = bus_index[int(branch.from_bus.object_id.value)]
        j = bus_index[int(branch.to_bus.object_id.value)]
        z = complex(branch.r.value, branch.x.value)
        if abs(z) == 0:
            raise ACPowerFlowError("AC_ZERO_BRANCH_IMPEDANCE")
        y = 1.0 / z
        branch_admittance.append(y)
        shunt = 1j * b_shunt / 2.0
        ybus[i][i] += y + shunt
        ybus[j][j] += y + shunt
        ybus[i][j] -= y
        ybus[j][i] -= y
    g = [[value.real for value in row] for row in ybus]
    b = [[value.imag for value in row] for row in ybus]
    base = parsed.network.base_mva.value

    def powers() -> tuple[list[float], list[float]]:
        p_calc = [0.0] * n
        q_calc = [0.0] * n
        for i in range(n):
            for k in range(n):
                delta = va[i] - va[k]
                p_calc[i] += vm[i] * vm[k] * (g[i][k] * math.cos(delta) + b[i][k] * math.sin(delta))
                q_calc[i] += vm[i] * vm[k] * (g[i][k] * math.sin(delta) - b[i][k] * math.cos(delta))
        return p_calc, q_calc

    converged = False
    residual_norm = math.inf
    iteration = 0
    failure = ""
    last_mismatch: list[float] = []
    try:
        for iteration in range(1, max_iterations + 1):
            p_calc, q_calc = powers()
            mismatch = [(p[i] / base) - p_calc[i] for i in angle_vars]
            mismatch.extend((q[i] / base) - q_calc[i] for i in voltage_vars)
            last_mismatch = mismatch[:]
            residual_norm = max((abs(value) for value in mismatch), default=0.0)
            if residual_norm < tolerance:
                converged = True
                break
            row_count = len(mismatch)
            jac = [[0.0] * row_count for _ in range(row_count)]
            for row, i in enumerate(angle_vars):
                for col, k in enumerate(angle_vars):
                    delta = va[i] - va[k]
                    if i == k:
                        jac[row][col] = -q_calc[i] - b[i][i] * vm[i] ** 2
                    else:
                        jac[row][col] = vm[i] * vm[k] * (g[i][k] * math.sin(delta) - b[i][k] * math.cos(delta))
                for col, k in enumerate(voltage_vars, len(angle_vars)):
                    delta = va[i] - va[k]
                    if i == k:
                        jac[row][col] = p_calc[i] / vm[i] + g[i][i] * vm[i]
                    else:
                        jac[row][col] = vm[i] * (g[i][k] * math.cos(delta) + b[i][k] * math.sin(delta))
            for offset, i in enumerate(voltage_vars, len(angle_vars)):
                row = offset
                for col, k in enumerate(angle_vars):
                    delta = va[i] - va[k]
                    if i == k:
                        jac[row][col] = p_calc[i] - g[i][i] * vm[i] ** 2
                    else:
                        jac[row][col] = -vm[i] * vm[k] * (g[i][k] * math.cos(delta) + b[i][k] * math.sin(delta))
                for col, k in enumerate(voltage_vars, len(angle_vars)):
                    delta = va[i] - va[k]
                    if i == k:
                        jac[row][col] = q_calc[i] / vm[i] - b[i][i] * vm[i]
                    else:
                        jac[row][col] = vm[i] * (g[i][k] * math.sin(delta) - b[i][k] * math.cos(delta))
            step = _linear_solve(jac, mismatch)
            for value, i in zip(step[: len(angle_vars)], angle_vars):
                va[i] += value
            for value, i in zip(step[len(angle_vars) :], voltage_vars):
                vm[i] += value
                if vm[i] <= 0 or not math.isfinite(vm[i]):
                    raise ACPowerFlowError("AC_INVALID_VOLTAGE_ITERATE")
        if not converged:
            failure = "AC_NONCONVERGENCE"
    except ACPowerFlowError as exc:
        failure = str(exc)

    reasons = IdentifierVector([] if converged else [Identifier(failure or "AC_ERROR")])
    if not last_mismatch:
        p_calc, q_calc = powers()
        last_mismatch = [(p[i] / base) - p_calc[i] for i in angle_vars]
        last_mismatch.extend((q[i] / base) - q_calc[i] for i in voltage_vars)
    scaled_residual = [value / tolerance for value in last_mismatch]
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
        residual_norm=FiniteFloat(residual_norm if math.isfinite(residual_norm) else 0.0),
        tolerance_reference=Identifier("ac_convergence_tolerance"),
        raw_primal_residual=FloatVector(last_mismatch),
        scaled_primal_residual=FloatVector(scaled_residual),
        network_input_hash=network_hash,
        p_injection_hash=p_hash,
        q_injection_hash=q_hash,
        solver_config_hash=config_hash,
        solver_environment_hash=environment_hash,
    )
    if not converged:
        return _failure_solution(parsed, solver_result, vm, va, hashes=(network_hash, p_hash, q_hash, config_hash, environment_hash), ref_index=ref_i)

    p_from: list[float] = []
    q_from: list[float] = []
    p_to: list[float] = []
    q_to: list[float] = []
    for branch, y, b_shunt in zip(parsed.network.branches, branch_admittance, parsed.branch_b_pu.values):
        i = bus_index[int(branch.from_bus.object_id.value)]
        j = bus_index[int(branch.to_bus.object_id.value)]
        vi = cmath.rect(vm[i], va[i])
        vj = cmath.rect(vm[j], va[j])
        shunt = 1j * b_shunt / 2.0
        sf = vi * (vi - vj).conjugate() * y.conjugate() + vi * vi.conjugate() * shunt.conjugate()
        st = vj * (vj - vi).conjugate() * y.conjugate() + vj * vj.conjugate() * shunt.conjugate()
        p_from.append(sf.real * base)
        q_from.append(sf.imag * base)
        p_to.append(st.real * base)
        q_to.append(st.imag * base)
    p_calc, q_calc = powers()
    bus_p_mismatch = [0.0 if i == ref_i else (p[i] / base - p_calc[i]) * base for i in range(n)]
    bus_q_mismatch = [0.0 if i not in pq else (q[i] / base - q_calc[i]) * base for i in range(n)]
    slack_p = p_calc[ref_i] * base
    slack_q = q_calc[ref_i] * base
    total_active_loss = sum(p_from) + sum(p_to)
    total_reactive_loss = sum(q_from) + sum(q_to)
    global_p_balance = slack_p + sum(p[index] for index in range(n) if index != ref_i) - total_active_loss
    global_q_balance = slack_q + sum(q[index] for index in range(n) if index != ref_i) - total_reactive_loss
    solution_hash = _canonical_solution_hash(
        solver_result, vm, [math.degrees(value) for value in va],
        p_from, q_from, p_to, q_to, str(parsed.network.buses[ref_i].bus_id),
        bus_p_mismatch, bus_q_mismatch, slack_p, slack_q,
        total_active_loss, total_reactive_loss, global_p_balance, global_q_balance,
    )
    solver_result = SolverResult(**{
        **{field: getattr(solver_result, field) for field in solver_result.__dataclass_fields__},
        "solution_hash": solution_hash,
    })
    return ACPowerFlowSolution(
        solver_result,
        FloatVector(vm),
        FloatVector(math.degrees(value) for value in va),
        FloatVector(p_from),
        FloatVector(q_from),
        FloatVector(p_to),
        FloatVector(q_to),
        Identifier(str(parsed.network.buses[ref_i].bus_id)),
        network_hash, p_hash, q_hash, config_hash, environment_hash, solution_hash,
        FloatVector(bus_p_mismatch), FloatVector(bus_q_mismatch),
        FiniteFloat(slack_p), FiniteFloat(slack_q),
        FiniteFloat(total_active_loss), FiniteFloat(total_reactive_loss),
        FiniteFloat(global_p_balance), FiniteFloat(global_q_balance),
    )
