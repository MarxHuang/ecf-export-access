"""Unbalanced OpenDSS screen for the Australian external replay.

The CSIRO/GridQube master file ends with a diagnostic ``export Voltages``.
This adapter executes a sanitized in-memory command stream and therefore never
writes into or edits the public source tree.  It screens customer-connection
terminal voltage (phase-neutral for wye, phase-phase for delta) and transformer
nameplate loading. Phase-earth node magnitudes are not service voltages in a
four-wire circuit with a displaced neutral. Line thermal limits are not
screened because the public package does not supply conductor ampacity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import opendssdirect as dss

from .australian_data import DSSLoadConnection, connection_point_power


class AustralianPowerFlowError(RuntimeError):
    """Raised when a replay state violates the OpenDSS input contract."""


@dataclass(frozen=True, slots=True)
class AustralianDSSScreenResult:
    converged: bool
    iterations: int
    customer_voltage_min_pu: float
    customer_voltage_max_pu: float
    customer_undervoltage_count: int
    customer_overvoltage_count: int
    transformer_max_loading_pu: float
    transformer_overload_count: int
    most_loaded_transformer: str | None
    source_terminal_p_kw: float
    source_terminal_q_kvar: float
    circuit_loss_kw: float
    circuit_loss_kvar: float
    physical_pair_pass: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "customer_voltage_min_pu": self.customer_voltage_min_pu,
            "customer_voltage_max_pu": self.customer_voltage_max_pu,
            "customer_undervoltage_count": self.customer_undervoltage_count,
            "customer_overvoltage_count": self.customer_overvoltage_count,
            "transformer_max_loading_pu": self.transformer_max_loading_pu,
            "transformer_overload_count": self.transformer_overload_count,
            "most_loaded_transformer": self.most_loaded_transformer,
            "source_terminal_p_kw": self.source_terminal_p_kw,
            "source_terminal_q_kvar": self.source_terminal_q_kvar,
            "circuit_loss_kw": self.circuit_loss_kw,
            "circuit_loss_kvar": self.circuit_loss_kvar,
            "physical_pair_pass": self.physical_pair_pass,
        }


class AustralianOpenDSSModel:
    """One process-local OpenDSS model for repeated snapshot solves."""

    def __init__(
        self,
        data_root: str | Path,
        connections: Sequence[DSSLoadConnection],
        *,
        load_power_factor: float = 0.95,
        voltage_min_pu: float = 0.90,
        voltage_max_pu: float = 1.10,
        transformer_loading_limit_pu: float = 1.0,
        require_transformer_screen: bool = True,
        solution_tolerance: float = 1.0e-10,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.master_path = self.data_root / "Master.dss"
        if not self.master_path.is_file():
            raise AustralianPowerFlowError("OpenDSS data root must contain Master.dss")
        self.connections = tuple(connections)
        if not self.connections:
            raise AustralianPowerFlowError("at least one DSS load connection is required")
        self.load_power_factor = float(load_power_factor)
        self.voltage_min_pu = float(voltage_min_pu)
        self.voltage_max_pu = float(voltage_max_pu)
        self.transformer_loading_limit_pu = float(transformer_loading_limit_pu)
        self.require_transformer_screen = bool(require_transformer_screen)
        self.solution_tolerance = float(solution_tolerance)
        if not 0.0 < self.load_power_factor <= 1.0:
            raise AustralianPowerFlowError("load power factor must lie in (0, 1]")
        if not 0.0 < self.voltage_min_pu < self.voltage_max_pu:
            raise AustralianPowerFlowError("invalid voltage limits")
        if self.transformer_loading_limit_pu <= 0.0:
            raise AustralianPowerFlowError("transformer limit must be positive")
        if not math.isfinite(self.solution_tolerance) or not 0 < self.solution_tolerance < 1:
            raise AustralianPowerFlowError("solution tolerance must be finite and lie in (0,1)")
        self._compile_read_only()
        self._bind_model_entities()

    def _compile_read_only(self) -> None:
        dss.Basic.ClearAll()
        for line_number, line in enumerate(
            self.master_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            command = line.strip()
            if not command or command.startswith("!") or command.startswith("//"):
                continue
            lowered = command.lower()
            if lowered == "solve" or lowered.startswith("export "):
                continue
            if lowered.startswith("redirect "):
                relative = command.split(None, 1)[1].strip().strip('"')
                target = (self.data_root / relative).resolve()
                if not target.is_file():
                    raise AustralianPowerFlowError(
                        f"missing redirect target at Master.dss:{line_number}: {target}"
                    )
                command = f'redirect "{target}"'
            dss.Text.Command(command)
            result = str(dss.Text.Result()).strip()
            if result and "error" in result.lower():
                raise AustralianPowerFlowError(
                    f"OpenDSS command failed at Master.dss:{line_number}: {result}"
                )
        # Preserve the frequency declared by each public circuit.  The main
        # Australian MV/LV case is a 50-Hz model; representative public LV
        # models may retain their native frequency.  Forcing 50 Hz after the
        # line-code definitions are loaded can yield a numerically converged
        # but physically undefined (zero per-unit) solution.
        # The public Master.dss is responsible for declaring voltage bases and
        # invoking CalcVoltageBases.  Reapplying it after all redirects can
        # alter OpenDSS's native base assignment for representative LV models,
        # producing artificial all-zero per-unit magnitudes despite convergence.
        # GridQube's top-level Master does not enumerate all LV voltage bases;
        # its native solve consequently leaves AllBusMagPu undefined.  Its
        # associated LV masters, and the taxonomy models, do declare their own
        # bases.  Supply the GridQube screen bases only when the Master omits a
        # voltage-base declaration; do not override a public model that already
        # provides one.  The gridQube line definitions are explicitly 50 Hz, so
        # its source frequency is set only in this branch.
        if not any(
            line.strip().lower().startswith("set voltagebases=")
            for line in self.master_path.read_text(encoding="utf-8").splitlines()
        ):
            dss.Text.Command("edit vsource.source frequency=50")
            dss.Text.Command("set voltagebases=[11,0.433,0.415]")
            dss.Text.Command("calcvoltagebases")
        dss.Text.Command("set mode=snapshot controlmode=off maxiterations=100")
        dss.Solution.Convergence(self.solution_tolerance)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise AustralianPowerFlowError("public topology did not converge at compile smoke state")

    def _bind_model_entities(self) -> None:
        model_loads = {name.lower() for name in dss.Loads.AllNames()}
        requested_loads = {item.load_name.lower() for item in self.connections}
        if model_loads != requested_loads:
            missing = sorted(requested_loads - model_loads)[:5]
            extra = sorted(model_loads - requested_loads)[:5]
            raise AustralianPowerFlowError(
                f"DSS load mapping differs from model; missing={missing}, extra={extra}"
            )
        node_index = {
            str(name).lower(): index for index, name in enumerate(dss.Circuit.AllNodeNames())
        }
        # The final appended complex zero represents grounded node 0. Node 4
        # (or another explicit neutral) must retain its solved complex voltage.
        # EPRI: https://opendss.epri.com/NeutralRules.html
        ground_index = len(node_index)
        voltage_from, voltage_to, voltage_bases, starts = [], [], [], []
        for item in self.connections:
            dss.Loads.Name(item.load_name)
            phases = int(dss.Loads.Phases())
            delta = bool(dss.Loads.IsDelta())
            conductors = int(dss.CktElement.NumConductors())
            nodes = list(dss.CktElement.NodeOrder())[:conductors]
            if dss.CktElement.NumTerminals() != 1:
                raise AustralianPowerFlowError(f"load {item.load_name} has an unsupported terminal layout")
            bus = str(dss.CktElement.BusNames()[0]).split('.')[0]
            if bus.lower() != item.bus_name.lower():
                raise AustralianPowerFlowError(f"load {item.load_name} has a different native bus")
            dss.Circuit.SetActiveBus(bus)
            base_v = float(dss.Bus.kVBase()) * 1000
            if not math.isfinite(base_v) or base_v <= 0:
                raise AustralianPowerFlowError(f"load {item.load_name} has no positive bus voltage base")
            if not delta and conductors == phases + 1:
                pairs = [(j, phases) for j in range(phases)]
            elif delta and phases == 1 and conductors == 2:
                pairs = [(0, 1)]
                base_v *= math.sqrt(3)
            elif delta and phases in (2, 3) and conductors == phases:
                pairs = [(j, (j + 1) % conductors) for j in range(phases)]
                base_v *= math.sqrt(3)
            else:
                raise AustralianPowerFlowError(f"load {item.load_name} has an unsupported phase/conductor layout")
            starts.append(len(voltage_from))
            for left, right in pairs:
                pair_indices = []
                for node in (nodes[left], nodes[right]):
                    key = f'{bus}.{node}'.lower()
                    if node != 0 and key not in node_index:
                        raise AustralianPowerFlowError(f"mapped terminal node is absent: {key}")
                    pair_indices.append(ground_index if node == 0 else node_index[key])
                voltage_from.append(pair_indices[0])
                voltage_to.append(pair_indices[1])
                voltage_bases.append(base_v)
        self._voltage_from = np.asarray(voltage_from, dtype=int)
        self._voltage_to = np.asarray(voltage_to, dtype=int)
        self._voltage_bases_v = np.asarray(voltage_bases, dtype=float)
        self._connection_voltage_starts = np.asarray(starts, dtype=int)

        transformers: list[tuple[str, float]] = []
        if dss.Transformers.First():
            while True:
                name = str(dss.Transformers.Name()).lower()
                dss.Transformers.Wdg(1)
                rating_kva = float(dss.Transformers.kVA())
                if not math.isfinite(rating_kva) or rating_kva <= 0.0:
                    raise AustralianPowerFlowError(f"transformer {name} has no positive kVA rating")
                transformers.append((name, rating_kva))
                if not dss.Transformers.Next():
                    break
        if not transformers and self.require_transformer_screen:
            raise AustralianPowerFlowError("public topology contains no transformers")
        self._transformers = tuple(transformers)

    @property
    def load_names(self) -> tuple[str, ...]:
        return tuple(item.load_name for item in self.connections)

    @property
    def transformer_ratings_kva(self) -> Mapping[str, float]:
        return dict(self._transformers)

    def solve(
        self,
        household_load_kw: Sequence[float],
        gross_pv_kw: Sequence[float],
        allocated_export_kw: Sequence[float],
        participant_mask: Sequence[bool],
    ) -> tuple[AustralianDSSScreenResult, np.ndarray, np.ndarray]:
        p_load, q_load, delivered, available = connection_point_power(
            household_load_kw,
            gross_pv_kw,
            allocated_export_kw,
            participant_mask,
            load_power_factor=self.load_power_factor,
        )
        if p_load.size != len(self.connections):
            raise AustralianPowerFlowError("PCC vectors do not align with DSS load mapping")
        for item, p_kw, q_kvar in zip(self.connections, p_load, q_load, strict=True):
            dss.Loads.Name(item.load_name)
            dss.Loads.kW(float(p_kw))
            dss.Loads.kvar(float(q_kvar))
            dss.Loads.Model(1)
        dss.Solution.Solve()
        converged = bool(dss.Solution.Converged())
        iterations = int(dss.Solution.Iterations())

        raw_voltage = np.asarray(dss.Circuit.AllBusVolts(), dtype=float).reshape(-1, 2)
        node_voltage = np.append(raw_voltage[:, 0] + 1j * raw_voltage[:, 1], 0j)
        terminal_voltage = np.abs(node_voltage[self._voltage_from] - node_voltage[self._voltage_to]) / self._voltage_bases_v
        customer_voltage_min = np.minimum.reduceat(terminal_voltage, self._connection_voltage_starts)
        customer_voltage_max = np.maximum.reduceat(terminal_voltage, self._connection_voltage_starts)
        finite_voltage = np.isfinite(customer_voltage_min) & np.isfinite(customer_voltage_max)
        if np.all(finite_voltage):
            minimum = float(np.min(customer_voltage_min))
            maximum = float(np.max(customer_voltage_max))
        else:
            minimum = float("nan")
            maximum = float("nan")
        undervoltage_count = int(np.sum(~finite_voltage | (customer_voltage_min < self.voltage_min_pu)))
        overvoltage_count = int(np.sum(~finite_voltage | (customer_voltage_max > self.voltage_max_pu)))

        maximum_loading = float("nan")
        most_loaded: str | None = None
        overload_count = 0
        for name, rating_kva in self._transformers:
            dss.Circuit.SetActiveElement(f"transformer.{name}")
            terminal_powers = np.asarray(dss.CktElement.TotalPowers(), dtype=float)
            if terminal_powers.size < 4 or terminal_powers.size % 2:
                raise AustralianPowerFlowError(f"invalid transformer powers for {name}")
            apparent = np.hypot(terminal_powers[0::2], terminal_powers[1::2])
            loading = float(np.max(apparent) / rating_kva)
            if not math.isfinite(maximum_loading) or loading > maximum_loading:
                maximum_loading = loading
                most_loaded = name
            if loading > self.transformer_loading_limit_pu + 1.0e-9:
                overload_count += 1

        source = np.asarray(dss.Circuit.TotalPower(), dtype=float)
        losses = np.asarray(dss.Circuit.Losses(), dtype=float) / 1000.0
        physical_pass = (
            converged
            and undervoltage_count == 0
            and overvoltage_count == 0
            and overload_count == 0
        )
        result = AustralianDSSScreenResult(
            converged=converged,
            iterations=iterations,
            customer_voltage_min_pu=minimum,
            customer_voltage_max_pu=maximum,
            customer_undervoltage_count=undervoltage_count,
            customer_overvoltage_count=overvoltage_count,
            transformer_max_loading_pu=float(maximum_loading),
            transformer_overload_count=overload_count,
            most_loaded_transformer=most_loaded,
            source_terminal_p_kw=float(source[0]),
            source_terminal_q_kvar=float(source[1]),
            circuit_loss_kw=float(losses[0]),
            circuit_loss_kvar=float(losses[1]),
            physical_pair_pass=physical_pass,
        )
        return result, delivered, available

    def solve_authorized_export(
        self,
        household_load_kw: Sequence[float],
        authorized_export_kw: Sequence[float],
        participant_mask: Sequence[bool],
    ) -> AustralianDSSScreenResult:
        """Screen the full authorized-export endpoint, not realised PV.

        At a participant connection, synthetic gross PV is set to household
        load plus the authorized export.  The resulting PCC injection is
        exactly the authorization while household reactive demand is retained.
        This prevents realised under-delivery from making the allocation rule
        appear network-feasible ex post.
        """

        load = np.asarray(household_load_kw, dtype=float)
        authorization = np.asarray(authorized_export_kw, dtype=float)
        participant = np.asarray(participant_mask, dtype=bool)
        if not (load.shape == authorization.shape == participant.shape):
            raise AustralianPowerFlowError("authorized-export vectors must align")
        if np.any(authorization[~participant] > 0.0):
            raise AustralianPowerFlowError("nonparticipants cannot receive export authorization")
        synthetic_pv = np.where(participant, load + authorization, 0.0)
        result, delivered, _ = self.solve(
            load,
            synthetic_pv,
            authorization,
            participant,
        )
        if not np.allclose(delivered, authorization, rtol=0.0, atol=1.0e-9):
            raise AustralianPowerFlowError("authorized-export endpoint did not reproduce its allocation")
        return result


__all__ = [
    "AustralianDSSScreenResult",
    "AustralianOpenDSSModel",
    "AustralianPowerFlowError",
    "connection_point_power",
]
