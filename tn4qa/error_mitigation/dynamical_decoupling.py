"""
Module 1: Dynamical Decoupling Transpiler
Transpiles a Qiskit circuit to a target backend and applies a chosen
dynamical decoupling (DD) sequence to idle qubit windows.
"""

from __future__ import annotations

from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    XGate,
    YGate,
)
from qiskit.providers import BackendV2
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)

DDScheme = Literal["XY4", "XY8", "CPMG", "X2", "EDD"]


_DD_SEQUENCES: dict[DDScheme, list] = {
    # Hahn echo / two-pulse: suppresses low-frequency Z noise
    "CPMG": [XGate(), XGate()],
    # Simple X echo
    "X2": [XGate(), XGate()],
    # XY-4: suppresses both X and Z noise to first order
    "XY4": [XGate(), YGate(), XGate(), YGate()],
    # XY-8: second-order suppression, robust to pulse errors
    "XY8": [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()],
    # Eulerian DD: constructed so gate errors also cancel
    "EDD": [XGate(), YGate(), XGate(), YGate(), XGate(), YGate(), XGate(), YGate()],
}

DEFAULT_GATE_TIMES = {
    "cx": 300,  # ns
    "cz": 300,
    "ecr": 280,
    "y": 35,
    "z": 35,
    "sx": 35,
    "x": 35,
    "rx": 35,
    "ry": 35,
    "rz": 0,
    "measure": 4000,
    "reset": 500,
}


class DynamicalDecouplingTranspiler:
    """
    Transpile a QuantumCircuit to a backend and insert a dynamical
    decoupling sequence on idle qubit windows.

    Parameters
    ----------
    backend : BackendV2
        Target backend (real or fake).
    dd_scheme : DDScheme
        One of "CPMG", "X2", "XY4", "XY8", "EDD".
    optimization_level : int
        Qiskit transpiler optimisation level (0–3).
    skip_reset_qubits : bool
        If True, DD pulses are not inserted after reset operations.
    """

    SCHEMES: list[DDScheme] = list(_DD_SEQUENCES.keys())

    def __init__(
        self,
        backend: BackendV2,
        dd_scheme: DDScheme = "XY4",
        optimization_level: int = 1,
        skip_reset_qubits: bool = True,
    ) -> None:
        if dd_scheme not in _DD_SEQUENCES:
            raise ValueError(
                f"Unknown DD scheme '{dd_scheme}'. " f"Choose from: {self.SCHEMES}"
            )
        self.backend = backend
        self.dd_scheme = dd_scheme
        self.optimization_level = optimization_level
        self.skip_reset_qubits = skip_reset_qubits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Transpile *circuit* to the backend and apply the chosen DD sequence.

        Returns
        -------
        QuantumCircuit
            A scheduled, DD-padded circuit ready for execution.
        """
        # Step 1 — standard transpilation
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            backend=self.backend,
        )
        transpiled = pm.run(circuit)

        # Step 2 — ALAP scheduling + DD padding
        num_qubits = self.backend.num_qubits
        coupling_map = self.backend.coupling_map
        instruction_duration_list = []
        for key, val in DEFAULT_GATE_TIMES.items():
            if key in ["cx", "cz", "ecr"]:
                for q1, q2 in coupling_map:
                    inst = (key, (q1, q2), val)
                    instruction_duration_list.append(inst)
            else:
                for q in range(num_qubits):
                    inst = (key, q, val)
                    instruction_duration_list.append(inst)

        durations = InstructionDurations(instruction_duration_list, dt=1e-9)

        dd_sequence = _DD_SEQUENCES[self.dd_scheme]
        dd_pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(
                    durations,
                    dd_sequences=dd_sequence,
                    skip_reset_qubits=self.skip_reset_qubits,
                ),
            ]
        )
        dd_circuit = dd_pm.run(transpiled)
        return dd_circuit

    def transpile_and_compare(
        self, circuit: QuantumCircuit
    ) -> tuple[QuantumCircuit, QuantumCircuit]:
        """
        Return both the plain transpiled circuit and the DD-augmented
        circuit so the caller can compare depth / gate counts.
        """
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            backend=self.backend,
        )
        transpiled = pm.run(circuit)

        dd_sequence = _DD_SEQUENCES[self.dd_scheme]
        num_qubits = self.backend.num_qubits
        coupling_map = self.backend.coupling_map
        instruction_duration_list = []
        for key, val in DEFAULT_GATE_TIMES.items():
            if key in ["cx", "cz", "ecr"]:
                for q1, q2 in coupling_map:
                    inst = (key, (q1, q2), val)
                    instruction_duration_list.append(inst)
            else:
                for q in range(num_qubits):
                    inst = (key, q, val)
                    instruction_duration_list.append(inst)

        durations = InstructionDurations(instruction_duration_list, dt=1e-9)
        dd_pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(
                    durations,
                    dd_sequences=dd_sequence,
                    skip_reset_qubits=self.skip_reset_qubits,
                ),
            ]
        )
        dd_circuit = dd_pm.run(transpiled)
        return transpiled, dd_circuit

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def describe_schemes() -> dict[DDScheme, str]:
        """Return a human-readable description of each available scheme."""
        return {
            "CPMG": "Carr-Purcell-Meiboom-Gill: [X, X]. "
            "Suppresses low-frequency dephasing (Z noise). "
            "Best when T2 << T1.",
            "X2": "Simple two-pulse X echo. Equivalent to CPMG "
            "with X pulses. Minimal overhead.",
            "XY4": "XY-4: [X, Y, X, Y]. First-order suppression of "
            "both X and Z noise. Good general-purpose choice.",
            "XY8": "XY-8: [X,Y,X,Y,Y,X,Y,X]. Second-order suppression, "
            "robust to systematic pulse errors.",
            "EDD": "Eulerian DD: 8-pulse sequence where rotation-axis "
            "errors also cancel. Best for high-coherence qubits "
            "with precise pulse control.",
        }
