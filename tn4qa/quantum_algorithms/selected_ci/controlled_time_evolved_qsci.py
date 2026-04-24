import copy
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from ...mps import MatrixProductState
from ...quantum_algorithms.hamiltonian_simulation.qdrift import (
    QDriftSimulation,
)
from ...quantum_algorithms.hamiltonian_simulation.trotterisation import (
    TrotterSimulation,
)
from ...tn_methods.mps_to_circuit import MPSAnalyticDecomposition
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import QiskitSimulatorBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import add_controls
from .qsci import QSCI


class ControlledTimeEvolvedQSCI(QuantumAlgorithm):
    def __init__(
        self,
        hamiltonian: dict,
        reference_state: MatrixProductState,
        duration: float = np.pi,
        num_circuits: int = 5,
        backend: QuantumBackend | None = None,
        qdrift: bool = True,
        num_qdrift_circuits: int | None = 10,
        qdrift_error: float | None = None,
        max_qdrift_terms: int | None = None,
        num_electrons: int | None = None,
        known_important_configurations: list[str] = [],
        known_unimportant_configurations: list[str] = [],
        postprocessing_function: Callable | None = None,
        postprocessing_args: dict | None = None,
    ) -> "ControlledTimeEvolvedQSCI":
        """
        Constructor for TE-QSCI class.
        """
        # self.duration = duration
        self.circuits = []
        self.num_circuits = num_circuits
        self.norm = None
        hamiltonian = self.sanitize_dict(hamiltonian)
        hamiltonian = self.normalise_hamiltonian(hamiltonian)
        self.hamiltonian = self.rescale_hamiltonian(hamiltonian)
        self.duration = (2 * self.norm * duration) / np.pi
        self.reference_state = reference_state
        self.reference_state_qc = self.create_reference_circuit()
        self.backend = self.set_backend(backend=backend)

        self.qdrift = qdrift
        self.num_qdrift_circuits = num_qdrift_circuits
        self.qdrift_error = qdrift_error
        self.max_qdrift_terms = max_qdrift_terms

        self.num_electrons = num_electrons
        self.important_configurations = known_important_configurations
        self.unimportant_configurations = known_unimportant_configurations
        self.postprocessing_function = postprocessing_function
        self.postprocessing_args = postprocessing_args

        self.circuit_depths = []
        self.energy = None

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit from the algorithm"""
        return self.circuits

    def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
        return {
            k: float(v.real) if isinstance(v, complex) else float(v)
            for k, v in d.items()
        }

    def normalise_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
        self.norm = np.sum([np.abs(x) for x in d.values()])
        return {k: v / self.norm for k, v in d.items()}

    def rescale_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
        num_qubits = len(list(d.keys())[0])
        d["I" * num_qubits] = d.get("I" * num_qubits, 0) + 1.0
        return {k: v * np.pi / 2 for k, v in d.items()}

    def create_reference_circuit(self) -> QuantumCircuit:
        """Create a circuit to prepare the reference state"""
        if self.reference_state.bond_dimension <= 2:
            mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
            qc = mpstocirc.bond_dim_2_to_qc_exact(self.reference_state)
        else:
            mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
            qc = mpstocirc.mps_to_qc_via_ttn(self.reference_state, 2)
        return qc

    def perform_controlled_time_evolution(self, duration: float) -> QuantumCircuit:
        """Add time evolution to the circuit"""
        if duration == 0.0:
            ref = copy.deepcopy(self.reference_state_qc)
            return ref
        sim = TrotterSimulation(self.hamiltonian, duration=duration)
        sim_circ = sim.circuit
        controlled_sim_circ = QuantumCircuit(sim_circ.num_qubits + 1)
        controlled_sim_circ.compose(
            sim_circ, qubits=range(1, sim_circ.num_qubits + 1), inplace=True
        )
        controlled_sim_circ = add_controls(controlled_sim_circ, [0])
        circ_copy = copy.deepcopy(self.reference_state_qc)
        ref = QuantumCircuit(circ_copy.num_qubits + 1)
        ref.h(0)
        ref.compose(circ_copy, qubits=range(1, circ_copy.num_qubits + 1), inplace=True)
        ref.compose(controlled_sim_circ, inplace=True)
        ref.h(0)
        return ref

    def perform_controlled_time_evolution_qdrift(
        self, duration: float, error: float | None = None
    ) -> QuantumCircuit:
        """Add qdrift time evolution to the circuit"""
        if duration == 0.0:
            ref = copy.deepcopy(self.reference_state_qc)
            return ref
        sim = QDriftSimulation(
            self.hamiltonian,
            duration=duration,
            error=error,
            max_num_terms=self.max_qdrift_terms,
        )
        sim_circ = sim.circuit
        controlled_sim_circ = QuantumCircuit(sim_circ.num_qubits + 1)
        controlled_sim_circ.compose(
            sim_circ, qubits=range(1, sim_circ.num_qubits + 1), inplace=True
        )
        controlled_sim_circ = add_controls(controlled_sim_circ, [0])
        circ_copy = copy.deepcopy(self.reference_state_qc)
        ref = QuantumCircuit(circ_copy.num_qubits + 1)
        ref.h(0)
        ref.compose(circ_copy, qubits=range(1, circ_copy.num_qubits + 1), inplace=True)
        ref.compose(controlled_sim_circ, inplace=True)
        ref.h(0)
        return ref

    def build_circuits_trotter(self, duration: float) -> list[QuantumCircuit]:
        """Get circuits using Trotterisation"""
        duration_per_circuit = duration / self.num_circuits
        circuits = []
        for idx in range(self.num_circuits):
            qc = self.perform_controlled_time_evolution(
                (idx + 1) * duration_per_circuit
            )
            circuits.append(qc)
        return circuits

    def build_circuits_qdrift(self, duration: float) -> list[QuantumCircuit]:
        """Get circuits using qDRIFT"""
        duration_per_circuit = duration / (self.num_circuits)
        circuits = []
        for idx in range(self.num_circuits):
            for _ in range(self.num_qdrift_circuits):
                qc = self.perform_controlled_time_evolution_qdrift(
                    (idx + 1) * duration_per_circuit, error=self.qdrift_error
                )
                circuits.append(qc)
        return circuits

    def build_circuits(self) -> list[QuantumCircuit]:
        if self.qdrift:
            circuits = self.build_circuits_qdrift(self.duration)
        else:
            circuits = self.build_circuits_trotter(self.duration)
        self.circuits = circuits
        return circuits

    def post_selection(self, counts: dict[str, int]) -> dict[str, int]:
        """Post select counts based on the ancilla output"""
        new_counts = {}
        for b, count in counts.items():
            if len(b) == self.reference_state.num_sites:
                new_counts[b] = new_counts.get(b, 0) + count
            elif b[0] == "0":
                new_counts[b[1:]] = new_counts.get(b[1:], 0) + count
        return new_counts

    def rescale_energy(self, energy_prime):
        return self.norm * ((2 / np.pi) * energy_prime - 1)

    def run(self, num_shots: int, subspace_size: int) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        circuits = self.build_circuits()
        if self.postprocessing_function is not None:
            postprocessing_funcs = [self.post_selection] + self.postprocessing_function
            func_args = [None] + self.postprocessing_args
        else:
            postprocessing_funcs = [self.post_selection]
            func_args = [None]
        qsci = QSCI(
            circuits,
            self.backend,
            self.hamiltonian,
            self.num_electrons,
            self.important_configurations,
            self.unimportant_configurations,
            postprocessing_funcs,
            func_args,
        )
        result = qsci.run(num_shots, subspace_size)
        rescaled_energy = self.rescale_energy(result.result[0])
        result.result = rescaled_energy, result.result[1]

        return result

    def set_backend(self, backend: QuantumBackend | None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = QiskitSimulatorBackend()
        self.backend = backend
        return
