import copy
from timeit import default_timer

import numpy as np
from qiskit import QuantumCircuit

from ...quantum_algorithms.hamiltonian_simulation.qdrift import (
    QDriftSimulation,
)
from ...quantum_algorithms.hamiltonian_simulation.trotterisation import (
    TrotterSimulation,
)
from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..result import Result
from .qsci import QSCI


class TimeEvolvedQSCI(QSCI):
    def __init__(
        self,
        hamiltonian: dict,
        backend: QuantumBackend | None = None,
        duration: float = np.pi,
        num_circuits: int = 5,
        qdrift: bool = True,
        **kwargs,
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        self.duration = duration
        self.num_circuits = num_circuits
        self.qdrift_config = kwargs
        self.qdrift = qdrift
        super().__init__(hamiltonian, backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def perform_time_evolution(self, duration: float) -> QuantumCircuit:
        """Add time evolution to the circuit"""
        sim = TrotterSimulation(self.hamiltonian, duration=duration)
        sim_circ = sim.circuit
        ref = copy.deepcopy(self.circuit)
        ref.compose(sim_circ, inplace=True)
        return ref

    def perform_time_evolution_qdrift(
        self, duration: float, error: float | None = None
    ) -> QuantumCircuit:
        """Add qdrift time evolution to the circuit"""
        sim = QDriftSimulation(self.hamiltonian, duration=duration, error=error)
        sim_circ = sim.circuit
        ref = copy.deepcopy(self.circuit)
        ref.compose(sim_circ, inplace=True)
        return ref

    def get_counts(
        self, duration: float, num_circuits: int, shots: int
    ) -> dict[str, int]:
        """Get counts using Trotterisation"""
        duration_per_circuit = duration / num_circuits
        counts = {}
        for idx in range(num_circuits):
            qc = self.perform_time_evolution((idx + 1) * duration_per_circuit)
            subcounts = self.backend.run(qc, shots=shots)
            for b, count in subcounts.items():
                counts[b] = counts.get(b, 0) + count
        return counts

    def get_counts_qdrift(
        self, duration: float, num_circuits: int, shots: int
    ) -> dict[str, int]:
        """Get counts using qDRIFT"""
        duration_per_circuit = duration / num_circuits
        counts = {}
        for idx in range(num_circuits):
            shots_per_circuit = int(shots / self.qdrift_config["num_qdrift_circuits"])
            for _ in range(self.qdrift_config["num_qdrift_circuits"]):
                qc = self.perform_time_evolution_qdrift(
                    (idx + 1) * duration_per_circuit, error=self.qdrift_config["error"]
                )
                subcounts = self.backend.run(qc, shots=shots_per_circuit)
                for b, count in subcounts.items():
                    counts[b] = counts.get(b, 0) + count

    def run(
        self, num_shots: int, subspace_size: int, num_iterations: int = 1
    ) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        for _ in range(num_iterations):
            self._circuit = self.prepare_state(self.state)
            if self.qdrift:
                counts = self.get_counts_qdrift(
                    self.duration, self.num_circuits, num_shots
                )
            else:
                counts = self.get_counts(self.duration, self.num_circuits, num_shots)
            cr_counts = self.configuration_recovery(counts)
            samples = self.gather_samples(cr_counts, subspace_size)
            projected_ham = self.project_hamiltonian(samples)
            self.state, self.energy = self.run_dmrg(projected_ham)
        end_time = default_timer()

        metadata = {
            "algorithm_name": "Time Evolved QSCI",
            "qdrift": self.qdrift,
            "num_shots": num_shots,
            "num_iterations": num_iterations,
            "max_subspace_size": subspace_size,
            "actual_subspace_size": len(samples),
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits
        if self.qdrift:
            metadata.update(self.qdrift_config)

        result = self.energy

        result = Result(
            result=result,
            measurements=counts,
            parameters=None,
            metadata=metadata,
        )
        return result

    def set_backend(self, backend: QuantumBackend | None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = TNQuantumBackend()
        self.backend = backend
        return
