import copy
from timeit import default_timer

from qiskit import QuantumCircuit

from ...quantum_algorithms.hamiltonian_simulation.trotterisation import (
    TrotterSimulation,
)
from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..result import Result
from .qsci import QSCI


class ControlledTimeEvolvedQSCI(QSCI):
    def __init__(
        self, hamiltonian: dict, backend: QuantumBackend | None = None
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
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

    def run(
        self, num_shots: int, subspace_size: int, num_iterations: int = 1
    ) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        for _ in range(num_iterations):
            self._circuit = self.prepare_state(self.state)
            counts = self.backend.run(self._circuit, shots=num_shots)
            cr_counts = self.configuration_recovery(counts)
            samples = self.gather_samples(cr_counts, subspace_size)
            projected_ham = self.project_hamiltonian(samples)
            self.state, self.energy = self.run_dmrg(projected_ham)
        end_time = default_timer()

        metadata = {
            "algorithm_name": "QSCI",
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
