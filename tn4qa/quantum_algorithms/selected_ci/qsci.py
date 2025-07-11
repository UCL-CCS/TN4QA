import copy
import heapq
from timeit import default_timer

from qiskit import QuantumCircuit

from ...dmrg import DMRG
from ...mpo import MatrixProductOperator
from ...mps import MatrixProductState
from ...quantum_algorithms.variational.ansatz_circuits import random_staircase_circuit
from ...tn_methods.mps_to_circuit import MPSOptimiser
from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result


class QSCI(QuantumAlgorithm):
    def __init__(
        self, hamiltonian: dict, backend: QuantumBackend | None = None
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        self.hamiltonian = hamiltonian
        self.state, self.energy = self.run_dmrg(self.hamiltonian)
        self._circuit = None
        self.set_backend(backend=backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def run_dmrg(
        self, hamiltonian: dict, max_bond: int = 16, maxiter: int = 10
    ) -> MatrixProductState:
        """Run DMRG"""
        dmrg = DMRG(hamiltonian, max_mps_bond=max_bond)
        dmrg.run(maxiter=maxiter)
        return dmrg.mps, dmrg.energy

    def prepare_state(self, mps: MatrixProductState) -> QuantumCircuit:
        """Prepare an MPS reference on quantum device"""
        qc = random_staircase_circuit(mps.num_sites, 1, 2)
        opt = MPSOptimiser(qc, mps)
        circ = opt.run()
        return circ

    def configuration_recovery(self, counts: dict) -> dict:
        """Perform configuration recovery"""
        return counts

    def gather_samples(self, cr_counts: dict, k: int) -> list[str]:
        """Collect the (at most) k most frequent samples to form the selected subspace"""
        top_samples = heapq.nlargest(k, cr_counts, key=cr_counts.get)
        return top_samples

    def project_hamiltonian(self, samples: list[str]) -> MatrixProductOperator:
        """Project Hamiltonian onto subspace"""
        ham = copy.deepcopy(self.hamiltonian)
        mpo = MatrixProductOperator.from_hamiltonian(ham)
        projector = MatrixProductOperator.projector_from_samples(samples, 128)
        projected_mpo = mpo.project_to_subspace(projector)
        return projected_mpo

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
