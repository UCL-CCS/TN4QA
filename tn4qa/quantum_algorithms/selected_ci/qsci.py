import copy
import heapq
from timeit import default_timer

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from scipy.sparse.linalg import LinearOperator, eigsh

from ...dmrg import DMRG
from ...mpo import MatrixProductOperator
from ...mps import MatrixProductState
from ...tn_methods.hf_suppression import HFSuppression
from ...tn_methods.mps_to_circuit import MPStoCircuit
from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result


class QSCI(QuantumAlgorithm):
    def __init__(
        self,
        hamiltonian: dict[str, complex],
        hf_state: str | None = None,
        backend: QuantumBackend | None = None,
        hfs: bool = True,
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        self.hamiltonian = hamiltonian
        self.hf_state = hf_state
        if hf_state is not None:
            self.num_electrons = hf_state.count("1")
            self.num_qubits = len(hf_state)
        else:
            self.num_electrons = None
            self.num_qubits = None
        self.hfs = hfs
        self.hamiltonian_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        self.state, self.energy = self.run_dmrg(self.hamiltonian)
        self._circuit = None
        self.set_backend(backend=backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def run_dmrg(
        self, hamiltonian: dict, max_bond: int = 8, maxiter: int = 6
    ) -> MatrixProductState:
        """Run DMRG"""
        hf_mps = None
        if self.hf_state is not None:
            hf_mps = MatrixProductState.from_bitstring(self.hf_state)
            hf_mps = hf_mps.expand_bond_dimension_list(
                1, list(range(1, self.num_qubits))
            )
        dmrg = DMRG(hamiltonian, max_mps_bond=max_bond, initial_state=hf_mps)
        dmrg.run(maxiter=maxiter)
        return dmrg.mps, dmrg.energy

    def hf_suppression(
        self, mps: MatrixProductState, mpo: MatrixProductOperator | None = None
    ) -> tuple[MatrixProductState, MatrixProductOperator]:
        """Perform HF suppression"""
        hfs = HFSuppression(self.hf_state, mps, mpo)
        hfs.run()
        return hfs.suppressed_mps, hfs.evolved_mpo

    def prepare_state(self, mps: MatrixProductState) -> QuantumCircuit:
        """Prepare an MPS reference on quantum device"""
        circ = MPStoCircuit(mps, 1, 1.0).run()
        return circ

    def configuration_recovery(
        self, counts: dict[str, int], particle_number: int | None = None
    ) -> dict:
        """Perform configuration recovery"""
        new_counts = {k: v for k, v in counts.items() if v > 2}
        if particle_number is None:
            return new_counts
        else:
            new_counts = {
                k: v for k, v in new_counts.items() if k.count("1") == particle_number
            }
            return new_counts

    def gather_samples(self, cr_counts: dict, k: int) -> list[str]:
        """Collect the (at most) k most frequent samples to form the selected subspace"""
        top_samples = heapq.nlargest(k, cr_counts, key=cr_counts.get)
        return top_samples

    def project_hamiltonian(self, samples: list[str]) -> ndarray:
        """Project Hamiltonian onto subspace"""
        ham_mpo = copy.deepcopy(self.hamiltonian_mpo)
        basis = [MatrixProductState.from_bitstring(s) for s in samples]
        n = len(basis)
        ham_proj = np.zeros((n, n), dtype=complex)

        for i in range(n):
            h_i = basis[i].apply_mpo(ham_mpo)
            for j in range(i, n):
                h_ij = h_i.compute_inner_product(basis[j])
                ham_proj[i, j] = h_ij
                if i != j:
                    ham_proj[j, i] = h_ij.conjugate()

        return ham_proj

    def exact_diagonalisation(
        self, hamiltonian_matrix: ndarray
    ) -> tuple[float, ndarray]:
        """Perform exact diagonalisation on the projected Hamiltonian"""
        eval, evec = eigsh(hamiltonian_matrix, k=1, which="SA")
        return eval[0], evec[:, 0]

    def linear_operator_diagonalisation(
        self, samples: list[str]
    ) -> tuple[float, ndarray]:
        """For larger subspaces this will be more efficient"""
        basis_states = [MatrixProductState.from_bitstring(s) for s in samples]
        n = len(basis_states)
        ham_mpo = copy.deepcopy(self.hamiltonian_mpo)

        def matvec(v):
            psi_v = basis_states[0]
            psi_v.multiply_by_constant(v[0])
            for j in range(1, n):
                temp_mps = basis_states[j]
                temp_mps.multiply_by_constant(v[j])
                psi_v = psi_v + temp_mps
            H_psi_v = psi_v.apply_mpo(ham_mpo)
            return np.array(
                [basis_states[i].compute_inner_product(H_psi_v) for i in range(n)]
            )

        H_linear = LinearOperator(shape=(n, n), matvec=matvec, dtype=np.complex128)
        eval, evec = eigsh(H_linear, k=1, which="SA")
        return eval[0], evec[:, 0]

    def reconstruct_mps(
        self, samples: list[str], groundstate_vec: ndarray
    ) -> MatrixProductState:
        """Construct an MPS from the approximate groundstate solution"""
        new_mps = MatrixProductState.from_bitstring(samples[0])
        new_mps.multiply_by_constant(groundstate_vec[0])
        for i in range(1, len(samples)):
            temp_mps = MatrixProductState.from_bitstring(samples[i])
            temp_mps.multiply_by_constant(groundstate_vec[i])
            new_mps = new_mps + temp_mps

        return new_mps

    def run(
        self, num_shots: int, subspace_size: int, num_iterations: int = 1
    ) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        for _ in range(num_iterations):
            if self.hfs:
                self.state, _ = self.hf_suppression(self.state)
            self._circuit = self.prepare_state(self.state)
            counts = self.backend.run(self._circuit, shots=num_shots)
            cr_counts = self.configuration_recovery(counts, self.num_electrons)
            samples = self.gather_samples(cr_counts, subspace_size)
            if self.hfs and self.hf_state not in samples:
                samples += self.hf_state
            if len(samples) <= 500:
                projected_ham = self.project_hamiltonian(samples)
                self.energy, groundstate_vec = self.exact_diagonalisation(projected_ham)
            else:
                self.energy, groundstate_vec = self.linear_operator_diagonalisation(
                    samples
                )
            self.state = self.reconstruct_mps(samples, groundstate_vec)
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
