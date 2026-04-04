import copy
import heapq
from multiprocessing import Pool, cpu_count
from timeit import default_timer
from typing import Callable

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from ...mpo import MatrixProductOperator
from ...mps import MatrixProductState
from ...tn import TensorNetwork
from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result


class QSCI(QuantumAlgorithm):
    def __init__(
        self,
        circuits: list[QuantumCircuit],
        backend: QuantumBackend | None,
        hamiltonian: dict[str, complex],
        num_electrons: int | None = None,
        postprocessing_functions: list[Callable] | None = None,
        postprocessing_args: list[dict] | None = None,
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        self.hamiltonian = hamiltonian
        self.num_electrons = num_electrons
        self.hamiltonian_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        self.circuits = circuits
        self.set_backend(backend=backend)
        self.important_configurations = []
        self.unimportant_configurations = []
        self.energy = None
        self.postprocessing_functions = postprocessing_functions
        self.postprocessing_args = postprocessing_args

    def execute_circuits(self, shots_per_circuit: int) -> dict:
        """Execute circuits and collect results"""
        all_counts = {}
        for circ in self.circuits:
            counts = self.backend.run(circ, shots_per_circuit)
            for b, c in counts.items():
                all_counts[b] = all_counts.get(b, 0) + c
        return all_counts

    def postprocessing(self, counts: dict[str, int], args: list[dict] | None):
        if self.postprocessing_functions is None:
            return counts
        else:
            assert len(self.postprocessing_functions) == len(self.postprocessing_args)
            for idx in range(len(self.postprocessing_functions)):
                func = self.postprocessing_functions[idx]
                func_args = args[idx]
                if func_args is None:
                    counts = func(counts)
                else:
                    counts = func(counts, **args)
            return counts

    def symmetry_verification(
        self, counts: dict[str, int], particle_number: int | None = None
    ) -> dict:
        """Perform symmetry verification"""
        new_counts = {k: v for k, v in counts.items() if v > 0}
        for sample in self.unimportant_configurations:
            counts[sample] = 0
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

    def compute_hij(self, args):
        i, j, basis, hamiltonian_mpo = args
        psi_i = MatrixProductState.from_bitstring(basis[i])
        psi_j = MatrixProductState.from_bitstring(basis[j])
        ham = copy.deepcopy(hamiltonian_mpo)
        psi_i.set_default_indices("X", "A")
        ham.set_default_indices("Y", "A", "B")
        psi_j.set_default_indices("Z", "B")
        tn = TensorNetwork(psi_i.tensors + ham.tensors + psi_j.tensors)
        h_ij = tn.contract_entire_network()
        return (i, j, h_ij)

    def project_hamiltonian(
        self, samples: list[str], reset_hamiltonian: bool = False
    ) -> ndarray:
        """Project Hamiltonian onto subspace"""
        n = len(samples)
        ham_proj = np.zeros((n, n), dtype=complex)

        # Prepare arguments for each task
        if reset_hamiltonian:
            ham = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        else:
            ham = self.hamiltonian_mpo
        args_list = [(i, j, samples, ham) for i in range(n) for j in range(i, n)]

        # Launch worker pool
        with Pool(processes=cpu_count() - 1) as pool:
            results = pool.map(self.compute_hij, args_list)

        # Fill in the matrix from results
        for i, j, h_ij in results:
            ham_proj[i, j] = h_ij
            if i != j:
                ham_proj[j, i] = h_ij.conjugate()

        return ham_proj

    def exact_diagonalisation(
        self, hamiltonian_matrix: ndarray
    ) -> tuple[float, ndarray]:
        """Perform exact diagonalisation on the projected Hamiltonian"""
        if hamiltonian_matrix.shape[0] >= 200:
            eval, evec = eigsh(hamiltonian_matrix, k=1, which="SA", tol=1e-10)
        else:
            eval, evec = eigh(hamiltonian_matrix)
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
        eval, evec = eigsh(H_linear, k=1, which="SA", tol=1e-10)
        return eval[0], evec[:, 0]

    def run(self, num_shots: int, subspace_size: int) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        shots_per_circuit = int(num_shots / len(self.circuits))
        counts = self.execute_circuits(self.circuits, shots_per_circuit)
        counts = self.postprocessing(counts, self.postprocessing_args)
        sv_counts = self.symmetry_verification(counts, self.num_electrons)
        samples = self.gather_samples(sv_counts, subspace_size)
        if len(samples) <= 500:
            projected_ham = self.project_hamiltonian(samples)
            self.energy, groundstate_vec = self.exact_diagonalisation(projected_ham)
        else:
            self.energy, groundstate_vec = self.linear_operator_diagonalisation(samples)
        for sidx in range(len(samples)):
            sample = samples[sidx]
            amp = groundstate_vec[sidx]
            if np.abs(amp) ** 2 > 0.0:
                self.important_configurations.append(sample)
            self.important_configurations = list(set(self.important_configurations))
        end_time = default_timer()

        metadata = {
            "algorithm_name": "QSCI",
            "num_shots": num_shots,
            "max_subspace_size": subspace_size,
            "actual_subspace_size": len(samples),
            "subspace": samples,
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits

        result = self.energy, groundstate_vec

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
