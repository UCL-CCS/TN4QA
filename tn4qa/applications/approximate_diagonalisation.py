import copy
import re

import numpy as np
from numpy import ndarray
from numpy.linalg import eig, svd
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from ..fidelity_metrics import hilbert_schmidt_inner_product
from ..mpo import MatrixProductOperator
from ..quantum_algorithms.variational.ansatz_circuits import random_brickwork_circuit
from ..tensor import Tensor
from ..tn import TensorNetwork


class ApproximateDiagonalisation:
    """
    A class for approximately diagonalising an MPO
    """

    def __init__(self, mpo: MatrixProductOperator, num_layers: int) -> None:
        """
        Class constructor.

        Args:
            mpo: The MPO that will be appoximately diagonalised
            num_layers: The number of layers to use in the ansatz circuit
        """
        self.reference = MatrixProductOperator.from_increasing_diagonal_matrix(
            mpo.num_sites
        )
        self.qc = random_brickwork_circuit(mpo.num_sites, num_layers, 3)
        self.num_qubits = mpo.num_sites
        self.mpo_to_diag = mpo
        self.set_ansatz()
        self.set_ansatz_dag()
        self.set_default_indices()
        self.build_tn()
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )

    def set_default_indices(self) -> None:
        """
        Set default indices for the full TN
        """

        def _index_splitter(idx):
            """Split the TN index into QW<number>, N<number>"""
            match = re.match(r"(QW\d+)(N\d+)", idx)
            qw, n = match.groups()
            return qw, n

        def _get_left_tn_indices():
            """Get all TN indices with N number 0"""
            left_tn_indices = [0] * self.num_qubits
            for t in self.ansatz.tensors:
                for idx in t.indices:
                    qw, n = _index_splitter(idx)
                    if n[1:] == "0":
                        left_tn_indices[int(qw[2:]) - 1] = idx
            return left_tn_indices

        def _get_right_tn_indices():
            """Get all TN indices with maximum N number for each QW number"""
            index_dict = {f"QW{x}": [] for x in range(1, self.num_qubits + 1)}
            for t in self.ansatz.tensors:
                for idx in t.indices:
                    qw, n = _index_splitter(idx)
                    index_dict[qw].append(int(n[1:]))
            right_tn_tensors = []
            for k, v in index_dict.items():
                max_n_number = max(v)
                index = k + "N" + str(max_n_number)
                right_tn_tensors.append(index)
            return right_tn_tensors

        left_tn_indices = _get_left_tn_indices()
        right_tn_indies = _get_right_tn_indices()

        self.reference.set_default_indices(
            internal_prefix="A", input_prefix="T", output_prefix="V"
        )
        for t in self.ansatz_dag.tensors:
            original_t_indices = t.indices
            new_t_indices = []
            for idx in original_t_indices:
                qw, _ = _index_splitter(idx)
                if idx in left_tn_indices:
                    new_t_indices.append(f"V{qw[2:]}")
                elif idx in right_tn_indies:
                    new_t_indices.append(f"W{qw[2:]}")
                else:
                    new_t_indices.append(idx)
            t.indices = new_t_indices
        self.mpo_to_diag.set_default_indices(
            internal_prefix="B", input_prefix="W", output_prefix="X"
        )
        for t in self.ansatz.tensors:
            original_t_indices = t.indices
            new_t_indices = []
            for idx in original_t_indices:
                qw, _ = _index_splitter(idx)
                if idx in left_tn_indices:
                    new_t_indices.append(f"X{qw[2:]}")
                elif idx in right_tn_indies:
                    new_t_indices.append(f"T{qw[2:]}")
                else:
                    new_t_indices.append(idx + "_")
            t.indices = new_t_indices
        return

    def update_circuit(self, variational_index: int, optimal_update: ndarray) -> None:
        """
        Update the quantum circuit with the optimal local update

        Args:
            variational_index: The local index to be updated
            optimal_value: The optimal update array
        """
        new_inst = UnitaryGate(optimal_update)
        qidxs = [
            self.qc.data[variational_index - 1].qubits[x]._index
            for x in range(len(self.qc.data[variational_index - 1].qubits))
        ]
        self.qc.data[variational_index - 1] = (new_inst, qidxs[::-1], [])
        return

    def set_ansatz(self) -> None:
        """Update ansatz after circuit update"""
        self.ansatz = TensorNetwork.from_qiskit_circuit(self.qc)
        for t in self.ansatz.tensors:
            t.labels.append(f"variational_site_{self.ansatz.tensors.index(t)+1}")
        return

    def set_ansatz_dag(self) -> None:
        """Update ansatz dag after circuit update"""
        self.ansatz_dag = TensorNetwork.from_qiskit_circuit(self.qc, dagger=True)
        for t in self.ansatz_dag.tensors:
            t.labels.append(f"variational_site_{self.ansatz_dag.tensors.index(t)+1}")
        return

    def build_tn(self) -> None:
        """Build the full tensor network assuming indices have been set"""
        self.tn = TensorNetwork(
            self.reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
        return

    def construct_approximately_diagonalised_mpo(self) -> "MatrixProductOperator":
        """
        Construct the approximately diagonalised MPO
        """
        mpo = copy.deepcopy(self.mpo_to_diag)
        ansatz_circ = copy.deepcopy(self.qc)
        ansatz_dag = ansatz_circ.inverse()
        for inst in ansatz_circ.data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            submpo = MatrixProductOperator.from_qiskit_gate(inst)
            mpo = mpo.contract_sub_mpo(submpo, qidxs, contract_right=True)
        for inst in ansatz_dag.data[::-1]:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            submpo = MatrixProductOperator.from_qiskit_gate(inst)
            mpo = mpo.contract_sub_mpo(submpo, qidxs, contract_right=False)
        return mpo

    def get_local_indices(self, variational_idx: int) -> tuple[list[str], list[str]]:
        """
        For the local site get the expected indices of the environment tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            output_inds, input_inds for the environment tensor
        """
        ansatz_tensor = self.ansatz.tensors[variational_idx - 1]
        ansatz_dag_tensor = self.ansatz_dag.tensors[variational_idx - 1]
        input_inds = ansatz_tensor.indices
        output_inds = ansatz_dag_tensor.indices
        return output_inds, input_inds

    def form_environment_matrix(self, variational_idx: int) -> ndarray:
        """
        Form the environment matrix for a local variational tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            A matrix for the environment
        """
        tn_copy = copy.deepcopy(self.tn)
        site_label = f"variational_site_{variational_idx}"
        tn_copy.pop_tensors_by_label([site_label])
        env_tensor = tn_copy.contract_entire_network()
        env_copy = copy.deepcopy(env_tensor)
        output_inds, input_inds = self.get_local_indices(variational_idx)
        env_copy.tensor_to_matrix(input_inds, output_inds)
        env_mat = env_copy.data.todense()
        return env_mat

    def get_maximum_eigenvector(self, mat: ndarray) -> ndarray:
        """
        For a given matrix, get the eigenvector associated to the maximum eigenvalue

        Args:
            mat: The input matrix

        Returns:
            The maximum eigenvector
        """
        evals, evecs = eig(mat)

        # Sort evals
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Find degeneracy in maximum eigenvector
        tol = 1e-8
        maxval = evals[0]
        degenerate_indices = np.where(np.abs(evals - maxval) < tol)[0]

        # Take a linear combination
        max_evec = evecs[:, degenerate_indices[0]]
        for idx in degenerate_indices[1:]:
            max_evec += evecs[:, idx]

        return max_evec

    def get_closest_unitary(self, mat: ndarray) -> ndarray:
        """
        Get the closest unitary to a given matrix

        Args:
            mat: The input matrix

        Returns:
            The closest unitary to mat under Frobenius norm
        """
        u, _, vh = svd(mat, full_matrices=False)
        unitary_part = u @ vh
        return unitary_part

    def local_update(self, variational_index: int) -> None:
        """
        Perform a local optimisation at the given index

        Args:
            variational_index: The index of the current local site
        """
        local_tensor = self.ansatz.tensors[variational_index - 1]
        dim = int(2 ** (len(local_tensor.indices) / 2))

        env_mat = self.form_environment_matrix(variational_index)
        max_evec = self.get_maximum_eigenvector(env_mat)
        new_site_data = max_evec.reshape((dim, dim))
        new_site_data = self.get_closest_unitary(new_site_data)

        self.update_circuit(variational_index, new_site_data)
        self.set_ansatz()
        self.set_ansatz_dag()
        self.set_default_indices()
        self.build_tn()

        return

    def run(self, num_sweeps: int = 10) -> QuantumCircuit:
        """
        Optimise the ansatz to approximately diagonalise the given MPO

        Args:
            num_sweeps: The number of sweeps to perform

        Returns:
            The optimised quantum circuit
        """
        for _ in range(num_sweeps):
            for idx in range(1, len(self.qc.data) + 1):
                self.local_update(idx)
            for idx in list(range(1, len(self.qc.data) + 1))[::-1]:
                self.local_update(idx)
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )
        return self.qc


class ApproximateDiagonalisationMPO:
    """
    A class for approximately diagonalising an MPO
    """

    def __init__(self, mpo: MatrixProductOperator, max_bond: int) -> None:
        """
        Class constructor.

        Args:
            mpo: The MPO that will be appoximately diagonalised
            max_bond: The maximum allowed bond dimension
        """
        reference = MatrixProductOperator.from_increasing_diagonal_matrix(mpo.num_sites)
        self.ansatz = MatrixProductOperator.random_mpo(mpo.num_sites, max_bond)
        self.mpo_to_diag = mpo
        self.max_bond = max_bond
        self.ansatz_dag = copy.deepcopy(self.ansatz)
        self.ansatz_dag.dagger()
        for t in self.ansatz.tensors:
            t.labels.append("variational")
            t.labels.append(f"variational_site_{self.ansatz.tensors.index(t)+1}")
        for t in self.ansatz_dag.tensors:
            t.labels.append(f"variational_site_{self.ansatz_dag.tensors.index(t)+1}")
        reference.set_default_indices(
            internal_prefix="A", input_prefix="T", output_prefix="V"
        )
        self.ansatz_dag.set_default_indices(
            internal_prefix="B", input_prefix="V", output_prefix="W"
        )
        self.mpo_to_diag.set_default_indices(
            internal_prefix="C", input_prefix="W", output_prefix="X"
        )
        self.ansatz.set_default_indices(
            internal_prefix="D", input_prefix="X", output_prefix="T"
        )
        self.tn = TensorNetwork(
            reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
        self.reference = reference
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )

    def construct_approximately_diagonalised_mpo(self) -> "MatrixProductOperator":
        """
        Construct the approximately diagonalised MPO
        """
        adag = copy.deepcopy(self.ansatz_dag)
        mpo = copy.deepcopy(self.mpo_to_diag)
        a = copy.deepcopy(self.ansatz)
        approximately_diagonalised_mpo = adag * mpo
        approximately_diagonalised_mpo = approximately_diagonalised_mpo * a
        return approximately_diagonalised_mpo

    def get_local_indices(self, variational_idx: int) -> tuple[list[str], list[str]]:
        """
        For the local site get the expected indices of the environment tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            output_inds, input_inds for the environment tensor
        """
        ansatz_tensor = self.ansatz.tensors[variational_idx - 1]
        ansatz_dag_tensor = self.ansatz_dag.tensors[variational_idx - 1]
        input_inds = ansatz_tensor.indices
        output_inds = ansatz_dag_tensor.indices
        return output_inds, input_inds

    def form_environment_matrix(self, variational_idx: int) -> ndarray:
        """
        Form the environment matrix for a local variational tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            A matrix for the environment
        """
        tn_copy = copy.deepcopy(self.tn)
        site_label = f"variational_site_{variational_idx}"
        tn_copy.pop_tensors_by_label([site_label])
        env_tensor = tn_copy.contract_entire_network()
        env_copy = copy.deepcopy(env_tensor)
        output_inds, input_inds = self.get_local_indices(variational_idx)
        env_copy.tensor_to_matrix(input_inds, output_inds)
        env_mat = env_copy.data.todense()
        return env_mat

    def get_maximum_eigenvector(self, mat: ndarray) -> ndarray:
        """
        For a given matrix, get the eigenvector associated to the maximum eigenvalue

        Args:
            mat: The input matrix

        Returns:
            The maximum eigenvector
        """
        evals, evecs = eig(mat)
        max_eval = max(evals)
        max_eval_index = list(evals).index(max_eval)
        max_evec = evecs[:, max_eval_index]
        return max_evec

    def get_closest_isometry(self, data: ndarray) -> ndarray:
        """
        The global unitary needs to be unitary. We can achieve this by ensuring each local update is an isometry.

        Args:
            data: The optimised data for a local update

        Returns:
            The data for the closest isometry
        """
        u, _, vh = svd(data, full_matrices=False)
        new_data = u @ vh
        return new_data

    def local_update(self, variational_index: int) -> None:
        """
        Perform a local optimisation at the given index

        Args:
            variational_index: The index of the current local site
        """
        local_tensor = self.ansatz.tensors[variational_index - 1]
        local_indices = local_tensor.indices
        local_dims = [local_tensor.get_dimension_of_index(idx) for idx in local_indices]
        local_labels = local_tensor.labels

        env_mat = self.form_environment_matrix(variational_index)
        max_evec = self.get_maximum_eigenvector(env_mat)
        new_site_data = max_evec.reshape(tuple(local_dims))
        new_site_data = self.get_closest_isometry(new_site_data)
        new_tensor = Tensor(new_site_data, indices=local_indices, labels=local_labels)
        site_label = f"variational_site_{variational_index}"
        self.ansatz.pop_tensors_by_label([site_label])
        self.ansatz.add_tensor(new_tensor, variational_index - 1)
        self.update_ansatz_dag()
        return

    def check_global_unitarity(self) -> None:
        """
        The output MPO will be a scaled version of a unitary. Here we enforce full unitarity.
        """
        # Check the magnitude
        ansatz = copy.deepcopy(self.ansatz)
        ansatz_copy = copy.deepcopy(self.ansatz)
        ip = hilbert_schmidt_inner_product(ansatz, ansatz_copy).real
        scale_factor = np.sqrt(ip / (2**self.ansatz.num_sites))
        self.ansatz.multiply_by_constant(1 / scale_factor)
        self.ansatz_dag.multiply_by_constant(1 / scale_factor)
        return

    def update_ansatz_dag(self) -> None:
        """
        Update ansatz_dag after changes to ansatz.
        """
        self.ansatz_dag = copy.deepcopy(self.ansatz)
        self.ansatz_dag.dagger()
        self.ansatz_dag.set_default_indices(
            internal_prefix="B", input_prefix="V", output_prefix="W"
        )
        return

    def update_tn(self) -> None:
        """
        Update tn after changes to ansatz.
        """
        self.tn = TensorNetwork(
            self.reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
        return

    def run(self, num_sweeps: int = 10) -> MatrixProductOperator:
        """
        Optimise the ansatz to approximately diagonalise the given MPO

        Args:
            num_sweeps: The number of sweeps to perform

        Returns:
            The optimised ansatz
        """
        for _ in range(num_sweeps):
            for idx in range(1, self.ansatz.num_sites + 1):
                self.local_update(idx)
                self.update_tn()
            for idx in list(range(1, self.ansatz.num_sites + 1))[::-1]:
                self.local_update(idx)
                self.update_tn()
        self.check_global_unitarity()
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )
        return self.ansatz
