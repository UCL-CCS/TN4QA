import copy
import re

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import svd

from ..fidelity_metrics import hilbert_schmidt_fidelity
from ..mps import MatrixProductOperator
from ..tn import TensorNetwork


class MPOOptimiser:
    """
    A class for locally optimising a quantum circuit with respect to a reference MPO and the HS distance
    """

    def __init__(self, qc: QuantumCircuit, reference: MatrixProductOperator) -> None:
        """
        Constructor

        Args:
            qc: The quantum circuit that will be optimised
            reference: The reference MPO
        """
        self.qc = qc
        self.num_qubits = qc.num_qubits
        self.tn = TensorNetwork.from_qiskit_circuit(qc)
        for t in self.tn.tensors:
            t.labels.append(f"variational_site_{self.tn.tensors.index(t)+1}")
        self.reference = reference
        self.left_tn_indices, self.right_tn_indices = self.get_tn_external_indices(
            self.tn
        )
        self.error = self.calculate_error()
        self.fidelity = self.get_fidelity()
        self.optimisation_dict = {
            "optimisation_iteration": [0],
            "error": [self.error],
            "fidelity": [self.fidelity],
        }

    def get_tn_external_indices(self, tn: TensorNetwork) -> tuple[list[str], list[str]]:
        """
        Get the left and right indices of TN
        """

        def _index_splitter(idx):
            """Split the TN index into QW<number>, N<number>"""
            match = re.match(r"(QW\d+)(N\d+)", idx)
            qw, n = match.groups()
            return qw, n

        def _get_left_tn_indices():
            """Get all TN indices with N number 0"""
            left_tn_indices = [0] * self.num_qubits
            for t in tn.tensors:
                for idx in t.indices:
                    qw, n = _index_splitter(idx)
                    if n[1:] == "0":
                        left_tn_indices[int(qw[2:]) - 1] = idx
            return left_tn_indices

        def _get_right_tn_indices():
            """Get all TN indices with maximum N number for each QW number"""
            index_dict = {f"QW{x}": [] for x in range(1, self.num_qubits + 1)}
            for t in tn.tensors:
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
        return left_tn_indices, right_tn_indies

    def trace_rrdag(self) -> complex:
        """
        Calculate Tr(RR^dag) where R is the reference MPO
        """
        r1 = copy.deepcopy(self.reference)
        r2 = copy.deepcopy(self.reference)
        r2.dagger()
        mpo = r1 * r2
        tr = mpo.trace()
        return tr

    def trace_rdagt(self) -> complex:
        """
        Calculate Tr(R^dagT) where R is the reference MPO and T is the TN
        """
        r = copy.deepcopy(self.reference)
        r.dagger()
        tn = copy.deepcopy(self.tn)

        def _index_splitter(idx):
            """Split the TN index into QW<number>, N<number>"""
            match = re.match(r"(QW\d+)(N\d+)", idx)
            qw, n = match.groups()
            return qw, n

        for t in tn.tensors:
            original_t_indices = t.indices
            new_t_indices = []
            for idx in original_t_indices:
                qw, _ = _index_splitter(idx)
                if idx in self.left_tn_indices:
                    new_t_indices.append(f"T{qw[2:]}")
                elif idx in self.right_tn_indices:
                    new_t_indices.append(f"V{qw[2:]}")
                else:
                    new_t_indices.append(idx)
            t.indices = new_t_indices
        r.set_default_indices("A", "V", "T")

        full_tn = TensorNetwork(tn.tensors + r.tensors)
        tr = full_tn.contract_entire_network()
        return tr

    def trace_tdagr(self) -> complex:
        """
        Calculate Tr(TR^dag) where R is the reference MPO and T is the TN
        """
        return self.trace_rdagt().conjugate()

    def trace_ttdag(self) -> complex:
        """
        Calculate Tr(TT^dag) where T is the TN
        """
        return 2**self.num_qubits

    def calculate_error(self) -> float:
        """
        Calculate the squared Frobenius norm between the reference MPO and the TN
        """
        error = (
            self.trace_rrdag()
            - self.trace_rdagt()
            - self.trace_tdagr()
            + self.trace_ttdag()
        )
        return max(error.real, 0.0)  # It will be real anyway

    def get_fidelity(self) -> float:
        """
        Get the fidelity
        """
        tn_mpo = MatrixProductOperator.from_qiskit_circuit(self.qc)
        fid = hilbert_schmidt_fidelity(self.reference, tn_mpo)
        return fid

    def build_trace_rdagt_tn(self) -> TensorNetwork:
        """
        Build the TN for Tr(R^dagT)
        """

        def _index_splitter(idx):
            """Split the TN index into QW<number>, N<number>"""
            match = re.match(r"(QW\d+)(N\d+)", idx)
            qw, n = match.groups()
            return qw, n

        r = copy.deepcopy(self.reference)
        r.dagger()
        tn = copy.deepcopy(self.tn)

        r.set_default_indices(internal_prefix="A", input_prefix="T", output_prefix="V")
        for t in tn.tensors:
            original_t_indices = t.indices
            new_t_indices = []
            for idx in original_t_indices:
                qw, _ = _index_splitter(idx)
                if idx in self.left_tn_indices:
                    new_t_indices.append(f"V{qw[2:]}")
                elif idx in self.right_tn_indices:
                    new_t_indices.append(f"T{qw[2:]}")
                else:
                    new_t_indices.append(idx)
            t.indices = new_t_indices

        tn = TensorNetwork(tn.tensors + r.tensors)
        return tn

    def get_environment_vector(self, variational_index: int) -> ndarray:
        """
        Get the environment vector at the given site

        Args:
            variational_index: The index of the current site

        Returns:
            The vector of the environment of the current site
        """
        tn = self.build_trace_rdagt_tn()
        site_label = f"variational_site_{variational_index}"
        popped_t = tn.pop_tensors_by_label([site_label])
        env_tensor = tn.contract_entire_network()
        env_copy = copy.deepcopy(env_tensor)
        output_inds = popped_t[0].indices
        env_copy.combine_indices(output_inds)
        env_vec = env_copy.data.todense()
        return env_vec

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

    def set_tn(self) -> None:
        """
        Reset TN after changes to qc
        """
        self.tn = TensorNetwork.from_qiskit_circuit(self.qc)
        for t in self.tn.tensors:
            t.labels.append(f"variational_site_{self.tn.tensors.index(t)+1}")

    def local_update(self, variational_index: int) -> None:
        """
        Perform a local update

        Args:
            variational_index: The index of the current site
        """
        local_tensor = self.tn.tensors[variational_index - 1]
        local_indices = local_tensor.indices
        local_dimensions = [
            local_tensor.get_dimension_of_index(idx) for idx in local_indices
        ]
        dim = np.prod(local_dimensions)

        env_vec = self.get_environment_vector(variational_index)
        update = env_vec
        update = update.reshape((int(np.sqrt(dim)), int(np.sqrt(dim))))
        unitary_update = self.get_closest_unitary(update)
        self.update_circuit(variational_index, unitary_update)
        self.set_tn()
        return

    def run(self, num_sweeps: int = 10) -> QuantumCircuit:
        """
        Optimise the ansatz to match the reference

        Args:
            num_sweeps: The number of sweeps to perform

        Returns:
            The optimised quantum circuit
        """
        for it_number in range(num_sweeps):
            for idx in range(1, len(self.qc.data) + 1):
                self.local_update(idx)
            for idx in list(range(1, len(self.qc.data) + 1))[::-1]:
                self.local_update(idx)
            self.error = self.calculate_error()
            self.fidelity = self.get_fidelity()
            self.optimisation_dict["optimisation_iteration"].append(it_number + 1)
            self.optimisation_dict["error"].append(self.error)
            self.optimisation_dict["fidelity"].append(self.fidelity)
        return self.qc
