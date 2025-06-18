import copy
import re

import numpy as np
from numpy import ndarray
from numpy.linalg import svd
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from ..mps import MatrixProductState
from ..tensor import Tensor
from ..tn import TensorNetwork


class MPSOptimiser:
    """
    A class for locally optimising a quantum circuit with respect to a reference MPS and the HS distance
    """

    def __init__(self, qc: QuantumCircuit, reference: MatrixProductState) -> None:
        """
        Constructor

        Args:
            qc: The quantum circuit that will be optimised
            reference: The reference MPS
        """
        self.qc = qc
        self.reference = reference
        self.num_qubits = qc.num_qubits
        self.set_tn()
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

    def apply_initial_state(self):
        """
        Apply the all zero initial state to tn to form a state
        """
        for idx in self.left_tn_indices:
            zero_data = np.array([1, 0], dtype=complex).reshape((2,))
            zero_tensor = Tensor(zero_data, [idx], ["Zero"])
            self.tn.add_tensor(zero_tensor)
        return

    def ip_rr(self) -> complex:
        """
        Calculate <R|R> where R is the reference MPS
        """
        return 1.0 + 0.0j

    def ip_tr(self) -> complex:
        """
        Calculate <T|R> where R is the reference MPS and T is the TN
        """
        return self.ip_rt().conjugate()

    def ip_rt(self) -> complex:
        """
        Calculate <R|T> where R is the reference MPS and T is the TN
        """
        tn = self.build_ip_rt_tn()
        ip = tn.contract_entire_network()
        return ip

    def ip_tt(self) -> complex:
        """
        Calculate <T|T> where T is the TN
        """
        return 1.0 + 0.0j

    def calculate_error(self) -> float:
        """
        Calculate the squared Frobenius norm between the reference MPS and the TN
        """
        err = self.ip_rr() - self.ip_rt() - self.ip_tr() + self.ip_tt()
        return max(err.real, 0.0)  # It will be real anyway

    def get_fidelity(self) -> float:
        """
        Get the fidelity
        """
        overlap = self.ip_tr()
        fid = np.abs(overlap) ** 2
        return fid

    def build_ip_rt_tn(self) -> TensorNetwork:
        def _index_splitter(idx):
            """Split the TN index into QW<number>, N<number>"""
            match = re.match(r"(QW\d+)(N\d+)", idx)
            qw, n = match.groups()
            return qw, n

        r = copy.deepcopy(self.reference)
        r.dagger()
        r.set_default_indices("A", "T")
        tn = copy.deepcopy(self.tn)
        for t in tn.tensors:
            original_t_indices = t.indices
            new_t_indices = []
            for idx in original_t_indices:
                qw, _ = _index_splitter(idx)
                if idx in self.right_tn_indices:
                    new_t_indices.append(f"T{qw[2:]}")
                else:
                    new_t_indices.append(idx)
            t.indices = new_t_indices

        full_tn = TensorNetwork(r.tensors + tn.tensors)
        return full_tn

    def get_environment_vector(self, variational_index: int) -> ndarray:
        tn = self.build_ip_rt_tn()
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
        self.num_variational_sites = len(self.tn.tensors)
        self.left_tn_indices, self.right_tn_indices = self.get_tn_external_indices(
            self.tn
        )
        self.apply_initial_state()
        return

    def local_update(self, variational_index: int) -> None:
        """
        Perform a local update

        Args:
            variational_index: The index of the current site
        """
        site_index = f"variational_site_{variational_index}"
        local_tensor = self.tn.get_tensors_from_label(site_index)[0]
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
