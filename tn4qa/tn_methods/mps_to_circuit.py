import copy
import re

import numpy as np
from numpy import ndarray
from numpy.linalg import svd
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import null_space, polar

from ..circuit_simulator import CircuitSimulator
from ..fidelity_metrics import state_uhlmann_fidelity
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


class MPSAnalyticDecomposition:
    """A class to analytically decompose MPS as quantum circuits"""

    def __init__(
        self, mps: MatrixProductState, max_layers: int, target_fidelity: float
    ):
        """Constructor

        Args:
            mps: The MPS to map to a quantum circuit
            max_layers: The maximum number of allowed staircase circuit layers
            target_fidelity: The target fidelity between the quantum circuit and the MPS
        """
        self.mps = mps
        self.num_sites = self.mps.num_sites
        self.max_layers = max_layers
        self.target_fidelity = target_fidelity
        self.qc = QuantumCircuit(mps.num_sites)
        self.num_layers = 0
        self.fidelity = 0.0

    def compress_to_bond_dim_2(self, mps: MatrixProductState) -> MatrixProductState:
        """Compress the current mps to bond dimension 2"""
        mps_copy = copy.deepcopy(mps)
        mps_copy.compress(2)
        mps_copy.normalise()
        return mps_copy

    def extend_to_unitary(
        self, tensor: Tensor, position: str | None = None
    ) -> np.ndarray:
        """Constructs a unitary matrix from a given tensor"""
        data = copy.deepcopy(tensor)

        # Determine reshape based on position
        if position == "first":
            data.reorder_indices([data.indices[1], data.indices[0]])
            matrix = data.data.todense().reshape((4, 1))
        elif position == "last":
            data.reorder_indices([data.indices[1], data.indices[0]])
            matrix = data.data.todense().reshape((2, 2))
        else:
            data.tensor_to_matrix(
                [tensor.indices[0]], [tensor.indices[2], tensor.indices[1]]
            )
            matrix = data.data.todense()

        shape = matrix.shape

        orthogonal_basis_1 = null_space(matrix)
        orthogonal_basis_2 = null_space(matrix.conj().T)

        if shape[0] > shape[1]:
            unitary = np.concatenate([matrix, orthogonal_basis_2], 1)
        elif shape[0] < shape[1]:
            unitary = np.concatenate([matrix.conj().T, orthogonal_basis_1], 1).conj().T
        else:
            unitary = matrix

        unitary, _ = polar(unitary)
        return unitary

    def bond_dim_2_to_qc_exact(
        self, bond_dim_2_mps: MatrixProductState
    ) -> QuantumCircuit:
        """Map a bond dimension 2 MPS to a quantum circuit exactly"""

        mps = bond_dim_2_mps
        mps.move_orthogonality_centre(1)

        mps_dims = [mps.tensors[idx].dimensions[0] for idx in range(1, mps.num_sites)]
        bond_dim_1_idxs = (
            [0] + [i + 1 for i, x in enumerate(mps_dims) if x == 1] + [mps.num_sites]
        )
        separate_mps_arrays = []
        for i in range(len(bond_dim_1_idxs) - 1):
            separate_mps_arrays.append(
                [
                    mps.tensors[idx].data.todense()
                    for idx in list(range(mps.num_sites))[
                        bond_dim_1_idxs[i] : bond_dim_1_idxs[i + 1]
                    ]
                ]
            )

        separate_mps = []

        for arrays in separate_mps_arrays:
            if len(arrays) == 1:
                array = copy.deepcopy(arrays[0])
                array = array.reshape((2,))
                separate_mps.append(MatrixProductState.from_arrays([array]))
                continue
            elif len(arrays) == 2:
                first_array = copy.deepcopy(arrays[0])
                first_array = first_array.reshape((2, 2))
                last_array = copy.deepcopy(arrays[1])
                last_array = last_array.reshape((2, 2))
                separate_mps.append(
                    MatrixProductState.from_arrays([first_array, last_array])
                )
                continue

            reshaped_arrays = []
            first_array = copy.deepcopy(arrays[0])
            if first_array.ndim == 2:
                pass
            else:
                first_array = first_array.reshape(
                    (first_array.shape[1], first_array.shape[2])
                )
            reshaped_arrays.append(first_array)
            for i in range(1, len(arrays) - 1):
                array = copy.deepcopy(arrays[i])
                reshaped_arrays.append(array)
            last_array = copy.deepcopy(arrays[-1])
            if last_array.ndim == 2:
                pass
            else:
                last_array = last_array.reshape(
                    (last_array.shape[0], last_array.shape[2])
                )
            reshaped_arrays.append(last_array)
            separate_mps.append(MatrixProductState.from_arrays(reshaped_arrays))

        qcs = []
        qidxs = []
        for sub_mps in separate_mps:
            if sub_mps.num_sites == 1:
                v = sub_mps.tensors[0].data.todense().reshape((2,))
                a, b = v
                v_perp = np.array([-np.conj(b), np.conj(a)])
                unitary = np.column_stack((v, v_perp))
                gate = UnitaryGate(unitary)
                qc = QuantumCircuit(1)
                qc.append(gate, [0])
                qcs.append(qc)
                if len(qidxs) == 0:
                    qidxs.append([0])
                else:
                    qidxs.append([qidxs[-1][-1] + 1])
                continue

            unitaries = []
            first_uni = self.extend_to_unitary(sub_mps.tensors[0], "first")
            unitaries.append(first_uni)
            for tidx in range(1, sub_mps.num_sites - 1):
                t = sub_mps.tensors[tidx]
                uni = self.extend_to_unitary(t)
                unitaries.append(uni)
            final_uni = self.extend_to_unitary(sub_mps.tensors[-1], "last")
            unitaries.append(final_uni)

            qc = QuantumCircuit(sub_mps.num_sites)
            if len(qidxs) == 0:
                qidxs.append(list(range(sub_mps.num_sites)))
            else:
                qidxs.append(
                    list(
                        range(qidxs[-1][-1] + 1, qidxs[-1][-1] + 1 + sub_mps.num_sites)
                    )
                )
            for uni_idx in range(sub_mps.num_sites - 2):
                uni = unitaries[uni_idx]
                uni = uni[[0, 2, 1, 3], :]
                gate = UnitaryGate(uni)
                qc.append(gate, [uni_idx, uni_idx + 1])
            penultimate_uni = unitaries[-2]
            penultimate_uni = penultimate_uni[[0, 2, 1, 3], :]
            final_uni = unitaries[-1]
            final_uni_2q = np.kron(np.eye(2), final_uni)
            final_uni_2q = final_uni_2q[[0, 2, 1, 3], :]
            final_uni_2q = final_uni_2q[:, [0, 2, 1, 3]]
            total_uni = final_uni_2q @ penultimate_uni
            last_gate = UnitaryGate(total_uni)
            qc.append(last_gate, [sub_mps.num_sites - 2, sub_mps.num_sites - 1])
            qcs.append(qc)

        final_qc = QuantumCircuit(mps.num_sites)
        for qc_idx in range(len(qcs)):
            final_qc.compose(qcs[qc_idx], qidxs[qc_idx], inplace=True)

        return final_qc

    def disentangle_mps(
        self, mps: MatrixProductState, qc_layer: QuantumCircuit
    ) -> MatrixProductState:
        """Update the current MPS by diesntangling with a circuit layer"""
        sim = CircuitSimulator(qc_layer.inverse(), mps)
        out = sim.run()
        return out

    def calculate_fidelity(self, circ) -> float:
        """Calculate current fidelity"""
        state = copy.deepcopy(self.mps)
        sim = CircuitSimulator(circ)
        output = sim.run()
        fid = state_uhlmann_fidelity(output, state)
        return fid

    def run(self) -> QuantumCircuit:
        """Run the analytic decomposition"""
        while (
            self.num_layers < self.max_layers and self.fidelity < self.target_fidelity
        ):
            original_mps = copy.deepcopy(self.mps)
            disentangled_mps = self.disentangle_mps(original_mps, self.qc)
            bond_dim_2_mps = self.compress_to_bond_dim_2(disentangled_mps)
            qc_layer = self.bond_dim_2_to_qc_exact(bond_dim_2_mps)
            temp_circ = self.qc.compose(qc_layer, front=True)
            new_fidelity = self.calculate_fidelity(temp_circ)
            # if new_fidelity < self.fidelity:
            #     break
            self.qc = temp_circ
            self.fidelity = new_fidelity
            self.num_layers += 1
        return self.qc


class MPStoCircuit:
    def __init__(
        self, mps: MatrixProductState, max_layers: int, target_fidelity: float
    ):
        self.mps = mps
        self.max_layers = max_layers
        self.target_fidelity = target_fidelity
        self.num_layers = 0
        self.fidelity = 0.0
        self.qc = QuantumCircuit(mps.num_sites)
        self.current_mps = copy.deepcopy(mps)

    def calculate_fidelity(self, circ) -> float:
        """Calculate current fidelity"""
        state = copy.deepcopy(self.mps)
        sim = CircuitSimulator(circ)
        output = sim.run()
        fid = state_uhlmann_fidelity(output, state)
        return fid

    def disentangle_mps(self, qc_layer: QuantumCircuit) -> None:
        """Update the current MPS by diesntangling with a circuit layer"""
        current_mps = copy.deepcopy(self.current_mps)
        sim = CircuitSimulator(qc_layer.inverse(), current_mps)
        self.current_mps = sim.run()
        return

    def run(self, num_optimiser_sweeps: int = 1) -> QuantumCircuit:
        while (
            self.num_layers < self.max_layers and self.fidelity < self.target_fidelity
        ):
            qc_layer = MPSAnalyticDecomposition(self.current_mps, 1, 1.0).run()
            # qc_layer = MPSOptimiser(qc_layer, self.current_mps).run(num_optimiser_sweeps)
            temp_circ = self.qc.compose(qc_layer, front=True)
            new_fidelity = self.calculate_fidelity(temp_circ)
            if new_fidelity < self.fidelity:
                break
            self.qc = temp_circ
            self.disentangle_mps(qc_layer)
            self.fidelity = new_fidelity
            self.num_layers += 1
        # self.qc = MPSOptimiser(self.qc, self.mps).run(num_optimiser_sweeps)
        return self.qc
