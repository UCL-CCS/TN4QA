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
        env_vec = env_copy.to_dense()
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
            t.labels.append(f"variational_site_{self.tn.tensors.index(t) + 1}")
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
        self,
        tensor: Tensor,
        position: str | None = None,
        reverse_direction: bool = False,
    ) -> np.ndarray:
        """Constructs a unitary matrix from a given tensor"""
        data = copy.deepcopy(tensor)

        # Determine reshape based on position
        if position == "first":
            if not reverse_direction:
                data.reorder_indices([data.indices[1], data.indices[0]])
            matrix = data.to_dense().reshape((4, 1))
        elif position == "last":
            data.reorder_indices([data.indices[1], data.indices[0]])
            matrix = data.to_dense().reshape((2, 2))
        elif position == "middle":
            data.reorder_indices([data.indices[0], data.indices[2], data.indices[1]])
            matrix = data.to_dense().reshape((8, 1))
        else:
            if reverse_direction:
                data.reorder_indices(
                    [data.indices[0], data.indices[2], data.indices[1]]
                )
                matrix = data.to_dense().reshape((4, 2))
            else:
                data.reorder_indices(
                    [data.indices[2], data.indices[1], data.indices[0]]
                )
                matrix = data.to_dense().reshape((4, 2))

        shape = matrix.shape

        orthogonal_basis_1 = null_space(matrix)
        orthogonal_basis_2 = null_space(matrix.conj().T)

        if shape[0] > shape[1]:
            unitary = np.concatenate([matrix, orthogonal_basis_2], 1)
        elif shape[0] < shape[1]:
            unitary = np.concatenate([matrix.conj().T, orthogonal_basis_1], 1).conj().T
        else:
            unitary = matrix

        if reverse_direction and position is None:
            unitary[:, [1, 2]] = unitary[:, [2, 1]]

        before_uni = copy.deepcopy(unitary)
        unitary, _ = polar(unitary)
        after_uni = copy.deepcopy(unitary)
        assert np.allclose(before_uni - after_uni, 0.0)

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
                    mps.tensors[idx].to_dense()
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
                v = sub_mps.tensors[0].to_dense().reshape((2,))
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

            if sub_mps.num_sites == 2:
                vec = sub_mps.to_dense_array()
                u = np.zeros((4, 4), dtype=np.complex128)
                u[:, 0] = vec
                u[:, 1:] = np.random.randn(4, 3) + 1j * np.random.randn(4, 3)
                Q, _ = np.linalg.qr(u)
                phase = np.vdot(vec, Q[:, 0])
                Q[:, 0] *= phase / abs(phase)
                gate = UnitaryGate(Q)
                qc = QuantumCircuit(2)
                qc.append(gate, [1, 0])
                qcs.append(qc)
                if len(qidxs) == 0:
                    qidxs.append([0, 1])
                else:
                    qidxs.append([qidxs[-1][-1] + 1, qidxs[-1][-1] + 2])
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

    def bond_dim_2_to_qc_middle_out(
        self, bond_dim_2_mps: MatrixProductState
    ) -> QuantumCircuit:
        """Map a bond dimension 2 MPS to a quantum circuit
        Gates are applied from the middle and move outwards"""
        mps = bond_dim_2_mps
        n = mps.num_sites
        # Define middle site for even and odd n
        if n % 2 == 0:
            mps.move_orthogonality_centre(int(n // 2))  # even
        else:
            mps.move_orthogonality_centre(int(n // 2 + 1))  # odd

        # identify cuts where bond-dim == 1 (so we split the MPS into pieces)
        mps_dims = [mps.tensors[idx].dimensions[0] for idx in range(1, mps.num_sites)]
        bond_dim_1_idxs = (
            [0] + [i + 1 for i, x in enumerate(mps_dims) if x == 1] + [mps.num_sites]
        )

        # collect arrays per piece
        separate_mps_arrays = []
        for i in range(len(bond_dim_1_idxs) - 1):
            separate_mps_arrays.append(
                [
                    mps.tensors[idx].to_dense()
                    for idx in list(range(mps.num_sites))[
                        bond_dim_1_idxs[i] : bond_dim_1_idxs[i + 1]
                    ]
                ]
            )

        # build MatrixProductState objects for the pieces (preserve reshaping rules)
        separate_mps = []
        for arrays in separate_mps_arrays:
            if len(arrays) == 1:
                array = copy.deepcopy(arrays[0]).reshape((2,))
                separate_mps.append(MatrixProductState.from_arrays([array]))
                continue

            if len(arrays) == 2:
                first_array = copy.deepcopy(arrays[0]).reshape((2, 2))
                last_array = copy.deepcopy(arrays[1]).reshape((2, 2))
                separate_mps.append(
                    MatrixProductState.from_arrays([first_array, last_array])
                )
                continue

            reshaped_arrays = []
            first_array = copy.deepcopy(arrays[0])
            if first_array.ndim != 2:
                first_array = first_array.reshape(
                    (first_array.shape[1], first_array.shape[2])
                )
            reshaped_arrays.append(first_array)

            for a in arrays[1:-1]:
                reshaped_arrays.append(copy.deepcopy(a))

            last_array = copy.deepcopy(arrays[-1])
            if last_array.ndim != 2:
                last_array = last_array.reshape(
                    (last_array.shape[0], last_array.shape[2])
                )
            reshaped_arrays.append(last_array)

            separate_mps.append(MatrixProductState.from_arrays(reshaped_arrays))

        # For each piece, build a circuit with middle-out gates
        qcs = []
        qidxs = []
        for sub_mps in separate_mps:
            n = sub_mps.num_sites
            if n % 2 == 0:
                mps.move_orthogonality_centre(int(n // 2))
            else:
                mps.move_orthogonality_centre(int(n // 2 + 1))
            # single-site: just make a single-qubit unitary mapping |0> -> v
            if n == 1:
                v = sub_mps.tensors[0].to_dense().reshape((2,))
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
            elif sub_mps.num_sites < 5:
                qc = self.bond_dim_2_to_qc_exact(sub_mps)
                qcs.append(qc)
                if len(qidxs) == 0:
                    qidxs.append(list(range(sub_mps.num_sites)))
                else:
                    qidxs.append(
                        [qidxs[-1][-1] + x + 1 for x in range(sub_mps.num_sites)]
                    )
                continue

            # build the "unitaries" list
            sub_n = int(sub_mps.num_sites)
            unitaries_left = []
            unitaries_right = []

            # define midpoints
            mid_mid = int(sub_n // 2) if sub_n % 2 == 0 else int(sub_n // 2 + 1)
            mid_right = mid_mid + 1
            mid_left = mid_mid - 1

            # middle tensor
            middle_unitary = self.extend_to_unitary(
                sub_mps.tensors[mid_mid - 1], "middle"
            )

            # grow outward
            if n % 2 == 1:
                for offset in range(mid_mid - 2):
                    left_idx = mid_left - offset
                    right_idx = mid_right + offset

                    left_uni = self.extend_to_unitary(
                        sub_mps.tensors[left_idx - 1], reverse_direction=True
                    )
                    unitaries_left.append(left_uni)
                    right_uni = self.extend_to_unitary(sub_mps.tensors[right_idx - 1])
                    unitaries_right.append(right_uni)

            else:
                for offset in range(mid_mid - 2):
                    left_idx = mid_left - offset
                    right_idx = mid_right + offset

                    left_uni = self.extend_to_unitary(
                        sub_mps.tensors[left_idx - 1], reverse_direction=True
                    )
                    unitaries_left.append(left_uni)
                    right_uni = self.extend_to_unitary(sub_mps.tensors[right_idx - 1])
                    unitaries_right.append(right_uni)
                extra_right_uni = self.extend_to_unitary(sub_mps.tensors[-2])
                unitaries_right.append(extra_right_uni)

            # last unitaries at edges
            last_left = self.extend_to_unitary(sub_mps.tensors[0], "last")
            last_right = self.extend_to_unitary(sub_mps.tensors[-1], "last")
            unitaries_left.append(last_left)
            unitaries_right.append(last_right)

            # combine streams into a single circuit
            qc = QuantumCircuit(sub_n)
            if len(qidxs) == 0:
                qidxs.append(list(range(sub_n)))
            else:
                qidxs.append(list(range(qidxs[-1][-1] + 1, qidxs[-1][-1] + 1 + sub_n)))

            # Apply centre gate
            centre_gate = UnitaryGate(middle_unitary)
            qc.append(centre_gate, [mid_right - 1, mid_mid - 1, mid_left - 1])

            # Apply left stream gates
            for uni_idx in range(len(unitaries_left)):
                uni = unitaries_left[uni_idx]
                gate = UnitaryGate(uni)
                if uni.shape == (4, 4):
                    uni = uni[[0, 2, 1, 3], :]
                    gate = UnitaryGate(uni)
                    qc.append(gate, [mid_left - uni_idx - 2, mid_left - uni_idx - 1])
                elif uni.shape == (2, 2):
                    gate = UnitaryGate(uni)
                    qc.append(gate, [mid_left - uni_idx - 1])

            # Apply right stream gates
            for uni_idx in range(len(unitaries_right)):
                uni = unitaries_right[uni_idx]
                if uni.shape == (4, 4):
                    uni = uni[[0, 2, 1, 3], :]
                    gate = UnitaryGate(uni)
                    qc.append(gate, [mid_right - 1 + uni_idx, mid_right + uni_idx])
                elif uni.shape == (2, 2):
                    gate = UnitaryGate(uni)
                    qc.append(gate, [mid_right - 1 + uni_idx])

            qcs.append(qc)
        # bring together all of the little circuits into the final big circuit
        final_qc = QuantumCircuit(mps.num_sites)
        for qc_idx in range(len(qcs)):
            final_qc.compose(qcs[qc_idx], qidxs[qc_idx], inplace=True)

        return final_qc

    def mps_to_qc_via_ttn(self, mps: MatrixProductState, max_dim: int | None):
        circuit_description = circuit_structure_for_mps(
            mps.num_sites, mps.bond_dimension, max_dim
        )
        unitaries_list = build_unitaries(mps, max_dim)
        qc = QuantumCircuit(mps.num_sites)

        padded_N = 1
        while padded_N < mps.num_sites:
            padded_N = 2 * padded_N
        num_layers = int(np.log2(padded_N))
        layers = list(range(1, num_layers + 1))[::-1]

        for k in layers:
            ngates_in_layer, _ = num_gates_layer_k(k, mps.num_sites)
            gates = list(range(1, ngates_in_layer + 1))
            for pos in gates:
                u = unitaries_list[(k, pos)]
                u_gate = UnitaryGate(u)
                qubits = circuit_description[(k, pos)]
                qc.append(u_gate, qubits[::-1])
        return qc

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


def num_gates_layer_k(k: int, N: int) -> tuple[int, bool]:
    # Check k valid
    padded_N = 1
    while padded_N < N:
        padded_N = 2 * padded_N
    if k > int(np.log2(padded_N)):
        raise ValueError(f"layer {k} does not exist")
    if k == int(np.log2(padded_N)):
        return 1, False

    if k == 1:
        if N % 2 == 0:
            rounded_N = N
            carry_bit = False
        else:
            rounded_N = N - 1
            carry_bit = True
        return int(rounded_N / 2), carry_bit
    gates_in_previous_layer, carry_bit = num_gates_layer_k(k - 1, N)
    if carry_bit:
        nsites = gates_in_previous_layer + 1
    else:
        nsites = gates_in_previous_layer

    if nsites % 2 == 0:
        rounded_nsites = nsites
        new_carry_bit = False
    else:
        rounded_nsites = nsites - 1
        new_carry_bit = True

    return int(rounded_nsites / 2), new_carry_bit


def gate_to_children(k: int, N: int, pos: int):
    if k == 1:
        return None, None

    ngates_in_layer, carry_bit = num_gates_layer_k(k, N)
    if pos < ngates_in_layer and not carry_bit:
        return (k - 1, 2 * pos - 1), (k - 1, 2 * pos)
    if pos <= ngates_in_layer and carry_bit:
        return (k - 1, 2 * pos - 1), (k - 1, 2 * pos)

    _, carry_bit = num_gates_layer_k(k - 1, N)
    if not carry_bit:
        return (k - 1, 2 * pos - 1), (k - 1, 2 * pos)

    second_child_layer = k - 1
    while carry_bit and second_child_layer > 0:
        second_child_pos, carry_bit = num_gates_layer_k(second_child_layer, N)
        second_child_layer -= 1

    if second_child_layer == 0 and carry_bit:
        return (k - 1, 2 * pos - 1), None
    else:
        return (k - 1, 2 * pos - 1), (second_child_layer + 1, second_child_pos)


def max_svd_dim(k: int, N: int, chi: int, pos: int, max_dim: int | None = None):
    ngates_in_layer, carry_bit = num_gates_layer_k(k, N)
    if k == 1:
        if pos == 1:
            r = min(4, chi)
            if max_dim:
                return min(r, max_dim)
            else:
                return r
        if not carry_bit and pos == ngates_in_layer:
            r = min(4, chi)
            if max_dim:
                return min(r, max_dim)
            else:
                return r
        else:
            r = min(4, chi**2)
            if max_dim:
                return min(r, max_dim)
            else:
                return r

    if k == 2:
        if pos == 1:
            r = min(16, chi)
            if max_dim:
                return min(r, max_dim)
            else:
                return r
        if not carry_bit and pos == ngates_in_layer:
            _, previous_carry_bit = num_gates_layer_k(1, N)
            if previous_carry_bit:
                r = min(8, chi)
            else:
                r = min(16, chi)
            if max_dim:
                return min(r, max_dim)
            else:
                return r
        else:
            r = min(4, chi)
            s = min(r**2, chi**2)
            if max_dim:
                return min(s, max_dim)
            else:
                return s

    child1, child2 = gate_to_children(k, N, pos)
    if child2 is None:
        child_dim2 = 2
    else:
        child_dim2 = max_svd_dim(child2[0], N, chi, child2[1], max_dim)
    child_dim1 = max_svd_dim(child1[0], N, chi, child1[1], max_dim)
    mat_dim1 = child_dim1 * child_dim2

    if pos == 1:
        mat_dim2 = chi
    elif not carry_bit and pos == ngates_in_layer:
        mat_dim2 = chi
    else:
        mat_dim2 = chi**2

    if max_dim:
        return min(mat_dim1, mat_dim2, max_dim)
    else:
        return min(mat_dim1, mat_dim2)


def output_dim_separated(
    k: int, N: int, chi: int, pos: int, max_dim: int | None = None
):
    if k == 1:
        return 2, 2
    child1, child2 = gate_to_children(k, N, pos)
    if child2 is None:
        dim2 = 2
    else:
        dim2 = max_svd_dim(child2[0], N, chi, child2[1], max_dim)
    dim1 = max_svd_dim(child1[0], N, chi, child1[1], max_dim)
    return dim1, dim2


def output_dim(k: int, N: int, chi: int, pos: int, max_dim: int | None = None):
    dim1, dim2 = output_dim_separated(k, N, chi, pos, max_dim)
    return dim1 * dim2


def nqubits_for_gate(k: int, N: int, chi: int, pos: int, max_dim: int | None = None):
    dim = output_dim(k, N, chi, pos, max_dim)
    nqubits = int(np.log2(dim))
    return nqubits


def parent_to_children_map(N: int):
    padded_N = 1
    while padded_N < N:
        padded_N = 2 * padded_N
    num_layers = int(np.log2(padded_N))

    map = {}

    layers = list(range(1, num_layers + 1))
    for k in layers:
        ngates_in_layer, _ = num_gates_layer_k(k, N)
        gates = list(range(1, ngates_in_layer + 1))
        for g in gates:
            child1, child2 = gate_to_children(k, N, g)
            map[(k, g)] = [child1, child2]

    return map


def child_to_parent_map(N: int):
    parent_to_child = parent_to_children_map(N)
    map = {}
    for parent, children in parent_to_child.items():
        child1, child2 = children
        if child1:
            map[child1] = parent
        if child2:
            map[child2] = parent

    padded_N = 1
    while padded_N < N:
        padded_N = 2 * padded_N
    num_layers = int(np.log2(padded_N))

    map[(num_layers, 1)] = None
    return map


def num_extra_inputs(k: int, N: int, chi: int, pos: int, max_dim: int | None = None):
    nqubits = nqubits_for_gate(k, N, chi, pos, max_dim)
    c2p = child_to_parent_map(N)
    p2c = parent_to_children_map(N)
    parent = c2p[(k, pos)]
    child1, _ = p2c[parent]
    child_idx = 0 if (k, pos) == child1 else 1
    dim1, dim2 = output_dim_separated(parent[0], N, chi, parent[1], max_dim)
    incoming_dim = dim1 if child_idx == 0 else dim2
    incoming_qubits = int(np.log2(incoming_dim))
    return nqubits - incoming_qubits


def gate_to_qubits(k: int, N: int, chi: int, pos: int, max_dim: int | None = None):
    if k == 1:
        return [2 * pos - 1, 2 * pos]

    p2c = parent_to_children_map(N)

    child1, child2 = p2c[(k, pos)]
    child1_qubits = gate_to_qubits(child1[0], N, chi, child1[1], max_dim)
    child1_extra_qubits = num_extra_inputs(child1[0], N, chi, child1[1], max_dim)
    if child2 is None:
        pass
    else:
        child2_qubits = gate_to_qubits(child2[0], N, chi, child2[1], max_dim)
        child2_extra_qubits = num_extra_inputs(child2[0], N, chi, child2[1], max_dim)
    if child2 is None:
        qubits = child1_qubits[child1_extra_qubits:] + [N]
    else:
        qubits = (
            child1_qubits[child1_extra_qubits:] + child2_qubits[child2_extra_qubits:]
        )
    return qubits


def circuit_structure_for_mps(N: int, chi: int, max_dim: int | None = None):
    padded_N = 1
    while padded_N < N:
        padded_N = 2 * padded_N
    num_layers = int(np.log2(padded_N))

    circuit_description = {}
    layers = list(range(1, num_layers + 1))
    for k in layers:
        ngates_in_layer, _ = num_gates_layer_k(k, N)
        gates = list(range(1, ngates_in_layer + 1))
        for g in gates:
            qubits = gate_to_qubits(k, N, chi, g, max_dim)
            qubits = [q - 1 for q in qubits]
            circuit_description[(k, g)] = qubits
    return circuit_description


def max_gate_size(N: int, chi: int, max_dim: int | None = None):
    circuit_description = circuit_structure_for_mps(N, chi, max_dim)
    gate_sizes = [len(qubits) for qubits in circuit_description.values()]
    return max(gate_sizes)


def pad_bond_dim(mps: MatrixProductState):
    bond_dim = mps.bond_dimension
    padded_bond_dim = 1
    while padded_bond_dim < bond_dim:
        padded_bond_dim = 2 * padded_bond_dim

    for idx in range(1, mps.num_sites):
        bond_dim = mps.tensors[idx].dimensions[0]
        if bond_dim < padded_bond_dim:
            mps = mps.expand_bond_dimension(padded_bond_dim - bond_dim, idx)

    return mps


def find_best_orthogonality_centre(mps: MatrixProductState, max_dim: int | None = None):
    mps = pad_bond_dim(mps)

    padded_N = 1
    while padded_N < mps.num_sites:
        padded_N = 2 * padded_N
    num_layers = int(np.log2(padded_N))

    circuit_description = circuit_structure_for_mps(
        mps.num_sites, mps.bond_dimension, max_dim
    )
    last_gate_qubits = circuit_description[(num_layers, 1)]
    return last_gate_qubits[0]


def build_unitaries(mps: MatrixProductState, max_dim: int | None = None):
    mps = pad_bond_dim(mps)
    n = mps.num_sites

    padded_N = 1
    while padded_N < mps.num_sites:
        padded_N = 2 * padded_N
    num_layers = int(np.log2(padded_N))

    all_unitaries = {}
    current_layer_n = n
    current_layer_tensors = mps.tensors
    for k in list(range(1, num_layers)):
        next_layer_tensors = []
        i = 0
        while i < current_layer_n:
            if i + 1 == current_layer_n:
                next_layer_tensors.append(current_layer_tensors[i])
                i += 1
            else:
                tensor0, tensor1 = (
                    current_layer_tensors[i],
                    current_layer_tensors[i + 1],
                )
                if len(tensor0.indices) == 2:
                    contraction = "ab,acd->bdc"
                    new_t_inds = ["down", f"p{i+1}"]
                    boundary = True
                elif len(tensor1.indices) == 2:
                    contraction = "abc,bd->cda"
                    new_t_inds = ["up", f"p{i+1}"]
                    boundary = True
                else:
                    contraction = "abc,bde->cead"
                    new_t_inds = ["up", "down", f"p{i+1}"]
                    boundary = False
                combined_tensor_data = np.einsum(
                    contraction, tensor0.to_dense(), tensor1.to_dense()
                )
                shape = combined_tensor_data.shape
                if boundary:
                    matrix = np.reshape(
                        combined_tensor_data, (shape[0] * shape[1], shape[2])
                    )
                else:
                    matrix = np.reshape(
                        combined_tensor_data,
                        (shape[0] * shape[1], shape[2] * shape[3]),
                    )

                u, s, vh = svd(matrix, full_matrices=False)

                if max_dim:
                    u = u[:, :max_dim]
                    s = s[:max_dim]
                    vh = vh[:max_dim, :]

                d1, d2 = u.shape[0], u.shape[0] - u.shape[1]
                Q, _ = np.linalg.qr(
                    np.random.randn(d1, d2) + 1j * np.random.randn(d1, d2)
                )
                Q = Q - u @ (u.conj().T @ Q)
                Q, _ = np.linalg.qr(Q)
                unitary = np.hstack([u, Q])
                gate_idx = int(i / 2 + 1)
                all_unitaries[(k, gate_idx)] = unitary

                next_data = np.diag(s) @ vh
                if boundary:
                    next_data = np.moveaxis(next_data, 0, -1)
                else:
                    first_dim = vh.shape[0]
                    next_data = np.reshape(
                        next_data, (first_dim, mps.bond_dimension, mps.bond_dimension)
                    )
                    next_data = np.moveaxis(next_data, 0, -1)
                next_tensor = Tensor(next_data, new_t_inds, ["TEMP"])
                next_layer_tensors.append(next_tensor)
                i += 2
        current_layer_n = len(next_layer_tensors)
        current_layer_tensors = next_layer_tensors

    last_contraction = "ab,ac->bc"
    last_data = np.einsum(
        last_contraction,
        next_layer_tensors[0].to_dense(),
        next_layer_tensors[1].to_dense(),
    )
    size = last_data.shape[0] * last_data.shape[1]
    vec = np.reshape(last_data, (size,))
    vec = vec / np.linalg.norm(vec)
    X = np.random.randn(size, size - 1) + 1j * np.random.randn(size, size - 1)
    X = X - vec[:, None] * (vec.conj() @ X)
    Q2, _ = np.linalg.qr(X)
    Q = np.column_stack([vec, Q2])
    all_unitaries[(num_layers, 1)] = Q
    return all_unitaries
