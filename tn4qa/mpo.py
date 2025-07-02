import copy
from itertools import islice
from typing import List, TypeAlias, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import sparse
from numpy import ndarray
from numpy.linalg import svd

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from scipy.sparse.linalg import svds
from sparse import SparseArray

from .tensor import Tensor
from .tn import TensorNetwork
from .utils import _update_array, _update_array_fermion

# Visualisation
from .visualisation import draw_mpo

DataOptions: TypeAlias = Union[ndarray, SparseArray]


class MatrixProductOperator(TensorNetwork):
    def __init__(self, tensors: List[Tensor], shape: str = "udrl") -> None:
        """
        Constructor for MatrixProductOperator class.

        Args:
            tensors: List of tensors to form the MPO.
            shape (optional): The order of the indices for the tensors. Default is 'udrl' (up, down, right, left).

        Returns
            An MPO.
        """
        if len(tensors) == 1:
            self.name = "MPO"
            self.tensors = tensors
            self.indices = tensors[0].indices
            self.num_sites = 1
            self.shape = shape
            self.internal_inds = []
            self.external_inds = tensors[0].indices
            self.bond_dims = []
            self.physical_dims = [tensors[0].dimensions[0], tensors[0].dimensions[1]]
            self.bond_dimension = None
            self.physical_dimension = self.physical_dims[0]
        else:
            super().__init__(tensors, "MPO")
            self.num_sites = len(tensors)
            self.shape = shape

            self.internal_inds = self.get_internal_indices()
            self.external_inds = self.get_external_indices()
            self.bond_dims = []
            self.physical_dims = []
            for idx in self.internal_inds:
                self.bond_dims.append(self.get_dimension_of_index(idx))
            for idx in self.external_inds:
                self.physical_dims.append(self.get_dimension_of_index(idx))
            self.bond_dimension = max(self.bond_dims)
            self.physical_dimension = max(self.physical_dims)

    @classmethod
    def from_arrays(
        cls, arrays: List[DataOptions], shape: str = "udrl"
    ) -> "MatrixProductOperator":
        """
        Create an MPO from a list of arrays.

        Args:
            arrays: The list of arrays.
            shape (optional): The order of the indices for the tensors. Default is 'udrl' (up, down, right, left).

        Returns:
            An MPO.
        """
        if len(arrays) == 1:
            idxs = ["R1", "L1"]
            tensor = Tensor(arrays[0], idxs, ["MPO_T1"])
            return cls([tensor], shape)

        tensors = []

        first_shape = shape.replace("u", "")
        right_idx_pos = first_shape.index("r")
        left_idx_pos = first_shape.index("l")
        down_idx_pos = first_shape.index("d")
        first_indices = ["", "", ""]
        first_indices[right_idx_pos] = "R1"
        first_indices[left_idx_pos] = "L1"
        first_indices[down_idx_pos] = "B1"
        first_tensor = Tensor(arrays[0], first_indices, ["MPO_T1"])
        tensors.append(first_tensor)

        right_idx_pos = shape.index("r")
        left_idx_pos = shape.index("l")
        down_idx_pos = shape.index("d")
        up_idx_pos = shape.index("u")
        for a_idx in range(1, len(arrays) - 1):
            a = arrays[a_idx]
            indices_k = ["", "", "", ""]
            indices_k[right_idx_pos] = f"R{a_idx+1}"
            indices_k[left_idx_pos] = f"L{a_idx+1}"
            indices_k[up_idx_pos] = f"B{a_idx}"
            indices_k[down_idx_pos] = f"B{a_idx+1}"
            tensor_k = Tensor(a, indices_k, [f"MPO_T{a_idx+1}"])
            tensors.append(tensor_k)

        last_shape = shape.replace("d", "")
        right_idx_pos = last_shape.index("r")
        left_idx_pos = last_shape.index("l")
        up_idx_pos = last_shape.index("u")
        last_indices = ["", "", ""]
        last_indices[right_idx_pos] = f"R{len(arrays)}"
        last_indices[left_idx_pos] = f"L{len(arrays)}"
        last_indices[up_idx_pos] = f"B{len(arrays)-1}"
        last_tensor = Tensor(arrays[-1], last_indices, [f"MPO_T{len(arrays)}"])
        tensors.append(last_tensor)

        mpo = cls(tensors, shape)
        mpo.reshape()
        return mpo

    @classmethod
    def identity_mpo(cls, num_sites: int) -> "MatrixProductOperator":
        """
        Create an MPO for the identity operation.

        Args:
            num_sites: The number of sites for the MPO.

        Returns:
            An MPO.
        """
        if num_sites == 1:
            arrays = [np.array([[1, 0], [0, 1]]).reshape(2, 2)]
            mpo = cls.from_arrays(arrays)
            return mpo
        end_array = np.array([[1, 0], [0, 1]]).reshape(1, 2, 2)
        middle_arrays = np.array([[1, 0], [0, 1]]).reshape(1, 1, 2, 2)
        arrays = [end_array] + [middle_arrays] * (num_sites - 2) + [end_array]
        mpo = cls.from_arrays(arrays)
        return mpo

    @classmethod
    def generalised_mcu_mpo(
        cls,
        num_sites: int,
        zero_ctrls: List[int],
        one_ctrls: List[int],
        target: int,
        unitary: DataOptions,
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a generalised MCU operation.

        Args:
            num_sites: The number of sites for the MPO.
            zero_ctrls: The sites with a zero control.
            one_ctrls: The sites with a one control.
            target: The target site.
            unitary: The U gate to apply.

        Returns:
            An MPO.
        """
        unitary = unitary.todense() if isinstance(unitary, SparseArray) else unitary
        unitary_gate = UnitaryGate(unitary)

        first_mcu_qubit = min(zero_ctrls + one_ctrls + [target])
        last_mcu_qubit = max(zero_ctrls + one_ctrls + [target])
        mcu_qubits = list(range(first_mcu_qubit, last_mcu_qubit + 1))

        tensors = []

        for qidx in range(1, first_mcu_qubit):
            if qidx == 1:
                first_indices = ["B1", "R1", "L1"]
                first_labels = ["MPO_T1"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 2, 2),
                    first_indices,
                    first_labels,
                )
                tensors.append(tensor)
            else:
                indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 1, 2, 2),
                    indices,
                    labels,
                )
                tensors.append(tensor)

        for qidx in mcu_qubits:
            if qidx == 1 or qidx == num_sites:
                indices = (
                    [f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                    if qidx == 1
                    else [f"B{qidx-1}", f"R{qidx}", f"L{qidx}"]
                )
                labels = [f"MPO_T{qidx}"]
                if qidx in zero_ctrls:
                    tensor = Tensor.rank_3_copy_open(indices, labels)
                elif qidx in one_ctrls:
                    tensor = Tensor.rank_3_copy(indices, labels)
                else:
                    tensor = Tensor.rank_3_qiskit_gate(unitary_gate, indices, labels)
                tensors.append(tensor)

            elif qidx == first_mcu_qubit:
                labels = [f"MPO_T{qidx}"]
                if qidx in zero_ctrls:
                    tensor = Tensor.rank_3_copy_open(labels=labels)
                elif qidx in one_ctrls:
                    tensor = Tensor.rank_3_copy(indices, labels)
                else:
                    tensor = Tensor.rank_3_qiskit_gate(unitary_gate, indices, labels)
                tensor.data = sparse.reshape(tensor.data, (1,) + tensor.dimensions)
                tensor.dimensions = (1,) + tensor.dimensions
                tensor.indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                tensor.rank = 4
                tensors.append(tensor)

            elif qidx == last_mcu_qubit:
                labels = [f"MPO_T{qidx}"]
                if qidx in zero_ctrls:
                    tensor = Tensor.rank_3_copy_open(labels=labels)
                elif qidx in one_ctrls:
                    tensor = Tensor.rank_3_copy(indices, labels)
                else:
                    tensor = Tensor.rank_3_qiskit_gate(unitary_gate, indices, labels)
                tensor.data = sparse.reshape(
                    tensor.data,
                    (tensor.dimensions[0],)
                    + (1,)
                    + (tensor.dimensions[1], tensor.dimensions[2]),
                )
                tensor.dimensions = (
                    (tensor.dimensions[0],)
                    + (1,)
                    + (tensor.dimensions[1], tensor.dimensions[2])
                )
                tensor.indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                tensor.rank = 4
                tensors.append(tensor)

            else:
                indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                if qidx in zero_ctrls:
                    tensor = Tensor.rank_4_copy_open(indices, labels)
                elif qidx in one_ctrls:
                    tensor = Tensor.rank_4_copy(indices, labels)
                elif qidx == target:
                    tensor = Tensor.rank_4_qiskit_gate(unitary_gate, indices, labels)
                else:
                    tensor = Tensor.from_array(
                        np.eye(4).reshape(2, 2, 2, 2), indices, labels
                    )
                tensors.append(tensor)

        for qidx in range(last_mcu_qubit + 1, num_sites + 1):
            if qidx == num_sites:
                last_indices = [f"B{num_sites-1}", f"R{num_sites}", f"L{num_sites}"]
                last_labels = [f"MPO_T{num_sites}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 2, 2),
                    last_indices,
                    last_labels,
                )
                tensors.append(tensor)
            else:
                indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 1, 2, 2),
                    indices,
                    labels,
                )
                tensors.append(tensor)

        mpo = cls(tensors)
        return mpo

    @classmethod
    def from_pauli_string(cls, ps: str) -> "MatrixProductOperator":
        """
        Create an MPO for a single Pauli string.

        Args:
            ps: The Pauli string.

        Returns:
            An MPO.
        """
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        pauli_id = np.array([[1, 0], [0, 1]], dtype=complex)
        pauli_dict = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z, "I": pauli_id}

        tensors = []

        if len(ps) == 1:
            indices = ["R1", "L1"]
            label = ["MPO_T!"]
            gate = pauli_dict[ps[0]]
            tensor = Tensor(gate, indices, label)
            tensors.append(tensor)
            mpo = cls(tensors)
            return mpo

        first_indices = ["B1", "R1", "L1"]
        first_labels = ["MPO_T1"]
        first_gate = pauli_dict[ps[0]].reshape(1, 2, 2)
        first_tensor = Tensor(first_gate, first_indices, first_labels)
        tensors.append(first_tensor)

        num_sites = len(ps)
        for qidx in range(2, num_sites):
            qidx_indices = [f"B{qidx-1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
            qidx_labels = [f"MPO_T{qidx}"]
            qidx_gate = pauli_dict[ps[qidx - 1]].reshape(1, 1, 2, 2)
            qidx_tensor = Tensor(qidx_gate, qidx_indices, qidx_labels)
            tensors.append(qidx_tensor)

        last_indices = [f"B{num_sites-1}", f"R{num_sites}", f"L{num_sites}"]
        last_labels = [f"MPO_T{num_sites}"]
        last_gate = pauli_dict[ps[-1]].reshape(1, 2, 2)
        last_tensor = Tensor(last_gate, last_indices, last_labels)
        tensors.append(last_tensor)

        mpo = cls(tensors)
        return mpo

    @classmethod
    def from_hamiltonian_adder(
        cls, ham: dict[str, complex], max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a Hamiltonian.

        Args:
            ham: The dict representation of the Hamiltonian {pauli_string : weight}.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        pauli_strings = list(ham.keys())
        first_ps = pauli_strings[0]
        mpo = cls.from_pauli_string(first_ps)
        mpo.multiply_by_constant(ham[first_ps])

        for ps in pauli_strings[1:]:
            temp_mpo = cls.from_pauli_string(ps)
            temp_mpo.multiply_by_constant(ham[ps])
            mpo = mpo + temp_mpo
            if max_bond:
                if mpo.bond_dimension > max_bond:
                    mpo.compress(max_bond)

        return mpo

    @classmethod
    def from_hamiltonian(
        cls,
        ham_dict: dict[str, complex],
        max_bond: int | None = None,
        batch: bool = False,
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a Hamiltonian.

        Args:
            ham: The dict representation of the Hamiltonian {pauli_string : weight}.
            max_bond: The maximum bond dimension allowed.
            batch: If True, batches the items in the Hamiltonian

        Returns:
            An MPO.
        """
        num_qubits = len(list(ham_dict.keys())[0])
        num_ham_terms = len(ham_dict.keys())

        if batch:
            if num_ham_terms / 2 > max_bond:
                first_batch = dict(
                    islice(ham_dict.items(), int(np.floor(max_bond / 2)))
                )
                mpo = cls.from_hamiltonian(first_batch)
                used = int(np.floor(max_bond / 2))
                while used < num_ham_terms:
                    batch = dict(
                        islice(
                            ham_dict.items(), used, used + int(np.floor(max_bond / 2))
                        )
                    )
                    temp_mpo = cls.from_hamiltonian(batch)
                    mpo = mpo + temp_mpo
                    if mpo.bond_dimension > max_bond:
                        mpo.compress(max_bond)
                    used += int(np.floor(max_bond / 2))
                return mpo

        first_array_coords: list[list[int]] = [[], [], []]
        middle_array_coords: list[list[list[int]]] = [
            [[], [], [], []] for _ in range(1, num_qubits - 1)
        ]
        last_array_coords: list[list[int]] = [[], [], []]
        first_array_data: list[complex] = []
        middle_array_data: list[list[complex]] = [[] for _ in range(1, num_qubits - 1)]
        last_array_data: list[complex] = []

        for p_string_idx, (p_string, weight) in enumerate(ham_dict.items()):
            # First Term
            _update_array(
                first_array_coords, first_array_data, weight, p_string_idx, p_string[0]
            )

            # Middle Terms
            for p_idx in range(1, num_qubits - 1):
                p = p_string[p_idx]
                _update_array(
                    middle_array_coords[p_idx - 1],
                    middle_array_data[p_idx - 1],
                    1,
                    p_string_idx,
                    p,
                    offset=True,
                )

            # Final Term
            _update_array(
                last_array_coords, last_array_data, 1, p_string_idx, p_string[-1]
            )

        first_array = sparse.COO(
            first_array_coords, first_array_data, shape=(num_ham_terms, 2, 2)
        )
        middle_arrays = [
            sparse.COO(
                middle_array_coords[i - 1],
                middle_array_data[i - 1],
                shape=(num_ham_terms, num_ham_terms, 2, 2),
            )
            for i in range(1, num_qubits - 1)
        ]
        last_array = sparse.COO(
            last_array_coords, last_array_data, shape=(num_ham_terms, 2, 2)
        )

        mpo = MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array]
        )
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        return mpo

    @classmethod
    def from_hamiltonian_approx(
        cls,
        ham_dict: dict[str, complex],
        max_bond: int | None = None,
        threshold: float = 1e-4,
    ) -> "MatrixProductOperator":
        """
        Create an approximate MPO representation of the Hamiltonian by discarding strings with small weights

        Args:
            ham_dict: The Hamiltonian
            max_bond: Maximum bond dimension
            threshold: Sets the cutoff parameter for which strings to keep

        Returns:
            An MPO
        """
        ham_norm = np.sum([np.abs(w) for w in list(ham_dict.values())])
        cutoff = ham_norm * threshold
        ham = {k: v for k, v in ham_dict.items() if np.abs(v) > cutoff}
        mpo = cls.from_hamiltonian(ham, max_bond)
        return mpo

    def apply_one_qubit_gate(self, data: SparseArray, site: int) -> None:
        """
        Apply a one-qubit gate in place

        Args:
            data: The one-qubit matrix
            site: Where to apply the gate to
        """
        if self.num_sites == 1:
            contraction = "ij,ki->kj"
        elif site == 1 or site == self.num_sites:
            contraction = "ijk,lj->ilk"
        else:
            contraction = "hijk,lj->hilk"
        self.tensors[site - 1].data = sparse.einsum(
            contraction, self.tensors[site - 1].data, data
        )
        return

    def move_site_to_location(
        self, site_source: int, site_destination: int, site_mapping: dict
    ) -> None:
        """
        Move a site to a different location

        Args:
            site_source: The starting location
            site_destination: The final location
            site_mapping: A mapping of physical to logical sites
        """
        if site_source < site_destination:
            for site in range(site_source, site_destination):
                self.swap_neighbouring_sites(site)
                reverse_map = {v: k for k, v in site_mapping.items()}
                (
                    site_mapping[reverse_map[str(site)]],
                    site_mapping[reverse_map[str(site + 1)]],
                ) = (
                    str(site + 1),
                    str(site),
                )
        else:
            for site in range(site_source, site_destination, -1):
                self.swap_neighbouring_sites(site - 1)
                reverse_map = {v: k for k, v in site_mapping.items()}
                (
                    site_mapping[reverse_map[str(site - 1)]],
                    site_mapping[reverse_map[str(site)]],
                ) = (
                    str(site),
                    str(site - 1),
                )
        return

    def apply_two_qubit_gate(
        self,
        data: SparseArray,
        sites: list[int],
        site_mapping: dict,
        max_bond: int | None = None,
        tol: float = 1e-12,
    ) -> None:
        """
        Apply a two qubit gate in place

        Args:
            data: The two-qubit matrix
            sites: The sites to apply it to
            site_mapping: Site mapping from physical to logical sites
            max_bond: The maximum allowed bond dimension
        """
        site0, site1 = sites[0], sites[1]

        if self.num_sites == 2:
            data = sparse.reshape(data, (2, 2, 2, 2))
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            if site1 < site0:
                data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            data = sparse.reshape(data, (4, 4))
            gate = UnitaryGate(data.todense())
            qc = QuantumCircuit(2)
            qc.append(gate, [site0 - 1, site1 - 1])
            gate_mpo = self.from_qiskit_gate(qc.data[0])
            self *= gate_mpo
            return

        data = sparse.reshape(data, (2, 2, 2, 2))
        if site1 < site0:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            if site1 == site0 - 1:
                pass
            else:
                self.move_site_to_location(site1, site0 - 1, site_mapping)
            tensor0 = self.tensors[site0 - 2]
            tensor1 = self.tensors[site0 - 1]
            if site0 - 1 == 1:
                contraction = "hij,hklm,noil->komnj"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor1.dimensions[3],
                    2,
                    tensor0.dimensions[2],
                )
                mat_shape = (
                    tensor1.dimensions[3] * 2 * tensor1.dimensions[1],
                    2 * tensor0.dimensions[2],
                )
            elif site0 == self.num_sites:
                contraction = "hijk,ilm,nojl->omhnk"
                output_shape = (
                    2,
                    tensor1.dimensions[2],
                    tensor0.dimensions[0],
                    2,
                    tensor0.dimensions[3],
                )
                mat_shape = (
                    2 * tensor1.dimensions[2],
                    tensor0.dimensions[0] * 2 * tensor0.dimensions[3],
                )
            else:
                contraction = "hijk,ilmn,opjm->lpnhok"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor1.dimensions[3],
                    tensor0.dimensions[0],
                    2,
                    tensor0.dimensions[3],
                )
                mat_shape = (
                    tensor1.dimensions[1] * 2 * tensor1.dimensions[3],
                    tensor0.dimensions[0] * 2 * tensor0.dimensions[3],
                )
        else:
            if site1 == site0 + 1:
                pass
            else:
                self.move_site_to_location(site1, site0 + 1, site_mapping)
            tensor0 = self.tensors[site0 - 1]
            tensor1 = self.tensors[site0]
            if site0 == 1:
                contraction = "hij,hklm,noil->komnj"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor1.dimensions[3],
                    2,
                    tensor0.dimensions[2],
                )
                mat_shape = (
                    tensor1.dimensions[3] * 2 * tensor1.dimensions[1],
                    2 * tensor0.dimensions[2],
                )
            elif site0 + 1 == self.num_sites:
                contraction = "hijk,ilm,nojl->omhnk"
                output_shape = (
                    2,
                    tensor1.dimensions[2],
                    tensor0.dimensions[0],
                    2,
                    tensor0.dimensions[3],
                )
                mat_shape = (
                    2 * tensor1.dimensions[2],
                    tensor0.dimensions[0] * 2 * tensor0.dimensions[3],
                )
            else:
                contraction = "hijk,ilmn,opjm->lpnhok"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor1.dimensions[3],
                    tensor0.dimensions[0],
                    2,
                    tensor0.dimensions[3],
                )
                mat_shape = (
                    tensor1.dimensions[1] * 2 * tensor1.dimensions[3],
                    tensor0.dimensions[0] * 2 * tensor0.dimensions[3],
                )

        output_data = sparse.einsum(contraction, tensor0.data, tensor1.data, data)
        output_data = np.reshape(output_data, mat_shape)

        if max_bond:
            bond_dim = min([max_bond, mat_shape[0], mat_shape[1]])
        else:
            bond_dim = min([mat_shape[0], mat_shape[1]])

        if bond_dim >= min([mat_shape[0], mat_shape[1]]) - 1:
            u, s, vh = svd(output_data.todense(), full_matrices=False)
        else:
            u, s, vh = svds(output_data, k=bond_dim)

        s = s[s > 1e-14]
        sq = s**2
        cumulative = np.cumsum(sq[::-1])[::-1]
        keep_dim = len(s)
        for k in range(len(s)):
            if cumulative[k] < tol**2:
                keep_dim = k + 1
                break
        keep_dim = min(keep_dim, bond_dim)

        new_data0 = sparse.COO.from_numpy(vh[:keep_dim, :])
        new_data1 = sparse.COO.from_numpy(u[:, :keep_dim] * s[:keep_dim])

        if site1 < site0:
            if site0 - 1 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data1 = sparse.reshape(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [3], [0])
            elif site0 == self.num_sites:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            else:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [3], [0])
            self.tensors[site0 - 2].data = new_data0
            self.tensors[site0 - 2].dimensions = self.tensors[site0 - 2].data.shape
            self.tensors[site0 - 1].data = new_data1
            self.tensors[site0 - 1].dimensions = self.tensors[site0 - 1].data.shape
            self.bond_dims = [t.dimensions[0] for t in self.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        else:
            if site0 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data1 = sparse.reshape(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [3], [0])
            elif site0 + 1 == self.num_sites:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            else:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [3], [0])
            self.tensors[site0 - 1].data = new_data0
            self.tensors[site0 - 1].dimensions = self.tensors[site0 - 1].data.shape
            self.tensors[site0].data = new_data1
            self.tensors[site0].dimensions = self.tensors[site0].data.shape
            self.bond_dims = [t.dimensions[0] for t in self.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        return

    def apply_general_gate(
        self,
        inst: CircuitInstruction,  # type: ignore
        site_mapping: dict,
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":  # type: ignore
        """
        Apply a gate with no better option

        Args:
            inst: The circuit instruction
            site_mapping: Site mapping of physical to logical sites
            max_bond: Maximum bond dimension
        """
        qidxs = [inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)]
        qidxs = [int(site_mapping[str(qidx)]) for qidx in qidxs]
        mpo = MatrixProductOperator.from_qiskit_gate(inst)
        mpo = self.contract_sub_mpo(mpo, qidxs, max_bond)
        return mpo

    @classmethod
    def from_qiskit_circuit(
        self,
        qc: QuantumCircuit,
        after_gate: int | None = None,
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """
        Build the MPO representing the quantum circuit

        Args:
            qc: The QuantumCircuit object
            after_gate: Builds the MPO representing the circuit up to after the given gate number. Defaults to full circuit
            max_bond: Maximum allowed bond dimension

        Returns:
            An MPO
        """
        if after_gate is not None:
            data = qc.data[:after_gate]
        else:
            data = qc.data
        mpo = MatrixProductOperator.identity_mpo(qc.num_qubits)
        site_mapping = {str(idx): str(idx) for idx in range(1, mpo.num_sites + 1)}
        for inst in data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            data = sparse.COO.from_numpy(Operator(inst.operation).reverse_qargs().data)
            if len(qidxs) == 1:
                site_loc = int(site_mapping[str(qidxs[0])])
                mpo.apply_one_qubit_gate(data, site_loc)
            elif len(qidxs) == 2:
                sites_locs = [int(site_mapping[str(site)]) for site in qidxs]
                mpo.apply_two_qubit_gate(
                    data, sites_locs, site_mapping, max_bond=max_bond
                )
            else:
                mpo = mpo.apply_general_gate(inst, site_mapping, max_bond=max_bond)

        reversed_mapping = {v: k for k, v in site_mapping.items()}
        target_site_ordering = [
            int(reversed_mapping[str(site)]) for site in range(1, mpo.num_sites + 1)
        ]
        mpo.reorder_sites(target_site_ordering)
        return mpo

    @classmethod
    def from_qiskit_gate(cls, inst: CircuitInstruction) -> "MatrixProductOperator":  # type: ignore
        """
        Create an MPO from a single Qiskit gate

        Args:
            inst: The Qiskit CircuitInstruction

        Returns:
            An MPO
        """
        qidxs = [inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)]
        indices = [f"out{qidxs[i]}" for i in range(inst.operation.num_qubits)] + [
            f"in{qidxs[i]}" for i in range(inst.operation.num_qubits)
        ]
        if len(qidxs) == 1:
            arrays = [Operator(inst.operation).reverse_qargs().data]
        elif len(qidxs) == 2:
            tensor = Tensor.from_qiskit_gate(inst, indices=indices)
            tn = TensorNetwork([tensor])
            tn.svd(
                tn.tensors[0],
                input_indices=[indices[0], indices[2]],
                output_indices=[indices[1], indices[3]],
                new_index_name=f"C{qidxs[0]}",
            )
            tn.tensors[0].reorder_indices(
                [f"C{qidxs[0]}", f"out{qidxs[0]}", f"in{qidxs[0]}"]
            )
            tn.tensors[1].reorder_indices(
                [f"C{qidxs[0]}", f"out{qidxs[1]}", f"in{qidxs[1]}"]
            )
            arrays = [tn.tensors[i].data for i in range(2)]
        else:
            tensor = Tensor.from_qiskit_gate(inst, indices=indices)
            tn = TensorNetwork([tensor])
            for idx in range(len(qidxs) - 1):
                t = tn.tensors[idx]
                input_inds = [indices[idx], indices[len(qidxs) + idx]]
                output_inds = (
                    indices[idx + 1 : len(qidxs)] + indices[len(qidxs) + idx + 1 :]
                )
                if idx != 0:
                    input_inds.insert(0, f"C{idx}")
                tn.svd(
                    t,
                    input_indices=input_inds,
                    output_indices=output_inds,
                    new_index_name=f"C{idx+1}",
                    new_labels=[[f"T{idx+1}"], [f"T{idx+2}"]],
                )
                if idx == 0:
                    new_idx_order1 = [
                        f"C{idx+1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                    new_idx_order2 = [f"C{idx+1}"] + output_inds
                else:
                    new_idx_order1 = [
                        f"C{idx}",
                        f"C{idx+1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                new_idx_order2 = [f"C{idx+1}"] + output_inds
                tn.tensors[idx].reorder_indices(new_idx_order1)
                tn.tensors[idx + 1].reorder_indices(new_idx_order2)
            arrays = [tn.tensors[i].data for i in range(len(qidxs))]
        mpo = cls.from_arrays(arrays)
        return mpo

    @classmethod
    def from_qiskit_circuit_zip_up(
        cls, qc: QuantumCircuit, max_bond: int
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a circuit using a zip up method.

        Args:
            qc: The quantum circuit.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        dag = circuit_to_dag(qc)
        all_layers = [label for label in dag.layers()]
        all_layers_circs = [dag_to_circuit(layer["graph"]) for layer in all_layers]
        all_layers_mpo = [
            MatrixProductOperator.from_qiskit_circuit(circ) for circ in all_layers_circs
        ]
        mpo = all_layers_mpo[0]
        for idx in range(1, len(all_layers_mpo)):
            mpo_to_zip = all_layers_mpo[idx]
            mpo = mpo.zip_up(mpo_to_zip, max_bond)
        return mpo

    @classmethod
    def zero_reflection_mpo(cls, num_sites: int) -> "MatrixProductOperator":
        """
        Create an MPO for the zero reflection operator.

        Args:
            num_sites: The number of sites for the MPO.

        Returns:
            An MPO.
        """
        x_layer = QuantumCircuit(num_sites)
        for idx in range(num_sites):
            x_layer.x(idx)
        x_layer_mpo = cls.from_qiskit_circuit(x_layer)

        z_gate = np.array([[1, 0], [0, -1]])
        mcz_mpo = cls.generalised_mcu_mpo(
            num_sites, [], list(range(1, num_sites)), num_sites, z_gate
        )

        mpo = copy.deepcopy(x_layer_mpo)
        mpo = mpo * mcz_mpo
        mpo = mpo * x_layer_mpo

        return mpo

    @classmethod
    def from_bitstring(cls, bs: str) -> "MatrixProductOperator":
        """
        Construct an MPO from a single bitstring.

        Args:
            bs: The bitstring.

        Returns:
            An MPO for the operator that projects onto the given bitstring.
        """
        proj_0_rank3 = np.array([[1, 0], [0, 0]], dtype=complex).reshape(1, 2, 2)
        proj_0_rank4 = np.array([[1, 0], [0, 0]], dtype=complex).reshape(1, 1, 2, 2)
        proj_1_rank3 = np.array([[0, 0], [0, 1]], dtype=complex).reshape(1, 2, 2)
        proj_1_rank4 = np.array([[0, 0], [0, 1]], dtype=complex).reshape(1, 1, 2, 2)

        if len(bs) == 1:
            if bs == "0":
                mpo = MatrixProductOperator.from_arrays([proj_0_rank3.reshape((2, 2))])
            else:
                mpo = MatrixProductOperator.from_arrays([proj_1_rank3.reshape((2, 2))])
            return mpo

        arrays = []

        first_array = proj_0_rank3 if bs[0] == "0" else proj_1_rank3
        arrays.append(first_array)

        for b in bs[1:-1]:
            array = proj_0_rank4 if b == "0" else proj_1_rank4
            arrays.append(array)

        last_array = proj_0_rank3 if bs[-1] == "0" else proj_1_rank3
        arrays.append(last_array)

        mpo = cls.from_arrays(arrays)
        return mpo

    @classmethod
    def projector_from_samples(
        cls, samples: List[str], max_bond: int
    ) -> "MatrixProductOperator":
        """
        Construct an MPO projector from bitstring samples. For use in QHCI.

        Args:
            samples: List of bitstrings.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        mpo = cls.from_bitstring(samples[0])
        for sample in samples[1:]:
            temp_mpo = cls.from_bitstring(sample)
            mpo = mpo + temp_mpo
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        return mpo

    @classmethod
    def from_fermionic_string(
        cls, num_sites: int, op_list: list[tuple]
    ) -> "MatrixProductOperator":
        """
        Construct an MPO from a Fermion operator consisting of a single string creation and annihilation operators.

        Args:
            num_sites: The total number of sites = number of spin-orbitals
            op:_list A list of tuples of the form (idx, o) where o is a creation ("+") or annihilation ("-") operator acting on the spin-orbital with index idx.

        Return:
            An MPO.
        """
        creation_op = np.array([[0, 0], [1, 0]], dtype=complex)
        annihilation_op = np.array([[0, 1], [0, 0]], dtype=complex)
        identity_op = np.array([[1, 0], [0, 1]], dtype=complex)
        z_op = np.array([[1, 0], [0, -1]], dtype=complex)

        strings = [""] * num_sites
        for o_qubit, o_val in op_list:
            for i in range(int(o_qubit)):
                strings[i] += "Z"
            strings[int(o_qubit)] += o_val

        arrays = [0] * num_sites

        # If the list is empty, assumes that its an identity operator
        if len(op_list) == 0:
            return MatrixProductOperator.identity_mpo(num_sites)

        for x in range(num_sites):
            total_op = identity_op.copy()
            for y in strings[x]:
                if x == "Z":
                    total_op = total_op @ z_op
                if x == "+":
                    total_op = total_op @ creation_op
                if x == "-":
                    total_op = total_op @ annihilation_op

            arrays[x] = (
                total_op.reshape(1, 2, 2)
                if x == 0 or x == num_sites - 1
                else total_op.reshape(1, 1, 2, 2)
            )

        return cls.from_arrays(arrays)

    @classmethod
    def from_fermionic_operator(
        cls, num_sites: int, ops: list[tuple], max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO from a linear combination of strings of fermionic creation and annihilation operators.

        Args:
            num_sites: The total number of sites = number of spin-orbitals
            ops: A list of tuples of the form (op, weight) where op is a single fermionic operator as defined in the from_fermionic_string method.

        Returns:
            An MPO.
        """
        mpo = MatrixProductOperator.from_fermionic_string(num_sites, ops[0][0])
        mpo.multiply_by_constant(ops[0][1])
        for op, weight in ops[1:]:
            temp_mpo = MatrixProductOperator.from_fermionic_string(num_sites, op)
            temp_mpo.multiply_by_constant(weight)
            mpo = mpo + temp_mpo
            if max_bond:
                if mpo.bond_dimension > max_bond:
                    mpo.compress(max_bond)
        return mpo

    @classmethod
    def from_electron_integral_arrays_adder(
        cls,
        one_elec_integrals: ndarray,
        two_elec_integrals: ndarray,
        max_bond: int | None = None,
    ):
        """
        Construct an MPO of a Fermionic Hamiltonian given as the arrays of one and two electron integrals. Slow method

        Args:
            one_elec_integrals: The 1e integrals in an (N,N) array.
            two_elec_integrals: The 2e integrals in an (N,N,N,N) array.

        Returns:
            An MPO.
        """
        ops = []
        num_sites = one_elec_integrals.shape[0]
        for i in range(num_sites):
            for j in range(num_sites):
                op_list = [(f"{i}", "+"), (f"{j}", "-")]
                ops.append((op_list, one_elec_integrals[i, j]))

        for i in range(num_sites):
            for j in range(num_sites):
                for k in range(num_sites):
                    for l in range(num_sites):
                        op_list = [
                            (f"{i}", "+"),
                            (f"{j}", "+"),
                            (f"{k}", "-"),
                            (f"{l}", "-"),
                        ]
                        ops.append((op_list, 0.5 * two_elec_integrals[i, j, k, l]))

        mpo = MatrixProductOperator.from_fermionic_operator(num_sites, ops)
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        return mpo

    @classmethod
    def from_electron_integral_arrays(
        cls,
        one_elec_integrals: ndarray,
        two_elec_integrals: ndarray,
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """
        Construct an MPO of a Fermionic Hamiltonian given as the arrays of one and two electron integrals. Fast method

        Args:
            one_elec_integrals: The 1e integrals in an (N,N) array.
            two_elec_integrals: The 2e integrals in an (N,N,N,N) array.

        Returns:
            An MPO.
        """
        num_qubits = len(one_elec_integrals)

        ops = []
        for i in range(num_qubits):
            for j in range(num_qubits):
                op_list = [(f"{i}", "+"), (f"{j}", "-")]
                ops.append((op_list, one_elec_integrals[i, j]))

        for i in range(num_qubits):
            for j in range(num_qubits):
                for k in range(num_qubits):
                    for l in range(num_qubits):
                        op_list = [
                            (f"{i}", "+"),
                            (f"{j}", "+"),
                            (f"{k}", "-"),
                            (f"{l}", "-"),
                        ]
                        ops.append((op_list, 0.5 * two_elec_integrals[i, j, k, l]))

        first_array_coords: list[list[int]] = [[], [], []]
        middle_array_coords: list[list[list[int]]] = [
            [[], [], [], []] for _ in range(1, num_qubits - 1)
        ]
        last_array_coords: list[list[int]] = [[], [], []]

        first_array_data: list[complex] = []
        middle_array_data: list[list[complex]] = [[] for _ in range(1, num_qubits - 1)]
        last_array_data: list[complex] = []

        op_idx = 0
        for op_list, weight in ops:
            if weight == 0.0:
                continue

            strings = [""] * num_qubits
            for o_qubit, o_val in op_list:
                for i in range(int(o_qubit)):
                    strings[i] += "Z"
                strings[int(o_qubit)] += o_val

            # First Term
            _update_array_fermion(
                first_array_coords, first_array_data, weight, op_idx, strings[0]
            )

            # Middle Terms
            for idx in range(1, num_qubits - 1):
                _update_array_fermion(
                    middle_array_coords[idx - 1],
                    middle_array_data[idx - 1],
                    1,
                    op_idx,
                    strings[idx],
                    offset=True,
                )

            # Final Term
            _update_array_fermion(
                last_array_coords, last_array_data, 1, op_idx, strings[-1]
            )

            op_idx += 1

        first_array = sparse.COO(
            first_array_coords, first_array_data, shape=(op_idx, 2, 2)
        )
        middle_arrays = [
            sparse.COO(
                middle_array_coords[i - 1],
                middle_array_data[i - 1],
                shape=(op_idx, op_idx, 2, 2),
            )
            for i in range(1, num_qubits - 1)
        ]
        last_array = sparse.COO(
            last_array_coords, last_array_data, shape=(op_idx, 2, 2)
        )

        mpo = MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array]
        )
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)

        return mpo

    @classmethod
    def from_electron_integral_arrays_approx(
        cls,
        one_elec_integrals: ndarray,
        two_elec_integrals: ndarray,
        max_bond: int | None = None,
        threshold: float = 1e-4,
    ) -> "MatrixProductOperator":
        """
        Construct an approximate MPO for second quantised Hamiltonian by discarding terms with small weights

        Args:
            one_elec_integrals: The 1e integrals in an (N,N) array.
            two_elec_integrals: The 2e integrals in an (N,N,N,N) array.

        Returns:
            An MPO.
        """
        n = len(one_elec_integrals)
        one_elec_vals = [one_elec_integrals[i, j] for i in range(n) for j in range(n)]
        two_elec_vals = [
            two_elec_integrals[i, j, k, l]
            for i in range(n)
            for j in range(n)
            for k in range(n)
            for l in range(n)
        ]
        all_vals = [np.abs(v) for v in one_elec_vals] + [
            0.5 * np.abs(v) for v in two_elec_vals
        ]
        norm = np.sum(all_vals)
        cutoff = norm * threshold
        one_elec_integrals = np.where(
            one_elec_integrals > cutoff, one_elec_integrals, 0.0
        )
        two_elec_integrals = np.where(
            two_elec_integrals > cutoff, two_elec_integrals, 0.0
        )
        mpo = cls.from_electron_integral_arrays(
            one_elec_integrals, two_elec_integrals, max_bond
        )
        return mpo

    @classmethod
    def from_diagonal_matrix(
        cls, diag: list[complex], max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO representation of a diagonal matrix.

        Args:
            diag: The list of diagonal entries, should be length 2^N
            max_bond: Maximum allowed bond dimension
        """
        num_sites = int(np.log2(len(diag)))
        mpo = MatrixProductOperator.from_bitstring("0" * num_sites)
        mpo.multiply_by_constant(diag[0])
        for i in range(1, len(diag)):
            bitstring = bin(i)[2:].zfill(num_sites)
            temp_mpo = MatrixProductOperator.from_bitstring(bitstring)
            temp_mpo.multiply_by_constant(diag[i])
            mpo = mpo + temp_mpo
            if max_bond:
                if mpo.bond_dimension > max_bond:
                    mpo.compress(max_bond)
        return mpo

    @classmethod
    def from_short_diagonal_matrix(
        cls, num_sites: int, diag: list[complex], max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO representation of a diagonal matrix of length k followed by 1s the rest of the way

        Args:
            num_sites: Total number of sites
            diag: List of length k < 2^num_sites

        Returns:
            MPO
        """
        mpo = MatrixProductOperator.identity_mpo(num_sites)
        for i in range(len(diag)):
            bitstring = bin(i)[2:].zfill(num_sites)
            temp_mpo = MatrixProductOperator.from_bitstring(bitstring)
            temp_mpo.multiply_by_constant(diag[i] - 1.0)
            mpo = mpo + temp_mpo
            if max_bond:
                if mpo.bond_dimension > max_bond:
                    mpo.compress(max_bond)

        return mpo

    @classmethod
    def from_diagonal_matrix_approx(
        cls, diag: list[complex]
    ) -> "MatrixProductOperator":
        """
        Constructs an MPO of bond dimension 2 that approximates a diagonal matrix.

        Args:
            diag: The list of entries defining the diagonal matrix
        """
        num_sites = int(np.log2(len(diag)))
        arrays = []

        # Loop over all positions
        for i in range(num_sites):
            if i == 0 or i == num_sites - 1:
                shape = (1, 2, 2)
            else:
                shape = (1, 1, 2, 2)
            site_tensor = np.zeros(shape, dtype=complex)
            for s in [0, 1]:
                # for every s, we filter the entries that have s at the i-th bit (from left)
                filtered_diag = [
                    d
                    for idx, d in enumerate(diag)
                    if ((idx >> (num_sites - 1 - i)) & 1) == s
                ]
                avg_value = np.mean(filtered_diag)
                if i == 0 or i == num_sites - 1:
                    site_tensor[0, s, s] = avg_value
                else:
                    site_tensor[0, 0, s, s] = avg_value
            arrays.append(site_tensor)

        mpo = MatrixProductOperator.from_arrays(arrays)

        return mpo

    @classmethod
    def from_increasing_diagonal_matrix(cls, num_sites: int) -> "MatrixProductOperator":
        """
        Construct an MPO representation of a diagonal matrix where the entries are increasing in size

        Args:
            num_sites: Number of sites.

        Returns:
            An MPO representing the diagonal matrix where the (i,i)-th entry is i/2^num_sites
        """
        arrays = []
        D = 2
        I = np.eye(2)
        P1 = np.array([[0, 0], [0, 1]])

        for site in range(num_sites):
            weight = 2 ** (num_sites - site - 1)
            A = weight * P1

            if site == 0:
                W = np.zeros((D, 2, 2))
                W[0] = I
                W[1] = A
            elif site == num_sites - 1:
                W = np.zeros((D, 2, 2))
                W[0] = A
                W[1] = I
            else:
                W = np.zeros((D, D, 2, 2))
                W[0, 0] = I
                W[0, 1] = A
                W[1, 1] = I

            arrays.append(W)

        mpo = MatrixProductOperator.from_arrays(arrays)
        mpo.multiply_by_constant(1 / 2**num_sites)
        return mpo

    @classmethod
    def from_short_increasing_diagonal_matrix(
        cls, num_sites: int, k: int
    ) -> "MatrixProductOperator":
        """
        Construct an MPO representing a diagonal matrix where the first k entries increase up to a value of 1
        after which point every entry is a 1

        Args:
            num_sites: Number of sites
            k: Number of increasing entries
        """
        mpo = MatrixProductOperator.identity_mpo(num_sites)
        for idx in range(k):
            weight = 1 - idx / k
            bitstring = bin(idx)[2:].zfill(num_sites)
            temp_mpo = MatrixProductOperator.from_bitstring(bitstring)
            temp_mpo.multiply_by_constant(weight)
            mpo -= temp_mpo

        return mpo

    @classmethod
    def random_mpo(cls, num_sites: int, max_bond: int) -> "MatrixProductOperator":
        """
        Create a random MPO

        Args:
            num_sites: The number of sites
            max_bond: Maximum bond dimension
        """
        first_array = np.random.random((max_bond, 2, 2))
        arrays = [first_array]
        for _ in range(num_sites - 2):
            array = np.random.random((max_bond, max_bond, 2, 2))
            arrays.append(array)
        last_array = np.random.random((max_bond, 2, 2))
        arrays.append(last_array)
        mpo = MatrixProductOperator.from_arrays(arrays)
        return mpo

    @classmethod
    def from_sparse_array(
        cls, array: SparseArray, max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO from a sparse array

        Args:
            array: The array
            max_bond: Maximum bond dimension

        Returns:
            MPO
        """
        dense_array = array.todense()
        return cls.from_dense_array(dense_array, max_bond)

    @classmethod
    def from_dense_array(
        cls, array: ndarray, max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO from a dense array

        Args:
            array: The array
            max_bond: Maximum bond dimension

        Returns:
            MPO
        """
        num_qubits = int(np.log2(array.shape[0]))
        array = array.reshape((2,) * (2 * num_qubits))
        indices = [f"R{x}" for x in range(1, num_qubits + 1)] + [
            f"L{x}" for x in range(1, num_qubits + 1)
        ]
        tensor = Tensor(array, indices, ["MPO"])
        tn = TensorNetwork([tensor])

        for idx in range(num_qubits - 1):
            t = tn.tensors[idx]
            input_inds = [indices[idx], indices[num_qubits + idx]]
            output_inds = (
                indices[idx + 1 : num_qubits] + indices[num_qubits + idx + 1 :]
            )
            if idx != 0:
                input_inds.insert(0, f"C{idx}")
            tn.svd(
                t,
                input_indices=input_inds,
                output_indices=output_inds,
                new_index_name=f"C{idx+1}",
                new_labels=[[f"T{idx+1}"], [f"T{idx+2}"]],
            )
            if idx == 0:
                new_idx_order1 = [
                    f"C{idx+1}",
                    "R1",
                    "L1",
                ]
            else:
                new_idx_order1 = [
                    f"C{idx}",
                    f"C{idx+1}",
                    f"R{idx+1}",
                    f"L{idx+1}",
                ]
            new_idx_order2 = [f"C{idx+1}"] + output_inds
            tn.tensors[idx].reorder_indices(new_idx_order1)
            tn.tensors[idx + 1].reorder_indices(new_idx_order2)
        arrays = [tn.tensors[i].data for i in range(num_qubits)]
        mpo = cls.from_arrays(arrays)
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        return mpo

    def to_sparse_array(self) -> SparseArray:
        """
        Converts MPO to a sparse matrix.
        """
        mpo = copy.deepcopy(self)
        mpo.reshape()
        mpo.set_default_indices()
        tensor = mpo.contract_entire_network()
        output_indices = [x for x in mpo.indices if x[0] == "R"]
        input_indices = [x for x in mpo.indices if x[0] == "L"]

        tensor.tensor_to_matrix(input_indices, output_indices)

        return tensor.data

    def to_dense_array(self) -> ndarray:
        """
        Converts MPO to a dense matrix.
        """
        mpo = copy.deepcopy(self)
        sparse_matrix = mpo.to_sparse_array()
        dense_matrix = sparse_matrix.todense()

        return dense_matrix

    def __add__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Defines MPO addition.
        """
        self.reshape()
        other.reshape()
        arrays = []

        t1 = self.tensors[0]
        t2 = other.tensors[0]

        t1_data = t1.data
        t2_data = t2.data
        t1_dimensions = t1.dimensions
        t2_dimensions = t2.dimensions

        t1_data = sparse.moveaxis(t1_data, [0, 1, 2], [1, 2, 0])
        t2_data = sparse.moveaxis(t2_data, [0, 1, 2], [1, 2, 0])
        data1 = sparse.reshape(
            t1_data, (t1_dimensions[2], t1_dimensions[0] * t1_dimensions[1])
        )
        data2 = sparse.reshape(
            t2_data, (t2_dimensions[2], t2_dimensions[0] * t2_dimensions[1])
        )

        new_data = sparse.concatenate([data1, data2], axis=1)
        new_data = sparse.moveaxis(new_data, [0, 1], [1, 0])
        new_data = sparse.reshape(
            new_data,
            (t1_dimensions[0] + t2_dimensions[0], t1_dimensions[1], t1_dimensions[2]),
        )
        new_data = sparse.moveaxis(new_data, [0, 1, 2], [0, 2, 1])
        arrays.append(new_data)

        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx]
            t2 = other.tensors[t_idx]

            t1_data = t1.data
            t2_data = t2.data
            t1_dimensions = t1.dimensions
            t2_dimensions = t2.dimensions

            data1 = sparse.moveaxis(t1_data, [0, 1, 2, 3], [0, 2, 1, 3])
            data2 = sparse.moveaxis(t2_data, [0, 1, 2, 3], [0, 2, 1, 3])

            data1 = sparse.reshape(
                data1,
                (
                    t1_dimensions[0] * t1_dimensions[2],
                    t1_dimensions[1] * t1_dimensions[3],
                ),
            )
            data2 = sparse.reshape(
                data2,
                (
                    t2_dimensions[0] * t2_dimensions[2],
                    t2_dimensions[1] * t2_dimensions[3],
                ),
            )

            zeros_top_right = sparse.COO.from_numpy(
                np.zeros((data1.shape[0], data2.shape[1]))
            )
            zeros_bottom_left = sparse.COO.from_numpy(
                np.zeros((data2.shape[0], data1.shape[1]))
            )

            new_data = sparse.concatenate(
                [
                    sparse.concatenate([data1, zeros_top_right], axis=1),
                    sparse.concatenate([zeros_bottom_left, data2], axis=1),
                ]
            )
            new_data = sparse.moveaxis(new_data, [0, 1], [1, 0])
            new_data = sparse.reshape(
                new_data,
                (
                    t1_dimensions[0] + t2_dimensions[0],
                    t1_dimensions[2],
                    t1_dimensions[1] + t2_dimensions[1],
                    t1_dimensions[3],
                ),
            )
            new_data = sparse.moveaxis(new_data, [0, 1, 2, 3], [0, 2, 1, 3])

            arrays.append(new_data)

        t1 = self.tensors[-1]
        t2 = other.tensors[-1]

        t1_data = t1.data
        t2_data = t2.data
        t1_dimensions = t1.dimensions
        t2_dimensions = t2.dimensions

        t1_data = sparse.moveaxis(t1_data, [0, 1, 2], [1, 2, 0])
        t2_data = sparse.moveaxis(t2_data, [0, 1, 2], [1, 2, 0])
        data1 = sparse.reshape(
            t1_data, (t1_dimensions[2], t1_dimensions[0] * t1_dimensions[1])
        )
        data2 = sparse.reshape(
            t2_data, (t2_dimensions[2], t2_dimensions[0] * t2_dimensions[1])
        )

        new_data = sparse.concatenate([data1, data2], axis=1)
        new_data = sparse.moveaxis(new_data, [0, 1], [1, 0])
        new_data = sparse.reshape(
            new_data,
            (t1_dimensions[0] + t2_dimensions[0], t1_dimensions[1], t1_dimensions[2]),
        )
        new_data = sparse.moveaxis(new_data, [0, 1, 2], [0, 2, 1])
        arrays.append(new_data)

        output = MatrixProductOperator.from_arrays(arrays)
        return output

    def __sub__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Defines MPO subtraction.
        """
        other.multiply_by_constant(-1.0)
        output = self + other
        return output

    def __mul__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Defines MPO multiplication.
        """
        mpo1 = copy.deepcopy(self)
        mpo2 = copy.deepcopy(other)
        mpo1.set_default_indices()
        mpo2.set_default_indices()
        arrays = []

        t1 = mpo1.tensors[0]
        t2 = mpo2.tensors[0]

        t1.indices = ["T1_DOWN", "TO_CONTRACT", "T1_LEFT"]
        t2.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

        tn = TensorNetwork([t1, t2])
        tn.contract_index("TO_CONTRACT")

        tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
        tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
        tensor.reorder_indices(["DOWN", "T2_RIGHT", "T1_LEFT"])
        arrays.append(tensor.data)

        for t_idx in range(1, self.num_sites - 1):
            t1 = mpo1.tensors[t_idx]
            t2 = mpo2.tensors[t_idx]

            t1.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT", "T1_LEFT"]
            t2.indices = ["T2_UP", "T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

            tn = TensorNetwork([t1, t2])
            tn.contract_index("TO_CONTRACT")

            tensor = Tensor(
                tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
            )
            tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
            tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
            tensor.reorder_indices(["UP", "DOWN", "T2_RIGHT", "T1_LEFT"])
            arrays.append(tensor.data)

        t1 = mpo1.tensors[-1]
        t2 = mpo2.tensors[-1]

        t1.indices = ["T1_UP", "TO_CONTRACT", "T1_LEFT"]
        t2.indices = ["T2_UP", "T2_RIGHT", "TO_CONTRACT"]

        tn = TensorNetwork([t1, t2])
        tn.contract_index("TO_CONTRACT")

        tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
        tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
        tensor.reorder_indices(["UP", "T2_RIGHT", "T1_LEFT"])
        arrays.append(tensor.data)

        output = MatrixProductOperator.from_arrays(arrays)
        return output

    def __imul__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Define in place multiplication

        Args:
            other: The other MPO to multiply with
        """
        mul = self * other
        self.tensors = mul.tensors
        for t in self.tensors:
            t.indices = mul.tensors[self.tensors.index(t)].indices
            t.dimensions = mul.tensors[self.tensors.index(t)].dimensions
            t.labels = mul.tensors[self.tensors.index(t)].labels

        self.num_sites = mul.num_sites
        self.shape = mul.shape

        self.internal_inds = mul.get_internal_indices()
        self.external_inds = mul.get_external_indices()
        self.bond_dims = []
        self.physical_dims = []
        for idx in self.internal_inds:
            self.bond_dims.append(mul.get_dimension_of_index(idx))
        for idx in self.external_inds:
            self.physical_dims.append(mul.get_dimension_of_index(idx))
        self.bond_dimension = max(self.bond_dims)
        self.physical_dimension = max(self.physical_dims)

        return self

    def zip_up(
        self, other: "MatrixProductOperator", max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Zip up two MPOs

        Args:
            other: The other MPO to zip up

        Returns:
            The new MPO
        """
        mpo1 = copy.deepcopy(self)
        mpo2 = copy.deepcopy(other)
        mpo1.set_default_indices()
        mpo2.set_default_indices()

        mpo1.move_orthogonality_centre()

        for tidx in range(mpo1.num_sites):
            t1 = mpo1.tensors[tidx]
            t2 = mpo2.tensors[tidx]
            t1_current_indices = t1.indices
            t1.indices = [
                f"D{tidx+1}" if x[0] == "R" else x for x in t1_current_indices
            ]
            t2_current_indices = t2.indices
            t2.indices = [
                f"D{tidx+1}" if x[0] == "L" else x + "_" for x in t2_current_indices
            ]

        all_tensors = mpo1.tensors + mpo2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        tn.contract_index(f"D{mpo1.num_sites}")
        tensor = tn.get_tensors_from_index_name(f"L{mpo1.num_sites}")[0]
        input_inds = [f"R{mpo1.num_sites}_", f"L{mpo1.num_sites}"]
        output_inds = [f"B{mpo1.num_sites-1}", f"B{mpo1.num_sites-1}_"]
        tn.svd(tensor, input_inds, output_inds, new_index_name=f"C{mpo1.num_sites-1}")
        for n in list(range(1, mpo1.num_sites - 1))[::-1]:
            tn.contract_index(f"D{n+1}")
            tn.combine_indices([f"B{n}", f"B{n}_"], new_index_name=f"B{n}")
            tn.contract_index(f"B{n}")
            tensor = tn.get_tensors_from_index_name(f"L{n+1}")[0]
            input_inds = [f"R{n+1}_", f"L{n+1}"]
            output_inds = [f"B{n}", f"B{n}_"]
            tn.svd(tensor, input_inds, output_inds, new_index_name=f"C{n}")
        tn.contract_index("D1")
        tn.combine_indices(["B1", "B1_"], new_index_name="B1")
        tn.contract_index("B1")

        for tidx in range(self.num_sites):
            t = tn.tensors[tidx]
            if tidx == 0:
                t.reorder_indices(["C1", "R1_", "L1"])
            elif tidx == self.num_sites - 1:
                t.reorder_indices([f"C{tidx}", f"R{tidx+1}_", f"L{tidx+1}"])
            else:
                t.reorder_indices(
                    [f"C{tidx}", f"C{tidx+1}", f"R{tidx+1}_", f"L{tidx+1}"]
                )

        arrays = [t.data for t in tn.tensors]
        mpo = MatrixProductOperator.from_arrays(arrays)
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        mpo.move_orthogonality_centre()

        return mpo

    def reshape(self, shape="udrl"):
        """
        Reshape the tensors in the MPO.

        Args:
            shape (optional): Default is 'udrl' (up, down, right, left) but any order is allowed.
        """
        if shape == self.shape:
            return

        first_tensor = self.tensors[0]
        first_current_shape = self.shape.replace("u", "")
        first_new_shape = shape.replace("u", "")
        current_indices = first_tensor.indices
        new_indices = [
            current_indices[first_current_shape.index(n)] for n in first_new_shape
        ]
        first_tensor.reorder_indices(new_indices)

        for t_idx in range(1, self.num_sites - 1):
            t = self.tensors[t_idx]
            current_indices = t.indices
            new_indices = [current_indices[self.shape.index(n)] for n in shape]
            t.reorder_indices(new_indices)

        last_tensor = self.tensors[-1]
        last_current_shape = self.shape.replace("d", "")
        last_new_shape = shape.replace("d", "")
        current_indices = last_tensor.indices
        new_indices = [
            current_indices[last_current_shape.index(n)] for n in last_new_shape
        ]
        last_tensor.reorder_indices(new_indices)

        self.shape = shape
        return

    def move_orthogonality_centre(self, where: int = None) -> None:
        """
        Move the orthogonality centre of the MPO.

        Args:
            where (optional): Defaults to the last tensor.
        """
        if not where:
            where = self.num_sites

        internal_indices = self.get_internal_indices()

        push_down = list(range(1, where))
        push_up = list(range(where, self.num_sites))[::-1]

        max_bond = self.bond_dimension

        for idx in push_down:
            index = internal_indices[idx - 1]
            self.compress_index(index, max_bond)

        for idx in push_up:
            index = internal_indices[idx - 1]
            self.compress_index(index, max_bond, reverse_direction=True)

        return

    def project_to_subspace(
        self, projector: "MatrixProductOperator"
    ) -> "MatrixProductOperator":
        """
        Project the MPO to a subspace.

        Args:
            projector: The projector onto the subspace in MPO form.
        """
        max_bond = self.bond_dimension
        self_copy = copy.deepcopy(self)
        mpo = projector * self_copy
        mpo = mpo * projector
        mpo.compress(max_bond)
        return mpo

    def multiply_by_constant(self, const: complex) -> None:
        """
        Scale the MPO by a constant.

        Args:
            const: The constant.
        """
        tensor = self.tensors[0]
        tensor.multiply_by_constant(const)
        return

    def draw(
        self,
        node_size: int | None = None,
        x_len: int | None = None,
        y_len: int | None = None,
    ):
        """
        Visualise tensor network.

        Args:
            node_size: Size of nodes in figure (optional)
            x_len: Figure width (optional)
            y_len: Figure height (optional)

        Returns:
            Displays plot.
        """
        draw_mpo(self.tensors, node_size, x_len, y_len)

    def dagger(self) -> None:
        """
        Take the conjugate transpose of the MPO.
        """
        for t in self.tensors:
            new_index_order = copy.deepcopy(t.indices)
            new_index_order[-2], new_index_order[-1] = (
                new_index_order[-1],
                new_index_order[-2],
            )
            t.reorder_indices(new_index_order)
            t.data = sparse.COO.conj(t.data)
        return

    def swap_neighbouring_sites(self, idx: int) -> None:
        """
        Swap two neighbouring sites of the MPO.

        Args:
            idx: The index of the first site
        """
        if idx == self.num_sites:
            return
        self.reshape()
        if self.num_sites == 2:
            bond = self.tensors[0].indices[0]
            right_idx1 = self.tensors[0].indices[1]
            left_idx1 = self.tensors[0].indices[2]
            right_idx2 = self.tensors[1].indices[1]
            left_idx2 = self.tensors[1].indices[2]
            self.contract_index(bond)
            self.svd(
                self.tensors[0],
                [right_idx2, left_idx2],
                [right_idx1, left_idx1],
                new_index_name=bond,
                max_bond=None,
                tol=1e-12,
            )
            self.tensors[0].reorder_indices([bond, right_idx2, left_idx2])
            self.tensors[1].reorder_indices([bond, right_idx1, left_idx1])

            self.indices = self.get_all_indices()
            return

        if idx == 1:
            bond = self.tensors[0].indices[0]
            right_idx1 = self.tensors[0].indices[1]
            left_idx1 = self.tensors[0].indices[2]
            right_idx2 = self.tensors[1].indices[2]
            left_idx2 = self.tensors[1].indices[3]
        elif idx == self.num_sites - 1:
            bond = self.tensors[idx - 1].indices[1]
            right_idx1 = self.tensors[idx - 1].indices[2]
            left_idx1 = self.tensors[idx - 1].indices[3]
            right_idx2 = self.tensors[idx].indices[1]
            left_idx2 = self.tensors[idx].indices[2]
        else:
            bond = self.tensors[idx - 1].indices[1]
            right_idx1 = self.tensors[idx - 1].indices[2]
            left_idx1 = self.tensors[idx - 1].indices[3]
            right_idx2 = self.tensors[idx].indices[2]
            left_idx2 = self.tensors[idx].indices[3]

        input_inds = copy.deepcopy(self.tensors[idx - 1].indices)
        input_inds.remove(bond)
        input_inds.remove(right_idx1)
        input_inds.remove(left_idx1)
        input_inds.append(right_idx2)
        input_inds.append(left_idx2)
        output_inds = copy.deepcopy(self.tensors[idx].indices)
        output_inds.remove(bond)
        output_inds.remove(right_idx2)
        output_inds.remove(left_idx2)
        output_inds.append(right_idx1)
        output_inds.append(left_idx1)
        self.contract_index(bond)
        self.svd(
            self.tensors[idx - 1],
            input_inds,
            output_inds,
            max_bond=None,
            tol=1e-12,
            new_index_name=bond,
        )

        if idx == 1:
            self.tensors[idx - 1].reorder_indices([bond] + input_inds)
        else:
            self.tensors[idx - 1].reorder_indices(
                [input_inds[0]] + [bond] + [input_inds[1], input_inds[2]]
            )
        self.tensors[idx].reorder_indices([bond] + output_inds)

        self.indices = self.get_all_indices()
        return

    def swap_sites(self, idx1: int, idx2: int) -> None:
        """
        Swap two sites of the MPO.

        Args:
            idx1: The index of the first site
            idx2: The index of the second site
        """
        if idx1 == idx2:
            return

        self.reshape()
        if idx1 < idx2:
            first_idx = idx1
            second_idx = idx2
        else:
            first_idx = idx2
            second_idx = idx1

        for idx in range(first_idx, second_idx):
            self.swap_neighbouring_sites(idx)
        for idx in list(range(first_idx, second_idx - 1))[::-1]:
            self.swap_neighbouring_sites(idx)
        return

    def reorder_sites(
        self, site_mapping: list[int], set_default_indices: bool = False
    ) -> None:
        """
        Reorder the sites of the MPO without changing the operator.

        Args:
            site_mapping: A list of the target ordering of sites
        """
        target_pos = [i - 1 for i in site_mapping]

        visited = [False] * self.num_sites

        for i in range(self.num_sites):
            if visited[i] or target_pos[i] == i:
                continue

            j = i
            cycle = []

            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = target_pos[j]

            for k in range(len(cycle) - 1, 0, -1):
                # Apply swaps on logical site indices (1-based)
                a = cycle[k - 1] + 1
                b = cycle[k] + 1
                self.swap_sites(a, b)

        if set_default_indices:
            self.set_default_indices()

        return

    def contract_sub_mpo(
        self,
        other: "MatrixProductOperator",
        sites: list[int],
        max_bond: int | None = None,
        contract_right: bool = True,
    ) -> "MatrixProductOperator":
        """
        Contract the MPO with a smaller MPO on the given sites

        Args:
            other: The smaller MPO
            sites: The list of sites where the smaller MPO acts
            max_bond: Maximum allowed bond dimension
            contract_right: If set to False the sub-MPO will be contracted on the left


        Returns:
            An MPO that is the output of the contraction
        """

        mpo1 = copy.deepcopy(self)
        mpo2 = copy.deepcopy(other)
        mpo1.set_default_indices()
        mpo2.set_default_indices()

        all_sites = list(range(1, self.num_sites + 1))
        target_site_ordering = [0] * self.num_sites
        for idx in sites:
            target_site_ordering[idx - 1] = sites.index(idx) + 1
            all_sites.remove(sites.index(idx) + 1)
        for site in all_sites:
            target_site_ordering[target_site_ordering.index(0)] = site
        site_mapping = {
            site: target_site_ordering[site - 1]
            for site in range(1, self.num_sites + 1)
        }
        reverse_mapping = {v: k for k, v in site_mapping.items()}
        restore_ordering = [
            reverse_mapping[idx] for idx in range(1, self.num_sites + 1)
        ]

        mpo1.reorder_sites(target_site_ordering, set_default_indices=True)

        if mpo2.num_sites == mpo1.num_sites:
            if contract_right:
                mpo = mpo1 * mpo2
            else:
                mpo = mpo2 * mpo1
            mpo.reorder_sites(restore_ordering, set_default_indices=True)
            if max_bond:
                if mpo.bond_dimension > max_bond:
                    mpo.compress(max_bond)
            return mpo

        contraction_prefix = "R" if contract_right else "L"
        sub_mpo_prefix = "L" if contract_right else "R"

        for tidx in range(mpo2.num_sites):
            t1 = mpo1.tensors[tidx]
            t2 = mpo2.tensors[tidx]
            t1_current_indices = t1.indices
            t1.indices = [
                f"D{tidx+1}" if x[0] == contraction_prefix else x
                for x in t1_current_indices
            ]
            t2_current_indices = t2.indices
            t2.indices = [
                f"D{tidx+1}" if x[0] == sub_mpo_prefix else x + "_"
                for x in t2_current_indices
            ]

        all_tensors = mpo1.tensors + mpo2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        for n in range(len(sites)):
            tn.contract_index(f"D{n+1}")
        for n in range(len(sites) - 1):
            tn.combine_indices([f"B{n+1}", f"B{n+1}_"], new_index_name=f"B{n+1}")
        if contract_right:
            tn.tensors[0].reorder_indices(["B1", "R1_", "L1"])
        else:
            tn.tensors[0].reorder_indices(["B1", "R1", "L1_"])
        for n in range(1, len(sites)):
            if contract_right:
                tn.tensors[n].reorder_indices(
                    [f"B{n}", f"B{n+1}", f"R{n+1}_", f"L{n+1}"]
                )
            else:
                tn.tensors[n].reorder_indices(
                    [f"B{n}", f"B{n+1}", f"R{n+1}", f"L{n+1}_"]
                )
        arrays = [t.data for t in tn.tensors]
        mpo = MatrixProductOperator.from_arrays(arrays)
        mpo.reorder_sites(restore_ordering, set_default_indices=True)
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(self.bond_dimension)
        return mpo

    def partial_trace(
        self, sites: list[int], matrix: bool = False, set_default_indices: bool = False
    ) -> Union[ndarray, "MatrixProductOperator"]:
        """
        Compute the partial trace.

        Args:
            sites: The list of sites to trace over.
            matrix: If True returns the reduced density matrix, otherwise returns a MPDO.
            set_default_indices: If True resets the index labels to default values

        Returns:
            The reduced state.
        """
        mpo = copy.deepcopy(self)
        num_sites_to_trace = len(sites)

        if not matrix:
            all_sites = list(range(1, self.num_sites + 1))
            target_site_ordering = [0] * self.num_sites
            for idx in sites:
                target_site_ordering[idx - 1] = sites.index(idx) + 1
                all_sites.remove(sites.index(idx) + 1)
            for site in all_sites:
                target_site_ordering[target_site_ordering.index(0)] = site
            mpo.reorder_sites(target_site_ordering, set_default_indices=True)
            for idx in range(num_sites_to_trace):
                if mpo.num_sites == 2:
                    output = sparse.einsum(
                        "brr,bcd->cd", mpo.tensors[0].data, mpo.tensors[1].data
                    )
                    new_indices = [f"R{idx+2}", f"L{idx+2}"]
                    new_dimensions = output.shape
                else:
                    output = sparse.einsum(
                        "brr,bcde->cde", mpo.tensors[0].data, mpo.tensors[1].data
                    )
                    new_indices = [f"B{idx+2}", f"R{idx+2}", f"L{idx+2}"]
                    new_dimensions = output.shape
                mpo.tensors.pop(0)
                mpo.tensors[0].data = output
                mpo.tensors[0].indices = new_indices
                mpo.tensors[0].dimensions = new_dimensions
                mpo.num_sites -= 1
            if set_default_indices:
                mpo.set_default_indices()
            return mpo
        else:
            all_sites = list(range(1, self.num_sites + 1))
            for idx in sites:
                all_sites.remove(idx)
                current_indices = mpo.tensors[idx - 1].indices
                mpo.tensors[idx - 1].indices = [
                    "R" + x[1:] if x[0] == "L" else x for x in current_indices
                ]
            result = mpo.contract_entire_network()
            output_inds = [f"R{x}" for x in all_sites]
            input_inds = [f"L{x}" for x in all_sites]
            result.tensor_to_matrix(input_idxs=input_inds, output_idxs=output_inds)
            return result

    def set_default_indices(
        self,
        internal_prefix: str | None = None,
        input_prefix: str | None = None,
        output_prefix: str | None = None,
    ) -> None:
        """
        Set default indices to an MPO

        Args:
            internal_prefix: If provided the internal bonds will have the form internal_prefix + index
            input_prefix: If provided the input bonds will have the form input_prefix + index
            output_prefix: If provided the output bonds will have the form output_prefix + index
        """
        if not internal_prefix:
            internal_prefix = "B"
        if not input_prefix:
            input_prefix = "L"
        if not output_prefix:
            output_prefix = "R"
        self.reshape("udrl")

        if self.num_sites == 1:
            self.tensors[0].indices = [output_prefix + "1", input_prefix + "1"]
            return

        new_indices_first = [
            internal_prefix + "1",
            output_prefix + "1",
            input_prefix + "1",
        ]
        self.tensors[0].indices = new_indices_first
        for tidx in range(1, self.num_sites - 1):
            t = self.tensors[tidx]
            new_indices_t = [
                internal_prefix + str(tidx),
                internal_prefix + str(tidx + 1),
                output_prefix + str(tidx + 1),
                input_prefix + str(tidx + 1),
            ]
            t.indices = new_indices_t
        new_indices_last = [
            internal_prefix + str(self.num_sites - 1),
            output_prefix + str(self.num_sites),
            input_prefix + str(self.num_sites),
        ]
        self.tensors[-1].indices = new_indices_last
        self.indices = self.get_all_indices()
        return

    def trace(self) -> complex:
        """
        Calculate the trace of the MPO

        Returns:
            The trace
        """
        mpo = copy.deepcopy(self)
        mpo.set_default_indices(
            internal_prefix="B", output_prefix="R", input_prefix="R"
        )
        trace = mpo.contract_entire_network()
        return trace

    def evolve_by_quantum_circuit(
        self, qc: QuantumCircuit, max_bond: int | None = None
    ) -> None:
        """
        Evolve the MPO under the action of a quantum circuit

        Args:
            qc: The QuantumCircuit
            max_bond: Maximum bond dimension
        """
        qc_mpo = MatrixProductOperator.from_qiskit_circuit(qc, max_bond)
        qc_inv_mpo = MatrixProductOperator.from_qiskit_circuit(qc.inverse(), max_bond)

        self = qc_inv_mpo * self
        self = self * qc_mpo

        if max_bond:
            if self.bond_dimension > max_bond:
                self.compress(max_bond)
        return
