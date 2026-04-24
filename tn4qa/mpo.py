import copy
from itertools import islice
from typing import List, TypeAlias, Union

import cotengra as ctg

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import scipy
import scipy.linalg
import sparse
from numpy import ndarray

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from scipy.sparse.linalg import svds
from sparse import SparseArray

from .quantum_algorithms.utils import exp_pauli_string_to_circ
from .tensor import StorageHint, Tensor, _as_dense, _as_sparse, _make_storage
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
        cls,
        arrays: List[DataOptions],
        shape: str = "udrl",
        storage_hint: StorageHint = StorageHint.DENSE,
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
        first_tensor = Tensor(arrays[0], first_indices, ["MPO_T1"], storage_hint)
        tensors.append(first_tensor)

        right_idx_pos = shape.index("r")
        left_idx_pos = shape.index("l")
        down_idx_pos = shape.index("d")
        up_idx_pos = shape.index("u")
        for a_idx in range(1, len(arrays) - 1):
            a = arrays[a_idx]
            indices_k = ["", "", "", ""]
            indices_k[right_idx_pos] = f"R{a_idx + 1}"
            indices_k[left_idx_pos] = f"L{a_idx + 1}"
            indices_k[up_idx_pos] = f"B{a_idx}"
            indices_k[down_idx_pos] = f"B{a_idx + 1}"
            tensor_k = Tensor(a, indices_k, [f"MPO_T{a_idx + 1}"], storage_hint)
            tensors.append(tensor_k)

        last_shape = shape.replace("d", "")
        right_idx_pos = last_shape.index("r")
        left_idx_pos = last_shape.index("l")
        up_idx_pos = last_shape.index("u")
        last_indices = ["", "", ""]
        last_indices[right_idx_pos] = f"R{len(arrays)}"
        last_indices[left_idx_pos] = f"L{len(arrays)}"
        last_indices[up_idx_pos] = f"B{len(arrays) - 1}"
        last_tensor = Tensor(
            arrays[-1], last_indices, [f"MPO_T{len(arrays)}"], storage_hint
        )
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
        mpo = cls.from_arrays(arrays, storage_hint=StorageHint.SPARSE)
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
                    storage_hint=StorageHint.DENSE,
                )
                tensors.append(tensor)
            else:
                indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 1, 2, 2),
                    indices,
                    labels,
                    storage_hint=StorageHint.DENSE,
                )
                tensors.append(tensor)

        for qidx in mcu_qubits:
            if qidx == 1 or qidx == num_sites:
                indices = (
                    [f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                    if qidx == 1
                    else [f"B{qidx - 1}", f"R{qidx}", f"L{qidx}"]
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
                tensor.indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
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
                if isinstance(tensor.data, np.ndarray):
                    tensor.data = np.reshape(
                        tensor.data,
                        (tensor.dimensions[0],)
                        + (1,)
                        + (tensor.dimensions[1], tensor.dimensions[2]),
                    )
                else:
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
                tensor.indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                tensor.rank = 4
                tensors.append(tensor)

            else:
                indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                if qidx in zero_ctrls:
                    tensor = Tensor.rank_4_copy_open(indices, labels)
                elif qidx in one_ctrls:
                    tensor = Tensor.rank_4_copy(indices, labels)
                elif qidx == target:
                    tensor = Tensor.rank_4_qiskit_gate(unitary_gate, indices, labels)
                else:
                    tensor = Tensor.from_array(
                        np.eye(4).reshape(2, 2, 2, 2),
                        indices,
                        labels,
                        storage_hint=StorageHint.SPARSE,
                    )
                tensors.append(tensor)

        for qidx in range(last_mcu_qubit + 1, num_sites + 1):
            if qidx == num_sites:
                last_indices = [f"B{num_sites - 1}", f"R{num_sites}", f"L{num_sites}"]
                last_labels = [f"MPO_T{num_sites}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 2, 2),
                    last_indices,
                    last_labels,
                    storage_hint=StorageHint.DENSE,
                )
                tensors.append(tensor)
            else:
                indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
                labels = [f"MPO_T{qidx}"]
                tensor = Tensor.from_array(
                    np.array([[1, 0], [0, 1]], dtype=complex).reshape(1, 1, 2, 2),
                    indices,
                    labels,
                    storage_hint=StorageHint.DENSE,
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
            tensor = Tensor(gate, indices, label, storage_hint=StorageHint.DENSE)
            tensors.append(tensor)
            mpo = cls(tensors)
            return mpo

        first_indices = ["B1", "R1", "L1"]
        first_labels = ["MPO_T1"]
        first_gate = pauli_dict[ps[0]].reshape(1, 2, 2)
        first_tensor = Tensor(
            first_gate, first_indices, first_labels, storage_hint=StorageHint.DENSE
        )
        tensors.append(first_tensor)

        num_sites = len(ps)
        for qidx in range(2, num_sites):
            qidx_indices = [f"B{qidx - 1}", f"B{qidx}", f"R{qidx}", f"L{qidx}"]
            qidx_labels = [f"MPO_T{qidx}"]
            qidx_gate = pauli_dict[ps[qidx - 1]].reshape(1, 1, 2, 2)
            qidx_tensor = Tensor(
                qidx_gate, qidx_indices, qidx_labels, storage_hint=StorageHint.DENSE
            )
            tensors.append(qidx_tensor)

        last_indices = [f"B{num_sites - 1}", f"R{num_sites}", f"L{num_sites}"]
        last_labels = [f"MPO_T{num_sites}"]
        last_gate = pauli_dict[ps[-1]].reshape(1, 2, 2)
        last_tensor = Tensor(
            last_gate, last_indices, last_labels, storage_hint=StorageHint.DENSE
        )
        tensors.append(last_tensor)

        mpo = cls(tensors)
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
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
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

    def apply_one_qubit_gate(
        self, data: SparseArray, site: int, dagger: bool = False
    ) -> None:
        """
        Apply a one-qubit gate in place

        Args:
            data: The one-qubit matrix
            site: Where to apply the gate to
            dagger: If true, applies the inverse of the gate to the left of the mpo
        """
        if dagger:
            data = data.todense()
            data = data.conj()
            data = np.transpose(data)
            data = sparse.COO.from_numpy(data)
        if self.num_sites == 1:
            contraction = "ij,jk->ik" if dagger else "ij,ki->kj"
        elif site == 1 or site == self.num_sites:
            contraction = "ijk,kl->ijl" if dagger else "ijk,lj->ilk"
        else:
            contraction = "hijk,kl->hijl" if dagger else "hijk,lj->hilk"
        self.tensors[site - 1].data = sparse.einsum(
            contraction, self.tensors[site - 1].data, data
        )
        return

    def apply_local_two_qubit_gate(
        self,
        data: SparseArray,
        sites: list[int],
        max_bond: int | None = None,
        tol: float = 1e-12,
    ) -> "MatrixProductOperator":
        """
        Apply a two qubit gate to neighbouring qubits

        Args:
            data: The two-qubit matrix
            sites: The sites to apply it to
            max_bond: The maximum allowed bond dimension
        """
        site0, site1 = sites[0], sites[1]

        if self.num_sites == 2:
            data = sparse.reshape(data, (2, 2, 2, 2))
            if site1 < site0:
                data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            data = sparse.reshape(data, (4, 4))
            gate = UnitaryGate(data.todense())
            qc = QuantumCircuit(2)
            qc.append(gate, [site0 - 1, site1 - 1])
            gate_mpo = self.from_qiskit_gate(qc.data[0])
            mpo = self.contract_sub_mpo(gate_mpo, [site0, site1], max_bond=max_bond)
            return mpo

        if isinstance(data, np.ndarray):
            data = sparse.COO.from_numpy(data)
        data = sparse.reshape(data, (2, 2, 2, 2))
        if site1 < site0:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            assert site1 == site0 - 1
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
            assert site1 == site0 + 1
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

        if tensor0.is_sparse() and tensor1.is_sparse():
            output_data = sparse.einsum(contraction, tensor0.data, tensor1.data, data)
            output_data = sparse.reshape(output_data, mat_shape)
            sh = StorageHint.SPARSE
        else:
            output_data = np.einsum(
                contraction, tensor0.to_dense(), tensor1.to_dense(), data.todense()
            )
            output_data = np.reshape(output_data, mat_shape)
            sh = StorageHint.DENSE

        if max_bond:
            bond_dim = min([max_bond, mat_shape[0], mat_shape[1]])
        else:
            bond_dim = min([mat_shape[0], mat_shape[1]])

        sparse_req = bond_dim < min([mat_shape[0], mat_shape[1]]) - 1
        if tensor0.is_sparse() and tensor1.is_sparse() and sparse_req:
            u, s, vh = svds(output_data, k=max_bond)
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[:, idx]
            vh = vh[idx, :]
        else:
            u, s, vh = scipy.linalg.svd(
                _as_dense(output_data), full_matrices=False, check_finite=False
            )

        u = np.asarray(u)
        s = np.asarray(s)
        vh = np.asarray(vh)

        s = s[s > 1e-14]
        sq = s**2
        cumulative = np.cumsum(sq[::-1])[::-1]
        keep_dim = len(s)
        for k in range(len(s)):
            if cumulative[k] < tol**2:
                keep_dim = k + 1
                break
        keep_dim = min(keep_dim, bond_dim)
        if keep_dim == 0:
            keep_dim += 1

        eps = 1e-16
        data0 = vh[:keep_dim, :]
        data1 = u[:, :keep_dim] * s[:keep_dim]
        data0[np.abs(data0) < eps] = 0.0
        data1[np.abs(data1) < eps] = 0.0

        new_data0 = _make_storage(data0, sh)
        new_data1 = _make_storage(data1, sh)

        if sh == StorageHint.SPARSE:
            reshape_func = sparse.reshape
            moveaxis_func = sparse.moveaxis
        else:
            reshape_func = np.reshape
            moveaxis_func = np.moveaxis

        if site1 < site0:
            if site0 - 1 == 1:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-2:])
                new_data1 = reshape_func(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [3], [0])
            elif site0 == self.num_sites:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = moveaxis_func(new_data0, [0], [1])
                new_data1 = reshape_func(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [2], [0])
            else:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = moveaxis_func(new_data0, [0], [1])
                new_data1 = reshape_func(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [3], [0])
            self.tensors[site0 - 2].data = new_data0
            self.tensors[site0 - 2].dimensions = self.tensors[site0 - 2].data.shape
            self.tensors[site0 - 1].data = new_data1
            self.tensors[site0 - 1].dimensions = self.tensors[site0 - 1].data.shape
            self.bond_dims = [t.dimensions[0] for t in self.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        else:
            if site0 == 1:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-2:])
                new_data1 = reshape_func(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [3], [0])
            elif site0 + 1 == self.num_sites:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = moveaxis_func(new_data0, [0], [1])
                new_data1 = reshape_func(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [2], [0])
            else:
                new_data0 = reshape_func(new_data0, (keep_dim,) + output_shape[-3:])
                new_data0 = moveaxis_func(new_data0, [0], [1])
                new_data1 = reshape_func(new_data1, output_shape[:3] + (keep_dim,))
                new_data1 = moveaxis_func(new_data1, [3], [0])
            self.tensors[site0 - 1].data = new_data0
            self.tensors[site0 - 1].dimensions = self.tensors[site0 - 1].data.shape
            self.tensors[site0].data = new_data1
            self.tensors[site0].dimensions = self.tensors[site0].data.shape
            self.bond_dims = [t.dimensions[0] for t in self.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        return self

    def apply_nonlocal_two_qubit_gate(
        self,
        data,
        sites: list,
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """
        Apply a 2-qubit gate between non-neighbouring sites using a SWAP network.

        Mirrors apply_nonlocal_two_qubit_gate from CircuitSimulator:
        1. Sort sites so site0 < site1; record if original order was flipped.
        2. If already neighbouring, call apply_local_two_qubit_gate directly.
        3. SWAP site1 leftward until adjacent to site0.
        4. Apply local gate (with qubit labels swapped in G if flipped).
        5. SWAP back rightward to original position.
        """
        site0_orig, site1_orig = sites
        site0, site1 = sorted(sites)
        flipped = site0_orig > site1_orig

        if site1 == site0 + 1:
            return self.apply_local_two_qubit_gate(data, sites, max_bond=max_bond)

        G = _as_dense(data).reshape(4, 4)
        if flipped:
            G = G.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

        # Move site1 leftward to be adjacent to site0
        for s in range(site1, site0 + 1, -1):
            _apply_swap_mpo(self, s - 1, max_bond)

        # Apply local gate at (site0, site0+1)
        self.apply_local_two_qubit_gate(
            _as_sparse(G), [site0, site0 + 1], max_bond=max_bond
        )

        # Swap back rightward
        for s in range(site0 + 1, site1):
            _apply_swap_mpo(self, s, max_bond)

        return self

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
        for inst in data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            data = sparse.COO.from_numpy(Operator(inst.operation).reverse_qargs().data)
            if len(qidxs) == 1:
                mpo.apply_one_qubit_gate(data, qidxs[0])
            elif len(qidxs) == 2:
                mpo = mpo.apply_nonlocal_two_qubit_gate(
                    data,
                    [qidxs[0], qidxs[1]],
                    max_bond=max_bond,
                )
        mpo.update_bond_information()
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        mpo.update_bond_information()
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
                    new_index_name=f"C{idx + 1}",
                    new_labels=[[f"T{idx + 1}"], [f"T{idx + 2}"]],
                )
                if idx == 0:
                    new_idx_order1 = [
                        f"C{idx + 1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                    new_idx_order2 = [f"C{idx + 1}"] + output_inds
                else:
                    new_idx_order1 = [
                        f"C{idx}",
                        f"C{idx + 1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                new_idx_order2 = [f"C{idx + 1}"] + output_inds
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

        mpo = mcz_mpo.multiply_and_compress_three(x_layer_mpo, x_layer_mpo)

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
        cls, samples: List[str], max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Construct an MPO projector from bitstring samples. For use in QHCI.

        Args:
            samples: List of bitstrings.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        num_sites = len(samples[0])
        num_states = len(samples)

        first_coords = [[], [], []]
        last_coords = [[], [], []]

        middle_coords = [[[], [], [], []] for _ in range(num_sites - 2)]

        first_data = []
        last_data = []
        middle_data = [[] for _ in range(num_sites - 2)]

        for s_idx, bitstring in enumerate(samples):
            b0 = int(bitstring[0])

            first_coords[0].append(s_idx)
            first_coords[1].append(b0)
            first_coords[2].append(b0)
            first_data.append(1.0)

            for site in range(1, num_sites - 1):
                b = int(bitstring[site])

                mid = site - 1

                middle_coords[mid][0].append(s_idx)
                middle_coords[mid][1].append(s_idx)
                middle_coords[mid][2].append(b)
                middle_coords[mid][3].append(b)

                middle_data[mid].append(1.0)

            bL = int(bitstring[-1])

            last_coords[0].append(s_idx)
            last_coords[1].append(bL)
            last_coords[2].append(bL)
            last_data.append(1.0)

        first_array = sparse.COO(first_coords, first_data, shape=(num_states, 2, 2))

        middle_arrays = [
            sparse.COO(
                middle_coords[i],
                middle_data[i],
                shape=(num_states, num_states, 2, 2),
            )
            for i in range(num_sites - 2)
        ]

        last_array = sparse.COO(last_coords, last_data, shape=(num_states, 2, 2))

        mpo = MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
        )

        if max_bond is not None and mpo.bond_dimension > max_bond:
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
        num_states = len(diag)

        first_coords = [[], [], []]
        middle_coords = [[[], [], [], []] for _ in range(num_sites - 2)]
        last_coords = [[], [], []]

        first_data = []
        middle_data = [[] for _ in range(num_sites - 2)]
        last_data = []

        for state_idx, value in enumerate(diag):
            bitstring = bin(state_idx)[2:].zfill(num_sites)
            b0 = int(bitstring[0])

            first_coords[0].append(state_idx)
            first_coords[1].append(b0)
            first_coords[2].append(b0)
            first_data.append(value)

            for site in range(1, num_sites - 1):
                b = int(bitstring[site])
                mid = site - 1

                middle_coords[mid][0].append(state_idx)
                middle_coords[mid][1].append(state_idx)
                middle_coords[mid][2].append(b)
                middle_coords[mid][3].append(b)

                middle_data[mid].append(1.0)

            bL = int(bitstring[-1])

            last_coords[0].append(state_idx)
            last_coords[1].append(bL)
            last_coords[2].append(bL)
            last_data.append(1.0)

        first_array = sparse.COO(first_coords, first_data, shape=(num_states, 2, 2))

        middle_arrays = [
            sparse.COO(
                middle_coords[i],
                middle_data[i],
                shape=(num_states, num_states, 2, 2),
            )
            for i in range(num_sites - 2)
        ]

        last_array = sparse.COO(last_coords, last_data, shape=(num_states, 2, 2))

        mpo = MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
        )

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
        num_states = 2**num_sites
        k = len(diag)

        first_coords = [[], [], []]
        last_coords = [[], [], []]

        middle_coords = [[[], [], [], []] for _ in range(num_sites - 2)]

        first_data = []
        last_data = []
        middle_data = [[] for _ in range(num_sites - 2)]

        for state_idx in range(num_states):
            bitstring = bin(state_idx)[2:].zfill(num_sites)

            # Determine coefficient
            coeff = diag[state_idx] if state_idx < k else 1.0

            # Correction relative to identity (only needed for first tensor)
            delta = coeff - 1.0

            b0 = int(bitstring[0])

            first_coords[0].append(state_idx)
            first_coords[1].append(b0)
            first_coords[2].append(b0)

            # identity + correction
            first_data.append(1.0 + delta)

            for site in range(1, num_sites - 1):
                b = int(bitstring[site])
                mid = site - 1

                middle_coords[mid][0].append(state_idx)
                middle_coords[mid][1].append(state_idx)
                middle_coords[mid][2].append(b)
                middle_coords[mid][3].append(b)

                middle_data[mid].append(1.0)

            bL = int(bitstring[-1])

            last_coords[0].append(state_idx)
            last_coords[1].append(bL)
            last_coords[2].append(bL)

            last_data.append(1.0)

        first_array = sparse.COO(first_coords, first_data, shape=(num_states, 2, 2))

        middle_arrays = [
            sparse.COO(
                middle_coords[i],
                middle_data[i],
                shape=(num_states, num_states, 2, 2),
            )
            for i in range(num_sites - 2)
        ]

        last_array = sparse.COO(last_coords, last_data, shape=(num_states, 2, 2))

        mpo = MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
        )

        if max_bond is not None and mpo.bond_dimension > max_bond:
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
        diag = [i / (2**num_sites) for i in range(num_sites)]
        mpo = cls.from_diagonal_matrix(diag)
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
        diag = [i / 2**k for i in range(k)]
        mpo = cls.from_short_diagonal_matrix(num_sites, diag)
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
                new_index_name=f"C{idx + 1}",
                new_labels=[[f"T{idx + 1}"], [f"T{idx + 2}"]],
            )
            if idx == 0:
                new_idx_order1 = [
                    f"C{idx + 1}",
                    "R1",
                    "L1",
                ]
            else:
                new_idx_order1 = [
                    f"C{idx}",
                    f"C{idx + 1}",
                    f"R{idx + 1}",
                    f"L{idx + 1}",
                ]
            new_idx_order2 = [f"C{idx + 1}"] + output_inds
            tn.tensors[idx].reorder_indices(new_idx_order1)
            tn.tensors[idx + 1].reorder_indices(new_idx_order2)
        arrays = [tn.tensors[i].data for i in range(num_qubits)]
        mpo = cls.from_arrays(arrays)
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        return mpo

    @classmethod
    def swap_mpo(cls, num_sites: int, target_sites: list[int]):
        """
        Qubit SWAP MPO with bond dimension 2.

        Args:
            num_sites: Total length of MPO
            target_sites: 1-indexed, sites to swap
        """
        qc = QuantumCircuit(num_sites)
        qc.swap(target_sites[0] - 1, target_sites[1] - 1)
        mpo = cls.from_qiskit_circuit(qc)
        return mpo

    @classmethod
    def purity_mpo(
        cls,
        num_sites: int,
        target_sites: list[int],
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """Build an MPO that calculates the purity of a RDM for an MPS.

        Constructs the MPO by directly applying SWAP gates between specified sites.

        Args:
            num_sites: The number of sites for the target MPS
            target_sites: The sites corresponding to the RDM whose purity we want to calculate (1-indexed)

        Returns:
            An MPO representing the purity measurement circuit
        """
        mpo = cls.identity_mpo(2 * num_sites)

        for idx in target_sites:
            qc = QuantumCircuit(2 * num_sites)
            qc.swap(idx - 1, num_sites + idx - 1)
            qc_mpo = cls.from_qiskit_circuit(qc)
            mpo = mpo.multiply_and_compress(qc_mpo, max_bond)

        return mpo

    @classmethod
    def from_hamiltonian_exponential(
        cls,
        hamiltonian: dict[str, complex],
        time: float,
        trotter_steps: int,
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """Build an MPO for e^{-iHt} using Trotterisation
        Args;
            hamiltonian: The Hamiltonian dictionary
            time: t
            trotter_steps: Number of Trotter steps to use in decomposition
        """

        pauli_strings = list(hamiltonian.keys())
        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        for _ in range(trotter_steps):
            for p in pauli_strings:
                temp_qc = exp_pauli_string_to_circ(
                    p, time / trotter_steps * hamiltonian[p]
                )
                qc.compose(temp_qc, inplace=True)

        mpo = cls.from_qiskit_circuit(qc, max_bond=max_bond)
        return mpo

    @classmethod
    def from_pauli_exponential(
        cls, pauli_string: str, x: float
    ) -> "MatrixProductOperator":
        """Construct the MPO for exp(ixP) = cos(x)*I + i*sin(x)*P

        Args:
            pauli_string: The Pauli string P
            x: The rotation coefficient x
        """
        num_sites = len(pauli_string)

        first_coords = [[], [], []]
        last_coords = [[], [], []]

        middle_coords = [[[], [], [], []] for _ in range(num_sites - 2)]

        first_data = []
        last_data = []
        middle_data = [[] for _ in range(num_sites - 2)]

        c = np.cos(x)
        s = 1j * np.sin(x)

        for site, p in enumerate(pauli_string):
            for s_idx in [0, 1]:
                for t_idx in [0, 1]:
                    # identity contribution always present
                    val = c if s_idx == t_idx else 0.0

                    # Pauli contribution
                    if p == "I":
                        val += s if s_idx == t_idx else 0.0
                    elif p == "X":
                        val += s if s_idx != t_idx else 0.0
                    elif p == "Z":
                        val += (
                            s * (1 if s_idx == t_idx else 0) * (1 if s_idx == 0 else -1)
                        )
                    elif p == "Y":
                        if s_idx == 0 and t_idx == 1:
                            val += -s
                        elif s_idx == 1 and t_idx == 0:
                            val += s

                    if site == 0:
                        first_coords[0].append(0)
                        first_coords[1].append(s_idx)
                        first_coords[2].append(t_idx)
                        first_data.append(val)

                    elif site == num_sites - 1:
                        last_coords[0].append(0)
                        last_coords[1].append(s_idx)
                        last_coords[2].append(t_idx)
                        last_data.append(val)

                    else:
                        mid = site - 1
                        middle_coords[mid][0].append(0)
                        middle_coords[mid][1].append(0)
                        middle_coords[mid][2].append(s_idx)
                        middle_coords[mid][3].append(t_idx)
                        middle_data[mid].append(val)

        first_array = sparse.COO(first_coords, first_data, shape=(1, 2, 2))

        middle_arrays = [
            sparse.COO(middle_coords[i], middle_data[i], shape=(1, 1, 2, 2))
            for i in range(num_sites - 2)
        ]

        last_array = sparse.COO(last_coords, last_data, shape=(1, 2, 2))

        return MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
        )

    def to_sparse_array(self) -> SparseArray:
        """
        Converts MPO to a sparse matrix.
        """
        mpo = copy.deepcopy(self)
        mpo.reshape()
        mpo.set_default_indices()
        for t in mpo.tensors:
            if isinstance(t.data, np.ndarray):
                t.data = sparse.COO.from_numpy(t.data)
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
        mpo.reshape()
        mpo.set_default_indices()
        for t in mpo.tensors:
            if not isinstance(t.data, np.ndarray):
                t.data = t.data.todense()
        tensor = mpo.contract_entire_network()
        output_indices = [x for x in mpo.indices if x[0] == "R"]
        input_indices = [x for x in mpo.indices if x[0] == "L"]

        tensor.tensor_to_matrix(input_indices, output_indices)

        return tensor.data

    def __add__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Defines MPO addition.
        """
        self.reshape()
        other.reshape()

        self_sparse = all(isinstance(t.data, SparseArray) for t in self.tensors)
        other_sparse = all(isinstance(t.data, SparseArray) for t in other.tensors)

        result_sparse = self_sparse and other_sparse

        arrays = []

        t1 = self.tensors[0].data
        t2 = other.tensors[0].data

        if result_sparse:
            new_data = sparse.concatenate([_as_sparse(t1), _as_sparse(t2)], axis=0)
        else:
            new_data = np.concatenate([_as_dense(t1), _as_dense(t2)], axis=0)

        arrays.append(new_data)

        # ============================================================
        # MIDDLE TENSORS
        # ============================================================
        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx].data
            t2 = other.tensors[t_idx].data

            if result_sparse:
                t1 = _as_sparse(t1)
                t2 = _as_sparse(t2)

                D1_up, D1_down, d_out, d_in = t1.shape
                D2_up, D2_down, _, _ = t2.shape

                zeros_tr = sparse.COO(np.zeros((D1_up, D2_down, d_out, d_in)))
                zeros_bl = sparse.COO(np.zeros((D2_up, D1_down, d_out, d_in)))

                top = sparse.concatenate([t1, zeros_tr], axis=1)
                bottom = sparse.concatenate([zeros_bl, t2], axis=1)

                new_data = sparse.concatenate([top, bottom], axis=0)

            else:
                t1 = _as_dense(t1)
                t2 = _as_dense(t2)

                D1_up, D1_down, d_out, d_in = t1.shape
                D2_up, D2_down, _, _ = t2.shape

                zeros_tr = np.zeros((D1_up, D2_down, d_out, d_in), dtype=complex)
                zeros_bl = np.zeros((D2_up, D1_down, d_out, d_in), dtype=complex)

                top = np.concatenate([t1, zeros_tr], axis=1)
                bottom = np.concatenate([zeros_bl, t2], axis=1)

                new_data = np.concatenate([top, bottom], axis=0)

            arrays.append(new_data)

        t1 = self.tensors[-1].data
        t2 = other.tensors[-1].data

        if result_sparse:
            new_data = sparse.concatenate([_as_sparse(t1), _as_sparse(t2)], axis=0)
        else:
            new_data = np.concatenate([_as_dense(t1), _as_dense(t2)], axis=0)

        arrays.append(new_data)

        sh = StorageHint.SPARSE if result_sparse else StorageHint.DENSE

        return MatrixProductOperator.from_arrays(arrays, storage_hint=sh)

    def __sub__(self, other: "MatrixProductOperator") -> "MatrixProductOperator":
        """
        Defines MPO subtraction.
        """
        self_copy = copy.deepcopy(self)
        other_copy = copy.deepcopy(other)
        other_copy.multiply_by_constant(-1.0)
        output = self_copy + other_copy
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

    def multiply_and_compress(
        self, other: "MatrixProductOperator", max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """Multiply and compress simultaneously

        Args:
            other: The other MPO acting to the right
            max_bond: Maximum allowed bond dimension
        """
        mpo1 = copy.deepcopy(self)
        mpo2 = copy.deepcopy(other)
        mpo1.set_default_indices("A", "B", "C")
        mpo2.set_default_indices("D", "C", "E")
        tn = TensorNetwork(mpo1.tensors + mpo2.tensors)
        new_tensors = []

        # First contraction
        tn.contract_index("C1")
        tensor = tn.get_tensors_from_index_name("B1")[0]
        tn.svd(
            tensor,
            input_indices=["B1", "E1"],
            output_indices=["A1", "D1"],
            new_index_name="F1",
            new_labels=[["FIRST"], []],
            max_bond=max_bond,
        )
        first_tensor = tn.get_tensors_from_label("FIRST")[0]
        first_tensor.reorder_indices(["F1", "E1", "B1"])
        new_tensors.append(first_tensor)

        # Middle contractions
        n = mpo1.num_sites
        for idx in range(1, n - 1):
            t1, t2 = tn.get_tensors_from_index_name(f"A{idx}")
            t1.labels.append("T1_TEMP_LABEL")
            t2.labels.append("T2_TEMP_LABEL")
            t3 = tn.get_tensors_from_index_name(f"E{idx + 1}")[0]
            t3.labels.append("T3_TEMP_LABEL")
            new_t_data = ctg.array_contract(
                arrays=[t1.data, t2.data, t3.data],
                inputs=[t1.indices, t2.indices, t3.indices],
                output=[
                    f"F{idx}",
                    f"B{idx + 1}",
                    f"E{idx + 1}",
                    f"A{idx + 1}",
                    f"D{idx + 1}",
                ],
                cache_expression=True,
                prefer_einsum=True,
            )
            new_t = Tensor(
                new_t_data,
                [f"F{idx}", f"B{idx + 1}", f"E{idx + 1}", f"A{idx + 1}", f"D{idx + 1}"],
                [f"NEW_LABEL_{idx}"],
            )
            tn.pop_tensors_by_label(t1.labels)
            tn.pop_tensors_by_label(t2.labels)
            tn.pop_tensors_by_label(t3.labels)
            tn.add_tensor(new_t)
            tensor = tn.get_tensors_from_index_name(f"B{idx + 1}")[0]
            tn.svd(
                tensor,
                input_indices=[f"F{idx}", f"B{idx + 1}", f"E{idx + 1}"],
                output_indices=[f"A{idx + 1}", f"D{idx + 1}"],
                new_index_name=f"F{idx + 1}",
                new_labels=[[f"NEXT{idx}"], []],
                max_bond=max_bond,
            )
            next_tensor = tn.get_tensors_from_label(f"NEXT{idx}")[0]
            next_tensor.reorder_indices(
                [f"F{idx}", f"F{idx + 1}", f"E{idx + 1}", f"B{idx + 1}"]
            )
            new_tensors.append(next_tensor)

        # Final contraction
        t1, t2 = tn.get_tensors_from_index_name(f"A{n - 1}")
        t3 = tn.get_tensors_from_index_name(f"E{n}")[0]
        new_t_data = ctg.array_contract(
            arrays=[t1.data, t2.data, t3.data],
            inputs=[t1.indices, t2.indices, t3.indices],
            output=[f"F{n-1}", f"E{n}", f"B{n}"],
            cache_expression=False,
            prefer_einsum=True,
        )
        new_t = Tensor(new_t_data, [f"F{n-1}", f"E{n}", f"B{n}"], [])
        tn.pop_tensors_by_label(t1.labels)
        tn.pop_tensors_by_label(t2.labels)
        tn.pop_tensors_by_label(t3.labels)
        tn.add_tensor(new_t)
        new_tensors.append(new_t)

        output_mpo = MatrixProductOperator(new_tensors)
        output_mpo.set_default_indices()

        return output_mpo

    def multiply_and_compress_three(
        self,
        left: "MatrixProductOperator",
        right: "MatrixProductOperator",
        max_bond: int | None = None,
    ) -> "MatrixProductOperator":
        """Mutiply and compress 3 MPOs simultaneously

        Args:
            left: Another MPO acting to the left
            right: Another MPO acting to the right
            max_bond: Maximum allowed bond dimension
        """
        mpo_centre = copy.deepcopy(self)
        mpo_left = copy.deepcopy(left)
        mpo_right = copy.deepcopy(right)

        mpo_left.set_default_indices("X", "A", "B")
        mpo_centre.set_default_indices("Y", "B", "C")
        mpo_right.set_default_indices("Z", "C", "D")
        tn = TensorNetwork(mpo_left.tensors + mpo_centre.tensors + mpo_right.tensors)
        new_tensors = []

        # First contraction
        tn.contract_index("B1")
        tn.contract_index("C1")
        tensor = tn.get_tensors_from_index_name("A1")[0]
        tn.svd(
            tensor,
            input_indices=["A1", "D1"],
            output_indices=["X1", "Y1", "Z1"],
            new_index_name="W1",
            new_labels=[["FIRST"], []],
            max_bond=max_bond,
        )
        first_tensor = tn.get_tensors_from_label("FIRST")[0]
        first_tensor.reorder_indices(["W1", "D1", "A1"])
        new_tensors.append(first_tensor)

        # Middle contractions
        n = self.num_sites
        for idx in range(1, n - 1):
            t1, t2 = tn.get_tensors_from_index_name(f"X{idx}")
            t1.labels.append("T1_TEMP_LABEL")
            t2.labels.append("T2_TEMP_LABEL")
            t3, t4 = tn.get_tensors_from_index_name(f"C{idx + 1}")
            t3.labels.append("T3_TEMP_LABEL")
            t4.labels.append("T4_TEMP_LABEL")
            new_t_data = ctg.array_contract(
                arrays=[t1.data, t2.data, t3.data, t4.data],
                inputs=[t1.indices, t2.indices, t3.indices, t4.indices],
                output=[
                    f"W{idx}",
                    f"A{idx + 1}",
                    f"D{idx + 1}",
                    f"X{idx + 1}",
                    f"Y{idx + 1}",
                    f"Z{idx + 1}",
                ],
                cache_expression=True,
                prefer_einsum=True,
            )
            new_t = Tensor(
                new_t_data,
                [
                    f"W{idx}",
                    f"A{idx + 1}",
                    f"D{idx + 1}",
                    f"X{idx + 1}",
                    f"Y{idx + 1}",
                    f"Z{idx + 1}",
                ],
                [f"NEW_LABEL_{idx}"],
            )
            tn.pop_tensors_by_label(t1.labels)
            tn.pop_tensors_by_label(t2.labels)
            tn.pop_tensors_by_label(t3.labels)
            tn.pop_tensors_by_label(t4.labels)
            tn.add_tensor(new_t)

            tensor = tn.get_tensors_from_index_name(f"A{idx + 1}")[0]
            tn.svd(
                tensor,
                input_indices=[f"W{idx}", f"A{idx + 1}", f"D{idx + 1}"],
                output_indices=[f"X{idx + 1}", f"Y{idx + 1}", f"Z{idx + 1}"],
                new_index_name=f"W{idx + 1}",
                new_labels=[[f"NEXT{idx}"], []],
                max_bond=max_bond,
            )
            next_tensor = tn.get_tensors_from_label(f"NEXT{idx}")[0]
            next_tensor.reorder_indices(
                [f"W{idx}", f"W{idx + 1}", f"D{idx + 1}", f"A{idx + 1}"]
            )
            new_tensors.append(next_tensor)

        # Final contraction
        t1, t2 = tn.get_tensors_from_index_name(f"X{n - 1}")
        t3, t4 = tn.get_tensors_from_index_name(f"C{n}")
        new_t_data = ctg.array_contract(
            arrays=[t1.data, t2.data, t3.data, t4.data],
            inputs=[t1.indices, t2.indices, t3.indices, t4.indices],
            output=[f"W{n - 1}", f"D{n}", f"A{n}"],
            cache_expression=True,
            prefer_einsum=True,
        )
        new_t = Tensor(new_t_data, [f"W{n - 1}", f"D{n}", f"A{n}"], [])
        tn.pop_tensors_by_label(t1.labels)
        tn.pop_tensors_by_label(t2.labels)
        tn.pop_tensors_by_label(t3.labels)
        tn.pop_tensors_by_label(t4.labels)
        tn.add_tensor(new_t)
        new_tensors.append(new_t)

        output_mpo = MatrixProductOperator(new_tensors)
        output_mpo.set_default_indices()

        return output_mpo

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
                f"D{tidx + 1}" if x[0] == "R" else x for x in t1_current_indices
            ]
            t2_current_indices = t2.indices
            t2.indices = [
                f"D{tidx + 1}" if x[0] == "L" else x + "_" for x in t2_current_indices
            ]

        all_tensors = mpo1.tensors + mpo2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        tn.contract_index(f"D{mpo1.num_sites}")
        tensor = tn.get_tensors_from_index_name(f"L{mpo1.num_sites}")[0]
        input_inds = [f"R{mpo1.num_sites}_", f"L{mpo1.num_sites}"]
        output_inds = [f"B{mpo1.num_sites - 1}", f"B{mpo1.num_sites - 1}_"]
        tn.svd(tensor, input_inds, output_inds, new_index_name=f"C{mpo1.num_sites - 1}")
        for n in list(range(1, mpo1.num_sites - 1))[::-1]:
            tn.contract_index(f"D{n + 1}")
            tn.combine_indices([f"B{n}", f"B{n}_"], new_index_name=f"B{n}")
            tn.contract_index(f"B{n}")
            tensor = tn.get_tensors_from_index_name(f"L{n + 1}")[0]
            input_inds = [f"R{n + 1}_", f"L{n + 1}"]
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
                t.reorder_indices([f"C{tidx}", f"R{tidx + 1}_", f"L{tidx + 1}"])
            else:
                t.reorder_indices(
                    [f"C{tidx}", f"C{tidx + 1}", f"R{tidx + 1}_", f"L{tidx + 1}"]
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
        self, projector: "MatrixProductOperator", max_bond: int | None = None
    ) -> "MatrixProductOperator":
        """
        Project the MPO to a subspace.

        Args:
            projector: The projector onto the subspace in MPO form.
        """
        self_copy = copy.deepcopy(self)
        projector_copy = copy.deepcopy(projector)
        mpo = self_copy.multiply_and_compress_three(
            projector, projector_copy, max_bond=max_bond
        )
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

    def extend_mpo_to_size(
        self, num_sites: int, sites: list[int]
    ) -> "MatrixProductOperator":
        physical_dim = self.tensors[0].dimensions[-1]
        if self.num_sites == 1:
            arrays = []
            for idx in range(1, num_sites + 1):
                if idx in sites:
                    array = self.tensors[0].data
                    if idx == 1 or idx == num_sites:
                        array = array.reshape((1, array.shape[0], array.shape[1]))
                    else:
                        array = array.reshape((1, 1, array.shape[0], array.shape[1]))
                    arrays.append(array)
                else:
                    if idx == 1 or idx == num_sites:
                        array = np.eye(physical_dim).reshape(
                            (1, physical_dim, physical_dim)
                        )
                    else:
                        array = np.eye(physical_dim).reshape(
                            (1, 1, physical_dim, physical_dim)
                        )
                    arrays.append(array)
            mpo = MatrixProductOperator.from_arrays(arrays)
            return mpo

        arrays = []
        current_counter = 0
        for idx in range(1, num_sites + 1):
            if idx in sites:
                array = self.tensors[current_counter].data
                if idx == 1:
                    assert current_counter == 0
                elif idx != 1 and current_counter == 0:
                    array = array.reshape(
                        (1, array.shape[0], array.shape[1], array.shape[2])
                    )
                elif idx == num_sites:
                    assert current_counter == self.num_sites - 1
                elif idx != num_sites and current_counter == self.num_sites - 1:
                    array = array.reshape(
                        (array.shape[0], 1, array.shape[1], array.shape[2])
                    )
                arrays.append(array)
                current_counter += 1
            else:
                if idx == 1:
                    array = np.eye(physical_dim).reshape(
                        (1, physical_dim, physical_dim)
                    )
                elif idx == num_sites:
                    array = np.eye(physical_dim).reshape(
                        (1, physical_dim, physical_dim)
                    )
                elif (
                    idx != 1 and idx != num_sites and current_counter != self.num_sites
                ):
                    array = np.array(
                        [[np.zeros((physical_dim, physical_dim))] * self.bond_dimension]
                        * self.bond_dimension
                    )
                    for x in range(self.bond_dimension):
                        array[x, x, :, :] = np.eye(physical_dim)
                elif (
                    idx != 1 and idx != num_sites and current_counter == self.num_sites
                ):
                    array = np.eye(physical_dim).reshape(
                        (1, 1, physical_dim, physical_dim)
                    )
                arrays.append(array)

        mpo = MatrixProductOperator.from_arrays(arrays)
        return mpo

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
        mpo2 = mpo2.extend_mpo_to_size(mpo1.num_sites, sites)

        if contract_right:
            mpo = mpo1.multiply_and_compress(mpo2, max_bond)
        else:
            mpo = mpo2.multiply_and_compress(mpo1, max_bond)
        mpo.update_bond_information()
        if max_bond:
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
        mpo.update_bond_information()
        return mpo

    def partial_trace(
        self, sites: list[int], matrix: bool = False
    ) -> Union[complex, ndarray, "MatrixProductOperator"]:
        """
        Compute the partial trace.
        Args:
            sites: The list of sites to trace over.
            matrix: If True returns the reduced density matrix as a 2D ndarray,
                    otherwise returns an MPDO.
        Returns:
            The reduced state as a complex scalar, ndarray, or MatrixProductOperator.
        """
        # Handle no trace
        if len(sites) == 0 and matrix:
            return self.to_dense_array()
        elif len(sites) == 0:
            return self

        # Handle full trace
        if len(sites) == self.num_sites:
            return self.trace()

        # Determine final storage hint before deepcopy
        all_sparse = all(t.storage_hint == StorageHint.SPARSE for t in self.tensors)
        final_storage_hint = StorageHint.SPARSE if all_sparse else StorageHint.DENSE

        mpo = copy.deepcopy(self)
        n = mpo.num_sites
        traced_sites = set(sites)

        # Step 1: Trace physical legs of each site to be traced.
        for site_idx in sites:
            tensor = mpo.tensors[site_idx - 1]
            einsum_fn = (
                sparse.einsum
                if tensor.storage_hint == StorageHint.SPARSE
                else np.einsum
            )
            if site_idx == 1:
                tensor.data = einsum_fn("acc->a", tensor.data)
            elif site_idx == n:
                tensor.data = einsum_fn("acc->a", tensor.data)
            else:
                tensor.data = einsum_fn("abcc->ab", tensor.data)

        # Step 2: Find contiguous runs of traced sites
        sorted_sites = sorted(traced_sites)
        runs = []
        run_start = sorted_sites[0]
        run_end = sorted_sites[0]
        for s in sorted_sites[1:]:
            if s == run_end + 1:
                run_end = s
            else:
                runs.append((run_start, run_end))
                run_start = s
                run_end = s
        runs.append((run_start, run_end))

        # Step 3: For each run, contract the chain of traced tensors together,
        # then absorb into the nearest untraced neighbour.
        # Process in reverse order so that popping tensors doesn't shift indices.
        for run_start, run_end in reversed(runs):
            # Contract the chain of traced tensors within this run
            chain_data = mpo.tensors[run_start - 1].data
            is_sparse = mpo.tensors[run_start - 1].storage_hint == StorageHint.SPARSE

            for site_idx in range(run_start + 1, run_end + 1):
                next_data = mpo.tensors[site_idx - 1].data
                next_is_sparse = (
                    mpo.tensors[site_idx - 1].storage_hint == StorageHint.SPARSE
                )

                # Reconcile sparse/dense
                if next_is_sparse and not is_sparse:
                    chain_data = sparse.COO.from_numpy(chain_data)
                    is_sparse = True
                elif not next_is_sparse and is_sparse:
                    next_data = sparse.COO.from_numpy(next_data)
                einsum_fn = sparse.einsum if is_sparse else np.einsum

                chain_rank = chain_data.ndim
                next_rank = next_data.ndim

                if chain_rank == 1 and next_rank == 1:
                    # vector . vector -> scalar
                    chain_data = einsum_fn("a,a->", chain_data, next_data)
                elif chain_rank == 1 and next_rank == 2:
                    # vector . matrix -> vector
                    chain_data = einsum_fn("a,ab->b", chain_data, next_data)
                elif chain_rank == 2 and next_rank == 1:
                    # matrix . vector -> vector
                    chain_data = einsum_fn("ab,b->a", chain_data, next_data)
                elif chain_rank == 2 and next_rank == 2:
                    # matrix . matrix -> matrix
                    chain_data = einsum_fn("ab,bc->ac", chain_data, next_data)

            # chain_data is now rank-0 (scalar), rank-1, or rank-2
            # depending on whether run boundaries are at MPO boundaries.

            # Find nearest untraced neighbour: prefer right (below), else left (above)
            if run_end < n and (run_end + 1) not in traced_sites:
                neighbour_idx = run_end + 1
                absorb_direction = "from_above"
            else:
                neighbour_idx = run_start - 1
                absorb_direction = "from_below"

            neighbour_tensor = mpo.tensors[neighbour_idx - 1]
            nb_is_sparse = neighbour_tensor.storage_hint == StorageHint.SPARSE

            # Reconcile sparse/dense between chain and neighbour
            if is_sparse and not nb_is_sparse:
                neighbour_tensor.data = sparse.COO.from_numpy(neighbour_tensor.data)
                nb_is_sparse = True
            elif not is_sparse and nb_is_sparse:
                chain_data = sparse.COO.from_numpy(chain_data)
            einsum_fn = sparse.einsum if nb_is_sparse else np.einsum

            nb_data = neighbour_tensor.data
            chain_rank = chain_data.ndim

            if absorb_direction == "from_above":
                # Chain is above neighbour: chain's down bond contracts with neighbour's up bond.
                # Neighbour is rank-4 [up, down, R, L] for interior, rank-3 [up, R, L] for site n.
                if chain_rank == 0:
                    neighbour_tensor.data = nb_data * chain_data
                elif chain_rank == 1:
                    # chain: [down]; neighbour up bond contracts with it
                    if neighbour_idx == n:
                        neighbour_tensor.data = einsum_fn(
                            "a,abc->bc", chain_data, nb_data
                        )
                    else:
                        neighbour_tensor.data = einsum_fn(
                            "a,abcd->bcd", chain_data, nb_data
                        )
                else:
                    # chain: [up, down]; contract chain's down with neighbour's up
                    if neighbour_idx == n:
                        neighbour_tensor.data = einsum_fn(
                            "ab,bcd->acd", chain_data, nb_data
                        )
                    else:
                        neighbour_tensor.data = einsum_fn(
                            "ab,bcde->acde", chain_data, nb_data
                        )
            else:
                # Chain is below neighbour: chain's up bond contracts with neighbour's down bond.
                # Neighbour is rank-4 [up, down, R, L] for interior, rank-3 [down, R, L] for site 1.
                if chain_rank == 0:
                    neighbour_tensor.data = nb_data * chain_data
                elif chain_rank == 1:
                    # chain: [up]; neighbour down bond contracts with it
                    if neighbour_idx == 1:
                        neighbour_tensor.data = einsum_fn(
                            "abc,a->bc", nb_data, chain_data
                        )
                    else:
                        neighbour_tensor.data = einsum_fn(
                            "abcd,b->acd", nb_data, chain_data
                        )
                else:
                    # chain: [up, down]; contract chain's up with neighbour's down
                    if neighbour_idx == 1:
                        neighbour_tensor.data = einsum_fn(
                            "abc,ad->dbc", nb_data, chain_data
                        )
                    else:
                        neighbour_tensor.data = einsum_fn(
                            "abcd,be->aecd", nb_data, chain_data
                        )

            neighbour_tensor.storage_hint = (
                StorageHint.SPARSE if nb_is_sparse else StorageHint.DENSE
            )

            # Remove traced tensors in this run (in reverse to preserve indices)
            for site_idx in range(run_end, run_start - 1, -1):
                mpo.tensors.pop(site_idx - 1)
            mpo.num_sites -= run_end - run_start + 1

        # Step 4: Rebuild clean MPO using from_arrays
        array_list = [t.data for t in mpo.tensors]
        if final_storage_hint == StorageHint.SPARSE:
            array_list = [
                sparse.COO.from_numpy(a) if not isinstance(a, sparse.COO) else a
                for a in array_list
            ]
        else:
            array_list = [
                a.todense() if isinstance(a, sparse.COO) else a for a in array_list
            ]

        output_mpo = MatrixProductOperator.from_arrays(
            array_list, storage_hint=final_storage_hint
        )

        if matrix:
            return output_mpo.to_dense_array()
        return output_mpo

    def set_default_indices(
        self,
        internal_prefix: str | None = None,
        input_prefix: str | None = None,
        output_prefix: str | None = None,
        index_from: int | None = None,
    ) -> None:
        """
        Set default indices to an MPO

        Args:
            internal_prefix: If provided the internal bonds will have the form internal_prefix + index
            input_prefix: If provided the input bonds will have the form input_prefix + index
            output_prefix: If provided the output bonds will have the form output_prefix + index
            index_from: Where to start counting from, default to 1
        """
        if not internal_prefix:
            internal_prefix = "B"
        if not input_prefix:
            input_prefix = "L"
        if not output_prefix:
            output_prefix = "R"
        if not index_from:
            index_from = 1
        self.reshape("udrl")

        if self.num_sites == 1:
            self.tensors[0].indices = [
                output_prefix + f"{index_from}",
                input_prefix + f"{index_from}",
            ]
            return

        new_indices_first = [
            internal_prefix + f"{index_from}",
            output_prefix + f"{index_from}",
            input_prefix + f"{index_from}",
        ]
        self.tensors[0].indices = new_indices_first
        for tidx in range(1, self.num_sites - 1):
            t = self.tensors[tidx]
            new_indices_t = [
                internal_prefix + str(tidx + index_from - 1),
                internal_prefix + str(tidx + index_from),
                output_prefix + str(tidx + index_from),
                input_prefix + str(tidx + index_from),
            ]
            t.indices = new_indices_t
        new_indices_last = [
            internal_prefix + str(index_from + self.num_sites - 2),
            output_prefix + str(index_from + self.num_sites - 1),
            input_prefix + str(index_from + self.num_sites - 1),
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

    def compress(self, max_bond: int) -> None:
        """Special compress method for MPO
        Args:
            max_bond: Bond dimension to compress to
        """
        midpoint = int(np.ceil(self.num_sites / 2))
        for tidx in range(midpoint):
            t = self.tensors[tidx]
            original_inds = copy.deepcopy(t.indices)
            original_next_inds = copy.deepcopy(self.tensors[tidx + 1].indices)
            bond_name = t.indices[0] if len(t.indices) == 3 else t.indices[1]
            input_indices = (
                t.indices[1:]
                if len(t.indices) == 3
                else [t.indices[0]] + [t.indices[2], t.indices[3]]
            )
            output_indices = [bond_name]
            self.svd(
                t,
                input_indices=input_indices,
                output_indices=output_indices,
                max_bond=max_bond,
                new_index_name="TEMP",
            )
            self.contract_index(bond_name)
            reordered_indices = (
                ["TEMP", original_inds[1], original_inds[2]]
                if len(original_inds) == 3
                else [original_inds[0], "TEMP", original_inds[2], original_inds[3]]
            )
            reordered_indices_next = ["TEMP"] + original_next_inds[1:]
            self.tensors[tidx].reorder_indices(reordered_indices)
            self.tensors[tidx + 1].reorder_indices(reordered_indices_next)
            self.tensors[tidx].indices = original_inds
            self.tensors[tidx + 1].indices = original_next_inds

        for tidx in range(1, midpoint + 1):
            tidx = -tidx
            t = self.tensors[tidx]
            original_inds = copy.deepcopy(t.indices)
            original_next_inds = copy.deepcopy(self.tensors[tidx - 1].indices)
            bond_name = t.indices[0]
            input_indices = t.indices[1:]
            output_indices = [bond_name]
            self.svd(
                t,
                input_indices=input_indices,
                output_indices=output_indices,
                max_bond=max_bond,
                new_index_name="TEMP",
            )
            self.contract_index(bond_name)
            reordered_indices = (
                ["TEMP", original_inds[1], original_inds[2]]
                if len(original_inds) == 3
                else ["TEMP", original_inds[1], original_inds[2], original_inds[3]]
            )
            reordered_indices_next = [
                original_next_inds[0],
                "TEMP",
            ] + original_next_inds[2:]
            self.tensors[tidx].reorder_indices(reordered_indices)
            self.tensors[tidx - 1].reorder_indices(reordered_indices_next)
            self.tensors[tidx].indices = original_inds
            self.tensors[tidx - 1].indices = original_next_inds
        self.update_bond_information()
        return

    def update_bond_information(self) -> None:
        """Update bond dimension information"""
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
        return


_MPO_SWAP = sparse.COO.from_numpy(
    np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
)


def _apply_swap_mpo(
    mpo: "MatrixProductOperator", site: int, max_bond: int | None
) -> None:
    """Apply SWAP between MPO sites `site` and `site+1` in-place."""
    mpo.apply_local_two_qubit_gate(_MPO_SWAP, [site, site + 1], max_bond=max_bond)
