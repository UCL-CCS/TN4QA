from __future__ import annotations

import copy
from enum import Enum, auto
from typing import Union

import numpy as np
import sparse
from numpy import ndarray
from numpy.linalg import svd
from qiskit.circuit import CircuitInstruction, Operation
from qiskit.quantum_info import Operator
from sparse import SparseArray

DataOptions = Union[ndarray, SparseArray]
QiskitOptions = Union[CircuitInstruction, Operation]

SPARSE_THRESHOLD = 0.10


class StorageHint(Enum):
    DENSE = auto()  # ndarray
    SPARSE = auto()  # always COO  (mainly for chemistry Hamiltonians)


def _make_storage(array: ndarray, hint: StorageHint = StorageHint.DENSE) -> DataOptions:
    """Return array in the appropriate storage format."""
    arr = np.asarray(array) if not isinstance(array, np.ndarray) else array
    if hint is StorageHint.SPARSE:
        return sparse.COO(arr)
    if hint is StorageHint.DENSE:
        return arr


def _as_dense(data: DataOptions) -> ndarray:
    """Return data as a dense ndarray."""
    if isinstance(data, np.ndarray):
        return data
    return data.todense()


def _as_sparse(data: DataOptions) -> SparseArray:
    """Return data as a COO sparse array."""
    if isinstance(data, np.ndarray):
        return sparse.COO(data)
    return data


_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


class Tensor:
    """
    Base class for tensors.

    Parameters
    ----------
    data : ndarray or sparse.COO
        Underlying array.
    indices : list[str]
        One name per axis.
    labels : list[str]
        Arbitrary metadata tags.
    storage_hint : StorageHint
        Recorded for use by operations that produce derived tensors
    """

    def __init__(
        self,
        data: DataOptions,
        indices: list[str],
        labels: list[str],
        storage_hint: StorageHint = StorageHint.DENSE,
    ) -> None:
        if isinstance(data, np.ndarray) and storage_hint is StorageHint.SPARSE:
            data = sparse.COO(data)
        self.data: DataOptions = data
        self.dimensions: tuple = data.shape
        self.rank: int = len(data.shape)
        self.indices: list[str] = list(indices)
        self.labels: list[str] = list(labels)
        self.storage_hint: StorageHint = storage_hint

    def as_dense(self) -> ndarray:
        return _as_dense(self.data)

    def as_sparse(self) -> SparseArray:
        return _as_sparse(self.data)

    def is_sparse(self) -> bool:
        return isinstance(self.data, SparseArray)

    @classmethod
    def from_array(
        cls,
        array: DataOptions,
        indices: list[str] | None = None,
        labels: list[str] | None = None,
        storage_hint: StorageHint = StorageHint.DENSE,
    ) -> Tensor:
        labels = labels or ["T1"]
        if isinstance(array, SparseArray) and storage_hint == StorageHint.SPARSE:
            data = array
        else:
            arr = _as_dense(array)
            data = _make_storage(arr, storage_hint)
        rank = len(data.shape) if data.size != 0 else 0
        if not indices:
            indices = ["B" + str(i + 1) for i in range(rank)]
        return cls(data, indices, labels, storage_hint)

    @classmethod
    def from_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: list[str] = None,
        labels: list[str] = ["T1"],
        dagger: bool = False,
    ) -> Tensor:
        """
        Construct a tensor object from the array.

        Args:
            gate: the underlying qiskit object.
            indices (optional): Default is "I<int>" for inputs and "O<int>" for outputs.
            labels (optional): Default is "T1".
            dagger: If True will construct a tensor for the conjugate transpose of the gate

        Returns:
            tensor: The Tensor object.
        """

        if isinstance(gate, CircuitInstruction):
            gate = gate.operation

        num_qubits = gate.num_qubits
        num_dims = 2 * num_qubits
        shape = [2] * num_dims
        data = Operator(gate).reverse_qargs().data
        if dagger:
            data = data.conj().T
        data = np.reshape(data, shape)
        if not indices:
            indices = [0] * num_dims
            for idx in range(1, num_qubits + 1):
                indices[idx - 1] = "O" + str(idx)
                indices[num_qubits + idx - 1] = "I" + str(idx)
        labels = labels + [gate.name]

        tensor = cls(data, indices, labels, StorageHint.DENSE)

        return tensor

    @classmethod
    def rank_3_copy(
        cls, indices: list[str] = ["B1", "R1", "L1"], labels: list[str] = ["T1"]
    ) -> Tensor:
        """
        Construct a tensor object for the rank-3 copy tensor.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        array = np.array(
            [[[1, 0], [0, 1]], [[0, 0], [0, 1j * np.sqrt(2)]]], dtype=complex
        ).reshape(2, 2, 2)
        indices = indices
        labels = labels + ["copy3"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    @classmethod
    def rank_4_copy(
        cls, indices: list[str] = ["B1", "B2", "R1", "L1"], labels: list[str] = ["T1"]
    ) -> Tensor:
        """
        Construct a tensor object for the rank-4 copy tensor.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "B2", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        array = np.array(
            [
                [[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
                [[[0, 0], [0, 0]], [[0, 0], [0, 1]]],
            ],
            dtype=complex,
        ).reshape(2, 2, 2, 2)
        indices = indices
        labels = labels + ["copy4"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    @classmethod
    def rank_3_copy_open(
        cls, indices: list[str] = ["B1", "R1", "L1"], labels: list[str] = ["T1"]
    ) -> Tensor:
        """
        Construct a tensor object for the rank-3 copy tensor with open control.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        array = np.array(
            [[[1, 0], [0, 1]], [[1j * np.sqrt(2), 0], [0, 0]]], dtype=complex
        ).reshape(2, 2, 2)
        indices = indices
        labels = labels + ["copy3open"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    @classmethod
    def rank_4_copy_open(
        cls, indices: list[str] = ["B1", "B2", "R1", "L1"], labels: list[str] = ["T1"]
    ) -> Tensor:
        """
        Construct a tensor object for the rank-4 copy tensor with open control.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "B2", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        array = np.array(
            [
                [[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
                [[[0, 0], [0, 0]], [[1, 0], [0, 0]]],
            ],
            dtype=complex,
        ).reshape(2, 2, 2, 2)
        indices = indices
        labels = labels + ["copy4open"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    @classmethod
    def rank_3_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: list[str] = ["B1", "R1", "L1"],
        labels: list[str] = ["T1"],
    ) -> Tensor:
        """
        Construct a tensor object for the rank-3 gate tensor.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        if isinstance(gate, CircuitInstruction):
            gate = gate.operation
        data = Operator(gate).reverse_qargs().data.reshape(2, 2)
        id_array = np.array([[1, 0], [0, 1]], dtype=complex).reshape(2, 2)
        array = np.array([id_array, (1j / np.sqrt(2)) * (id_array - data)]).reshape(
            2, 2, 2
        )
        indices = indices
        labels = labels + [f"rank3{gate.name}"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    @classmethod
    def rank_4_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: list[str] = ["B1", "B2", "R1", "L1"],
        labels: list[str] = ["T1"],
    ) -> Tensor:
        """
        Construct a tensor object for the rank-4 gate tensor.

        Args:
            indices (optional): The list of indices. Defaults to ["B1", "B2", "R1", "L1"].
            labels (optional): The list of labels.

        Returns:
            A tensor.
        """
        if isinstance(gate, CircuitInstruction):
            gate = gate.operation
        data = Operator(gate).reverse_qargs().data.reshape(2, 2)
        id_array = np.array([[1, 0], [0, 1]], dtype=complex).reshape(2, 2)
        zero_array = np.array([[0, 0], [0, 0]], dtype=complex).reshape(2, 2)
        array = np.array(
            [[id_array, zero_array], [zero_array, -0.5 * (data - id_array)]]
        ).reshape(2, 2, 2, 2)
        indices = indices
        labels = labels + [f"rank4{gate.name}"]
        tensor = cls(array, indices, labels, StorageHint.DENSE)
        return tensor

    def __str__(self) -> str:
        fmt = "sparse" if self.is_sparse() else "dense"
        return f"Tensor(shape={self.dimensions}, indices={self.indices}, {fmt})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dense(self) -> ndarray:
        data = self.data
        return data.todense() if hasattr(data, "todense") else data

    def reorder_indices(self, index_order: list[str]) -> None:
        """
        Used to change the order of indices in the tensor object.

        Args:
            index_order: The desired new ordering of indices.
        """
        old_indices = list(range(len(self.indices)))
        new_indices = [index_order.index(idx) for idx in self.indices]
        if isinstance(self.data, np.ndarray):
            new_data = np.moveaxis(self.data, old_indices, new_indices)
        else:
            new_data = sparse.moveaxis(self.data, old_indices, new_indices)
        self.data = new_data
        self.indices = index_order
        self.dimensions = new_data.shape
        return

    def new_index_name(self, prefix: str = "B", n: int = 1) -> str | list[str]:
        """Generate n fresh index names with given prefix."""
        existing = []
        for idx in self.indices:
            if idx.startswith(prefix) and idx[len(prefix) :].isdigit():
                existing.append(int(idx[len(prefix) :]))
        base = max(existing, default=0)
        names = [prefix + str(base + i) for i in range(1, n + 1)]
        return names[0] if n == 1 else names

    def get_dimension_of_index(self, index_name: str) -> int:
        return self.dimensions[self.indices.index(index_name)]

    def get_total_dimension_of_indices(self, idxs: list[str]) -> int:
        return int(np.prod([self.get_dimension_of_index(i) for i in idxs]))

    def combine_indices(self, idxs: list[str], new_index_name: str = None) -> None:
        """
        Merge two or more indices together in the tensor.

        Args:
            idxs: The indices to merge.
            new_index_name (optional): What to call the resulting merged index. Defaults to a new index name.
        """
        original_index_ordering = self.indices
        combined_index_dim = self.get_total_dimension_of_indices(idxs)

        temp_index_ordering = [idx for idx in idxs]
        temp_shape = [self.get_dimension_of_index(idx) for idx in idxs]
        for idx in original_index_ordering:
            if idx not in idxs:
                temp_index_ordering.append(idx)
                temp_shape.append(self.get_dimension_of_index(idx))
        self.reorder_indices(temp_index_ordering)

        new_shape = [combined_index_dim] + temp_shape[len(idxs) :]
        if isinstance(self.data, np.ndarray):
            new_data = np.reshape(self.data, shape=new_shape)
        else:
            new_data = sparse.reshape(self.data, new_shape)
        if not new_index_name:
            new_index_name = self.new_index_name()
        new_index_ordering = [new_index_name] + temp_index_ordering[len(idxs) :]
        new_rank = len(new_index_ordering)

        self.data = new_data
        self.indices = new_index_ordering
        self.dimensions = tuple(new_shape)
        self.rank = new_rank

        return

    def tensor_to_matrix(self, input_idxs: list[str], output_idxs: list[str]) -> None:
        """
        Reshape the tensor into a matrix.

        Args:
            input_idxs: The indices to be treated as matrix inputs.
            output_idxs: The indices to be treated as matrix outputs.
        """
        if len(input_idxs) > 0:
            self.combine_indices(input_idxs, new_index_name="I1")
        if len(output_idxs) > 0:
            self.combine_indices(output_idxs, new_index_name="O1")
        return

    def multiply_by_constant(self, const: complex) -> None:
        self.data = self.data * const

    def dagger(self) -> None:
        """Conjugate (not transpose — use reorder_indices for that)."""
        self.data = self.data.conj()

    def get_closest_unitary(
        self,
        input_indices: list[str],
        output_indices: list[str],
    ) -> Tensor:
        """
        Polar decomposition to find the closest unitary.

        Works on a *copy* so the original tensor is not mutated.
        """
        tmp = copy.deepcopy(self)
        in_dims = [tmp.get_dimension_of_index(i) for i in input_indices]
        out_dims = [tmp.get_dimension_of_index(i) for i in output_indices]
        tmp.tensor_to_matrix(input_indices, output_indices)
        mat = _as_dense(tmp.data)
        u, _, vh = svd(mat, full_matrices=False)
        unitary = (u @ vh).reshape(tuple(out_dims + in_dims))
        new_t = Tensor(unitary, output_indices + input_indices, list(self.labels))
        new_t.reorder_indices(self.indices)
        return new_t
