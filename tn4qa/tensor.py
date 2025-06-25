from typing import List, TypeAlias, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import sparse
from numpy import ndarray
from numpy.linalg import svd
from qiskit.circuit import CircuitInstruction, Operation

# Qiskit quantum circuit integration
from qiskit.quantum_info import Operator
from sparse import SparseArray

DataOptions: TypeAlias = Union[ndarray, SparseArray]
QiskitOptions: TypeAlias = Union[CircuitInstruction, Operation]  # type: ignore


class Tensor:
    def __init__(
        self, data: DataOptions, indices: List[str], labels: List[str]
    ) -> None:
        """
        Constructor for the Tensor class.

        Args:
            data: The underlying data for the tensor. Internally this will always be stored as a sparse array.
            indices: Names for each dimension of the data array.
            labels: A list of labels to be associated with this tensor.
        """
        if isinstance(data, np.ndarray):
            sparse_array = sparse.COO(data)
        else:
            sparse_array = data
        self.data = sparse_array
        self.dimensions = data.shape
        self.rank = len(self.dimensions)
        self.indices = indices
        self.labels = labels

    @classmethod
    def from_array(
        cls, array: DataOptions, indices: List[str] = None, labels: List[str] = ["T1"]
    ) -> "Tensor":
        """
        Construct a tensor object from the array.

        Args:
            array: the underlying data for the tensor.
            index_prefix (optional): Default is "I".
            labels (optional): Default is "T1".

        Returns:
            tensor: The Tensor object.
        """

        data = array
        if array.size == 0:  # Check if the array is empty
            indices = []
        else:
            rank = len(array.shape)  # Use array.shape to calculate rank
            if not indices:
                index_prefix = "B"
                indices = [index_prefix + str(d_idx + 1) for d_idx in range(rank)]
            else:
                indices = indices
        labels = labels

        tensor = cls(data, indices, labels)

        return tensor

    @classmethod
    def from_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: List[str] = None,
        labels: List[str] = ["T1"],
        dagger: bool = False,
    ) -> "Tensor":
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

        tensor = cls(data, indices, labels)

        return tensor

    @classmethod
    def rank_3_copy(
        cls, indices: List[str] = ["B1", "R1", "L1"], labels: List[str] = ["T1"]
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    @classmethod
    def rank_4_copy(
        cls, indices: List[str] = ["B1", "B2", "R1", "L1"], labels: List[str] = ["T1"]
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    @classmethod
    def rank_3_copy_open(
        cls, indices: List[str] = ["B1", "R1", "L1"], labels: List[str] = ["T1"]
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    @classmethod
    def rank_4_copy_open(
        cls, indices: List[str] = ["B1", "B2", "R1", "L1"], labels: List[str] = ["T1"]
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    @classmethod
    def rank_3_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: List[str] = ["B1", "R1", "L1"],
        labels: List[str] = ["T1"],
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    @classmethod
    def rank_4_qiskit_gate(
        cls,
        gate: QiskitOptions,
        indices: List[str] = ["B1", "B2", "R1", "L1"],
        labels: List[str] = ["T1"],
    ) -> "Tensor":
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
        tensor = cls(array, indices, labels)
        return tensor

    def __str__(self) -> str:
        """
        Defines output of print.
        """
        shape = str(self.dimensions)
        indices = str(self.indices)
        return f"Tensor with shape {shape} and indices {indices}"

    def reorder_indices(self, index_order: List[str]) -> None:
        """
        Used to change the order of indices in the tensor object.

        Args:
            index_order: The desired new ordering of indices.
        """
        old_indices = list(range(len(self.indices)))
        new_indices = [index_order.index(idx) for idx in self.indices]
        new_data = sparse.moveaxis(self.data, old_indices, new_indices)
        self.data = new_data
        self.indices = index_order
        self.dimensions = new_data.shape
        return

    def new_index_name(
        self, index_prefix: str = "B", num_new_indices: int = 1
    ) -> Union[str, List[str]]:
        """
        Generate a new index name not already in use.

        Args:
            index_prefix (optional): Default is "B".
            num_new_indices (optional): Number of new names required. Default is 1.

        Returns:
            The new index name. Returned as a str if num_new_indices=1, otherwise returned as List[str].
        """
        current_indices = [x for x in self.indices if len(x) > len(index_prefix)]
        current_vals = []
        for idx in current_indices:
            if (
                idx[: len(index_prefix)] == index_prefix
                and idx[len(index_prefix) :].isdigit()
            ):
                current_vals.append(int(idx[len(index_prefix) :]))
        if len(current_vals) > 0:
            max_current_val = max(current_vals)
        else:
            max_current_val = 0
        new_indices = [
            index_prefix + str(max_current_val + i)
            for i in range(1, num_new_indices + 1)
        ]

        if num_new_indices == 1:
            return new_indices[0]
        return new_indices

    def get_dimension_of_index(self, index_name: str) -> int:
        """
        Get the dimension of an index of the tensor.

        Args:
            index_name: The name of the index.

        Returns:
            The dimension associated to index_name.
        """
        return self.dimensions[self.indices.index(index_name)]

    def get_total_dimension_of_indices(self, idxs: List[str]) -> int:
        """
        Get the total dimension of a list of indices of the tensor.

        Args:
            idxs: The names of the indices.

        Returns:
            The product of dimensions associated to each index in idxs.
        """
        dims = [self.get_dimension_of_index(idx) for idx in idxs]
        total = np.prod(dims)
        return total

    def combine_indices(self, idxs: List[str], new_index_name: str = None) -> None:
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

    def tensor_to_matrix(self, input_idxs: List[str], output_idxs: List[str]) -> None:
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
        """
        Multiply the tensor by a constant.

        Args:
            const: The constant to multiply by.
        """
        self.data = self.data * const
        return

    def dagger(self) -> None:
        """
        Get the conjugate transpose
        """
        self.data = self.data.conj()
        return

    def get_closest_unitary(
        self, input_indices: list[str], output_indices: list[str]
    ) -> "Tensor":
        """
        Perform a polar decomposition to determine the closest unitary matrix to the given tensor.

        Args:
            input_indices: The indices to treat as the input indices for the matrix representation
            output_indices: The indices to treat as the output indices for the matrix representation

        Returns:
            A new tensor that is unitary as a matrix with index ordering same as self
        """
        matrix = self.tensor_to_matrix(input_indices, output_indices)
        u, _, vh = svd(matrix, full_matrices=False)
        unitary_part = u @ vh
        input_dims = [self.get_dimension_of_index(i) for i in input_indices]
        output_dims = [self.get_dimension_of_index(i) for i in output_indices]
        shape = tuple(input_dims + output_dims)
        unitary_part = unitary_part.reshape(
            shape
        )  # This will have index ordering output_inds, input_inds
        new_t = Tensor(unitary_part, output_indices + input_indices, labels=self.labels)
        new_t.reorder_indices(self.indices)
        return new_t
