from typing import List, Union

# Contraction path finding is offloaded to Cotengra
import cotengra as ctg
import numpy as np

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import sparse
from numpy.linalg import svd

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from scipy.sparse.linalg import svds

from .tensor import Tensor

# Visualisation
from .visualisation import draw_arbitrary_tn, draw_quantum_circuit


class TensorNetwork:
    def __init__(
        self, tensors: List[Tensor], name: str = "TN", count_from: int = 1
    ) -> None:
        """
        Constructor for the TensorNetwork class.

        Args:
            tensors: A list of Tensor objects.
            name (optional): A name for the tensor network.

        Returns:
            A tensor network.
        """
        self.name = name
        i = count_from
        for t in tensors:
            t.labels.append(f"TN_T{i}")
            i += 1
        self.tensors = tensors
        self.indices = self.get_all_indices()

    def __str__(self) -> str:
        """
        Defines output of print.
        """
        output = "Tensor Network containing: \n"
        for t in self.tensors:
            shape = str(t.dimensions)
            indices = str(t.indices)
            output += f"Tensor with shape {shape} and indices {indices} \n"
        return output

    def __add__(self, other: "TensorNetwork") -> "TensorNetwork":
        """
        Defines addition for tensor networks.
        """
        current_tensor_number = len(self.tensors)
        self.tensors = self.tensors + other.tensors
        i = current_tensor_number
        for t in self.tensors[current_tensor_number:]:
            t.labels.append(f"TN_T{i}")
            i += 1
        self.indices = self.get_all_indices()
        return self

    @classmethod
    def from_qiskit_circuit(
        cls, qc: QuantumCircuit, dagger: bool = False
    ) -> "TensorNetwork":
        """
        Construct a tensor network from a Qiskit QuantumCircuit object.

        Args:
            qc: The QuantumCircuit.
            dagger: If True will construct a TN for the inverse circuit

        Returns:
            A tensor network.
        """
        num_qubits = qc.num_qubits
        index_prefixes = [f"QW{x}" for x in range(1, num_qubits + 1)]
        wire_counts = {str(x): 0 for x in range(num_qubits)}
        tensors = []

        data = qc.data
        if dagger:
            data = data[::-1]

        tensor_number = 1
        for inst in data:
            inst_num_qubits = inst.operation.num_qubits
            qidxs = [inst.qubits[i]._index for i in range(inst_num_qubits)]
            indices = [0] * (2 * len(qidxs))
            labels = [f"T{tensor_number}"]
            for qidx in qidxs:
                qw = index_prefixes[qidx]
                wire_in = str(wire_counts[str(qidx)])
                wire_out = str(wire_counts[str(qidx)] + 1)
                indices[qidxs.index(qidx)] = qw + "N" + wire_out
                indices[len(qidxs) + qidxs.index(qidx)] = qw + "N" + wire_in
                labels.append(f"Q{qidx}")
                wire_counts[str(qidx)] += 1
            inst_tensor = Tensor.from_qiskit_gate(inst, indices, labels, dagger)
            tensors.append(inst_tensor)
            tensor_number += 1

        tn = TensorNetwork(tensors, name="QuantumCircuit")
        return tn

    def get_index_to_tensor_dict(self) -> dict:
        """
        Build a dictionary mapping indices to their tensors.

        Returns:
            A dictionary of the form {idx : [tensor1,...]}
        """
        tn_dict = {}
        for t in self.tensors:
            for idx in t.indices:
                if idx in tn_dict:
                    tn_dict[idx].append(t)
                else:
                    tn_dict[idx] = [t]
        return tn_dict

    def get_label_to_tensor_dict(self) -> dict:
        """
        Build a dictionary mapping labels to their tensors.

        Returns:
            A dictionary of the form {label : [tensor1,...]}
        """
        tn_dict = {}
        for t in self.tensors:
            for label in t.labels:
                if label in tn_dict:
                    tn_dict[label].append(t)
                else:
                    tn_dict[label] = [t]
        return tn_dict

    def get_dimension_of_index(self, idx: str) -> int:
        """
        Get the dimension of an index.

        Args:
            idx: The index name.

        Returns:
            The dimension of idx.
        """
        for t in self.tensors:
            if idx in t.indices:
                return t.dimensions[t.indices.index(idx)]
        raise ValueError

    def get_internal_indices(self) -> List[str]:
        """
        Get the internal indices of the tensor network.

        Returns:
            Indices that are connected to 2 tensors in the network.
        """
        tn_dict = self.get_index_to_tensor_dict()
        internal_bonds = [idx for idx in tn_dict.keys() if len(tn_dict[idx]) == 2]
        return internal_bonds

    def get_external_indices(self) -> List[str]:
        """
        Get the external bonds of the tensor network.

        Returns:
            Indices that are connected to 1 tensor in the network.
        """
        tn_dict = self.get_index_to_tensor_dict()
        external_bonds = [idx for idx in tn_dict.keys() if len(tn_dict[idx]) == 1]
        return external_bonds

    def get_all_indices(self) -> List[str]:
        """
        Get all indices in the tensor network.

        Returns:
            A list of all index names.
        """
        dict = self.get_index_to_tensor_dict()
        return list(dict.keys())

    def get_all_labels(self) -> List[str]:
        """
        Get all labels in the tensor network.

        Returns:
            A list of all label names.
        """
        all_labels = []
        for t in self.tensors:
            for l in t.labels:
                all_labels.append(l)
        return list(set(all_labels))

    def get_new_label(self, tensor_prefix: str = "TN_T") -> str:
        """
        Get a new tensor label with the specified prefix.

        Args:
            tensor_prefix (optional): Defaults to "TN_T".

        Returns:
            A new label that doesn't already appear in the network starting with tensor_prefix.
        """
        all_labels = [x for x in self.get_all_labels() if len(x) > len(tensor_prefix)]

        current_vals = []
        for label in all_labels:
            if (
                label[: len(tensor_prefix)] == tensor_prefix
                and label[len(tensor_prefix) :].isdigit()
            ):
                current_vals.append(int(label[len(tensor_prefix) :]))
        if len(current_vals) > 0:
            max_current_val = max(current_vals)
        else:
            max_current_val = 0
        new_label = tensor_prefix + str(max_current_val + 1)

        return new_label

    def get_tensors_from_index_name(self, idx: str) -> List[Tensor]:
        """
        Get all tensors connected to a given index.

        Args:
            idx: The index name.

        Returns
            A list of tensors with idx as one of their indices.
        """
        return [t for t in self.tensors if idx in t.indices]

    def get_tensors_from_label(self, label: str) -> List[Tensor]:
        """
        Get all tensors connected to a given label.

        Args:
            label: The label name.

        Returns
            A list of tensors with label as one of their labels.
        """
        return [t for t in self.tensors if label in t.labels]

    def contract_index(self, idx: str) -> None:
        """
        Contract an index in the tensor network.

        Args:
            idx: The name of the index to contract.
        """
        tensors = self.get_tensors_from_index_name(idx)

        array0, array1 = tensors[0].data, tensors[1].data
        indices0, indices1 = tensors[0].indices, tensors[1].indices

        output_indices = [i for i in indices0 if i != idx] + [
            i for i in indices1 if i != idx
        ]
        new_data = ctg.array_contract(
            arrays=[array0, array1],
            inputs=[indices0, indices1],
            output=output_indices,
            cache_expression=True,
            prefer_einsum=True,
        )
        new_labels = [self.get_new_label()]
        if len(new_data.shape) > len(output_indices):
            new_data = new_data.reshape(new_data.shape[1:])
        new_tensor = Tensor(new_data, output_indices, new_labels)

        pos = self.tensors.index(tensors[0])
        self.tensors.remove(tensors[0])
        self.tensors.remove(tensors[1])
        self.indices.remove(idx)
        self.add_tensor(new_tensor, position=pos)

        return

    def compress_index(
        self,
        idx: str,
        max_bond: int,
        reverse_direction: bool = False,
        tol: float = 1e-12,
    ) -> None:
        """
        Compress a given index using SVD.

        Args:
            idx: The index to compress.
            max_bond: The maximum bond dimension for this index.
            reverse_direction: If True the second tensor will become an isometry
            tol: Tolerance for discarding singular values
        """
        if reverse_direction:
            tensors = self.get_tensors_from_index_name(idx)[::-1]
        else:
            tensors = self.get_tensors_from_index_name(idx)

        array0, array1 = tensors[0].data, tensors[1].data
        indices0, indices1 = tensors[0].indices, tensors[1].indices
        dims0, dims1 = tensors[0].dimensions, tensors[1].dimensions

        output_indices = [i for i in indices0 if i != idx] + [
            i for i in indices1 if i != idx
        ]

        new_data = ctg.array_contract(
            arrays=[array0, array1],
            inputs=[indices0, indices1],
            output=output_indices,
            cache_expression=True,
            prefer_einsum=True,
        )
        temp_tensor = Tensor(new_data, output_indices, ["TEMP"])

        input_idxs = [i for i in indices0 if i != idx]
        output_idxs = [i for i in indices1 if i != idx]
        temp_tensor.tensor_to_matrix(input_idxs, output_idxs)

        bond_dim = min([max_bond, temp_tensor.data.shape[0], temp_tensor.data.shape[1]])
        if bond_dim >= min([temp_tensor.data.shape[0], temp_tensor.data.shape[1]]) - 1:
            u, s, vh = svd(temp_tensor.data.todense(), full_matrices=False)
        else:
            u, s, vh = svds(temp_tensor.data, k=bond_dim)

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

        idx_pos0 = indices0.index(idx)
        idx_pos1 = indices1.index(idx)
        new_dims0 = (keep_dim,) + dims0[:idx_pos0] + dims0[idx_pos0 + 1 :]
        new_dims1 = dims1[:idx_pos1] + dims1[idx_pos1 + 1 :] + (keep_dim,)
        new_indices0 = [idx] + indices0[:idx_pos0] + indices0[idx_pos0 + 1 :]
        new_indices1 = indices1[:idx_pos1] + indices1[idx_pos1 + 1 :] + [idx]

        new_data0 = new_data0.reshape(new_dims0)
        new_data1 = new_data1.reshape(new_dims1)

        tensors[0].data = new_data0
        tensors[1].data = new_data1
        tensors[0].indices = new_indices0
        tensors[1].indices = new_indices1
        tensors[0].dimensions = new_dims0
        tensors[1].dimensions = new_dims1

        tensors[0].reorder_indices(indices0)
        tensors[1].reorder_indices(indices1)

        return

    def pop_tensors_by_label(self, labels: List[str]) -> List[Tensor]:
        """
        Remove tensors from the network given a set of labels.

        Args:
            labels: The list of labels to search for.

        Returns:
            The (possibly empty) list of removed tensors.
        """
        tensors = []
        for tensor in self.tensors:
            has_all_labels = True
            for label in labels:
                if label not in tensor.labels:
                    has_all_labels = False
            if has_all_labels:
                tensors.append(tensor)

        tn_dict = self.get_index_to_tensor_dict()
        for t in tensors:
            self.tensors.remove(t)

        for idx in tn_dict:
            still_exists = False
            if len(tn_dict[idx]) > 0:
                still_exists = True
            if not still_exists:
                self.indices.remove(idx)

        return tensors

    def add_tensor(
        self, tensor: Tensor, position: int = None, add_label: bool = False
    ) -> None:
        """
        Add a tensor to the network.

        Args:
            tensor: The tensor to add.
        """
        if add_label:
            unique_label = self.get_new_label("TN_T")
            tensor.labels.append(unique_label)
        if position is None:
            self.tensors.append(tensor)
        else:
            self.tensors.insert(position, tensor)
        for idx in tensor.indices:
            if idx not in self.indices:
                self.indices.append(idx)
        return

    def contract_entire_network(self) -> Union[Tensor, complex]:
        """
        Contracts all internal indices in the network.

        Returns:
            A tensor whose indices were the external indices of the network, or a float if there were no external indices.
        """
        output_indices = self.get_external_indices()
        output_labels = [self.get_new_label("TN_T")]
        arrays = [t.data for t in self.tensors]
        input_indices = [t.indices for t in self.tensors]

        output_tensor_data = ctg.array_contract(
            arrays=arrays,
            inputs=input_indices,
            output=output_indices,
            cache_expression=True,
            prefer_einsum=True,
        )
        if len(output_indices) == 0:
            return complex(output_tensor_data.flatten()[0])
        else:
            output_tensor = Tensor(output_tensor_data, output_indices, output_labels)
            return output_tensor

    def compute_environment_tensor_by_label(
        self, labels: List[str], replace_tensor: bool = False
    ) -> Union[Tensor, None]:
        """
        Compute the environment of a tensor in the network given a set of labels.

        Args:
            labels: The labels to look for.
            replace_tensor (optional): When True replaces the original tensor in the network by its environment. Default is False.

        Returns:
            If replace_tensor is True, return type is None. Otherwise, returns the environment tensor.
        """
        popped_tensor = self.pop_tensors_by_label(labels)[0]
        output_tensor = self.contract_entire_network()

        if replace_tensor:
            self = TensorNetwork([output_tensor])
        else:
            self.add_tensor(popped_tensor)
            return output_tensor

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

    def combine_indices(
        self, idxs: List[str], new_index_name: str | None = None
    ) -> None:
        """
        Combine two or more indices within the network. Only valid when all indices are between the same two tensors.

        Args:
            idxs: The indices to combine.
            new_index_name (optional): What to call the resulting combined index.
        """
        tensors = self.get_tensors_from_index_name(idxs[0])
        for t in tensors:
            t.combine_indices(idxs, new_index_name)
        self.indices = self.get_all_indices()
        return

    def svd(
        self,
        tensor: Tensor,
        input_indices: List[str],
        output_indices: List[str],
        max_bond: int | None = None,
        new_index_name: str | None = None,
        new_labels: List[List[str]] | None = None,
        tol: float | None = 1e-12,
    ) -> None:
        """
        Perform an SVD on a tensor.

        Args:
            tensor: The tensor.
            input_indices: Indices to be treated as one side of SVD.
            output_indices: Indices to be treated as other side of SVD.
            max_bond: The maximum bond dimension allwoed.
            new_index_name (optional): What to call the resulting new index.
            tol: Tolerance for keeping singular values
        """
        original_position = self.tensors.index(tensor)
        original_labels = tensor.labels
        original_input_dims = [tensor.get_dimension_of_index(x) for x in input_indices]
        original_output_dims = [
            tensor.get_dimension_of_index(x) for x in output_indices
        ]
        tensor.tensor_to_matrix(input_indices, output_indices)
        if not max_bond:
            max_bond = min([tensor.dimensions[0], tensor.dimensions[1]])
        else:
            max_bond = min([max_bond, tensor.dimensions[0], tensor.dimensions[1]])

        if max_bond >= min([tensor.dimensions[0], tensor.dimensions[1]]) - 1:
            u, s, vh = svd(tensor.data.todense(), full_matrices=False)
        else:
            u, s, vh = svds(tensor.data, k=max_bond)

        s = s[s > 1e-14]
        sq = s**2
        cumulative = np.cumsum(sq[::-1])[::-1]
        keep_dim = len(s)
        for k in range(len(s)):
            if cumulative[k] < tol**2:
                keep_dim = k + 1
                break
        keep_dim = min(keep_dim, max_bond)

        threshold = 1e-14
        data0 = vh[:keep_dim, :]
        data0[np.abs(data0) < threshold] = 0.0
        data1 = u[:, :keep_dim] * s[:keep_dim]
        data1[np.abs(data1) < threshold] = 0.0

        tensor0_data = sparse.COO.from_numpy(data0)
        tensor1_data = sparse.COO.from_numpy(data1)

        if not new_index_name:
            new_index_name = self.new_index_name()

        tensor0_labels = [self.get_new_label("TN_T")]
        tensor1_labels = [self.get_new_label("TN_T")]
        if new_labels:
            tensor0_labels = tensor0_labels + new_labels[0]
            tensor1_labels = tensor1_labels + new_labels[1]

        tensor0_indices = [new_index_name] + input_indices
        tensor1_indices = output_indices + [new_index_name]

        tensor0_dims = [keep_dim] + original_input_dims
        tensor1_dims = original_output_dims + [keep_dim]

        tensor0_data = sparse.reshape(tensor0_data, tensor0_dims)
        tensor1_data = sparse.reshape(tensor1_data, tensor1_dims)

        tensor0 = Tensor(tensor0_data, tensor0_indices, tensor0_labels)
        tensor1 = Tensor(tensor1_data, tensor1_indices, tensor1_labels)

        self.pop_tensors_by_label(original_labels)
        self.add_tensor(tensor0, original_position)
        self.add_tensor(tensor1, original_position + 1)
        return

    def compress(self, max_bond: int) -> None:
        """
        Compress the tensor network using SVD.

        Args:
            max_bond: The maximum bond dimension allowed.
        """
        internal_indices = self.get_internal_indices()
        for index in internal_indices:
            self.compress_index(index, max_bond)
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
        if self.name == "QuantumCircuit":
            draw_quantum_circuit(self.tensors, node_size, x_len, y_len)

        else:
            draw_arbitrary_tn(self.tensors)
