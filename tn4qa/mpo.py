import copy
from typing import List, TypeAlias, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import sparse
from numpy import ndarray

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
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
        cls, ham: dict[str, complex], max_bond: int
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
        if mpo.bond_dimension > max_bond:
            mpo.compress(max_bond)

        return mpo

    @classmethod
    def from_hamiltonian(
        cls, ham_dict: dict[str, complex], max_bond: int
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a Hamiltonian.

        Args:
            ham: The dict representation of the Hamiltonian {pauli_string : weight}.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        num_qubits = len(list(ham_dict.keys())[0])
        num_ham_terms = len(ham_dict.keys())

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

        return MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array]
        )

    @classmethod
    def from_qiskit_layer(
        cls, layer: QuantumCircuit, layer_number: int = 1
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a circuit layer.

        Args:
            layer: The quantum circuit layer (should only contain one and two qubit gates with nearest neighbour inetractions).
            layer_number (optional): The layer number within a larger circuit. Default to 1.

        Returns:
            An MPO.
        """
        tn = TensorNetwork.from_qiskit_layer(layer, layer_number)
        arrays = [0] * (layer.num_qubits)
        tensors = [t for t in tn.tensors]
        for t in tensors:
            num_qubits = int(len(t.indices) / 2)
            if num_qubits == 2:
                qidx_labels = [label for label in t.labels if label[0] == "Q"]
                q1, q2 = qidx_labels[0][1:], qidx_labels[1][1:]

                q1_indices = [t.indices[0], t.indices[2]]
                q2_indices = [t.indices[1], t.indices[3]]

                new_index_name = "TEMP_INDEX"
                new_labels = [["TEMP_LABEL_1"], ["TEMP_LABEL_2"]]
                tn.svd(
                    t,
                    q1_indices,
                    q2_indices,
                    new_index_name=new_index_name,
                    new_labels=new_labels,
                )

                tensor0 = tn.get_tensors_from_label("TEMP_LABEL_1")[0]
                tensor1 = tn.get_tensors_from_label("TEMP_LABEL_2")[0]
                tensor0.labels.remove("TEMP_LABEL_1")
                tensor1.labels.remove("TEMP_LABEL_2")
                tensor0.reorder_indices(["TEMP_INDEX"] + q1_indices)
                tensor1.reorder_indices(["TEMP_INDEX"] + q2_indices)

                tensor0_data = tensor0.data
                tensor1_data = tensor1.data

                if int(q1) == 0:
                    tensor0_shape = tensor0_data.shape
                else:
                    tensor0_shape = (1,) + tensor0_data.shape

                if int(q2) == layer.num_qubits - 1:
                    tensor1_shape = tensor1_data.shape
                else:
                    tensor1_shape = (
                        (tensor1_data.shape[0],)
                        + (1,)
                        + (tensor1_data.shape[1],)
                        + (tensor1_data.shape[2],)
                    )

                tensor0_data = sparse.reshape(tensor0_data, tensor0_shape)
                tensor1_data = sparse.reshape(tensor1_data, tensor1_shape)
                arrays[int(q1)] = tensor0_data
                arrays[int(q2)] = tensor1_data

            else:
                qidx_labels = [label for label in t.labels if label[0] == "Q"]
                qidx = qidx_labels[0][1:]

                data = t.data

                if int(qidx) == 0 or int(qidx) == layer.num_qubits - 1:
                    new_shape = (1,) + t.dimensions
                else:
                    new_shape = (
                        1,
                        1,
                    ) + t.dimensions

                data = sparse.reshape(data, new_shape)
                arrays[int(qidx)] = data

        mpo = cls.from_arrays(arrays)
        return mpo

    @classmethod
    def from_qiskit_circuit(
        cls, qc: QuantumCircuit, max_bond: int
    ) -> "MatrixProductOperator":
        """
        Create an MPO for a circuit.

        Args:
            qc: The quantum circuit.
            max_bond: The maximum bond dimension allowed.

        Returns:
            An MPO.
        """
        dag = circuit_to_dag(qc)
        all_layers = [label for label in dag.layers()]
        first_layer = all_layers[0]
        first_layer_as_circ = dag_to_circuit(first_layer["graph"])
        mpo = cls.from_qiskit_layer(first_layer_as_circ, layer_number=1)
        layer_number = 2
        for layer in all_layers[1:]:
            layer_as_circ = dag_to_circuit(layer["graph"])
            temp_mpo = cls.from_qiskit_layer(layer_as_circ, layer_number)
            mpo = mpo * temp_mpo
            if mpo.bond_dimension > max_bond:
                mpo.compress(max_bond)
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
        x_layer_mpo = cls.from_qiskit_layer(x_layer)

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
        cls, num_sites: int, ops: list[tuple]
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
        return mpo

    @classmethod
    def from_electron_integral_arrays_adder(
        cls, one_elec_integrals: ndarray, two_elec_integrals: ndarray
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

        return MatrixProductOperator.from_fermionic_operator(num_sites, ops)

    @classmethod
    def from_electron_integral_arrays(
        cls, one_elec_integrals: ndarray, two_elec_integrals: ndarray
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

        return MatrixProductOperator.from_arrays(
            [first_array] + middle_arrays + [last_array]
        )

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
        self.reshape()
        other.reshape()
        arrays = []

        t1 = self.tensors[0]
        t2 = other.tensors[0]

        t1.indices = ["T1_DOWN", "TO_CONTRACT", "T1_LEFT"]
        t2.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

        tn = TensorNetwork([t1, t2])
        tn.contract_index("TO_CONTRACT")

        tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
        tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
        tensor.reorder_indices(["DOWN", "T2_RIGHT", "T1_LEFT"])
        arrays.append(tensor.data)

        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx]
            t2 = other.tensors[t_idx]

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

        t1 = self.tensors[-1]
        t2 = other.tensors[-1]

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
        self_copy = copy.deepcopy(self)
        mpo = projector * self_copy
        mpo = mpo * projector
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
            )
            self.tensors[0].reorder_indices([bond, right_idx2, left_idx2])
            self.tensors[1].reorder_indices([bond, right_idx1, left_idx1])

            self.indices = self.get_all_indices()

            # right_idx1_pos = self.indices.index(right_idx1)
            # left_idx1_pos = self.indices.index(left_idx1)
            # right_idx2_pos = self.indices.index(right_idx2)
            # left_idx2_pos = self.indices.index(left_idx2)
            # self.indices[right_idx1_pos], self.indices[right_idx2_pos] = self.indices[right_idx2_pos], self.indices[right_idx1_pos]
            # self.indices[left_idx1_pos], self.indices[left_idx2_pos] = self.indices[left_idx2_pos], self.indices[left_idx1_pos]
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
        self.svd(self.tensors[idx - 1], input_inds, output_inds, new_index_name=bond)

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

        n = len(site_mapping)
        visited = [False] * n

        for i in range(n):
            if visited[i] or target_pos[i] == i:
                continue

            j = i
            cycle = []

            # Follow the cycle of positions
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = target_pos[j]

            # Now perform swaps to rotate elements in the cycle
            for k in range(len(cycle) - 1, 0, -1):
                self.swap_sites(cycle[0] + 1, cycle[k] + 1)

        if set_default_indices:
            self.set_default_indices()

        return

    def contract_sub_mpo(
        self, other: "MatrixProductOperator", sites: list[int]
    ) -> "MatrixProductOperator":
        """
        Contract the MPO with a smaller MPO on the given sites

        Args:
            sites: The list of sites where the smaller MPO acts

        Returns:
            An MPO that is the output of the contraction
        """

        mpo1 = copy.deepcopy(self)
        mpo2 = copy.deepcopy(other)
        mpo1.set_default_indices()
        mpo2.set_default_indices()

        target_site_ordering = copy.deepcopy(sites)
        for idx in range(1, self.num_sites + 1):
            if idx not in sites:
                target_site_ordering.append(idx)
        restore_ordering = []
        for idx in range(1, self.num_sites + 1):
            restore_ordering.append(target_site_ordering.index(idx) + 1)
        mpo1.reorder_sites(target_site_ordering, set_default_indices=True)

        for tidx in range(mpo2.num_sites):
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
        for n in range(len(sites)):
            tn.contract_index(f"D{n+1}")
        for n in range(len(sites) - 1):
            tn.combine_indices([f"B{n+1}", f"B{n+1}_"], new_index_name=f"B{n+1}")
        tn.tensors[0].reorder_indices([f"B{n+1}", f"R{n+1}_", f"L{n+1}"])
        for n in range(1, len(sites)):
            tn.tensors[n].reorder_indices([f"B{n}", f"B{n+1}", f"R{n+1}_", f"L{n+1}"])

        arrays = [t.data for t in tn.tensors]
        mpo = MatrixProductOperator.from_arrays(arrays)
        mpo.reorder_sites(restore_ordering, set_default_indices=True)
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
        remaining_sites = list(range(1, self.num_sites + 1))
        for site in sites:
            remaining_sites.remove(site)
        mpo.reorder_sites(sites + remaining_sites, set_default_indices=True)

        for idx in range(num_sites_to_trace):
            current_indices = mpo.tensors[idx].indices
            mpo.tensors[idx].indices = [
                "R" + x[1:] if x[0] == "L" else x for x in current_indices
            ]

        if not matrix:
            for idx in range(num_sites_to_trace):
                mpo.contract_index("R" + str(idx + 1))
                mpo.contract_index("B" + str(idx + 1))
            if set_default_indices:
                mpo.set_default_indices()
            return mpo
        else:
            result = mpo.contract_entire_network()
            output_inds = [
                f"R{x}" for x in list(range(num_sites_to_trace + 1, self.num_sites + 1))
            ]
            input_inds = [
                f"L{x}" for x in list(range(num_sites_to_trace + 1, self.num_sites + 1))
            ]
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
