import copy
from collections import Counter
from typing import List, TypeAlias, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import sparse
from numpy import ndarray

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from sparse import SparseArray
from symmer import QuantumState

from .mpo import MatrixProductOperator
from .tensor import Tensor
from .tn import TensorNetwork

# Visualisation
from .visualisation import draw_mps

DataOptions: TypeAlias = Union[ndarray, SparseArray]


class MatrixProductState(TensorNetwork):
    def __init__(self, tensors: List[Tensor], shape: str = "udp") -> None:
        """
        Constructor for MatrixProductState class.

        Args:
            tensors: List of tensors to form the MPS.
            shape (optional): The order of the indices for the tensors. Default is 'udp' (up, down, physical)

        Returns
            An MPS.
        """
        if len(tensors) == 1:
            self.name = "MPS"
            self.tensors = tensors
            self.indices = tensors[0].indices
            self.num_sites = 1
            self.shape = shape
            self.internal_inds = []
            self.external_inds = [tensors[0].indices]
            self.bond_dims = []
            self.physical_dims = [tensors[0].dimensions[0]]
            self.bond_dimension = None
            self.physical_dimension = self.physical_dims[0]
        else:
            super().__init__(tensors, "MPS")
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
        cls, arrays: List[DataOptions], shape: str = "udp"
    ) -> "MatrixProductState":
        """
        Create an MPS from a list of arrays.

        Args:
            arrays: The list of arrays.
            shape (optional): The order of the indices for the tensors. Default is 'udp' (up, down, physical)

        Returns:
            An MPS.
        """
        if len(arrays) == 1:
            idx = "P1"
            tensor = Tensor(arrays[0], [idx], ["MPS_T1"])
            return cls([tensor], shape)
        tensors = []

        first_shape = shape.replace("u", "")
        physical_idx_pos = first_shape.index("p")
        virtual_input_idx_pos = first_shape.index("d")
        first_indices = ["", ""]
        first_indices[physical_idx_pos] = "P1"
        first_indices[virtual_input_idx_pos] = "B1"
        first_tensor = Tensor(arrays[0], first_indices, ["MPS_T1"])
        tensors.append(first_tensor)

        physical_idx_pos = shape.index("p")
        virtual_output_idx_pos = shape.index("u")
        virtual_input_idx_pos = shape.index("d")
        for a_idx in range(1, len(arrays) - 1):
            a = arrays[a_idx]
            indices_k = ["", "", ""]
            indices_k[physical_idx_pos] = f"P{a_idx+1}"
            indices_k[virtual_output_idx_pos] = f"B{a_idx}"
            indices_k[virtual_input_idx_pos] = f"B{a_idx+1}"
            tensor_k = Tensor(a, indices_k, [f"MPS_T{a_idx+1}"])
            tensors.append(tensor_k)

        last_shape = shape.replace("d", "")
        physical_idx_pos = last_shape.index("p")
        virtual_output_idx_pos = last_shape.index("u")
        last_indices = ["", ""]
        last_indices[physical_idx_pos] = f"P{len(arrays)}"
        last_indices[virtual_output_idx_pos] = f"B{len(arrays)-1}"
        last_tensor = Tensor(arrays[-1], last_indices, [f"MPS_T{len(arrays)}"])
        tensors.append(last_tensor)

        mps = cls(tensors, shape)
        mps.reshape()
        return mps

    @classmethod
    def from_bitstring(cls, bitstring: str) -> "MatrixProductState":
        """
        Create an MPS for the given bitstring |b>

        Args:
            bitstring: The computational basis state to be prepared.

        Returns:
            An MPS.
        """
        zero = np.array([1, 0], dtype=complex)
        one = np.array([0, 1], dtype=complex)

        if len(bitstring) == 1:
            arrays = []
            if bitstring == "0":
                arrays.append(zero.reshape(2))
            else:
                arrays.append(one.reshape(2))
            return cls.from_arrays(arrays)

        arrays = []
        if bitstring[0] == "0":
            arrays.append(zero.reshape((1, 2)))
        else:
            arrays.append(one.reshape((1, 2)))

        for bit in bitstring[1:-1]:
            if bit == "0":
                arrays.append(zero.reshape((1, 1, 2)))
            else:
                arrays.append(one.reshape((1, 1, 2)))
        if bitstring[-1] == "0":
            arrays.append(zero.reshape((1, 2)))
        else:
            arrays.append(one.reshape((1, 2)))

        return cls.from_arrays(arrays)

    @classmethod
    def all_zero_mps(cls, num_sites: int) -> "MatrixProductState":
        """
        Create an MPS for the all zero state |000...0>

        Args:
            num_sites: The number of sites for the MPS

        Returns:
            An MPS.
        """
        return cls.from_bitstring("0" * num_sites)

    @classmethod
    def from_hf_state(cls, num_spin_orbs: int, num_electrons: int):
        """
        Create an MPS for the HF state. Currently only valid for fermionic systems and JW encoded qubit systems.
        This is because the HF state is assumed to be |111000...0>.

        Args:
            num_spin_orbs: The number of spin orbitals in the system.
            num_electrons: The number of electrons in the system.

        Returns:
            A MPS.
        """
        bitstring = "1" * num_electrons + "0" * (num_spin_orbs - num_electrons)
        return cls.from_bitstring(bitstring)

    @classmethod
    def from_symmer_quantumstate(
        cls, quantum_state: QuantumState, max_bond: int | None = None
    ):
        """
        Create an MPS from a Symmer QuantumState object.

        Args:
            quantum_state: The quantum state.

        Returns:
            An MPS.
        """
        state_dict = quantum_state.to_dictionary
        bitstrings = list(state_dict.keys())
        weights = list(state_dict.values())
        mps = MatrixProductState.from_bitstring(bitstrings[0])
        mps.multiply_by_constant(weights[0])
        for idx in range(1, len(bitstrings)):
            temp_mps = MatrixProductState.from_bitstring(bitstrings[idx])
            temp_mps.multiply_by_constant(weights[idx])
            mps = mps + temp_mps
            if max_bond:
                if mps.bond_dimension > max_bond:
                    mps.compress(max_bond)

        return mps

    @classmethod
    def random_mps(
        cls, num_sites: int, bond_dim: int, physical_dim: int
    ) -> "MatrixProductState":
        """
        Create a random MPS.

        Args:
            num_sites: The number of sites for the MPS.
            bond_dim: The internal bond dimension to use.
            physical_dim: The physical dimension to use.

        Returns:
            An MPS.
        """
        if num_sites == 1:
            array = np.random.rand(physical_dim)
            return cls.from_arrays([array], shape="udp")

        arrays = []
        first_array = np.random.rand(bond_dim, physical_dim)
        arrays.append(first_array)

        for _ in range(1, num_sites - 1):
            array = np.random.rand(bond_dim, bond_dim, physical_dim)
            arrays.append(array)

        last_array = np.random.rand(bond_dim, physical_dim)
        arrays.append(last_array)

        return cls.from_arrays(arrays, shape="udp")

    @classmethod
    def random_quantum_state_mps(
        cls, num_sites: int, bond_dim: int, physical_dim: int = 2
    ) -> "MatrixProductState":
        """
        Create a random MPS corresponding to a valid quantum state.

        Args:
            num_sites: The number of sites for the MPS.
            bond_dim: The internal bond dimension to use.
            physical_dim (optional): The physical dimension to use. Default is 2 (for qubits).

        Returns:
            An MPS.
        """
        mps = cls.random_mps(num_sites, bond_dim, physical_dim)
        mps.normalise()
        return mps

    @classmethod
    def equal_superposition_mps(cls, num_sites: int) -> "MatrixProductState":
        """
        Create an MPS for the equal superposition state |+++...+>

        Args:
            num_sites: The number of sites for the MPS.

        Returns:
            An MPS.
        """
        if num_sites == 1:
            h = np.array([np.sqrt(1 / 2), np.sqrt(1 / 2)], dtype=complex).reshape(2)
            return cls.from_arrays([h], shape="udp")

        h_end = np.array([np.sqrt(1 / 2), np.sqrt(1 / 2)], dtype=complex).reshape(1, 2)
        h_middle = np.array([np.sqrt(1 / 2), np.sqrt(1 / 2)], dtype=complex).reshape(
            1, 1, 2
        )
        arrays = [h_end] + [h_middle] * (num_sites - 2) + [h_end]
        return cls.from_arrays(arrays, shape="udp")

    @classmethod
    def from_qiskit_circuit(
        cls,
        qc: QuantumCircuit,
        max_bond: int | None = None,
        input_mps: "MatrixProductState" = None,
    ) -> "MatrixProductState":
        """
        Create an MPS for the output of a Qiskit QuantumCircuit.

        Args:
            qc: The QuantumCircuit object.
            max_bond: The maximum bond dimension to allow.
            input (optional): The input MPS. Default is the all zero MPS.

        Returns:
            An MPS.
        """
        qc_mpo = MatrixProductOperator.from_qiskit_circuit(qc, max_bond)
        if not input_mps:
            mps = cls.all_zero_mps(qc.num_qubits)
        else:
            mps = input_mps
        mps = mps.apply_mpo(qc_mpo, max_bond)
        return mps

    @classmethod
    def from_sparse_array(
        cls, array: SparseArray, max_bond: int | None = None
    ) -> "MatrixProductState":
        """
        Create an MPS from a sparse array

        Args:
            array: The array
            max_bond: Maximum bond dimension

        Returns:
            MPS
        """
        dense_array = array.todense()
        return cls.from_dense_array(dense_array, max_bond)

    @classmethod
    def from_dense_array(
        cls, array: ndarray, max_bond: int | None = None
    ) -> "MatrixProductState":
        """
        Create an MPS from a dense array

        Args:
            array: The array
            max_bond: Maximum bond dimension

        Returns:
            MPS
        """
        num_qubits = int(np.log2(len(array)))
        array = array.reshape((2,) * (num_qubits))
        indices = [f"P{x}" for x in range(1, num_qubits + 1)]
        tensor = Tensor(array, indices, ["MPS"])
        tn = TensorNetwork([tensor])

        for idx in range(num_qubits - 1):
            t = tn.tensors[idx]
            input_inds = [indices[idx]]
            output_inds = indices[idx + 1 : num_qubits]
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
                new_idx_order1 = [f"C{idx+1}", "P1"]
            else:
                new_idx_order1 = [
                    f"C{idx}",
                    f"C{idx+1}",
                    f"P{idx+1}",
                ]
            new_idx_order2 = [f"C{idx+1}"] + output_inds
            tn.tensors[idx].reorder_indices(new_idx_order1)
            tn.tensors[idx + 1].reorder_indices(new_idx_order2)
        arrays = [tn.tensors[i].data for i in range(num_qubits)]
        mps = cls.from_arrays(arrays)
        if max_bond:
            if mps.bond_dimension > max_bond:
                mps.compress(max_bond)
        return mps

    def __add__(self, other: "MatrixProductState") -> "MatrixProductState":
        """
        Defines MPS addition.
        """
        self.reshape()
        other.reshape()
        arrays = []

        t1 = self.tensors[0]
        t2 = other.tensors[0]

        t1_data = t1.data
        t2_data = t2.data
        t1_data = sparse.reshape(t1_data, (1, t1.dimensions[0], t1.dimensions[1]))
        t2_data = sparse.reshape(t2_data, (1, t2.dimensions[0], t2.dimensions[1]))
        t1_dimensions = (1, t1.dimensions[0], t1.dimensions[1])
        t2_dimensions = (1, t2.dimensions[0], t2.dimensions[1])

        data1 = sparse.reshape(
            t1_data, (t1_dimensions[0] * t1_dimensions[2], t1_dimensions[1])
        )
        data2 = sparse.reshape(
            t2_data, (t2_dimensions[0] * t2_dimensions[2], t2_dimensions[1])
        )

        new_data = sparse.concatenate([data1, data2], axis=1)
        new_data = sparse.moveaxis(new_data, [0, 1], [1, 0])
        arrays.append(new_data)

        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx]
            t2 = other.tensors[t_idx]

            t1_data = t1.data
            t2_data = t2.data
            t1_dimensions = t1.dimensions
            t2_dimensions = t2.dimensions

            data1 = sparse.moveaxis(t1_data, [0, 1, 2], [0, 2, 1])
            data2 = sparse.moveaxis(t2_data, [0, 1, 2], [0, 2, 1])

            data1 = sparse.reshape(
                data1, (t1_dimensions[0] * t1_dimensions[2], t1_dimensions[1])
            )
            data2 = sparse.reshape(
                data2, (t2_dimensions[0] * t2_dimensions[2], t2_dimensions[1])
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
                    t1_dimensions[1] + t2_dimensions[1],
                    t1_dimensions[2],
                ),
            )

            arrays.append(new_data)

        t1 = self.tensors[-1]
        t2 = other.tensors[-1]

        t1_data = t1.data
        t2_data = t2.data
        t1_data = sparse.reshape(t1_data, (t1.dimensions[0], 1, t1.dimensions[1]))
        t2_data = sparse.reshape(t2_data, (t2.dimensions[0], 1, t2.dimensions[1]))
        t1_dimensions = (t1.dimensions[0], 1, t1.dimensions[1])
        t2_dimensions = (t2.dimensions[0], 1, t2.dimensions[1])

        data1 = sparse.reshape(
            t1_data, (t1_dimensions[0] * t1_dimensions[2], t1_dimensions[1])
        )
        data2 = sparse.reshape(
            t2_data, (t2_dimensions[0] * t2_dimensions[2], t2_dimensions[1])
        )

        new_data = sparse.concatenate([data1, data2], axis=1)
        new_data = sparse.moveaxis(new_data, [0, 1], [1, 0])
        arrays.append(new_data)

        output = MatrixProductState.from_arrays(arrays)
        return output

    def __sub__(self, other: "MatrixProductState") -> "MatrixProductState":
        """
        Defines MPS subtraction.
        """
        other.multiply_by_constant(-1.0)
        output = self + other
        return output

    def to_sparse_array(self) -> SparseArray:
        """
        Convert the MPS to a sparse array.
        """
        mps = copy.deepcopy(self)
        output = mps.contract_entire_network()
        output.combine_indices(output.indices, output.indices[0])
        return output.data

    def to_dense_array(self) -> ndarray:
        """
        Convert the MPS to a dense array.
        """
        mps = copy.deepcopy(self)
        sparse_array = mps.to_sparse_array()
        dense_array = sparse_array.todense()
        return dense_array

    def reshape(self, shape: str = "udp") -> None:
        """
        Reshape the tensors in the MPS.

        Args:
            shape (optional): Default is 'udp' (up, down, physical) but any order is allowed.
        """
        if len(self.tensors) == 1:
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

    def multiply_by_constant(self, const: complex) -> None:
        """
        Scale the MPS by a constant.

        Args:
            const: The constant to multiply by.
        """
        first_tensor = self.tensors[0]
        first_tensor.multiply_by_constant(const)
        return

    def dagger(self) -> None:
        """
        Take the conjugate transpose of the MPS. Leaves indices unchanged.
        """
        for t in self.tensors:
            t.data = sparse.COO.conj(t.data)
        return

    def move_orthogonality_centre(self, where: int = None, current: int = None) -> None:
        """
        Move the orthogonality centre of the MPS.

        Args:
            where (optional): Defaults to the last tensor.
            current (optional): Where the orthogonality centre is currently (if known)
        """
        if not where:
            where = self.num_sites

        internal_indices = self.get_internal_indices()

        if current == where:
            return

        if not current:
            push_down = list(range(1, where))
            push_up = list(range(where, self.num_sites))[::-1]
        elif current < where:
            push_down = list(range(current, where))
            push_up = []
        else:
            push_down = []
            push_up = list(range(where, current))[::-1]

        max_bond = self.bond_dimension

        for idx in push_down:
            index = internal_indices[idx - 1]
            self.compress_index(index, max_bond)

        for idx in push_up:
            index = internal_indices[idx - 1]
            self.compress_index(index, max_bond, reverse_direction=True)

        return

    def apply_sub_mpo(
        self, mpo: MatrixProductOperator, sites: list[int], max_bond: int | None = None
    ) -> "MatrixProductState":
        """
        Apply a smaller MPO to the MPS

        Args:
            mpo: The MPO to apply.
            sites: The list of site indices where the MPO will apply.

        Returns:
            The new MPS.
        """
        mps = copy.deepcopy(self)
        mpo = copy.deepcopy(mpo)
        mps.set_default_indices()
        mpo.set_default_indices()

        target_site_ordering = copy.deepcopy(sites)
        for idx in range(1, self.num_sites + 1):
            if idx not in sites:
                target_site_ordering.append(idx)
        restore_ordering = []
        for idx in range(1, self.num_sites + 1):
            restore_ordering.append(target_site_ordering.index(idx) + 1)
        mps.reorder_sites(target_site_ordering, set_default_indices=True)

        if len(sites) == mps.num_sites:
            mps = self.apply_mpo(mpo, max_bond)
        else:
            for tidx in range(mpo.num_sites):
                t1 = mps.tensors[tidx]
                t2 = mpo.tensors[tidx]
                t1_current_indices = t1.indices
                t1.indices = [
                    f"D{tidx+1}" if x[0] == "P" else x for x in t1_current_indices
                ]
                t2_current_indices = t2.indices
                t2.indices = [
                    f"D{tidx+1}" if x[0] == "L" else x + "_" for x in t2_current_indices
                ]

            all_tensors = mps.tensors + mpo.tensors

            tn = TensorNetwork(all_tensors, "TotalTN")
            for n in range(len(sites)):
                tn.contract_index(f"D{n+1}")
            for n in range(len(sites) - 1):
                tn.combine_indices([f"B{n+1}", f"B{n+1}_"], new_index_name=f"B{n+1}")
            tn.tensors[0].reorder_indices(["B1", "R1_"])
            for n in range(1, len(sites)):
                tn.tensors[n].reorder_indices([f"B{n}", f"B{n+1}", f"R{n+1}_"])

            arrays = [t.data for t in tn.tensors]
            mps = MatrixProductState.from_arrays(arrays)

        mps.reorder_sites(restore_ordering, set_default_indices=True)

        if max_bond:
            if mps.bond_dimension > max_bond:
                mps.compress(max_bond)

        return mps

    def apply_mpo(
        self, mpo: MatrixProductOperator, max_bond: int | None = None
    ) -> "MatrixProductState":
        """
        Apply a MPO to the MPS.

        Args:
            mpo: The MPO to apply.

        Returns:
            The new MPS.
        """
        self.reshape()
        mpo.reshape()
        arrays = []

        t1 = self.tensors[0]
        t2 = mpo.tensors[0]

        t1.indices = ["T1_DOWN", "TO_CONTRACT"]
        t2.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

        tn = TensorNetwork([t1, t2])
        tn.contract_index("TO_CONTRACT")

        tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
        tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
        tensor.reorder_indices(["DOWN", "T2_RIGHT"])
        arrays.append(tensor.data)

        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx]
            t2 = mpo.tensors[t_idx]

            t1.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT"]
            t2.indices = ["T2_UP", "T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

            tn = TensorNetwork([t1, t2])
            tn.contract_index("TO_CONTRACT")

            tensor = Tensor(
                tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
            )
            tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
            tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
            tensor.reorder_indices(["UP", "DOWN", "T2_RIGHT"])
            arrays.append(tensor.data)

        t1 = self.tensors[-1]
        t2 = mpo.tensors[-1]

        t1.indices = ["T1_UP", "TO_CONTRACT"]
        t2.indices = ["T2_UP", "T2_RIGHT", "TO_CONTRACT"]

        tn = TensorNetwork([t1, t2])
        tn.contract_index("TO_CONTRACT")

        tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
        tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
        tensor.reorder_indices(["UP", "T2_RIGHT"])
        arrays.append(tensor.data)
        mps = MatrixProductState.from_arrays(arrays)
        if max_bond:
            if mps.bond_dimension > max_bond:
                mps.compress(max_bond)
        return mps

    def set_default_indices(
        self, internal_prefix: str | None = None, external_prefix: str | None = None
    ) -> None:
        """
        Rename all indices to a standard form.

        Args:
            internal_prefix: If provided the internal bonds will have the form internal_prefix + index
            external_prefix: If provided the external bonds will have the form external_prefix + index
        """
        if not internal_prefix:
            internal_prefix = "B"
        if not external_prefix:
            external_prefix = "P"
        self.reshape("udp")

        if self.num_sites == 1:
            self.tensors[0].indices = [external_prefix + "1"]
            return

        new_indices_first = [internal_prefix + "1", external_prefix + "1"]
        self.tensors[0].indices = new_indices_first
        for tidx in range(1, self.num_sites - 1):
            t = self.tensors[tidx]
            new_indices_t = [
                internal_prefix + str(tidx),
                internal_prefix + str(tidx + 1),
                external_prefix + str(tidx + 1),
            ]
            t.indices = new_indices_t
        new_indices_last = [
            internal_prefix + str(self.num_sites - 1),
            external_prefix + str(self.num_sites),
        ]
        self.tensors[-1].indices = new_indices_last
        self.indices = self.get_all_indices()
        return

    def compute_inner_product(self, other: "MatrixProductState") -> complex:
        """
        Calculate the inner product with another MPS.

        Args:
            other: The other MPS.

        Returns
            The inner product <self | other>.
        """
        mps1 = copy.deepcopy(self)
        mps2 = copy.deepcopy(other)
        mps1.reshape()
        mps2.reshape()
        mps2.dagger()
        mps1.set_default_indices()
        mps2.set_default_indices()

        for t in mps2.tensors:
            current_indices = t.indices
            new_indices = [x if x[0] == "P" else x + "_" for x in current_indices]
            t.indices = new_indices
        all_tensors = mps1.tensors + mps2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        for n in range(self.num_sites - 1):
            tn.contract_index(f"P{n+1}")
            tn.contract_index(f"B{n+1}")
            tn.combine_indices([f"P{n+2}", f"B{n+1}_"], new_index_name=f"P{n+2}")

        tn.contract_index(f"P{self.num_sites}")
        val = complex(tn.tensors[0].data.flatten()[0])

        return val

    def compute_expectation_value(self, mpo: MatrixProductOperator) -> float:
        """
        Calculate an expectation value of the form <MPS | MPO | MPS>.

        Args:
            mpo: The MPO whose expectation value will be calculated.

        Returns:
            The expectation value.
        """
        mps1 = copy.deepcopy(self)
        mpo1 = copy.deepcopy(mpo)

        mpo1.reshape("udrl")
        mps1.reshape("udp")

        mps1 = mps1.apply_mpo(mpo1)

        exp_val = self.compute_inner_product(mps1)
        return exp_val

    def outer_product(self, other: "MatrixProductState") -> MatrixProductOperator:
        """
        Take the outer product with another MPS.

        Args:
            other: Another MPS
            normalise: Whether to normalise the resulting outer product

        Returns:
            |self><other| as a MPO
        """
        if self.num_sites == 1:
            ket = self.to_dense_array()
            bra = other.to_dense_array()
            prod = np.outer(ket, bra)
            return MatrixProductOperator.from_arrays([prod])
        ket = copy.deepcopy(self)
        bra = copy.deepcopy(other)
        bra.dagger()

        arrays = []
        ket_tensors = [t.data for t in ket.tensors]
        bra_tensors = [t.data for t in bra.tensors]
        for A_ket, A_bra in zip(ket_tensors, bra_tensors):
            if A_ket.ndim == 2:
                D_ket, d = A_ket.shape
                D_bra, d = A_bra.shape
                coords = []
                data = []

                for s in range(d):
                    for sp in range(d):
                        ket_s = A_ket[:, s]
                        bra_sp = A_bra[:, sp]
                        kron = sparse.kron(ket_s, bra_sp)

                        for idx, val in zip(kron.coords[0], kron.data):
                            coords.append((idx, s, sp))
                            data.append(val)

                shape = (D_ket * D_bra, d, d)
                array = sparse.COO(
                    coords=np.array(coords).T, data=np.array(data), shape=shape
                )
                arrays.append(array)
                continue

            Dl_ket, Dr_ket, d = A_ket.shape
            Dl_bra, Dr_bra, d = A_bra.shape

            coords = []
            data = []

            for s in range(d):
                for sp in range(d):
                    ket_s = A_ket[:, :, s]
                    bra_sp = A_bra[:, :, sp]

                    kron = sparse.kron(ket_s, bra_sp)
                    kron = kron.reshape((Dl_ket, Dl_bra, Dr_ket, Dr_bra))

                    for (i, j, k, l), val in zip(zip(*kron.coords), kron.data):
                        left_index = i * Dl_ket + j
                        right_index = k * Dr_bra + l
                        coords.append((left_index, right_index, s, sp))
                        data.append(val)

            shape = (Dl_ket * Dl_bra, Dr_ket * Dr_bra, d, d)
            array = sparse.COO(
                coords=np.array(coords).T, data=np.array(data), shape=shape
            )
            arrays.append(array)

        mpdo = MatrixProductOperator.from_arrays(arrays)

        return mpdo

    def form_density_operator(self) -> MatrixProductOperator:
        """
        Form the density matrix representation of the state.

        Returns:
            An MPDO
        """
        return self.outer_product(self)

    def partial_trace(
        self, sites: list[int], matrix: bool = False
    ) -> Tensor | MatrixProductOperator:
        """
        Compute the partial trace.

        Args:
            sites: The list of sites to trace over.
            matrix: If True returns the reduced density matrix, otherwise returns a MPDO.

        Returns:
            The reduced state.
        """
        mps = copy.deepcopy(self)
        mpdo = mps.form_density_operator()
        return mpdo.partial_trace(sites, matrix, True)

    def normalise(self) -> None:
        """
        Normalise the MPS.
        """
        norm = self.compute_inner_product(self).real
        self.multiply_by_constant(np.sqrt(1 / norm))
        return

    def expand_bond_dimension(self, diff: int, bond_idx: int) -> "MatrixProductState":
        """
        Expand the internal bond dimension by padding with 0s.

        Args:
            diff: The amount to pad the bond dimension by
            bond_idx: The bond to expand
        """
        arrays = [t.data for t in self.tensors]
        self.reshape("udp")
        if bond_idx - 1 == 0:
            arrays[bond_idx - 1] = sparse.pad(arrays[bond_idx - 1], ((0, diff), (0, 0)))
        else:
            arrays[bond_idx - 1] = sparse.pad(
                arrays[bond_idx - 1], ((0, 0), (0, diff), (0, 0))
            )
        if bond_idx == self.num_sites - 1:
            arrays[bond_idx] = sparse.pad(arrays[bond_idx], ((0, diff), (0, 0)))
        else:
            arrays[bond_idx] = sparse.pad(arrays[bond_idx], ((0, diff), (0, 0), (0, 0)))
        mps = MatrixProductState.from_arrays(arrays)

        return mps

    def expand_bond_dimension_list(
        self, diff: int, bond_idxs: list[int]
    ) -> "MatrixProductState":
        """
        Expand multiple bonds.

        Args:
            diff: The amount to pad the bond dimension by
            bond_idxs: The bonds to expand
        """
        mps = self
        for idx in bond_idxs:
            mps = mps.expand_bond_dimension(diff, idx)
        return mps

    def draw(
        self,
        node_size: int | None = None,
        x_len: int | None = None,
        y_len: int | None = None,
    ):
        """
        Visualise MPS.

        Args:
            node_size: Size of nodes in figure (optional)
            x_len: Figure width (optional)
            y_len: Figure height (optional)

        Returns:
            Displays plot.
        """
        draw_mps(self.tensors, node_size, x_len, y_len)

    def swap_neighbouring_sites(self, idx: int) -> None:
        """
        Swap two neighbouring sites of the MPS.

        Args:
            idx: The index of the first site
        """
        if idx == self.num_sites:
            return

        self.reshape()

        if self.num_sites == 2:
            bond = self.tensors[0].indices[0]
            phys_idx1 = self.tensors[0].indices[1]
            phys_idx2 = self.tensors[1].indices[1]
            self.contract_index(bond)
            self.svd(
                self.tensors[0],
                [phys_idx2],
                [phys_idx1],
                max_bond=None,
                new_index_name=bond,
                tol=1e-12,
            )
            self.tensors[0].reorder_indices([bond, phys_idx2])
            self.tensors[1].reorder_indices([bond, phys_idx1])
            return

        if idx == 1:
            bond = self.tensors[0].indices[0]
            phys_idx1 = self.tensors[0].indices[1]
            phys_idx2 = self.tensors[1].indices[2]
        elif idx == self.num_sites - 1:
            bond = self.tensors[idx - 1].indices[1]
            phys_idx1 = self.tensors[idx - 1].indices[2]
            phys_idx2 = self.tensors[idx].indices[1]
        else:
            bond = self.tensors[idx - 1].indices[1]
            phys_idx1 = self.tensors[idx - 1].indices[2]
            phys_idx2 = self.tensors[idx].indices[2]

        input_inds = copy.deepcopy(self.tensors[idx - 1].indices)
        input_inds.remove(bond)
        input_inds.remove(phys_idx1)
        input_inds.append(phys_idx2)
        output_inds = copy.deepcopy(self.tensors[idx].indices)
        output_inds.remove(bond)
        output_inds.remove(phys_idx2)
        output_inds.append(phys_idx1)
        self.contract_index(bond)
        self.svd(
            self.tensors[idx - 1],
            input_inds,
            output_inds,
            max_bond=None,
            new_index_name=bond,
            tol=1e-12,
        )

        if idx == 1:
            self.tensors[idx - 1].reorder_indices([bond] + input_inds)
        else:
            self.tensors[idx - 1].reorder_indices(
                [input_inds[0]] + [bond] + [input_inds[1]]
            )
        self.tensors[idx].reorder_indices([bond] + output_inds)

        return

    def swap_sites(self, idx1: int, idx2: int) -> None:
        """
        Swap two sites of the MPS.

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
        Reorder the sites of the MPS without changing the state.

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

    def contract_sub_mps(
        self,
        other: "MatrixProductState",
        sites: list[int],
        set_default_indices: bool = False,
    ) -> "MatrixProductState":
        """
        Contract the MPS with a smaller MPS on the given sites

        Args:
            sites: The list of sites where the smaller MPS acts
            set_default_indices: Whether or not to reset the index labels to default

        Returns:
            A smaller MPS that is the output of the partial inner product
        """

        mps1 = copy.deepcopy(self)
        mps2 = copy.deepcopy(other)
        mps1.set_default_indices()
        mps2.set_default_indices()
        mps2.dagger()

        mps1.reshape()
        mps2.reshape()

        all_sites = list(range(1, self.num_sites + 1))
        target_site_ordering = [0] * self.num_sites
        for idx in sites:
            target_site_ordering[idx - 1] = sites.index(idx) + 1
            all_sites.remove(sites.index(idx) + 1)
        for site in all_sites:
            target_site_ordering[target_site_ordering.index(0)] = site

        mps1.reorder_sites(target_site_ordering, set_default_indices=True)

        output_indices = []
        for site_idx in range(len(sites), self.num_sites):
            t = self.tensors[site_idx]
            output_indices.append(t.indices)

        for tidx in range(mps2.num_sites):
            t = mps2.tensors[tidx]
            current_indices = t.indices
            new_indices = [x if x[0] == "P" else x + "_" for x in current_indices]
            t.indices = new_indices
        all_tensors = mps1.tensors + mps2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        for n in range(len(sites) - 1):
            tn.contract_index(f"P{n+1}")
            tn.contract_index(f"B{n+1}")
            tn.combine_indices([f"P{n+2}", f"B{n+2}_"], new_index_name=f"P{n+2}")
        tn.contract_index(f"P{len(sites)}")
        tn.contract_index(f"B{len(sites)}")

        mps = MatrixProductState(tn.tensors)
        for t_idx in range(mps.num_sites):
            t = mps.tensors[t_idx]
            t.indices = (
                output_indices[t_idx][1:] if t_idx == 0 else output_indices[t_idx]
            )

        if set_default_indices:
            self.set_default_indices()

        if mps.bond_dimension and self.bond_dimension:
            if mps.bond_dimension > self.bond_dimension:
                mps.compress(self.bond_dimension)

        return mps

    def get_probability_distribution(self) -> dict[str, float]:
        """
        Compute the probability distribution of an MPS.

        Returns:
            A dictionary of the form {bitstring:probability}
        """
        dist = {}
        sparse_array = self.to_sparse_array()
        for idx, val in zip(sparse_array.coords[0], sparse_array.data):
            bitstring = bin(idx)[2:].zfill(self.num_sites)
            probability = np.abs(val) ** 2
            dist[bitstring] = probability
        return dist

    def sample_bitstrings(self, num_bitstrings: int = 1) -> dict[str, int]:
        """
        Sample bitstrings from an MPS

        Args:
            num_bitstrings: The number of samples to take

        Returns:
            A dictionary of the form {bitstring : counts}
        """
        if self.num_sites <= 15:
            dist = self.get_probability_distribution()
            samples = np.random.choice(
                list(dist.keys()), size=num_bitstrings, p=list(dist.values())
            )
            output = dict(Counter(list(samples)))
            return output

        samples = {}
        prefix_prob_dict = {}  # {prefix : (prob0, prob1)}
        sample_prob_dict = {}  # {bitstring : probabiity}

        samples_collected = 0
        self_copy = copy.deepcopy(self)
        while samples_collected < num_bitstrings:
            prob_existing_sample = np.sum(list(sample_prob_dict.values()))
            choose_existing_sample = np.random.choice(
                ["y", "n"], p=[prob_existing_sample, 1 - prob_existing_sample]
            )
            if choose_existing_sample == "y":
                probs_norm = np.sum(list(sample_prob_dict.values()))
                normalised_probs = [x / probs_norm for x in sample_prob_dict.values()]
                bitstring = np.random.choice(
                    list(sample_prob_dict.keys()), p=normalised_probs
                )
                samples[bitstring] += 1
                samples_collected += 1
                continue
            bitstring = ""
            current_mps = copy.deepcopy(self_copy)
            current_mpo = current_mps.form_density_operator()
            total_prob = 1.0
            temp_prob = 1.0
            for site in range(1, self.num_sites + 1):
                if bitstring in prefix_prob_dict:
                    prob0, prob1 = prefix_prob_dict[bitstring]
                else:
                    num_sites_to_trace = len(bitstring) - (
                        self.num_sites - current_mpo.num_sites
                    )
                    if len(bitstring) > 0 and num_sites_to_trace > 0:
                        sub_mpo_bitstring = bitstring[-num_sites_to_trace:]
                        sub_mpo = MatrixProductOperator.from_bitstring(
                            sub_mpo_bitstring
                        )
                        current_mpo = current_mpo.contract_sub_mpo(
                            sub_mpo, list(range(1, num_sites_to_trace + 1))
                        )
                        current_mpo = current_mpo.partial_trace(
                            list(range(1, num_sites_to_trace + 1)),
                            set_default_indices=True,
                        )
                        current_mpo.multiply_by_constant(1 / temp_prob)
                        current_mpo.indices = current_mpo.get_all_indices()
                        temp_prob = 1.0
                    if site != self.num_sites:
                        site_rdm = (
                            current_mpo.partial_trace(
                                list(range(2, current_mpo.num_sites + 1))
                            )
                            .tensors[0]
                            .data.todense()
                        )
                    else:
                        site_rdm = current_mpo.tensors[0].data.todense()
                    prob0 = min(site_rdm[0, 0].real, 1.0)
                    prob1 = 1.0 - prob0

                site_bit = np.random.choice(["0", "1"], p=[prob0, prob1])
                bitstring += site_bit
                if site_bit == "0":
                    temp_prob *= prob0
                    total_prob *= prob0
                else:
                    temp_prob *= prob1
                    total_prob *= prob1
                if bitstring[:-1] not in prefix_prob_dict:
                    prefix_prob_dict[bitstring[:-1]] = (prob0, prob1)
            if bitstring not in sample_prob_dict:
                if bitstring in samples:
                    samples[bitstring] += 1
                else:
                    samples[bitstring] = 1
                samples_collected += 1
                sample_prob_dict[bitstring] = total_prob

        return samples

    def get_approximate_probability_distribution(
        self, sample_size: int = 1000
    ) -> dict[str, float]:
        """
        Compute the approximate probability distribution of an MPS using samples

        Returns:
            A dictionary of the form {bitstring:probability}
        """
        samples = self.sample_bitstrings(sample_size)
        approx_pd = {k: v / sample_size for k, v in samples.items()}
        return approx_pd
