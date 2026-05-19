import copy
from typing import List, TypeAlias, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
import scipy.linalg
import sparse
from numpy import ndarray

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import svd
from sparse import SparseArray

from .mpo import MatrixProductOperator
from .tensor import StorageHint, Tensor, _as_dense, _as_sparse
from .tn import TensorNetwork

# Visualisation
from .visualisation import draw_mps

DataOptions: TypeAlias = Union[ndarray, SparseArray]

_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def _truncate_sv(s: ndarray, max_bond: int, tol: float) -> int:
    s_sq = s**2
    cum = np.cumsum(s_sq[::-1])[::-1]
    keep = len(s)
    for k in range(len(s)):
        if s[k] < 1e-14:
            keep = k
            break
        if cum[k] < tol**2:
            keep = k + 1
            break
    return max(1, min(keep, max_bond))


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
        cls,
        arrays: List[DataOptions],
        shape: str = "udp",
        storage_hint: StorageHint = StorageHint.DENSE,
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
        first_tensor = Tensor(arrays[0], first_indices, ["MPS_T1"], storage_hint)
        tensors.append(first_tensor)

        physical_idx_pos = shape.index("p")
        virtual_output_idx_pos = shape.index("u")
        virtual_input_idx_pos = shape.index("d")
        for a_idx in range(1, len(arrays) - 1):
            a = arrays[a_idx]
            indices_k = ["", "", ""]
            indices_k[physical_idx_pos] = f"P{a_idx + 1}"
            indices_k[virtual_output_idx_pos] = f"B{a_idx}"
            indices_k[virtual_input_idx_pos] = f"B{a_idx + 1}"
            tensor_k = Tensor(a, indices_k, [f"MPS_T{a_idx + 1}"], storage_hint)
            tensors.append(tensor_k)

        last_shape = shape.replace("d", "")
        physical_idx_pos = last_shape.index("p")
        virtual_output_idx_pos = last_shape.index("u")
        last_indices = ["", ""]
        last_indices[physical_idx_pos] = f"P{len(arrays)}"
        last_indices[virtual_output_idx_pos] = f"B{len(arrays) - 1}"
        last_tensor = Tensor(
            arrays[-1], last_indices, [f"MPS_T{len(arrays)}"], storage_hint
        )
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

        return cls.from_arrays(arrays, storage_hint=StorageHint.SPARSE)

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
    def from_bitstring_dict(
        cls, bitstring_dict: dict[str, complex], max_bond: int | None = None
    ):
        """
        Create an MPS from a dictionary {bitstring : amplitude}

        Args:
            bitstring_dict: The dictionary
            max_bond: Maximum bond dimension
        """
        bitstrings = list(bitstring_dict.keys())
        weights = list(bitstring_dict.values())

        num_states = len(bitstrings)
        num_sites = len(bitstrings[0])

        first_coords = [[], []]
        middle_coords = [[[], [], []] for _ in range(num_sites - 2)]
        last_coords = [[], []]

        first_data = []
        middle_data = [[] for _ in range(num_sites - 2)]
        last_data = []

        for s_idx, (bitstring, weight) in enumerate(zip(bitstrings, weights)):
            b0 = int(bitstring[0])

            first_coords[0].append(s_idx)
            first_coords[1].append(b0)
            first_data.append(weight)

            for site in range(1, num_sites - 1):
                b = int(bitstring[site])
                mid = site - 1

                middle_coords[mid][0].append(s_idx)
                middle_coords[mid][1].append(s_idx)
                middle_coords[mid][2].append(b)

                middle_data[mid].append(1.0)

            bL = int(bitstring[-1])

            last_coords[0].append(s_idx)
            last_coords[1].append(bL)
            last_data.append(1.0)

        first_array = sparse.COO(first_coords, first_data, shape=(num_states, 2))

        middle_arrays = [
            sparse.COO(
                middle_coords[i], middle_data[i], shape=(num_states, num_states, 2)
            )
            for i in range(num_sites - 2)
        ]

        last_array = sparse.COO(last_coords, last_data, shape=(num_states, 2))

        mps = MatrixProductState.from_arrays(
            [first_array] + middle_arrays + [last_array],
            storage_hint=StorageHint.SPARSE,
        )

        if max_bond is not None and mps.bond_dimension > max_bond:
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
    def random_physical_mps(
        cls,
        num_sites: int,
        max_bond: int = 16,
        correlation_length: float = 2.0,
        num_layers: int = 3,
        seed: int | None = None,
    ):
        """
        Construct a random MPS with physically realistic correlations:
        - exponential decay of correlations
        - controlled entanglement growth

        Args:
            num_sites: number of sites
            max_bond: maximum bond dimension
            correlation_length: sets decay scale of correlations
            num_layers: number of circuit layers
            seed: RNG seed
        """

        rng = np.random.default_rng(seed)

        mps = cls.from_bitstring("0" * num_sites)

        def random_unitary(strength):
            H = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            H = H + H.conj().T
            return scipy.linalg.expm(-1j * strength * H)

        for _ in range(num_layers):
            # loop over distances
            for dist in range(1, num_sites):
                # exponential decay of interaction strength
                strength = np.exp(-dist / correlation_length)

                # skip negligible interactions
                if strength < 1e-3:
                    continue

                for i in range(num_sites - dist):
                    j = i + dist

                    U = random_unitary(strength)
                    gate = UnitaryGate(U)

                    # Apply gate to sites (i, j)
                    gate_mpo = MatrixProductOperator.from_qiskit_gate(gate)
                    mps.apply_sub_mpo(gate_mpo, [i + 1, j + 1], max_bond=max_bond)

        mps.normalise()
        return mps

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
        mps.normalise()
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
                new_index_name=f"C{idx + 1}",
                new_labels=[[f"T{idx + 1}"], [f"T{idx + 2}"]],
            )
            if idx == 0:
                new_idx_order1 = [f"C{idx + 1}", "P1"]
            else:
                new_idx_order1 = [
                    f"C{idx}",
                    f"C{idx + 1}",
                    f"P{idx + 1}",
                ]
            new_idx_order2 = [f"C{idx + 1}"] + output_inds
            tn.tensors[idx].reorder_indices(new_idx_order1)
            tn.tensors[idx + 1].reorder_indices(new_idx_order2)
        arrays = [tn.tensors[i].data for i in range(num_qubits)]
        mps = cls.from_arrays(arrays)
        if max_bond:
            if mps.bond_dimension > max_bond:
                mps.compress(max_bond)
        return mps

    @classmethod
    def from_mpo(cls, mpo: MatrixProductOperator) -> "MatrixProductState":
        """Convert MPO to MPS with twice the number of sites"""
        mpo_tensors = mpo.tensors
        mps_arrays = []

        first_t = mpo_tensors[0]
        data = first_t.to_dense()
        data_reshaped = np.moveaxis(data, [0, 1, 2], [0, 2, 1])
        shape = data_reshaped.shape
        data_mat = np.reshape(data_reshaped, (shape[0] * shape[1], shape[2]))
        u, s, vh = svd(data_mat, full_matrices=False)
        new_index_dim = vh.shape[0]
        us = u @ np.diag(s)
        us = np.reshape(us, (shape[0], shape[1], new_index_dim))
        us = np.moveaxis(us, 2, 0)
        vh = np.reshape(vh, (new_index_dim, shape[2]))

        mps_arrays.append(vh)
        mps_arrays.append(us)

        for t in mpo_tensors[1:-1]:
            t_data = t.to_dense()
            t_data_reshaped = np.moveaxis(t_data, [0, 1, 2, 3], [2, 0, 1, 3])
            shape = t_data_reshaped.shape
            t_mat = np.reshape(
                t_data_reshaped, (shape[0] * shape[1], shape[2] * shape[3])
            )
            u, s, vh = svd(t_mat, full_matrices=True)
            new_index_dim = vh.shape[0]
            us = u @ np.diag(s)
            us = np.reshape(us, (shape[0], shape[1], new_index_dim))
            us = np.moveaxis(us, 2, 0)
            vh = np.reshape(vh, (new_index_dim, shape[2], shape[3]))
            vh = np.moveaxis(vh, 0, 1)

            mps_arrays.append(vh)
            mps_arrays.append(us)

        last_t = mpo_tensors[-1]
        data = last_t.to_dense()
        data_reshaped = np.moveaxis(data, [0, 1, 2], [1, 2, 0])
        shape = data_reshaped.shape
        data_mat = np.reshape(data_reshaped, (shape[0], shape[1] * shape[2]))
        u, s, vh = svd(data_mat, full_matrices=False)
        new_index_dim = vh.shape[0]
        us = u @ np.diag(s)
        us = np.reshape(us, (shape[0], new_index_dim))
        us = np.moveaxis(us, 0, 1)
        vh = np.reshape(vh, (new_index_dim, shape[1], shape[2]))
        vh = np.moveaxis(vh, 0, 1)

        mps_arrays.append(vh)
        mps_arrays.append(us)

        mps = cls.from_arrays(mps_arrays)
        return mps

    def __add__(self, other: "MatrixProductState") -> "MatrixProductState":
        """
        Defines MPS addition.
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

        for t_idx in range(1, self.num_sites - 1):
            t1 = self.tensors[t_idx].data
            t2 = other.tensors[t_idx].data

            if result_sparse:
                t1 = _as_sparse(t1)
                t2 = _as_sparse(t2)

                D1_up, D1_down, d = t1.shape
                D2_up, D2_down, _ = t2.shape

                zeros_tr = sparse.COO(np.zeros((D1_up, D2_down, d)))
                zeros_bl = sparse.COO(np.zeros((D2_up, D1_down, d)))

                # block structure in (up, down)
                top = sparse.concatenate([t1, zeros_tr], axis=1)
                bottom = sparse.concatenate([zeros_bl, t2], axis=1)

                new_data = sparse.concatenate([top, bottom], axis=0)

            else:
                t1 = _as_dense(t1)
                t2 = _as_dense(t2)

                D1_up, D1_down, d = t1.shape
                D2_up, D2_down, _ = t2.shape

                zeros_tr = np.zeros((D1_up, D2_down, d), dtype=complex)
                zeros_bl = np.zeros((D2_up, D1_down, d), dtype=complex)

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

        return MatrixProductState.from_arrays(arrays, storage_hint=sh)

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
        for t in mps.tensors:
            if isinstance(t.data, np.ndarray):
                t.data = sparse.COO.from_numpy(t.data)
        output = mps.contract_entire_network()
        output.combine_indices(output.indices, output.indices[0])
        return output.data

    def to_dense_array(self) -> ndarray:
        """
        Convert the MPS to a dense array.
        """
        mps = copy.deepcopy(self)
        for t in mps.tensors:
            if not isinstance(t.data, np.ndarray):
                t.data = t.data.todense()
        output = mps.contract_entire_network()
        output.combine_indices(output.indices, output.indices[0])
        return output.data

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

    def _apply_swap(self, site: int, max_bond: int | None = None) -> None:
        """Apply SWAP gate between sites site and site+1 in-place."""
        self._apply_local_gate_dense(_SWAP, site, site + 1, max_bond)

    def _apply_local_gate_dense(
        self,
        gate: ndarray,
        site0: int,
        site1: int,
        max_bond: int | None,
        tol: float = 1e-12,
    ) -> None:
        """
        Apply a local 2-qubit gate to neighbouring sites (site1 == site0 + 1).
        Dense throughout. Updates MPS in-place.

        Index convention:
            Site 1 (left boundary) : (down, phys)
            Site N (right boundary): (up, phys)
            Interior               : (up, down, phys)
        """
        assert site1 == site0 + 1
        n = self.num_sites
        G = gate.reshape(2, 2, 2, 2)  # (out0, out1, in0, in1)

        t0 = _as_dense(self.tensors[site0 - 1].data)
        t1 = _as_dense(self.tensors[site0].data)
        eps = 1e-14

        if site0 == 1 and site1 == n:
            # t0: (d,)  t1: (d,)
            merged = np.einsum("i,j,klij->kl", t0, t1, G)
            u, s, vh = scipy.linalg.svd(merged, full_matrices=False, check_finite=False)
            keep = _truncate_sv(
                s, min(2, 2) if max_bond is None else min(max_bond, 2, 2), tol
            )
            new_t0 = vh[:keep, :]  # (k, d)
            new_t1 = np.moveaxis((u[:, :keep] * s[:keep]), 1, 0)  # (k, d)

        elif site0 == 1:
            # t0: (a, i)   t1: (a, b, j)
            b = t1.shape[1]
            merged = np.einsum("ai,abj,klij->klb", t0, t1, G)
            mat = merged.transpose(0, 2, 1).reshape(2, b * 2)
            bond = min(max_bond, 2, b * 2) if max_bond else min(2, b * 2)
            u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
            keep = _truncate_sv(s, bond, tol)
            new_t0 = np.moveaxis((u[:, :keep] * s[:keep]), 1, 0)  # (k, d)
            new_t1 = vh[:keep, :].reshape(keep, b, 2)  # (k, b, d)

        elif site1 == n:
            # t0: (a, b, i)   t1: (b, j)
            a = t0.shape[0]
            merged = np.einsum("abi,bj,klij->akl", t0, t1, G)
            mat = merged.reshape(a * 2, 2)
            bond = min(max_bond, a * 2, 2) if max_bond else min(a * 2, 2)
            u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
            keep = _truncate_sv(s, bond, tol)
            new_t0 = (
                (u[:, :keep] * s[:keep]).reshape(a, 2, keep).transpose(0, 2, 1)
            )  # (a, k, d)
            new_t1 = vh[:keep, :]  # (k, d)

        else:
            # t0: (a, b, i)   t1: (b, c, j)
            a, c = t0.shape[0], t1.shape[1]
            merged = np.einsum("abi,bcj,klij->aklc", t0, t1, G)
            mat = merged.reshape(a * 2, 2 * c)
            bond = min(max_bond, a * 2, 2 * c) if max_bond else min(a * 2, 2 * c)
            u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
            keep = _truncate_sv(s, bond, tol)
            new_t0 = (
                (u[:, :keep] * s[:keep]).reshape(a, 2, keep).transpose(0, 2, 1)
            )  # (a, k, d)
            new_t1 = vh[:keep, :].reshape(keep, 2, c).transpose(0, 2, 1)  # (k, c, d)

        new_t0[np.abs(new_t0) < eps] = 0.0
        new_t1[np.abs(new_t1) < eps] = 0.0

        self.tensors[site0 - 1].data = new_t0
        self.tensors[site0 - 1].dimensions = new_t0.shape
        self.tensors[site0].data = new_t1
        self.tensors[site0].dimensions = new_t1.shape
        self.update_bond_information()

    def apply_mpo(
        self,
        mpo: MatrixProductOperator,
        max_bond: int | None = None,
    ) -> "MatrixProductState":
        """
        Apply an MPO to the full MPS.

        Args:
            mpo: The MPO to apply.
            max_bond: Optional maximum bond dimension.

        Returns:
            The new MPS.
        """
        self.reshape()
        mpo.reshape()
        arrays = []

        # Left boundary: MPS rank-2 (down, phys), MPO rank-3 (down, R, L)
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

        # Interior sites: MPS rank-3 (up, down, phys), MPO rank-4 (up, down, R, L)
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

        # Right boundary: MPS rank-2 (up, phys), MPO rank-3 (up, R, L)
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
        if max_bond and mps.bond_dimension > max_bond:
            mps.compress(max_bond)
        return mps

    def apply_sub_mpo(
        self,
        mpo: MatrixProductOperator,
        sites: list[int],
        max_bond: int | None = None,
    ) -> "MatrixProductState":
        """
        Apply a smaller MPO to the MPS at the given contiguous sites,
        using a SWAP network to bring non-contiguous sites together.

        Args:
            mpo: The MPO to apply.
            sites: The list of contiguous site indices where the MPO acts.
            max_bond: Optional maximum bond dimension.

        Returns:
            The new MPS.
        """
        mps = copy.deepcopy(self)
        mpo = copy.deepcopy(mpo)
        mps.set_default_indices()
        mpo.set_default_indices()
        mps.reshape()
        mpo.reshape()

        sorted_sites = sorted(sites)
        n_sub = len(sorted_sites)

        # SWAP non-contiguous sites into a contiguous block at their natural positions.
        # For each gap between consecutive target sites, SWAP the intervening
        # sites outward so that the target sites become adjacent.
        # We track where each target site actually is after swaps.
        current_positions = list(sorted_sites)

        for i in range(n_sub - 1):
            target = current_positions[i] + 1
            while current_positions[i + 1] > target:
                # SWAP current_positions[i+1] leftward by one
                swap_site = current_positions[i + 1] - 1
                mps._apply_swap(swap_site, max_bond)
                current_positions[i + 1] -= 1

        # current_positions is now a contiguous block starting at sorted_sites[0]
        block_start = current_positions[0]  # == sorted_sites[0]

        # Apply the sub-MPO site by site across the contiguous block
        n = mps.num_sites

        if n_sub == 1:
            # Single site: just contract the physical leg directly
            t_mps = mps.tensors[block_start - 1]
            t_mpo = mpo.tensors[0]
            if block_start == 1:
                # MPS: (down, phys), MPO: (down, R, L)
                t_mps.indices = ["T1_DOWN", "TO_CONTRACT"]
                t_mpo.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]
            elif block_start == n:
                # MPS: (up, phys), MPO: (up, R, L)
                t_mps.indices = ["T1_UP", "TO_CONTRACT"]
                t_mpo.indices = ["T2_UP", "T2_RIGHT", "TO_CONTRACT"]
            else:
                # MPS: (up, down, phys), MPO: (up, down, R, L)
                t_mps.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT"]
                t_mpo.indices = ["T2_UP", "T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]
            tn = TensorNetwork([t_mps, t_mpo])
            tn.contract_index("TO_CONTRACT")
            tensor = Tensor(
                tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
            )
            if block_start == 1:
                tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
                tensor.reorder_indices(["DOWN", "T2_RIGHT"])
            elif block_start == n:
                tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
                tensor.reorder_indices(["UP", "T2_RIGHT"])
            else:
                tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
                tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
                tensor.reorder_indices(["UP", "DOWN", "T2_RIGHT"])
            mps.tensors[block_start - 1].data = tensor.data
            mps.tensors[block_start - 1].dimensions = tensor.data.shape

        else:
            arrays_sub = []

            # First site of the block
            t1 = mps.tensors[block_start - 1]
            t2 = mpo.tensors[0]
            if block_start == 1:
                # MPS rank-2 (down, phys), MPO rank-3 (down, R, L)
                t1.indices = ["T1_DOWN", "TO_CONTRACT"]
                t2.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]
                tn = TensorNetwork([t1, t2])
                tn.contract_index("TO_CONTRACT")
                tensor = Tensor(
                    tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
                )
                tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
                tensor.reorder_indices(["DOWN", "T2_RIGHT"])
            else:
                # MPS rank-3 (up, down, phys), MPO rank-3 (down, R, L)
                t1.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT"]
                t2.indices = ["T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]
                tn = TensorNetwork([t1, t2])
                tn.contract_index("TO_CONTRACT")
                tensor = Tensor(
                    tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
                )
                tensor.combine_indices(["T1_DOWN", "T2_DOWN"], new_index_name="DOWN")
                tensor.reorder_indices(["T1_UP", "DOWN", "T2_RIGHT"])
            arrays_sub.append(tensor.data)

            # Interior sites of the block
            for i in range(1, n_sub - 1):
                t1 = mps.tensors[block_start + i - 1]
                t2 = mpo.tensors[i]
                # MPS rank-3 (up, down, phys), MPO rank-4 (up, down, R, L)
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
                arrays_sub.append(tensor.data)

            # Last site of the block
            t1 = mps.tensors[block_start + n_sub - 2]
            t2 = mpo.tensors[-1]
            block_end = block_start + n_sub - 1
            if block_end == n:
                # MPS rank-2 (up, phys), MPO rank-3 (up, R, L)
                t1.indices = ["T1_UP", "TO_CONTRACT"]
                t2.indices = ["T2_UP", "T2_RIGHT", "TO_CONTRACT"]
                tn = TensorNetwork([t1, t2])
                tn.contract_index("TO_CONTRACT")
                tensor = Tensor(
                    tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
                )
                tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
                tensor.reorder_indices(["UP", "T2_RIGHT"])
            else:
                # MPS rank-3 (up, down, phys), MPO rank-3 (up, R, L)
                t1.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT"]
                t2.indices = ["T2_UP", "T2_RIGHT", "TO_CONTRACT"]
                tn = TensorNetwork([t1, t2])
                tn.contract_index("TO_CONTRACT")
                tensor = Tensor(
                    tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels()
                )
                tensor.combine_indices(["T1_UP", "T2_UP"], new_index_name="UP")
                tensor.reorder_indices(["UP", "T1_DOWN", "T2_RIGHT"])
            arrays_sub.append(tensor.data)

            # Write the contracted block back into the MPS tensors
            for i, arr in enumerate(arrays_sub):
                mps.tensors[block_start + i - 1].data = arr
                mps.tensors[block_start + i - 1].dimensions = arr.shape

        # SWAP non-contiguous sites back to their original positions
        for i in range(n_sub - 2, -1, -1):
            target = sorted_sites[i + 1] if i + 1 < n_sub else None
            if target is not None:
                while current_positions[i + 1] < sorted_sites[i + 1]:
                    swap_site = current_positions[i + 1]
                    mps._apply_swap(swap_site, max_bond)
                    current_positions[i + 1] += 1

        arrays = [t.data for t in mps.tensors]
        result_mps = MatrixProductState.from_arrays(arrays)

        if max_bond and result_mps.bond_dimension > max_bond:
            result_mps.compress(max_bond)

        return result_mps

    def contract_sub_mps(
        self,
        other: "MatrixProductState",
        sites: list[int],
    ) -> "MatrixProductState":
        """
        Contract the MPS with a smaller MPS on the given sites,
        using a SWAP network to bring non-contiguous sites together.

        Args:
            other: The smaller MPS to contract with.
            sites: The list of site indices where the smaller MPS acts.

        Returns:
            A smaller MPS that is the output of the partial inner product.
        """
        mps1 = copy.deepcopy(self)
        mps2 = copy.deepcopy(other)
        mps1.set_default_indices()
        mps2.set_default_indices()
        mps2.dagger()
        mps1.reshape()
        mps2.reshape()

        sorted_sites = sorted(sites)
        n_sub = len(sorted_sites)
        n = mps1.num_sites

        # SWAP non-contiguous sites into a contiguous block
        current_positions = list(sorted_sites)
        for i in range(n_sub - 1):
            target = current_positions[i] + 1
            while current_positions[i + 1] > target:
                swap_site = current_positions[i + 1] - 1
                mps1._apply_swap(swap_site)
                current_positions[i + 1] -= 1

        block_start = current_positions[0]
        block_end = current_positions[-1]

        # Rename mps2 virtual bonds to avoid clashes
        for tidx in range(mps2.num_sites):
            t = mps2.tensors[tidx]
            t.indices = [x if x[0] == "P" else x + "_" for x in t.indices]

        # Contract mps2 into the block site by site
        # After contraction of physical legs and virtual bonds within the block,
        # we are left with a matrix/vector connecting the dangling bonds
        # at the block boundaries

        all_tensors = mps1.tensors[block_start - 1 : block_end] + mps2.tensors
        tn = TensorNetwork(all_tensors)

        # Contract physical legs and internal virtual bonds
        for i in range(n_sub):
            tn.contract_index(f"P{block_start + i}")
        for i in range(n_sub - 1):
            tn.contract_index(f"B{block_start + i}")
            tn.combine_indices(
                [f"B{block_start + i + 1}", f"B{block_start + i + 1}_"],
                new_index_name=f"B{block_start + i + 1}",
            )

        # The result is a matrix/vector connecting the left and right
        # dangling bonds of the block. Absorb into the neighbouring site.
        # The remaining tensor in tn connects B{block_start-1} (if exists)
        # and B{block_end+1} (if exists) — i.e. it is rank-0, rank-1, or rank-2.
        result_data = tn.tensors[0].data
        result_data = _as_dense(result_data)

        # Remove the block tensors from mps1
        for _ in range(n_sub):
            mps1.tensors.pop(block_start - 1)
        mps1.num_sites -= n_sub

        if result_data.ndim == 0:
            # Scalar: multiply into left neighbour if exists, else right
            if block_start > 1:
                neighbour = mps1.tensors[block_start - 2]
                neighbour.data = _as_dense(neighbour.data) * result_data
            else:
                neighbour = mps1.tensors[0]
                neighbour.data = _as_dense(neighbour.data) * result_data

        elif result_data.ndim == 1:
            # Vector: one dangling bond. Contract into the appropriate neighbour.
            if block_start > 1 and block_end < n:
                # Shouldn't happen for a proper contiguous block — both ends open
                raise ValueError("Unexpected rank-1 result with two open boundaries.")
            elif block_start == 1:
                # Block was at the left end: dangling bond points right to site block_end+1
                # which is now site 1 of the remaining MPS
                neighbour = mps1.tensors[0]
                nb_data = _as_dense(neighbour.data)
                # neighbour is new left boundary: rank-2 (down, phys)
                # result_data has index B{block_end+1}
                neighbour.data = np.einsum("a,ab->b", result_data, nb_data)
            else:
                # Block was at the right end: dangling bond points left
                neighbour = mps1.tensors[block_start - 2]
                nb_data = _as_dense(neighbour.data)
                # neighbour is new right boundary: rank-2 (up, phys)
                neighbour.data = np.einsum("ab,b->a", nb_data, result_data)
            neighbour.dimensions = neighbour.data.shape

        else:
            # Matrix: two dangling bonds. Absorb into the right neighbour
            # (the site immediately after the block, now at index block_start-1)
            if block_start > 1 and block_end < n:
                # General middle case: absorb into right neighbour
                neighbour = mps1.tensors[block_start - 1]
                nb_data = _as_dense(neighbour.data)
                # result_data: (B{block_start-1}, B{block_end+1})
                # neighbour (right of block): rank-3 (up, down, phys) or rank-2 (up, phys)
                # Its up bond = B{block_end+1} contracts with result's second index
                # result's first index B{block_start-1} fuses with left neighbour's down bond
                # We absorb result into right neighbour, giving it an extra left bond
                if block_end + 1 == n:
                    # right neighbour is right boundary: rank-2 (up, phys)
                    neighbour.data = np.einsum("ab,bc->ac", result_data, nb_data)
                else:
                    # right neighbour is interior: rank-3 (up, down, phys)
                    neighbour.data = np.einsum("ab,bcd->acd", result_data, nb_data)
                neighbour.dimensions = neighbour.data.shape
            else:
                raise ValueError("Unexpected matrix result at MPS boundary.")

        arrays = [_as_dense(t.data) for t in mps1.tensors]
        result_mps = MatrixProductState.from_arrays(arrays)

        if result_mps.bond_dimension and self.bond_dimension:
            if result_mps.bond_dimension > self.bond_dimension:
                result_mps.compress(self.bond_dimension)

        return result_mps

    def set_default_indices(
        self,
        internal_prefix: str | None = None,
        external_prefix: str | None = None,
        index_from: int | None = None,
    ) -> None:
        """
        Rename all indices to a standard form.

        Args:
            internal_prefix: If provided the internal bonds will have the form internal_prefix + index
            external_prefix: If provided the external bonds will have the form external_prefix + index
            index_from: Where to index from, default to 1
        """
        if not internal_prefix:
            internal_prefix = "B"
        if not external_prefix:
            external_prefix = "P"
        if not index_from:
            index_from = 1
        self.reshape("udp")

        if self.num_sites == 1:
            self.tensors[0].indices = [external_prefix + f"{index_from}"]
            return

        new_indices_first = [
            internal_prefix + f"{index_from}",
            external_prefix + f"{index_from}",
        ]
        self.tensors[0].indices = new_indices_first
        for tidx in range(1, self.num_sites - 1):
            t = self.tensors[tidx]
            new_indices_t = [
                internal_prefix + str(index_from + tidx - 1),
                internal_prefix + str(index_from + tidx),
                external_prefix + str(index_from + tidx),
            ]
            t.indices = new_indices_t
        new_indices_last = [
            internal_prefix + str(index_from + self.num_sites - 2),
            external_prefix + str(index_from + self.num_sites - 1),
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
            The inner product <other | self>.
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
            tn.contract_index(f"P{n + 1}")
            tn.contract_index(f"B{n + 1}")
            tn.combine_indices([f"P{n + 2}", f"B{n + 1}_"], new_index_name=f"P{n + 2}")

        tn.contract_index(f"P{self.num_sites}")
        val = complex(tn.tensors[0].data.flatten()[0])

        return val

    def compute_expectation_value(self, mpo: MatrixProductOperator) -> complex:
        """
        Calculate an expectation value of the form <MPS | MPO | MPS>.

        Args:
            mpo: The MPO whose expectation value will be calculated.

        Returns:
            The expectation value.
        """
        mps_ket = copy.deepcopy(self)
        mpo_op = copy.deepcopy(mpo)

        mps_bra = copy.deepcopy(mps_ket)
        mps_bra.dagger()

        # Relabel
        # MPS ket: internal A*, external B*
        mps_ket.set_default_indices(internal_prefix="A", external_prefix="B")

        # MPO: input = B* (to match ket physical), output = D*, internal = C*
        mpo_op.set_default_indices(
            input_prefix="B", output_prefix="D", internal_prefix="C"
        )

        # MPS bra: internal E*, physical D* (to match MPO output)
        mps_bra.set_default_indices(internal_prefix="E", external_prefix="D")

        tn = TensorNetwork(mps_bra.tensors + mpo_op.tensors + mps_ket.tensors)
        val = tn.contract_entire_network()

        return complex(val)

    def outer_product(self, other: "MatrixProductState") -> MatrixProductOperator:
        """
        Take the outer product with another MPS.

        Args:
            other: Another MPS
            normalise: Whether to normalise the resulting outer product

        Returns:
            |self><other| as a MPO
        """
        assert self.num_sites == other.num_sites

        if self.num_sites == 1:
            ket = self.to_dense_array()
            bra = other.to_dense_array()
            return MatrixProductOperator.from_arrays([np.outer(ket, bra)])

        ket = copy.deepcopy(self)
        bra = copy.deepcopy(other)
        bra.dagger()

        self_sparse = all(isinstance(t.data, SparseArray) for t in ket.tensors)
        other_sparse = all(isinstance(t.data, SparseArray) for t in bra.tensors)
        result_sparse = self_sparse and other_sparse

        arrays = []

        for A_ket, A_bra in zip(ket.tensors, bra.tensors):
            A_k = A_ket.data
            A_b = A_bra.data

            if result_sparse:
                A_k = _as_sparse(A_k)
                A_b = _as_sparse(A_b)

                if A_k.ndim == 2 or (A_k.ndim == 3 and A_k.shape[-1] == 1):
                    if A_k.ndim == 2:
                        Dk, d = A_k.shape
                        A_k = A_k.reshape((Dk, 1, d))
                    else:
                        Dl_k, d, Dr_k = A_k.shape
                        A_k = A_k.reshape((Dl_k, 1, d))

                    if A_b.ndim == 2:
                        Db, _ = A_b.shape
                        A_b = A_b.reshape((Db, 1, d))
                    else:
                        Dl_b, d_b, Dr_b = A_b.shape
                        A_b = A_b.reshape((Dl_b, 1, d_b))

                    boundary = True
                else:
                    boundary = False

                Dl_k, Dr_k, d = A_k.shape
                Dl_b, Dr_b, _ = A_b.shape

                # broadcast outer product over physical indices
                # result: (Dl_k, Dl_b, Dr_k, Dr_b, d, d)
                W = sparse.einsum("a b s, c d t -> a c b d s t", A_k, A_b)

                # reshape bonds
                W = W.reshape((Dl_k * Dl_b, Dr_k * Dr_b, d, d))

                if boundary:
                    W = W.reshape((Dl_k * Dl_b, d, d))

            else:
                A_k = _as_dense(A_k)
                A_b = _as_dense(A_b)

                if A_k.ndim == 2 or (A_k.ndim == 3 and A_k.shape[-1] == 1):
                    if A_k.ndim == 2:
                        Dk, d = A_k.shape
                        A_k = A_k.reshape((Dk, 1, d))
                    else:
                        Dl_k, d, Dr_k = A_k.shape
                        A_k = A_k.reshape((Dl_k, 1, d))

                    if A_b.ndim == 2:
                        Db, _ = A_b.shape
                        A_b = A_b.reshape((Db, 1, d))
                    else:
                        Dl_b, d_b, Dr_b = A_b.shape
                        A_b = A_b.reshape((Dl_b, 1, d_b))

                    boundary = True
                else:
                    boundary = False

                Dl_k, Dr_k, d = A_k.shape
                Dl_b, Dr_b, _ = A_b.shape

                W = np.einsum("a b s, c d t -> a c b d s t", A_k, A_b)

                W = W.reshape((Dl_k * Dl_b, Dr_k * Dr_b, d, d))

                if boundary:
                    W = W.reshape((Dl_k * Dl_b, d, d))

            arrays.append(W)

        sh = StorageHint.SPARSE if result_sparse else StorageHint.DENSE

        return MatrixProductOperator.from_arrays(arrays, storage_hint=sh)

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
        return mpdo.partial_trace(sites, matrix)

    def normalise(self, value: float = 1.0) -> None:
        """
        Normalise the MPS.
        """
        norm = self.compute_inner_product(self).real
        self.multiply_by_constant(np.sqrt(value / norm))
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

    def sample_bitstrings(
        self,
        num_samples: int = 1,
        seed: int | None = None,
    ) -> dict[str, int]:
        """
        Draw `num_bitstrings` samples from the Born distribution of `mps`.

        Parameters
        ----------
        mps            : MatrixProductState in any gauge.
        num_bitstrings : Number of samples to draw (with replacement).
        seed           : Optional RNG seed for reproducibility.

        Returns
        -------
        dict mapping bitstring → count.

        Algorithm
        ---------
        1. Canonicalise once to top-canonical form (orthogonality centre at 1).
        Cost: O(N d χ³).

        2. For each sample, sweep top-to-bottom:
        - At site i, compute marginal probs from the accumulated env (χ×χ matrix).
        - Sample a bit, accumulate env update.  Cost per site: O(d χ²).
        - Total per sample: O(N d χ²).  [χ² not χ³ because no SVD needed]

        The overall cost is  O(N d χ³)  for canonicalisation  +
                            O(S · N · d · χ²)  for S samples.
        """
        mps = copy.deepcopy(self)

        def _dense(tensor: Tensor) -> np.ndarray:
            """Return tensor data as a dense numpy array."""
            arr = tensor.to_dense()
            return np.asarray(arr)

        def _site_tensor(mps: "MatrixProductState", site: int) -> np.ndarray:
            """
            Return the dense array for the 1-indexed site.
            Index layout:
            Interior  (1 < site < N)  →  [up, down, physical]   shape (χ_u, χ_d, d)
            Top       (site == 1)     →  [down, physical]         shape (χ_d, d)
            Bottom    (site == N)     →  [up, physical]           shape (χ_u, d)
            """
            return _dense(mps.tensors[site - 1])

        rng = np.random.default_rng(seed)
        N = mps.num_sites

        canonical_mps = copy.deepcopy(mps)
        canonical_mps.move_orthogonality_centre(1)

        tensors: list[np.ndarray] = [
            _site_tensor(canonical_mps, s) for s in range(1, N + 1)
        ]

        def _phys_dim(site_idx: int) -> int:
            """Physical dim is always the last index of the tensor."""
            return tensors[site_idx].shape[-1]

        counts: dict[str, int] = {}

        for _ in range(num_samples):
            bits: list[str] = []

            env: np.ndarray | float = 1.0

            for site_idx in range(N):
                A = tensors[site_idx]
                is_top = site_idx == 0
                is_bottom = site_idx == N - 1
                d = _phys_dim(site_idx)

                if is_top:
                    raw_probs = np.array(
                        [np.real(np.dot(A[:, k].conj(), A[:, k])) for k in range(d)]
                    )
                elif is_bottom:
                    raw_probs = np.array(
                        [
                            np.real(np.einsum("i,ij,j->", A[:, k].conj(), env, A[:, k]))
                            for k in range(d)
                        ]
                    )
                else:
                    raw_probs = np.array(
                        [
                            np.real(
                                np.einsum(
                                    "ij,ik,jk->", env, A[:, :, k].conj(), A[:, :, k]
                                )
                            )
                            for k in range(d)
                        ]
                    )

                raw_probs = np.clip(raw_probs, 0.0, None)
                total = raw_probs.sum()
                if total < 1e-15:
                    raise ValueError(
                        f"Probability at site {site_idx + 1} is numerically zero. "
                        f"Check MPS normalisation."
                    )
                probs = raw_probs / total

                bit = int(rng.choice(d, p=probs))
                bits.append(str(bit))

                if not is_bottom:
                    if is_top:
                        v = A[:, bit]
                        norm = np.sqrt(np.real(np.dot(v.conj(), v)))
                        v = v / (norm if norm > 1e-15 else 1.0)
                        env = np.outer(v.conj(), v)
                    else:
                        M = A[:, :, bit]
                        env_new = np.einsum("ij,ik,jl->kl", env, M.conj(), M)
                        norm = np.trace(env_new).real
                        env = env_new / (norm if norm > 1e-15 else 1.0)

            bitstring = "".join(bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

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

    def to_two_copy_mps(self) -> "MatrixProductState":
        """Build the MPS representation of |psi>|psi>"""
        all_sparse = all(isinstance(t.data, SparseArray) for t in self.tensors)

        def reshape(x, shape):
            if isinstance(x, SparseArray):
                return sparse.reshape(x, shape)
            else:
                return np.reshape(x, shape)

        doubled_mps_arrays = []
        for idx in range(self.num_sites):
            array = copy.deepcopy(self.tensors[idx].data)

            if idx == self.num_sites - 1:
                array = reshape(array, (array.shape[0], 1, array.shape[1]))

            doubled_mps_arrays.append(array)

        for idx in range(self.num_sites):
            array = copy.deepcopy(self.tensors[idx].data)

            if idx == 0:
                array = reshape(array, (1, array.shape[0], array.shape[1]))

            doubled_mps_arrays.append(array)

        sh = StorageHint.SPARSE if all_sparse else StorageHint.DENSE

        return MatrixProductState.from_arrays(doubled_mps_arrays, storage_hint=sh)

    def householder_map(self, other: "MatrixProductState") -> MatrixProductOperator:
        """
        Construct an MPO representing the Householder-like unitary V that swaps
        MPS |psi_C⟩ and |psi_D⟩, and acts as identity on the orthogonal complement.

        V = |other><self| + |self><other| + (I - |self><self| - |other><other|)

        Args:
            other: MatrixProductState representing |other>

        Returns:
            MatrixProductOperator representing the unitary V
        """
        assert (
            self.num_sites == other.num_sites
        ), "psi_C and psi_D must have the same number of sites"
        psi_C = copy.deepcopy(self)
        psi_D = copy.deepcopy(other)

        # Compute outer product MPOs
        proj_DC = psi_D.outer_product(psi_C)  # calculate |D><C|
        proj_CD = psi_C.outer_product(psi_D)  # calculate |C><D|
        proj_CC = psi_C.outer_product(psi_C)  # calculate |C><C|
        proj_DD = psi_D.outer_product(psi_D)  # calculate |D><D|

        # Identity MPO
        identity = MatrixProductOperator.identity_mpo(self.num_sites)

        # Build V = |D><C| + |C><D| + I - |C><C| - |D><D|
        V = proj_DC + proj_CD + identity - proj_CC - proj_DD

        return V

    def compress(self, max_bond: int) -> None:
        """Special compress method for MPS
        Args:
            max_bond: Bond dimension to compress to
        """
        for tidx in range(self.num_sites - 1):
            t = self.tensors[tidx]
            original_inds = copy.deepcopy(t.indices)
            original_next_inds = copy.deepcopy(self.tensors[tidx + 1].indices)
            bond_name = t.indices[0] if len(t.indices) == 2 else t.indices[1]
            input_indices = (
                t.indices[1:]
                if len(t.indices) == 2
                else [t.indices[0]] + [t.indices[2]]
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
                ["TEMP", original_inds[1]]
                if len(original_inds) == 2
                else [original_inds[0], "TEMP", original_inds[2]]
            )
            reordered_indices_next = ["TEMP"] + original_next_inds[1:]
            self.tensors[tidx].reorder_indices(reordered_indices)
            self.tensors[tidx + 1].reorder_indices(reordered_indices_next)
            self.tensors[tidx].indices = original_inds
            self.tensors[tidx + 1].indices = original_next_inds
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
