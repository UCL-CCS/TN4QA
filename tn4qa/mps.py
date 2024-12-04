from typing import List, Union, TypeAlias
import copy

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
from numpy import ndarray
import sparse
from sparse import SparseArray
from .tensor import Tensor
from .tn import TensorNetwork
from .mpo import MatrixProductOperator

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit

# Block2 integration
import block2

DataOptions : TypeAlias = Union[ndarray, SparseArray]

class MatrixProductState(TensorNetwork):
    def __init__(self, tensors : List[Tensor], shape : str="udp") -> "MatrixProductState":
        """
        Constructor for MatrixProductState class.
        
        Args:
            tensors: List of tensors to form the MPS.
            shape (optional): The order of the indices for the tensors. Default is 'udp' (up, down, physical)

        Returns
            An MPS.
        """
        super().__init__(tensors, "MPS")
        self.num_sites = len(tensors)
        self.shape = shape

        internal_inds = self.get_internal_indices()
        external_inds = self.get_external_indices()
        bond_dims = []
        physical_dims = []
        for idx in internal_inds:
            bond_dims.append(self.get_dimension_of_index(idx))
        for idx in external_inds:
            physical_dims.append(self.get_dimension_of_index(idx))
        self.bond_dimension = max(bond_dims)
        self.physical_dimension = max(physical_dims)

    @classmethod
    def from_arrays(cls, arrays : List[DataOptions], shape : str="udp") -> "MatrixProductState":
        """
        Create an MPS from a list of arrays.
        
        Args:
            arrays: The list of arrays.
            shape (optional): The order of the indices for the tensors. Default is 'udp' (up, down, physical)
        
        Returns:
            An MPS.
        """
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
        for a_idx in range(1, len(arrays)-1):
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
    def all_zero_mps(cls, num_sites : int) -> "MatrixProductState":
        """
        Create an MPS for the all zero state |000...0>

        Args:
            num_sites: The number of sites for the MPS

        Returns:
            An MPS.
        """
        zero_end = np.array([1,0], dtype=complex).reshape(1,2)
        zero_middle = np.array([1,0], dtype=complex).reshape(1,1,2)
        arrays = [zero_end] + [zero_middle]*(num_sites-2) + [zero_end]

        return cls.from_arrays(arrays, shape="udp")

    @classmethod
    def random_mps(cls, num_sites : int, bond_dim : int, physical_dim : int) -> "MatrixProductState":
        """
        Create a random MPS.
        
        Args:
            num_sites: The number of sites for the MPS.
            bond_dim: The internal bond dimension to use.
            physical_dim: The physical dimension to use. 

        Returns:
            An MPS.
        """
        arrays = []
        first_array = np.random.rand(bond_dim, physical_dim)
        arrays.append(first_array)

        for _ in range(1, num_sites-1):
            array = np.random.rand(bond_dim, bond_dim, physical_dim)
            arrays.append(array)
        
        last_array = np.random.rand(bond_dim, physical_dim)
        arrays.append(last_array)

        return cls.from_arrays(arrays, shape="udp")

    @classmethod
    def random_quantum_state_mps(cls, num_sites : int, bond_dim : int, physical_dim : int=2) -> "MatrixProductState":
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
    def equal_superposition_mps(cls, num_sites : int) -> "MatrixProductState":
        """
        Create an MPS for the equal superposition state |+++...+>
        
        Args:
            num_sites: The number of sites for the MPS.

        Returns:
            An MPS.
        """
        h_end = np.array([np.sqrt(1/2),np.sqrt(1/2)], dtype=complex).reshape(1,2)
        h_middle = np.array([np.sqrt(1/2),np.sqrt(1/2)], dtype=complex).reshape(1,1,2)
        arrays = [h_end] + [h_middle]*(num_sites-2) + [h_end]
        return cls.from_arrays(arrays, shape="udp")

    @classmethod
    def from_qiskit_circuit(cls, qc : QuantumCircuit, max_bond : int, input_mps : "MatrixProductState"=None) -> "MatrixProductState":
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
        mps = mps.apply_mpo(qc_mpo)
        return mps
    
    def __add__(self, other : "MatrixProductState") -> "MatrixProductState":
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

        data1 = sparse.reshape(t1_data, (t1_dimensions[0]*t1_dimensions[2], t1_dimensions[1]))
        data2 = sparse.reshape(t2_data, (t2_dimensions[0]*t2_dimensions[2], t2_dimensions[1]))

        new_data = sparse.concatenate([data1, data2],axis=1)
        new_data = sparse.moveaxis(new_data, [0,1], [1,0])
        arrays.append(new_data)

        for t_idx in range(1, self.num_sites-1):
            t1 = self.tensors[t_idx]
            t2 = other.tensors[t_idx]

            t1_data = t1.data 
            t2_data = t2.data
            t1_dimensions = t1.dimensions
            t2_dimensions = t2.dimensions

            data1 = sparse.moveaxis(t1_data, [0,1,2], [0,2,1])
            data2 = sparse.moveaxis(t2_data, [0,1,2], [0,2,1])

            data1 = sparse.reshape(data1, (t1_dimensions[0]*t1_dimensions[2], t1_dimensions[1]))
            data2 = sparse.reshape(data2, (t2_dimensions[0]*t2_dimensions[2], t2_dimensions[1]))

            zeros_top_right = sparse.COO.from_numpy(np.zeros((data1.shape[0], data2.shape[1])))
            zeros_bottom_left = sparse.COO.from_numpy(np.zeros((data2.shape[0], data1.shape[1])))

            new_data = sparse.concatenate([sparse.concatenate([data1, zeros_top_right],axis=1), sparse.concatenate([zeros_bottom_left, data2],axis=1)])
            new_data = sparse.moveaxis(new_data, [0,1], [1,0])
            new_data = sparse.reshape(new_data, (t1_dimensions[0]+t2_dimensions[0], t1_dimensions[1]+t2_dimensions[1], t1_dimensions[2]))

            arrays.append(new_data)

        t1 = self.tensors[-1]
        t2 = other.tensors[-1]

        t1_data = t1.data
        t2_data = t2.data
        t1_data = sparse.reshape(t1_data, (t1.dimensions[0], 1, t1.dimensions[1]))
        t2_data = sparse.reshape(t2_data, (t2.dimensions[0], 1, t2.dimensions[1]))
        t1_dimensions = (t1.dimensions[0], 1, t1.dimensions[1])
        t2_dimensions = (t2.dimensions[0], 1, t2.dimensions[1])

        data1 = sparse.reshape(t1_data, (t1_dimensions[0]*t1_dimensions[2], t1_dimensions[1]))
        data2 = sparse.reshape(t2_data, (t2_dimensions[0]*t2_dimensions[2], t2_dimensions[1]))

        new_data = sparse.concatenate([data1, data2],axis=1)
        new_data = sparse.moveaxis(new_data, [0,1], [1,0])
        arrays.append(new_data)

        output = MatrixProductState.from_arrays(arrays)
        return output

    def __sub__(self, other : "MatrixProductState") -> "MatrixProductState":
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

    def reshape(self, shape : str="udp") -> None:
        """
        Reshape the tensors in the MPS.
        
        Args:
            shape (optional): Default is 'udp' (up, down, physical) but any order is allowed.
        """
        first_tensor = self.tensors[0]
        first_current_shape = self.shape.replace("u", "")
        first_new_shape = shape.replace("u", "")
        current_indices = first_tensor.indices
        new_indices = [current_indices[first_current_shape.index(n)] for n in first_new_shape]
        first_tensor.reorder_indices(new_indices)

        for t_idx in range(1, self.num_sites-1):
            t = self.tensors[t_idx]
            current_indices = t.indices
            new_indices = [current_indices[self.shape.index(n)] for n in shape]
            t.reorder_indices(new_indices)

        last_tensor = self.tensors[-1]
        last_current_shape = self.shape.replace("d", "")
        last_new_shape = shape.replace("d", "")
        current_indices = last_tensor.indices
        new_indices = [current_indices[last_current_shape.index(n)] for n in last_new_shape]
        last_tensor.reorder_indices(new_indices)

        self.shape = shape
        return

    def multiply_by_constant(self, const : complex) -> None:
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

    def move_orthogonality_centre(self, where : int=None) -> None:
        """
        Move the orthogonality centre of the MPS.
        
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
            index = internal_indices[idx-1]
            self.compress_index(index, max_bond)
        
        for idx in push_up:
            index = internal_indices[idx-1]
            self.compress_index(index, max_bond, reverse_direction=True)

        return

    def apply_mpo(self, mpo : MatrixProductOperator) -> "MatrixProductState":
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

        for t_idx in range(1, self.num_sites-1):
            t1 = self.tensors[t_idx]
            t2 = mpo.tensors[t_idx]

            t1.indices = ["T1_UP", "T1_DOWN", "TO_CONTRACT"]
            t2.indices = ["T2_UP", "T2_DOWN", "T2_RIGHT", "TO_CONTRACT"]

            tn = TensorNetwork([t1, t2])
            tn.contract_index("TO_CONTRACT")

            tensor = Tensor(tn.tensors[0].data, tn.get_all_indices(), tn.get_all_labels())
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
        return mps

    def compute_inner_product(self, other : "MatrixProductState") -> complex:
        """
        Calculate the inner product with another MPS.

        Args:
            other: The other MPS.

        Returns
            The inner product <self | other>.
        """
        mps1 = copy.deepcopy(self)
        mps2 = copy.deepcopy(other)
        mps1.reshape("udp")
        mps2.reshape("udp")
        mps2.dagger()
        for t in mps2.tensors:
            current_indices = t.indices
            new_indices = [x if x[0]=="P" else x+"_" for x in current_indices]
            t.indices = new_indices
        all_tensors = mps1.tensors + mps2.tensors

        tn = TensorNetwork(all_tensors, "TotalTN")
        for n in range(self.num_sites-1):
            tn.contract_index(f"P{n+1}")
            tn.contract_index(f"B{n+1}")
            tn.combine_indices([f"P{n+2}", f"B{n+1}_"], new_index_name=f"P{n+2}")

        tn.contract_index(f"P{self.num_sites}")
        val = complex(tn.tensors[0].data.flatten()[0])

        return val

    def normalise(self) -> None:
        """
        Normalise the MPS.
        """
        norm = self.compute_inner_product(self).real
        self.multiply_by_constant(np.sqrt(1/norm))
        return
