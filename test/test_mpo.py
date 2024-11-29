import numpy as np 
from numpy import ndarray
import sparse
from sparse import SparseArray
from tensor import Tensor 
from tn import TensorNetwork
from mpo import MatrixProductOperator

np.random.seed(999)

TEST_ARRAY_1 = np.random.rand(2, 2, 2)
TEST_ARRAY_2 = np.random.rand(2, 2, 2)

arrays_valid = [
  sparse.COO.from_numpy(TEST_ARRAY_1),
  sparse.COO.from_numpy(TEST_ARRAY_2),
]

t1 = Tensor(TEST_ARRAY_1, ["A", "B", "C"], ["T1"])
t2 = Tensor(TEST_ARRAY_2, ["A", "E", "F"], ["T2"])


def test_constructor():
  mpo = MatrixProductOperator([t1,t2])
  print(mpo)
  
  # Assert
  assert mpo.num_sites == 2, "Number of sites mismatch"
  assert mpo.shape == "udrl", "Shape mismatch"
  assert mpo.bond_dimension == 2, "Bond dimension not correct"
  assert mpo.physical_dimension == 2, "Physical dimension not correct"

def test_MatrixProductOperator_empty_tensors():
    try:
        MatrixProductOperator([], shape="udrl")
    except ValueError:
      pass
    else:
        assert False, "Expected ValueError for empty tensors"

def test_from_arrays():
    mpo = MatrixProductOperator.from_arrays(arrays_valid, shape="udrl")
    assert mpo.num_sites == 2, "Number of sites mismatch"
    assert mpo.shape == "udrl", "Shape mismatch"
    assert mpo.bond_dimension == 2, "Bond dimension not correct"
    assert mpo.physical_dimension == 2, "Physical dimension not correct"

def test_identity_mpo():
  for n in range(2,10):
    mpo = MatrixProductOperator.identity_mpo(n)
    dense_matrix = mpo.to_dense_array()
    assert dense_matrix.all() == np.identity(n).all(), "Does not return Identity Matrix"

def test_generalised_mcu_mpo():
    return 

def test_from_pauli_string():
    return 

def test_from_hamiltonian():
    return 

def test_from_qiskit_layer():
    return 

def test_from_qiskit_circuit():
    return 

def test_tnqem_mpo_construction():
    return 

def test_zero_reflection_mpo():
    return 

def test_from_bitstring():
    return 

def test_projector_from_samples():
    return 

def test_to_sparse_array():
    # Arrange: Create a valid MPO
    mpo = MatrixProductOperator([t1,t2])
    
    # Act: Convert the MPO to a sparse array
    sparse_matrix = mpo.to_sparse_array()
    print(sparse_matrix)
    
    # Assert: Check the returned object is a SparseArray and is non-empty
    assert isinstance(sparse_matrix, SparseArray), "Output is not a SparseArray"
    assert sparse_matrix.nnz > 0, "Sparse array is empty"

def test_to_dense_array():
    # Arrange: Create a valid MPO
    mpo = MatrixProductOperator([t1,t2])
    
    # Act: Convert the MPO to a dense array
    dense_matrix = mpo.to_dense_array()
    
    # Assert: Check the returned object is a NumPy ndarray and has expected dimensions
    assert isinstance(dense_matrix, np.ndarray), "Output is not a dense NumPy ndarray"
    assert dense_matrix.shape == (4, 4), "Dense matrix dimensions are incorrect"
    assert np.any(dense_matrix), "Dense array is empty or all zeros"

def sparse_and_dense_equivalence():
    mpo = MatrixProductOperator([t1,t2])
    
    # Act: Convert MPO to both sparse and dense arrays
    sparse_matrix = mpo.to_sparse_array()
    dense_matrix = mpo.to_dense_array()
    
    # Assert: Check equivalence between sparse and dense representations
    assert np.allclose(sparse_matrix.todense(), dense_matrix), "Sparse and dense matrix representations are not equivalent"

def test_add():
    return 

def test_subtract():
    return

def test_multiply():
    return 

def test_reshape():
    return 

def test_move_orthogonality_centre():
    return 

def test_project_to_subspace():
    return 

def test_multiply_by_constant():
  for n in range(1,10):
    mpo = MatrixProductOperator([t1,t2])
    dense_matrix = mpo.to_dense_array()
    result_a = dense_matrix * 2
    mpo.multiply_by_constant(2)
    dense_matrix = mpo.to_dense_array()
    result_b = dense_matrix
    assert result_a.all() == result_b.all()

def test_dmrg():
    return 
