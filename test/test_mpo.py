import numpy as np 
import sparse
from sparse import SparseArray
from tn4qa.tensor import Tensor
from tn4qa.mpo import MatrixProductOperator

np.random.seed(999)

TEST_ARRAY_1 = np.random.rand(2, 2, 2)
TEST_ARRAY_2 = np.random.rand(2, 2, 2)
TEST_ARRAYS = [TEST_ARRAY_1, TEST_ARRAY_2]

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
  assert mpo.bond_dimension == 2, "Bond dimension not set"
  assert mpo.physical_dimension == 2, "Physical dimension not set"

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
    assert mpo.bond_dimension == 2, "Bond dimension not set"
    assert mpo.physical_dimension == 2, "Physical dimension not set"

def test_identity_mpo():
    mpo = MatrixProductOperator.identity_mpo(8)
    id_mat = np.eye(2**8)

    mpo_dense = mpo.to_dense_array()
    assert np.allclose(mpo_dense, id_mat)

    return 

def test_generalised_mcu_mpo():
    x_gate = np.array([[0,1],[1,0]])
    mpo = MatrixProductOperator.generalised_mcu_mpo(4, [1], [3], 2, x_gate)
    mpo_dense = mpo.to_dense_array()

    expected_array = np.kron(np.array([
        [1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ]), np.eye(2))

    assert np.allclose(expected_array, mpo_dense)
    
    return 

def test_from_pauli_string():
    mpo = MatrixProductOperator.from_pauli_string("XYZIZYIX")
    mpo_dense = mpo.to_dense_array()

    xmat = np.array([[0,1],[1,0]], dtype=complex)
    ymat = np.array([[0,-1j],[1j,0]], dtype=complex)
    zmat = np.array([[1,0],[0,-1]], dtype=complex)
    idmat = np.array([[1,0],[0,1]], dtype=complex)

    expected_output = np.kron(xmat, np.kron(ymat, np.kron(zmat, np.kron(idmat, np.kron(zmat, np.kron(ymat, np.kron(idmat, xmat)))))))

    assert np.allclose(expected_output, mpo_dense)

    return 

def test_from_hamiltonian():
    ham = {"IXIY" : -1.2+0.2j, "YYYX" : 0.4-0.8j, "ZIXX" : 1.1-0.1j}
    mpo = MatrixProductOperator.from_hamiltonian(ham, 8)
    mpo_dense = mpo.to_dense_array()

    xmat = np.array([[0,1],[1,0]], dtype=complex)
    ymat = np.array([[0,-1j],[1j,0]], dtype=complex)
    zmat = np.array([[1,0],[0,-1]], dtype=complex)
    idmat = np.array([[1,0],[0,1]], dtype=complex)

    expected_output = (-1.2+0.2j)*np.kron(idmat, np.kron(xmat, np.kron(idmat,ymat))) + (0.4-0.8j)*np.kron(ymat, np.kron(ymat,np.kron(ymat,xmat))) + (1.1-0.1j)*np.kron(zmat, np.kron(idmat,np.kron(xmat,xmat)))

    assert np.allclose(expected_output, mpo_dense)
    return 

def test_from_qiskit_layer():
    return 

def test_from_qiskit_circuit():
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
    mps = MatrixProductOperator.from_arrays(TEST_ARRAYS)
    mps.multiply_by_constant(-3.7+1.2j)

    expected_output = [
        TEST_ARRAYS[0] * (-3.7+1.2j),
        TEST_ARRAYS[1],
    ]

    for i in range(2):
        assert np.allclose(mps.tensors[i].data.todense(), expected_output[i])

    return 
