import numpy as np 
import sparse
from sparse import SparseArray
from tn4qa.tensor import Tensor
from tn4qa.mpo import MatrixProductOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

np.random.seed(999)

# Define test arrays
TEST_ARRAY_1 = np.random.rand(2, 2, 2)
TEST_ARRAY_2 = np.random.rand(2, 2, 2)
TEST_ARRAYS = [TEST_ARRAY_1, TEST_ARRAY_2]

# Convert test arrays to sparse
arrays_valid = [
  sparse.COO.from_numpy(TEST_ARRAY_1),
  sparse.COO.from_numpy(TEST_ARRAY_2),
]

# Initialise tensors
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
    # Test creating an MPO with empty tensors
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
  # Test identity MPO creation
  for n in range(2,10):
    mpo = MatrixProductOperator.identity_mpo(n)
    dense_matrix = mpo.to_dense_array()
    assert dense_matrix.all() == np.identity(n).all(), f"Does not return Identity Matrix for size {n}"
    return 

def test_generalised_mcu_mpo():
    # Test for single qubit gate: X Gate 
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

    assert np.allclose(expected_array, mpo_dense), "Generalised MCU MPO output mismatch"
    
    return 

def test_from_pauli_string():
    mpo = MatrixProductOperator.from_pauli_string("XYZIZYIX")
    mpo_dense = mpo.to_dense_array()

    xmat = np.array([[0,1],[1,0]], dtype=complex)
    ymat = np.array([[0,-1j],[1j,0]], dtype=complex)
    zmat = np.array([[1,0],[0,-1]], dtype=complex)
    idmat = np.array([[1,0],[0,1]], dtype=complex)

    expected_output = np.kron(xmat, np.kron(ymat, np.kron(zmat, np.kron(idmat, np.kron(zmat, np.kron(ymat, np.kron(idmat, xmat)))))))

    assert np.allclose(expected_output, mpo_dense), "Pauli string MPO output mismatch"

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

    assert np.allclose(expected_output, mpo_dense), "Hamiltomian MPO output mismatch"
    return 

def test_from_qiskit_layer():
    qc = QuantumCircuit(8)
    qc.h(0)      # H Gate
    qc.x(1)      # X Gate
    qc.cx(2,3)   # CX Gate
    qc.cz(4,5)   # CZ Gate
    qc.h(6)      # H Gate
    qc.y(7)      # Y Gate
    expected_op = Operator.from_circuit(qc).reverse_qargs().data
    mpo = MatrixProductOperator.from_qiskit_layer(qc)
    mpo_dense = mpo.to_dense_array()
    assert np.allclose(mpo_dense, expected_op), "Qiskit Layer MPO output mismatch"

    return 

def test_from_qiskit_circuit():
    qc = QuantumCircuit(5)
    for _ in range(5):
        qc.h([0,1,2])  # H Gate
        qc.x([3,4])    # X Gate
        qc.cx(0,1)     # CX Gate 
        qc.cz(2,3)     # CZ Gate 
        qc.cx(3,4)     # CX Gate 
        qc.z([0,1])    # Z Gate 
        qc.h([2,3,4])  # H Gate 
        
    expected_op = Operator.from_circuit(qc).reverse_qargs().data
    mpo = MatrixProductOperator.from_qiskit_circuit(qc, 64)
    mpo_dense = mpo.to_dense_array()
    assert np.allclose(mpo_dense, expected_op), "Qiskit Circuit MPO output mismatch"
    return 

def test_zero_reflection_mpo():
    mpo = MatrixProductOperator.zero_reflection_mpo(8)
    mpo_dense = mpo.to_dense_array()
    expected = np.eye(2**8)
    expected[0][0] = -1
    assert np.allclose(expected, mpo_dense), "Zero Reflection MPO output mismatch"
    return 

def test_from_bitstring():
    bs = "01101001"
    mpo = MatrixProductOperator.from_bitstring(bs)
    mpo_dense = mpo.to_dense_array()

    bs_int = int(bs, 2)
    expected = np.zeros((2**8, 2**8))
    expected[bs_int][bs_int] = 1

    assert np.allclose(mpo_dense, expected), "Bitstring MPO output mismatch"
    return 

def test_projector_from_samples():
    # Test creation of a projector MPO from a list of bitstrings
    list_bs = ["010", "111", "101"]
    expected = np.zeros((8,8))
    for bs in list_bs:
        bs_int = int(bs,2)
        expected[bs_int][bs_int] = 1
    mpo = MatrixProductOperator.projector_from_samples(list_bs, 8)
    mpo_dense = mpo.to_dense_array()
    assert np.allclose(mpo_dense, expected), "Projector MPO does not match the expected result"
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
    # Test addition of two MPOs
    ham1 = {"XX" : 0.2, "YY" : 0.3}
    ham2 = {"ZZ" : 0.4, "II" : 0.5}
    total_ham = {"XX" : 0.2, "YY" : 0.3, "ZZ" : 0.4, "II" : 0.5}
    mpo1 = MatrixProductOperator.from_hamiltonian(ham1, 8)
    mpo2 = MatrixProductOperator.from_hamiltonian(ham2, 8)

    out = mpo1 + mpo2
    expected = MatrixProductOperator.from_hamiltonian(total_ham, 8)

    assert np.allclose(out.to_dense_array(), expected.to_dense_array()), "MPO addition result is incorrect"
    return 

def test_subtract():
    # Test subtraction of two MPOs
    ham1 = {"XX" : 0.2, "YY" : 0.3}
    ham2 = {"ZZ" : 0.4, "II" : 0.5}
    total_ham = {"XX" : 0.2, "YY" : 0.3, "ZZ" : 0.4, "II" : 0.5}
    mpo1 = MatrixProductOperator.from_hamiltonian(ham1, 8)
    mpo2 = MatrixProductOperator.from_hamiltonian(total_ham, 8)

    out = mpo2 - mpo1
    expected = MatrixProductOperator.from_hamiltonian(ham2, 8)

    assert np.allclose(out.to_dense_array(), expected.to_dense_array()), "MPO subtraction result is incorrect"
    return

def test_multiply():
    # Test multiplication of two MPOs
    qc1 = QuantumCircuit(4)
    qc1.h([0,1,2,3])    # H Gate
    qc1.cx(0,1)         # CX Gate
    qc1.cx(2,3)         # CX Gate
    qc1.x([0,1,2,3])    # X Gate

    qc2 = QuantumCircuit(4)
    qc2.h([0,1,2,3])    # H Gate
    qc2.cz(0,1)         # CZ Gate
    qc2.cz(2,3)         # CZ Gate
    qc2.y([0,1,2,3])    # Y Gate

    totalqc = qc1.compose(qc2)

    mpo1 = MatrixProductOperator.from_qiskit_circuit(qc1, 8)
    mpo2 = MatrixProductOperator.from_qiskit_circuit(qc2, 8)

    out = mpo1 * mpo2
    expected = MatrixProductOperator.from_qiskit_circuit(totalqc, 64)

    assert np.allclose(out.to_dense_array(), expected.to_dense_array()), "MPO multiplication result is incorrect"
    return 

def test_reshape():
    mpo = MatrixProductOperator.from_arrays(TEST_ARRAYS)
    mpo.reshape("rudl")

    TEST_ARRAYS_RESHAPED = [np.moveaxis(TEST_ARRAY_1, [0,1,2], [1,0,2]), np.moveaxis(TEST_ARRAY_2, [0,1,2], [1,0,2])]
    new_mpo = MatrixProductOperator.from_arrays(TEST_ARRAYS_RESHAPED)

    for tidx in range(2):
        assert np.allclose(mpo.tensors[tidx].data.todense(), new_mpo.tensors[tidx].data.todense()), f"Reshaping failed for tensor {tidx}"

    return 

def test_move_orthogonality_centre():
    # Test moving the orthogonality centre of an MPO
    ham = {"XYXY" : 0.8, "IZZZ" : -1.2, "XYIZ" : 0.2-1.1j}
    mpo = MatrixProductOperator.from_hamiltonian(ham, 8)
    # Move the orthogonality centre to the tensor at position 2
    mpo.move_orthogonality_centre(2)

    t = mpo.tensors[0]
    t.tensor_to_matrix([t.indices[1], t.indices[2]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat), "Tensor 0 is not left-orthogonal after moving orthogonality centre"
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T), "Tensor 0 is not right-orthogonal after moving orthogonality centre"

    t = mpo.tensors[2]
    t.tensor_to_matrix([t.indices[1], t.indices[2], t.indices[3]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat), "Tensor 2 is not left-orthogonal after moving orthogonality centre"
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T), "Tensor 2 is not right-orthogonal after moving orthogonality centre"

    t = mpo.tensors[3]
    t.tensor_to_matrix([t.indices[1], t.indices[2]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat), "Tensor 3 is not left-orthogonal after moving orthogonality centre"
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T), "Tensor 3 is not right-orthogonal after moving orthogonality centre"

    return 

def test_project_to_subspace():
    # Test projection of an MPO Hamiltonian onto a subspace
    ham = {"XYZ" : 0.2+1.1j, "IXI" : 1.1-0.8j, "YYY" : -0.8-0.2j}
    bs_list = ["101", "001"]
    mpo = MatrixProductOperator.from_hamiltonian(ham, 8)
    proj = MatrixProductOperator.projector_from_samples(bs_list, 8)
    # Project the MPO onto the subspace and get the dense representation
    mpo = mpo.project_to_subspace(proj) 
    mpo_dense = mpo.to_dense_array()

    # Manually compute the expected result
    xmat = np.array([[0,1],[1,0]],dtype=complex)
    ymat = np.array([[0,-1j],[1j,0]],dtype=complex)
    zmat = np.array([[1,0],[0,-1]],dtype=complex)
    idmat = np.array([[1,0],[0,1]],dtype=complex)
    ham_mat = (0.2+1.1j)*np.kron(xmat, np.kron(ymat,zmat)) + (1.1-0.8j)*np.kron(idmat, np.kron(xmat,idmat)) + (-0.8-0.2j)*np.kron(ymat, np.kron(ymat,ymat))
    proj_mat = np.zeros((8,8))
    for bs in bs_list:
        bs_int = int(bs,2)
        proj_mat[bs_int][bs_int] = 1
    expected = proj_mat @ ham_mat @ proj_mat 

    assert np.allclose(expected, mpo_dense), "Projected MPO does not match the expected dense matrix"
    return 

def test_multiply_by_constant():
  # Test scalar multiplication of an MPO
  for n in range(1,10):
    mpo = MatrixProductOperator([t1,t2])
    dense_matrix = mpo.to_dense_array()
    result_a = dense_matrix * n
    mpo.multiply_by_constant(n)
    dense_matrix = mpo.to_dense_array()
    result_b = dense_matrix
    assert result_a.all() == result_b.all(), "Multiply by constant test failed"

    return 
