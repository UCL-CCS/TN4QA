import numpy as np
from tn4qa.tensor import Tensor
from tn4qa.mps import MatrixProductState
from tn4qa.mpo import MatrixProductOperator
from qiskit import QuantumCircuit

np.random.seed(12)

TEST_ARRAYS = [np.random.rand(4,2), np.random.rand(4,6,2), np.random.rand(6,2)]
CONTRACTED_TEST_ARRAY = np.einsum(TEST_ARRAYS[0], [0,1], np.einsum(TEST_ARRAYS[1], [0,2,3], TEST_ARRAYS[2], [2,4], [0,3,4]), [0,3,4], [1,3,4]).reshape(1,8)


def test_constructor():
    t1 = Tensor(TEST_ARRAYS[0], ["B1", "P1"], ["MPS_T1"])
    t2 = Tensor(TEST_ARRAYS[1], ["B1", "B2", "P2"], ["MPS_T2"])
    t3 = Tensor(TEST_ARRAYS[2], ["B2", "P3"], ["MPS_T3"])
    tensors = [t1,t2,t3]
    mps = MatrixProductState(tensors)

    assert mps.name == "MPS"
    assert mps.shape == "udp"
    assert mps.physical_dimension == 2
    assert mps.bond_dimension == 6
    assert mps.num_sites == 3
    assert all(x in mps.indices for x in ["B1", "B2", "P1", "P2", "P3"])

    return 

def test_from_arrays():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)

    assert mps.name == "MPS"
    assert mps.shape == "udp"
    assert mps.physical_dimension == 2
    assert mps.bond_dimension == 6
    assert mps.num_sites == 3
    assert all(x in mps.indices for x in ["B1", "B2", "P1", "P2", "P3"])

    return 

def test_all_zero_mps():
    mps = MatrixProductState.all_zero_mps(8)

    assert mps.num_sites == 8
    assert mps.bond_dimension == 1
    assert mps.physical_dimension == 2
    for tensor in mps.tensors:
        assert np.array_equal(tensor.data.todense().flatten(), np.array([1,0]))
    
    return 

def test_random_mps():
    mps = MatrixProductState.random_mps(12, 8, 2)

    assert mps.num_sites == 12
    assert mps.physical_dimension == 2
    assert mps.bond_dimension == 8
    assert len(mps.indices) == 23
    assert all(x in mps.indices for x in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"])
    assert all(x in mps.indices for x in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"])

    return 

def test_random_quantum_state_mps():
    mps = MatrixProductState.random_quantum_state_mps(3, 4)
    contracted_mps = np.einsum(mps.tensors[0].data.todense(), [0,1], np.einsum(mps.tensors[1].data.todense(), [0,2,3], mps.tensors[2].data.todense(), [2,4], [0,3,4]), [0,3,4], [1,3,4]).reshape(1,8)
    inner_prod = np.dot(contracted_mps, contracted_mps.conj().T)

    assert np.isclose(inner_prod, 1.0)

    return 

def test_equal_superposition_mps():
    mps = MatrixProductState.equal_superposition_mps(3)
    contracted_mps = np.einsum(mps.tensors[0].data.todense(), [0,1], np.einsum(mps.tensors[1].data.todense(), [0,2,3], mps.tensors[2].data.todense(), [2,4], [0,3,4]), [0,3,4], [1,3,4]).reshape(1,8)
    inner_prod = np.dot(contracted_mps, contracted_mps.conj().T)

    assert np.isclose(inner_prod, 1.0)

    expected_state = np.array([[np.sqrt(1/2), np.sqrt(1/2)]])
    for tensor in mps.tensors:
        assert np.allclose(tensor.data.todense().flatten(), expected_state)

    return 

def test_from_qiskit_circuit_1():
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.h(i)
    mps = MatrixProductState.from_qiskit_circuit(qc, 4)

    expected_state = np.array([[np.sqrt(1/2), np.sqrt(1/2)]])
    for tensor in mps.tensors:
        assert np.allclose(tensor.data.todense().flatten(), expected_state)

def test_from_qiskit_circuit_2():
    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(5):
        qc.cx(i, i+1)
    mps = MatrixProductState.from_qiskit_circuit(qc, 2)

    output = mps.contract_entire_network()
    output.combine_indices(["P1", "P2", "P3", "P4", "P5", "P6"])
    output_data = output.data.todense()

    assert np.isclose(output_data[0], np.sqrt(1/2))
    assert np.isclose(output_data[63], np.sqrt(1/2))
    for i in range(1, 63):
        assert np.isclose(output_data[i], 0.0)

    return 

def test_add():
    mps1 = MatrixProductState.all_zero_mps(4)
    mps2 = MatrixProductState.equal_superposition_mps(4)
    output = mps1 + mps2 
    contracted = output.contract_entire_network()
    contracted.combine_indices(["P1", "P2", "P3", "P4"])
    output_data = contracted.data.todense()

    assert np.isclose(output_data[0], 1.25)
    for i in range(1, 16):
        assert np.isclose(output_data[i], 0.25)

    return 

def test_subtract():
    mps1 = MatrixProductState.all_zero_mps(4)
    mps2 = MatrixProductState.equal_superposition_mps(4)
    output = mps2 - mps1
    contracted = output.contract_entire_network()
    contracted.combine_indices(["P1", "P2", "P3", "P4"])
    output_data = contracted.data.todense()

    assert np.isclose(output_data[0], -0.75)
    for i in range(1, 16):
        assert np.isclose(output_data[i], 0.25)

    return 

def test_to_sparse_array():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)
    sparse_array = mps.to_sparse_array()

    assert np.allclose(sparse_array.todense().flatten(), CONTRACTED_TEST_ARRAY.flatten())

    return 

def test_to_dense_array():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)
    sparse_array = mps.to_dense_array()
    
    assert np.allclose(sparse_array.flatten(), CONTRACTED_TEST_ARRAY.flatten())

    return 

def test_reshape():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)
    mps.reshape("pud")

    expected_output = [
        np.moveaxis(TEST_ARRAYS[0], [0,1], [1,0]),
        np.moveaxis(TEST_ARRAYS[1], [0,1,2], [1,2,0]),
        np.moveaxis(TEST_ARRAYS[2], [0,1], [1,0])
    ]

    for i in range(3):
        assert np.allclose(mps.tensors[i].data.todense(), expected_output[i])
    
    return 

def test_multiply_by_constant():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)
    mps.multiply_by_constant(-3.7+1.2j)

    expected_output = [
        TEST_ARRAYS[0] * (-3.7+1.2j),
        TEST_ARRAYS[1],
        TEST_ARRAYS[2]
    ]

    for i in range(3):
        assert np.allclose(mps.tensors[i].data.todense(), expected_output[i])

    return 

def test_dagger():
    mps = MatrixProductState.from_arrays(TEST_ARRAYS)
    mps.dagger()

    contracted = mps.contract_entire_network()
    contracted.combine_indices(["P1", "P2", "P3"])
    output_data = contracted.data.todense()

    expected_output = CONTRACTED_TEST_ARRAY.conj().T

    assert np.allclose(output_data.flatten(), expected_output.flatten())

    return 

def test_move_orthogonality_centre():
    mps = MatrixProductState.random_mps(4, 4, 2)
    mps.move_orthogonality_centre(2)

    t = mps.tensors[0]
    t.tensor_to_matrix([t.indices[1]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat)
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T)

    t = mps.tensors[2]
    t.tensor_to_matrix([t.indices[1], t.indices[2]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat)
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T)

    t = mps.tensors[3]
    t.tensor_to_matrix([t.indices[1]], [t.indices[0]])
    t_mat = t.data.todense()
    if t.dimensions[0] >= t.dimensions[1]:
        id_mat = np.eye(t.dimensions[1])
        assert np.allclose(id_mat, t_mat.conj().T @ t_mat)
    else:
        id_mat = np.eye(t.dimensions[0])
        assert np.allclose(id_mat, t_mat @ t_mat.conj().T)

    return 

def test_apply_mpo_1():
    mps = MatrixProductState.equal_superposition_mps(4)
    qc = QuantumCircuit(4)
    qc.h([0,1,2,3])
    mpo = MatrixProductOperator.from_qiskit_circuit(qc, 4)

    output = mps.apply_mpo(mpo).to_dense_array()
    expected_output = np.array([0]*16, dtype=complex)
    expected_output[0] = 1

    assert np.allclose(output, expected_output)

    return 

def test_apply_mpo_2():
    mps = MatrixProductState.all_zero_mps(12)
    qc = QuantumCircuit(12)
    qc.h(0)
    for i in range(11):
        qc.cx(i,i+1)
    mpo = MatrixProductOperator.from_qiskit_circuit(qc, 64)
    output = mps.apply_mpo(mpo).to_dense_array()

    expected_output = np.array([0]*(2**12), dtype=complex)
    expected_output[0] = np.sqrt(1/2)
    expected_output[-1] = np.sqrt(1/2)

    print(output)

    assert np.allclose(output, expected_output, atol=0.1)

    return

def test_compute_inner_product():
    mps1 = MatrixProductState.all_zero_mps(5)
    mps2 = MatrixProductState.equal_superposition_mps(5)

    prod1 = mps1.compute_inner_product(mps1)
    prod2 = mps2.compute_inner_product(mps2)
    prod3 = mps1.compute_inner_product(mps2)
    prod4 = mps2.compute_inner_product(mps1)

    assert np.isclose(prod1, 1.0)
    assert np.isclose(prod2, 1.0)
    assert np.isclose(prod3, np.sqrt(1/2**5))
    assert np.isclose(prod4, np.sqrt(1/2**5))

    return 

def test_normalise():
    mps = MatrixProductState.random_mps(5, 4, 2)
    mps.normalise() 

    prod = mps.compute_inner_product(mps)
    assert np.isclose(prod, 1.0)
    