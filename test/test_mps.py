import numpy as np
from tn4qa.tensor import Tensor
from tn4qa.mps import MatrixProductState
from qiskit import QuantumCircuit

np.random.seed(12)

TEST_ARRAYS = [np.random.rand(4,2), np.random.rand(4,6,2), np.random.rand(6,2)]
contracted_array = np.einsum(TEST_ARRAYS[0], [0,1], np.einsum(TEST_ARRAYS[1], [0,2,3], TEST_ARRAYS[2], [2,4], [0,3,4]), [0,3,4], [1,3,4]).reshape(1,8)

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
    return 

def test_subtract():
    return 

def test_to_sparse_array():
    return 

def test_to_dense_array():
    return 

def test_reshape():
    return 

def test_multiply_by_constant():
    return 

def test_dagger():
    return 

def test_move_orthogonality_centre():
    return 

def test_apply_mpo():
    return 

def test_compute_inner_product():
    return 

def test_sample_configuration():
    return 

def test_to_staircase_circuit():
    return 

def test_warmstart_ansatz_circuit():
    return 

def test_to_preparation_mpo():
    return 

def test_perfect_amplitude_amplification():
    return 

def test_perfect_amplitude_suppression():
    return 

def test_get_grovers_operator():
    return 