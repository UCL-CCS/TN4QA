import numpy as np
from tn4qa.tensor import Tensor
from tn4qa.mps import MatrixProductState

np.random.seed(12)

TEST_ARRAYS = [np.random.rand(4,2), np.random.rand(4,6,2), np.random.rand(6,2)]
contracted_array = np.einsum(TEST_ARRAYS[0], [0,1], np.einsum(TEST_ARRAYS[1], [0,2,3], TEST_ARRAYS[2], [2,4], [0,3,4]), [0,3,4], [1,3,4])

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
    return 

def test_all_zero_mps():
    return 

def test_random_mps():
    return 

def test_random_quantum_state_mps():
    return 

def test_equal_superposition_mps():
    return 

def test_from_qiskit_circuit():
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