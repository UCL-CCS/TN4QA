import numpy as np
from qiskit.quantum_info import Statevector

from tn4qa.mps import MatrixProductState
from tn4qa.tn_methods.mps_to_circuit import MPSAnalyticDecomposition


def reverse_qubits(s):
    n = len(s.data)
    num_qubits = int(np.log2(n))
    new_s = np.zeros((n,), dtype=complex)
    for i in range(n):
        bin_i = bin(i)[2:].zfill(num_qubits)
        reversed_bin_i = bin_i[::-1]
        reversed_i = int(reversed_bin_i, 2)
        s_val = s[reversed_i]
        new_s[i] = s_val
    return new_s


def get_statevec(qcirc):
    s = Statevector.from_instruction(qcirc)
    new_s = reverse_qubits(s)
    return new_s


def test_mps_to_circuit_staircase():
    ## Should check even and odd num_qubits
    num_qubits = 7
    mps = MatrixProductState.random_quantum_state_mps(num_qubits, 2, 2)
    state_exact = mps.to_dense_array()
    mapper = MPSAnalyticDecomposition(mps, max_layers=10, target_fidelity=1 - 1e-2)
    # Convert MPS to Quantum Circuit
    qc_exact = mapper.bond_dim_2_to_qc_exact(mps)
    # Verify that the circuits produce the same state as the MPS
    state_stair = get_statevec(qc_exact)
    assert np.allclose(state_stair.data, state_exact)

    num_qubits = 8
    mps = MatrixProductState.random_quantum_state_mps(num_qubits, 2, 2)
    state_exact = mps.to_dense_array()
    mapper = MPSAnalyticDecomposition(mps, max_layers=10, target_fidelity=1 - 1e-2)
    # Convert MPS to Quantum Circuit
    qc_staircase = mapper.bond_dim_2_to_qc_exact(mps)
    # Verify that the circuits produce the same state as the MPS
    state_stair = get_statevec(qc_staircase)
    assert np.allclose(state_stair.data, state_exact)


def test_mps_to_circuit_middle_out():
    num_qubits = 7
    mps = MatrixProductState.random_quantum_state_mps(num_qubits, 2, 2)
    state_exact = mps.to_dense_array()
    mapper = MPSAnalyticDecomposition(mps, max_layers=10, target_fidelity=1 - 1e-2)
    # Convert MPS to Quantum Circuit
    qc_middle_out = mapper.bond_dim_2_to_qc_middle_out(mps)
    # Verify that the circuits produce the same state as the MPS
    state_middle_out = get_statevec(qc_middle_out)

    assert np.allclose(state_middle_out.data, state_exact)
