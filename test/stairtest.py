import numpy as np
from qiskit import QuantumCircuit
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


num_qubits = 7
# Create an MPS


def GHZ():
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


# Convert GHZ state to MPS
# mps = MatrixProductState.from_qiskit_circuit(GHZ())
mps = MatrixProductState.random_quantum_state_mps(num_qubits, 2, 2)

state_exact = mps.to_dense_array()

mapper = MPSAnalyticDecomposition(mps, max_layers=10, target_fidelity=1 - 1e-2)

# Convert MPS to Quantum Circuit
qc_exact = mapper.bond_dim_2_to_qc_exact(mps)
qc_parallel = mapper.bond_dim_2_to_qc_middle_out(mps)


print("Depth (staircase):", qc_exact.depth())
print("Depth (parallel): ", qc_parallel.depth())

print("\nStaircase circuit:")
print(qc_exact)
print("\nParallel circuit:")
print(qc_parallel)

# Verify that the circuits produce the same state as the MPS
state_stair = get_statevec(qc_exact)
state_parallel = get_statevec(qc_parallel)
print("State from exact:", state_exact.round(3))
print("State from staircase:", state_stair.round(3))
print("State from parallel: ", state_parallel.round(3))

if np.allclose(state_stair.data, state_parallel.data):
    print("\nThe two circuits produce the same state.")
else:
    print("\nThe two circuits produce different states.")
