import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from tn4qa.mps import MatrixProductState
from tn4qa.tn_methods.mps_to_circuit import MPSAnalyticDecomposition

num_qubits = 6
# Create a random MPS
#mps = MatrixProductState.equal_superposition_mps(num_qubits)

def GHZ():
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

# Convert GHZ state to MPS
mps = MatrixProductState.from_qiskit_circuit(GHZ())

mapper = MPSAnalyticDecomposition(mps, max_layers=10, target_fidelity=1-1e-2)

# Convert MPS to Quantum Circuit
#qc_exact = mapper.bond_dim_2_to_qc_exact(mps)
qc_parallel = mapper.bond_dim_2_to_qc_parallel(mps)

#print("Depth (staircase):", qc_exact.depth())
print("Depth (parallel): ", qc_parallel.depth())

#print("\nStaircase circuit:")
#print(qc_exact)
print("\nParallel circuit:")
print(qc_parallel)

# Verify that the circuits produce the same state as the MPS
#state_exact = Statevector.from_instruction(qc_exact)
state_parallel = Statevector.from_instruction(qc_parallel)
#print("State from staircase:", state_exact.data.round(3))
print("State from parallel: ", state_parallel.data.round(3))

#if np.allclose(state_exact.data, state_parallel.data):
#    print("\nThe two circuits produce the same state.")
#else:
#    print("\nThe two circuits produce different states.")