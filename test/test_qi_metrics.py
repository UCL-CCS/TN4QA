import numpy as np
import pyscf
from tn4qa.mps import MatrixProductState
from tn4qa.dmrg import DMRG
from tn4qa.utils import ham_dict_from_scf
from tn4qa.qi_metrics import get_one_orbital_rdm, get_two_orbital_rdm
from test.data.h2_string import get_h2_string

# Define 4x4 O matrices
O_matrices = []
for i in range(4):
    for j in range(4):
        O_matrix = np.zeros((4, 4))
        O_matrix[i,j] = 1
        O_matrices.append(O_matrix)

# Ground state wavefunctions - Tom's code
H2_array = get_h2_string(1.0, 'sto-3g')
ground_states = {
    "H2": np.array(H2_array)
}

# Define the ground state wavefunction - DMRG way
def get_mps_from_mf(mf):
    ham = ham_dict_from_scf(mf, qubit_transformation="JW")
    dmrg = DMRG(hamiltonian=ham, max_mps_bond=16, method="one-site")
    dmrg.run(20)  # 20 DMRG sweeps
    return dmrg.mps

identity = np.eye(4)


for molecule, wf in ground_states.items():

    # Normalise wavefunction
    wf = wf / np.linalg.norm(wf)
    print(f"\nRDM1 for {molecule}:")
    
    # Build RDM1 (4x4 matrix)
    RDM1 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            O_index = i * 4 + j
            RDM1[i, j] = np.real(np.conj(wf).T @ np.kron(O_matrices[O_index], identity) @ wf)
    print(RDM1)


    # Build RDM2 as pairwise expectation values
    RDM2 = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            RDM2[i, j] = np.real(np.conj(wf).T @ (np.kron(O_matrices[int(4*int(np.floor(i/4))+np.floor(j/4))], O_matrices[int(4*(i%4)+(j%4))])) @ wf) 
    print(f"\nRDM2 for {molecule}:")
    print(RDM2)

#-----------------------------------

mol = pyscf.M(
    atom="H 0 0 0; H 0 0 1", 
    basis="sto-3g",
)
mf = pyscf.scf.RHF(mol).run()
mps_H2 = get_mps_from_mf(mf)


# Get the one and two orbital RDMs using tn4qa
our_RDM1_H2 = get_one_orbital_rdm(mps_H2, orbital_idx=1)
our_RDM2_H2 = get_two_orbital_rdm(mps_H2, sites=[1,2])

# Set a cutoff to remove very small values from the RDMs bc they were too big
cutoff = 1e-15
def truncate_matrix(matrix, precision=5, cutoff=cutoff):
    truncated_matrix = np.array([
        [round(c.real, precision) if abs(c) > cutoff else 0 
         for c in row] 
        for row in matrix
    ])
    return truncated_matrix
truncated_RDM1 = truncate_matrix(our_RDM1_H2)
truncated_RDM2 = truncate_matrix(our_RDM2_H2)

# Print results
print("\ntn4qa RDM1:\n", truncated_RDM1)
print("\ntn4qa RDM2:\n", truncated_RDM2)

# Compare with Tom's RDMs
tol = 1e-5
def test_rdm1_consistency():
    # Check if they are the same
    try:
        np.testing.assert_allclose(RDM1, truncated_RDM1, atol=tol)
        print("Test passed: Tom's RDM1 and DMRG RDM1 are identical within tolerance", tol)
    except AssertionError as e:
        difference = RDM1 - truncated_RDM1
        print("Test failed: Tom's RDM1 and DMRG RDM1 are not identical.")
        print("Maximum absolute difference:", np.max(np.abs(difference)))
        print("Difference matrix:\n", difference)
        raise e

def test_rdm2_consistency():
    # Check if they are the same
    try:
        np.testing.assert_allclose(RDM2, truncated_RDM2, atol=tol)
        print("Test passed: Tom's RDM2 and DMRG RDM2 are identical within tolerance", tol)
    except AssertionError as e:
        difference = RDM2 - truncated_RDM2
        print("Test failed: Tom's RDM2 and DMRG RDM2 are not identical.")
        print("Maximum absolute difference:", np.max(np.abs(difference)))
        print("Difference matrix:\n", difference)
        raise e

test_rdm1_consistency()
test_rdm2_consistency()
