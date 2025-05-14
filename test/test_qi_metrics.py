import numpy as np
import pyscf
from tn4qa.mps import MatrixProductState
from tn4qa.dmrg import DMRG
from tn4qa.utils import ham_dict_from_scf
from tn4qa.qi_metrics import get_one_orbital_rdm, get_two_orbital_rdm
from test.data.h2_string import get_string

# Define 4x4 O matrices
O_matrices = []
for i in range(4):
    for j in range(4):
        O_matrix = np.zeros((4, 4))
        O_matrix[i,j] = 1
        O_matrices.append(O_matrix)

identity = np.eye(4)
#-----------------------------------
# Set a cutoff to remove very small values from the RDMs bc they were too big
cutoff = 1e-15
def truncate_matrix(matrix, precision=10, cutoff=cutoff):
    truncated_matrix = np.array([
        [round(c.real, precision) if abs(c) > cutoff else 0 
         for c in row] 
        for row in matrix
    ])
    return truncated_matrix
#-----------------------------------

mol_dict = {
    'H2': {
        'xyz_path': 'test/data/mols/h2.xyz',
        'charge': 0
    },
    'HeH': {
        'xyz_path': 'test/data/mols/heh.xyz',
        'charge': 1
    }
}

# Define the ground state wavefunction - DMRG way
basis = 'STO-3G'
ground_states = {}
mps_dict = {}

for mol_name, mol_info in mol_dict.items():
    # Create the molecule
    mymol = pyscf.M(
        atom=mol_info['xyz_path'],
        charge=mol_info['charge'],
        basis=basis
    )

    def get_mps_from_mol(mymol):
        mf = pyscf.scf.RHF(mymol).run()
        ham = ham_dict_from_scf(mf, qubit_transformation="JW")
        dmrg = DMRG(hamiltonian=ham, max_mps_bond=4, method="one-site")
        dmrg.run(40)  # 20 DMRG sweeps
        return dmrg.mps
    
    # Generate the ground state wavefunction
    array = get_string(xyz_path=mol_info['xyz_path'], charge=mol_info['charge'], basis=basis)
    ground_states[mol_name] = np.array(array)

    mps_dict[mol_name] = get_mps_from_mol(mymol)

# Set tolerance for comparison
tol = 1e-2

for molecule, wf in ground_states.items():
    # Make sure the file paths match for comparison
    expected_path = mol_dict[molecule]['xyz_path']
    print(f"Comparing {molecule} with expected path: {expected_path}")

    # Normalise wavefunction
    wf = wf / np.linalg.norm(wf)

    # Build RDM1 (4x4 matrix)
    RDM1 = np.zeros((4, 4))
    for i in range(4):
            for j in range(4):
                O_index = i * 4 + j
                RDM1[i, j] = np.real(np.conj(wf).T @ np.kron(O_matrices[O_index], identity) @ wf)

    # Build RDM2 (16x16 matrix)
    RDM2 = np.zeros((16, 16))
    for i in range(16):
            for j in range(16):
                RDM2[i, j] = np.real(np.conj(wf).T @ (np.kron(O_matrices[int(4*int(np.floor(i/4))+np.floor(j/4))], O_matrices[int(4*(i%4)+(j%4))])) @ wf)

    mps = mps_dict[molecule]
    # Get the one and two orbital RDMs using tn4qa
    dmrg_RDM1 = get_one_orbital_rdm(mps, orbital_idx=1)
    dmrg_RDM2 = get_two_orbital_rdm(mps, sites=[1,2])

    # Truncate small values
    truncated_RDM1 = truncate_matrix(dmrg_RDM1)
    truncated_RDM2 = truncate_matrix(dmrg_RDM2)

    #print("\ntn4qa RDM1 for", molecule ,":\n", truncated_RDM1)
    #print("\ntn4qa RDM2 for", molecule ,":\n", truncated_RDM2)

    # Compare with Tom's RDMs
    def test_rdm1_consistency(molecule):
        # Check if they are the same
        try:
            np.testing.assert_allclose(RDM1, truncated_RDM1, atol=tol)
            print("Test passed: Tom's RDM1 and DMRG RDM1 are identical within tolerance", tol, "for molecule:", molecule)
        except AssertionError as e:
            difference = RDM1 - truncated_RDM1
            print("Test failed: Tom's RDM1 and DMRG RDM1 are not identical for molecule:", molecule)
            print("Maximum absolute difference:", np.max(np.abs(difference)))
            print("Difference matrix:\n", difference)
            raise e

    def test_rdm2_consistency(molecule):
        # Check if they are the same
        try:
            np.testing.assert_allclose(RDM2, truncated_RDM2, atol=tol)
            print("Test passed: Tom's RDM2 and DMRG RDM2 are identical within tolerance", tol, "for molecule:", molecule)
        except AssertionError as e:
            difference = RDM2 - truncated_RDM2
            print("Test failed: Tom's RDM2 and DMRG RDM2 are not identical for molecule:", molecule)
            print("Maximum absolute difference:", np.max(np.abs(difference)))
            print("Difference matrix:\n", difference)
            raise e

    test_rdm1_consistency(molecule)
    test_rdm2_consistency(molecule)
