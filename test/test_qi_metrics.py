import numpy as np
import pyscf
import pyscf.fci
import json
from pyscf.scf import chkfile
from tn4qa.mps import MatrixProductState
from tn4qa.dmrg import DMRG
from tn4qa.utils import ham_dict_from_scf
from tn4qa.qi_metrics import get_one_orbital_rdm, get_two_orbital_rdm
from test.data.h2_string import civec_to_state
from benchmarking.utils import load_scf_from_chk

# Define 4x4 O matrices
O_matrices = [np.eye(4)[[i]] * 4 for i in range(4)]
O_matrices = [np.zeros((4, 4)).astype(float) for _ in range(16)]
for i in range(4):
    for j in range(4):
        O_matrices[i*4 + j][i, j] = 1

identity = np.eye(4)

# Set a cutoff to remove very small values from the RDMs
cutoff = 1e-15
def truncate_matrix(matrix, precision=10, cutoff=cutoff):
    return np.array([
        [round(c.real, precision) if abs(c) > cutoff else 0 for c in row]
        for row in matrix
    ])

def load_hamiltonian(ham_file_path):
    with open(ham_file_path, "r") as f:
        return json.load(f)

def get_string(scf_file_path: str, charge: int = 0, basis: str = 'sto-3g') -> np.ndarray:
    mol = chkfile.load_mol(scf_file_path)
    rhf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    fci_obj = pyscf.fci.FCI(rhf_obj).run(verbose=0)
    statevector = civec_to_state(ci_obj=fci_obj, zero_threhold=None).to_sparse_matrix.toarray()
    return statevector

def get_mps_from_path(ham_file_path):
    ham = load_hamiltonian(ham_file_path)
    ham = {k:complex(v[0]) for k,v in ham.items()}
    dmrg = DMRG(hamiltonian=ham, max_mps_bond=4, method="one-site")
    dmrg.run(40)
    return dmrg.mps

# Specify your molecules using chk and json paths
mol_dict = {
        'HeH': {
        'chk': 'molecules/scf/sto_3g/HeH.chk',
        'json': 'molecules/hamiltonians/sto_3g/HeH.json',
        'charge': 1
    }
}

basis = 'sto-3g'
ground_states = {}
mps_dict = {}

for mol_name, mol_info in mol_dict.items():
    chk_path = mol_info['chk']
    json_path = mol_info['json']
    ground_states[mol_name] = get_string(scf_file_path=chk_path, charge=mol_info['charge'], basis=basis)
    mps_dict[mol_name] = get_mps_from_path(json_path)

tol = 1e-2

for molecule, wf in ground_states.items():
    print(f"\n--- Comparing RDMs for {molecule} ---")
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
    dmrg_RDM1 = get_one_orbital_rdm(mps, orbital_idx=1)
    dmrg_RDM2 = get_two_orbital_rdm(mps, sites=[1, 2])

    truncated_RDM1 = truncate_matrix(dmrg_RDM1)
    truncated_RDM2 = truncate_matrix(dmrg_RDM2)

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