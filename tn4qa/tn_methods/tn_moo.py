from tn4qa.tn_methods.active_space_selection import ActiveSpaceSelection
from tn4qa.qi_cost_functions import cost_mutual_info_active_inactive
import os
import numpy as np

cwd = os.getcwd()

from tn4qa.utils import ReadMoleculeData

import pyscf

def stitch_matrices(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    """
    Stitch one or two N x N matrices to produce an N x 2N matrix with alternating columns.
    
    If only A is provided, it is duplicated and interleaved with itself.
    If both A and B are provided, they must be N x N and of the same shape.
    
    Args:
        A (np.ndarray): First N x N matrix.
        B (np.ndarray, optional): Second N x N matrix. Defaults to None.
        
    Returns:
        np.ndarray: N x 2N matrix with alternating columns from A and B (or A and A).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be an N x N square matrix.")

    if B is None:
        B = A.copy()
    elif B.shape != A.shape:
        raise ValueError("Both matrices must be N x N and the same shape.")

    N = A.shape[0]
    C = np.empty((N, 2 * N), dtype=A.dtype)
    C[:, ::2] = A
    C[:, 1::2] = B
    return C

mol_obj = pyscf.M(
    atom='H 0 0 0; H 0 0 1',
    basis='sto-3g',
)

rhf_obj = pyscf.scf.RHF(mol_obj).run()
uhf_obj = pyscf.scf.UHF(mol_obj).run()

rhf_C = rhf_obj.mo_coeff    # shape is (N,N) - spatial orbitals
uhf_C = uhf_obj.mo_coeff    # shape is (2,N,N) - alpha and beta

# For restricted HF, alpha and beta coefficients are the same.
# ActiveSpaceSelection expects a square spatial MO coefficient matrix (N x N) for restricted cases.
coeff_matrix = rhf_C  # shape (N, N)

location = os.path.join(cwd, "molecules/H2.json")
mol_data = ReadMoleculeData(location)
hamiltonian = mol_data.fermionic_hamiltonian

ass = ActiveSpaceSelection(hamiltonian, coeff_matrix)

opt_coeff = ass.run(1, cost_mutual_info_active_inactive, dmrg_max_mps_bond=8, cost_function_max_bond=8, rotation_mpo_max_bond=8)
