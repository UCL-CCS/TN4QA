import math

import numpy as np
import pyscf
from symmer import QuantumState


def index_to_bitstring(
    bitstring_len: int, number_particles: int, idx: int
) -> np.ndarray:
    """
    Given bitstring length and number of paricles and Fock space index... get binary Fock sector string
    This expects Fock space to be ordered Lexicographically.

    e.g. for 2 particles in 4 orbs... we expect the 6 unique indices to give:

        0 index ==> [1,1,0,0]
        1 index ==> [1,0,1,0]
        2 index ==> [1,0,0,1]
        3 index ==> [0,1,1,0]
        4 index ==> [0,1,0,1]
        5 index ==> [0,0,1,1]

    Args:
        bitstring_len (int): length of bitstring
        number_particles (int): number of paricles
        idx (int): Fock space index
    Returns:
        bitstring (np.array): bitstring array
    """
    bitstring = np.zeros(bitstring_len, dtype=int)
    n_part = np.int_(number_particles)  # needed for assert

    for n in np.arange(bitstring_len, 0, -1):
        if n > number_particles & number_particles >= 0:
            y = math.comb(n - 1, number_particles)
        else:
            y = 0
        if idx >= y:
            idx = idx - y
            bitstring[n - 1] = 1
            number_particles = number_particles - 1

    assert np.equal(
        np.sum(bitstring), n_part
    ), f"number of particles not correct. Expect: {n_part} got: {np.sum(bitstring)}"
    return bitstring


def civec_to_state(ci_obj, zero_threhold=None) -> QuantumState:
    """
    Function to convert a PySCF FCIvector into a Quantum State

    Note: fci_vec matrix is indexed as:
                        fci_vec[alpha_bitstring_index, beta_bitstring_index] = amplitude

    Note the dimension of this matrix will therefore be:
    (M_spatial_orbs choose n_alpha_electrons) by (M_spatial_orbs choose n_beta_electrons)

    Args:
        ci_obj (PySCF): pyscf CI object (either ci or fci)
    Returns:
        psi (QuantumState): QuantumState version of FCI vector
    """
    norb = ci_obj.mol.nao
    nelec_a, nelec_b = ci_obj.mol.nelec

    try:
        ## CI object
        ci_amps = ci_obj.to_fcivec(
            cisdvec=ci_obj.ci,
            norb=norb,
            nelec=ci_obj.mol.nelec,
            # cisdvec=ci_obj.ci, nmo=norb, nocc=ci_obj.mol.nelec # variables named differently if using ROHF...? PySCF 2.6.2
        )
    except TypeError:
        ## FCI object
        ci_amps = ci_obj.ci

    # get nonzero amplitudes!
    if zero_threhold is None:
        alpha_ket_int_index, beta_ket_int_index = ci_amps.nonzero()
    else:
        assert zero_threhold > 0, "cannot have zero threshold less than 0"
        alpha_ket_int_index, beta_ket_int_index = np.nonzero(
            np.abs(ci_amps) > zero_threhold
        )

    nq = 2 * norb
    state_matrix = np.zeros((len(alpha_ket_int_index), nq), dtype=int)
    coeff_vec = np.zeros(len(alpha_ket_int_index), dtype=complex)
    for i, (a_idx, b_idx) in enumerate(zip(alpha_ket_int_index, beta_ket_int_index)):
        alpha_arr = index_to_bitstring(norb, nelec_a, a_idx)
        beta_arr = index_to_bitstring(norb, nelec_b, b_idx)
        sign = (-1) ** np.sum(
            [np.sum(beta_arr[:ind]) for ind in alpha_arr.nonzero()[0]]
        )
        state_matrix[i, 0::2] = alpha_arr
        state_matrix[i, 1::2] = beta_arr
        coeff_vec[i] = ci_amps[a_idx, b_idx] * sign

    psi = QuantumState(state_matrix, coeff_vec).normalize.sort()
    return psi


def get_string(xyz_path: str, charge: int = 0, basis: str = "sto-3g") -> np.ndarray:
    """Runs FCI on a molecule from an XYZ file and returns the coefficients in full statevector form.

    Args:
        xyz_path (str): Path to the XYZ file.
        charge (int, optional): Charge of the molecule. Defaults to 0.
        basis (str, optional): Basis set to use. Defaults to 'sto-3g'.

    Returns:
        np.ndarray: Statevector version of the FCI vector.
    """
    # Build the molecule directly from the XYZ path
    mol = pyscf.M(atom=xyz_path, charge=charge, basis=basis)

    # Run RHF and FCI calculations
    rhf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    fci_obj = pyscf.fci.FCI(rhf_obj).run(verbose=0)
    print(fci_obj.e_tot)

    # Convert FCI vector to full statevector
    statevector = civec_to_state(ci_obj=fci_obj).to_sparse_matrix.toarray()

    return statevector
