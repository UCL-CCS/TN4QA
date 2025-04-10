import numpy as np
from numpy import ndarray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState


def get_one_particle_rdm(mps: MatrixProductState) -> ndarray:
    """
    Calculate 1-RDM.

    Args:
        mps: The quantum state whose 1-RDM we want

    Returns:
        An array of shape (N,N)
    """
    rdm = np.zeros((mps.num_sites, mps.num_sites))

    # We need to calculate pairs (i,j) for i <= j
    for j in range(mps.num_sites):
        for i in range(j + 1):
            op = "I" * mps.num_sites
            op = op[:i] + "+" + op[i + 1 : j] + "-" + op[j + 1 :]
            mpo = MatrixProductOperator.from_fermionic_operator(op)
            expval = mps.compute_expectation_value(mpo)
            rdm[i, j] = expval
            if i != j:
                rdm[j, i] = np.conj(expval)

    return rdm


def get_two_particle_rdm(mps: MatrixProductState) -> ndarray:
    """
    Calculate 2-RDM.

    Args:
        mps: The quantum state whose 2-RDM we want

    Returns:
        An array of shape (N,N,N,N)
    """
    rdm = np.zeros((mps.num_sites, mps.num_sites, mps.num_sites, mps.num_sites))

    # For pairs (pq),(rs) the 2-RDM is antisymmetric within each pair and Hermitian between the pairs.
    pairs = [(p, q) for q in range(mps.num_sites) for p in range(q + 1)]

    for pair1_idx in range(len(pairs)):
        for pair2_idx in range(pair1_idx):
            p, q = pairs[pair1_idx]
            r, s = pairs[pair2_idx]
            op = ""
            for idx in range(mps.num_sites):
                if idx == p or idx == q:
                    op += "+"
                elif idx == r or idx == s:
                    op += "-"
                else:
                    op += "I"
            mpo = MatrixProductOperator.from_fermionic_operator(op)
            expval = mps.compute_expectation_value(mpo)
            rdm[p, q, r, s] = expval
            rdm[q, p, r, s] = -expval
            rdm[p, q, s, r] = -expval
            rdm[r, s, p, q] = np.conj(expval)
            rdm[s, r, p, q] = -np.conj(expval)
            rdm[r, s, q, p] = -np.conj(expval)

    return rdm


def get_one_orbital_rdm(
    mps: MatrixProductState, site: int, direct: bool = True
) -> ndarray:
    """
    Calculate the one orbital RDM.

    Args:
        mps: The quantum state.
        site: The location of the orbital
        direct: If True the RDM is calculated via direct contraction, otherwise calculated via matrix elements.

    Return:
        The (4,4) array for the one-orbital RDM
    """


def get_two_orbital_rdm(mps: MatrixProductState, site: int) -> ndarray:
    """
    Calculate the two orbital RDM.
    """


def get_one_orbital_entropy(mps: MatrixProductState, site: int) -> float:
    """
    Calculate the one orbital entropy.
    """


def get_two_orbital_entropy(mps: MatrixProductState, sites: list[int]) -> float:
    """
    Calculate the two orbital entropy.
    """


def get_mutual_information(mps: MatrixProductState, sites: list[int]) -> float:
    """
    Calculate the mutual information between two orbitals.
    """


def get_all_mutual_information(mps: MatrixProductState) -> float:
    """
    Calculate the mutual information between every pair of orbitals.
    """
