import numpy as np
from numpy import ndarray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState

# NOTATION:
# Each tuple is (list, weight) where the first term in list is the operators acting on the up spin-orbital and the second term is the operators acting on the down spin-orbital
ONE_ORBITAL_OPERATORS = np.array(
    [
        [
            [([], 1), (["+-", ""], -1), (["", "+-"], -1), (["+-", "+-"], 1)],
            [(["", "-"], 1), (["+-", "-"], -1)],
            [(["-", ""], 1), (["-", "+-"], -1)],
            [(["-", "-"], 1)],
        ],
        [
            [(["", "+"], 1), (["+-", "+"], -1)],
            [(["", "+-"], 1), (["+-", "+-"], -1)],
            [(["-", "+"], 1)],
            [(["-", "+-"], -1)],
        ],
        [
            [(["+", ""], 1), (["+", "+-"], -1)],
            [(["+", "-"], 1)],
            [(["+-", ""], 1), (["+-", "+-"], -1)],
            [(["+-", "-"], 1)],
        ],
        [
            [(["+", "+"], 1)],
            [(["+", "+-"], 1)],
            [(["+-", "+"], 1)],
            [(["+-", "+-"], 1)],
        ],
    ]
)


def get_one_orbital_operator(idx1: int, idx2: int, orbital_idx: int) -> list[tuple]:
    """
    Get a one orbital Fermionic operator.

    Args:
        idx1: The index of the iniital occupation state.
        idx2: The index of the final occupation state.
        orbital_idx: The index of the orbital.

    Returns:
        A list of tuples of the form (op, weight) where op is a single string of Fermionic creation/annihilation operators.
    """
    up_idx = 2 * orbital_idx - 1
    down_idx = 2 * orbital_idx
    ops = []
    for op_list, weight in ONE_ORBITAL_OPERATORS[idx1, idx2]:
        op = []
        for up_op, down_op in op_list:
            for o in up_op:
                op.append((str(up_idx), o))
            for o in down_op:
                op.append((str(down_idx), o))
        ops.append((op, weight))
    return ops


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
            op_list = [(f"{i}", "+"), (f"{j}", "-")]
            mpo = MatrixProductOperator.from_fermionic_string(op_list)
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
            op_list = [
                (f"{p}", "+"),
                (f"{q}", "+"),
                (f"{r}", "-"),
                (f"{s}", "-"),
            ]
            mpo = MatrixProductOperator.from_fermionic_operator(op_list)
            expval = mps.compute_expectation_value(mpo)
            rdm[p, q, r, s] = expval
            rdm[q, p, r, s] = -expval
            rdm[p, q, s, r] = -expval
            rdm[r, s, p, q] = np.conj(expval)
            rdm[s, r, p, q] = -np.conj(expval)
            rdm[r, s, q, p] = -np.conj(expval)

    return rdm


def get_one_orbital_rdm(
    mps: MatrixProductState,
    orbital_idx: int,
    direct: bool = True,
    enforce_symmetry: bool = False,
) -> ndarray:
    """
    Calculate the one orbital RDM.

    Args:
        mps: The quantum state.
        site: The location of the orbital
        direct: If True the RDM is calculated via direct contraction, otherwise calculated via matrix elements.
        enforce_symmetry: If True then enforces spin and particle number symmetry in RDM.

    Return:
        The (4,4) array for the one-orbital RDM
    """
    if direct:
        spin_orbitals_to_remove = list(range(1, mps.num_sites + 1))
        spin_orbitals_to_remove.remove(2 * orbital_idx - 1)
        spin_orbitals_to_remove.remove(2 * orbital_idx)
        rdm = mps.partial_trace(spin_orbitals_to_remove, matrix=True)
        if enforce_symmetry:
            for j in range(4):
                for i in range(j):
                    rdm[i, j] = 0
                    rdm[j, i] = 0
        return rdm
    else:
        rdm = np.zeros((4, 4))
        for i in range(4):
            op = get_one_orbital_operator(i, i, orbital_idx)
            mpo = MatrixProductOperator.from_fermionic_operator(mps.num_sites, op)
            rdm[i, i] = mps.compute_expectation_value(mpo)
        if enforce_symmetry:
            return rdm
        else:
            for i in range(4):
                for j in range(4):
                    if i == j:
                        pass
                    else:
                        op = get_one_orbital_operator(i, j, orbital_idx)
                        mpo = MatrixProductOperator.from_fermionic_operator(
                            mps.num_sites, op
                        )
                        rdm[i, j] = mps.compute_expectation_value(mpo)
            return rdm


def get_two_orbital_rdm(
    mps: MatrixProductState,
    sites: list[int],
    direct: bool = True,
    enforce_symmetry: bool = False,
) -> ndarray:
    """
    Calculate the two orbital RDM.

    Args:
        mps: The quantum state.
        site: The location of the orbitals
        direct: If True the RDM is calculated via direct contraction, otherwise calculated via matrix elements.
        enforce_symmetry: If True then enforces spin and particle number symmetry in RDM.

    Return:
        The (16,16) array for the two-orbital RDM
    """
    if direct:
        spin_orbitals_to_remove = list(range(1, mps.num_sites + 1))
        spin_orbitals_to_remove.remove(2 * sites[0] - 1)
        spin_orbitals_to_remove.remove(2 * sites[0])
        spin_orbitals_to_remove.remove(2 * sites[1] - 1)
        spin_orbitals_to_remove.remove(2 * sites[1])
        rdm = mps.partial_trace(spin_orbitals_to_remove, matrix=True)
        if enforce_symmetry:
            # TODO: enable enforcing symmetries
            pass
        return rdm
    else:
        # TODO: implement calculation via matrix elements
        return


def get_one_orbital_entropy(mps: MatrixProductState, site: int) -> float:
    """
    Calculate the one orbital entropy.
    """
    # TODO: Method - just calculate the one orbital RDM using above function then calculate eigenvalues and sub into entropy formula.
    # N.B. if symmetries are enforced in the RDM calculation then the RDM is diagonal so the eigenvalues are easy to get.


def get_two_orbital_entropy(mps: MatrixProductState, sites: list[int]) -> float:
    """
    Calculate the two orbital entropy.
    """
    # TODO: Method - just calculate the two orbital RDM using above function then calculate eigenvalues and sub into entropy formula.


def get_mutual_information(mps: MatrixProductState, sites: list[int]) -> float:
    """
    Calculate the mutual information between two orbitals.
    """
    # TODO: Method - calculate one and two orbital entropies using above functions and sub into mutual information formula


def get_all_mutual_information(mps: MatrixProductState) -> float:
    """
    Calculate the mutual information between every pair of orbitals.
    """
    # TODO: Method - just iterate the previous function over all pairs of orbitals.
