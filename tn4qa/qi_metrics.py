import numpy as np
from numpy import ndarray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState

# NOTATION:
# The following is a matrix of one orbital operators.
# Each matrix element is a linear combination of strings of creation and annihilation operators given as a list of tuples.
# Each tuple is (list, weight) where the first term in list is the operators acting on the up spin-orbital and the second term is the operators acting on the down spin-orbital
ONE_ORBITAL_OPERATORS = np.array(
    [
        [
            [(["", ""], 1), (["+-", ""], -1), (["", "+-"], -1), (["+-", "+-"], 1)],
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
    ],
    dtype=object,
).reshape((4, 4))


def two_orbital_basis_reorder(idx: int, particle_to_orbital: bool = True) -> int:
    """
    The basis for a two-orbital RDM can be ordered particle-first or orbital-first.
    Particle-first makes the two-orbital RDM block diagonal however we tend to use orbital-first.

    Args:
        idx: The index of the basis element.
        particle_to_orbital: If true, finds the index of the basis element in the orbital basis.

    Returns:
        int: The index of the basis element in the reordered basis.
    """
    # key = index in particle-first ordering, value = index in orbital-first ordering
    reordering_dict = {
        "0": "0",
        "1": "2",
        "2": "8",
        "3": "1",
        "4": "4",
        "5": "10",
        "6": "3",
        "7": "9",
        "8": "6",
        "9": "12",
        "10": "5",
        "11": "11",
        "12": "14",
        "13": "7",
        "14": "13",
        "15": "15",
    }
    # If reordering in the other direction, flip the dictionary
    if not particle_to_orbital:
        reordering_dict = {v: k for k, v in reordering_dict.items()}
    return int(reordering_dict[str(idx)])


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
    up_idx = 2 * orbital_idx - 2
    down_idx = 2 * orbital_idx - 1
    ops = []
    for op_list, weight in ONE_ORBITAL_OPERATORS[idx1, idx2]:
        op = []
        up_op = op_list[0]
        down_op = op_list[1]
        for o in up_op:
            op.append((str(up_idx), o))
        for o in down_op:
            op.append((str(down_idx), o))
        ops.append((op, weight))
    return ops


def get_one_particle_rdm(mps: MatrixProductState) -> ndarray:
    """Calculate 1-RDM.

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
    """Calculate 2-RDM.

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
        The (4,4) array for the one-orbital RDM.

    Example:
        >>> mps = MatrixProductState(...)
        >>> rdm = get_one_orbital_rdm(mps, 1)
        >>> print(rdm)
    """
    if direct:
        spin_orbitals_to_remove = list(range(1, mps.num_sites + 1))
        spin_orbitals_to_remove.remove(2 * orbital_idx - 1)
        spin_orbitals_to_remove.remove(2 * orbital_idx)
        rdm = mps.partial_trace(spin_orbitals_to_remove, matrix=True)
        rdm = rdm.data.todense()
        if enforce_symmetry:
            for j in range(4):
                for i in range(j):
                    rdm[i, j] = 0
                    rdm[j, i] = 0
        return rdm
    else:
        rdm = np.zeros((4, 4), dtype=complex)
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
    """Calculate the two orbital RDM.

    Args:
        mps: The quantum state.
        site: The location of the orbitals
        direct: If True the RDM is calculated via direct contraction, otherwise calculated via matrix elements.
        enforce_symmetry: If True then enforces spin and particle number symmetry in RDM.

    Return:
        The (16,16) array for the two-orbital RDM

    Raises:
        ValueError: If the sites are not two different sites.

    Example:
        >>> mps = MatrixProductState(...)
        >>> rdm = get_two_orbital_rdm(mps, [1, 2])
        >>> print(rdm)
    """
    if len(set(sites)) != 2:
        raise ValueError("Must provide two different sites.")

    expected_non_zero_pairs_particle_basis = [
        (0, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 6),
        (7, 7),
        (7, 8),
        (7, 9),
        (8, 6),
        (8, 7),
        (8, 8),
        (8, 9),
        (9, 6),
        (9, 7),
        (9, 8),
        (9, 9),
        (10, 10),
        (11, 11),
        (11, 12),
        (12, 11),
        (12, 12),
        (13, 13),
        (13, 14),
        (14, 13),
        (14, 14),
        (15, 15),
    ]
    expected_non_zero_pairs_orbital_basis = [
        (two_orbital_basis_reorder(x), two_orbital_basis_reorder(y))
        for x, y in expected_non_zero_pairs_particle_basis
    ]
    if direct:
        spin_orbitals_to_remove = list(range(1, mps.num_sites + 1))
        spin_orbitals_to_remove.remove(2 * sites[0] - 1)
        spin_orbitals_to_remove.remove(2 * sites[0])
        spin_orbitals_to_remove.remove(2 * sites[1] - 1)
        spin_orbitals_to_remove.remove(2 * sites[1])
        rdm = mps.partial_trace(spin_orbitals_to_remove, matrix=True)
        rdm = rdm.data.todense()
        if enforce_symmetry:
            for i in range(16):
                for j in range(16):
                    if (i, j) in expected_non_zero_pairs_orbital_basis:
                        continue
                    else:
                        rdm[i, j] = 0
        return rdm
    else:
        rdm = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            for j in range(16):
                if (
                    i,
                    j,
                ) not in expected_non_zero_pairs_orbital_basis and enforce_symmetry:
                    rdm[i, j] = 0
                else:
                    op1 = get_one_orbital_operator(
                        int(np.floor(i / 4)), int(np.floor(j / 4)), sites[0]
                    )
                    op2 = get_one_orbital_operator(int(i % 4), int(j % 4), sites[1])
                    mpo1 = MatrixProductOperator.from_fermionic_operator(
                        mps.num_sites, op1
                    )
                    mpo2 = MatrixProductOperator.from_fermionic_operator(
                        mps.num_sites, op2
                    )
                    mpo = mpo1 * mpo2
                    rdm[i, j] = mps.compute_expectation_value(mpo)
        return rdm


def get_one_orbital_entropy(mps: MatrixProductState, site: int) -> float:
    """Calculate the one orbital entropy.

    Args:
        mps (MatrixProductState): A matrix product state.
        site (int): The index of the orbital (1-indexed).
    Returns:
        float: The one orbital entropy.

    Example:
        >>> mps = MatrixProductState(...)
        >>> entropy = get_one_orbital_entropy(mps, 1)
        >>> print(entropy)
    """
    rdm1 = get_one_orbital_rdm(mps, site, direct=True, enforce_symmetry=False)
    # Calculate eigenvalues
    eigvals = np.linalg.eigvalsh(rdm1)
    # Calculate entropy
    entropy = -np.sum(
        eigvals * np.log2(eigvals + 1e-12)
    )  # Add small value to avoid log(0)
    return entropy


def get_two_orbital_entropy(mps: MatrixProductState, sites: list[int]) -> float:
    """Calculate the two orbital entropy.

    Args:
        mps (MatrixProductState): A matrix product state.
        sites (list[int]): A list of two orbital indices (1-indexed).
    Returns:
        float: The two orbital entropy.

    Example:
        >>> mps = MatrixProductState(...)
        >>> entropy = get_two_orbital_entropy(mps, [1, 2])
        >>> print(entropy)
    """
    match n_sites := len(set(sites)):
        case 1:
            return get_one_orbital_entropy(mps, sites[0])
        case 2:
            rdm2 = get_two_orbital_rdm(mps, sites, direct=True, enforce_symmetry=False)
            # Calculate eigenvalues
            eigvals = np.linalg.eigvalsh(rdm2)
            # Calculate entropy
            entropy = -np.sum(
                eigvals * np.log2(eigvals + 1e-12)
            )  # Add small value to avoid log(0)
            return entropy
        case _:
            raise ValueError(
                "Incorrect number of sites provided for two orbital entropy %s", n_sites
            )


def get_mutual_information(mps: MatrixProductState, sites: list[int]) -> float:
    """
    Calculate the mutual information between two orbitals.
    I(i, j) = S(i) + S(j) - S(i,j)

    Args:
        mps (MatrixProductState): A matrix product state.
        sites (list[int]): A list of two orbital indices (1-indexed).
    Returns:
        float: The mutual information between the two orbitals.

    Raises:
        ValueError: If the sites list does not contain exactly two different sites.

    Example:
        >>> mps = MatrixProductState(...)
        >>> mutual_info = get_mutual_information(mps, [1, 2])
        >>> print(mutual_info)
    """
    match n_sites := len(set(sites)):
        case 1:
            mutual_info = get_one_orbital_entropy(mps, sites[0])
        case 2:
            s1 = get_one_orbital_entropy(mps, sites[0])
            s2 = get_one_orbital_entropy(mps, sites[1])
            s12 = get_two_orbital_entropy(mps, sites)
            mutual_info = s1 + s2 - s12
        case _:
            raise ValueError(
                "Incorrect number of sites provided for mutual information %s", n_sites
            )
    return mutual_info


def get_all_mutual_information(mps: MatrixProductState) -> np.typing.NDArray:
    """
    Calculate the mutual information between every pair of orbitals.
    Mutual Information matrix where M[i,j] = I(i,j)
    I(i,j) = S(i) + S(j) - S(i,j)

    Args:
        mps (MatrixProductState): A matrix product state.
    Returns:
        ndarray: A symmetric matrix of shape (N, N) where N is the number of orbitals.
                 M[i, j] = I(i, j) is the mutual information between orbitals i and j.

    Example:
        >>> mps = MatrixProductState(...)
        >>> mutual_info_matrix = get_all_mutual_information(mps)
        >>> print(mutual_info_matrix)
    """
    n_orbs = mps.num_sites // 2  # Number of orbitals
    M = np.zeros((n_orbs, n_orbs))
    for i in range(0, n_orbs):
        M[i, i] = get_one_orbital_entropy(mps, i + 1)
        for j in range(i + 1, n_orbs):
            # mps sites are 1-indexed
            M[i, j] = get_mutual_information(mps, [i + 1, j + 1])
            M[j, i] = M[i, j]
    return M
