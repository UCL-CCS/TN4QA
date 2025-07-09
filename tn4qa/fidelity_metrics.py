import copy

import numpy as np
import scipy
import scipy.linalg

from .mpo import MatrixProductOperator
from .mps import MatrixProductState


def trace_nrom(
    A: MatrixProductState | MatrixProductOperator, unitary: bool = False
) -> float:
    """
    Calculate the trace norm of a quantum state or operator

    Args:
        A: The MPS or MPO to calculate the trace norm of
        unitary: If True the MPO is assumed to represent a unitary operator

    Returns:
        The Schatten 1-norm of A, Tr(sqrt(A^dag A))
    """

    # Pure states have trace norm = 1
    if isinstance(A, MatrixProductState):
        return 1

    # Unitary operators have trace norm = dimension
    if unitary:
        return (A.physical_dimension) ** A.num_sites

    # Calculate directly
    # N.B. only doable for small operators
    Adag = A.dagger()
    prod = Adag * A
    prod_mat = prod.to_dense_array()
    sqrt_prod_mat = scipy.linalg.sqrtm(prod_mat)
    return np.trace(sqrt_prod_mat)


def hilbert_schmidt_inner_product(
    A: MatrixProductOperator, B: MatrixProductOperator
) -> complex:
    """
    Calculate the HS inner product of two operators.

    Args:
        A: The first operator
        B: The second operator

    Returns:
        Tr(A^dag B)
    """
    A.dagger()
    prod = A * B
    return prod.trace()


def frobenius_norm(
    A: MatrixProductState | MatrixProductOperator, unitary: bool = False
) -> float:
    """
    Calculate the Frobenius norm of a quantum state or operator

    Args:
        A: The MPS or MPO to calculate the trace norm of
        unitary: If True the MPO is assumed to represent a unitary operator

    Returns:
        The Schatten 2-norm of A, sqrt(Tr(A^dag A))
    """
    # The Frobenius norm of a pure state = 1
    if isinstance(A, MatrixProductState):
        return 1

    # The Frobenius norm of a unitary matrix = sqrt(dimension)
    if unitary:
        return np.sqrt((A.physical_dimension) ** A.num_sites)

    # Otherwise call the HS inner product
    ip = hilbert_schmidt_inner_product(A, A)
    return np.sqrt(ip.real)  # N.B. ip will be real anyway since A^dag A is Hermitian


def trace_distance(
    psi: MatrixProductState | MatrixProductOperator,
    phi: MatrixProductState | MatrixProductOperator,
) -> float:
    """
    Calculate the trace distance between two quantum states or operators

    Args:
        psi: The first state or operator
        phi: The second state or operator

    Returns:
        D_tr = 1/2 * ||psi - phi||_1
    """
    # Pure states is easy
    if isinstance(psi, MatrixProductState) and isinstance(phi, MatrixProductState):
        uhlmann_fid = state_uhlmann_fidelity(psi, phi)
        return np.sqrt(1 - uhlmann_fid)

    # Otherwise call the trace norm
    if isinstance(psi, MatrixProductState):
        psi = psi.form_density_operator()
    if isinstance(phi, MatrixProductState):
        phi = phi.form_density_operator()
    diff = psi - phi
    return 0.5 * trace_nrom(diff)


def hilbert_schmidt_distance(
    psi: MatrixProductState | MatrixProductOperator,
    phi: MatrixProductState | MatrixProductOperator,
    unitary: bool = False,
) -> float:
    """
    Calculate the HS distance between two quantum states or operators

    Args:
        psi: The first state or operator
        phi: The second state or operator

    Returns:
        ||psi - phi||_2
    """
    # Pure states is easy
    if isinstance(psi, MatrixProductState) and isinstance(phi, MatrixProductState):
        overlap = psi.compute_inner_product(phi)
        return np.sqrt(2 - 2 * overlap.real)

    # Unitary matrices are easy
    if unitary:
        ip = hilbert_schmidt_inner_product(psi, phi)
        dim = (psi.physical_dimension) ** psi.num_sites
        return np.sqrt(2 * dim - 2 * ip.real)

    # Otherwise call the Frobenius norm
    if isinstance(psi, MatrixProductState):
        psi = psi.form_density_operator()
    if isinstance(phi, MatrixProductState):
        phi = phi.form_density_operator()
    diff = psi - phi
    return frobenius_norm(diff)


def state_uhlmann_fidelity(
    psi: MatrixProductState | MatrixProductOperator,
    phi: MatrixProductState | MatrixProductOperator,
) -> float:
    """
    Calculate the Uhlmann (trace) fidelity between two quantum states.

    Args:
        psi: The first state
        phi: The second state

    Returns:
        F_U = Tr(sqrt(sqrt(psi) phi sqrt(psi)))**2
    """

    mps = copy.deepcopy(psi)
    other = copy.deepcopy(phi)

    # Pure states fidelity is easy
    if isinstance(psi, MatrixProductState):
        assert isinstance(phi, MatrixProductState)
        inner_prod = mps.compute_inner_product(other)
        return np.abs(inner_prod) ** 2

    # Otherwise it's hard, only possible for small systems
    psi_mat = psi.to_dense_array()
    phi_mat = phi.to_dense_array()
    sqrt_psi = scipy.linalg.sqrtm(psi_mat)
    prod = sqrt_psi @ phi_mat @ sqrt_psi
    sqrt_prod = scipy.linalg.sqrtm(prod)
    fid = np.trace(sqrt_prod) ** 2
    return fid


def hilbert_schmidt_fidelity(
    psi: MatrixProductState | MatrixProductOperator,
    phi: MatrixProductState | MatrixProductOperator,
    unitary: bool = False,
) -> float:
    """
    Calculate the HS fidelity between two quantum states or operators

    Args:
        psi: The first state or operator
        phi: The second state or operator

    Returns:
        F_HS = Tr(psi phi) / Tr(psi psi)
    """
    # Pure states is easy
    if isinstance(psi, MatrixProductState) and isinstance(phi, MatrixProductState):
        return state_uhlmann_fidelity(psi, phi)

    # Unitaries are easy
    if unitary:
        ip = hilbert_schmidt_inner_product(psi, phi)
        dim = (psi.physical_dimension) ** psi.num_sites
        return (1 / dim**2) * np.abs(ip) ** 2

    # Otherwise call the Frobenius norm and HS inner product
    if isinstance(psi, MatrixProductState):
        psi = psi.form_density_operator()
    if isinstance(phi, MatrixProductState):
        phi = phi.form_density_operator()
    ip = hilbert_schmidt_inner_product(psi, phi)
    return np.abs(ip) ** 2 / ((frobenius_norm(psi) ** 2) * (frobenius_norm(phi) ** 2))


def total_variation_distance(
    output_distribution: dict[str, float] | MatrixProductState,
    expected_distribution: dict[str, float],
    sample_size: int = None,
) -> float:
    """
    Calculate the total variation distance between the probability distribution of an mps with an expected distribution

    Args:
        mps: The MPS
        expected_distribution: The expected distribution
        sample_size: The number of samples to take from the MPS to form the probability distribution. If none, gets exact distribution

    Returns:
        TVD = 1/2 * sum_x |P(x) - Q(x)|
    """
    if isinstance(output_distribution, MatrixProductState):
        mps = output_distribution
        if not sample_size:
            probability_dist = mps.get_probability_distribution()
        else:
            probability_dist = mps.get_approximate_probability_distribution(sample_size)
    else:
        probability_dist = output_distribution

    # Get the union of keys from both distributions
    all_keys = set(probability_dist.keys()).union(expected_distribution.keys())

    # Sum over absolute differences of probabilities for each key (default 0 if key missing)
    total_diff = sum(
        abs(probability_dist.get(k, 0) - expected_distribution.get(k, 0))
        for k in all_keys
    )

    # TVD is half the total difference
    return 0.5 * total_diff
