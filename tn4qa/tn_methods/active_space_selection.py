from fidelity_metrics import hilbert_schmidt_distance
from mpo import build_trotterised_unitary
import numpy as np
from scipy.optimize import minimize


def hs_squared_distance(V, W) -> float:
    hs = hilbert_schmidt_distance(V, W)
    # Return the squared distance
    return hs**2

def vector_to_antihermitian(theta: np.ndarray, N: int) -> np.ndarray:
    """
    Convert a vector of parameters to an anti-Hermitian matrix.
    Map real parameters to a skew-Hermitian K∈C^(NxN) matrix.

    Args:
        theta: A vector of parameters.
        N: The number of sites (or dimensions).

    Returns:
        An anti-Hermitian matrix of shape (N, N).
    """
    K = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(i + 1, N):
            K[i, j] = -1j * theta[i * (N - 1) + j - 1]
            K[j, i] = 1j * theta[i * (N - 1) + j - 1]
    return K

def cost(theta: np.ndarray, V, W_init) -> float:
    N = V.num_sites
    K = vector_to_antihermitian(theta, N)
    W_rotated = build_trotterised_unitary(K)  # returns an MPO
    return hs_squared_distance(V, W_rotated)

def optimise_K(V, W_init):
    """
    Run BFGS optimization over K such that W(K) ≈ V.

    Args:
        V: Target MPO
        W_init: Initial MPO (e.g. identity or guess)

    Returns:
        Optimal real-valued parameter vector θ defining anti-Hermitian K
    """
    N = V.num_sites
    num_params = N * (N - 1)  
    theta0 = np.zeros(num_params)

    result = minimize(
        cost,
        theta0,
        args=(V, W_init),
        method='BFGS',
        options={'disp': True}
    )

    return result.x