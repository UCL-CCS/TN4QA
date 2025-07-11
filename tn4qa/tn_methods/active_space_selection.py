from fidelity_metrics import hilbert_schmidt_distance
from mpo import build_trotterised_unitary, MatrixProductOperator
from mps import MatrixProductState 
import numpy as np
from scipy.optimize import minimize


def hs_squared_distance(V, W) -> float:
    hs = hilbert_schmidt_distance(V, W)
    # Return the squared distance
    return hs**2

def vector_to_antihermitian(theta: np.ndarray, N: int) -> np.ndarray:
    """
    Converts a real vector of length N^2 into an anti-Hermitian matrix K ∈ C^{N x N}.
    
    Diagonal entries are pure imaginary: iθ
    Off-diagonal: K[p,q] = a + ib, K[q,p] = -a + ib
    """
    assert len(theta) == N**2, "theta must have length N^2"
    
    K = np.zeros((N, N), dtype=complex)
    idx = 0

    # Fill diagonals: all imaginary
    for i in range(N):
        K[i, i] = 1j * theta[idx]
        idx += 1

    # Fill upper triangle, set lower triangle with Hermitian conjugate
    for i in range(N):
        for j in range(i + 1, N):
            real = theta[idx]
            imag = theta[idx + 1]
            K[i, j] = real + 1j * imag
            K[j, i] = -real + 1j * imag  # = -conj(K[i,j])
            idx += 2

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

#----------------------------------------------------------------------------------------------------------
def householder_map(psi_C, psi_D):
    """
    Construct an MPO representing the Householder-like unitary V that swaps
    MPS |psi_C⟩ and |psi_D⟩, and acts as identity on the orthogonal complement.

    V = |D><C| + |C><D| + (I - |C><C| - |D><D|)

    Args:
        psi_C: MatrixProductState representing |psi_C⟩
        psi_D: MatrixProductState representing |psi_D⟩

    Returns:
        MatrixProductOperator representing the unitary V
    """
    assert psi_C.num_sites == psi_D.num_sites, "psi_C and psi_D must have the same number of sites"
    N = psi_C.num_sites

    # Compute outer product MPOs
    proj_DC = psi_D.outer_product(psi_C)  # calculate |D><C|
    proj_CD = psi_C.outer_product(psi_D)  # calculate |C><D|
    proj_CC = psi_C.outer_product(psi_C)  # calculate |C><C|
    proj_DD = psi_D.outer_product(psi_D)  # calculate |D><D|

    # Identity MPO
    identity = MatrixProductOperator.identity_mpo(N)

    # Build V = |D><C| + |C><D| + I - |C><C| - |D><D|
    V = proj_DC + proj_CD + identity - proj_CC - proj_DD

    return V



def exponentiate_K(K: np.ndarray) -> np.ndarray:
    """
    Compute U = exp(K) using eigendecomposition, where K is anti-Hermitian.

    Args:
    K: Anti-Hermitian matrix of shape (N, N)

    Returns:
    U = exp(K): a unitary matrix
    """
    assert K.shape[0] == K.shape[1], "K must be square"
    assert np.allclose(K + K.conj().T, 0), "K must be anti-Hermitian"

    # Eigendecomposition: K = V D V^{-1}
    eigvals, eigvecs = np.linalg.eig(K)

    # Compute exp(K) = V exp(D) V^{-1)
    exp_D = np.diag(np.exp(eigvals))
    V_inv = np.linalg.inv(eigvecs)
    U = eigvecs @ exp_D @ V_inv
    return U
