from fidelity_metrics import hilbert_schmidt_distance
from mpo import MatrixProductOperator
from mps import MatrixProductState 
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from symmer.operators import PauliwordOp
from quantum_algorithms.hamiltonian_simulation.trotterisation import TrotterSimulation
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner


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

def exponential_hopping_term(p: int, q: int, theta: complex, num_sites: int) -> MatrixProductOperator:
    """
    Construct the MPO for exp(theta * a_p† a_q - theta* * a_q† a_p).

    Args:
        p, q: Indices of orbitals (must be different)
        theta: Complex parameter
        num_sites: Total number of spin-orbitals

    Returns:
        MatrixProductOperator for exp(H), where H = theta * a_p† a_q - conj(theta) * a_q† a_p
    """
    assert p != q, "Cannot build on-site hopping term with p == q"

    # Build the FermionOperator, 1 is the creation operator, 0 is the annihilation operator
    # H = theta * a_p† a_q - conj(theta) * a_q†
    h_fermion = FermionOperator(((p, 1), (q, 0)), theta) - FermionOperator(((q, 1), (p, 0)), np.conj(theta))

    # Map to QubitOperator using Jordan-Wigner
    h_qubit = jordan_wigner(h_fermion)

    # Convert to PauliwordOp 
    h_pauli = PauliwordOp.from_openfermion(h_qubit)

    # Convert to a dictionary
    h_dict = h_pauli.to_dictionary()
    h_dict = {k:v.real for k,v in h_dict.items()}

    # Create a circuit
    sim = TrotterSimulation(H_dict, duration=1.0, num_steps=1)
    qc = sim.from_qiskit_circuit() 

    # Convert Qiskit circuit to MPO
    u_mpo = MatrixProductOperator.from_qiskit_circuit(qc)

    return u_mpo

def build_trotterised_unitary(K: np.ndarray, trotter_steps=1) -> MatrixProductOperator:
    """
    Build an MPO approximation of the fermionic unitary:
        U = exp(Σ_{pq} K_{pq} a†_p a_q)
    
    using first-order Trotter decomposition.

    Args:
        K: Anti-Hermitian matrix (N x N)
        trotter_steps: Number of Trotter steps

    Returns:
        MatrixProductOperator representing the unitary
    """
    N = K.shape[0]
    assert K.shape[1] == N
    assert np.allclose(K + K.conj().T, 0, atol=1e-10), "K must be anti-Hermitian"

    u_mpo = MatrixProductOperator.identity_mpo(N)
    dt = 1.0 / trotter_steps

    for _ in range(trotter_steps):
        for p in range(N):
            for q in range(N):
                if abs(K[p, q]) > 1e-12:
                    theta = dt * K[p, q]
                    hop_exp_mpo = exponential_hopping_term(p, q, theta, N)
                    u_mpo = u_mpo @ hop_exp_mpo

    return u_mpo

def cost(theta: np.ndarray, V) -> float:
    N = V.num_sites
    K = vector_to_antihermitian(theta, N)
    W_rotated = build_trotterised_unitary(K)  # returns an MPO
    return hs_squared_distance(V, W_rotated)

def optimise_K(V, W_init):
    """
    Run BFGS optimisation over K such that W(K) ≈ V.

    Args:
        V: Target MPO
        W_init: Initial MPO (e.g. identity or guess)

    Returns:
        Optimal real-valued parameter vector θ defining anti-Hermitian K
    """
    N = V.num_sites
    num_params = N**2
    theta0 = np.zeros(num_params)

    result = minimize(
        cost,
        theta0,
        args=(V, W_init),
        method='BFGS',
        options={'disp': True}
    )

    return result.x

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

#----------------------------------------------------------------------------------------------------------

def active_space_selection(coeff_matrix: np.ndarray, num_active_orbitals: int) -> np.ndarray:
    """
    Perform active space selection by optimising a unitary transformation of the orbital coefficients.

    Args:
        coeff_matrix: HF coefficient matrix of shape (N, N)
        num_active_orbitals: Number of active orbitals to select

    Returns:
        Transformed coefficient matrix with optimal active orbitals
    """
    N = coeff_matrix.shape[0]
    assert coeff_matrix.shape[1] == N, "Coefficient matrix must be square"

    # Convert the coefficient matrix into an MPO 
    # Using identity MPO as a placeholder for V
    V = MatrixProductOperator.identity_mpo(N)

    # Initialise an identity MPO (initial guess W_init)
    # Using identity MPO as a placeholder for W_init
    W_init = MatrixProductOperator.identity_mpo(N)

    # Run BFGS optimisation to find optimal K
    theta_opt = optimise_K(V, W_init)

    # Exponentiate K to get a unitary U = exp(K)
    K = vector_to_antihermitian(theta_opt, N)
    U = exponentiate_K(K)

    # Apply U to the input coefficient matrix, returning the transformed coefficient matrix (the new basis)
    transformed_coeff_matrix = U @ coeff_matrix

    # Truncate to the desired number of active orbitals
    active_coeff_matrix = transformed_coeff_matrix[:, :num_active_orbitals]

    return active_coeff_matrix




