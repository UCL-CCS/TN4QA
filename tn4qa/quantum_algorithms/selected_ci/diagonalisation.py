"""
Subspace Hamiltonian Projection & Exact Diagonalisation
========================================================
Given:
  - A qubit Hamiltonian as {pauli_string: weight}  e.g. {"XZI": 0.5, "ZZI": -0.3}
  - A list of computational basis bitstrings        e.g. ["1100", "0011", "1001"]

Computes:
  - The projected Hamiltonian matrix H[i,j] = <i|H|j>
  - Its lowest eigenvalue and corresponding eigenvector

Method: symplectic (binary) representation of Pauli strings.
  Each n-qubit Pauli P = ⊗ Pₖ is stored as two binary vectors (z, x) ∈ {0,1}ⁿ
  where:
      I → z=0, x=0
      Z → z=1, x=0
      X → z=0, x=1
      Y → z=1, x=1   (Y = iXZ, phase tracked separately)

For a computational basis state |b⟩, P|b⟩ is always ±1 or ±i times another
basis state |b'⟩.  The matrix element <bᵢ|P|bⱼ> is therefore either 0 (if
P|bⱼ⟩ ≠ |bᵢ⟩ up to phase) or the phase itself.  Both are computed with
pure integer/bitwise arithmetic — no 2^n vectors are ever allocated.
"""

from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Symplectic Pauli engine
# ---------------------------------------------------------------------------


def _parse_paulis(
    hamiltonian: dict[str, complex],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert {pauli_string: weight} to three arrays (z, x, weights).

    z, x : uint8 arrays of shape (n_terms, n_qubits)
    weights: complex128 array of shape (n_terms,)

    Qubit ordering: character index 0 in the Pauli string = qubit 0.
    Make sure this matches your bitstring convention.
    """
    terms = [(ps.upper(), w) for ps, w in hamiltonian.items() if w != 0]
    if not terms:
        raise ValueError("Hamiltonian is empty.")

    n_qubits = len(terms[0][0])
    n_terms = len(terms)

    z = np.zeros((n_terms, n_qubits), dtype=np.uint8)
    x = np.zeros((n_terms, n_qubits), dtype=np.uint8)
    weights = np.zeros(n_terms, dtype=np.complex128)

    for k, (ps, w) in enumerate(terms):
        if len(ps) != n_qubits:
            raise ValueError(
                f"Pauli string '{ps}' has wrong length (expected {n_qubits})."
            )
        for q, c in enumerate(ps):
            if c == "I":
                pass
            elif c == "Z":
                z[k, q] = 1
            elif c == "X":
                x[k, q] = 1
            elif c == "Y":
                z[k, q] = 1
                x[k, q] = 1
            else:
                raise ValueError(f"Unknown Pauli character '{c}'.")
        weights[k] = complex(w)

    return z, x, weights


def _bitstrings_to_ints(samples: list[str]) -> np.ndarray:
    """Convert list of bitstrings to integer indices. '1100' → 12."""
    return np.array([int(b, 2) for b in samples], dtype=np.int64)


def _pauli_action(
    z_row: np.ndarray,  # (n_qubits,) uint8
    x_row: np.ndarray,  # (n_qubits,) uint8
    bj_int: int,  # integer index of |bⱼ⟩
    n_qubits: int,
) -> tuple[int, complex]:
    """
    Apply a single Pauli P (given by its z,x row) to computational basis state |bⱼ⟩.

    Returns (new_basis_int, phase) such that P|bⱼ⟩ = phase * |new_basis⟩.

    Phase from Y = iXZ:
      Each Y on qubit q contributes a factor of i if bⱼ[q]=0 (X flips 0→1)
      and -i if bⱼ[q]=1 (X flips 1→0), but the standard convention is:
          Y|0⟩ =  i|1⟩
          Y|1⟩ = -i|0⟩
      Equivalently, the total phase is i^(n_Y) * (-1)^(popcount of z·b),
      where the (-1) comes from the Z part acting before the X part (ZX = iY).
    """
    # Bits of |bⱼ⟩ as an array
    b = np.array([(bj_int >> q) & 1 for q in range(n_qubits)], dtype=np.uint8)

    # X flips the bits where x=1
    b_new = b ^ x_row

    # Phase from Z: (-1)^(z · b)   [Z acts on original b]
    z_phase_exp = int(np.dot(z_row, b)) % 2  # 0 or 1
    phase = (-1) ** z_phase_exp  # ±1 (real so far)

    # Phase from Y: each Y contributes an extra factor of i
    # Y = iXZ, so n_Y factors of i total, times the sign from Z already counted
    # Net extra factor per Y qubit:  i  (absorbed into the convention above)
    n_Y = int(np.sum(z_row & x_row))
    if n_Y > 0:
        phase = phase * (1j**n_Y)

    # Reconstruct new basis integer
    b_new_int = int(sum(int(b_new[q]) << q for q in range(n_qubits)))

    return b_new_int, phase


# ---------------------------------------------------------------------------
# Core: build projected Hamiltonian
# ---------------------------------------------------------------------------


def project_hamiltonian(
    hamiltonian: dict[str, complex],
    samples: list[str],
) -> ndarray:
    """
    Build the n×n projected Hamiltonian matrix H_proj[i,j] = <bᵢ|H|bⱼ>.

    Parameters
    ----------
    hamiltonian : {pauli_string: coefficient}
                  e.g. {"IZ": -0.5, "ZI": -0.5, "XX": 0.2}
    samples     : list of computational basis bitstrings, all same length.
                  Qubit 0 = leftmost character.

    Returns
    -------
    ham_proj : complex ndarray of shape (len(samples), len(samples))
    """
    n_samples = len(samples)
    n_qubits = len(samples[0])

    z, x, weights = _parse_paulis(hamiltonian)
    basis_ints = _bitstrings_to_ints(samples)

    # Build a lookup: basis_int → row index (for O(1) lookup of <bᵢ|)
    basis_index = {b: idx for idx, b in enumerate(basis_ints.tolist())}

    ham_proj = np.zeros((n_samples, n_samples), dtype=np.complex128)

    # For each Pauli term P_k with weight w_k:
    #   H_proj[i,j] += w_k * <bᵢ|P_k|bⱼ>
    #                = w_k * phase   if P_k|bⱼ⟩ = phase * |bᵢ⟩
    #                = 0             otherwise
    #
    # We iterate over j (ket) and k (Pauli term); the bra index i is determined
    # by which basis state P_k maps |bⱼ⟩ to — a single dict lookup.
    # Total work: O(n_samples * n_terms) with no 2^n allocations.

    for j, bj in enumerate(basis_ints.tolist()):
        for k in range(len(weights)):
            new_basis, phase = _pauli_action(z[k], x[k], bj, n_qubits)
            i = basis_index.get(new_basis)
            if i is not None:
                ham_proj[i, j] += weights[k] * phase

    return ham_proj


# ---------------------------------------------------------------------------
# Exact diagonalisation
# ---------------------------------------------------------------------------


def exact_diagonalisation(
    ham_proj: ndarray,
    sparse_threshold: int = 200,
) -> tuple[float, ndarray]:
    """
    Diagonalise the projected Hamiltonian and return the ground state.

    Parameters
    ----------
    ham_proj          : Hermitian complex ndarray (n×n)
    sparse_threshold  : Use sparse eigensolver (eigsh) for n ≥ this value.

    Returns
    -------
    energy   : float, ground state energy
    gs_coeff : ndarray of shape (n,), ground state coefficients in the
               subspace spanned by `samples`
    """
    # Symmetrise numerical noise
    ham_proj = (ham_proj + ham_proj.conj().T) * 0.5

    n = ham_proj.shape[0]
    if n >= sparse_threshold:
        evals, evecs = eigsh(ham_proj, k=1, which="SA", tol=1e-12)
    else:
        evals, evecs = eigh(ham_proj)

    return float(evals[0].real), evecs[:, 0]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def subspace_energy(
    hamiltonian: dict[str, complex],
    samples: list[str],
    sparse_threshold: int = 200,
    return_matrix: bool = False,
) -> tuple[float, ndarray] | tuple[float, ndarray, ndarray]:
    """
    Project `hamiltonian` into the subspace spanned by `samples` and
    return the ground state energy and coefficient vector.

    Parameters
    ----------
    hamiltonian      : {pauli_string: coefficient}
    samples          : list of bitstrings (qubit 0 = leftmost character)
    sparse_threshold : forwarded to exact_diagonalisation
    return_matrix    : if True, also return the projected matrix

    Returns
    -------
    energy   : float
    gs_coeff : ndarray of shape (len(samples),)
    ham_proj : ndarray of shape (n,n)  — only if return_matrix=True
    """
    ham_proj = project_hamiltonian(hamiltonian, samples)
    energy, gs_coeff = exact_diagonalisation(ham_proj, sparse_threshold)

    if return_matrix:
        return energy, gs_coeff, ham_proj
    return energy, gs_coeff


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def print_groundstate_summary(
    samples: list[str],
    gs_coeff: ndarray,
    energy: float,
    k: int = 10,
) -> None:
    """Print the dominant configurations in the ground state."""
    weights = np.abs(gs_coeff) ** 2
    order = np.argsort(weights)[::-1]

    print(f"\nGround state energy: {energy:.10f}")
    print(f"\n{'Bitstring':<{len(samples[0])+2}}  {'Amplitude':>14}  {'Weight':>10}")
    print("-" * (len(samples[0]) + 32))
    for idx in order[:k]:
        if weights[idx] < 1e-12:
            break
        print(f"|{samples[idx]}>  {gs_coeff[idx]:+.8f}  {weights[idx]:.8f}")
    print(f"\nTotal weight: {weights.sum():.10f}")


# ---------------------------------------------------------------------------
# Quick verification against numpy full diagonalisation
# ---------------------------------------------------------------------------


def verify_against_full_diagonalisation(
    hamiltonian: dict[str, complex],
    n_qubits: int,
    samples: Optional[list[str]] = None,
) -> None:
    """
    Build the full 2^n Hamiltonian directly and compare against the
    subspace result.  Only feasible for small n (≤ 14 or so).
    """
    dim = 2**n_qubits
    full_ham = np.zeros((dim, dim), dtype=np.complex128)

    _ = [format(i, f"0{n_qubits}b") for i in range(dim)]
    z, x, weights = _parse_paulis(hamiltonian)
    _ = {i: i for i in range(dim)}

    for j in range(dim):
        for k in range(len(weights)):
            new_basis, phase = _pauli_action(z[k], x[k], j, n_qubits)
            full_ham[new_basis, j] += weights[k] * phase

    full_ham = (full_ham + full_ham.conj().T) * 0.5
    full_evals = np.linalg.eigvalsh(full_ham)
    print(f"Full ({dim}×{dim}) ground state energy: {full_evals[0]:.10f}")

    if samples is not None:
        e_sub, _ = exact_diagonalisation(project_hamiltonian(hamiltonian, samples))
        print(f"Subspace ({len(samples)}×{len(samples)}) energy:     {e_sub:.10f}")
        print(f"Subspace error: {abs(e_sub - full_evals[0]):.2e} Ha")
