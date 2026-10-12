"""
Spin-aware symmetry verification for TN4QA's QSCI.

WHAT THIS FIXES
---------------
The Jordan-Wigner molecular Hamiltonian conserves the alpha and beta electron
counts separately, not just the total, and therefore S_z.  Verifying only the
total lets the sampled subspace span several S_z blocks, and the consequence is
not inefficiency but a wrong answer:

  H8 chain / STO-3G, one short step from Hartree-Fock, 1e5 shots
    total number only : subspace 168, only  32 determinants with non-zero
                        amplitude (19% live); the returned state has 100% of
                        its weight in the S_z = +2 block and reports
                        E = -3.0935 against an S_z = 0 FCI ground state
    number + S_z      : subspace 110, all 110 live (100%), S_z = 0 throughout

So the diagonalisation was silently returning a high-spin state and its energy
was being compared with the singlet ground state.  In the iterative loop this
also starves the reference: Hartree-Fock has S_z = 0, gets no amplitude, and
the "important configurations" pool fills with S_z = +-2 determinants.

S^2 is NOT checked and cannot be at this stage: it is not diagonal in the
computational basis, so it is not a property of a single sample.  Filtering on
(N_alpha, N_beta) captures the whole abelian part of the spin symmetry, which
is the most per-shot verification can do.  A spin-adapted projection would
have to act on the gathered subspace.

HOW TO APPLY
------------
1. Paste the three module-level helpers into qsci.py above `class QSCI`.
2. Replace `QSCI.symmetry_verification` with the method below.
3. See PATCHES.md for the four small wiring changes (constructor arguments,
   the call in QSCI.run, metadata, and forwarding from the TE/CTE/iterative
   classes).  The ordering matters: QSCI is constructed POSITIONALLY by its
   callers, so new arguments must go at the END of its signature.
"""


def spin_orbital_partition(n_qubits: int, order: str) -> tuple[set[int], set[int]]:
    """Alpha and beta bit positions, with bit 0 the LEFTMOST character.

    'interleaved' is OpenFermion's convention: spin orbital 2p is alpha and
    2p+1 is beta for spatial orbital p.  'blocked' puts all alpha orbitals
    first.  Getting this wrong discards essentially every sample, so it is
    detected rather than assumed -- see detect_spin_orbital_order.
    """
    if order == "interleaved":
        alpha = set(range(0, n_qubits, 2))
    elif order == "blocked":
        alpha = set(range(0, n_qubits // 2))
    else:
        raise ValueError(f"unknown spin orbital order {order!r}")
    return alpha, set(range(n_qubits)) - alpha


def detect_spin_orbital_order(hamiltonian: dict[str, complex]) -> str | None:
    """Infer the spin orbital ordering from the Hamiltonian itself.

    A Pauli string acts on a computational basis state by flipping the bits
    carrying X or Y.  For the term to preserve the alpha and beta counts
    separately it must flip an even number of alpha positions and an even
    number of beta positions.  Requiring that of every term in the
    Hamiltonian singles out the correct convention: on the H4 and H8 STO-3G
    benchmarks the interleaved partition satisfies it and the blocked
    partition does not.

    Returns 'interleaved', 'blocked', or None when neither holds -- in which
    case the Hamiltonian does not conserve S_z in any basis ordering this
    recognises, and only the total electron number should be verified.
    """
    if not hamiltonian:
        return None
    n = len(next(iter(hamiltonian)))
    candidates = []
    for order in ("interleaved", "blocked"):
        alpha, beta = spin_orbital_partition(n, order)
        good = True
        for string in hamiltonian:
            flips = [i for i, c in enumerate(string) if c in "XY"]
            if (
                sum(1 for i in flips if i in alpha) % 2
                or sum(1 for i in flips if i in beta) % 2
            ):
                good = False
                break
        if good:
            candidates.append(order)
    # Both conventions agree only on a Hamiltonian with no spin-flipping
    # structure at all, where the choice cannot matter.
    return candidates[0] if candidates else None


def spin_occupations(
    bitstring: str, alpha: set[int], beta: set[int]
) -> tuple[int, int]:
    """(N_alpha, N_beta) for one bitstring, bit 0 leftmost."""
    na = sum(1 for i in alpha if bitstring[i] == "1")
    nb = sum(1 for i in beta if bitstring[i] == "1")
    return na, nb
