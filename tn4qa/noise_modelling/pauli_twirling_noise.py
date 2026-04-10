"""
Pauli-Twirled Noise Channel — Superoperator / Evolution-Matrix Form
==============================================================================

After Pauli twirling, any two-qubit gate error channel is converted into a
Pauli channel:

    N(rho) = Σ_{P ∈ P^⊗2}  p_P  P rho P†

where {p_P} are the Pauli-channel probabilities inferred from the gate
error rate ε (depolarising assumption).

Representation
--------------
We work in the *Pauli Transfer Matrix* (PTM) / Liouville superoperator
representation.  A density matrix rho (4^n vector in Pauli basis) evolves as

    vec(rho')  =  L_noise · L_gate · vec(rho)

so the full noisy gate superoperator is  L' = L_noise · L_gate.

For an n-qubit Pauli channel the PTM is diagonal:

    [L_noise]_{PP} = 1 - Σ_{Q≠P} 2 p_Q

The inverse (noise-inversion channel) is therefore also diagonal:

    [L_noise^{-1}]_{PP} = 1 / [L_noise]_{PP}

allowing

    L_gate · vec(rho) = L_noise^{-1} · L' · vec(rho)

Public API
----------
PauliTwirlNoise
    .noise_ptm()                 → 16×16 real superoperator  N
    .noise_inverse_ptm()         → 16×16 real superoperator  N^{-1}
    .noisy_gate_ptm(gate_u)      → 16×16 superoperator  N · U_ptm
    .apply_noise(state_ptm)      → evolved Pauli vector
    .apply_inverse(state_ptm)    → de-noised Pauli vector
    .pauli_channel_probs         → dict {pauli_label: probability}
    .analytic_description()      → human-readable summary

PauliChannelFromProcess
    Construct a PauliTwirlNoise from an arbitrary process matrix / Kraus ops.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pauli basis (4^2 = 16 two-qubit operators in Pauli basis)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_P1 = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
_LABELS_2Q = [p + q for p in "IXYZ" for q in "IXYZ"]
_BASIS_2Q = [np.kron(_P1[p], _P1[q]) for p in "IXYZ" for q in "IXYZ"]

_DIM = 4  # single-qubit Hilbert space dimension
_DIM2 = 16  # two-qubit Pauli basis size


def _unitary_to_ptm(U: np.ndarray) -> np.ndarray:
    """
    Convert a 4×4 unitary U to its 16×16 Pauli Transfer Matrix (PTM).

    PTM_{ij} = (1/4) Tr[ sigma_i · U · sigma_j · U† ]
    """
    ptm = np.zeros((_DIM2, _DIM2), dtype=float)
    for j, Pj in enumerate(_BASIS_2Q):
        rotated = U @ Pj @ U.conj().T
        for i, Pi in enumerate(_BASIS_2Q):
            ptm[i, j] = np.real(np.trace(Pi @ rotated)) / _DIM
    return ptm


def _pauli_channel_ptm(probs: dict[str, float]) -> np.ndarray:
    """
    Build the 16×16 PTM of a two-qubit Pauli channel.

    N(rho) = Σ_P p_P P rho P†

    The PTM of a Pauli channel is diagonal:
        PTM[i,i] = Σ_P p_P (-1)^{anticommutes(sigma_i, P)}

    Equivalently, for the diagonal element corresponding to sigma_i:
        lambda_i = Σ_P p_P  s(sigma_i, P)
    where s = +1 if [sigma_i, P] = 0 and s = -1 if {sigma_i, P} = 0.

    Note: sigma_I always has eigenvalue +1.
    """
    ptm = np.zeros((_DIM2, _DIM2), dtype=float)
    for i, (label_i, Pi) in enumerate(zip(_LABELS_2Q, _BASIS_2Q)):
        lam = 0.0
        for label_P, p_P in probs.items():
            # Commutator sign: (+1) if Pi commutes with P, (-1) if anticommutes
            P_mat = _BASIS_2Q[_LABELS_2Q.index(label_P)]
            comm = Pi @ P_mat - P_mat @ Pi
            sign = +1.0 if np.allclose(comm, 0) else -1.0
            lam += p_P * sign
        ptm[i, i] = lam
    return ptm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PauliTwirlNoise:
    """
    Pauli-twirled noise model for a two-qubit gate with a given error rate.

    After twirling, a depolarising channel with total error rate ε becomes
    a Pauli channel.  Under the assumption that the underlying noise is
    depolarising:

        N(rho) = (1 − ε) rho  +  (ε/15) Σ_{P≠II} P rho P†

    which is the uniform Pauli channel with

        p_{II} = 1 − ε + ε/15 = 1 − 14ε/15
        p_P    = ε/15           for all P ≠ II  (15 terms)

    The noise PTM is diagonal with eigenvalues:

        λ_{II} = 1                       (identity row always 1)
        λ_P    = 1 − (16/15) ε           for all non-identity Paulis

    The inverse noise PTM is diagonal with eigenvalues 1/λ_P.

    Parameters
    ----------
    error_rate : float
        Total depolarising error rate ε ∈ [0, 1].  For a CNOT this is
        typically the reported "two-qubit gate error" from backend properties.
    gate_unitary : np.ndarray, optional
        4×4 unitary of the ideal gate.  Used only for ``noisy_gate_ptm``.
        Defaults to the CNOT unitary.
    """

    def __init__(
        self,
        error_rate: float,
        gate_unitary: np.ndarray | None = None,
    ) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1].")
        self.error_rate = error_rate

        if gate_unitary is None:
            # Default: CNOT
            U = np.eye(4, dtype=complex)
            U[2, 2], U[3, 3] = 0, 0
            U[2, 3], U[3, 2] = 1, 1
            self.gate_unitary = U
        else:
            self.gate_unitary = np.array(gate_unitary, dtype=complex)

        self._probs: dict[str, float] = self._compute_probs()

    # ------------------------------------------------------------------
    # Channel probabilities
    # ------------------------------------------------------------------

    def _compute_probs(self) -> dict[str, float]:
        """Pauli channel probabilities for a uniform depolarising channel."""
        eps = self.error_rate
        n_non_identity = _DIM2 - 1  # 15 for 2 qubits
        p_II = 1.0 - eps + eps / (n_non_identity + 1)
        p_P = eps / (n_non_identity + 1)
        probs = {label: p_P for label in _LABELS_2Q}
        probs["II"] = p_II
        return probs

    @property
    def pauli_channel_probs(self) -> dict[str, float]:
        """Dict mapping two-qubit Pauli label → probability."""
        return dict(self._probs)

    # ------------------------------------------------------------------
    # PTM representations
    # ------------------------------------------------------------------

    def noise_ptm(self) -> np.ndarray:
        """
        16×16 real Pauli Transfer Matrix of the noise channel N.

        Diagonal matrix; diagonal entry for Pauli P is:

            λ_P = 1                 if P = II
            λ_P = 1 − (16/15) ε    otherwise (depolarising assumption)
        """
        return _pauli_channel_ptm(self._probs)

    def noise_inverse_ptm(self) -> np.ndarray:
        """
        16×16 real PTM of N^{-1} (the noise inversion channel).

        Because the noise PTM is diagonal, inversion is element-wise:

            [N^{-1}]_{PP} = 1 / [N]_{PP}

        This is the analytic inverse under the depolarising / Pauli twirl
        assumption:

            [N^{-1}]_{II,II} = 1
            [N^{-1}]_{P,P}   = 1 / (1 − 16ε/15)   for P ≠ II

        Note: if ε = 15/16, the non-identity eigenvalues vanish and the
        channel is not invertible (completely depolarising).
        """
        N = self.noise_ptm()
        diag = np.diag(N)
        if np.any(np.abs(diag) < 1e-14):
            raise ValueError(
                "Noise channel is not invertible (error rate too high or "
                "completely depolarising). Cannot construct N^{-1}."
            )
        return np.diag(1.0 / diag)

    def gate_ptm(self) -> np.ndarray:
        """16×16 PTM of the ideal gate unitary."""
        return _unitary_to_ptm(self.gate_unitary)

    def noisy_gate_ptm(self) -> np.ndarray:
        """
        16×16 PTM of the full noisy gate:

            L' = N · U_ptm

        The noisy evolution of a state vec(rho) is:

            vec(rho') = L' · vec(rho)  =  N · U · vec(rho)
        """
        return self.noise_ptm() @ self.gate_ptm()

    # ------------------------------------------------------------------
    # Application helpers
    # ------------------------------------------------------------------

    def apply_noise(self, state_ptm: np.ndarray) -> np.ndarray:
        """Apply noise channel N to a Pauli-basis state vector."""
        return self.noise_ptm() @ state_ptm

    def apply_inverse(self, state_ptm: np.ndarray) -> np.ndarray:
        """Apply inverse noise N^{-1} to a Pauli-basis state vector."""
        return self.noise_inverse_ptm() @ state_ptm

    def apply_noisy_gate(self, state_ptm: np.ndarray) -> np.ndarray:
        """Apply the full noisy gate L' = N·U to a Pauli-basis state vector."""
        return self.noisy_gate_ptm() @ state_ptm

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def analytic_description(self) -> str:
        eps = self.error_rate
        lam = 1.0 - (16.0 / 15.0) * eps
        lam_inv = 1.0 / lam if abs(lam) > 1e-14 else float("inf")
        return f"""
Pauli-Twirled Noise Channel — Analytic Description
====================================================
Gate error rate  ε = {eps:.6g}

After Pauli twirling, the depolarising error channel becomes a Pauli channel:

  N(ρ) = p_II · ρ  +  Σ_{{P≠II}} p_P · P ρ P†

Probabilities:
  p_II = 1 − 14ε/15 = {self._probs['II']:.6g}
  p_P  = ε/15       = {eps/15:.6g}   (for each of the 15 non-identity Paulis)

Noise PTM (diagonal):
  λ_II = 1
  λ_P  = 1 − (16/15)ε = {lam:.6g}   (for all P ≠ II)

Noisy gate superoperator (in Pauli basis):
  L'·|ρ⟩⟩ = N · U_ptm · |ρ⟩⟩

Noise inversion channel N^{{-1}} (diagonal PTM):
  [N^{{-1}}]_II  = 1
  [N^{{-1}}]_PP  = 1 / (1 − 16ε/15) = {lam_inv:.6g}   (for P ≠ II)

Recovery identity:
  U_ptm · |ρ⟩⟩  =  N^{{-1}} · L' · |ρ⟩⟩

This holds exactly when the twirled error is a perfect depolarising channel.
For real hardware the Pauli channel probabilities p_P will not be uniform;
use process tomography to obtain the true p_P values.
"""


# ---------------------------------------------------------------------------
# Construct from arbitrary process / Kraus operators
# ---------------------------------------------------------------------------


class PauliChannelFromProcess:
    """
    Derive a PauliTwirlNoise-compatible Pauli channel from an arbitrary
    two-qubit process (given as Kraus operators or a chi matrix).

    The Pauli-twirled version of any channel E is:

        T[E](rho) = Σ_P  (1/16) Σ_Q  (P⊗Q) E[(P⊗Q) rho (P⊗Q)†] (P⊗Q)†

    which is always a Pauli channel with probabilities:

        p_P = (1/16) Tr[ χ_PP ]   (diagonal of the chi matrix in Pauli basis)

    Parameters
    ----------
    kraus_ops : list[np.ndarray]
        List of 4×4 Kraus operators {K_i} satisfying Σ K_i†K_i = I.
    gate_unitary : np.ndarray, optional
        4×4 ideal gate unitary (for constructing PTMs).
    """

    def __init__(
        self,
        kraus_ops: list[np.ndarray],
        gate_unitary: np.ndarray | None = None,
    ) -> None:
        self.kraus_ops = [np.array(K, dtype=complex) for K in kraus_ops]
        self.gate_unitary = gate_unitary

    def chi_matrix(self) -> np.ndarray:
        """16×16 chi (process) matrix in the two-qubit Pauli basis."""
        chi = np.zeros((_DIM2, _DIM2), dtype=complex)
        for K in self.kraus_ops:
            # Expand K in Pauli basis: K = Σ_i α_i P_i,  α_i = Tr(P_i K)/4
            alphas = np.array([np.trace(P @ K) / _DIM for P in _BASIS_2Q])
            chi += np.outer(alphas, alphas.conj())
        return chi

    def pauli_channel_probs(self) -> dict[str, float]:
        """Diagonal of chi matrix gives Pauli channel probabilities."""
        chi = self.chi_matrix()
        return {label: float(np.real(chi[i, i])) for i, label in enumerate(_LABELS_2Q)}

    def effective_error_rate(self) -> float:
        """
        Effective depolarising error rate matching this Pauli channel,
        defined as ε = (15/16)(1 − λ) where λ is the average non-identity
        PTM eigenvalue.
        """
        probs = self.pauli_channel_probs()
        N_ptm = _pauli_channel_ptm(probs)
        non_identity_eigs = [N_ptm[i, i] for i in range(1, _DIM2)]
        lam_avg = float(np.mean(non_identity_eigs))
        return (15.0 / 16.0) * (1.0 - lam_avg)

    def to_pauli_twirl_noise(self) -> PauliTwirlNoise:
        """Convert to a PauliTwirlNoise using the effective error rate."""
        eps = self.effective_error_rate()
        obj = PauliTwirlNoise(eps, gate_unitary=self.gate_unitary)
        # Override probs with exact values from process
        obj._probs = self.pauli_channel_probs()
        return obj
