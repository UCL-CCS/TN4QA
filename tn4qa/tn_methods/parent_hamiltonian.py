"""
Computes the parent Hamiltonian of an MPS in the Pauli-string basis,
and optionally builds the corresponding MPO.
"""

from collections import defaultdict
from itertools import product as iproduct
from typing import Optional

import numpy as np

from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState

# ── Pauli matrices ─────────────────────────────────────────────────────────────

_PAULI_NAMES = ["I", "X", "Y", "Z"]
_PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ── ParentHamiltonian class ────────────────────────────────────────────────────


class ParentHamiltonian:
    """
    Computes and stores the parent Hamiltonian of an MPS.

    Parameters
    ----------
    mps        : your MPS object
    block_size : number of contiguous sites per local projector term.
                 If None (default), the minimum valid block size is chosen
                 automatically based on bond dimension and physical dimension.
    tol        : numerical threshold for null-space eigenvalues and
                 near-zero Pauli coefficients.
    build_mpo  : if True, also construct an MPO representation.

    Attributes (populated after construction)
    ------------------------------------------
    hamiltonian : dict  {pauli_string : real_coefficient}
    mpo         : list of numpy arrays (None if build_mpo=False)
    N           : number of sites
    d           : physical dimension
    chi         : bond dimension
    block_size  : block size actually used
    """

    def __init__(
        self,
        mps: MatrixProductState,
        block_size: Optional[int] = None,
        tol: float = 1e-10,
        build_mpo: bool = False,
    ):
        self.tol = tol
        self.N = len(mps.tensors)

        # ── intrinsic MPS properties ──────────────────────────────────────────
        self.d = self._physical_dim(mps)
        self.chi = self._bond_dim(mps)

        # ── choose block size ─────────────────────────────────────────────────
        min_block = self._min_block_size()
        if block_size is None:
            self.block_size = min_block
        else:
            if block_size < min_block:
                print(
                    f"[ParentHamiltonian] Warning: requested block_size="
                    f"{block_size} may be too small for χ={self.chi}, "
                    f"d={self.d} (minimum recommended: {min_block}). "
                    f"Proceeding with requested value."
                )
            self.block_size = block_size

        # ── compute ───────────────────────────────────────────────────────────
        self.hamiltonian = self._build_hamiltonian(mps)
        self.mpo = self._build_mpo() if build_mpo else None

    # ── public helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_terms = len(self.hamiltonian)
        return (
            f"ParentHamiltonian(N={self.N}, d={self.d}, χ={self.chi}, "
            f"block_size={self.block_size}, terms={n_terms})"
        )

    def print_hamiltonian(
        self, max_terms: Optional[int] = None, sort_by: str = "weight"
    ) -> None:
        """
        Pretty-print the Pauli-string decomposition.

        Parameters
        ----------
        max_terms : if set, only the top `max_terms` terms are shown.
        sort_by   : 'weight' (descending |coeff|) or 'string'.
        """
        items = list(self.hamiltonian.items())
        if sort_by == "weight":
            items.sort(key=lambda x: -abs(x[1]))
        else:
            items.sort(key=lambda x: x[0])
        if max_terms is not None:
            items = items[:max_terms]

        width = self.N
        print(f"\n{'Pauli string':<{width+2}}  Coefficient")
        print("─" * (width + 18))
        for pstr, w in items:
            print(f"  {pstr}  {w:+.8f}")
        print(f"\n  {len(self.hamiltonian)} terms total")

    def matrix(self) -> np.ndarray:
        """
        Return the full 2^N × 2^N Hamiltonian matrix (only feasible for small N).
        """
        dim = 2**self.N
        H = np.zeros((dim, dim), dtype=complex)
        for pstr, w in self.hamiltonian.items():
            H += w * self._pauli_string_matrix(pstr)
        return H

    # ── private: MPS utilities ─────────────────────────────────────────────────

    @staticmethod
    def _get_dense(tensor) -> np.ndarray:
        return np.asarray(tensor.data.todense())

    def _physical_dim(self, mps) -> int:
        """Physical dimension = last index of any tensor."""
        return mps.tensors[0].dimensions[-1]

    def _bond_dim(self, mps) -> int:
        """
        Bond dimension = 'down' index of the top tensor (index 0),
        which equals the bond connecting sites 0 and 1.
        Returns 1 for a single-site MPS.
        """
        if self.N == 1:
            return 1
        # top tensor: (down, physical) → dimensions[0] is the bond
        return mps.tensors[0].dimensions[0]

    def _min_block_size(self) -> int:
        """
        Smallest block size n such that d^n > χ^2,
        guaranteeing a non-trivial null space in the block RDM.
        For χ=1 (product state) this is 1.
        """
        n = 1
        while self.d**n <= self.chi**2:
            n += 1
        return n

    # ── private: contraction ───────────────────────────────────────────────────

    def _site_to_lrp(self, mps, i: int) -> np.ndarray:
        """
        Return site tensor reshaped to (χ_left, χ_right, d),
        inserting dummy bond axes of size 1 at the open boundaries.
        """
        t = self._get_dense(mps.tensors[i])
        if i == 0:  # (down, phys) → (1, down, phys)
            t = t[np.newaxis, :]
        elif i == self.N - 1:  # (up, phys)   → (up, 1, phys)
            t = t[:, np.newaxis, :]
        # middle tensors are already (up, down, phys)
        return t

    def _block_tensor(self, mps, start: int, stop: int) -> np.ndarray:
        """
        Contract MPS sites [start..stop] into a single tensor:
            shape: (χ_L, χ_R, d_start, ..., d_stop)
        """
        block = self._site_to_lrp(mps, start)  # (χ_L, χ_R, d)

        for i in range(start + 1, stop + 1):
            t = self._site_to_lrp(mps, i)  # (χ_L', χ_R', d_i)
            # contract right bond of block with left bond of t
            block = np.tensordot(block, t, axes=([1], [0]))
            # resulting shape: (χ_L, *phys_so_far, χ_R', d_i)
            # move χ_R' (axis n_phys+1) to position 1
            n_phys = i - start  # physical axes accumulated so far
            order = [0, n_phys + 1] + list(range(1, n_phys + 1)) + [n_phys + 2]
            block = block.transpose(order)
            # shape now: (χ_L, χ_R', d_start, ..., d_i)

        return block

    def _block_rdm(self, mps, start: int, stop: int) -> np.ndarray:
        """
        Reduced density matrix for sites [start..stop], shape (D, D)
        where D = d^(stop-start+1). Traced over virtual (bond) indices.
        """
        block = self._block_tensor(mps, start, stop)
        chi_L, chi_R = block.shape[0], block.shape[1]
        D = int(np.prod(block.shape[2:]))

        mat = block.reshape(chi_L * chi_R, D)
        rho = mat.conj().T @ mat
        tr = np.trace(rho)
        if tr > self.tol:
            rho /= tr
        return rho

    # ── private: algebra ───────────────────────────────────────────────────────

    def _null_projector(self, rho: np.ndarray) -> np.ndarray:
        """
        Projector onto the null space of rho (eigenvectors with
        eigenvalue < tol).
        """
        eigvals, eigvecs = np.linalg.eigh(rho)
        null_vecs = eigvecs[:, eigvals < self.tol]
        if null_vecs.shape[1] == 0:
            return np.zeros_like(rho)
        return null_vecs @ null_vecs.conj().T

    def _decompose_paulis(self, matrix: np.ndarray, n_qubits: int) -> dict:
        """
        Expand `matrix` in the n-qubit Pauli basis.
        Returns {pauli_string: coefficient}, dropping near-zero terms.
        """
        d = 2**n_qubits
        coeffs = {}
        for names in iproduct(_PAULI_NAMES, repeat=n_qubits):
            label = "".join(names)
            P = _PAULI_MATS[names[0]]
            for name in names[1:]:
                P = np.kron(P, _PAULI_MATS[name])
            c = np.trace(P @ matrix) / d
            if abs(c) > self.tol:
                coeffs[label] = c
        return coeffs

    @staticmethod
    def _embed(local_label: str, start: int, N: int) -> str:
        """Pad a local Pauli string with identities to length N."""
        n_local = len(local_label)
        return "I" * start + local_label + "I" * (N - start - n_local)

    def _pauli_string_matrix(self, pstr: str) -> np.ndarray:
        """Full 2^N matrix for a Pauli string of length N."""
        mat = _PAULI_MATS[pstr[0]]
        for p in pstr[1:]:
            mat = np.kron(mat, _PAULI_MATS[p])
        return mat

    # ── private: main build ────────────────────────────────────────────────────

    def _build_hamiltonian(self, mps) -> dict:
        H = defaultdict(complex)

        for start in range(self.N - self.block_size + 1):
            stop = start + self.block_size - 1

            rho = self._block_rdm(mps, start, stop)
            proj = self._null_projector(rho)

            if np.allclose(proj, 0, atol=self.tol):
                print(
                    f"[ParentHamiltonian] Block [{start},{stop}] has trivial "
                    f"null space — consider increasing block_size."
                )
                continue

            local_paulis = self._decompose_paulis(proj, self.block_size)

            for local_label, coeff in local_paulis.items():
                global_label = self._embed(local_label, start, self.N)
                H[global_label] += coeff

        # Drop cancelled terms; Hamiltonian is Hermitian so coefficients are real
        return {k: v.real for k, v in H.items() if abs(v) > self.tol}

    # ── private: MPO build ─────────────────────────────────────────────────────

    def _build_mpo(self) -> list:
        """
        Construct an MPO for the Hamiltonian by summing one MPO layer
        per Pauli string.  Bond dimension = number of terms.

        Shapes (your convention):
            site 0   : (χ_down, d, d)
            site i   : (χ_up, χ_down, d, d)
            site N-1 : (χ_up, d, d)
        """
        terms = list(self.hamiltonian.items())
        n_terms = len(terms)
        d = self.d
        N = self.N
        mpo = []

        for i in range(N):
            layers = []
            for pstr, w in terms:
                mat = _PAULI_MATS[pstr[i]]
                if i == 0:
                    mat = w * mat  # absorb weight into first site
                layers.append(mat)  # (d, d)

            stack = np.stack(layers, axis=0)  # (n_terms, d, d)

            if i == 0:
                # (χ_down=n_terms, d, d)
                mpo.append(stack)
            elif i == N - 1:
                # (χ_up=n_terms, d, d)
                mpo.append(stack)
            else:
                # (χ_up=n_terms, χ_down=n_terms, d, d) — diagonal in bond space
                t = np.zeros((n_terms, n_terms, d, d), dtype=complex)
                for k in range(n_terms):
                    t[k, k] = stack[k]
                mpo.append(t)

        mpo = MatrixProductOperator.from_arrays(mpo)
        return mpo
