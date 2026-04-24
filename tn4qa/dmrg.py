from __future__ import annotations

import numpy as np
from numpy import ndarray
from pyblock2.algebra.io import MPSTools
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from .mps import MatrixProductState


class DMRG:
    def __init__(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int = 16,
    ) -> DMRG:
        self.hamiltonian = hamiltonian
        pauli = {
            key[::-1]: value
            for key, value in hamiltonian.items()
            if np.abs(value) > 0.0
        }
        pauli = list(pauli.items())
        self.pauli_terms = pauli
        self.nqubits = len(self.pauli_terms[0][0])
        self.max_bond = max_mps_bond

        self.driver = self.set_driver()
        self.mpo = self.set_mpo()
        self.ket = self.set_initial_state()

        self.energy = None
        self.mps = None
        return

    def set_driver(self):
        driver_obj = DMRGDriver(
            symm_type=SymmetryTypes.SGB,
            n_threads=4,
        )
        driver_obj.initialize_system(
            n_sites=self.nqubits,
            pauli_mode=True,
        )
        return driver_obj

    def set_mpo(self):
        mpo = self.driver.get_mpo_any_pauli(self.pauli_terms)
        return mpo

    def set_initial_state(self):
        ket = self.driver.get_random_mps(tag="GS", bond_dim=2, nroots=1)
        ket = self.driver.get_random_mps(tag="GS", bond_dim=2, nroots=1)
        return ket

    def run(
        self,
        nsweeps: int,
        bond_dims: list[int] | None = None,
        noises: list[int] | None = None,
        thrds: list[int] | None = None,
    ):
        if not bond_dims:
            bond_dims = [
                round(2 + i * (self.max_bond - 2) / (nsweeps - 1))
                for i in range(nsweeps)
            ]
        if not noises:
            noises = (
                [1e-2] * max(1, nsweeps // 3)
                + [1e-4] * max(1, nsweeps // 3)
                + [1e-6] * max(1, nsweeps - 2 * (nsweeps // 3) - 1)
                + [0]
            )
        if not thrds:
            thrds = [1e-12] * len(bond_dims)
        self.energy = self.driver.dmrg(
            self.mpo,
            self.ket,
            n_sweeps=nsweeps,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=0,
        )
        self.mps = self.ket_to_tn4qa_mps()
        return self.energy, self.mps

    def ket_to_tn4qa_mps(self):
        ket = self.driver.adjust_mps(self.ket, dot=1)[0]
        pyket = MPSTools.from_block2(ket)

        arrays = []
        for tensor in pyket.tensors:
            arrays.append(np.array(tensor.blocks[0].reduced))
        mps = MatrixProductState.from_arrays(arrays, shape="upd")
        mps.reshape()
        return mps


# ---------------------------------------------------------------------------
# LinearOperator factory
# ---------------------------------------------------------------------------


def _make_matvec(left, mpo, right):
    # left[χ_bra_l, χ_mpo_l, χ_ket_l]  — all 'up' indices of site
    # mpo[χ_mpo_up=χ_mpo_l, χ_mpo_down=χ_mpo_r, d_out, d_in]
    # right[χ_bra_r, χ_mpo_r, χ_ket_r] — all 'down' indices of site
    # state v[χ_ket_l, d_in, χ_ket_r]
    # result[χ_bra_l, d_out, χ_bra_r]

    # Pre-contract: H[χ_bra_l, d_out, χ_bra_r, χ_ket_l, d_in, χ_ket_r]
    #   = Σ_{χ_mpo_l, χ_mpo_r} left[χ_bra_l, χ_mpo_l, χ_ket_l]
    #                          * mpo[χ_mpo_l, χ_mpo_r, d_out, d_in]
    #                          * right[χ_bra_r, χ_mpo_r, χ_ket_r]
    H = np.einsum(
        "ijk,jlmn,plk->imponk",  # WRONG — fix:
        left,
        mpo,
        right,
    )
    # Correct:
    # left: (a=χ_bra_l, b=χ_mpo_l, c=χ_ket_l)
    # mpo:  (b=χ_mpo_l, d=χ_mpo_r, e=d_out,   f=d_in)
    # right:(g=χ_bra_r, d=χ_mpo_r, h=χ_ket_r)
    H = np.einsum("abc,bdef,gdb->aegcfh", left, mpo, right, optimize="optimal")
    # H: (χ_bra_l, d_out, χ_bra_r, χ_ket_l, d_in, χ_ket_r)
    chi_l = left.shape[0]
    chi_r = right.shape[0]
    d = mpo.shape[2]
    dim = chi_l * d * chi_r
    H_mat = H.reshape(dim, dim)

    def matvec(v_flat):
        return H_mat @ v_flat

    return matvec, dim


def _make_matvec_sparse_mpo(left: ndarray, mpo_coo, right: ndarray):
    """
    Matvec for large sparse MPO tensors (quantum chemistry regime).

    Instead of converting the MPO to dense, we iterate over the d² physical
    index pairs and use scipy sparse matrix-vector products for the bond part.
    """
    import scipy.sparse as scs

    chi_l = left.shape[0]
    chi_r = right.shape[0]
    chi_ml = mpo_coo.shape[0]
    chi_mr = mpo_coo.shape[1]
    d = mpo_coo.shape[2]

    coords = mpo_coo.coords  # (4, nnz)
    vals = mpo_coo.data

    # Pre-build sparse W matrix per (σ, σ') pair — done once per site
    W_slices = {}
    for sig in range(d):
        for sigp in range(d):
            mask = (coords[2] == sig) & (coords[3] == sigp)
            if not mask.any():
                continue
            W_slices[(sig, sigp)] = scs.csr_matrix(
                (vals[mask], (coords[0][mask], coords[1][mask])),
                shape=(chi_ml, chi_mr),
            )

    dim = chi_l * d * chi_r

    def matvec(v_flat: ndarray) -> ndarray:
        v = v_flat.reshape(chi_l, d, chi_r)
        result = np.zeros_like(v)
        for (sig, sigp), W in W_slices.items():
            # Contract: left[i,k,i2] @ v[i2, sig, j2] → lv[i, k, j2]
            lv = np.einsum("ijk,jl->ikl", left, v[:, sig, :])  # (χ_l, χ_ml, χ_r)
            # Apply W along mpo bond: lv[i, k, j2] @ W[k, m] → lvW[i, m, j2]
            lvW = np.einsum("ikj,km->imj", lv, W.toarray())  # dense for now
            # Contract with right: lvW[i, m, j2] @ right[j, m, j2] → res[i, j]
            result[:, sigp, :] += np.einsum("imj,jmk->ik", lvW, right)
        return result.ravel()

    return matvec, dim
