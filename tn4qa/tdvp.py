from __future__ import annotations

import numpy as np
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import FermionSite


class TeNPyTDVPAdapter:
    """
    Real-time TDVP wrapper around a custom MPS/MPO implementation.

    Assumptions about your objects:
    - mps.tensors[i].data is numpy array
    - MPO tensors are rank-4 arrays
    - open boundary conditions
    """

    # ------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------
    def __init__(self, my_mps, my_mpo, dt: float):
        self.my_mps = my_mps
        self.my_mpo = my_mpo
        self.dt = dt

        self.tenpy_mps = None
        self.tenpy_mpo = None

    # ============================================================
    # 1. MPS CONVERSION
    # ============================================================
    def _to_tenpy_mps(self) -> MPS:
        """
        Convert custom MPS -> TeNPy MPS in (Dl, d, Dr) format.
        """

        tensors = []

        for i, t in enumerate(self.my_mps.tensors):
            A = np.array(t.data, copy=True)

            # ----------------------------------------------------
            # Normalize shape
            # ----------------------------------------------------

            if A.ndim == 2:
                # boundary tensors
                if i == 0:
                    # (d, Dr)
                    A = np.expand_dims(A, axis=0)
                else:
                    # (Dl, d)
                    A = np.expand_dims(A, axis=2)

            elif A.ndim == 3:
                # Ensure (Dl, d, Dr)

                # Heuristic based on your convention:
                # you said: (up, down, phys) or similar
                # but standard MPS is bond-phys-bond

                if t.indices[1] in ("phys", "p"):
                    # already (Dl, d, Dr)
                    pass
                else:
                    # assume (Dl, Dr, d)
                    A = np.transpose(A, (0, 2, 1))

            else:
                raise ValueError(f"Unexpected tensor rank {A.ndim}")

            tensors.append(A)

        # --------------------------------------------------------
        # Build Fermionic site (IMPORTANT for quantum chemistry)
        # --------------------------------------------------------
        #
        # This is the correct default for electronic structure:
        # local basis = |0>, |↑>, |↓>, |↑↓>
        #
        # If your model is spinless, we can swap this later.
        # --------------------------------------------------------

        L = len(tensors)
        sites = [FermionSite(conserve=None) for _ in range(L)]

        psi = MPS.from_Bflat(sites, tensors, bc="finite")

        psi.canonical_form()

        self.tenpy_mps = psi
        return psi

    # ============================================================
    # 2. MPO CONVERSION
    # ============================================================
    def _to_tenpy_mpo(self) -> MPO:
        """
        Convert custom MPO -> TeNPy MPO.

        Assumes your MPO tensors are:
        (up, down, right, left) OR similar rank-4 structure.
        """

        W_list = []

        for W in self.my_mpo.tensors:
            W = np.array(W.data, copy=True)

            if W.ndim != 4:
                raise ValueError("MPO tensor must be rank-4")

            # ----------------------------------------------------
            # Convert to TeNPy ordering:
            # (Dl, d_out, d_in, Dr)
            # ----------------------------------------------------

            # your assumed ordering: (p_out, p_in, Dr, Dl)
            # => transpose to (Dl, p_out, p_in, Dr)

            W = np.transpose(W, (3, 0, 1, 2))

            W_list.append(W)

        L = len(W_list)

        sites = [FermionSite(conserve=None) for _ in range(L)]

        mpo = MPO(sites, W_list)

        self.tenpy_mpo = mpo
        return mpo

    # ============================================================
    # 3. RUN TDVP
    # ============================================================
    def run(self, steps: int, order: int = 2):
        """
        Real-time TDVP evolution using Two-site TDVP.
        """

        if self.tenpy_mps is None:
            psi = self._to_tenpy_mps()
        else:
            psi = self.tenpy_mps

        if self.tenpy_mpo is None:
            H = self._to_tenpy_mpo()
        else:
            H = self.tenpy_mpo

        eng = TwoSiteTDVPEngine(
            psi,
            H,
            dt=self.dt,
            order=order,
            trunc_params={
                "chi_max": 200,  # adjust for your system
                "svd_min": 1e-10,
            },
        )

        for _ in range(steps):
            eng.run_one_step()

        self.tenpy_mps = psi

        return self._from_tenpy_mps()

    # ============================================================
    # 4. BACK CONVERSION
    # ============================================================
    def _from_tenpy_mps(self):
        """
        TeNPy MPS -> custom MPS
        """

        psi = self.tenpy_mps

        arrays = []

        for B in psi.get_Bs():
            B = np.array(B, copy=True)
            arrays.append(B)

        # --------------------------------------------------------
        # restore boundary conventions
        # --------------------------------------------------------

        arrays[0] = arrays[0][0, :, :]  # remove dummy left bond
        arrays[-1] = arrays[-1][:, :, 0]  # remove dummy right bond

        return self.my_mps.__class__.from_arrays(arrays, shape="upd")
