from __future__ import annotations

import copy

import numpy as np
import scipy.linalg

from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .tensor import _as_dense


def _lanczos_expm_apply(matvec, v: np.ndarray, tau: complex, krylov_dim: int = 25):
    """
    Compute exp(tau * H) v for Hermitian H given only its matvec, via the
    Lanczos algorithm with full reorthogonalisation.
    """
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-14:
        return v.copy()

    dim = v.size
    m = min(krylov_dim, dim)

    V = np.zeros((m, dim), dtype=complex)
    alphas = np.zeros(m)
    betas = np.zeros(m - 1) if m > 1 else np.zeros(0)

    V[0] = v / norm_v
    w = matvec(V[0])
    alphas[0] = np.real(np.vdot(V[0], w))
    w = w - alphas[0] * V[0]

    used = 1
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-12:
            break
        V[j] = w / beta
        for k in range(j):
            V[j] -= np.vdot(V[k], V[j]) * V[k]
        V[j] /= np.linalg.norm(V[j])
        betas[j - 1] = beta
        w = matvec(V[j])
        alphas[j] = np.real(np.vdot(V[j], w))
        w = w - alphas[j] * V[j] - beta * V[j - 1]
        used = j + 1

    T = np.diag(alphas[:used]).astype(complex)
    if used > 1:
        T += np.diag(betas[: used - 1], 1) + np.diag(betas[: used - 1], -1)

    e0 = np.zeros(used, dtype=complex)
    e0[0] = 1.0
    small = scipy.linalg.expm(tau * T) @ e0
    return norm_v * (V[:used].T @ small)


def _truncated_svd(mat: np.ndarray, max_bond: int, tol: float = 1e-12):
    u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
    if s[0] < 1e-14:
        keep = 1
    else:
        keep = int(np.sum(s > tol * s[0]))
        keep = max(1, min(keep, max_bond))
    return u[:, :keep], s[:keep], vh[:keep, :], keep


def _sample_qdrift_hamiltonian(
    hamiltonian: dict[str, complex],
    num_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    ... (docstring unchanged) ...
    """
    terms = list(hamiltonian.keys())
    # .real: hamiltonian may carry complex-typed values with a negligible
    # (already validated by TDVP.__init__, |Im| < 1e-10) imaginary part --
    # e.g. if the caller's original Hamiltonian dict used complex(...)
    # throughout, as in the qDRIFT-TDVP test script. Safe specifically
    # because that check has already happened before this function is
    # ever called; this function does NOT re-validate it itself.
    weights = np.array([hamiltonian[p] for p in terms], dtype=complex).real
    lam = np.sum(np.abs(weights))
    if lam == 0.0:
        return {}
    probs = np.abs(weights) / lam

    sampled_indices = rng.choice(len(terms), size=num_samples, p=probs)

    per_draw_magnitude = lam / num_samples
    aggregated: dict[str, float] = {}
    for idx in sampled_indices:
        p = terms[idx]
        sign = 1.0 if weights[idx] >= 0 else -1.0
        aggregated[p] = aggregated.get(p, 0.0) + sign * per_draw_magnitude

    return aggregated


class TDVP:
    """
    Time-Dependent Variational Principle evolution of an MPS under a
    Hamiltonian given as a weighted sum of Pauli strings, H = sum_k w_k P_k.

    [... existing class docstring content unchanged ...]

    QDRIFT HAMILTONIAN OPTION (use_qdrift_hamiltonian=True)
    ---------------------------------------------------------
    Chemistry Hamiltonians have L ~ O(n^4) terms, and an uncompressed
    Hamiltonian MPO has bond dimension chi_H ~ O(L) -- catastrophically
    large before compression (see the theoretical TEBD-vs-TDVP analysis
    developed earlier in this project). The standard fix, SVD-compressing
    the Hamiltonian MPO down to a fixed mpo_max_bond, is deterministic and
    one-shot: the same compressed Hamiltonian is used for the entire
    trajectory.

    This option replaces that with a STOCHASTIC alternative, inspired by
    (but NOT identical to) qDRIFT: at the start of EVERY TDVP time step, a
    fresh qDRIFT-style random subsample of qdrift_num_samples terms is
    drawn (see _sample_qdrift_hamiltonian above) and used to build a small
    Hamiltonian MPO for that step only, replacing self._W before the
    sweep proceeds exactly as normal.

    Re-sampling EVERY step, rather than once for the whole trajectory, is
    essential, not a minor detail: standard qDRIFT's own error guarantee
    relies on averaging over many independent random slices so that
    per-step errors partially cancel over the trajectory. A single
    one-shot random Hamiltonian used deterministically for the entire
    simulation has no such averaging and can simply give a biased
    trajectory with no mechanism to self-correct.

    IMPORTANT: this is a NEW, UNPROVEN heuristic, not an established
    method with a known error bound. Unlike the qDRIFT-for-TEBD variant
    built elsewhere in this project (which reuses qDRIFT's own proven
    mechanism unchanged), this applies a qDRIFT-INSPIRED importance-sampled
    estimator to a completely different part of the algorithm (Hamiltonian-
    MPO construction, not time-stepping) that qDRIFT's own convergence
    proof says nothing about. Treat any result from this option as
    something to verify empirically (see the accompanying test script),
    not as inheriting qDRIFT's theoretical guarantees by analogy.
    """

    def __init__(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int = 16,
        initial_mps: MatrixProductState | None = None,
        two_site: bool = True,
        mpo_max_bond: int | None = None,
        krylov_dim: int = 25,
        svd_tol: float = 1e-12,
        use_qdrift_hamiltonian: bool = False,
        qdrift_num_samples: int = 50,
        qdrift_seed: int | None = None,
    ) -> TDVP:
        """
        New parameters relative to the earlier version:

        use_qdrift_hamiltonian : if True, replaces the fixed, SVD-
            compressed Hamiltonian MPO with a fresh qDRIFT-style
            stochastic estimator re-sampled at every time step -- see the
            class docstring above. mpo_max_bond is IGNORED when this is
            True; qdrift_num_samples controls the resulting Hamiltonian-
            MPO's (much smaller) bond dimension instead.
        qdrift_num_samples : N in the stochastic Hamiltonian estimator
            (see _sample_qdrift_hamiltonian). Larger N -> lower estimator
            variance but a larger resampled Hamiltonian-MPO bond
            dimension each step (at most N, generally fewer after
            repeated draws of the same term collapse).
        qdrift_seed : optional seed for the persistent RNG used across
            all resampling calls in a single TDVP instance's lifetime
            (NOT reseeded every step -- reseeding every step with the
            same seed would draw the identical realization every time,
            defeating the point of re-randomizing).
        """
        self.hamiltonian = hamiltonian
        pauli = {
            key: value for key, value in hamiltonian.items() if np.abs(value) > 1e-12
        }
        self.pauli_terms = list(pauli.items())
        self.nqubits = len(self.pauli_terms[0][0])
        self.max_bond = max_mps_bond
        self.two_site = two_site
        self.krylov_dim = krylov_dim
        self.svd_tol = svd_tol

        self.use_qdrift_hamiltonian = use_qdrift_hamiltonian
        self.qdrift_num_samples = qdrift_num_samples
        self._qdrift_rng = np.random.default_rng(qdrift_seed)
        self.qdrift_bond_dim_history: list[int] = []

        for p_string, weight in self.pauli_terms:
            if np.abs(np.imag(weight)) > 1e-10:
                raise ValueError(
                    f"Hamiltonian coefficient for {p_string} is not real: {weight}"
                )

        mpo = MatrixProductOperator.from_hamiltonian(dict(self.pauli_terms))
        if mpo_max_bond is not None and mpo.bond_dimension > mpo_max_bond:
            mpo.compress(mpo_max_bond)
        self.mpo = mpo
        self._W = self._extract_mpo_arrays(mpo)

        if initial_mps is None:
            initial_mps = MatrixProductState.all_zero_mps(self.nqubits)
        self.initial_state = initial_mps
        self._A = self._extract_mps_arrays(initial_mps)

        self.times = [0.0]
        self.mps: MatrixProductState | None = None
        return

    # ------------------------------------------------------------------
    # qDRIFT Hamiltonian resampling
    # ------------------------------------------------------------------

    def _resample_qdrift_hamiltonian(self) -> None:
        """
        Draw a fresh qDRIFT-style stochastic Hamiltonian estimator and
        REPLACE self.mpo/self._W with it, for use in the upcoming step's
        local updates. Called once per full TDVP time step by run() when
        use_qdrift_hamiltonian=True -- see the class docstring for why
        resampling every step (not once for the whole trajectory) is
        essential.
        """
        simplified_ham = _sample_qdrift_hamiltonian(
            dict(self.pauli_terms), self.qdrift_num_samples, self._qdrift_rng
        )
        if not simplified_ham:
            return  # pathological all-zero draw; keep the previous self._W
        mpo = MatrixProductOperator.from_hamiltonian(simplified_ham)
        self.mpo = mpo
        self._W = self._extract_mpo_arrays(mpo)
        self.qdrift_bond_dim_history.append(mpo.bond_dimension)

    # ------------------------------------------------------------------
    # Conversion between the repo's tensor classes and padded dense arrays
    # ------------------------------------------------------------------

    def _extract_mps_arrays(self, mps: MatrixProductState) -> list[np.ndarray]:
        mps = copy.deepcopy(mps)
        mps.reshape("udp")
        arrays = []
        for i, t in enumerate(mps.tensors):
            a = np.asarray(_as_dense(t.data), dtype=complex)
            if i == 0:
                a = a[np.newaxis, :, :]
            elif i == mps.num_sites - 1:
                a = a[:, np.newaxis, :]
            arrays.append(a)
        return arrays

    def _extract_mpo_arrays(self, mpo: MatrixProductOperator) -> list[np.ndarray]:
        mpo = copy.deepcopy(mpo)
        mpo.reshape("udrl")
        arrays = []
        for i, t in enumerate(mpo.tensors):
            a = np.asarray(_as_dense(t.data), dtype=complex)
            if i == 0:
                a = a[np.newaxis, :, :, :]
            elif i == mpo.num_sites - 1:
                a = a[:, np.newaxis, :, :]
            arrays.append(a)
        return arrays

    def _rebuild_mps(self) -> MatrixProductState:
        arrays = []
        n = len(self._A)
        for i, a in enumerate(self._A):
            if i == 0:
                arrays.append(a[0, :, :])
            elif i == n - 1:
                arrays.append(a[:, 0, :])
            else:
                arrays.append(a)
        mps = MatrixProductState.from_arrays(arrays, shape="udp")
        mps.normalise()
        return mps

    # ------------------------------------------------------------------
    # Canonicalisation and environments
    # ------------------------------------------------------------------

    def _right_canonicalise(self) -> None:
        n = len(self._A)
        for i in range(n - 1, 0, -1):
            u, d, p = self._A[i].shape
            mat = self._A[i].transpose(0, 1, 2).reshape(u, d * p)
            q, r = np.linalg.qr(mat.conj().T)
            k = q.shape[1]
            self._A[i] = q.conj().T.reshape(k, d, p)
            self._A[i - 1] = np.einsum("udp,dk->ukp", self._A[i - 1], r.conj().T)

    def _transfer_down(self, env: np.ndarray, A: np.ndarray, W: np.ndarray):
        return np.einsum("bwk,bBp,wWpq,kKq->BWK", env, A.conj(), W, A, optimize=True)

    def _transfer_up(self, env: np.ndarray, A: np.ndarray, W: np.ndarray):
        return np.einsum("BWK,bBp,wWpq,kKq->bwk", env, A.conj(), W, A, optimize=True)

    def _bot_envs_from_scratch(self) -> list[np.ndarray]:
        n = len(self._A)
        bot = [None] * n
        bot[n - 1] = np.ones((1, 1, 1), dtype=complex)
        for i in range(n - 1, 0, -1):
            bot[i - 1] = self._transfer_up(bot[i], self._A[i], self._W[i])
        return bot

    # ------------------------------------------------------------------
    # Effective Hamiltonian matvecs
    # ------------------------------------------------------------------

    def _matvec_two_site(self, top, Wi, Wj, bot, shape):
        def mv(x):
            theta = x.reshape(shape)
            y = np.einsum(
                "bwk,wvpq,vurs,BuK,kqsK->bprB",
                top,
                Wi,
                Wj,
                bot,
                theta,
                optimize=True,
            )
            return y.ravel()

        return mv

    def _matvec_one_site(self, top, Wi, bot, shape):
        def mv(x):
            a = x.reshape(shape)
            y = np.einsum("bwk,wWpq,BWK,kKq->bBp", top, Wi, bot, a, optimize=True)
            return y.ravel()

        return mv

    def _matvec_zero_site(self, top, bot, shape):
        def mv(x):
            c = x.reshape(shape)
            y = np.einsum("bwk,BwK,kK->bB", top, bot, c, optimize=True)
            return y.ravel()

        return mv

    # ------------------------------------------------------------------
    # Sweeps (unchanged from the earlier version)
    # ------------------------------------------------------------------

    def _step_two_site(self, dt: float) -> None:
        n = len(self._A)
        A, W = self._A, self._W
        fwd = -1j * dt / 2
        bwd = +1j * dt / 2

        bot = self._bot_envs_from_scratch()
        top = [None] * n
        top[0] = np.ones((1, 1, 1), dtype=complex)

        for i in range(n - 1):
            chiL = A[i].shape[0]
            chiR = A[i + 1].shape[1]
            theta = np.einsum("udp,dDq->upqD", A[i], A[i + 1], optimize=True)
            shape = theta.shape
            mv = self._matvec_two_site(top[i], W[i], W[i + 1], bot[i + 1], shape)
            theta = _lanczos_expm_apply(
                mv, theta.ravel(), fwd, self.krylov_dim
            ).reshape(shape)

            mat = theta.reshape(chiL * 2, 2 * chiR)
            u, s, vh, keep = _truncated_svd(mat, self.max_bond, self.svd_tol)
            s = s / np.linalg.norm(s)
            A[i] = u.reshape(chiL, 2, keep).transpose(0, 2, 1)
            top[i + 1] = self._transfer_down(top[i], A[i], W[i])

            M = (np.diag(s) @ vh).reshape(keep, 2, chiR).transpose(0, 2, 1)
            if i < n - 2:
                shape1 = M.shape
                mv1 = self._matvec_one_site(top[i + 1], W[i + 1], bot[i + 1], shape1)
                M = _lanczos_expm_apply(mv1, M.ravel(), bwd, self.krylov_dim).reshape(
                    shape1
                )
            A[i + 1] = M

        bot[n - 1] = np.ones((1, 1, 1), dtype=complex)
        for i in range(n - 2, -1, -1):
            chiL = A[i].shape[0]
            chiR = A[i + 1].shape[1]
            theta = np.einsum("udp,dDq->upqD", A[i], A[i + 1], optimize=True)
            shape = theta.shape
            mv = self._matvec_two_site(top[i], W[i], W[i + 1], bot[i + 1], shape)
            theta = _lanczos_expm_apply(
                mv, theta.ravel(), fwd, self.krylov_dim
            ).reshape(shape)

            mat = theta.reshape(chiL * 2, 2 * chiR)
            u, s, vh, keep = _truncated_svd(mat, self.max_bond, self.svd_tol)
            s = s / np.linalg.norm(s)
            A[i + 1] = vh.reshape(keep, 2, chiR).transpose(0, 2, 1)
            bot[i] = self._transfer_up(bot[i + 1], A[i + 1], W[i + 1])

            M = (u @ np.diag(s)).reshape(chiL, 2, keep).transpose(0, 2, 1)
            if i > 0:
                shape1 = M.shape
                mv1 = self._matvec_one_site(top[i], W[i], bot[i], shape1)
                M = _lanczos_expm_apply(mv1, M.ravel(), bwd, self.krylov_dim).reshape(
                    shape1
                )
            A[i] = M

    def _step_one_site(self, dt: float) -> None:
        n = len(self._A)
        A, W = self._A, self._W
        fwd = -1j * dt / 2
        bwd = +1j * dt / 2

        bot = self._bot_envs_from_scratch()
        top = [None] * n
        top[0] = np.ones((1, 1, 1), dtype=complex)

        for i in range(n):
            shape1 = A[i].shape
            mv1 = self._matvec_one_site(top[i], W[i], bot[i], shape1)
            A[i] = _lanczos_expm_apply(mv1, A[i].ravel(), fwd, self.krylov_dim).reshape(
                shape1
            )

            if i == n - 1:
                break

            u, d, p = A[i].shape
            mat = A[i].transpose(0, 2, 1).reshape(u * p, d)
            q, r = np.linalg.qr(mat)
            k = q.shape[1]
            A[i] = q.reshape(u, p, k).transpose(0, 2, 1)
            top[i + 1] = self._transfer_down(top[i], A[i], W[i])

            shape0 = r.shape
            mv0 = self._matvec_zero_site(top[i + 1], bot[i], shape0)
            r = _lanczos_expm_apply(mv0, r.ravel(), bwd, self.krylov_dim).reshape(
                shape0
            )
            A[i + 1] = np.einsum("kd,dDp->kDp", r, A[i + 1], optimize=True)

        bot[n - 1] = np.ones((1, 1, 1), dtype=complex)
        for i in range(n - 1, -1, -1):
            shape1 = A[i].shape
            mv1 = self._matvec_one_site(top[i], W[i], bot[i], shape1)
            A[i] = _lanczos_expm_apply(mv1, A[i].ravel(), fwd, self.krylov_dim).reshape(
                shape1
            )

            if i == 0:
                break

            u, d, p = A[i].shape
            mat = A[i].transpose(0, 1, 2).reshape(u, d * p)
            q, r = np.linalg.qr(mat.conj().T)
            k = q.shape[1]
            A[i] = q.conj().T.reshape(k, d, p)
            bot[i - 1] = self._transfer_up(bot[i], A[i], W[i])

            r = r.conj().T
            shape0 = r.shape
            mv0 = self._matvec_zero_site(top[i], bot[i - 1], shape0)
            r = _lanczos_expm_apply(mv0, r.ravel(), bwd, self.krylov_dim).reshape(
                shape0
            )
            A[i - 1] = np.einsum("uKp,Kk->ukp", A[i - 1], r, optimize=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        time: float,
        steps: int,
        callback=None,
    ) -> MatrixProductState:
        """
        Evolve the MPS to exp(-i * H * time) |initial>.

        When use_qdrift_hamiltonian=True, a fresh stochastic Hamiltonian
        is resampled at the START of every step (before that step's
        sweep), via _resample_qdrift_hamiltonian -- see the class
        docstring for why this matters.
        """
        dt = time / steps
        self._right_canonicalise()

        for step in range(steps):
            if self.use_qdrift_hamiltonian:
                self._resample_qdrift_hamiltonian()
            if self.two_site:
                self._step_two_site(dt)
            else:
                self._step_one_site(dt)
            self.times.append(self.times[-1] + dt)
            if callback is not None:
                callback(step, self.times[-1], self._rebuild_mps())

        self.mps = self._rebuild_mps()
        return self.mps
