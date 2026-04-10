"""
Module 3: Readout Error Mitigation
Implements matrix-inversion and least-squares readout error mitigation
on raw measurement counts from a quantum circuit.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit.result import Result


class ReadoutErrorMitigator:
    """
    Calibrate and apply readout error mitigation.

    Two mitigation strategies are available:

    ``"matrix_inversion"``
        Build the full 2^n × 2^n assignment matrix A where
        A[i,j] = P(measure i | prepared j), then apply A^{-1} to the
        raw probability vector.  Exact but exponential in qubit count;
        practical up to ~10–12 qubits.

    ``"tensored"``
        Assume readout errors are independent per qubit.  Build n separate
        2×2 calibration matrices and apply their tensor product.  Scales to
        many qubits but ignores correlated readout errors.

    Parameters
    ----------
    backend : BackendV2
        Backend used for calibration circuits (can be a simulator).
    qubits : list[int]
        Physical qubit indices to calibrate.
    method : str
        ``"matrix_inversion"`` (default) or ``"tensored"``.
    shots : int
        Shots per calibration circuit.
    """

    def __init__(
        self,
        backend: BackendV2,
        qubits: list[int],
        method: str = "tensored",
        shots: int = 8192,
    ) -> None:
        if method not in ("matrix_inversion", "tensored"):
            raise ValueError("method must be 'matrix_inversion' or 'tensored'")
        self.backend = backend
        self.qubits = qubits
        self.method = method
        self.shots = shots
        self.n = len(qubits)
        self._cal_matrix: np.ndarray | None = None  # full 2^n × 2^n
        self._single_matrices: list[np.ndarray] | None = None  # n × 2×2

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        """
        Run calibration circuits on the backend and build the assignment
        matrix (or per-qubit matrices for tensored method).
        """
        if self.method == "matrix_inversion":
            self._cal_matrix = self._build_full_cal_matrix()
        else:
            self._single_matrices = self._build_tensored_cal_matrices()

    def _build_full_cal_matrix(self) -> np.ndarray:
        """2^n × 2^n assignment matrix via prepare-all-bitstrings calibration."""
        dim = 2**self.n
        cal_matrix = np.zeros((dim, dim), dtype=float)

        for j, bitstring in enumerate(itertools.product("01", repeat=self.n)):
            # Prepare state |bitstring>
            qc = QuantumCircuit(self.n, self.n)
            for k, bit in enumerate(reversed(bitstring)):  # qiskit qubit ordering
                if bit == "1":
                    qc.x(k)
            qc.measure(range(self.n), range(self.n))
            job = self.backend.run(transpile(qc, self.backend), shots=self.shots)
            counts = job.result().get_counts()

            # Normalise counts into column j of the assignment matrix
            total = sum(counts.values())
            for i, out_bitstring in enumerate(itertools.product("01", repeat=self.n)):
                key = "".join(out_bitstring)
                cal_matrix[i, j] = counts.get(key, 0) / total

        return cal_matrix

    def _build_tensored_cal_matrices(self) -> list[np.ndarray]:
        """One 2×2 assignment matrix per qubit."""
        matrices = []
        for q_idx, qubit in enumerate(self.qubits):
            mat = np.zeros((2, 2), dtype=float)
            for j, state in enumerate([0, 1]):
                qc = QuantumCircuit(1, 1)
                if state == 1:
                    qc.x(0)
                qc.measure(0, 0)
                job = self.backend.run(
                    transpile(
                        qc,
                        self.backend,
                        initial_layout=[qubit],
                        layout_method="trivial",
                        routing_method="none",
                    ),
                    shots=self.shots,
                )
                counts = job.result().get_counts()
                total = sum(counts.values())
                mat[0, j] = counts.get("0", 0) / total
                mat[1, j] = counts.get("1", 0) / total
            matrices.append(mat)
        return matrices

    # ------------------------------------------------------------------
    # Mitigation
    # ------------------------------------------------------------------

    def mitigate_counts(self, raw_counts: dict[str, int]) -> dict[str, float]:
        """
        Apply readout error mitigation to raw measurement counts.

        Parameters
        ----------
        raw_counts : dict[str, int]
            Raw counts dict from ``result.get_counts()``.

        Returns
        -------
        dict[str, float]
            Mitigated quasi-probability distribution.  Values may be
            slightly negative; clip to zero if non-negative probabilities
            are required.
        """
        if self._cal_matrix is None and self._single_matrices is None:
            raise RuntimeError("Call calibrate() before mitigate_counts().")

        raw_probs = self._counts_to_probs(raw_counts)

        if self.method == "matrix_inversion":
            mit_probs = self._mitigate_full(raw_probs)
        else:
            mit_probs = self._mitigate_tensored(raw_probs)

        return mit_probs

    def mitigate_result(self, result: Result, circuit_idx: int = 0) -> dict[str, float]:
        """Convenience wrapper — accept a Qiskit ``Result`` object directly."""
        return self.mitigate_counts(result.get_counts(circuit_idx))

    def _counts_to_probs(self, counts: dict[str, int]) -> np.ndarray:
        dim = 2**self.n
        prob_vec = np.zeros(dim, dtype=float)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            # Strip spaces qiskit sometimes inserts
            key = bitstring.replace(" ", "")
            idx = int(key, 2)
            if idx < dim:
                prob_vec[idx] = count / total
        return prob_vec

    def _mitigate_full(self, raw_probs: np.ndarray) -> dict[str, float]:
        A = self._cal_matrix
        try:
            mit_vec = np.linalg.solve(A, raw_probs)
        except np.linalg.LinAlgError:
            mit_vec = np.linalg.lstsq(A, raw_probs, rcond=None)[0]

        # Renormalise (least-norm non-negative solution)
        mit_vec = mit_vec.clip(min=0)
        s = mit_vec.sum()
        if s > 0:
            mit_vec /= s
        else:
            warnings.warn("Mitigated distribution sums to zero; returning raw.")
            mit_vec = raw_probs

        return {
            format(i, f"0{self.n}b"): float(v) for i, v in enumerate(mit_vec) if v > 0
        }

    def _mitigate_tensored(self, raw_probs: np.ndarray) -> dict[str, float]:
        # Build the full inverse as a tensor product of per-qubit inverses
        inv_mat = np.array([[1.0]])
        for mat in self._single_matrices:
            try:
                inv_2x2 = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                inv_2x2 = np.linalg.pinv(mat)
            inv_mat = np.kron(inv_mat, inv_2x2)

        mit_vec = inv_mat @ raw_probs
        mit_vec = mit_vec.clip(min=0)
        s = mit_vec.sum()
        if s > 0:
            mit_vec /= s

        return {
            format(i, f"0{self.n}b"): float(v) for i, v in enumerate(mit_vec) if v > 0
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def assignment_matrix(self) -> np.ndarray:
        """Return the full assignment matrix (matrix_inversion method only)."""
        if self._cal_matrix is None:
            raise RuntimeError("Not calibrated yet.")
        return self._cal_matrix.copy()

    def readout_fidelity(self) -> float:
        """Average assignment fidelity: mean of diagonal of the cal matrix."""
        if self.method == "matrix_inversion":
            if self._cal_matrix is None:
                raise RuntimeError("Not calibrated yet.")
            return float(np.mean(np.diag(self._cal_matrix)))
        else:
            if self._single_matrices is None:
                raise RuntimeError("Not calibrated yet.")
            fids = [float(np.mean(np.diag(m))) for m in self._single_matrices]
            return float(np.mean(fids))
